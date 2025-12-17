"""
Helper functions for scene processing.
"""
from __future__ import annotations
import json
import threading
from typing import Optional, Any, TYPE_CHECKING
from pathlib import Path
from queue import Queue
import numpy as np
import cv2
from PIL import Image

if TYPE_CHECKING:
    from core.config import Config
    from core.logger import AppLogger
    from core.models import Scene, SceneState, AnalysisParameters
    from core.managers import ThumbnailManager, ModelRegistry
    import gradio as gr

from core.utils import create_frame_map, render_mask_overlay, draw_bbox, _to_json_safe
from core.managers import initialize_analysis_models
from core.scene_utils.subject_masker import SubjectMasker
from core.shared import build_scene_gallery_items


def draw_boxes_preview(img: np.ndarray, boxes_xyxy: list[list[int]], cfg: 'Config') -> np.ndarray:
    """
    Draw bounding boxes on an image for preview.
    
    Args:
        img: RGB image
        boxes_xyxy: List of boxes in [x1, y1, x2, y2] format
        cfg: Config with visualization settings
        
    Returns:
        Image with boxes drawn
    """
    img = img.copy()
    for x1, y1, x2, y2 in boxes_xyxy:
        cv2.rectangle(
            img, 
            (int(x1), int(y1)), 
            (int(x2), int(y2)), 
            cfg.visualization_bbox_color, 
            cfg.visualization_bbox_thickness
        )
    return img


def save_scene_seeds(
    scenes_list: list['Scene'], 
    output_dir_str: str, 
    logger: 'AppLogger'
) -> None:
    """
    Save scene seed information to JSON file.
    
    Args:
        scenes_list: List of Scene objects
        output_dir_str: Output directory path
        logger: Application logger
    """
    if not scenes_list or not output_dir_str:
        return
    scene_seeds = {}
    for s in scenes_list:
        data = {
            'best_frame': s.best_frame,
            'seed_frame_idx': s.seed_frame_idx,
            'seed_type': s.seed_type,
            'seed_config': s.seed_config,
            'status': s.status,
            'seed_metrics': s.seed_metrics
        }
        scene_seeds[str(s.shot_id)] = data
    try:
        path = Path(output_dir_str) / "scene_seeds.json"
        path.write_text(json.dumps(_to_json_safe(scene_seeds), indent=2), encoding='utf-8')
        logger.info("Saved scene_seeds.json")
    except Exception:
        logger.error("Failed to save scene_seeds.json", exc_info=True)


def get_scene_status_text(scenes_list: list['Scene']) -> tuple[str, Any]:
    """
    Generate status text and button update for scene list.
    
    Args:
        scenes_list: List of Scene objects
        
    Returns:
        Tuple of (status_text, gr.update for button)
    """
    import gradio as gr
    
    if not scenes_list:
        return "No scenes loaded.", gr.update(interactive=False)
    
    included_scenes = [s for s in scenes_list if s.status == 'included']
    ready_for_propagation_count = sum(
        1 for s in included_scenes 
        if s.seed_result and s.seed_result.get('bbox')
    )
    total_count = len(scenes_list)
    included_count = len(included_scenes)
    
    # Count rejection reasons
    rejection_counts = {}
    for scene in scenes_list:
        if scene.status == 'excluded' and scene.rejection_reasons:
            for reason in scene.rejection_reasons:
                rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
    
    status_text = f"{included_count}/{total_count} scenes included for propagation."
    if rejection_counts:
        reasons_summary = ", ".join(
            f"{reason}: {count}" for reason, count in rejection_counts.items()
        )
        status_text += f" (Rejected: {reasons_summary})"
    
    button_text = f"🔬 Propagate Masks on {ready_for_propagation_count} Ready Scenes"
    return status_text, gr.update(value=button_text, interactive=ready_for_propagation_count > 0)


def toggle_scene_status(
    scenes_list: list['Scene'],
    selected_shot_id: int,
    new_status: str,
    output_folder: str,
    logger: 'AppLogger'
) -> tuple[list, str, str, Any]:
    """
    Toggle the status of a selected scene.
    
    Args:
        scenes_list: List of Scene objects
        selected_shot_id: ID of the scene to toggle
        new_status: New status ('included' or 'excluded')
        output_folder: Output folder path
        logger: Application logger
        
    Returns:
        Tuple of (updated_scenes, status_text, message, button_update)
    """
    if selected_shot_id is None or not scenes_list:
        status_text, button_update = get_scene_status_text(scenes_list)
        return scenes_list, status_text, "No scene selected.", button_update
    
    scene_to_update = next(
        (s for s in scenes_list if s.shot_id == selected_shot_id), 
        None
    )
    if scene_to_update:
        scene_to_update.status = new_status
        scene_to_update.manual_status_change = True
        save_scene_seeds(scenes_list, output_folder, logger)
        status_text, button_update = get_scene_status_text(scenes_list)
        return (
            scenes_list, 
            status_text, 
            f"Scene {selected_shot_id} status set to {new_status}.", 
            button_update
        )
    
    status_text, button_update = get_scene_status_text(scenes_list)
    return (
        scenes_list, 
        status_text, 
        f"Could not find scene {selected_shot_id}.", 
        button_update
    )


def _create_analysis_context(config: 'Config', logger: 'AppLogger', thumbnail_manager: 'ThumbnailManager',
                             cuda_available: bool, ana_ui_map_keys: list[str], ana_input_components: list,
                             model_registry: 'ModelRegistry') -> 'SubjectMasker':
    """Helper to initialize a SubjectMasker from UI arguments."""
    from core.models import AnalysisParameters
    ui_args = dict(zip(ana_ui_map_keys, ana_input_components))
    if 'outputfolder' in ui_args and 'output_folder' not in ui_args: ui_args['output_folder'] = ui_args.pop('outputfolder')
    output_folder_str = ui_args.get('output_folder')
    if not output_folder_str or isinstance(output_folder_str, bool):
        logger.error(f"Output folder is not valid (was '{output_folder_str}', type: {type(output_folder_str)}). This is likely due to a UI argument mapping error.", component="analysis")
        raise FileNotFoundError(f"Output folder is not valid or does not exist: {output_folder_str}")
    if not Path(output_folder_str).exists(): raise FileNotFoundError(f"Output folder is not valid or does not exist: {output_folder_str}")
    resolved_outdir = Path(output_folder_str).resolve()
    ui_args['output_folder'] = str(resolved_outdir)
    params = AnalysisParameters.from_ui(logger, config, **ui_args)
    models = initialize_analysis_models(params, config, logger, model_registry)
    frame_map = create_frame_map(resolved_outdir, logger)
    if not frame_map: raise RuntimeError("Failed to create frame map. Check if frame_map.json exists and is valid.")
    return SubjectMasker(
        params=params, progress_queue=Queue(), cancel_event=threading.Event(), config=config,
        frame_map=frame_map, face_analyzer=models["face_analyzer"],
        reference_embedding=models["ref_emb"],
        niqe_metric=None, thumbnail_manager=thumbnail_manager, logger=logger,
        face_landmarker=models["face_landmarker"], device=models["device"],
        model_registry=model_registry
    )


def _recompute_single_preview(scene_state: 'SceneState', masker: 'SubjectMasker', overrides: dict,
                              thumbnail_manager: 'ThumbnailManager', logger: 'AppLogger'):
    """Re-runs the seeding process for a single scene and updates its preview image."""
    scene = scene_state.scene # Use .scene property if using refactored SceneState
    out_dir = Path(masker.params.output_folder)
    best_frame_num = scene.best_frame or scene.start_frame
    if best_frame_num is None: raise ValueError(f"Scene {scene.shot_id} has no best frame number.")
    fname = masker.frame_map.get(int(best_frame_num))
    if not fname: raise FileNotFoundError(f"Best frame {best_frame_num} not found in project's frame map.")
    thumb_rgb = thumbnail_manager.get(out_dir / "thumbs" / f"{Path(fname).stem}.webp")
    if thumb_rgb is None: raise FileNotFoundError(f"Thumbnail for frame {best_frame_num} not found on disk.")
    seed_config = {**masker.params.model_dump(), **overrides}
    if overrides.get("text_prompt", "").strip():
        seed_config['primary_seed_strategy'] = "📝 By Text"
        logger.info(f"Recomputing scene {scene.shot_id} with text-first strategy due to override.", extra={'prompt': overrides.get("text_prompt")})
    bbox, details = masker.get_seed_for_frame(thumb_rgb, seed_config=seed_config, scene=scene)
    scene_state.update_seed_result(bbox, details)
    scene.seed_config.update(overrides)
    new_score = details.get('final_score') or details.get('conf') or details.get('dino_conf')
    if new_score is not None:
        if not scene.seed_metrics: scene.seed_metrics = {}
        scene.seed_metrics['score'] = new_score
    mask = masker.get_mask_for_bbox(thumb_rgb, bbox) if bbox else None
    if mask is not None:
        h, w = mask.shape[:2]; area = (h * w)
        if not scene.seed_result.get('details'): scene.seed_result['details'] = {}
        scene.seed_result['details']['mask_area_pct'] = (np.sum(mask > 0) / area * 100.0) if area > 0 else 0.0
    overlay_rgb = render_mask_overlay(thumb_rgb, mask, 0.6, logger=logger) if mask is not None else masker.draw_bbox(thumb_rgb, bbox)
    previews_dir = out_dir / "previews"; previews_dir.mkdir(parents=True, exist_ok=True)
    preview_path = previews_dir / f"scene_{int(scene.shot_id):05d}.jpg"
    try:
        Image.fromarray(overlay_rgb).save(preview_path)
        scene.preview_path = str(preview_path)
    except Exception: logger.error(f"Failed to save preview for scene {scene.shot_id}", exc_info=True)


def _wire_recompute_handler(config: 'Config', logger: 'AppLogger', thumbnail_manager: 'ThumbnailManager',
                            scenes: list['Scene'], shot_id: int, outdir: str, text_prompt: str,
                            view: str, ana_ui_map_keys: list[str],
                            ana_input_components: list, cuda_available: bool, model_registry: 'ModelRegistry') -> tuple:
    """Gradio event handler for the 'Recompute' button in the scene editor."""
    import gradio as gr
    from core.models import SceneState
    try:
        if not text_prompt or not text_prompt.strip(): return scenes, gr.update(), gr.update(), "Enter a text prompt to use advanced seeding."
        ui_args = dict(zip(ana_ui_map_keys, ana_input_components))
        ui_args['output_folder'] = outdir
        masker = _create_analysis_context(config, logger, thumbnail_manager, cuda_available, ana_ui_map_keys, ana_input_components, model_registry)
        scene_idx = next((i for i, s in enumerate(scenes) if s.shot_id == shot_id), None)
        if scene_idx is None: return scenes, gr.update(), gr.update(), f"Error: Scene {shot_id} not found."
        overrides = {"text_prompt": text_prompt}
        scene_state = SceneState(scenes[scene_idx])
        _recompute_single_preview(scene_state, masker, overrides, thumbnail_manager, logger)
        save_scene_seeds(scenes, outdir, logger)
        gallery_items, index_map, _ = build_scene_gallery_items(scenes, view, outdir)
        msg = f"Scene {shot_id} preview recomputed successfully."
        return scenes, gr.update(value=gallery_items), gr.update(value=index_map), msg
    except Exception as e:
        logger.error("Failed to recompute scene preview", exc_info=True)
        gallery_items, index_map, _ = build_scene_gallery_items(scenes, view, outdir)
        return scenes, gr.update(value=gallery_items), gr.update(value=index_map), f"[ERROR] Recompute failed: {str(e)}"
