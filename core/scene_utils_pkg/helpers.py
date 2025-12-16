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
