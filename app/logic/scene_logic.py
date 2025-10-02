import json
from pathlib import Path
from dataclasses import asdict
from queue import Queue
import threading

import cv2
import numpy as np

from app.core.config import Config
from app.core.logging import UnifiedLogger
from app.core.utils import _to_json_safe
from app.domain.models import AnalysisParameters
from app.io.frames import render_mask_overlay
from app.masking.subject_masker import SubjectMasker


def save_scene_seeds(scenes_list, output_dir_str, logger):
    """Save scene seeds to JSON file."""
    if not scenes_list or not output_dir_str:
        return
    output_dir = Path(output_dir_str)
    scene_seeds = {
        str(s['shot_id']): {
            'seed_frame_idx': s.get('best_seed_frame'),
            'seed_type': s.get('seed_result', {}).get('details', {}).get('type'),
            'seed_config': s.get('seed_config', {}),
            'status': s.get('status', 'pending'),
            'seed_metrics': s.get('seed_metrics', {})
        }
        for s in scenes_list
    }
    try:
        (output_dir / "scene_seeds.json").write_text(
            json.dumps(_to_json_safe(scene_seeds), indent=2),
            encoding='utf-8'
        )
        logger.info("Saved scene_seeds.json")
    except Exception as e:
        logger.error("Failed to save scene_seeds.json", exc_info=True)


def get_scene_status_text(scenes_list):
    """Get status text for scenes."""
    if not scenes_list:
        return "No scenes loaded."
    num_included = sum(1 for s in scenes_list if s['status'] == 'included')
    return f"{num_included}/{len(scenes_list)} scenes included for propagation."


def toggle_scene_status(scenes_list, selected_shot_id, new_status,
                        output_folder, logger):
    """Toggle the status of a specific scene."""
    if selected_shot_id is None or not scenes_list:
        return (scenes_list, get_scene_status_text(scenes_list),
               "No scene selected.")

    scene_found = False
    for s in scenes_list:
        if s['shot_id'] == selected_shot_id:
            s['status'] = new_status
            s['manual_status_change'] = True
            scene_found = True
            break

    if scene_found:
        save_scene_seeds(scenes_list, output_folder, logger)
        return (scenes_list, get_scene_status_text(scenes_list),
               f"Scene {selected_shot_id} status set to {new_status}.")
    else:
        return (scenes_list, get_scene_status_text(scenes_list),
               f"Could not find scene {selected_shot_id}.")


def apply_bulk_scene_filters(scenes, min_mask_area, min_face_sim,
                             enable_face_filter, output_folder, logger):
    """Apply bulk filters to scenes."""
    if not scenes:
        return [], "No scenes to filter."

    for scene in scenes:
        # Reset manual status on bulk filter application so sliders work
        # after "Include/Exclude All" has been clicked.
        scene['manual_status_change'] = False

        is_excluded = False
        seed_result = scene.get('seed_result', {})
        details = seed_result.get('details', {})
        seed_metrics = scene.get('seed_metrics', {})

        mask_area = details.get('mask_area_pct', 101)
        if mask_area < min_mask_area:
            is_excluded = True

        if enable_face_filter and not is_excluded:
            # Get face_sim from seed_metrics, not details.
            # Default to a high value so scenes without a score are not excluded.
            face_sim = seed_metrics.get('face_sim', 1.01)
            if face_sim < min_face_sim:
                is_excluded = True

        scene['status'] = 'excluded' if is_excluded else 'included'

    save_scene_seeds(scenes, output_folder, logger)
    return scenes, get_scene_status_text(scenes)


def apply_scene_overrides(scenes_list, selected_shot_id, prompt,
                        box_th, text_th, output_folder, ana_ui_map_keys,
                        ana_input_components, cuda_available,
                        thumbnail_manager, logger):
    """Apply overrides to a specific scene."""
    if selected_shot_id is None or not scenes_list:
        return (None, scenes_list,
               "No scene selected to apply overrides.")

    scene_idx, scene_dict = next(
        ((i, s) for i, s in enumerate(scenes_list)
         if s['shot_id'] == selected_shot_id), (None, None)
    )
    if scene_dict is None:
        return (None, scenes_list,
               "Error: Selected scene not found in state.")

    try:
        scene_dict['seed_config'] = {
            'text_prompt': prompt,
            'box_threshold': box_th,
            'text_threshold': text_th,
        }

        ui_args = dict(zip(ana_ui_map_keys, ana_input_components))
        ui_args['output_folder'] = output_folder
        params = AnalysisParameters.from_ui(**ui_args)

        face_analyzer, ref_emb, person_detector = None, None, None
        device = "cuda" if cuda_available else "cpu"
        if params.enable_face_filter:
            from app.ml.face import get_face_analyzer
            face_analyzer = get_face_analyzer(params.face_model_name)
            if params.face_ref_img_path:
                ref_img = cv2.imread(params.face_ref_img_path)
                if ref_img is not None:
                    faces = face_analyzer.get(ref_img)
                    if faces:
                        ref_emb = max(faces, key=lambda x: x.det_score).normed_embedding
        from app.ml.person import get_person_detector
        person_detector = get_person_detector(params.person_detector_model, device)

        masker = SubjectMasker(params, Queue(), threading.Event(),
                             face_analyzer=face_analyzer,
                             reference_embedding=ref_emb,
                             person_detector=person_detector,
                             thumbnail_manager=thumbnail_manager)
        masker.frame_map = masker._create_frame_map(output_folder)

        fname = masker.frame_map.get(scene_dict['best_seed_frame'])
        if not fname:
            raise ValueError("Framemap lookup failed for re-seeding.")

        thumb_path = (Path(output_folder) / "thumbs" /
                     f"{Path(fname).stem}.webp")
        thumb_rgb = thumbnail_manager.get(thumb_path)

        bbox, details = masker.get_seed_for_frame(thumb_rgb,
                                                scene_dict['seed_config'])
        scene_dict['seed_result'] = {'bbox': bbox, 'details': details}

        save_scene_seeds(scenes_list, output_folder, logger)

        updated_gallery_previews = _regenerate_all_previews(
            scenes_list, output_folder, masker, thumbnail_manager
        )

        return (updated_gallery_previews, scenes_list,
               f"Scene {selected_shot_id} updated and saved.")

    except Exception as e:
        logger.error("Failed to apply scene overrides", exc_info=True)
        return None, scenes_list, f"[ERROR] {e}"


def _regenerate_all_previews(scenes_list, output_folder, masker,
                            thumbnail_manager):
    """Regenerate all scene previews."""
    previews = []
    output_dir = Path(output_folder)

    for scene_dict in scenes_list:
        fname = masker.frame_map.get(scene_dict['best_seed_frame'])
        if not fname:
            continue

        thumb_path = output_dir / "thumbs" / f"{Path(fname).stem}.webp"
        thumb_rgb = thumbnail_manager.get(thumb_path)
        if thumb_rgb is None:
            continue

        bbox = scene_dict.get('seed_result', {}).get('bbox')
        details = scene_dict.get('seed_result', {}).get('details', {})

        mask = masker.get_mask_for_bbox(thumb_rgb, bbox) if bbox else None
        overlay_rgb = (render_mask_overlay(thumb_rgb, mask)
                      if mask is not None
                      else masker.draw_bbox(thumb_rgb, bbox))

        caption = (f"Scene {scene_dict['shot_id']} "
                  f"(Seed: {scene_dict['best_seed_frame']}) | "
                  f"{details.get('type', 'N/A')}")
        previews.append((overlay_rgb, caption))
    return previews