import json
from pathlib import Path
import gradio as gr

from app.logic.events import SessionLoadEvent

def load_session_logic(event: SessionLoadEvent, logger):
    """Loads a session from a previous run."""
    session_path = Path(event.session_path)
    config_path = session_path / "run_config.json"

    if not config_path.exists():
        logger.error(f"Session load failed: run_config.json not found in {session_path}")
        return {
            "unified_log": f"[ERROR] Could not find 'run_config.json' in the specified directory: {session_path}",
            "unified_status": "Session load failed."
        }

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            run_config = json.load(f)

        logger.info(f"Loaded run configuration from {config_path}")

        # Prepare updates for all UI components based on the loaded config
        updates = {
            # Extraction Tab
            'source_input': gr.update(value=run_config.get('source_path', '')),
            'max_resolution': gr.update(value=run_config.get('max_resolution', '1080')),
            'thumbnails_only_input': gr.update(value=run_config.get('thumbnails_only', True)),
            'thumb_megapixels_input': gr.update(value=run_config.get('thumb_megapixels', 0.5)),
            'ext_scene_detect_input': gr.update(value=run_config.get('scene_detect', True)),
            'method_input': gr.update(value=run_config.get('method', 'scene')),
            'use_png_input': gr.update(value=run_config.get('use_png', False)),

            # Analysis Tab
            'frames_folder_input': gr.update(value=run_config.get('output_folder', '')),
            'analysis_video_path_input': gr.update(value=run_config.get('video_path', '')),
            'pre_analysis_enabled_input': gr.update(value=run_config.get('pre_analysis_enabled', True)),
            'pre_sample_nth_input': gr.update(value=run_config.get('pre_sample_nth', 1)),
            'enable_face_filter_input': gr.update(value=run_config.get('enable_face_filter', False)),
            'face_model_name_input': gr.update(value=run_config.get('face_model_name', 'buffalo_l')),
            'face_ref_img_path_input': gr.update(value=run_config.get('face_ref_img_path', '')),
            'text_prompt_input': gr.update(value=run_config.get('text_prompt', '')),
            'seed_strategy_input': gr.update(value=run_config.get('seed_strategy', 'Largest Person')),
            'person_detector_model_input': gr.update(value=run_config.get('person_detector_model', 'yolo11x.pt')),
            'dam4sam_model_name_input': gr.update(value=run_config.get('dam4sam_model_name', 'sam21pp-T')),
            'enable_dedup_input': gr.update(value=run_config.get('enable_dedup', False)),

            # States
            'extracted_video_path_state': run_config.get('video_path', ''),
            'extracted_frames_dir_state': run_config.get('output_folder', ''),
            'analysis_output_dir_state': run_config.get('output_folder', ''),
        }

        # Check for scene seeds to restore scene selection state
        scene_seeds_path = session_path / "scene_seeds.json"
        if scene_seeds_path.exists():
            with open(scene_seeds_path, 'r', encoding='utf-8') as f:
                scenes = json.load(f)
            updates['scenes_state'] = scenes
            updates['propagate_masks_button'] = gr.update(interactive=True)
            logger.info(f"Loaded {len(scenes)} scenes from {scene_seeds_path}")

        # Check for full analysis metadata to restore filtering tab
        metadata_path = session_path / "metadata.json"
        if metadata_path.exists():
            updates['analysis_metadata_path_state'] = str(metadata_path)
            updates['filtering_tab'] = gr.update(interactive=True)
            logger.info(f"Found analysis metadata at {metadata_path}. Filtering tab will be enabled.")

        # Hide manual input elements since we are loading a session
        updates['manual_input_group'] = gr.update(visible=False)
        updates['load_analysis_for_filtering_button'] = gr.update(visible=False)

        updates['unified_log'] = f"Successfully loaded session from: {session_path}"
        updates['unified_status'] = "Session loaded. You can now proceed from where you left off."

        return updates

    except Exception as e:
        logger.error(f"Error loading session from {session_path}: {e}", exc_info=True)
        return {
            "unified_log": f"[ERROR] An unexpected error occurred while loading the session: {e}",
            "unified_status": "Session load failed."
        }