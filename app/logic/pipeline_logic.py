import json
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from queue import Queue, Empty
from pathlib import Path
import threading
from dataclasses import asdict

import cv2
import gradio as gr
import numpy as np

from app.core.config import Config
from app.core.logging import UnifiedLogger
from app.domain.models import Scene, AnalysisParameters
from app.logic.events import (ExtractionEvent, PreAnalysisEvent, PropagationEvent,
                              SessionLoadEvent)
from app.logic.scene_logic import get_scene_status_text
from app.masking.subject_masker import SubjectMasker
from app.pipelines.extract import ExtractionPipeline
from app.pipelines.analyze import AnalysisPipeline
from app.io.frames import render_mask_overlay, create_frame_map


def run_pipeline_logic(event, progress_queue, cancel_event, logger, config,
                       thumbnail_manager, cuda_available):
    """Dispatcher for different pipeline logic events."""
    if isinstance(event, ExtractionEvent):
        yield from execute_extraction(event, progress_queue, cancel_event,
                                     logger, config)
    elif isinstance(event, PreAnalysisEvent):
        yield from execute_pre_analysis(event, progress_queue, cancel_event,
                                        logger, config, thumbnail_manager,
                                        cuda_available)
    elif isinstance(event, PropagationEvent):
        yield from execute_propagation(event, progress_queue, cancel_event,
                                       logger, config, thumbnail_manager,
                                       cuda_available)
    elif isinstance(event, SessionLoadEvent):
        yield from execute_session_load(event, logger, config, thumbnail_manager)


def execute_extraction(event: ExtractionEvent, progress_queue: Queue,
                       cancel_event: threading.Event, logger: UnifiedLogger,
                       config: Config):
    """Execute extraction pipeline."""
    yield {"unified_log": "", "unified_status": "Starting extraction..."}

    params_dict = asdict(event)
    if event.upload_video:
        source = params_dict.pop('upload_video')
        dest = str(config.DIRS['downloads'] / Path(source).name)
        shutil.copy2(source, dest)
        params_dict['source_path'] = dest

    params = AnalysisParameters.from_ui(**params_dict)
    pipeline = ExtractionPipeline(params, progress_queue, cancel_event)

    result = yield from _run_task(pipeline.run, progress_queue, cancel_event,
                                 logger)

    if result.get("done"):
        yield {
            "unified_log": "Extraction complete.",
            "unified_status": f"Output: {result['output_dir']}",
            "extracted_video_path_state": result.get("video_path", ""),
            "extracted_frames_dir_state": result["output_dir"]
        }


def execute_pre_analysis(event: PreAnalysisEvent, progress_queue: Queue,
                         cancel_event: threading.Event, logger: UnifiedLogger,
                         config: Config, thumbnail_manager, cuda_available):
    """Execute pre-analysis pipeline."""
    import pyiqa

    yield {"unified_log": "", "unified_status": "Starting Pre-Analysis..."}

    params_dict = asdict(event)
    if event.face_ref_img_upload:
        ref_upload = params_dict.pop('face_ref_img_upload')
        dest = config.DIRS['downloads'] / Path(ref_upload).name
        shutil.copy2(ref_upload, dest)
        params_dict['face_ref_img_path'] = str(dest)

    params = AnalysisParameters.from_ui(**params_dict)

    output_dir = Path(params.output_folder)

    # Save the run configuration for resuming
    run_config_path = output_dir / "run_config.json"
    try:
        # Create a serializable copy of the event parameters
        config_to_save = params_dict.copy()
        config_to_save.pop('face_ref_img_upload', None)  # Not needed for resume

        with run_config_path.open('w', encoding='utf-8') as f:
            json.dump(config_to_save, f, indent=4)
        logger.info(f"Saved run configuration to {run_config_path}")
    except TypeError as e:
        logger.error(f"Could not serialize run configuration: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"Failed to save run configuration: {e}", exc_info=True)

    scenes_path = output_dir / "scenes.json"
    if not scenes_path.exists():
        yield {"unified_log": "[ERROR] scenes.json not found. Run extraction with scene detection."}
        return

    with scenes_path.open('r', encoding='utf-8') as f:
        shots = json.load(f)
    scenes = [Scene(shot_id=i, start_frame=s, end_frame=e) for i, (s, e) in enumerate(shots)]

    scene_seeds_path = output_dir / "scene_seeds.json"
    if scene_seeds_path.exists() and params.resume:
        logger.info("Loading existing scene_seeds.json")
        with scene_seeds_path.open('r', encoding='utf-8') as f:
            loaded_seeds = json.load(f)
        for scene in scenes:
            seed_data = loaded_seeds.get(str(scene.shot_id))
            if seed_data:
                scene.best_seed_frame = seed_data.get('seed_frame_idx')
                scene.seed_config = seed_data.get('seed_config', {})
                scene.status = seed_data.get('status', 'pending')
                scene.seed_metrics = seed_data.get('seed_metrics', {})

    niqe_metric, face_analyzer, ref_emb, person_detector = None, None, None, None
    device = "cuda" if cuda_available else "cpu"

    if params.pre_analysis_enabled:
        niqe_metric = pyiqa.create_metric('niqe', device=device)
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

    masker = SubjectMasker(params, progress_queue, cancel_event,
                         face_analyzer=face_analyzer,
                         reference_embedding=ref_emb,
                         person_detector=person_detector,
                         niqe_metric=niqe_metric,
                         thumbnail_manager=thumbnail_manager)
    masker.frame_map = masker._create_frame_map(str(output_dir))

    previews = []
    for i, scene in enumerate(scenes):
        progress_queue.put({"stage": f"Pre-analyzing scene {i+1}/{len(scenes)}", "total": len(scenes), "progress": i})

        if not scene.best_seed_frame:
            masker._select_best_seed_frame_in_scene(scene, str(output_dir))

        fname = masker.frame_map.get(scene.best_seed_frame)
        if not fname:
            logger.warning(f"Could not find best_seed_frame {scene.best_seed_frame} in frame_map for scene {scene.shot_id}")
            continue

        thumb_path = output_dir / "thumbs" / f"{Path(fname).stem}.webp"
        thumb_rgb = thumbnail_manager.get(thumb_path)

        if thumb_rgb is None:
            logger.warning(f"Could not load thumbnail for best_seed_frame {scene.best_seed_frame} at path {thumb_path}")
            continue

        bbox, details = masker.get_seed_for_frame(thumb_rgb, seed_config=scene.seed_config or params)
        scene.seed_result = {'bbox': bbox, 'details': details}

        mask = masker.get_mask_for_bbox(thumb_rgb, bbox) if bbox else None
        if mask is not None:
            h, w = mask.shape[:2]
            area_pct = ((np.sum(mask > 0) / (h * w)) * 100 if (h * w) > 0 else 0.0)
            scene.seed_result['details']['mask_area_pct'] = area_pct

        overlay_rgb = (render_mask_overlay(thumb_rgb, mask) if mask is not None else masker.draw_bbox(thumb_rgb, bbox))

        caption = f"Scene {scene.shot_id} (Seed: {scene.best_seed_frame}) | {details.get('type', 'N/A')}"
        previews.append((overlay_rgb, caption))
        scene.preview_path = "dummy"
        if scene.status == 'pending':
            scene.status = 'included'

    scenes_as_dict = [asdict(s) for s in scenes]
    # self._save_scene_seeds(scenes_as_dict, str(output_dir)) # This will be handled by scene_logic
    progress_queue.put({"stage": "Pre-analysis complete", "progress": len(scenes)})

    yield {
        "unified_log": "Pre-analysis complete.",
        "unified_status": f"{len(scenes)} scenes found.",
        "seeding_preview_gallery": gr.update(value=previews),
        "scenes_state": scenes_as_dict,
        "propagate_masks_button": gr.update(interactive=True),
        # "scene_filter_status": self._get_scene_status_text(scenes_as_dict) # This will be handled by scene_logic
    }


def execute_session_load(event: SessionLoadEvent, logger: UnifiedLogger, config: Config, thumbnail_manager):
    """Loads a session from a previous run and prepares the UI."""
    session_path = Path(event.session_path)
    config_path = session_path / "run_config.json"

    if not config_path.exists():
        logger.error(f"Session load failed: run_config.json not found in {session_path}")
        yield {
            "unified_log": f"[ERROR] Could not find 'run_config.json' in the specified directory: {session_path}",
            "unified_status": "Session load failed."
        }
        return

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            run_config = json.load(f)
        logger.info(f"Loaded run configuration from {config_path}")

        updates = {
            'source_input': gr.update(value=run_config.get('source_path', '')),
            'max_resolution': gr.update(value=run_config.get('max_resolution', '1080')),
            'thumbnails_only_input': gr.update(value=run_config.get('thumbnails_only', True)),
            'thumb_megapixels_input': gr.update(value=run_config.get('thumb_megapixels', 0.5)),
            'ext_scene_detect_input': gr.update(value=run_config.get('scene_detect', True)),
            'method_input': gr.update(value=run_config.get('method', 'scene')),
            'use_png_input': gr.update(value=run_config.get('use_png', False)),
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
            'extracted_video_path_state': run_config.get('video_path', ''),
            'extracted_frames_dir_state': run_config.get('output_folder', ''),
            'analysis_output_dir_state': run_config.get('output_folder', ''),
        }

        scene_seeds_path = session_path / "scene_seeds.json"
        if scene_seeds_path.exists():
            with open(scene_seeds_path, 'r', encoding='utf-8') as f:
                scenes_from_file = json.load(f)

            scenes_as_dict = []
            for shot_id, scene_data in scenes_from_file.items():
                scene_data['shot_id'] = int(shot_id)
                scenes_as_dict.append(scene_data)

            updates['scenes_state'] = scenes_as_dict
            updates['propagate_masks_button'] = gr.update(interactive=True)
            logger.info(f"Loaded {len(scenes_as_dict)} scenes from {scene_seeds_path}")

            output_dir = Path(run_config.get('output_folder'))
            frame_map = create_frame_map(output_dir)
            previews = []
            for scene in scenes_as_dict:
                seed_frame_idx = scene.get('seed_frame_idx')
                if seed_frame_idx is not None:
                    fname = frame_map.get(seed_frame_idx)
                    if not fname:
                        logger.warning(f"Could not find seed_frame_idx {seed_frame_idx} in frame_map for scene {scene['shot_id']}")
                        continue

                    thumb_path = output_dir / "thumbs" / f"{Path(fname).stem}.webp"
                    thumb_rgb = thumbnail_manager.get(thumb_path)

                    if thumb_rgb is None:
                        logger.warning(f"Could not load thumbnail for seed_frame_idx {seed_frame_idx} at path {thumb_path}")
                        continue

                    caption = f"Scene {scene['shot_id']} (Seed: {seed_frame_idx})"
                    previews.append((thumb_rgb, caption))

            updates['seeding_preview_gallery'] = gr.update(value=previews)
            updates['scene_filter_status'] = get_scene_status_text(scenes_as_dict)

            has_face_sim = any(
                s.get('seed_metrics', {}).get('best_face_sim') is not None
                for s in scenes_as_dict
            )
            updates['scene_face_sim_min_input'] = gr.update(visible=has_face_sim)

        metadata_path = session_path / "metadata.json"
        if metadata_path.exists():
            updates['analysis_metadata_path_state'] = str(metadata_path)
            updates['filtering_tab'] = gr.update(interactive=True)
            logger.info(f"Found analysis metadata at {metadata_path}. Filtering tab will be enabled.")

        updates['unified_log'] = f"Successfully loaded session from: {session_path}"
        updates['unified_status'] = "Session loaded. You can now proceed from where you left off."

        yield updates

    except Exception as e:
        logger.error(f"Error loading session from {session_path}: {e}", exc_info=True)
        yield {
            "unified_log": f"[ERROR] An unexpected error occurred while loading the session: {e}",
            "unified_status": "Session load failed."
        }


def execute_propagation(event: PropagationEvent, progress_queue: Queue,
                        cancel_event: threading.Event, logger: UnifiedLogger,
                        config: Config, thumbnail_manager, cuda_available):
    """Execute propagation pipeline."""
    scenes_to_process = [Scene(**s) for s in event.scenes if s['status'] == 'included']
    if not scenes_to_process:
        yield {"unified_log": "No scenes were included for propagation.", "unified_status": "Propagation skipped."}
        return

    yield {"unified_log": "", "unified_status": f"Starting propagation on {len(scenes_to_process)} scenes..."}

    params = AnalysisParameters.from_ui(**asdict(event.analysis_params))
    pipeline = AnalysisPipeline(params, progress_queue, cancel_event, thumbnail_manager=thumbnail_manager)

    result = yield from _run_task(lambda: pipeline.run_full_analysis(scenes_to_process), progress_queue, cancel_event, logger)

    if result.get("done"):
        yield {
            "unified_log": "Propagation and analysis complete.",
            "unified_status": f"Metadata saved to {result['metadata_path']}",
            "analysis_output_dir_state": result['output_dir'],
            "analysis_metadata_path_state": result['metadata_path'],
            "filtering_tab": gr.update(interactive=True)
        }


def _run_task(task_func, progress_queue, cancel_event, logger):
    """Run a task with progress tracking."""
    log_buffer, processed, total, stage = [], 0, 1, "Initializing"
    start_time, last_yield = time.time(), 0.0
    last_task_result = {}

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(task_func)
        while future.running():
            if cancel_event.is_set():
                break
            try:
                msg = progress_queue.get(timeout=0.1)
                if "log" in msg:
                    log_buffer.append(msg["log"])
                if "stage" in msg:
                    stage, processed, start_time = msg["stage"], 0, time.time()
                if "total" in msg:
                    total = msg["total"] or 1
                if "progress" in msg:
                    processed += msg["progress"]

                if time.time() - last_yield > 0.25:
                    elapsed = time.time() - start_time
                    rate = processed / elapsed if elapsed > 0 else 0
                    eta = (total - processed) / rate if rate > 0 else 0
                    status = (f"**{stage}:** {processed}/{total} "
                              f"({processed/total:.1%}) | {rate:.1f} items/s | "
                              f"ETA: {int(eta//60):02d}:{int(eta%60):02d}")
                    yield {"unified_log": "\n".join(log_buffer), "unified_status": status}
                    last_yield = time.time()
            except Empty:
                pass

    last_task_result = future.result() or {}
    if "log" in last_task_result:
        log_buffer.append(last_task_result["log"])
    if "error" in last_task_result:
        log_buffer.append(f"[ERROR] {last_task_result['error']}")

    if cancel_event.is_set():
        status_text = "⏹️ Cancelled."
    elif 'error' in last_task_result:
        status_text = f"❌ Error: {last_task_result.get('error')}"
    else:
        status_text = "✅ Complete."

    yield {"unified_log": "\n".join(log_buffer), "unified_status": status_text}

    # Return the final result so the caller can use it
    return last_task_result