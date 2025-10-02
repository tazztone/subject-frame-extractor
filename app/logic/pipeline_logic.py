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
from app.logic.events import ExtractionEvent, PreAnalysisEvent, PropagationEvent
from app.masking.subject_masker import SubjectMasker
from app.pipelines.extract import ExtractionPipeline
from app.pipelines.analyze import AnalysisPipeline
from app.io.frames import render_mask_overlay


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
                                       logger, config, thumbnail_manager)


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
            "extracted_frames_dir_state": result["output_dir"],
            "frames_folder_input": result["output_dir"],
            "analysis_video_path_input": result.get("video_path", "")
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


def execute_propagation(event: PropagationEvent, progress_queue: Queue,
                        cancel_event: threading.Event, logger: UnifiedLogger,
                        config: Config, thumbnail_manager):
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