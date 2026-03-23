import json
import shutil
import threading
from pathlib import Path
from queue import Queue
from typing import TYPE_CHECKING, Callable, Generator, Optional

if TYPE_CHECKING:
    from core.config import Config
    from core.logger import AppLogger
    from core.managers import ThumbnailManager

from core.events import ExtractionEvent, PreAnalysisEvent, PropagationEvent
from core.managers import (
    AnalysisPipeline,
    ExtractionPipeline,
    ModelRegistry,
    PreAnalysisPipeline,
    VideoManager,
    _load_analysis_scenes,
    _load_scenes,
)
from core.managers import (
    execute_session_load as _execute_session_load,
)
from core.managers import (
    validate_session_dir as _validate_session_dir,
)
from core.models import AnalysisParameters
from core.progress import AdvancedProgressTracker
from core.utils import (
    estimate_totals,
    handle_common_errors,
)


def _initialize_extraction_params(event_dict, config, logger):
    return AnalysisParameters.from_ui(logger, config, **event_dict)


def _handle_pre_analysis_uploads(event_dict, config):
    if event_dict.get("face_ref_img_upload"):
        ref = event_dict.pop("face_ref_img_upload")
        dest = Path(config.downloads_dir) / Path(ref).name
        shutil.copy2(ref, dest)
        event_dict["face_ref_img_path"] = str(dest)
    return event_dict


@handle_common_errors
def execute_extraction(
    event: "ExtractionEvent",
    progress_queue: Queue,
    cancel_event: threading.Event,
    logger: "AppLogger",
    config: "Config",
    thumbnail_manager: Optional["ThumbnailManager"] = None,
    cuda_available: Optional[bool] = None,
    progress: Optional[Callable] = None,
    model_registry: Optional["ModelRegistry"] = None,
) -> Generator[dict, None, None]:
    try:
        from core.pipelines import _handle_extraction_uploads  # Local import or define here

        event_dict = _handle_extraction_uploads(event.model_dump(), config)
        from core.models import AnalysisParameters

        params = AnalysisParameters.from_ui(logger, config, **event_dict)
        tracker = AdvancedProgressTracker(progress, progress_queue, logger, ui_stage_name="Extracting")
        pipeline = ExtractionPipeline(
            config, logger, params, progress_queue, cancel_event, model_registry=model_registry
        )
        result = pipeline.run(tracker=tracker)
        if result and result.get("done"):
            try:
                from core.fingerprint import create_fingerprint, save_fingerprint

                ext_settings = {
                    k: getattr(params, k)
                    for k in ["method", "nth_frame", "max_resolution", "scene_detect", "thumb_megapixels"]
                    if hasattr(params, k)
                }
                save_fingerprint(
                    create_fingerprint(video_path=result.get("video_path"), extraction_settings=ext_settings),
                    result.get("output_dir"),
                )
            except Exception:
                pass
            yield {
                "unified_log": "Extraction complete.",
                "extracted_video_path_state": result.get("video_path", ""),
                "extracted_frames_dir_state": result["output_dir"],
                "done": True,
            }
        else:
            yield {"unified_log": f"Extraction failed: {result.get('log')}", "done": False}
    except Exception as e:
        logger.error("Extraction failed", exc_info=True)
        yield {"unified_log": f"Error: {e}", "done": False}


def _handle_extraction_uploads(event_dict: dict, config: "Config") -> dict:
    if event_dict.get("upload_video"):
        import shutil

        source = event_dict.pop("upload_video")
        dest = str(Path(config.downloads_dir) / Path(source).name)
        shutil.copy2(source, dest)
        event_dict["source_path"] = dest
    return event_dict


@handle_common_errors
def execute_pre_analysis(
    event: "PreAnalysisEvent",
    progress_queue: Queue,
    cancel_event: threading.Event,
    logger: "AppLogger",
    config: "Config",
    thumbnail_manager: "ThumbnailManager",
    cuda_available: bool,
    progress: Optional[Callable] = None,
    model_registry: "ModelRegistry" = None,
) -> Generator[dict, None, None]:
    import gradio as gr

    try:
        tracker = AdvancedProgressTracker(progress, progress_queue, logger, ui_stage_name="Pre-Analysis")
        event_dict = _handle_pre_analysis_uploads(event.model_dump(), config)
        from core.models import AnalysisParameters

        params = AnalysisParameters.from_ui(logger, config, **event_dict)
        out_dir = Path(params.output_folder)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "run_config.json").write_text(
            json.dumps({k: v for k, v in event_dict.items() if k != "face_ref_img_upload"}, indent=4)
        )
        scenes = _load_scenes(out_dir)
        tracker.start(len(scenes), desc="Pre-analyzing")
        pipeline = PreAnalysisPipeline(
            config, logger, params, progress_queue, cancel_event, thumbnail_manager, model_registry
        )
        processed = pipeline.run(scenes, tracker=tracker)
        yield {
            "unified_log": "Pre-analysis complete.",
            "scenes": [s.model_dump() for s in processed],
            "output_dir": str(out_dir),
            "done": True,
            "seeding_results_column": gr.update(visible=True),
            "propagation_group": gr.update(visible=True),
        }
    except Exception as e:
        logger.error("Pre-analysis failed", exc_info=True)
        yield {"unified_log": f"Error: {e}", "done": False}


def validate_session_dir(path):
    return _validate_session_dir(path)


def execute_session_load(event, logger):
    return _execute_session_load(event, logger)


def execute_propagation(
    event: PropagationEvent,
    progress_queue: Queue,
    cancel_event: threading.Event,
    logger: "AppLogger",
    config: "Config",
    thumbnail_manager,
    cuda_available,
    progress=None,
    model_registry=None,
) -> Generator[dict, None, None]:
    try:
        from core.models import AnalysisParameters

        params = AnalysisParameters.from_ui(logger, config, **event.analysis_params.model_dump())
        is_folder = not params.video_path
        scenes = _load_analysis_scenes(event.scenes, is_folder)
        if not scenes:
            yield {"done": True}
            return
        tracker = AdvancedProgressTracker(progress, progress_queue, logger, ui_stage_name="Analysis")
        if is_folder:
            tracker.start(len(scenes), desc="Analyzing Images")
        else:
            v_info = VideoManager.get_video_info(params.video_path)
            totals = estimate_totals(params, v_info, scenes)
            tracker.start(totals.get("propagation", 0) + len(scenes), desc="Propagating Masks")
        pipeline = AnalysisPipeline(
            config, logger, params, progress_queue, cancel_event, thumbnail_manager, model_registry
        )
        result = pipeline.run_full_analysis(scenes, tracker=tracker)
        if result and result.get("done"):
            masks_dir = Path(result["output_dir"]) / "masks"
            n = len(list(masks_dir.glob("*.png"))) if masks_dir.exists() else 0
            yield {
                "unified_log": f"Propagation complete. {n} masks generated.",
                "output_dir": result["output_dir"],
                "done": True,
            }
        else:
            yield {"unified_log": f"Propagation failed: {result.get('error')}", "done": False}
    except Exception as e:
        logger.error("Propagation failed", exc_info=True)
        yield {"unified_log": f"Error: {e}", "done": False}


@handle_common_errors
def execute_analysis(
    event: PropagationEvent,
    progress_queue: Queue,
    cancel_event: threading.Event,
    logger: "AppLogger",
    config: "Config",
    thumbnail_manager,
    cuda_available,
    progress=None,
    model_registry=None,
) -> Generator[dict, None, None]:
    try:
        from core.models import AnalysisParameters

        params = AnalysisParameters.from_ui(logger, config, **event.analysis_params.model_dump())
        scenes = _load_analysis_scenes(event.scenes, not params.video_path)
        if not scenes:
            yield {"done": True}
            return
        tracker = AdvancedProgressTracker(progress, progress_queue, logger, ui_stage_name="Analyzing")
        tracker.start(sum(s.end_frame - s.start_frame for s in scenes), desc="Analyzing")
        pipeline = AnalysisPipeline(
            config, logger, params, progress_queue, cancel_event, thumbnail_manager, model_registry
        )
        result = pipeline.run_analysis_only(scenes, tracker=tracker)
        if result and result.get("done"):
            yield {
                "unified_log": "Analysis complete.",
                "output_dir": result["output_dir"],
                "metadata_path": str(Path(result["output_dir"]) / "metadata.db"),
                "done": True,
            }
        else:
            yield {"unified_log": f"Analysis failed: {result.get('error')}", "done": False}
    except Exception as e:
        logger.error("Analysis failed", exc_info=True)
        yield {"unified_log": f"Error: {e}", "done": False}
