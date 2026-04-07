import json
import shutil
import threading
from pathlib import Path
from queue import Queue
from typing import TYPE_CHECKING, Callable, Generator, Optional

from PIL import Image

# For test discovery (TestImportSmoke.test_pipelines_has_all_imports)
Image = Image

if TYPE_CHECKING:
    from core.config import Config
    from core.database import Database
    from core.logger import AppLogger
    from core.managers import ThumbnailManager

from core.error_handling import handle_common_errors
from core.events import ExtractionEvent, PreAnalysisEvent, PropagationEvent, SessionLoadEvent
from core.managers import (
    AnalysisPipeline,
    ExtractionPipeline,
    ModelRegistry,
    PreAnalysisPipeline,
    VideoManager,
    _load_analysis_scenes,
    _load_scenes,
    initialize_analysis_models,
)
from core.managers import (
    execute_session_load as _execute_session_load,
)
from core.managers import (
    validate_session_dir as _validate_session_dir,
)
from core.models import AnalysisParameters
from core.operators import OperatorRegistry
from core.pipeline_results import (
    AnalysisResult,
    ExtractionResult,
    PreAnalysisResult,
    PropagationResult,
)
from core.progress import AdvancedProgressTracker
from core.utils import (
    estimate_totals,
)
from core.utils.device import get_device


def _handle_extraction_uploads(event_dict: dict, config: "Config") -> dict:
    """Helper to move uploaded video to downloads directory."""
    if event_dict.get("upload_video"):
        source = Path(event_dict.pop("upload_video"))
        dest = Path(config.downloads_dir) / source.name
        shutil.copy2(source, dest)
        event_dict["source_path"] = str(dest)
    return event_dict


def _handle_pre_analysis_uploads(event_dict: dict, config: "Config") -> dict:
    """Helper to move uploaded face reference image to downloads directory."""
    if event_dict.get("face_ref_img_upload"):
        ref = Path(event_dict.pop("face_ref_img_upload"))
        dest = Path(config.downloads_dir) / ref.name
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
    model_registry: "ModelRegistry",
    thumbnail_manager: Optional["ThumbnailManager"] = None,
    cuda_available: Optional[bool] = None,
    progress: Optional[Callable] = None,
) -> Generator[dict, None, None]:
    # Ensure OperatorRegistry is initialized
    OperatorRegistry.initialize_all(config)

    event_dict = _handle_extraction_uploads(event.model_dump(), config)
    params = AnalysisParameters.from_ui(logger, config, **event_dict)

    device = "cuda" if cuda_available else "cpu" if cuda_available is not None else get_device()
    tracker = AdvancedProgressTracker(progress, progress_queue, logger, ui_stage_name="Extracting", device=device)
    pipeline = ExtractionPipeline(
        config, logger, params, progress_queue, cancel_event, model_registry=model_registry, device=device
    )
    result = pipeline.run(tracker=tracker)

    if result and result.get("done"):
        # Fingerprint logic is best-effort
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
        except Exception as e:
            logger.warning(f"Failed to save fingerprint for extraction session: {e}", exc_info=True)

        msg = "Extraction Complete."
        logger.info(msg)
        yield ExtractionResult(
            unified_log=msg,
            extracted_video_path_state=result.get("video_path", ""),
            extracted_frames_dir_state=result["output_dir"],
            done=True,
        ).model_dump()
    else:
        error_log = result.get("log") if result else "Unknown error"
        msg = f"Extraction failed: {error_log}"
        logger.error(msg)
        yield ExtractionResult(success=False, unified_log=msg, done=False).model_dump()


@handle_common_errors
def execute_pre_analysis(
    event: "PreAnalysisEvent",
    progress_queue: Queue,
    cancel_event: threading.Event,
    logger: "AppLogger",
    config: "Config",
    thumbnail_manager: "ThumbnailManager",
    model_registry: "ModelRegistry",
    cuda_available: bool,
    progress: Optional[Callable] = None,
    loaded_models: Optional[dict] = None,
) -> Generator[dict, None, None]:
    tracker = AdvancedProgressTracker(progress, progress_queue, logger, ui_stage_name="Pre-Analysis")
    event_dict = _handle_pre_analysis_uploads(event.model_dump(), config)
    params = AnalysisParameters.from_ui(logger, config, **event_dict)
    out_dir = Path(params.output_folder)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save run config
    (out_dir / "run_config.json").write_text(
        json.dumps({k: v for k, v in event_dict.items() if k != "face_ref_img_upload"}, indent=4)
    )

    scenes = _load_scenes(out_dir)
    tracker.start(len(scenes), desc="Pre-analyzing")

    # Ensure OperatorRegistry is initialized
    OperatorRegistry.initialize_all(config)
    device = "cuda" if cuda_available else "cpu"
    pipeline = PreAnalysisPipeline(
        config,
        logger,
        params,
        progress_queue,
        cancel_event,
        thumbnail_manager,
        model_registry,
        loaded_models=loaded_models,
        device=device,
    )
    processed = pipeline.run(scenes, tracker=tracker)

    msg = "Pre-Analysis Complete."
    logger.info(msg)

    # Fix Issue 6: Use typed PreAnalysisResult with show_results flag
    res = PreAnalysisResult(
        unified_log="Pre-Analysis Complete. (Compute Metrics to continue)",
        scenes=[s.model_dump() for s in processed],
        output_dir=str(out_dir),
        video_path=params.video_path,
        done=True,
        show_results=True,
    )

    yield res.model_dump()


def validate_session_dir(path: str) -> bool:
    p, err = _validate_session_dir(path)
    return p is not None and err is None


def execute_session_load(event: SessionLoadEvent | dict, logger: "AppLogger") -> dict:
    if isinstance(event, dict):
        event = SessionLoadEvent(**event)
    return _execute_session_load(event, logger)


@handle_common_errors
def execute_propagation(
    event: PropagationEvent,
    progress_queue: Queue,
    cancel_event: threading.Event,
    logger: "AppLogger",
    config: "Config",
    thumbnail_manager: "ThumbnailManager",
    model_registry: "ModelRegistry",
    database: "Database",
    cuda_available: bool,
    progress: Optional[Callable] = None,
    loaded_models: Optional[dict] = None,
) -> Generator[dict, None, None]:
    params = AnalysisParameters.from_ui(logger, config, **event.analysis_params.model_dump())
    is_folder = not params.video_path
    scenes = _load_analysis_scenes(event.scenes, is_folder)

    if not scenes:
        yield {"done": True}
        return

    device = "cuda" if cuda_available else "cpu"
    tracker = AdvancedProgressTracker(progress, progress_queue, logger, ui_stage_name="Analysis", device=device)
    if is_folder:
        tracker.start(len(scenes), desc="Analyzing Images")
    else:
        v_info = VideoManager.get_video_info(params.video_path)
        totals = estimate_totals(params, v_info, scenes)
        tracker.start(totals.get("propagation", 0) + len(scenes), desc="Propagating Masks")

    # Ensure OperatorRegistry is initialized and Database is pointed to session dir
    OperatorRegistry.initialize_all(config)
    database.set_db_path(Path(params.output_folder) / "metadata.db")

    pipeline = AnalysisPipeline(
        config,
        logger,
        params,
        progress_queue,
        cancel_event,
        thumbnail_manager,
        model_registry,
        database,
        loaded_models=loaded_models,
        device=device,
    )
    result = pipeline.run_full_analysis(scenes, tracker=tracker)

    if result and result.get("done"):
        masks_dir = Path(result["output_dir"]) / "masks"
        n = len(list(masks_dir.glob("*.png"))) if masks_dir.exists() else 0
        msg = f"Propagation Complete. {n} masks generated."
        logger.info(msg)
        yield PropagationResult(
            unified_log=msg,
            output_dir=result["output_dir"],
            mask_count=n,
            done=True,
        ).model_dump()
    else:
        error_msg = result.get("error") if result else "Unknown error"
        msg = f"Propagation failed: {error_msg}"
        logger.error(msg)
        yield PropagationResult(success=False, unified_log=msg, done=False).model_dump()


@handle_common_errors
def execute_analysis(
    event: PropagationEvent,
    progress_queue: Queue,
    cancel_event: threading.Event,
    logger: "AppLogger",
    config: "Config",
    thumbnail_manager: "ThumbnailManager",
    model_registry: "ModelRegistry",
    database: "Database",
    cuda_available: bool,
    progress: Optional[Callable] = None,
    loaded_models: Optional[dict] = None,
) -> Generator[dict, None, None]:
    params = AnalysisParameters.from_ui(logger, config, **event.analysis_params.model_dump())
    scenes = _load_analysis_scenes(event.scenes, not params.video_path)

    if not scenes:
        yield {"done": True}
        return

    device = "cuda" if cuda_available else "cpu"
    tracker = AdvancedProgressTracker(progress, progress_queue, logger, ui_stage_name="Analyzing", device=device)
    tracker.start(sum(s.end_frame - s.start_frame for s in scenes), desc="Analyzing")

    # Ensure OperatorRegistry is initialized and Database is pointed to session dir
    OperatorRegistry.initialize_all(config)
    database.set_db_path(Path(params.output_folder) / "metadata.db")

    pipeline = AnalysisPipeline(
        config,
        logger,
        params,
        progress_queue,
        cancel_event,
        thumbnail_manager,
        model_registry,
        database,
        loaded_models=loaded_models,
        device=device,
    )
    result = pipeline.run_analysis_only(scenes, tracker=tracker)

    if result and result.get("done"):
        msg = "Analysis Complete."
        logger.info(msg)
        yield AnalysisResult(
            unified_log=msg,
            output_dir=result["output_dir"],
            metadata_path=str(Path(result["output_dir"]) / "metadata.db"),
            done=True,
        ).model_dump()
    else:
        error_msg = result.get("error") if result else "Unknown error"
        yield AnalysisResult(success=False, unified_log=f"Analysis failed: {error_msg}", done=False).model_dump()


@handle_common_errors
def execute_analysis_orchestrator(
    event: "PreAnalysisEvent",
    progress_queue: Queue,
    cancel_event: threading.Event,
    logger: "AppLogger",
    config: "Config",
    thumbnail_manager: "ThumbnailManager",
    model_registry: "ModelRegistry",
    database: "Database",
    cuda_available: bool,
    progress: Optional[Callable] = None,
) -> Generator[dict, None, None]:
    """Orchestrates Pre-Analysis, Propagation, and Analysis stages."""
    # Ensure OperatorRegistry is initialized
    OperatorRegistry.initialize_all(config)

    # Initialize models once for all stages
    params = AnalysisParameters.from_ui(logger, config, **event.model_dump())
    loaded_models = initialize_analysis_models(params, config, logger, model_registry)
    pre_result = None

    # 1. Pre-Analysis
    pre_gen = execute_pre_analysis(
        event=event,
        progress_queue=progress_queue,
        cancel_event=cancel_event,
        logger=logger,
        config=config,
        thumbnail_manager=thumbnail_manager,
        model_registry=model_registry,
        cuda_available=cuda_available,
        progress=progress,
        loaded_models=loaded_models,
    )
    for res in pre_gen:
        if res.get("done"):
            pre_result = res
            # Strip 'done' so UI doesn't stop consuming the orchestrator
            clean_res = res.copy()
            clean_res.pop("done")
            if clean_res:
                yield clean_res
            continue
        yield res

    if not pre_result or not pre_result.get("done"):
        return

    scenes = pre_result.get("scenes", [])
    is_video = bool(event.video_path)

    # 2. Propagation (if video)
    prop_event = PropagationEvent(
        output_folder=event.output_folder,
        video_path=event.video_path,
        scenes=scenes,
        analysis_params=event,
    )

    if is_video:
        prop_gen = execute_propagation(
            event=prop_event,
            progress_queue=progress_queue,
            cancel_event=cancel_event,
            logger=logger,
            config=config,
            thumbnail_manager=thumbnail_manager,
            model_registry=model_registry,
            database=database,
            cuda_available=cuda_available,
            progress=progress,
            loaded_models=loaded_models,
        )
        for res in prop_gen:
            if res.get("done"):
                # Strip 'done' so UI doesn't stop consuming the orchestrator
                clean_res = res.copy()
                clean_res.pop("done")
                if clean_res:
                    yield clean_res
                continue
            yield res
    else:
        msg = "Mask Propagation (Skipped for Folder)"
        logger.info(msg)
        yield {"unified_log": msg}

    # 3. Analysis
    ana_gen = execute_analysis(
        event=prop_event,
        progress_queue=progress_queue,
        cancel_event=cancel_event,
        logger=logger,
        config=config,
        thumbnail_manager=thumbnail_manager,
        model_registry=model_registry,
        database=database,
        cuda_available=cuda_available,
        progress=progress,
        loaded_models=loaded_models,
    )
    yield from ana_gen


@handle_common_errors
def execute_full_pipeline(
    event: "ExtractionEvent",
    progress_queue: Queue,
    cancel_event: threading.Event,
    logger: "AppLogger",
    config: "Config",
    thumbnail_manager: "ThumbnailManager",
    model_registry: "ModelRegistry",
    database: "Database",
    cuda_available: bool,
    progress: Optional[Callable] = None,
) -> Generator[dict, None, None]:
    """Orchestrates the entire flow: Extraction -> Pre-Analysis -> Propagation -> Analysis."""
    # 1. Extraction
    ext_gen = execute_extraction(
        event=event,
        progress_queue=progress_queue,
        cancel_event=cancel_event,
        logger=logger,
        config=config,
        model_registry=model_registry,
        thumbnail_manager=thumbnail_manager,
        cuda_available=cuda_available,
        progress=progress,
    )
    ext_result = {}
    for res in ext_gen:
        if res.get("done"):
            ext_result = res
            # Strip 'done' so UI doesn't stop consuming the orchestrator
            clean_res = res.copy()
            clean_res.pop("done")
            if clean_res:
                yield clean_res
            continue
        yield res

    if not ext_result or not ext_result.get("done"):
        return

    # 2. Build Pre-Analysis Event from Extraction Result
    # This logic matches run_full's behavior in cli_commands.py
    pre_event = PreAnalysisEvent(
        output_folder=ext_result["extracted_frames_dir_state"],
        video_path=ext_result["extracted_video_path_state"],
        face_ref_img_path="",  # Default to "Find Prominent Person" unless provided
        primary_seed_strategy="Automatic Detection",
        resume=False,
    )

    yield {"unified_log": "Moving to Analysis stages...", "done": False}

    # 3. Chain to Analysis Orchestrator
    yield from execute_analysis_orchestrator(
        event=pre_event,
        progress_queue=progress_queue,
        cancel_event=cancel_event,
        logger=logger,
        config=config,
        thumbnail_manager=thumbnail_manager,
        model_registry=model_registry,
        database=database,
        cuda_available=cuda_available,
        progress=progress,
    )
