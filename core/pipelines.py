from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Generator

import gradio as gr
from PIL import Image

# For test discovery (TestImportSmoke.test_pipelines_has_all_imports)
Image = Image
gr = gr

if TYPE_CHECKING:
    from core.config import Config
    from core.events import ExtractionEvent
    from core.logger import AppLogger

from pydantic import BaseModel

from core.context import AnalysisContext
from core.events import PreAnalysisEvent, PropagationEvent, SessionLoadEvent
from core.managers import (
    AnalysisPipeline,
    ExtractionPipeline,
    MediaSession,
    ModelRegistry,
    PreAnalysisPipeline,
    _load_scenes,
    initialize_analysis_models,
)
from core.managers.media_session import (
    execute_session_load as _execute_session_load,
)
from core.managers.media_session import (
    load_analysis_scenes as _load_analysis_scenes,
)
from core.managers.media_session import (
    validate_dir as _validate_session_dir,
)
from core.models import (
    AnalysisParameters,
    AnalysisResult,
    ExtractionResult,
    PipelineFailure,
    PreAnalysisResult,
    PropagationResult,
)
from core.progress import AdvancedProgressTracker
from core.utils import (
    estimate_totals,
    handle_common_errors,
)


def _handle_extraction_uploads(event_dict: dict, config: Config) -> dict:
    """Helper to move uploaded video to downloads directory."""
    if event_dict.get("upload_video"):
        source = Path(event_dict.pop("upload_video"))
        dest = Path(config.downloads_dir) / source.name
        shutil.copy2(source, dest)
        event_dict["source_path"] = str(dest)
    return event_dict


def _handle_pre_analysis_uploads(event_dict: dict, config: Config) -> dict:
    """Helper to move uploaded face reference image to downloads directory."""
    if event_dict.get("face_ref_img_upload"):
        ref = Path(event_dict.pop("face_ref_img_upload"))
        dest = Path(config.downloads_dir) / ref.name
        shutil.copy2(ref, dest)
        event_dict["face_ref_img_path"] = str(dest)
    return event_dict


@handle_common_errors(ExtractionResult)
def execute_extraction(
    event: ExtractionEvent,
    context: AnalysisContext,
) -> Generator[BaseModel, None, None]:
    progress_queue = context.progress_queue
    cancel_event = context.cancel_event
    logger = context.logger
    config = context.config
    progress = context.progress
    model_registry = context.model_registry

    event_dict = _handle_extraction_uploads(event.model_dump(), config)
    params = AnalysisParameters.from_ui(logger, config, **event_dict)
    tracker = AdvancedProgressTracker(progress, progress_queue, logger, ui_stage_name="Extracting")
    pipeline = ExtractionPipeline(config, logger, params, progress_queue, cancel_event, model_registry=model_registry)
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
            video_path=result.get("video_path", ""),
            output_dir=result["output_dir"],
            done=True,
        )
    else:
        error_log = result.get("log") if result else "Unknown error"
        msg = f"Extraction failed: {error_log}"
        logger.error(msg)
        yield PipelineFailure(
            unified_log=msg,
            status_message="Extraction failed",
            error_message=error_log,
        )


@handle_common_errors(PreAnalysisResult)
def execute_pre_analysis(
    event: PreAnalysisEvent,
    context: AnalysisContext,
) -> Generator[BaseModel, None, None]:
    progress_queue = context.progress_queue
    cancel_event = context.cancel_event
    logger = context.logger
    config = context.config
    thumbnail_manager = context.thumbnail_manager
    progress = context.progress
    model_registry = context.model_registry
    loaded_models = context.loaded_models

    # Import gradio only when needed to keep test dependencies light
    try:
        import gradio as gr
    except ImportError:
        gr = None

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

    # Ensure model_registry is not None
    mr = model_registry or ModelRegistry(logger=logger)

    pipeline = PreAnalysisPipeline(
        config, logger, params, progress_queue, cancel_event, thumbnail_manager, mr, loaded_models=loaded_models
    )
    processed = pipeline.run(scenes, tracker=tracker)

    msg = "Pre-Analysis Complete."
    logger.info(msg)

    # Use typed PreAnalysisResult
    res = PreAnalysisResult(
        unified_log="Pre-Analysis Complete. (Compute Metrics to continue)",
        scenes=[s.model_dump() for s in processed],
        output_dir=str(out_dir),
        video_path=params.video_path,
        done=True,
    )

    if gr:
        res.seeding_results_column = gr.update(visible=True)
        res.propagation_group = gr.update(visible=True)

    yield res


def validate_session_dir(path: str) -> bool:
    p, err = _validate_session_dir(path)
    return p is not None and err is None


def execute_session_load(event: SessionLoadEvent | dict, logger: AppLogger) -> dict:
    if isinstance(event, dict):
        event = SessionLoadEvent(**event)
    return _execute_session_load(event, logger)


@handle_common_errors(PropagationResult)
def execute_propagation(
    event: PropagationEvent,
    context: AnalysisContext,
) -> Generator[BaseModel, None, None]:
    progress_queue = context.progress_queue
    cancel_event = context.cancel_event
    logger = context.logger
    config = context.config
    thumbnail_manager = context.thumbnail_manager
    progress = context.progress
    model_registry = context.model_registry
    loaded_models = context.loaded_models

    params = AnalysisParameters.from_ui(logger, config, **event.analysis_params.model_dump())
    is_folder = not params.video_path
    scenes = _load_analysis_scenes(event.scenes, is_folder)

    if not scenes:
        yield PropagationResult(unified_log="No scenes to propagate.", output_dir="", done=True)
        return

    tracker = AdvancedProgressTracker(progress, progress_queue, logger, ui_stage_name="Analysis")
    if is_folder:
        tracker.start(len(scenes), desc="Analyzing Images")
    else:
        v_info = MediaSession.get_video_info(params.video_path)
        totals = estimate_totals(params, v_info, scenes)
        tracker.start(totals.get("propagation", 0) + len(scenes), desc="Propagating Masks")

    mr = model_registry or ModelRegistry(logger=logger)
    pipeline = AnalysisPipeline(
        config, logger, params, progress_queue, cancel_event, thumbnail_manager, mr, loaded_models=loaded_models
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
            done=True,
        )
    else:
        error_msg = result.get("error") if result else "Unknown error"
        msg = f"Propagation failed: {error_msg}"
        logger.error(msg)
        yield PipelineFailure(
            unified_log=msg,
            status_message="Propagation failed",
            error_message=error_msg,
        )


@handle_common_errors(AnalysisResult)
def execute_analysis(
    event: PropagationEvent,
    context: AnalysisContext,
) -> Generator[BaseModel, None, None]:
    progress_queue = context.progress_queue
    cancel_event = context.cancel_event
    logger = context.logger
    config = context.config
    thumbnail_manager = context.thumbnail_manager
    progress = context.progress
    model_registry = context.model_registry
    loaded_models = context.loaded_models

    params = AnalysisParameters.from_ui(logger, config, **event.analysis_params.model_dump())
    scenes = _load_analysis_scenes(event.scenes, not params.video_path)

    if not scenes:
        yield AnalysisResult(unified_log="No scenes to analyze.", output_dir="", done=True)
        return

    tracker = AdvancedProgressTracker(progress, progress_queue, logger, ui_stage_name="Analyzing")
    tracker.start(sum(s.end_frame - s.start_frame for s in scenes), desc="Analyzing")

    mr = model_registry or ModelRegistry(logger=logger)
    pipeline = AnalysisPipeline(
        config, logger, params, progress_queue, cancel_event, thumbnail_manager, mr, loaded_models=loaded_models
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
        )
    else:
        error_msg = result.get("error") if result else "Unknown error"
        yield PipelineFailure(
            unified_log=f"Analysis failed: {error_msg}",
            status_message="Analysis failed",
            error_message=error_msg,
        )


@handle_common_errors(AnalysisResult)
def execute_analysis_orchestrator(
    event: PreAnalysisEvent,
    context: AnalysisContext,
) -> Generator[BaseModel, None, None]:
    """Orchestrates Pre-Analysis, Propagation, and Analysis stages."""
    logger = context.logger
    config = context.config
    model_registry = context.model_registry

    mr = model_registry or ModelRegistry(logger=logger, config=config)
    params = AnalysisParameters.from_ui(logger, config, **event.model_dump())

    if not context.loaded_models:
        context.loaded_models = initialize_analysis_models(params, config, logger, mr)

    # 1. Pre-Analysis
    pre_result = None
    pre_gen = execute_pre_analysis(event, context)
    for update in pre_gen:
        if isinstance(update, PreAnalysisResult):
            pre_result = update
        elif isinstance(update, PipelineFailure):
            yield update
            return
        else:
            yield update

    if not pre_result:
        return

    scenes = pre_result.scenes
    is_video = bool(event.video_path)

    # 2. Propagation (if video)
    prop_event = PropagationEvent(
        output_folder=event.output_folder,
        video_path=event.video_path,
        scenes=scenes,
        analysis_params=event,
    )

    if is_video:
        prop_gen = execute_propagation(prop_event, context)
        for update in prop_gen:
            if isinstance(update, PipelineFailure):
                yield update
                return
            yield update
    else:
        msg = "Mask Propagation (Skipped for Folder)"
        logger.info(msg)
        yield PropagationResult(unified_log=msg, output_dir=event.output_folder)

    # 3. Analysis
    ana_gen = execute_analysis(prop_event, context)
    for update in ana_gen:
        yield update


@handle_common_errors(AnalysisResult)
def execute_full_pipeline(
    event: ExtractionEvent,
    context: AnalysisContext,
) -> Generator[BaseModel, None, None]:
    """Orchestrates the entire flow: Extraction -> Pre-Analysis -> Propagation -> Analysis."""

    # 1. Extraction
    ext_result = None
    ext_gen = execute_extraction(event, context)
    for update in ext_gen:
        if isinstance(update, ExtractionResult):
            ext_result = update
        elif isinstance(update, PipelineFailure):
            yield update
            return
        else:
            yield update

    if not ext_result:
        return

    # 2. Build Pre-Analysis Event from Extraction Result
    pre_event = PreAnalysisEvent(
        output_folder=ext_result.output_dir,
        video_path=ext_result.video_path,
        face_ref_img_path="",  # Default to "Find Prominent Person" unless provided
        primary_seed_strategy="Automatic Detection",
        resume=False,
    )

    yield PropagationResult(unified_log="Moving to Analysis stages...")

    # 3. Chain to Analysis Orchestrator
    yield from execute_analysis_orchestrator(pre_event, context)
