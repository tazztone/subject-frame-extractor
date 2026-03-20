"""
System health checks and diagnostic report generation.
"""

import shutil
import subprocess
import sys
from collections import deque
from pathlib import Path
from typing import Any, Generator, List

import torch

from core.events import ExportEvent, ExtractionEvent, PreAnalysisEvent, PropagationEvent
from core.export import export_kept_frames
from core.pipelines import (
    execute_analysis,
    execute_extraction,
    execute_pre_analysis,
    execute_propagation,
)


def check_environment() -> List[str]:
    """Checks the basic Python and PyTorch/CUDA environment."""
    report = ["\n[SECTION 1: System & Environment]"]
    try:
        report.append(
            f"  - Python Version: OK ({sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro})"
        )
    except Exception as e:
        report.append(f"  - Python Version: FAILED ({e})")

    try:
        report.append(f"  - PyTorch Version: OK ({torch.__version__})")
        if torch.cuda.is_available():
            report.append(
                f"  - CUDA: OK (Version: {torch.version.cuda}, GPU: {torch.cuda.get_device_name(0)})"
            )
        else:
            report.append("  - CUDA: NOT AVAILABLE (Running in CPU mode)")
    except Exception as e:
        report.append(f"  - PyTorch/CUDA Check: FAILED ({e})")
    return report


def check_dependencies() -> List[str]:
    """Checks for the presence of core dependencies."""
    report = ["\n[SECTION 2: Core Dependencies]"]
    for dep in ["cv2", "gradio", "imagehash", "mediapipe", "sam3"]:
        try:
            __import__(dep.split(".")[0])
            report.append(f"  - {dep}: OK")
        except ImportError:
            report.append(f"  - {dep}: FAILED (Not Installed)")
    return report


def check_paths_and_assets(config: Any) -> List[str]:
    """Checks for required paths and assets."""
    report = ["\n[SECTION 3: Paths & Assets]"]

    # Check ExifTool
    exiftool_path = shutil.which("exiftool")
    if exiftool_path:
        try:
            ver = subprocess.run(
                [exiftool_path, "-ver"], capture_output=True, text=True
            ).stdout.strip()
            report.append(f"  - ExifTool: OK (Version: {ver})")
        except Exception:
            report.append(f"  - ExifTool: FOUND but check failed (Path: {exiftool_path})")
    else:
        report.append("  - ExifTool: FAILED (Not Found - Required for Photo Mode)")

    paths = {
        "Models Directory": Path(config.models_dir),
        "Dry Run Assets": Path("dry-run-assets"),
        "Sample Video": Path("dry-run-assets/sample.mp4"),
        "Sample Image": Path("dry-run-assets/sample.jpg"),
    }

    for name, path in paths.items():
        report.append(f"  - {name}: {'OK' if path.exists() else 'FAILED'} (Path: {path})")
    return report


def simulate_pipeline(
    config: Any,
    logger: Any,
    progress_queue: Any,
    cancel_event: Any,
    thumbnail_manager: Any,
    cuda_available: bool,
) -> List[str]:
    """Simulates the E2E pipeline with sample assets."""
    report = [
        "\n[SECTION 4: Model Loading Simulation]",
        "  - Skipping Model Loading Simulation (Models loaded on demand)",
        "\n[SECTION 5: E2E Pipeline Simulation]",
    ]

    temp_output_dir = Path(config.downloads_dir) / "dry_run_output"
    shutil.rmtree(temp_output_dir, ignore_errors=True)
    temp_output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Stage 1: Extraction
        report.append("  - Stage 1: Frame Extraction...")
        ext_event = ExtractionEvent(
            source_path="dry-run-assets/sample.mp4",
            method="interval",
            interval="1.0",
            max_resolution="720",
            thumbnails_only=True,
            thumb_megapixels=0.2,
            scene_detect=True,
        )
        ext_result = deque(
            execute_extraction(ext_event, progress_queue, cancel_event, logger, config),
            maxlen=1,
        )[0]
        if not ext_result.get("done"):
            raise RuntimeError("Extraction failed")
        report[-1] += " OK"

        # Stage 2: Pre-analysis
        report.append("  - Stage 2: Pre-analysis...")
        pre_ana_event = PreAnalysisEvent(
            output_folder=ext_result["extracted_frames_dir_state"],
            video_path=ext_result["extracted_video_path_state"],
            scene_detect=True,
            pre_analysis_enabled=True,
            pre_sample_nth=1,
            primary_seed_strategy="🧑‍🤝‍🧑 Find Prominent Person",
            face_model_name="buffalo_l",
            tracker_model_name="sam3",
            min_mask_area_pct=1.0,
            sharpness_base_scale=2500.0,
            edge_strength_base_scale=100.0,
        )
        pre_ana_result = deque(
            execute_pre_analysis(
                pre_ana_event,
                progress_queue,
                cancel_event,
                logger,
                config,
                thumbnail_manager,
                cuda_available,
            ),
            maxlen=1,
        )[0]
        if not pre_ana_result.get("done"):
            raise RuntimeError(f"Pre-analysis failed: {pre_ana_result}")
        report[-1] += " OK"

        scenes = pre_ana_result["scenes"]

        # Stage 3: Mask Propagation
        report.append("  - Stage 3: Mask Propagation...")
        prop_event = PropagationEvent(
            output_folder=pre_ana_result["output_dir"],
            video_path=ext_result["extracted_video_path_state"],
            scenes=scenes,
            analysis_params=pre_ana_event,
        )
        prop_result = deque(
            execute_propagation(
                prop_event,
                progress_queue,
                cancel_event,
                logger,
                config,
                thumbnail_manager,
                cuda_available,
            ),
            maxlen=1,
        )[0]
        if not prop_result.get("done"):
            raise RuntimeError("Propagation failed")
        report[-1] += " OK"

        # Stage 4: Frame Analysis
        report.append("  - Stage 4: Frame Analysis...")
        ana_result = deque(
            execute_analysis(
                prop_event,
                progress_queue,
                cancel_event,
                logger,
                config,
                thumbnail_manager,
                cuda_available,
            ),
            maxlen=1,
        )[0]
        if not ana_result.get("done"):
            raise RuntimeError("Analysis failed")
        report[-1] += " OK"

        output_dir = ana_result["output_dir"]

        # Stage 5: Filtering
        from core.filtering import apply_all_filters_vectorized, load_and_prep_filter_data

        all_frames, _ = load_and_prep_filter_data(
            output_dir, lambda: ["quality_score", "face_sim"], config
        )
        report.append("  - Stage 5: Filtering...")
        kept, _, _, _ = apply_all_filters_vectorized(
            all_frames,
            {"require_face_match": False, "dedup_thresh": -1},
            config,
            output_dir=output_dir,
        )
        report[-1] += f" OK (kept {len(kept)} frames)"

        # Stage 6: Export
        report.append("  - Stage 6: Export...")
        export_event = ExportEvent(
            all_frames_data=all_frames,
            output_dir=output_dir,
            video_path=ext_result["extracted_video_path_state"],
            enable_crop=False,
            crop_ars="",
            crop_padding=0,
            filter_args={"require_face_match": False, "dedup_thresh": -1},
        )
        export_msg = export_kept_frames(
            export_event, config, logger, thumbnail_manager, cancel_event
        )
        if "Error" in export_msg:
            raise RuntimeError(f"Export failed: {export_msg}")
        report[-1] += " OK"

    except Exception as e:
        error_message = f"FAILED ({e})"
        if report and "..." in report[-1]:
            report[-1] += error_message
        else:
            report.append(f"  - Pipeline Simulation: {error_message}")
        logger.error("Dry run pipeline failed", exc_info=True)

    return report


def generate_full_diagnostic_report(
    config: Any,
    logger: Any,
    progress_queue: Any,
    cancel_event: Any,
    thumbnail_manager: Any,
    cuda_available: bool,
) -> Generator[str, None, None]:
    """Generates a full diagnostic report as a generator."""
    report = ["\n\n--- System Diagnostics Report ---"]
    report.extend(check_environment())
    report.extend(check_dependencies())
    report.extend(check_paths_and_assets(config))
    report.extend(
        simulate_pipeline(
            config,
            logger,
            progress_queue,
            cancel_event,
            thumbnail_manager,
            cuda_available,
        )
    )

    final_report = "\n".join(report)
    yield final_report
