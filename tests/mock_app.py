import os
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

# --- 1. Mock Heavy Dependencies ---
# We must mock these BEFORE importing app.py

mock_torch = MagicMock(name="torch")
mock_torch.cuda.is_available.return_value = False

sys.modules["torch"] = mock_torch
mock_torch.__version__ = "2.0.0"
# Mock torch classes used in type hints or inheritance
mock_torch.nn.Module = MagicMock
mock_torch.Tensor = MagicMock

mock_sam3 = MagicMock(name="sam3")
mock_sam3.model_builder = MagicMock()

modules_to_mock = {
    "torch": mock_torch,
    "torchvision": MagicMock(),
    "torchvision.ops": MagicMock(),
    "torchvision.transforms": MagicMock(),
    "insightface": MagicMock(),
    "insightface.app": MagicMock(),
    "sam3": mock_sam3,
    "sam3.model_builder": mock_sam3.model_builder,
    "sam3.model.sam3_video_predictor": MagicMock(),
    "mediapipe": MagicMock(),
    "mediapipe.tasks": MagicMock(),
    "mediapipe.tasks.python": MagicMock(),
    "mediapipe.tasks.python.vision": MagicMock(),
    "pyiqa": MagicMock(),
    "scenedetect": MagicMock(),
    "yt_dlp": MagicMock(),
    "numba": MagicMock(),
    "lpips": MagicMock(),
    "matplotlib": MagicMock(),
    "matplotlib.pyplot": MagicMock(),
    "matplotlib.ticker": MagicMock(),
    "matplotlib.figure": MagicMock(),
    "matplotlib.backends": MagicMock(),
    "matplotlib.backends.backend_agg": MagicMock(),
    "skimage": MagicMock(),
    "skimage.metrics": MagicMock(),
    "safetensors": MagicMock(),
    "safetensors.torch": MagicMock(),
    "ftfy": MagicMock(),
    "regex": MagicMock(),
    "iopath": MagicMock(),
    "decord": MagicMock(),
    "onnxruntime": MagicMock(),
}

# Patch sys.modules
patch.dict(sys.modules, modules_to_mock).start()

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# --- 2. Import App and Core Modules ---
import app
import core
import core.managers
import core.photo_utils
import core.pipelines
import core.utils
import core.xmp_writer
from core.models import Scene

# --- 3. Patch Pipeline Logic for E2E Speed ---


def mock_extraction_run(self, tracker=None):
    """Mocks the extraction process."""
    print("[Mock] Running Extraction...")
    # Simulate processing time
    if tracker:
        tracker.start(10, desc="Mock Extraction")
        for i in range(10):
            time.sleep(0.1)
            tracker.step(1)
        tracker.set_stage("Extraction Complete")
        tracker.done_stage()

    # Create fake output
    output_dir = os.path.join(self.config.downloads_dir, "mock_video")
    os.makedirs(output_dir, exist_ok=True)
    thumb_dir = os.path.join(output_dir, "thumbs")
    os.makedirs(thumb_dir, exist_ok=True)

    # Create actual dummy thumbnail files so ThumbnailManager can load them
    import numpy as np
    from PIL import Image

    for i in range(1, 11):
        thumb_path = os.path.join(thumb_dir, f"frame_{i:06d}.webp")
        if not os.path.exists(thumb_path):
            # Create a small random image
            img_data = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            img = Image.fromarray(img_data)
            img.save(thumb_path, "WEBP")

    # Create fake frame map
    import json

    frame_map = {i: f"frame_{i:06d}.webp" for i in range(1, 11)}
    with open(os.path.join(output_dir, "frame_map.json"), "w") as f:
        json.dump(list(frame_map.keys()), f)

    msg = "Extraction complete."
    if hasattr(self, "logger") and self.logger:
        self.logger.info(msg)

    return {"done": True, "output_dir": output_dir, "video_path": "mock_video.mp4"}


def mock_pre_analysis_execution(
    event,
    progress_queue,
    cancel_event,
    logger,
    config,
    thumbnail_manager,
    cuda_available,
    progress=None,
    model_registry=None,
):
    """Mocks execute_pre_analysis generator."""
    print("[Mock] Running Pre-Analysis...")

    scenes = [
        Scene(
            shot_id=1,
            start_frame=0,
            end_frame=25,
            status="included",
            filename="frame_000001.webp",
            seed_result={"bbox": [10, 10, 100, 100], "details": {"type": "mock"}},
            seed_metrics={"score": 10.0, "best_face_sim": 0.9},
        ).model_dump(),
        Scene(
            shot_id=2,
            start_frame=26,
            end_frame=50,
            status="included",
            filename="frame_000003.webp",
            seed_metrics={"score": 5.0, "best_face_sim": 0.5},
        ).model_dump(),
        Scene(
            shot_id=3,
            start_frame=51,
            end_frame=75,
            status="included",
            filename="frame_000005.webp",
            seed_metrics={"score": 8.0, "best_face_sim": 0.7},
        ).model_dump(),
        Scene(
            shot_id=4,
            start_frame=76,
            end_frame=100,
            status="included",
            filename="frame_000007.webp",
            seed_metrics={"score": 12.0, "best_face_sim": 0.95},
        ).model_dump(),
    ]

    output_dir = os.path.join(config.downloads_dir, "mock_video")
    previews_dir = os.path.join(output_dir, "previews")
    os.makedirs(previews_dir, exist_ok=True)

    # Create dummy preview files
    import numpy as np
    from PIL import Image

    for s_dict in scenes:
        shot_id = s_dict["shot_id"]
        preview_path = os.path.join(previews_dir, f"scene_{shot_id:05d}.jpg")
        if not os.path.exists(preview_path):
            img_data = np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8)
            img = Image.fromarray(img_data)
            img.save(preview_path, "JPEG")

    # Yield progress update
    msg = "Pre-Analysis complete."
    if logger:
        logger.info(msg)
    yield {
        "unified_log": msg,
        "scenes": scenes,
        "output_dir": output_dir,
        "done": True,
        # Omit UI updates to let app.py use defaults (gr.update)
    }


def mock_propagation_execution(
    event,
    progress_queue,
    cancel_event,
    logger,
    config,
    thumbnail_manager,
    cuda_available,
    progress=None,
    model_registry=None,
):
    print("[Mock] Running Propagation...")
    msg = "Propagation complete."
    if logger:
        logger.info(msg)
    yield {
        "unified_log": msg,
        "output_dir": event.output_folder,
        "done": True,
        "scenes": event.scenes,  # Pass back scenes
        "metadata_path": os.path.join(event.output_folder, "metadata.db"),  # Add for compatibility
    }


def mock_analysis_execution(
    event,
    progress_queue,
    cancel_event,
    logger,
    config,
    thumbnail_manager,
    cuda_available,
    progress=None,
    model_registry=None,
):
    print("[Mock] Running Analysis...")
    output_dir = event.output_folder
    metadata_path = os.path.join(output_dir, "metadata.db")
    msg = "Analysis complete."
    if logger:
        logger.info(msg)
    yield {
        "unified_log": msg,
        "output_dir": output_dir,
        "metadata_path": metadata_path,
        "done": True,
    }


def mock_ingest_folder(folder_path, output_dir):
    print(f"[Mock] Ingesting folder: {folder_path}")
    dummy_path = str(Path(__file__).parent / "ui" / "dummy.jpg")
    return [
        {
            "id": f"photo_{i}",
            "source": f"photo_{i}.CR2",
            "preview": dummy_path,
            "type": "raw",
            "status": "unreviewed",
            "scores": {},
        }
        for i in range(1, 6)
    ]


def mock_apply_scores_to_photos(photos, weights):
    print(f"[Mock] Scoring photos with weights: {weights}")
    for p in photos:
        p["scores"] = {"quality_score": 85.0, "sharpness": 90.0, "entropy": 80.0}
    return photos


def mock_export_xmps_for_photos(photos, thresholds=None):
    print("[Mock] Exporting XMPs")
    return len(photos)


def mock_export_kept_frames(*args, **kwargs):
    print("[Mock] Running Export...")
    logger = None
    if len(args) > 2:
        logger = args[2]
    elif "logger" in kwargs:
        logger = kwargs["logger"]

    msg = "✅ Export Complete. Exported 10 items (MOCKED)."
    if logger:
        logger.info(msg)
    return msg


# Apply patches
import core.export
import ui.app_ui

core.pipelines.ExtractionPipeline._run_impl = mock_extraction_run
# We mock the `execute_*` functions directly as they are what the UI calls via `_run_pipeline`
ui.app_ui.AppUI.preload_models = MagicMock(side_effect=lambda *args: None)
core.pipelines.execute_pre_analysis = ui.app_ui.execute_pre_analysis = mock_pre_analysis_execution
core.pipelines.execute_propagation = ui.app_ui.execute_propagation = mock_propagation_execution
core.pipelines.execute_analysis = ui.app_ui.execute_analysis = mock_analysis_execution
ui.app_ui.export_kept_frames = mock_export_kept_frames
core.export.export_kept_frames = mock_export_kept_frames
core.photo_utils.ingest_folder = mock_ingest_folder
core.xmp_writer.export_xmps_for_photos = mock_export_xmps_for_photos
# Patch download_model to avoid network calls
core.utils.download_model = MagicMock()
core.managers.download_model = MagicMock()


# --- 4. Launch App ---

if __name__ == "__main__":
    print("Starting Mock App for E2E Testing...")
    import os

    os.environ["APP_DEBUG_MODE"] = "true"
    app.main()
