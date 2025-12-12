import sys
import os
import threading
import time
from unittest.mock import MagicMock, patch

# --- 1. Mock Heavy Dependencies ---
# We must mock these BEFORE importing app.py

mock_torch = MagicMock(name='torch')
mock_torch.cuda.is_available.return_value = False
mock_torch.__version__ = "2.0.0"
# Mock torch classes used in type hints or inheritance
mock_torch.nn.Module = MagicMock
mock_torch.Tensor = MagicMock

mock_sam3 = MagicMock(name='sam3')
mock_sam3.model_builder = MagicMock()

modules_to_mock = {
    'torch': mock_torch,
    'torchvision': MagicMock(),
    'torchvision.ops': MagicMock(),
    'torchvision.transforms': MagicMock(),
    'insightface': MagicMock(),
    'insightface.app': MagicMock(),
    'sam3': mock_sam3,
    'sam3.model_builder': mock_sam3.model_builder,
    'sam3.model.sam3_video_predictor': MagicMock(),
    'mediapipe': MagicMock(),
    'mediapipe.tasks': MagicMock(),
    'mediapipe.tasks.python': MagicMock(),
    'mediapipe.tasks.python.vision': MagicMock(),
    'pyiqa': MagicMock(),
    'scenedetect': MagicMock(),
    'yt_dlp': MagicMock(),
    'ultralytics': MagicMock(),
    'groundingdino': MagicMock(),
    'numba': MagicMock(),
    'lpips': MagicMock(),
}

# Patch sys.modules
patch.dict(sys.modules, modules_to_mock).start()

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- 2. Import App and Core Modules ---
import app
from app import Config, AppLogger, ThumbnailManager
import core.pipelines
import core.utils
import core.managers
from core.models import Scene

# --- 3. Patch Pipeline Logic for E2E Speed ---

def mock_extraction_run(self, tracker=None):
    """Mocks the extraction process."""
    print("[Mock] Running Extraction...")
    # Simulate processing time
    if tracker:
        tracker.start(10, desc="Mock Extraction")
        for i in range(10):
            time.sleep(0.01)
            tracker.step(1)
        tracker.done_stage("Mock Extraction Complete")

    # Create fake output
    output_dir = os.path.join(self.config.downloads_dir, "mock_video")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "thumbs"), exist_ok=True)

    # Create fake frame map
    import json
    frame_map = {i: f"frame_{i:06d}.webp" for i in range(1, 11)}
    with open(os.path.join(output_dir, "frame_map.json"), 'w') as f:
        json.dump(list(frame_map.keys()), f)

    return {"done": True, "output_dir": output_dir, "video_path": "mock_video.mp4"}

def mock_pre_analysis_execution(event, progress_queue, cancel_event, logger, config, thumbnail_manager, cuda_available, progress=None, model_registry=None):
    """Mocks execute_pre_analysis generator."""
    print("[Mock] Running Pre-Analysis...")

    scenes = [
        Scene(shot_id=1, start_frame=0, end_frame=50, status="included", seed_result={'bbox': [10, 10, 100, 100], 'details': {'type': 'mock'}}).model_dump(),
        Scene(shot_id=2, start_frame=51, end_frame=100, status="included").model_dump()
    ]

    output_dir = os.path.join(config.downloads_dir, "mock_video")

    # Yield progress update
    yield {
        "unified_log": "Pre-analysis complete (MOCKED).",
        "scenes": scenes,
        "output_dir": output_dir,
        "done": True,
        # Omit UI updates to let app.py use defaults (gr.update)
    }

def mock_propagation_execution(event, progress_queue, cancel_event, logger, config, thumbnail_manager, cuda_available, progress=None, model_registry=None):
    print("[Mock] Running Propagation...")
    yield {
        "unified_log": "Propagation complete (MOCKED).",
        "output_dir": event.output_folder,
        "done": True,
        "scenes": event.scenes # Pass back scenes
    }

def mock_analysis_execution(event, progress_queue, cancel_event, logger, config, thumbnail_manager, cuda_available, progress=None, model_registry=None):
    print("[Mock] Running Analysis...")
    output_dir = event.output_folder
    metadata_path = os.path.join(output_dir, "metadata.db")
    yield {
        "unified_log": "Analysis complete (MOCKED).",
        "output_dir": output_dir,
        "metadata_path": metadata_path,
        "done": True
    }

# Apply patches
core.pipelines.ExtractionPipeline._run_impl = mock_extraction_run
# We mock the `execute_*` functions directly as they are what the UI calls via `_run_pipeline`
core.pipelines.execute_pre_analysis = mock_pre_analysis_execution
core.pipelines.execute_propagation = mock_propagation_execution
core.pipelines.execute_analysis = mock_analysis_execution
# Patch download_model to avoid network calls
core.utils.download_model = MagicMock()
core.managers.download_model = MagicMock()


# --- 4. Launch App ---
if __name__ == "__main__":
    print("Starting Mock App for E2E Testing...")
    # Use a specific port for testing
    os.environ['GRADIO_SERVER_PORT'] = '7860'
    app.main()
