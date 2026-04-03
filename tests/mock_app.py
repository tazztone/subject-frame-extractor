import os
import sys
import time
import types
from pathlib import Path
from unittest.mock import MagicMock

# --- 1. Mock Heavy Dependencies ---
# We must mock these BEFORE importing app.py


def create_mock_module(name, attributes=None):
    """Creates a proper ModuleType instance populated with mocks/attributes."""
    mock_mod = types.ModuleType(name)
    if attributes:
        for attr, val in attributes.items():
            setattr(mock_mod, attr, val)
    return mock_mod


# Initialize mock objects
mock_torch = MagicMock(name="torch")
mock_torch.cuda.is_available.return_value = False
mock_torch.__version__ = "2.0.0"
mock_torch.nn.Module = MagicMock
mock_torch.Tensor = MagicMock

mock_sam3 = MagicMock(name="sam3")
mock_sam3.model_builder = MagicMock()


# Create Stable exception classes
class OutOfMemoryError(RuntimeError):
    pass


class VideoOpenFailure(RuntimeError):
    pass


class TransparentContext:
    """Empty context manager that does nothing but allows 'with' blocks."""

    def __enter__(self, *args, **kwargs):
        return self

    def __exit__(self, *args, **kwargs):
        pass

    def __call__(self, func=None):
        if func is not None:
            return func
        return self


# Stub torch creation functions to return mocks with correct shape
import numpy as np


def _create_mock_tensor(name="tensor", shape=None, value=None, **kwargs):
    class MockTensor(MagicMock):
        def __len__(self):
            if hasattr(self, "_mock_shape") and self._mock_shape is not None:
                return self._mock_shape[0] if len(self._mock_shape) > 0 else 0
            return 0

        def __getitem__(self, idx):
            new_shape = None
            if hasattr(self, "_mock_shape") and self._mock_shape is not None:
                if len(self._mock_shape) > 0:
                    new_shape = self._mock_shape[1:]
            return _create_mock_tensor(f"{self._mock_name}[{idx}]", shape=new_shape)

        def __gt__(self, other):
            return _create_mock_tensor(f"{self._mock_name} > {other}", shape=getattr(self, "_mock_shape", None))

        def __bool__(self):
            return True

        def cpu(self):
            return self

        def numpy(self):
            s = getattr(self, "_mock_shape", None)
            if s is None:
                s = (100, 100)
            if isinstance(s, int):
                s = (s,)
            return np.zeros(s, dtype=np.float32)

        def __repr__(self):
            return f"MockTensor(name={self._mock_name}, shape={getattr(self, '_mock_shape', None)})"

        @property
        def ndim(self):
            s = getattr(self, "_mock_shape", None)
            return len(s) if s is not None else 0

    mock_t = MockTensor(name=name)
    mock_t._mock_shape = shape
    if shape is not None:
        mock_t.shape = shape
    mock_t.device = mock_torch.device("cpu")
    mock_t.dtype = mock_torch.float32
    mock_t.size.side_effect = lambda dim=None: shape if dim is None else shape[dim]
    mock_t.__mul__ = MagicMock(return_value=mock_t)
    mock_t.__add__ = MagicMock(return_value=mock_t)
    mock_t.__sub__ = MagicMock(return_value=mock_t)
    mock_t.__truediv__ = MagicMock(return_value=mock_t)
    if value is not None:
        if hasattr(value, "__getitem__") and len(value) > 0:
            try:
                # Handle nested lists/arrays
                flat_val = value
                while hasattr(flat_val, "__getitem__") and not isinstance(flat_val, (str, bytes)):
                    flat_val = flat_val[0]
                mock_t.item.return_value = flat_val
            except Exception:
                mock_t.item.return_value = 1.0
        else:
            mock_t.item.return_value = value
    else:
        mock_t.item.return_value = 1.0
    return mock_t


mock_torch.cuda.OutOfMemoryError = OutOfMemoryError
mock_torch.cuda.get_device_name = MagicMock(return_value="Mock GPU")
mock_torch.cuda.empty_cache = MagicMock()
mock_torch.float = MagicMock(name="torch.float")
mock_torch.float32 = MagicMock(name="torch.float32")
mock_torch.float16 = MagicMock(name="torch.float16")
mock_torch.bfloat16 = MagicMock(name="torch.bfloat16")
mock_torch.uint8 = MagicMock(name="torch.uint8")
mock_torch.int64 = MagicMock(name="torch.int64")
mock_torch.device = MagicMock()
mock_torch.from_numpy = MagicMock(side_effect=lambda np_arr: _create_mock_tensor("from_numpy", np_arr.shape))
mock_torch.zeros = MagicMock(side_effect=lambda shape, **kwargs: _create_mock_tensor("zeros", shape))
mock_torch.ones = MagicMock(side_effect=lambda shape, **kwargs: _create_mock_tensor("ones", shape))
mock_torch.tensor = MagicMock(
    side_effect=lambda data, **kwargs: _create_mock_tensor(
        "tensor",
        getattr(data, "shape", getattr(data, "__len__", lambda: (1,))() if hasattr(data, "__len__") else ()),
        data,
    )
)
mock_torch.no_grad = TransparentContext
mock_torch.inference_mode = TransparentContext

# Define the modules to mock and their structure
# We use ModuleType to avoid "Environment Pollution" (MagicMock in sys.modules)
modules_map = {
    "torch": create_mock_module(
        "torch",
        {
            "cuda": mock_torch.cuda,
            "nn": mock_torch.nn,
            "Tensor": mock_torch.Tensor,
            "__version__": "2.0.0",
            "device": mock_torch.device,
            "float": mock_torch.float,
            "uint8": mock_torch.uint8,
            "float32": mock_torch.float32,
            "float16": mock_torch.float16,
            "bfloat16": mock_torch.bfloat16,
            "int64": mock_torch.int64,
            "from_numpy": mock_torch.from_numpy,
            "zeros": mock_torch.zeros,
            "ones": mock_torch.ones,
            "tensor": mock_torch.tensor,
            "no_grad": mock_torch.no_grad,
            "inference_mode": mock_torch.inference_mode,
        },
    ),
    "torch.cuda": mock_torch.cuda,
    "torch.nn": mock_torch.nn,
    "torchvision": create_mock_module("torchvision", {"ops": MagicMock(), "transforms": MagicMock()}),
    "torchvision.ops": MagicMock(),
    "torchvision.transforms": MagicMock(),
    "insightface": create_mock_module(
        "insightface", {"app": create_mock_module("insightface.app", {"FaceAnalysis": MagicMock()})}
    ),
    "insightface.app": create_mock_module("insightface.app", {"FaceAnalysis": MagicMock()}),
    "sam3": create_mock_module("sam3", {"model_builder": mock_sam3.model_builder}),
    "sam3.model_builder": mock_sam3.model_builder,
    "sam3.model.sam3_video_predictor": MagicMock(),
    "mediapipe": create_mock_module("mediapipe", {"tasks": MagicMock()}),
    "mediapipe.tasks": create_mock_module("mediapipe.tasks", {"python": MagicMock()}),
    "mediapipe.tasks.python": create_mock_module("mediapipe.tasks.python", {"vision": MagicMock()}),
    "mediapipe.tasks.python.vision": MagicMock(),
    "pyiqa": MagicMock(),
    "scenedetect": create_mock_module(
        "scenedetect", {"detect": MagicMock(), "VideoOpenFailure": VideoOpenFailure, "ContentDetector": MagicMock()}
    ),
    "scenedetect.detectors": create_mock_module("scenedetect.detectors", {"ContentDetector": MagicMock()}),
    "yt_dlp": create_mock_module(
        "yt_dlp",
        {"utils": create_mock_module("yt_dlp.utils", {"DownloadError": type("DownloadError", (Exception,), {})})},
    ),
    "yt_dlp.utils": create_mock_module("yt_dlp.utils", {"DownloadError": type("DownloadError", (Exception,), {})}),
    "numba": create_mock_module("numba", {"njit": lambda f: f, "jit": lambda f: f, "cuda": MagicMock()}),
    "lpips": MagicMock(),
    "matplotlib": create_mock_module(
        "matplotlib",
        {
            "pyplot": MagicMock(),
            "ticker": MagicMock(),
            "figure": MagicMock(),
            "backends": MagicMock(),
            "get_backend": MagicMock(return_value="agg"),
            "use": MagicMock(),
            "rcParams": {},
        },
    ),
    "matplotlib.pyplot": MagicMock(),
    "matplotlib.ticker": MagicMock(),
    "matplotlib.figure": MagicMock(),
    "matplotlib.backends": MagicMock(),
    "matplotlib.backends.backend_agg": MagicMock(),
    "skimage": create_mock_module("skimage", {"metrics": MagicMock()}),
    "skimage.metrics": create_mock_module("skimage.metrics", {"structural_similarity": MagicMock()}),
    "safetensors": create_mock_module("safetensors", {"torch": MagicMock()}),
    "safetensors.torch": MagicMock(),
    "ftfy": MagicMock(),
    "regex": MagicMock(),
    "iopath": MagicMock(),
    "decord": MagicMock(),
    "onnxruntime": MagicMock(),
}

# Ensure sam3 submodules are also mocked for patches
modules_map["sam3.model"] = create_mock_module("sam3.model")
modules_map["sam3.utils"] = create_mock_module("sam3.utils")
modules_map["sam3.model.sam3_video_predictor"] = create_mock_module("sam3.model.sam3_video_predictor")
modules_map["sam3.model.sam3_video_predictor"].SAM3VideoPredictor = MagicMock()
modules_map["sam3.model.sam3_video_inference"] = create_mock_module("sam3.model.sam3_video_inference")
modules_map["sam3.model.sam3_video_inference"].SAM3VideoInference = MagicMock()


# Patch sys.modules with proper ModuleType objects where possible
for mod_name, mod_obj in modules_map.items():
    if not isinstance(mod_obj, types.ModuleType):
        # Fallback to MagicMock wrapped in ModuleType if it represents a package
        mock_val = mod_obj
        mod_obj = types.ModuleType(mod_name)
        # Populate the module with common attributes from the mock if it's a MagicMock
        if isinstance(mock_val, MagicMock):
            for attr in dir(mock_val):
                if not attr.startswith("__"):
                    setattr(mod_obj, attr, getattr(mock_val, attr))
    sys.modules[mod_name] = mod_obj

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
            time.sleep(0.01)
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

    return {
        "done": True,
        "output_dir": output_dir,
        "video_path": "mock_video.mp4",
        "extracted_frames_dir_state": output_dir,
        "extracted_video_path_state": "mock_video.mp4",
        "unified_status_state": "Extraction Complete.",
    }


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
        s_dict["preview_path"] = preview_path
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
        "video_path": event.video_path,
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
    import json
    import sqlite3

    print("[Mock] Running Analysis...")
    output_dir = event.output_folder
    metadata_path = os.path.join(output_dir, "metadata.db")
    os.makedirs(output_dir, exist_ok=True)

    # Create a minimal real schema-compliant DB
    conn = sqlite3.connect(metadata_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS metadata (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT UNIQUE,
            metrics TEXT,
            face_sim REAL,
            face_conf REAL,
            shot_id INTEGER,
            seed_type TEXT,
            seed_face_sim REAL,
            mask_area_pct REAL,
            mask_empty INTEGER,
            error TEXT,
            phash TEXT,
            dedup_thresh INTEGER,
            error_severity TEXT
        )
    """)
    # Insert 10 dummy rows
    for i in range(1, 11):
        metrics = json.dumps({"quality_score": 80.0, "sharpness": 75.0, "eyes_open": 1.0})
        conn.execute(
            "INSERT OR IGNORE INTO metadata (id, filename, metrics, face_sim, shot_id) VALUES (?, ?, ?, ?, ?)",
            (i, f"frame_{i:06d}.webp", metrics, 0.95, 1),
        )
    conn.commit()
    conn.close()

    msg = "Analysis complete."
    if logger:
        logger.info(msg)
    yield {
        "unified_log": msg,
        "output_dir": output_dir,
        "metadata_path": metadata_path,
        "done": True,
    }


def mock_analysis_orchestrator(
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
    """Mocks the analysis orchestrator workflow."""
    print("[Mock] Running Analysis Orchestrator...")
    # 1. Pre-Analysis
    pre_gen = mock_pre_analysis_execution(
        event, progress_queue, cancel_event, logger, config, thumbnail_manager, cuda_available, progress, model_registry
    )
    for res in pre_gen:
        yield res

    # 2. Propagation
    yield from mock_propagation_execution(
        event, progress_queue, cancel_event, logger, config, thumbnail_manager, cuda_available, progress, model_registry
    )

    # 3. Analysis
    yield from mock_analysis_execution(
        event, progress_queue, cancel_event, logger, config, thumbnail_manager, cuda_available, progress, model_registry
    )


def mock_full_pipeline_execution(
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
    """Mocks the full pipeline orchestrator workflow."""
    print("[Mock] Running Full Pipeline Orchestrator...")
    # 1. Extraction
    # We need a dummy object that has .run because mock_extraction_run is patched on the class
    # Actually, we can just call it or simulate it.
    yield {"unified_log": "Starting Full Pipeline (Mock)...", "done": False}

    # Simulate Extraction
    output_dir = os.path.join(config.downloads_dir, "mock_video")
    os.makedirs(output_dir, exist_ok=True)
    yield {
        "unified_log": "Extraction Complete (Mock)",
        "extracted_video_path_state": "mock_video.mp4",
        "extracted_frames_dir_state": output_dir,
        "done": True,
    }

    yield {"unified_log": "Moving to Analysis stages...", "done": False}

    # Chain to Analysis Orchestrator
    yield from mock_analysis_orchestrator(
        event, progress_queue, cancel_event, logger, config, thumbnail_manager, cuda_available, progress, model_registry
    )


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
import ui.handlers.pipeline_handlers

core.pipelines.ExtractionPipeline._run_impl = mock_extraction_run
# We mock the `execute_*` functions directly as they are what the UI calls via `_run_pipeline`
ui.app_ui.AppUI.preload_models = MagicMock(side_effect=lambda *args: None)

# Patch in all locations where these are imported/used
for target in [core.pipelines, ui.app_ui, ui.handlers.pipeline_handlers]:
    if hasattr(target, "execute_pre_analysis"):
        target.execute_pre_analysis = mock_pre_analysis_execution
    if hasattr(target, "execute_propagation"):
        target.execute_propagation = mock_propagation_execution
    if hasattr(target, "execute_analysis"):
        target.execute_analysis = mock_analysis_execution
    if hasattr(target, "execute_analysis_orchestrator"):
        target.execute_analysis_orchestrator = mock_analysis_orchestrator
    if hasattr(target, "execute_full_pipeline"):
        target.execute_full_pipeline = mock_full_pipeline_execution

# Patch fingerprinting to avoid FileNotFoundError in mock tests
import core.fingerprint

core.fingerprint.create_fingerprint = MagicMock(return_value={})
core.fingerprint.save_fingerprint = MagicMock()

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
