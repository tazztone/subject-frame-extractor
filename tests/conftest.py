"""
Shared pytest fixtures and configuration.

This module provides common fixtures for operator tests and regression tests,
reducing code duplication and ensuring consistency.
"""

import sys
import threading
import types
from queue import Queue
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

# --- 1. Global Mocking Infrastructure (Run immediately on import) ---


# Module creation helper
def _create_mock_module(name, attributes=None):
    """Creates a proper ModuleType instance populated with mocks/attributes."""
    mock_mod = types.ModuleType(name)
    if attributes:
        for attr, val in attributes.items():
            setattr(mock_mod, attr, val)
    return mock_mod


# Stable exception classes
class OutOfMemoryError(RuntimeError):
    """Mock CUDA OutOfMemoryError."""

    pass


class VideoOpenFailure(RuntimeError):
    """Mock PySceneDetect VideoOpenFailure."""

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


# Build the base torch mock
_mock_torch_obj = MagicMock(name="torch")


# Promote torch.cuda to a real ModuleType instance for stable patching
_cuda_mod = _create_mock_module(
    "torch.cuda",
    {
        "is_available": MagicMock(return_value=False),
        "device_count": MagicMock(return_value=0),
        "get_device_name": MagicMock(return_value="Mock GPU"),
        "empty_cache": MagicMock(),
        "memory_summary": MagicMock(return_value="Mock Memory Summary"),
        "memory_allocated": MagicMock(return_value=0),
        "OutOfMemoryError": OutOfMemoryError,
    },
)

_mock_torch_obj.cuda = _cuda_mod
_mock_torch_obj.__version__ = "2.0.0"
_mock_torch_obj.nn.Module = MagicMock
_mock_torch_obj.Tensor = MagicMock
_mock_torch_obj.device = MagicMock
_mock_torch_obj.float = MagicMock(name="torch.float")
_mock_torch_obj.float32 = MagicMock(name="torch.float32")
_mock_torch_obj.float16 = MagicMock(name="torch.float16")
_mock_torch_obj.bfloat16 = MagicMock(name="torch.bfloat16")
_mock_torch_obj.uint8 = MagicMock(name="torch.uint8")
_mock_torch_obj.int64 = MagicMock(name="torch.int64")
_mock_torch_obj.version = MagicMock()
_mock_torch_obj.version.cuda = "12.1"


# Stub torch creation functions to return mocks with correct shape
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
    mock_t.device = _mock_torch_obj.device("cpu")
    mock_t.dtype = _mock_torch_obj.float32
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


_mock_torch_obj.from_numpy = MagicMock(side_effect=lambda np_arr: _create_mock_tensor("from_numpy", np_arr.shape))
_mock_torch_obj.zeros = MagicMock(side_effect=lambda shape, **kwargs: _create_mock_tensor("zeros", shape))
_mock_torch_obj.ones = MagicMock(side_effect=lambda shape, **kwargs: _create_mock_tensor("ones", shape))
_mock_torch_obj.rand = MagicMock(side_effect=lambda *shape, **kwargs: _create_mock_tensor("rand", shape))
_mock_torch_obj.randn = MagicMock(side_effect=lambda *shape, **kwargs: _create_mock_tensor("randn", shape))
_mock_torch_obj.empty = MagicMock(side_effect=lambda *shape, **kwargs: _create_mock_tensor("empty", shape))
_mock_torch_obj.tensor = MagicMock(
    side_effect=lambda data, **kwargs: _create_mock_tensor(
        "tensor",
        getattr(data, "shape", getattr(data, "__len__", lambda: (1,))() if hasattr(data, "__len__") else ()),
        data,
    )
)
_mock_torch_obj.no_grad = TransparentContext
_mock_torch_obj.inference_mode = TransparentContext
_mock_torch_obj.SymFloat = MagicMock
_mock_torch_obj.SymInt = MagicMock

# Patch sys.modules globally and immediately
modules_to_mock = {
    "torch": _create_mock_module(
        "torch",
        {
            "cuda": _cuda_mod,
            "nn": _mock_torch_obj.nn,
            "version": _mock_torch_obj.version,
            "Tensor": _mock_torch_obj.Tensor,
            "device": _mock_torch_obj.device,
            "from_numpy": _mock_torch_obj.from_numpy,
            "zeros": _mock_torch_obj.zeros,
            "ones": _mock_torch_obj.ones,
            "tensor": _mock_torch_obj.tensor,
            "stack": MagicMock(side_effect=lambda tensors, **kwargs: MagicMock(name="stacked")),
            "cat": MagicMock(side_effect=lambda tensors, **kwargs: MagicMock(name="catted")),
            "rand": MagicMock(side_effect=lambda *args, **kwargs: MagicMock(name="rand")),
            "randn": MagicMock(side_effect=lambda *args, **kwargs: MagicMock(name="randn")),
            "float": _mock_torch_obj.float,
            "float32": _mock_torch_obj.float32,
            "float16": _mock_torch_obj.float16,
            "bfloat16": _mock_torch_obj.bfloat16,
            "uint8": _mock_torch_obj.uint8,
            "int64": _mock_torch_obj.int64,
            "no_grad": _mock_torch_obj.no_grad,
            "inference_mode": _mock_torch_obj.no_grad,
            "set_float32_matmul_precision": MagicMock(),
            "SymFloat": _mock_torch_obj.SymFloat,
            "SymInt": _mock_torch_obj.SymInt,
            "__version__": "2.0.0",
            "manual_seed": MagicMock(),
        },
    ),
    "torch.cuda": _cuda_mod,
    "torch.nn": _mock_torch_obj.nn,
    "torch.version": _mock_torch_obj.version,
    "torchvision": _create_mock_module("torchvision", {"ops": MagicMock(), "transforms": MagicMock()}),
    "torchvision.ops": MagicMock(),
    "insightface": _create_mock_module("insightface", {"app": MagicMock()}),
    "insightface.app": MagicMock(),
    "sam3": _create_mock_module("sam3", {"model_builder": MagicMock()}),
    "sam3.model_builder": MagicMock(),
    "sam3.model": _create_mock_module("sam3.model", {"sam3_video_predictor": MagicMock()}),
    "sam3.model.sam3_video_predictor": MagicMock(),
    "sam3.model.decoder": MagicMock(),
    "sam3.model.sam3_image_processor": MagicMock(),
    "sam3.model.sam3_video_inference": MagicMock(),
    "sam3.model.edt": MagicMock(),
    "sam3.perflib": _create_mock_module("sam3.perflib", {"connected_components": MagicMock()}),
    "sam3.perflib.connected_components": MagicMock(),
    "sam2": _create_mock_module("sam2", {"build_sam2": MagicMock()}),
    "sam2.build_sam": MagicMock(),
    "sam2.build_sam_video_predictor": MagicMock(),
    "mediapipe": _create_mock_module("mediapipe", {"tasks": MagicMock()}),
    "mediapipe.tasks": _create_mock_module("mediapipe.tasks", {"python": MagicMock()}),
    "mediapipe.tasks.python": _create_mock_module("mediapipe.tasks.python", {"vision": MagicMock()}),
    "pyiqa": MagicMock(),
    "lpips": _create_mock_module("lpips", {"LPIPS": MagicMock()}),
    "skimage.metrics": MagicMock(),
    "scenedetect": _create_mock_module(
        "scenedetect",
        {
            "detect": MagicMock(),
            "VideoOpenFailure": VideoOpenFailure,
            "ContentDetector": MagicMock(),
            "__all__": ["VideoOpenFailure", "ContentDetector"],
        },
    ),
    "scenedetect.detectors": _create_mock_module("scenedetect.detectors", {"ContentDetector": MagicMock()}),
    "scenedetect.video_stream": _create_mock_module("scenedetect.video_stream", {"VideoOpenFailure": VideoOpenFailure}),
    "torchvision.transforms": MagicMock(),
}

# Patch sys.modules globally ONLY if we are in a mock-safe environment (unit tests)
# Integration and UI tests MUST use real dependencies.
import os

# Check if we should skip global mocking (e.g. for integration tests)
_skip_mocks = os.environ.get("PYTEST_INTEGRATION_MODE") == "true" or any(
    arg for arg in sys.argv if (("tests/integration" in arg and "smoke" not in arg) or "tests/ui" in arg)
)

if not _skip_mocks:
    for mod_name, mod_obj in modules_to_mock.items():
        sys.modules[mod_name] = mod_obj
else:
    # Optional: log that we are in "Real Mode"
    print("\n[PyTest] Integration/UI Mode detected: Global mocks DISABLED.")


# NOW import application modules
from core.operators import OperatorRegistry


@pytest.fixture(autouse=True)
def clean_registry():
    """Ensure OperatorRegistry is clean before each test."""
    OperatorRegistry.clear()
    yield
    OperatorRegistry.clear()


@pytest.fixture(scope="session")
def requires_cuda():
    """Skip test if CUDA is not available."""
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")


@pytest.fixture(scope="session")
def check_assets():
    """Ensure E2E sample assets exist."""
    from pathlib import Path

    assets_dir = Path("tests/assets")
    required = ["sample.mp4", "sample.jpg"]
    missing = [f for f in required if not (assets_dir / f).exists()]
    if missing:
        pytest.skip(f"Missing required test assets: {', '.join(missing)}")


@pytest.fixture
def mock_logger():
    """Mock Application Logger."""
    return MagicMock()


@pytest.fixture
def sample_image():
    """100x100 RGB image with random noise (seed 42 for consistency)."""
    np.random.seed(42)
    return np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)


@pytest.fixture
def sample_mask():
    """100x100 grayscale mask (center region active)."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[25:75, 25:75] = 255
    return mask


@pytest.fixture
def sharp_image():
    """High-frequency checkerboard pattern (sharp)."""
    pattern = np.indices((100, 100)).sum(axis=0) % 2
    img = (pattern * 255).astype(np.uint8)
    return np.stack([img, img, img], axis=-1)


@pytest.fixture
def blurry_image():
    """Gaussian blurred uniform gray (blurry)."""
    gray = np.full((100, 100, 3), 128, dtype=np.uint8)
    return cv2.GaussianBlur(gray, (21, 21), 0)


@pytest.fixture
def mock_config():
    """Mock Config object with common parameters."""
    config = MagicMock()
    # Quality params
    config.sharpness_base_scale = 2500.0
    config.edge_strength_base_scale = 500.0
    config.quality_contrast_clamp = 50.0
    config.ffmpeg_thumbnail_quality = 100
    config.cache_cleanup_threshold = 0.9
    config.cache_size = 100
    config.cache_thumbnail_max_size = 100
    config.thumbnail_cache_max_mb = 100
    config.cache_eviction_factor = 0.5
    config.seeding_iou_threshold = 0.5
    # Analysis params
    config.analysis_default_workers = 4
    config.analysis_default_batch_size = 8
    # System params
    config.downloads_dir = "/tmp/downloads"
    config.retry_max_attempts = 1
    config.utility_max_filename_length = 100
    config.utility_image_extensions = [".jpg", ".png", ".webp", ".jpeg"]
    config.image_extensions = config.utility_image_extensions  # Fallback
    config.filename_max_length = config.utility_max_filename_length  # Fallback

    # Utils params
    config.masking_close_kernel_size = 3
    config.masking_keep_largest_only = True
    config.visualization_bbox_color = (255, 0, 0)
    config.visualization_bbox_thickness = 2

    # Filter defaults
    config.filter_default_quality_score = {"default_min": 0.0, "default_max": 100.0}
    config.filter_default_face_sim = {"default_min": 0.0, "default_max": 1.0}
    config.filter_default_mask_area_pct = {"default_min": 0.0}
    config.filter_default_eyes_open = {"default_min": 0.0}
    config.filter_default_sharpness = {"default_min": 0.0, "default_max": 100.0}
    config.filter_default_edge_strength = {"default_min": 0.0, "default_max": 100.0}
    config.filter_default_contrast = {"default_min": 0.0, "default_max": 100.0}
    config.filter_default_brightness = {"default_min": 0.0, "default_max": 100.0}
    config.filter_default_entropy = {"default_min": 0.0, "default_max": 100.0}
    config.filter_default_niqe = {"default_min": 0.0, "default_max": 100.0}
    config.filter_default_yaw = {"min": -180, "max": 180}
    config.filter_default_pitch = {"min": -180, "max": 180}

    # Mock model_dump
    config.model_dump.return_value = {
        "sharpness_base_scale": 2500.0,
        "edge_strength_base_scale": 500.0,
        "quality_contrast_clamp": 50.0,
        "default_thumb_megapixels": 0.5,
        "quality_weights_sharpness": 20.0,
        "quality_weights_edge_strength": 20.0,
        "quality_weights_contrast": 20.0,
        "quality_weights_brightness": 10.0,
        "quality_weights_entropy": 10.0,
        "quality_weights_niqe": 20.0,
        "filter_default_quality_score": {"default_min": 0.0, "default_max": 100.0},
        "filter_default_face_sim": {"default_min": 0.0, "default_max": 1.0},
        "filter_default_mask_area_pct": {"default_min": 0.0},
        "filter_default_eyes_open": {"default_min": 0.0},
    }
    return config


@pytest.fixture
def mock_config_simple(mock_config):
    """Alias for mock_config used by some tests."""
    return mock_config


@pytest.fixture
def sample_frames_data():
    """Provides sample frame metadata for filtering tests."""
    return [
        {"filename": "frame_01.png", "face_sim": 0.8, "mask_area_pct": 15.0, "metrics": {"quality_score": 85}},
        {"filename": "frame_02.png", "face_sim": 0.7, "mask_area_pct": 12.0, "metrics": {"quality_score": 75}},
        {"filename": "frame_03.png", "face_sim": 0.6, "mask_area_pct": 20.0, "metrics": {"quality_score": 65}},
        {"filename": "frame_04.png", "face_sim": 0.4, "mask_area_pct": 10.0, "metrics": {"quality_score": 55}},
        {"filename": "frame_05.png", "face_sim": 0.9, "mask_area_pct": 5.0, "metrics": {"quality_score": 45}},
        {"filename": "frame_06.png", "face_sim": None, "mask_area_pct": 0.0, "metrics": {"quality_score": 35}},
    ]


@pytest.fixture
def mock_ui_state():
    """Provides a dictionary with default values for UI-related event models."""
    return {
        "source_path": "video.mp4",
        "video_path": "video.mp4",
        "output_folder": "/tmp/out",
        "method": "🤖 Automatic",
        "interval": 1.0,
        "max_resolution": "720p",
        "disable_parallel": False,
        "resume": False,
        "enable_face_filter": False,
        "face_ref_img_path": "",
        "face_model_name": "ghostface",
        "enable_subject_mask": False,
        "tracker_model_name": "sam2",
        "best_frame_strategy": "🤖 Automatic",
        "scene_detect": True,
        "nth_frame": 1,
        "require_face_match": False,
        "text_prompt": "",
        "thumbnails_only": True,
        "thumb_megapixels": 0.5,
        "pre_analysis_enabled": True,
        "pre_sample_nth": 1,
        "primary_seed_strategy": "🧑\u200d🤝\u200d🧑 Find Prominent Person",
        "min_mask_area_pct": 1.0,
        "sharpness_base_scale": 2500.0,
        "edge_strength_base_scale": 100.0,
        "compute_quality_score": True,
        "compute_sharpness": True,
        "compute_edge_strength": True,
        "compute_contrast": True,
        "compute_brightness": True,
        "compute_entropy": True,
        "compute_eyes_open": True,
        "compute_yaw": True,
        "compute_pitch": True,
        "compute_face_sim": True,
        "compute_subject_mask_area": True,
        "compute_niqe": True,
        "compute_phash": True,
    }


@pytest.fixture
def mock_params(mock_ui_state):
    """Provides an AnalysisParameters instance for testing."""
    from core.models import AnalysisParameters

    return AnalysisParameters(**mock_ui_state)


@pytest.fixture
def mock_thumbnail_manager():
    """Provides a mock ThumbnailManager."""
    tm = MagicMock()
    tm.get.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
    return tm


@pytest.fixture
def mock_model_registry():
    """Provides a mock ModelRegistry."""
    return MagicMock()


@pytest.fixture
def mock_progress_queue():
    """Provides a mock progress queue."""
    return Queue()


@pytest.fixture
def mock_cancel_event():
    """Provides a mock cancel event."""
    return threading.Event()


@pytest.fixture
def sample_scenes():
    """Provides a list of sample Scene objects."""
    from core.models import Scene

    return [
        Scene(shot_id=1, start_frame=0, end_frame=100, status="pending"),
        Scene(shot_id=2, start_frame=101, end_frame=200, status="included"),
        Scene(shot_id=3, start_frame=201, end_frame=300, status="excluded"),
    ]


def pytest_sessionstart(session):
    """Mocks are now initialized at the module level for early interception."""
    pass


def pytest_sessionfinish(session, exitstatus):
    """Restore original modules after the session."""
    if hasattr(session, "original_modules"):
        for mod_name in session.original_modules:
            sys.modules[mod_name] = session.original_modules[mod_name]


@pytest.fixture
def mock_torch():
    """Fixture to access the mocked torch module."""
    return sys.modules["torch"]


def pytest_addoption(parser):
    """Add command line options."""
    parser.addoption(
        "--capture-golden",
        action="store_true",
        default=False,
        help="Capture current legacy metrics as golden reference for regression tests",
    )
