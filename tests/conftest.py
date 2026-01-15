"""
Centralized pytest fixtures for Frame Extractor & Analyzer tests.

This module provides reusable mock fixtures for testing, avoiding duplication
across test files and improving test maintainability.
"""

import os
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pydantic
import pytest

# Add project root to the Python path to allow for submodule imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# =============================================================================
# MODULE-LEVEL MOCK SETUP
# =============================================================================

# --- Create mock modules for heavy dependencies ---


def _create_mock_torch():
    """Create a comprehensive mock for torch and its submodules."""
    mock_torch = MagicMock(name="torch")
    mock_torch.__version__ = "2.0.0"
    mock_torch.__path__ = ["fake"]
    mock_torch.__spec__ = MagicMock()
    mock_torch.hub = MagicMock(name="torch.hub")
    mock_torch.cuda = MagicMock(name="torch.cuda")
    mock_torch.cuda.is_available.return_value = False
    mock_torch.distributed = MagicMock(name="torch.distributed")
    mock_torch.multiprocessing = MagicMock(name="torch.multiprocessing")
    mock_torch.amp = MagicMock(name="torch.amp")
    return mock_torch


def _create_mock_torch_submodules(mock_torch):
    """Create mocks for torch submodules like nn, optim, utils."""
    mock_torch_autograd = MagicMock(name="torch.autograd")
    mock_torch_autograd.Variable = MagicMock(name="torch.autograd.Variable")

    mock_torch_nn = MagicMock(name="torch.nn")
    mock_torch_nn.__path__ = ["fake"]

    class MockNNModule:
        """Mock for torch.nn.Module to allow class inheritance."""

        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, *args, **kwargs):
            return MagicMock()

    mock_torch_nn.Module = MockNNModule
    mock_torch_nn.attention = MagicMock(name="torch.nn.attention")

    mock_torch_nn_init = MagicMock(name="torch.nn.init")
    mock_torch_nn_functional = MagicMock(name="torch.nn.functional")
    mock_torch_optim = MagicMock(name="torch.optim")

    mock_torch_utils = MagicMock(name="torch.utils")
    mock_torch_utils.__path__ = ["fake"]
    mock_torch_utils_data = MagicMock(name="torch.utils.data")
    mock_torch_utils_checkpoint = MagicMock(name="torch.utils.checkpoint")
    mock_torch_utils_pytree = MagicMock(name="torch.utils._pytree")

    return {
        "torch.autograd": mock_torch_autograd,
        "torch.nn": mock_torch_nn,
        "torch.nn.attention": mock_torch_nn.attention,
        "torch.nn.init": mock_torch_nn_init,
        "torch.nn.functional": mock_torch_nn_functional,
        "torch.optim": mock_torch_optim,
        "torch.utils": mock_torch_utils,
        "torch.utils.data": mock_torch_utils_data,
        "torch.utils.checkpoint": mock_torch_utils_checkpoint,
        "torch.utils._pytree": mock_torch_utils_pytree,
    }


def _create_mock_torchvision():
    """Create a mock for torchvision."""
    mock = MagicMock(name="torchvision")
    mock.ops = MagicMock(name="torchvision.ops")
    mock.ops.roi_align = MagicMock(name="torchvision.ops.roi_align")
    mock.ops.misc = MagicMock(name="torchvision.ops.misc")
    mock.datasets = MagicMock(name="torchvision.datasets")
    mock.datasets.vision = MagicMock(name="torchvision.datasets.vision")
    mock.transforms = MagicMock(name="torchvision.transforms")
    mock.transforms.functional = MagicMock(name="torchvision.transforms.functional")
    mock.utils = MagicMock(name="torchvision.utils")
    return mock


def _create_mock_psutil():
    """Create a mock for psutil with expected return values."""
    mock = MagicMock(name="psutil")
    mock.cpu_percent.return_value = 50.0
    mock.virtual_memory.return_value = MagicMock(percent=50.0, available=1024 * 1024 * 1024)
    mock.disk_usage.return_value = MagicMock(percent=50.0)
    mock_process = mock.Process.return_value
    mock_process.memory_info.return_value.rss = 100 * 1024 * 1024
    mock_process.cpu_percent.return_value = 10.0
    return mock


def _create_mock_matplotlib():
    """Create a mock for matplotlib."""
    mock = MagicMock(name="matplotlib")
    mock.__path__ = ["fake"]
    mock.ticker = MagicMock(name="matplotlib.ticker")
    mock.figure = MagicMock(name="matplotlib.figure")
    mock.backends = MagicMock(name="matplotlib.backends")
    mock.backends.backend_agg = MagicMock(name="matplotlib.backends.backend_agg")
    return mock


def _create_mock_scenedetect():
    """Create a mock for scenedetect with proper Exception classes."""
    mock = MagicMock(name="scenedetect")

    class MockVideoOpenFailure(Exception):
        """Mock VideoOpenFailure exception."""

        pass

    mock.VideoOpenFailure = MockVideoOpenFailure
    return mock


def build_modules_to_mock():
    """Build the complete dictionary of modules to mock."""
    mock_torch = _create_mock_torch()
    torch_submodules = _create_mock_torch_submodules(mock_torch)
    mock_torchvision = _create_mock_torchvision()
    mock_psutil = _create_mock_psutil()
    mock_matplotlib = _create_mock_matplotlib()
    mock_scenedetect = _create_mock_scenedetect()

    mock_insightface = MagicMock(name="insightface")
    mock_insightface.app = MagicMock(name="insightface.app")

    mock_timm = MagicMock(name="timm")
    mock_timm.models = MagicMock(name="timm.models")
    mock_timm.models.layers = MagicMock(name="timm.models.layers")

    mock_pycocotools = MagicMock(name="pycocotools")
    mock_pycocotools.mask = MagicMock(name="pycocotools.mask")

    # Mock pydantic_settings if not available
    mock_pydantic_settings = MagicMock(name="pydantic_settings")
    mock_pydantic_settings.BaseSettings = pydantic.BaseModel
    mock_pydantic_settings.SettingsConfigDict = dict

    mock_skimage = MagicMock(name="skimage")
    mock_skimage.measure = MagicMock(name="skimage.measure")

    modules = {
        # Torch
        "torch": mock_torch,
        "torch.hub": mock_torch.hub,
        "torch.distributed": mock_torch.distributed,
        "torch.multiprocessing": mock_torch.multiprocessing,
        **torch_submodules,
        # Torchvision
        "torchvision": mock_torchvision,
        "torchvision.ops": mock_torchvision.ops,
        "torchvision.ops.roi_align": mock_torchvision.ops.roi_align,
        "torchvision.ops.misc": mock_torchvision.ops.misc,
        "torchvision.datasets": mock_torchvision.datasets,
        "torchvision.datasets.vision": mock_torchvision.datasets.vision,
        "torchvision.transforms": mock_torchvision.transforms,
        "torchvision.transforms.functional": mock_torchvision.transforms.functional,
        "torchvision.utils": mock_torchvision.utils,
        # Other ML/Vision libs
        "insightface": mock_insightface,
        "insightface.app": mock_insightface.app,
        "timm": mock_timm,
        "timm.models": mock_timm.models,
        "timm.models.layers": mock_timm.models.layers,
        "onnxruntime": MagicMock(name="onnxruntime"),
        # SAM3 mocks
        "sam3": MagicMock(name="sam3"),
        "sam3.model_builder": MagicMock(name="sam3.model_builder"),
        "sam3.model.sam3_video_predictor": MagicMock(name="sam3.model.sam3_video_predictor"),
        # Utils
        "GPUtil": MagicMock(getGPUs=lambda: [MagicMock(memoryUtil=0.5)]),
        "pycocotools": mock_pycocotools,
        "pycocotools.mask": mock_pycocotools.mask,
        "psutil": mock_psutil,
        # Matplotlib
        "matplotlib": mock_matplotlib,
        "matplotlib.ticker": mock_matplotlib.ticker,
        "matplotlib.figure": mock_matplotlib.figure,
        "matplotlib.backends": mock_matplotlib.backends,
        "matplotlib.backends.backend_agg": mock_matplotlib.backends.backend_agg,
        "matplotlib.pyplot": MagicMock(),
        # Other dependencies
        "scenedetect": mock_scenedetect,
        "yt_dlp": MagicMock(),
        "pyiqa": MagicMock(name="pyiqa"),
        "mediapipe": MagicMock(),
        "mediapipe.tasks": MagicMock(),
        "mediapipe.tasks.python": MagicMock(),
        "mediapipe.tasks.python.vision": MagicMock(),
        "lpips": MagicMock(name="lpips"),
        "numba": MagicMock(name="numba"),
        "skimage": mock_skimage,
        "skimage.measure": mock_skimage.measure,
        "skimage.metrics": MagicMock(name="skimage.metrics"),
        "pydantic_settings": mock_pydantic_settings,
    }
    return modules


# Apply module mocks at import time - SKIP for integration tests
# Check if running integration tests by looking at command line args


def _should_apply_mocks():
    """Check if we should apply mocks (skip for integration/smoke/signature/gpu_e2e tests)."""
    import sys

    # Tests that should NOT use mocks
    no_mock_tests = ["test_integration", "test_smoke", "test_signatures", "test_gpu_e2e"]
    no_mock_markers = ["integration", "smoke", "signature", "gpu_e2e"]

    # Check if running specific test files
    if any(test in arg for arg in sys.argv for test in no_mock_tests):
        return False

    # Check if -m marker is specified
    if "-m" in sys.argv:
        try:
            idx = sys.argv.index("-m")
            if idx + 1 < len(sys.argv):
                marker_arg = sys.argv[idx + 1]
                if any(marker in marker_arg for marker in no_mock_markers):
                    return False
        except ValueError:
            pass
    return True


if _should_apply_mocks():
    MODULES_TO_MOCK = build_modules_to_mock()
    patch.dict(sys.modules, MODULES_TO_MOCK).start()
else:
    MODULES_TO_MOCK = {}


# =============================================================================
# PYTEST FIXTURES
# =============================================================================


@pytest.fixture(scope="session")
def mock_torch():
    """Session-scoped mock for torch module."""
    return MODULES_TO_MOCK["torch"]


@pytest.fixture
def mock_config(tmp_path):
    """
    Provides a test Config with temporary directories.

    Use this for tests that need a valid Config object
    with writable paths.
    """
    from core.config import Config

    config = Config(
        logs_dir=str(tmp_path / "logs"),
        models_dir=str(tmp_path / "models"),
        downloads_dir=str(tmp_path / "downloads"),
    )
    return config


@pytest.fixture
def mock_logger(mock_config):
    """Provides a mock AppLogger for testing."""
    from core.logger import AppLogger

    return AppLogger(config=mock_config, log_to_console=False, log_to_file=False)


@pytest.fixture
def mock_thumbnail_manager(mock_logger, mock_config):
    """Provides a mock ThumbnailManager."""
    mock = MagicMock()
    mock.get.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
    mock.clear_cache = MagicMock()
    return mock


@pytest.fixture
def mock_model_registry(mock_logger):
    """Provides a mock ModelRegistry."""
    mock = MagicMock()
    mock.get_or_load = MagicMock(return_value=MagicMock())
    mock.get_tracker = MagicMock(return_value=MagicMock())
    mock.clear = MagicMock()
    return mock


@pytest.fixture
def mock_progress_queue():
    """Provides a mock progress queue."""
    from queue import Queue

    return Queue()


@pytest.fixture
def mock_cancel_event():
    """Provides a mock cancel event."""
    import threading

    return threading.Event()


@pytest.fixture
def mock_ui_state():
    """
    Provides a dictionary with default values for UI-related event models.

    Useful for testing event validation and pipeline execution.
    """
    return {
        "source_path": "test.mp4",
        "upload_video": None,
        "method": "interval",
        "interval": "1.0",
        "nth_frame": "5",
        "max_resolution": "720",
        "thumbnails_only": True,
        "thumb_megapixels": 0.2,
        "scene_detect": True,
        "output_folder": "/fake/output",
        "video_path": "/fake/video.mp4",
        "resume": False,
        "enable_face_filter": False,
        "face_ref_img_path": "",
        "face_ref_img_upload": None,
        "face_model_name": "buffalo_l",
        "enable_subject_mask": False,
        "tracker_model_name": "sam3",
        "best_frame_strategy": "Largest Person",
        "text_prompt": "",
        "min_mask_area_pct": 1.0,
        "sharpness_base_scale": 2500.0,
        "edge_strength_base_scale": 100.0,
        "pre_analysis_enabled": True,
        "pre_sample_nth": 1,
        "primary_seed_strategy": "🧑‍🤝‍🧑 Find Prominent Person",
    }


@pytest.fixture
def sample_frames_data():
    """
    Provides sample frame metadata for filtering tests.

    Includes a mix of good and bad frames to test various filters.
    """
    return [
        {
            "filename": "frame_01.png",
            "phash": "a" * 16,
            "metrics": {"sharpness_score": 50, "contrast_score": 50},
            "face_sim": 0.8,
            "mask_area_pct": 20,
        },
        {
            "filename": "frame_02.png",
            "phash": "a" * 16,
            "metrics": {"sharpness_score": 50, "contrast_score": 50},
            "face_sim": 0.8,
            "mask_area_pct": 20,
        },  # Duplicate of frame_01
        {
            "filename": "frame_03.png",
            "phash": "b" * 16,
            "metrics": {"sharpness_score": 5, "contrast_score": 50},
            "face_sim": 0.8,
            "mask_area_pct": 20,
        },  # Low sharpness
        {
            "filename": "frame_04.png",
            "phash": "c" * 16,
            "metrics": {"sharpness_score": 50, "contrast_score": 50},
            "face_sim": 0.2,
            "mask_area_pct": 20,
        },  # Low face_sim
        {
            "filename": "frame_05.png",
            "phash": "d" * 16,
            "metrics": {"sharpness_score": 50, "contrast_score": 50},
            "face_sim": 0.8,
            "mask_area_pct": 2,
        },  # Low mask_area
        {
            "filename": "frame_06.png",
            "phash": "e" * 16,
            "metrics": {"sharpness_score": 50, "contrast_score": 50},
            "mask_area_pct": 20,
        },  # No face_sim
    ]


@pytest.fixture
def sample_scenes():
    """
    Provides sample Scene objects for scene-related tests.
    """
    from core.models import Scene

    scenes_data = [
        {
            "shot_id": 1,
            "start_frame": 0,
            "end_frame": 100,
            "status": "pending",
            "seed_result": {"details": {"mask_area_pct": 50}},
            "seed_metrics": {"best_face_sim": 0.9, "score": 0.95},
        },
        {
            "shot_id": 2,
            "start_frame": 101,
            "end_frame": 200,
            "status": "pending",
            "seed_result": {"details": {"mask_area_pct": 5}},  # Low mask area
            "seed_metrics": {"best_face_sim": 0.8, "score": 0.9},
        },
        {
            "shot_id": 3,
            "start_frame": 201,
            "end_frame": 300,
            "status": "pending",
            "seed_result": {"details": {"mask_area_pct": 60}},
            "seed_metrics": {"best_face_sim": 0.4, "score": 0.8},
        },  # Low face_sim
        {
            "shot_id": 4,
            "start_frame": 301,
            "end_frame": 400,
            "status": "pending",
            "seed_result": {"details": {"mask_area_pct": 70}},
            "seed_metrics": {"score": 0.7},
        },  # No face_sim
    ]
    return [Scene(**data) for data in scenes_data]


@pytest.fixture
def sample_image_rgb():
    """Provides a sample RGB image for testing."""
    return np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)


@pytest.fixture
def sample_mask():
    """Provides a sample binary mask for testing."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[25:75, 25:75] = 255  # Simple square mask
    return mask


@pytest.fixture
def mock_params(tmp_path):
    """Provides mock AnalysisParameters for pipeline tests."""
    from core.models import AnalysisParameters

    # Create output directory so database tests don't fail
    output_folder = tmp_path / "output"
    output_folder.mkdir(exist_ok=True)

    return AnalysisParameters(
        source_path="test_video.mp4",
        video_path="test_video.mp4",
        output_folder=str(output_folder),
        thumbnails_only=True,
        tracker_model_name="sam3",
    )


@pytest.fixture
def mock_config_simple(tmp_path):
    """Provides a MagicMock config for tests needing attribute flexibility."""
    mock = MagicMock()

    # Create directories so database tests don't fail
    downloads_dir = tmp_path / "downloads"
    models_dir = tmp_path / "models"
    logs_dir = tmp_path / "logs"
    downloads_dir.mkdir(exist_ok=True)
    models_dir.mkdir(exist_ok=True)
    logs_dir.mkdir(exist_ok=True)

    mock.downloads_dir = downloads_dir
    mock.models_dir = models_dir
    mock.logs_dir = logs_dir
    mock.ffmpeg_thumbnail_quality = 80
    mock.retry_max_attempts = 1
    mock.retry_backoff_seconds = (0.1,)
    mock.monitoring_memory_warning_threshold_mb = 1000
    mock.analysis_default_workers = 1
    mock.analysis_default_batch_size = 1
    mock.sharpness_base_scale = 1.0
    mock.edge_strength_base_scale = 1.0
    mock.utility_max_filename_length = 255
    mock.cache_size = 10
    mock.cache_cleanup_threshold = 0.8
    mock.cache_eviction_factor = 0.5
    mock.default_max_resolution = "1080"
    mock.seeding_iou_threshold = 0.5
    mock.seeding_face_contain_score = 10
    mock.seeding_confidence_score_multiplier = 1
    mock.seeding_iou_bonus = 5
    mock.seeding_balanced_score_weights = {"area": 1, "confidence": 1, "edge": 1}
    mock.seeding_face_to_body_expansion_factors = [1.5, 3.0, 1.0]
    mock.seeding_final_fallback_box = [0.25, 0.25, 0.75, 0.75]
    mock.visualization_bbox_color = (0, 255, 0)
    mock.visualization_bbox_thickness = 2
    return mock
