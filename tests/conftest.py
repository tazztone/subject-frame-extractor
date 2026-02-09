"""
Shared pytest fixtures and configuration.

This module provides common fixtures for operator tests and regression tests,
reducing code duplication and ensuring consistency.
"""

import threading
from queue import Queue
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest
from core.operators import OperatorRegistry


@pytest.fixture(autouse=True)
def clean_registry():
    """Ensure OperatorRegistry is clean before each test."""
    OperatorRegistry.clear()
    yield
    OperatorRegistry.clear()


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
    # Analysis params
    config.analysis_default_workers = 4
    config.analysis_default_batch_size = 8
    # System params
    config.downloads_dir = "/tmp/downloads"
    config.retry_max_attempts = 1
    
    # Utils params
    config.masking_close_kernel_size = 3
    config.masking_keep_largest_only = True
    config.visualization_bbox_color = (255, 0, 0)
    config.visualization_bbox_thickness = 2

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
        "tracker_model_name": "sam3",
        "best_frame_strategy": "🤖 Automatic",
        "scene_detect": True,
        "nth_frame": 1,
        "require_face_match": False,
        "text_prompt": "",
        "thumbnails_only": True,
        "thumb_megapixels": 0.5,
        "pre_analysis_enabled": True,
        "pre_sample_nth": 1,
        "primary_seed_strategy": "🧑‍🤝‍🧑 Find Prominent Person",
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


def pytest_addoption(parser):
    """Add command line options."""
    parser.addoption(
        "--capture-golden",
        action="store_true",
        default=False,
        help="Capture current legacy metrics as golden reference for regression tests",
    )