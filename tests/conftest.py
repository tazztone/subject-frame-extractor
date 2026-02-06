"""
Shared pytest fixtures and configuration.

This module provides common fixtures for operator tests and regression tests,
reducing code duplication and ensuring consistency.
"""

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
    # Analysis params
    config.analysis_default_workers = 4
    config.analysis_default_batch_size = 8
    # System params
    config.downloads_dir = "/tmp/downloads"
    config.retry_max_attempts = 1
    return config


def pytest_addoption(parser):
    """Add command line options."""
    parser.addoption(
        "--capture-golden",
        action="store_true",
        default=False,
        help="Capture current legacy metrics as golden reference for regression tests",
    )
