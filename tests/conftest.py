import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# 1. BOOTSTRAP STABLE MOCK HELPERS
# We use a proper class for MockTensor to ensure stability in parallel execution
from tests.helpers.mock_tensor import create_mock_tensor

# 2. BOOTSTRAP MOCK MODULES

# Alias for backward compatibility
createmocktensor = create_mock_tensor


# 3. INJECTION LOGIC


def pytest_configure(config):
    """Ensure mocks are injected before test modules are collected."""
    if os.environ.get("PYTEST_INTEGRATION_MODE", "").lower() != "true":
        from tests.helpers.sys_mock_modules import inject_mocks_into_sys

        inject_mocks_into_sys()


# 5. INITIALIZE TEST ENVIRONMENT
# Top-level injection removed; now handled by pytest_configure to avoid
# early import pollutants and respect integration mode more reliably.


@pytest.fixture(scope="session", autouse=True)
def initialize_operators():
    """Initialize operators once per session after mocks are injected."""
    if os.environ.get("PYTEST_INTEGRATION_MODE", "").lower() == "true":
        yield
        return

    try:
        from core.operators import OperatorRegistry

        # Avoid double-initialization in some environments
        if not OperatorRegistry.list_names():
            from core.operators import discover_operators

            discover_operators()

        # We use a mock config to satisfy initialization
        from unittest.mock import MagicMock

        OperatorRegistry.initialize_all(MagicMock())
    except Exception as e:
        # Don't let initialization failure crash the entire session if some tests don't need it
        print(f"\nWarning: Failed to initialize operators in conftest: {e}")
    yield


# --- GLOBAL FIXTURES ---


@pytest.fixture
def mock_config():
    from core.config import Config

    return Config()


@pytest.fixture
def mock_config_simple(mock_config):
    return mock_config


@pytest.fixture
def mock_logger():
    return MagicMock(name="mock_logger")


@pytest.fixture
def mock_progress_queue():
    import queue

    return queue.Queue()


@pytest.fixture
def mock_cancel_event():
    import threading

    return threading.Event()


@pytest.fixture
def mock_thumbnail_manager():
    return MagicMock(name="mock_thumbnail_manager")


@pytest.fixture
def mock_ui_state():
    """Provides a dictionary with default values for UI-related event models."""
    return {
        "video_path": "test.mp4",
        "output_folder": "/tmp",
        "primary_seed_strategy": "Automatic Detection",
    }


@pytest.fixture
def mock_params(mock_ui_state):
    """Provides an AnalysisParameters instance for testing."""
    from core.models import AnalysisParameters

    return AnalysisParameters(**mock_ui_state)


@pytest.fixture
def mock_model_registry():
    return MagicMock(name="mock_model_registry")


@pytest.fixture
def mock_database():
    db = MagicMock(name="mock_database")
    db.__enter__.return_value = db
    return db


@pytest.fixture
def sample_image():
    """100x100 RGB image with random noise (seed 42 for consistency)."""

    np.random.seed(42)
    return np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)


@pytest.fixture
def sample_mask():
    """100x100 grayscale mask (center region active)."""

    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[25:75, 25:75] = 255
    return mask


@pytest.fixture
def sharp_image():
    """High-frequency checkerboard pattern (sharp)."""

    img = np.zeros((100, 100, 3), dtype=np.uint8)
    # 10x10 checkerboard
    for i in range(10):
        for j in range(10):
            if (i + j) % 2 == 0:
                img[i * 10 : (i + 1) * 10, j * 10 : (j + 1) * 10] = 255
    return img


@pytest.fixture
def blurry_image():
    """Uniform gray (low frequency/blurry)."""

    return np.full((100, 100, 3), 128, dtype=np.uint8)


@pytest.fixture
def sample_frames_data():
    """Provides sample frame metadata for filtering tests."""
    return [
        {"filename": "frame_0000.jpg", "metrics": {"quality_score": 85.0, "niqe": 15.0}},
        {"filename": "frame_0001.jpg", "metrics": {"quality_score": 40.0, "niqe": 45.0}},
        {"filename": "frame_0002.jpg", "metrics": {"quality_score": 92.0, "niqe": 12.0}},
    ]


@pytest.fixture
def sample_scenes():
    """Provides a list of sample Scene objects."""
    from core.models import Scene

    return [
        Scene(shot_id=1, start_frame=0, end_frame=100, status="included"),
        Scene(shot_id=2, start_frame=100, end_frame=250, status="excluded", rejection_reasons=["blurry"]),
        Scene(shot_id=3, start_frame=250, end_frame=400, status="included"),
    ]


@pytest.fixture(autouse=True)
def force_cpu_device():
    """Forces core.utils.device to return 'cpu' and False for CUDA availability.

    This ensures that unit tests running in the same process as the mock
    infrastructure behave deterministically.
    """
    if os.environ.get("PYTEST_INTEGRATION_MODE", "").lower() == "true":
        # Smoke assertion: even in integration mode, we need a valid device detection state
        from core.utils.device import get_device

        try:
            get_device()
        except Exception as e:
            pytest.fail(f"Integration mode failed smoke device check: {e}")
        yield
        return

    with (
        patch("core.utils.device.get_device", return_value="cpu"),
        patch("core.utils.device.is_cuda_available", return_value=False),
        patch("core.utils.device.get_gpu_memory_pressure", return_value=0.0),
    ):
        yield
