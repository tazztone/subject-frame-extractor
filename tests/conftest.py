import os
import sys
import types
from unittest.mock import MagicMock

import numpy as np

# --- 1. Mock Environment (Must run before any core/ imports) ---
from tests.helpers.mock_env import (
    OutOfMemoryError as OutOfMemoryError,
)
from tests.helpers.mock_env import (
    TransparentContext as TransparentContext,
)
from tests.helpers.mock_env import (
    _cuda_is_available_mock as _cuda_is_available_mock,
)
from tests.helpers.mock_env import (
    _torch_mod as _torch_mod,
)
from tests.helpers.mock_env import (
    install_mocks as install_mocks,
)
from tests.helpers.mock_env import (
    make_svmem as make_svmem,
)
from tests.helpers.mock_env import (
    modules_to_mock as modules_to_mock,
)
from tests.helpers.mock_env import (
    set_cuda_available as set_cuda_available,
)
from tests.helpers.mock_env import (
    svmem as svmem,
)


def _should_skip_mocks():
    return os.environ.get("PYTEST_INTEGRATION_MODE") == "true" or "tests/ui" in " ".join(sys.argv)


if not _should_skip_mocks():
    install_mocks()

# --- 2. Now do standard imports ---
import pytest


@pytest.fixture(autouse=True)
def _restore_mock_modules():
    """Snapshot mock module attributes before each test and restore after.

    Prevents unittest.mock.patch() teardown from permanently corrupting our
    types.ModuleType mock objects. patch() uses delattr() on teardown for
    attributes it created, and may also leave replaced values if teardown
    fails. This fixture unconditionally restores the exact pre-test state.

    Uses vars() (not dir()) to snapshot only __dict__ entries, avoiding
    inherited/read-only properties that can't be re-set.
    """
    if _should_skip_mocks():
        yield
        return

    # BEFORE test: snapshot __dict__ of every ModuleType mock + re-inject sys.modules
    snapshots = {}
    for name, mod in modules_to_mock.items():
        if isinstance(mod, types.ModuleType):
            snapshots[name] = dict(vars(mod))
        sys.modules[name] = mod  # re-inject in case a prior test replaced it

    yield

    # AFTER test: unconditionally restore attribute values on ModuleType mocks
    for name, attrs in snapshots.items():
        mod = modules_to_mock[name]
        if isinstance(mod, types.ModuleType):
            # Restore original attributes
            for k, v in attrs.items():
                try:
                    setattr(mod, k, v)
                except (AttributeError, TypeError):
                    pass
            # Remove attributes that were added during the test
            for k in list(vars(mod).keys()):
                if k not in attrs:
                    try:
                        delattr(mod, k)
                    except (AttributeError, TypeError):
                        pass

    # Re-inject modules in case test replaced them
    for name, mod in modules_to_mock.items():
        if sys.modules.get(name) is not mod:
            sys.modules[name] = mod


@pytest.fixture(autouse=True)
def clean_gpu_mock():
    """Ensure CUDA mock is reset between tests."""
    if not _should_skip_mocks():
        _cuda_is_available_mock.return_value = False
    yield


@pytest.fixture
def mock_config():
    from core.config import Config

    return Config()


@pytest.fixture
def mock_config_simple():
    from core.config import Config

    return Config()


@pytest.fixture
def mock_logger():
    return MagicMock()


@pytest.fixture
def mock_queue():
    from queue import Queue

    return Queue()


@pytest.fixture
def mock_cancel_event():
    import threading

    return threading.Event()


@pytest.fixture
def mock_thumbnail_manager():
    tm = MagicMock()
    tm.get.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
    return tm


@pytest.fixture
def mock_params():
    from core.models import AnalysisParameters

    return AnalysisParameters(
        output_folder="/tmp/test",
        video_path="test.mp4",
    )


@pytest.fixture
def mock_model_registry():
    mr = MagicMock()
    mr.get_tracker.return_value = MagicMock()
    mr.get_subject_detector.return_value = MagicMock()
    mr.get_face_analyzer.return_value = MagicMock()
    return mr


@pytest.fixture
def sample_scenes():
    from core.models import Scene

    return [
        Scene(shot_id=1, start_frame=0, end_frame=100),
        Scene(shot_id=2, start_frame=100, end_frame=200),
        Scene(shot_id=3, start_frame=200, end_frame=300),
        Scene(shot_id=4, start_frame=300, end_frame=400),
    ]


@pytest.fixture
def mock_progress_queue():
    from queue import Queue

    return Queue()


@pytest.fixture
def mock_ui_state():
    return {
        "video_path": "test.mp4",
        "output_folder": "/tmp/out",
        "thumbnails_only": True,
        "scene_detect": True,
        "max_resolution": "1080",
        "pre_analysis_enabled": True,
    }


@pytest.fixture
def sample_frames_data():
    return [
        {"filename": "frame_01.png", "metrics": {"quality_score": 85, "sharpness": 90}},
        {"filename": "frame_02.png", "metrics": {"quality_score": 70, "sharpness": 80}},
        {"filename": "frame_03.png", "metrics": {"quality_score": 40, "sharpness": 30}},
    ]


@pytest.fixture
def sample_image():
    return np.zeros((100, 100, 3), dtype=np.uint8)


@pytest.fixture
def sample_mask():
    return np.zeros((100, 100), dtype=np.uint8)


@pytest.fixture
def sharp_image():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[50:, :] = 255  # Hard edge
    return img
