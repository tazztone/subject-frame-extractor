import sys
from collections import namedtuple
from types import ModuleType
from unittest.mock import MagicMock

import numpy as np
import pytest

# 1. BOOTSTRAP STABLE MOCK HELPERS
# We use a proper class for MockTensor to ensure stability in parallel execution
from tests.helpers.mock_tensor import create_mock_tensor

DeviceProps = namedtuple("DeviceProps", ["total_memory"])

# 2. DEFINITION OF ESSENTIAL TOP-LEVEL MOCKS
# We use ModuleType to satisfy the import system (prevent AttributeError: __spec__)
# and use __getattr__ to automatically mock submodules.


class MockModule(ModuleType):
    def __init__(self, name, attrs=None):
        super().__init__(name)
        self.__file__ = f"<mock {name}>"
        self.__path__ = []
        if attrs:
            for k, v in attrs.items():
                setattr(self, k, v)

    def __getattr__(self, name):
        # Return a fresh mock for any missing attribute/submodule
        # We now use setattr to allow patching to work correctly
        m = MagicMock(name=f"{self.__name__}.{name}")
        setattr(self, name, m)
        return m


def _create_mock_module(name, attrs=None):
    return MockModule(name, attrs)


from tests.helpers.exceptions import OutOfMemoryError, VideoOpenFailure

# Alias for backward compatibility in existing tests
createmocktensor = create_mock_tensor

cuda_mock = _create_mock_module(
    "torch.cuda",
    {
        "is_available": MagicMock(return_value=False),
        "device_count": MagicMock(return_value=0),
        "OutOfMemoryError": OutOfMemoryError,
        "empty_cache": MagicMock(),
        "memory_allocated": MagicMock(return_value=0),
        "memory_reserved": MagicMock(return_value=0.0),
        "get_device_properties": MagicMock(return_value=DeviceProps(total_memory=8 * 1024**3)),
    },
)

# 3. CORE MODULE MAP
modules_to_mock = {
    "torch": _create_mock_module(
        "torch",
        {
            "Tensor": MagicMock,
            "from_numpy": MagicMock(side_effect=lambda arr, **k: create_mock_tensor("from_numpy", shape=arr.shape)),
            "zeros": MagicMock(side_effect=lambda shape, *a, **k: create_mock_tensor("zeros", shape=shape)),
            "ones": MagicMock(side_effect=lambda shape, *a, **k: create_mock_tensor("ones", shape=shape)),
            "randn": MagicMock(side_effect=lambda shape, *a, **k: create_mock_tensor("randn", shape=shape)),
            "cat": MagicMock(side_effect=lambda tensors, *a, **k: create_mock_tensor("cat", shape=tensors[0].shape)),
            "stack": MagicMock(
                side_effect=lambda tensors, *a, **k: create_mock_tensor(
                    "stack", shape=(len(tensors),) + tensors[0].shape
                )
            ),
            "device": MagicMock,
            "nn": _create_mock_module("torch.nn", {"Module": MagicMock, "Parameter": MagicMock}),
            "cuda": cuda_mock,
            "float": MagicMock(),
            "float32": MagicMock(),
            "float16": MagicMock(),
            "bfloat16": MagicMock(),
            "uint8": MagicMock(),
            "int64": MagicMock(),
            "int32": MagicMock(),
            "bool": MagicMock(),
            "long": MagicMock(),
        },
    ),
    "torch.cuda": cuda_mock,
    "torchvision": _create_mock_module("torchvision", {"transforms": MagicMock(), "ops": MagicMock()}),
    "cv2": _create_mock_module(
        "cv2",
        {
            "imread": MagicMock(return_value=np.zeros((100, 100, 3), dtype=np.uint8)),
            "cvtColor": MagicMock(side_effect=lambda x, *a, **k: x),
            "resize": MagicMock(
                side_effect=lambda x, size, **k: (
                    np.zeros((size[1], size[0]) + x.shape[2:], dtype=getattr(x, "dtype", np.uint8))
                    if hasattr(x, "shape") and len(x.shape) > 2
                    else np.zeros((size[1], size[0]), dtype=getattr(x, "dtype", np.uint8))
                )
            ),
            "addWeighted": MagicMock(
                side_effect=lambda src1, a1, src2, a2, g, **k: (
                    src1.astype(float) * a1 + src2.astype(float) * a2 + g
                ).astype(np.uint8)
            ),
            "morphologyEx": MagicMock(side_effect=lambda x, op, k, **kwargs: x),
            "getStructuringElement": MagicMock(return_value=np.ones((3, 3), dtype=np.uint8)),
            "rectangle": MagicMock(
                side_effect=lambda img, pt1, pt2, color, *a, **k: img.__setitem__(
                    (slice(max(0, pt1[1]), max(0, pt2[1])), slice(max(0, pt1[0]), max(0, pt2[0]))),
                    (max(color) if isinstance(color, (tuple, list, np.ndarray)) and len(color) > 0 else 255),
                )
            ),
            "putText": MagicMock(),
            "Sobel": MagicMock(side_effect=lambda x, *a, **k: x),
            "Laplacian": MagicMock(side_effect=lambda x, *a, **k: x),
            "calcHist": MagicMock(
                side_effect=lambda images, *a, **k: (
                    np.histogram(images[0], bins=256, range=(0, 256))[0].astype(np.float32).reshape(-1, 1)
                )
            ),
            "connectedComponentsWithStats": MagicMock(
                side_effect=lambda mask, **k: (
                    2,
                    (mask > 0).astype(np.int32),
                    np.array([[0, 0, 0, 0, 0], [0, 0, mask.shape[1], mask.shape[0], np.sum(mask > 0)]], dtype=np.int32),
                    np.array([[0, 0], [mask.shape[1] // 2, mask.shape[0] // 2]], dtype=np.float32),
                )
            ),
            "findContours": MagicMock(return_value=([], None)),
            "getTextSize": MagicMock(return_value=((50, 20), 5)),
            "IMREAD_UNCHANGED": 0,
            "IMREAD_COLOR": 1,
            "IMREAD_GRAYSCALE": 2,
            "COLOR_RGB2BGR": 4,
            "COLOR_BGR2RGB": 4,
            "COLOR_RGB2GRAY": 6,
            "COLOR_BGR2GRAY": 6,
            "INTER_AREA": 3,
            "INTER_LINEAR": 1,
            "INTER_NEAREST": 0,
            "CV_64F": 6,
            "CV_32F": 5,
            "CV_8U": 0,
            "MORPH_CLOSE": 1,
            "MORPH_ELLIPSE": 1,
            "CC_STAT_AREA": 4,
        },
    ),
    "sam2": _create_mock_module("sam2", {"build_sam2_video_predictor": MagicMock()}),
    "sam2.build_sam": _create_mock_module("sam2.build_sam", {"build_sam2_video_predictor": MagicMock()}),
    "sam3": _create_mock_module("sam3", {"build_sam3_video_model": MagicMock()}),
    "sam3.model_builder": _create_mock_module("sam3.model_builder", {"build_sam3_predictor": MagicMock()}),
    "scenedetect": _create_mock_module(
        "scenedetect",
        {
            "ContentDetector": MagicMock(),
            "VideoOpenFailure": VideoOpenFailure,
            "detect": MagicMock(),
        },
    ),
    "scenedetect.detectors": _create_mock_module("scenedetect.detectors", {"ContentDetector": MagicMock()}),
    "insightface": _create_mock_module("insightface", {"app": MagicMock()}),
    "insightface.app": _create_mock_module("insightface.app", {"FaceAnalysis": MagicMock()}),
    "pyiqa": _create_mock_module("pyiqa", {"create_metric": MagicMock()}),
    "lpips": _create_mock_module("lpips", {"LPIPS": MagicMock()}),
    "onnxruntime": _create_mock_module(
        "onnxruntime",
        {"InferenceSession": MagicMock(), "SessionOptions": MagicMock(), "GraphOptimizationLevel": MagicMock()},
    ),
}


# 4. INJECTION LOGIC
def _should_skip_mocks():
    import os

    if os.environ.get("PYTEST_INTEGRATION_MODE") == "true":
        return True
    # If explicitly running integration or E2E tests, skip mocks
    integration_paths = ["tests/integration/", "tests/e2e/", "tests/ui/"]
    for arg in sys.argv:
        if any(p in arg for p in integration_paths) and "tests/unit/" not in arg:
            return True
    return False


def _inject_global_mocks():
    if _should_skip_mocks():
        return

    # 1. First Pass: Inject all into sys.modules
    for mod_name, mod_obj in modules_to_mock.items():
        sys.modules[mod_name] = mod_obj

    # 2. Second Pass: Structural Linkage (Parent -> Child attributes)
    # Ensure that for 'a.b.c', sys.modules['a'].b is sys.modules['a.b']
    for mod_name in sorted(modules_to_mock.keys()):
        if "." in mod_name:
            parent_name, child_name = mod_name.rsplit(".", 1)
            if parent_name in sys.modules:
                parent_mod = sys.modules[parent_name]
                child_mod = sys.modules[mod_name]
                # Use object.__setattr__ to bypass MockModule.__setattr__ or auto-generation
                object.__setattr__(parent_mod, child_name, child_mod)

    # 3. CRITICAL: Specific sync for torch.cuda
    # Even if handled by the loop above, we force it for absolute certainty
    torch_mod = sys.modules.get("torch")
    cuda_mod = sys.modules.get("torch.cuda")
    if torch_mod is not None and cuda_mod is not None:
        object.__setattr__(torch_mod, "cuda", cuda_mod)


# 5. INITIALIZE TEST ENVIRONMENT
_inject_global_mocks()


@pytest.fixture(scope="session", autouse=True)
def initialize_operators():
    """Initialize operators once per session after mocks are injected."""
    if _should_skip_mocks():
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
    import numpy as np

    np.random.seed(42)
    return np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)


@pytest.fixture
def sample_mask():
    """100x100 grayscale mask (center region active)."""
    import numpy as np

    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[25:75, 25:75] = 255
    return mask


@pytest.fixture
def sharp_image():
    """High-frequency checkerboard pattern (sharp)."""
    import numpy as np

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
    import numpy as np

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
