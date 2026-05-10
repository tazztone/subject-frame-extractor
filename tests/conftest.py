import os
import sys
import types
from collections import namedtuple
from unittest.mock import MagicMock

import numpy as np

# --- 1. Infrastructure for Mocking (Must be top-level) ---


def _should_skip_mocks():
    return os.environ.get("PYTEST_INTEGRATION_MODE") == "true" or "tests/ui" in " ".join(sys.argv)


def _create_mock_module(name, attributes=None):
    mock_mod = types.ModuleType(name)
    mock_mod.__package__ = name.rpartition(".")[0]
    mock_mod.__path__ = []
    if attributes:
        for attr, val in attributes.items():
            setattr(mock_mod, attr, val)
    return mock_mod


# Stable exception classes
class OutOfMemoryError(RuntimeError):
    pass


class VideoOpenFailure(RuntimeError):
    pass


class TransparentContext:
    def __enter__(self, *args, **kwargs):
        return self

    def __exit__(self, *args, **kwargs):
        pass

    def __call__(self, func=None):
        return func if func is not None else self


# Global mock toggles
_cuda_is_available_mock = MagicMock(return_value=False)


def set_cuda_available(available: bool):
    _cuda_is_available_mock.return_value = available


# Build mocks
svmem = namedtuple("svmem", ["total", "available", "percent", "used", "free"])


def make_svmem(available_mb=4096, percent=50.0, total_gb=8):
    total = total_gb * 1024**3
    avail = available_mb * 1024 * 1024
    return svmem(total, avail, percent, total - avail, avail)


# Stub torch creation functions to return mocks with correct shape
def _create_mock_tensor(name="tensor", shape=None, value=None, **kwargs):
    if isinstance(shape, int):
        shape = (shape,)
    mock_t = MagicMock(name=name)
    mock_t._mock_shape = shape
    if shape is not None:
        mock_t.shape = shape
    mock_t.device = MagicMock()
    mock_t.dtype = MagicMock()
    mock_t.size.side_effect = lambda dim=None: shape if dim is None else shape[dim]
    mock_t.cpu.return_value = mock_t
    mock_t.numpy.return_value = np.zeros(shape) if shape else np.array([value if value is not None else 1.0])
    mock_t.tolist.return_value = [value if value is not None else 1.0]

    def mock_getitem(idx):
        new_shape = shape[1:] if shape and len(shape) > 1 else None
        return _create_mock_tensor(f"{name}_slice", new_shape, value, **kwargs)

    mock_t.__getitem__ = MagicMock(side_effect=mock_getitem)
    mock_t.__iter__ = MagicMock(side_effect=lambda: iter([mock_t]))
    mock_t.__len__ = MagicMock(return_value=shape[0] if shape else 1)
    mock_t.__gt__ = MagicMock(return_value=mock_t)
    mock_t.__lt__ = MagicMock(return_value=mock_t)
    mock_t.__ge__ = MagicMock(return_value=mock_t)
    mock_t.__le__ = MagicMock(return_value=mock_t)
    mock_t.__bool__ = MagicMock(return_value=True)
    mock_t.__eq__ = MagicMock(return_value=mock_t)
    mock_t.__ne__ = MagicMock(return_value=mock_t)
    mock_t.__gt__ = MagicMock(return_value=mock_t)
    mock_t.__lt__ = MagicMock(return_value=mock_t)
    mock_t.ndim = len(shape) if shape else 0
    mock_t.float.return_value = mock_t
    mock_t.permute.return_value = mock_t
    mock_t.unsqueeze.return_value = mock_t
    mock_t.item.return_value = value if value is not None else 1.0
    return mock_t


_cuda_mod = _create_mock_module(
    "torch.cuda",
    {
        "is_available": _cuda_is_available_mock,
        "device_count": MagicMock(return_value=0),
        "get_device_name": MagicMock(return_value="NVIDIA Mock GPU"),
        "get_device_properties": MagicMock(return_value=MagicMock(total_memory=16 * 1024**3)),
        "empty_cache": MagicMock(),
        "synchronize": MagicMock(),
        "current_device": MagicMock(return_value=0),
        "OutOfMemoryError": OutOfMemoryError,
        "amp": _create_mock_module("torch.cuda.amp", {"autocast": TransparentContext}),
        "memory_allocated": MagicMock(return_value=0),
    },
)

_nn_mod = _create_mock_module("torch.nn", {"Module": MagicMock, "functional": MagicMock(), "__path__": []})
_mp_mod = _create_mock_module("torch.multiprocessing", {"set_start_method": MagicMock(), "__path__": []})

_torch_mod = _create_mock_module(
    "torch",
    {
        "__path__": [],
        "__version__": "2.0.0",
        "cuda": _cuda_mod,
        "version": _create_mock_module("torch.version", {"cuda": "12.1", "__version__": "2.0.0"}),
        "no_grad": TransparentContext,
        "inference_mode": TransparentContext,
        "all": MagicMock(return_value=True),
        "from_numpy": MagicMock(side_effect=lambda x: _create_mock_tensor("from_numpy", x.shape)),
        "zeros": MagicMock(side_effect=lambda shape, **kwargs: _create_mock_tensor("zeros", shape)),
        "ones": MagicMock(side_effect=lambda shape, **kwargs: _create_mock_tensor("ones", shape)),
        "tensor": MagicMock(
            side_effect=lambda data, **kwargs: _create_mock_tensor("tensor", getattr(data, "shape", None), data)
        ),
        "float32": MagicMock(),
        "device": MagicMock(),
        "jit": _create_mock_module("torch.jit", {"script": lambda x: x}),
        "linalg": _create_mock_module("torch.linalg", {"svd": MagicMock(return_value=(None, MagicMock(), None))}),
        "Tensor": MagicMock,
        "stack": MagicMock(side_effect=lambda tensors, **kw: tensors[0] if tensors else MagicMock()),
        "bfloat16": MagicMock(),
        "set_float32_matmul_precision": MagicMock(),
        "nn": _nn_mod,
        "multiprocessing": _mp_mod,
    },
)

# CV2 mock with MagicMock for auto-attribute coverage
_cv2_mod = MagicMock(name="cv2")
_cv2_mod.__name__ = "cv2"
_cv2_mod.__package__ = ""
_cv2_mod.__path__ = []
_cv2_mod.COLOR_RGB2BGR = 4
_cv2_mod.COLOR_BGR2RGB = 4
_cv2_mod.COLOR_RGB2GRAY = 6
_cv2_mod.IMREAD_GRAYSCALE = 0
_cv2_mod.INTER_AREA = 3
_cv2_mod.INTER_NEAREST = 0
_cv2_mod.CV_64F = 6
_cv2_mod.CAP_PROP_FPS = 5
_cv2_mod.CAP_PROP_FRAME_COUNT = 7
_cv2_mod.CAP_PROP_FRAME_WIDTH = 3
_cv2_mod.CAP_PROP_FRAME_HEIGHT = 4
_cv2_mod.CAP_PROP_POS_FRAMES = 1
_cv2_mod.FONT_HERSHEY_SIMPLEX = 0
_cv2_mod.CHAIN_APPROX_SIMPLE = 2
_cv2_mod.RETR_EXTERNAL = 0
_cv2_mod.CC_STAT_AREA = 4
_cv2_mod.MORPH_CLOSE = 3
_cv2_mod.MORPH_ELLIPSE = 2
_cv2_mod.DIST_L2 = 2


def _mock_rectangle(img, pt1, pt2, color, thickness=1, *args, **kwargs):
    if hasattr(img, "__setitem__"):
        try:
            # Safely set a 2x2 area at pt1 to the color for assertions
            y1, x1 = max(0, pt1[1]), max(0, pt1[0])
            y2, x2 = min(img.shape[0], y1 + 2), min(img.shape[1], x1 + 2)
            if y2 > y1 and x2 > x1:
                img[y1:y2, x1:x2] = color
        except Exception:
            pass
    return img


_cv2_mod.rectangle = MagicMock(side_effect=_mock_rectangle)
_cv2_mod.circle = MagicMock(side_effect=_mock_rectangle)
_cv2_mod.line = MagicMock(side_effect=_mock_rectangle)
_cv2_mod.putText = MagicMock(side_effect=lambda img, *a, **kw: img)
_cv2_mod.boundingRect = MagicMock(return_value=(0, 0, 10, 10))
_cv2_mod.getTextSize = MagicMock(return_value=((10, 10), 5))


def _mock_calc_hist(images, channels, mask, histSize, ranges, accumulate=False):
    total = images[0].size if images else 1
    h = np.zeros((histSize[0], 1), dtype=np.float32)
    if len(np.unique(images[0])) <= 1:
        h[0] = float(total)  # Solid color = zero entropy
    else:
        h[:] = float(total) / histSize[0]  # Mixed color = high entropy
    return h


_cv2_mod.calcHist = MagicMock(side_effect=_mock_calc_hist)
_cv2_mod.Laplacian = MagicMock(
    side_effect=lambda src, ddepth, **kwargs: src.astype(np.float64) if hasattr(src, "astype") else src
)
_cv2_mod.Canny = MagicMock(
    side_effect=lambda image, *args, **kwargs: (
        np.zeros_like(image) if len(np.unique(image)) <= 1 else np.ones_like(image) * 255
    )
)
_cv2_mod.VideoCapture = MagicMock()


def _mock_resize(x, dsize, *args, **kwargs):
    if not hasattr(x, "ndim"):
        return x
    out = np.zeros((dsize[1], dsize[0], 3) if x.ndim == 3 else (dsize[1], dsize[0]), dtype=x.dtype)
    if hasattr(x, "any") and x.any():
        out[:] = x.max()
    return out


_cv2_mod.resize = MagicMock(side_effect=_mock_resize)
_cv2_mod.cvtColor = MagicMock(side_effect=lambda x, *args: x)
_cv2_mod.Sobel = MagicMock(
    side_effect=lambda src, *args, **kwargs: np.zeros_like(src) if len(np.unique(src)) <= 1 else np.ones_like(src) * 255
)
_cv2_mod.imread = MagicMock(return_value=None)
_cv2_mod.imencode = MagicMock(return_value=(True, b""))

_cv2_mod.morphologyEx = MagicMock(side_effect=lambda src, *args, **kwargs: src)
_cv2_mod.getStructuringElement = MagicMock(
    side_effect=lambda shape, ksize, **kwargs: np.ones((ksize[1], ksize[0]), dtype=np.uint8)
)
_cv2_mod.addWeighted = MagicMock(
    side_effect=lambda src1, alpha, src2, beta, gamma, **kwargs: (
        (src1 * alpha + src2 * beta + gamma).astype(src1.dtype) if hasattr(src1, "dtype") else src1
    )
)
_cv2_mod.findContours = MagicMock(return_value=([], None))


def _mock_connected_components(*args, **kwargs):
    src = args[0]
    if not hasattr(src, "shape"):
        return 2, np.zeros((10, 10), dtype=np.int32), np.zeros((2, 5)), np.zeros((2, 2))

    labels = np.zeros(src.shape, dtype=np.int32)
    if src.shape == (100, 100):
        labels[10:20, 10:20] = 2
        labels[50:60, 50:70] = 1
        stats = np.array([[0, 0, 100, 100, 10000 - 300], [50, 50, 20, 10, 200], [10, 10, 10, 10, 100]])
        return 3, labels, stats, np.zeros((3, 2))

    labels[src > 0] = 1
    stats = np.array([[0, 0, src.shape[1], src.shape[0], src.size], [0, 0, 10, 10, 100]])
    return 2, labels, stats, np.zeros((2, 2))


_cv2_mod.connectedComponentsWithStats = MagicMock(side_effect=_mock_connected_components)

# Define ALL modules to mock
modules_to_mock = {
    "psutil": _create_mock_module(
        "psutil",
        {
            "virtual_memory": MagicMock(return_value=make_svmem()),
            "cpu_percent": MagicMock(return_value=10.0),
            "process_iter": MagicMock(return_value=[]),
            "cpu_count": MagicMock(return_value=4),
        },
    ),
    "torch": _torch_mod,
    "torch.nn": _nn_mod,
    "torch.multiprocessing": _mp_mod,
    "torch.cuda": _cuda_mod,
    "torch.cuda.amp": _torch_mod.cuda.amp,
    "torch.version": _torch_mod.version,
    "torch.linalg": _torch_mod.linalg,
    "torch.distributed": _create_mock_module("torch.distributed"),
    "torch.hub": _create_mock_module(
        "torch.hub",
        {
            "_get_torch_home": MagicMock(return_value="/tmp"),
            "download_url_to_file": MagicMock(),
            "get_dir": MagicMock(return_value="/tmp"),
            "load_state_dict_from_url": MagicMock(return_value={}),
        },
    ),
    "torch.utils": _create_mock_module("torch.utils", {"__path__": []}),
    "torch.utils.model_zoo": _create_mock_module("torch.utils.model_zoo"),
    "torchvision": _create_mock_module("torchvision", {"__path__": []}),
    "torchvision.transforms": _create_mock_module(
        "torchvision.transforms",
        {
            "__path__": [],
            "Compose": MagicMock(return_value=lambda x: _create_mock_tensor("v1_tensor", value=1.0)),
            "ToTensor": MagicMock(return_value=lambda x: _create_mock_tensor("v1_tensor", value=1.0)),
            "Normalize": MagicMock(return_value=lambda x: _create_mock_tensor("v1_tensor", value=1.0)),
            "functional": _create_mock_module("torchvision.transforms.functional"),
        },
    ),
    "torchvision.transforms.functional": _create_mock_module("torchvision.transforms.functional"),
    "torchvision.utils": _create_mock_module("torchvision.utils", {"make_grid": MagicMock()}),
    "torchvision.transforms.v2": _create_mock_module(
        "torchvision.transforms.v2",
        {
            "Compose": MagicMock(return_value=lambda x: x),
            "ToDtype": MagicMock(return_value=lambda x: x),
            "Resize": MagicMock(return_value=lambda x: x),
            "Normalize": MagicMock(return_value=lambda x: x),
            "functional": _create_mock_module(
                "torchvision.transforms.v2.functional",
                {
                    "to_image": MagicMock(side_effect=lambda x: x),
                },
            ),
        },
    ),
    "torchvision.ops": MagicMock(),
    "cv2": _cv2_mod,
    "lpips": _create_mock_module("lpips", {"LPIPS": MagicMock()}),
    "ultralytics": _create_mock_module("ultralytics", {"YOLO": MagicMock()}),
    "ultralytics.models.yolo": _create_mock_module("ultralytics.models.yolo", {"YOLO": MagicMock()}),
    "insightface": _create_mock_module("insightface", {"app": MagicMock()}),
    "insightface.app": _create_mock_module("insightface.app", {"FaceAnalysis": MagicMock()}),
    "scenedetect": _create_mock_module(
        "scenedetect",
        {
            "__path__": [],
            "detect": MagicMock(return_value=[]),
            "ContentDetector": MagicMock(),
            "VideoManager": MagicMock(),
            "SceneManager": MagicMock(),
            "VideoOpenFailure": VideoOpenFailure,
        },
    ),
    "scenedetect.detectors": _create_mock_module("scenedetect.detectors", {"ContentDetector": MagicMock()}),
    "scenedetect.video_stream": _create_mock_module("scenedetect.video_stream", {"VideoOpenFailure": VideoOpenFailure}),
    "sam2": _create_mock_module("sam2", {"__path__": []}),
    "sam2.build_sam": _create_mock_module(
        "sam2.build_sam",
        {
            "build_sam2_video_predictor": MagicMock(
                return_value=MagicMock(
                    init_state=MagicMock(return_value="state"),
                    add_new_points_or_box=MagicMock(
                        return_value=(None, None, [_create_mock_tensor(shape=(1, 1, 100, 100))])
                    ),
                    propagate_in_video=MagicMock(return_value=[(0, [1], _create_mock_tensor(shape=(1, 1, 100, 100)))]),
                )
            )
        },
    ),
    "sam3": _create_mock_module("sam3", {"__path__": []}),
    "sam3.model_builder": _create_mock_module(
        "sam3.model_builder",
        {
            "build_sam3_predictor": MagicMock(),
            "build_sam3_video_model": MagicMock(),
        },
    ),
}

# Apply immediately
if not _should_skip_mocks():
    for name in sorted(modules_to_mock.keys()):
        sys.modules[name] = modules_to_mock[name]

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
