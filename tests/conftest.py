import os
import sys
import types
from collections import namedtuple
from unittest.mock import MagicMock

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
    mock_t.__getitem__ = MagicMock(side_effect=lambda x: mock_t)
    mock_t.__iter__ = MagicMock(side_effect=lambda: iter([mock_t]))
    mock_t.__len__ = MagicMock(return_value=shape[0] if shape else 1)
    mock_t.__gt__ = MagicMock(return_value=mock_t)
    mock_t.__lt__ = MagicMock(return_value=mock_t)
    mock_t.__ge__ = MagicMock(return_value=mock_t)
    mock_t.__le__ = MagicMock(return_value=mock_t)
    mock_t.__bool__ = MagicMock(return_value=True)
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
        "get_device_name": MagicMock(return_value="Mock GPU"),
        "empty_cache": MagicMock(),
        "synchronize": MagicMock(),
        "current_device": MagicMock(return_value=0),
        "OutOfMemoryError": OutOfMemoryError,
        "amp": _create_mock_module("torch.cuda.amp", {"autocast": TransparentContext}),
    },
)

_torch_mod = _create_mock_module(
    "torch",
    {
        "__path__": [],
        "cuda": _cuda_mod,
        "no_grad": TransparentContext,
        "inference_mode": TransparentContext,
        "from_numpy": MagicMock(side_effect=lambda x: _create_mock_tensor("from_numpy", x.shape)),
        "zeros": MagicMock(side_effect=lambda shape, **kwargs: _create_mock_tensor("zeros", shape)),
        "ones": MagicMock(side_effect=lambda shape, **kwargs: _create_mock_tensor("ones", shape)),
        "tensor": MagicMock(
            side_effect=lambda data, **kwargs: _create_mock_tensor("tensor", getattr(data, "shape", None), data)
        ),
        "float32": MagicMock(),
        "device": MagicMock(),
        "jit": _create_mock_module("torch.jit", {"script": lambda x: x}),
        "Tensor": MagicMock,
    },
)

# Define ALL modules to mock
modules_to_mock = {
    "psutil": _create_mock_module(
        "psutil",
        {
            "virtual_memory": MagicMock(return_value=make_svmem()),
            "cpu_percent": MagicMock(return_value=10.0),
            "process_iter": MagicMock(return_value=[]),
        },
    ),
    "torch": _torch_mod,
    "torch.cuda": _cuda_mod,
    "torch.cuda.amp": _torch_mod.cuda.amp,
    "torch.distributed": _create_mock_module("torch.distributed"),
    "torch.hub": _create_mock_module("torch.hub", {"_get_torch_home": MagicMock(return_value="/tmp")}),
    "torch.utils": _create_mock_module("torch.utils", {"__path__": []}),
    "torch.utils.model_zoo": _create_mock_module("torch.utils.model_zoo"),
    "torchvision": _create_mock_module("torchvision", {"__path__": []}),
    "torchvision.transforms": _create_mock_module("torchvision.transforms", {"__path__": []}),
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
    "cv2": _create_mock_module(
        "cv2",
        {
            "VideoCapture": MagicMock(),
            "cvtColor": MagicMock(side_effect=lambda x, *args: x),
            "COLOR_RGB2BGR": 4,
            "COLOR_BGR2RGB": 4,
            "imencode": MagicMock(return_value=(True, b"")),
        },
    ),
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
    "sam2.build_sam": _create_mock_module("sam2.build_sam", {"build_sam2_video_predictor": MagicMock()}),
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
import numpy as np
import pytest


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
def sample_scenes():
    from core.models import Scene

    return [
        Scene(shot_id=1, start_frame=0, end_frame=100),
        Scene(shot_id=2, start_frame=100, end_frame=200),
        Scene(shot_id=3, start_frame=200, end_frame=300),
        Scene(shot_id=4, start_frame=300, end_frame=400),
    ]
