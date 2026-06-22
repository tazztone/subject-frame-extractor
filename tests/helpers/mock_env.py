"""Single source of truth for sys.modules mocks of heavy ML/CV libraries.

Called at collection time by conftest.py (unit tests) and mock_app.py (UI test server).
Every heavy third-party package that causes import crashes in test environments
(PortAudio, CUDA, native libs) is mocked here once. No other file should define
sys.modules mocks for these packages.
"""

import sys
import types
from collections import namedtuple
from unittest.mock import MagicMock

import numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_mock_module(name, attributes=None):
    mock_mod = types.ModuleType(name)
    mock_mod.__package__ = name.rpartition(".")[0]
    mock_mod.__path__ = []
    if attributes:
        for attr, val in attributes.items():
            setattr(mock_mod, attr, val)
    return mock_mod


# ---------------------------------------------------------------------------
# Stable exception classes (importable by tests for assertions)
# ---------------------------------------------------------------------------


class OutOfMemoryError(RuntimeError):
    pass


class VideoOpenFailure(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# No-op context manager replacing torch.no_grad / torch.inference_mode
# ---------------------------------------------------------------------------


class TransparentContext:
    def __enter__(self, *args, **kwargs):
        return self

    def __exit__(self, *args, **kwargs):
        pass

    def __call__(self, func=None):
        return func if func is not None else self


# ---------------------------------------------------------------------------
# CUDA availability toggle
# ---------------------------------------------------------------------------

_cuda_is_available_mock = MagicMock(return_value=False)


def set_cuda_available(available: bool):
    _cuda_is_available_mock.return_value = available


# ---------------------------------------------------------------------------
# psutil memory helpers
# ---------------------------------------------------------------------------

svmem = namedtuple("svmem", ["total", "available", "percent", "used", "free"])


def make_svmem(available_mb=4096, percent=50.0, total_gb=8):
    total = total_gb * 1024**3
    avail = available_mb * 1024 * 1024
    return svmem(total, avail, percent, total - avail, avail)


# ---------------------------------------------------------------------------
# Mock tensor factory
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Deep mock modules (behavioural stubs with real numpy side-effects)
# ---------------------------------------------------------------------------

# --- torch ---
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

# --- cv2 (deep: 20+ behavioural stubs) ---
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
        h[0] = float(total)
    else:
        h[:] = float(total) / histSize[0]
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

# --- matplotlib (wired so `import matplotlib.pyplot as plt` gets our mock) ---
_matplotlib_pyplot_mod = MagicMock(name="matplotlib.pyplot")
_matplotlib_ticker_mod = MagicMock(name="matplotlib.ticker")
_matplotlib_mod = _create_mock_module(
    "matplotlib",
    {
        "__path__": [],
        "pyplot": _matplotlib_pyplot_mod,
        "ticker": _matplotlib_ticker_mod,
        "figure": MagicMock(),
        "backends": MagicMock(),
        "get_backend": MagicMock(return_value="agg"),
        "use": MagicMock(),
        "rcParams": {},
    },
)

# --- yt_dlp (wired so `from yt_dlp.utils import DownloadError` works) ---
_yt_dlp_utils_mod = _create_mock_module(
    "yt_dlp.utils",
    {"DownloadError": type("DownloadError", (Exception,), {})},
)
_yt_dlp_mod = _create_mock_module(
    "yt_dlp",
    {"__path__": [], "utils": _yt_dlp_utils_mod, "YoutubeDL": MagicMock()},
)

# --- skimage (wired so `from skimage.metrics import structural_similarity` works) ---
_skimage_metrics_mod = _create_mock_module(
    "skimage.metrics",
    {"structural_similarity": MagicMock(return_value=1.0)},
)
_skimage_mod = _create_mock_module(
    "skimage",
    {"__path__": [], "metrics": _skimage_metrics_mod},
)


# ---------------------------------------------------------------------------
# The canonical mock registry — union of all modules needed by both
# conftest.py (unit tests) and mock_app.py (UI test server).
# --- mediapipe (wired hierarchy to fix PortAudio crash) ---
# face.py does `from mediapipe.tasks.python import vision` at module level.
# For unittest.mock.patch() to reach the same object, the parent's attribute
# must point to the same mock we put in sys.modules.
_mp_vision_mod = _create_mock_module(
    "mediapipe.tasks.python.vision",
    {"FaceLandmarker": MagicMock(), "FaceLandmarkerOptions": MagicMock()},
)
_mp_tasks_python_mod = _create_mock_module(
    "mediapipe.tasks.python",
    {"vision": _mp_vision_mod, "BaseOptions": MagicMock()},
)
_mp_tasks_mod = _create_mock_module(
    "mediapipe.tasks",
    {"python": _mp_tasks_python_mod},
)
_mp_mod = _create_mock_module(
    "mediapipe",
    {"__path__": [], "tasks": _mp_tasks_mod},
)


# ---------------------------------------------------------------------------

modules_to_mock = {
    # --- psutil ---
    "psutil": _create_mock_module(
        "psutil",
        {
            "virtual_memory": MagicMock(return_value=make_svmem()),
            "cpu_percent": MagicMock(return_value=10.0),
            "process_iter": MagicMock(return_value=[]),
            "cpu_count": MagicMock(return_value=4),
        },
    ),
    # --- torch tree ---
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
    # --- torchvision tree ---
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
    # --- cv2 ---
    "cv2": _cv2_mod,
    # --- ML model libraries ---
    "lpips": _create_mock_module("lpips", {"LPIPS": MagicMock()}),
    "ultralytics": _create_mock_module("ultralytics", {"YOLO": MagicMock()}),
    "ultralytics.models.yolo": _create_mock_module("ultralytics.models.yolo", {"YOLO": MagicMock()}),
    "insightface": _create_mock_module("insightface", {"app": MagicMock()}),
    "insightface.app": _create_mock_module("insightface.app", {"FaceAnalysis": MagicMock()}),
    # --- scenedetect ---
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
    # --- SAM3 (full submodule tree) ---
    "sam3": _create_mock_module("sam3", {"__path__": []}),
    "sam3.model_builder": _create_mock_module(
        "sam3.model_builder",
        {
            "build_sam3_predictor": MagicMock(),
            "build_sam3_video_model": MagicMock(),
        },
    ),
    "sam3.model": MagicMock(),
    "sam3.utils": MagicMock(),
    "sam3.model.sam3_video_predictor": MagicMock(),
    "sam3.model.sam3_video_inference": MagicMock(),
    "sam3.model.sam3_base_predictor": MagicMock(),
    "sam3.model.sam3_multiplex_video_predictor": MagicMock(),
    "sam3.model.sam3_multiplex_tracking": MagicMock(),
    "sam3.model.sam3_multiplex_base": MagicMock(),
    # --- mediapipe (PortAudio crash fix, must be wired as a proper hierarchy) ---
    "mediapipe.tasks.python.vision": _mp_vision_mod,
    "mediapipe.tasks.python": _mp_tasks_python_mod,
    "mediapipe.tasks": _mp_tasks_mod,
    "mediapipe": _mp_mod,
    # --- Other heavy transitive dependencies (wired parent→child) ---
    "yt_dlp": _yt_dlp_mod,
    "yt_dlp.utils": _yt_dlp_utils_mod,
    "numba": _create_mock_module("numba", {"njit": lambda f=None, **kw: f if f else (lambda g: g)}),
    "onnxruntime": MagicMock(),
    "skimage": _skimage_mod,
    "skimage.metrics": _skimage_metrics_mod,
    "matplotlib": _matplotlib_mod,
    "matplotlib.pyplot": _matplotlib_pyplot_mod,
    "matplotlib.ticker": _matplotlib_ticker_mod,
    "safetensors": MagicMock(),
    "pyiqa": MagicMock(),
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def install_mocks() -> None:
    """Inject all mock modules into sys.modules. Idempotent."""
    for name in sorted(modules_to_mock):
        sys.modules[name] = modules_to_mock[name]
