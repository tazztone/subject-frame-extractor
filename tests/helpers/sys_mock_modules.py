import os
import sys
import types
from unittest.mock import MagicMock

import numpy as np

from tests.helpers.exceptions import OutOfMemoryError, VideoOpenFailure
from tests.helpers.mock_tensor import create_mock_tensor


def _mock_module(name, **attrs):
    """Create a mock module with explicit attributes. No __getattr__ magic."""
    m = types.ModuleType(name)
    m.__file__ = f"<mock {name}>"
    m.__path__ = []
    # Populate the dict with provided attributes
    m.__dict__.update(attrs)
    return m


def build_mock_modules():
    """Builds the comprehensive map of mock modules for the extraction system."""
    from collections import namedtuple

    DeviceProps = namedtuple("DeviceProps", ["total_memory"])

    cuda_mock = _mock_module(
        "torch.cuda",
        is_available=MagicMock(return_value=False),
        device_count=MagicMock(return_value=0),
        OutOfMemoryError=OutOfMemoryError,
        empty_cache=MagicMock(),
        memory_allocated=MagicMock(return_value=0),
        memory_reserved=MagicMock(return_value=0.0),
        get_device_properties=MagicMock(return_value=DeviceProps(total_memory=8 * 1024**3)),
        get_device_name=MagicMock(return_value="Mock GPU"),
    )

    modules = {
        "torch": _mock_module(
            "torch",
            __version__="2.0.0+mock",
            Tensor=MagicMock,
            tensor=MagicMock,
            from_numpy=MagicMock(side_effect=lambda arr, **k: create_mock_tensor("from_numpy", shape=arr.shape)),
            zeros=MagicMock(side_effect=lambda shape, *a, **k: create_mock_tensor("zeros", shape=shape)),
            ones=MagicMock(side_effect=lambda shape, *a, **k: create_mock_tensor("ones", shape=shape)),
            randn=MagicMock(side_effect=lambda shape, *a, **k: create_mock_tensor("randn", shape=shape)),
            cat=MagicMock(side_effect=lambda tensors, *a, **k: create_mock_tensor("cat", shape=tensors[0].shape)),
            stack=MagicMock(
                side_effect=lambda tensors, *a, **k: create_mock_tensor(
                    "stack", shape=(len(tensors),) + tensors[0].shape
                )
            ),
            device=MagicMock,
            no_grad=MagicMock,
            inference_mode=MagicMock,
            autocast=MagicMock,
            amp=MagicMock,
            jit=_mock_module("torch.jit", script=MagicMock(side_effect=lambda f: f)),
            nn=_mock_module(
                "torch.nn",
                Module=MagicMock,
                Parameter=MagicMock,
                GELU=MagicMock,
                Linear=MagicMock,
                Conv2d=MagicMock,
                LayerNorm=MagicMock,
                Dropout=MagicMock,
                Embedding=MagicMock,
                functional=MagicMock(),
                Sequential=MagicMock,
            ),
            version=_mock_module("torch.version", cuda="11.8"),
            cuda=cuda_mock,
            float=MagicMock(),
            float32=MagicMock(),
            float16=MagicMock(),
            bfloat16=MagicMock(),
            uint8=MagicMock(),
            int64=MagicMock(),
            int32=MagicMock(),
            bool=MagicMock(),
            long=MagicMock(),
            int=MagicMock(),
        ),
        "torch.cuda": cuda_mock,
        "torchvision": _mock_module("torchvision", transforms=MagicMock(), ops=MagicMock()),
        "cv2": _mock_module(
            "cv2",
            imread=MagicMock(return_value=np.zeros((100, 100, 3), dtype=np.uint8)),
            cvtColor=MagicMock(side_effect=lambda x, *a, **k: x),
            resize=MagicMock(
                side_effect=lambda x, size, **k: (
                    np.zeros((size[1], size[0]) + x.shape[2:], dtype=getattr(x, "dtype", np.uint8))
                    if hasattr(x, "shape") and len(x.shape) > 2
                    else np.zeros((size[1], size[0]), dtype=getattr(x, "dtype", np.uint8))
                )
            ),
            addWeighted=MagicMock(
                side_effect=lambda src1, a1, src2, a2, g, **k: (
                    src1.astype(float) * a1 + src2.astype(float) * a2 + g
                ).astype(np.uint8)
            ),
            morphologyEx=MagicMock(side_effect=lambda x, op, k, **kwargs: x),
            getStructuringElement=MagicMock(return_value=np.ones((3, 3), dtype=np.uint8)),
            rectangle=MagicMock(
                side_effect=lambda img, pt1, pt2, color, *a, **k: img.__setitem__(
                    (slice(max(0, pt1[1]), max(0, pt2[1])), slice(max(0, pt1[0]), max(0, pt2[0]))),
                    (max(color) if isinstance(color, (tuple, list, np.ndarray)) and len(color) > 0 else 255),
                )
            ),
            putText=MagicMock(),
            Sobel=MagicMock(side_effect=lambda x, *a, **k: x),
            Laplacian=MagicMock(side_effect=lambda x, *a, **k: x),
            calcHist=MagicMock(
                side_effect=lambda images, *a, **k: (
                    np.histogram(images[0], bins=256, range=(0, 256))[0].astype(np.float32).reshape(-1, 1)
                )
            ),
            connectedComponentsWithStats=MagicMock(
                side_effect=lambda mask, **k: (
                    2,
                    (mask > 0).astype(np.int32),
                    np.array([[0, 0, 0, 0, 0], [0, 0, mask.shape[1], mask.shape[0], np.sum(mask > 0)]], dtype=np.int32),
                    np.array([[0, 0], [mask.shape[1] // 2, mask.shape[0] // 2]], dtype=np.float32),
                )
            ),
            fundametalMat=MagicMock(),
            findFundamentalMat=MagicMock(return_value=(None, None)),
            imwrite=MagicMock(return_value=True),
            VideoCapture=MagicMock(),
            VideoWriter=MagicMock(),
            VideoWriter_fourcc=MagicMock(return_value=0),
            findContours=MagicMock(return_value=([], None)),
            getTextSize=MagicMock(return_value=((50, 20), 5)),
            IMREAD_UNCHANGED=0,
            IMREAD_COLOR=1,
            IMREAD_GRAYSCALE=2,
            COLOR_RGB2BGR=4,
            COLOR_BGR2RGB=4,
            COLOR_RGB2GRAY=6,
            COLOR_BGR2GRAY=6,
            INTER_AREA=3,
            INTER_LINEAR=1,
            INTER_NEAREST=0,
            CV_64F=6,
            CV_32F=5,
            CV_8U=0,
            MORPH_CLOSE=1,
            MORPH_ELLIPSE=1,
            CC_STAT_AREA=4,
            CAP_PROP_FPS=5,
            CAP_PROP_FRAME_COUNT=7,
            CAP_PROP_FRAME_HEIGHT=4,
            CAP_PROP_FRAME_WIDTH=3,
            CAP_PROP_POS_FRAMES=1,
            CHAIN_APPROX_SIMPLE=2,
            DIST_L2=2,
            FONT_HERSHEY_SIMPLEX=0,
            RETR_EXTERNAL=0,
            circle=MagicMock(),
            boundingRect=MagicMock(return_value=(0, 0, 10, 10)),
            distanceTransform=MagicMock(side_effect=lambda x, *a, **k: x),
        ),
        "sam2": _mock_module("sam2", build_sam2_video_predictor=MagicMock()),
        "sam2.build_sam": _mock_module("sam2.build_sam", build_sam2_video_predictor=MagicMock()),
        "sam3": _mock_module("sam3", build_sam3_video_model=MagicMock(), build_sam3_predictor=MagicMock()),
        "sam3.model_builder": _mock_module(
            "sam3.model_builder", build_sam3_predictor=MagicMock(), build_sam3_video_model=MagicMock()
        ),
        "scenedetect": _mock_module(
            "scenedetect",
            ContentDetector=MagicMock(),
            VideoOpenFailure=VideoOpenFailure,
            detect=MagicMock(),
        ),
        "scenedetect.detectors": _mock_module("scenedetect.detectors", ContentDetector=MagicMock()),
        "insightface": _mock_module("insightface", app=MagicMock()),
        "insightface.app": _mock_module("insightface.app", FaceAnalysis=MagicMock()),
        "pyiqa": _mock_module("pyiqa", create_metric=MagicMock()),
        "lpips": _mock_module("lpips", LPIPS=MagicMock()),
        "onnxruntime": _mock_module(
            "onnxruntime",
            InferenceSession=MagicMock(),
            SessionOptions=MagicMock(),
            GraphOptimizationLevel=MagicMock(),
        ),
    }
    return modules


_mocks_injected = False
_original_sys_modules = {}


def inject_mocks_into_sys():
    """Manually injects mocks into sys.modules. Used by mock_app.py and pytest_configure."""
    global _mocks_injected
    if _mocks_injected:
        return

    if os.environ.get("PYTEST_INTEGRATION_MODE", "").lower() == "true":
        return

    mocks = build_mock_modules()

    for name in mocks.keys():
        _original_sys_modules[name] = sys.modules.get(name)

    for name, b_mod in mocks.items():
        sys.modules[name] = b_mod

    # Structural linkage
    for name in sorted(mocks):
        if "." in name:
            parent_name, child_name = name.rsplit(".", 1)
            if parent_name in sys.modules:
                setattr(sys.modules[parent_name], child_name, sys.modules[name])

    # Force torch.cuda sync
    torch_mod = sys.modules.get("torch")
    cuda_mod = sys.modules.get("torch.cuda")
    if torch_mod is not None and cuda_mod is not None:
        setattr(torch_mod, "cuda", cuda_mod)

    _mocks_injected = True


def remove_mocks_from_sys():
    """Removes injected mocks from sys.modules to restore original state."""
    global _mocks_injected, _original_sys_modules
    if not _mocks_injected:
        return

    for name, orig_mod in _original_sys_modules.items():
        if orig_mod is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = orig_mod

    # Also remove structural linkage injected
    mocks = build_mock_modules()
    for name in sorted(mocks):
        if "." in name:
            parent_name, child_name = name.rsplit(".", 1)
            # We don't definitively revert child attributes on parents unless
            # they were dynamically created, but we can attempt basic cleanup if parent exists.
            if parent_name in sys.modules and hasattr(sys.modules[parent_name], child_name):
                # We skip deep setattr rollback to keep it safe and focus on module un-poisoning.
                pass

    _original_sys_modules.clear()
    _mocks_injected = False
