import os

import numpy as np
import onnxruntime as ort
import pytest
import torch


def test_torch_cuda_health():
    """Verify Torch can talk to the GPU."""
    assert torch.cuda.is_available(), "Torch reports CUDA is NOT available."
    device_count = torch.cuda.device_count()
    assert device_count > 0, "No CUDA devices found by Torch."

    # Simple operation
    t = torch.tensor([1.0, 2.0]).cuda()
    assert t.is_cuda
    assert t.sum().item() == 3.0


def test_onnx_cuda_health():
    """Verify ONNXRuntime can actually load CUDA providers and their shared libraries."""
    providers = ort.get_available_providers()
    assert "CUDAExecutionProvider" in providers, (
        f"CUDAExecutionProvider not in available providers: {providers}. "
        "Check if onnxruntime-gpu is installed correctly."
    )

    # Try preloading if available
    if hasattr(ort, "preload_dlls"):
        try:
            ort.preload_dlls()
        except Exception as e:
            print(f"preload_dlls() failed: {e}")

    # Use the real model from the project to verify it loads on GPU
    model_path = "models/yolo12l-person-seg-extended.onnx"
    if not os.path.exists(model_path):
        pytest.skip(f"Model {model_path} not found. Skip real model load test.")

    try:
        session = ort.InferenceSession(model_path, providers=["CUDAExecutionProvider"])
        assert "CUDAExecutionProvider" in session.get_providers()

        # Run a tiny dummy inference (zeros)
        input_info = session.get_inputs()[0]
        input_name = input_info.name
        input_shape = input_info.shape

        # Input shape for YOLO is typically [1, 3, H, W]
        # Some dimensions might be strings (dynamic), replace them with 640 or 1
        processed_shape = []
        for s in input_shape:
            if isinstance(s, str):
                processed_shape.append(640 if s in ("H", "W", "height", "width") else 1)
            else:
                processed_shape.append(s)

        dummy_input = np.zeros(processed_shape, dtype=np.float32)
        session.run(None, {input_name: dummy_input})

    except Exception as e:
        pytest.fail(
            f"ONNX CUDA session initialization failed for {model_path}: {e}. "
            "This usually means a shared library (like libcublasLt.so.12) is missing."
        )


def test_niqe_cuda_health():
    """Verify pyiqa NIQE metric can run on GPU."""
    try:
        import pyiqa

        device = "cuda" if torch.cuda.is_available() else "cpu"
        niqe = pyiqa.create_metric("niqe", device=device)

        # Test input
        img = torch.rand(1, 3, 224, 224).to(device)
        score = niqe(img)
        assert score is not None
        assert isinstance(float(score), float)
    except ImportError:
        pytest.skip("pyiqa not installed")
    except Exception as e:
        pytest.fail(f"NIQE GPU inference failed: {e}")


def test_gpu_coexistence():
    """Verify Torch (NIQE) and ONNX (YOLO) can run in the same process/thread."""
    try:
        import pyiqa

        device = "cuda" if torch.cuda.is_available() else "cpu"
        niqe = pyiqa.create_metric("niqe", device=device)

        model_path = "models/yolo12l-person-seg-extended.onnx"
        if not os.path.exists(model_path):
            pytest.skip("YOLO model not found")

        session = ort.InferenceSession(model_path, providers=["CUDAExecutionProvider"])

        # 1. Run NIQE
        img_torch = torch.rand(1, 3, 224, 224).to(device)

        niqe(img_torch)

        # 2. Run ONNX
        input_name = session.get_inputs()[0].name
        # Simple input
        img_onnx = np.zeros((1, 3, 640, 640), dtype=np.float32)
        session.run(None, {input_name: img_onnx})

        assert True  # Reached here without hanging
    except ImportError:
        pytest.skip("pyiqa not installed")
    except Exception as e:
        pytest.fail(f"GPU coexistence test failed: {e}")
