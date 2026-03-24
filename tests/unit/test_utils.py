from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from pydantic import BaseModel

from core.models import AnalysisParameters, Scene
from core.utils import _to_json_safe, estimate_totals, handle_common_errors, monitor_memory_usage, safe_resource_cleanup


def test_handle_common_errors_success():
    @handle_common_errors
    def success_func():
        return "ok"

    assert success_func() == "ok"


def test_handle_common_errors_file_not_found():
    @handle_common_errors
    def fail_func():
        raise FileNotFoundError("missing")

    res = fail_func()
    assert res["done"] is False
    assert "File not found" in res["status_message"]


def test_handle_common_errors_value_error():
    @handle_common_errors
    def fail_func():
        raise ValueError("bad value")

    res = fail_func()
    assert res["done"] is False
    assert "Invalid input" in res["status_message"]


def test_handle_common_errors_cuda_oom():
    @handle_common_errors
    def fail_func():
        raise RuntimeError("CUDA out of memory")

    res = fail_func()
    assert res["done"] is False
    assert "GPU memory error" in res["status_message"]


def test_handle_common_errors_generator():
    @handle_common_errors
    def gen_func():
        yield 1
        raise ValueError("fail")

    it = gen_func()
    assert next(it) == 1
    res = next(it)
    assert res["done"] is False
    assert "Invalid input" in res["status_message"]


def test_monitor_memory_usage():
    logger = MagicMock()
    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.memory_allocated", return_value=9000 * 1024**2),
        patch("torch.cuda.empty_cache") as mock_empty,
    ):
        monitor_memory_usage(logger, "cuda", threshold_mb=8000)
        assert logger.warning.called
        assert mock_empty.called


def test_estimate_totals():
    params = AnalysisParameters(method="every_nth_frame", nth_frame=2)
    video_info = {"frame_count": 100}
    scenes = [Scene(shot_id=1, start_frame=0, end_frame=10), Scene(shot_id=2, start_frame=20, end_frame=30)]

    totals = estimate_totals(params, video_info, scenes)
    assert totals["extraction"] == 50
    assert totals["pre_analysis"] == 2
    assert totals["propagation"] == (11 + 11)


def test_estimate_totals_keyframes():
    params = AnalysisParameters(method="keyframes")
    video_info = {"frame_count": 1000}
    totals = estimate_totals(params, video_info, None)
    assert totals["extraction"] == 150  # 15%


class MockModel(BaseModel):
    name: str


def test_to_json_safe():
    data = {
        "int": np.int64(1),
        "float": np.float32(1.5),
        "bool": np.bool_(True),
        "arr": np.array([1, 2, 3]),
        "path": Path("/tmp/test"),
        "model": MockModel(name="test"),
        "nested": [{"a": 1}],
    }
    safe = _to_json_safe(data)
    assert isinstance(safe["int"], int)
    assert isinstance(safe["float"], float)
    assert isinstance(safe["bool"], bool)
    assert isinstance(safe["arr"], list)
    assert isinstance(safe["path"], str)
    assert isinstance(safe["model"], dict)
    assert safe["nested"][0]["a"] == 1


def test_safe_resource_cleanup():
    with (
        patch("gc.collect") as mock_gc,
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.empty_cache") as mock_empty,
    ):
        with safe_resource_cleanup(device="cuda"):
            pass
        assert mock_gc.called
        assert mock_empty.called


def test_monitor_memory_usage_low():
    logger = MagicMock()
    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.memory_allocated", return_value=1000 * 1024**2),
        patch("torch.cuda.empty_cache") as mock_empty,
    ):
        monitor_memory_usage(logger, "cuda", threshold_mb=8000)
        assert not logger.warning.called
        assert not mock_empty.called


def test_handle_common_errors_gen_exceptions():
    @handle_common_errors
    def gen_fail(exc_type):
        yield 1
        raise exc_type("fail")

    # Test FileNotFoundError
    it = gen_fail(FileNotFoundError)
    assert next(it) == 1
    res = next(it)
    assert "File not found" in res["status_message"]

    # Test RuntimeError (non-CUDA)
    it = gen_fail(RuntimeError)
    assert next(it) == 1
    res = next(it)
    assert "Processing error" in res["status_message"]

    # Test Generic Exception
    it = gen_fail(Exception)
    assert next(it) == 1
    res = next(it)
    assert "Critical error" in res["status_message"]


def test_handle_common_errors_non_gen_exceptions():
    @handle_common_errors
    def fail_func(exc_type):
        raise exc_type("fail")

    assert "Runtime error" in fail_func(RuntimeError)["log"]
    assert "Unexpected error" in fail_func(Exception)["log"]


def test_estimate_totals_default():
    params = AnalysisParameters(method="unknown")
    video_info = {"frame_count": 100}
    totals = estimate_totals(params, video_info, None)
    assert totals["extraction"] == 100


def test_estimate_totals_all():
    params = AnalysisParameters(method="all")
    video_info = {"frame_count": 100}
    totals = estimate_totals(params, video_info, None)
    assert totals["extraction"] == 100


def test_safe_resource_cleanup_no_cuda():
    with (
        patch("gc.collect") as mock_gc,
        patch("torch.cuda.is_available", return_value=False),
        patch("torch.cuda.empty_cache") as mock_empty,
    ):
        with safe_resource_cleanup(device="cuda"):
            pass
        assert mock_gc.called
        assert not mock_empty.called
