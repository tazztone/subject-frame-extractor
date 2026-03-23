from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

# Mock sam3 before importing SAM3Wrapper
mock_sam3 = MagicMock()
import sys

sys.modules["sam3"] = mock_sam3
sys.modules["sam3.model_builder"] = mock_sam3.model_builder

from core.managers.sam3 import SAM3Wrapper


@pytest.fixture
def mock_predictor():
    predictor = MagicMock()

    # Mock handle_request to return session_id
    def handle_request(*args, **kwargs):
        # Handle both positional and keyword 'request'
        request = args[0] if args else kwargs.get("request")

        if request.get("type") == "start_session":
            return {"session_id": "test_session"}
        elif request.get("type") == "add_prompt":
            return {"outputs": {"out_binary_masks": torch.zeros((1, 100, 100)), "out_obj_ids": [1]}}
        elif request.get("type") == "detect_objects":
            return {"outputs": [{"bbox": [0, 0, 10, 10], "label": "person"}]}
        elif request.get("type") == "close_session":
            return {"status": "ok"}
        elif request.get("type") == "remove_object":
            return {"status": "ok"}
        elif request.get("type") == "clear_prompts":
            return {"status": "ok"}
        return {}

    predictor.handle_request.side_effect = handle_request
    predictor.handle_stream_request.return_value = [
        {"frame_index": 0, "outputs": {"out_binary_masks": torch.zeros((1, 100, 100)), "out_obj_ids": [1]}}
    ]
    return predictor


@patch("sam3.model_builder.build_sam3_video_predictor")
@patch("core.managers.sam3.torch.cuda.is_available", return_value=False)
def test_sam3_wrapper_init(mock_cuda, mock_build, mock_predictor):
    mock_build.return_value = mock_predictor
    wrapper = SAM3Wrapper(device="cpu")
    assert wrapper.device == "cpu"
    assert mock_build.called


@patch("sam3.model_builder.build_sam3_video_predictor")
def test_sam3_wrapper_session_lifecycle(mock_build, mock_predictor):
    mock_build.return_value = mock_predictor
    wrapper = SAM3Wrapper()

    session_id = wrapper.init_video("video.mp4")
    assert session_id == "test_session"
    assert wrapper.session_id == "test_session"

    wrapper.close_session()
    assert wrapper.session_id is None
    assert mock_predictor.handle_request.called
    # Check that it was called with type=close_session
    called_with_close = False
    for call in mock_predictor.handle_request.call_args_list:
        req = call[0][0] if call[0] else call[1].get("request")
        if req and req.get("type") == "close_session":
            called_with_close = True
    assert called_with_close


@patch("sam3.model_builder.build_sam3_video_predictor")
def test_sam3_wrapper_add_bbox_prompt(mock_build, mock_predictor):
    mock_build.return_value = mock_predictor
    wrapper = SAM3Wrapper()
    wrapper.session_id = "test_session"

    mask = wrapper.add_bbox_prompt(frame_idx=0, obj_id=1, bbox_xywh=[10, 10, 50, 50], img_size=(100, 100))

    assert isinstance(mask, np.ndarray)
    assert mask.shape == (100, 100)
    assert mock_predictor.handle_request.called


@patch("sam3.model_builder.build_sam3_video_predictor")
def test_sam3_wrapper_propagate(mock_build, mock_predictor):
    mock_build.return_value = mock_predictor
    wrapper = SAM3Wrapper()
    wrapper.session_id = "test_session"

    results = list(wrapper.propagate(start_idx=0, max_frames=1))

    assert len(results) == 1
    frame_idx, oid, mask = results[0]
    assert frame_idx == 0
    assert oid == 1
    assert mask.shape == (100, 100)
    assert mock_predictor.handle_stream_request.called


@patch("sam3.model_builder.build_sam3_video_predictor")
def test_sam3_wrapper_detect_objects(mock_build, mock_predictor):
    mock_build.return_value = mock_predictor
    wrapper = SAM3Wrapper()

    results = wrapper.detect_objects(np.zeros((100, 100, 3), dtype=np.uint8), "person")

    assert len(results) == 1
    assert results[0]["label"] == "person"


@patch("sam3.model_builder.build_sam3_video_predictor")
def test_sam3_wrapper_utility_methods(mock_build, mock_predictor):
    mock_build.return_value = mock_predictor
    wrapper = SAM3Wrapper()
    wrapper.session_id = "test_session"

    wrapper.add_text_prompt(0, "test")
    wrapper.add_point_prompt(0, 1, [[10, 10]], [1], (100, 100))
    wrapper.remove_object(1)
    wrapper.clear_prompts()
    wrapper.reset_session()

    assert mock_predictor.handle_request.called


def test_triton_mocking():
    import sys

    from core.managers.sam3 import _setup_triton_mock

    # Temporarily remove triton from sys.modules
    if "triton" in sys.modules:
        del sys.modules["triton"]

    # Directly test setup_triton_mock logic by ensuring it populates sys.modules
    _setup_triton_mock()
    assert "triton" in sys.modules
    assert hasattr(sys.modules["triton"], "jit")
    # Just check it's a mock or original
    assert sys.modules["triton"].jit is not None
