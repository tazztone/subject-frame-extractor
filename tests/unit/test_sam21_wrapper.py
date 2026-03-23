"""
Unit tests for SAM21Wrapper API completeness and functionality.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

# We will patch the build function directly where it is used in SAM21Wrapper
from core.managers.sam21 import SAM21Wrapper


@pytest.fixture
def mock_predictor():
    predictor = MagicMock()
    # Mock return values for predictor methods
    predictor.init_state.return_value = "test_state"
    predictor.add_new_points_or_box.return_value = (None, None, torch.zeros((1, 1, 100, 100)))
    predictor.propagate_in_video.return_value = [
        (0, [1], torch.zeros((1, 1, 100, 100))),
        (1, [1], torch.zeros((1, 1, 100, 100))),
    ]
    return predictor


@patch("core.managers.sam21.build_sam2_video_predictor")
@patch("core.managers.sam21.torch.cuda.is_available", return_value=False)
def test_sam21_wrapper_init(mock_cuda, mock_build, mock_predictor):
    mock_build.return_value = mock_predictor

    wrapper = SAM21Wrapper(checkpoint_path="dummy.pt", device="cpu")
    assert wrapper.device == "cpu"
    assert mock_build.called
    assert wrapper.predictor == mock_predictor


@patch("core.managers.sam21.build_sam2_video_predictor")
def test_sam21_wrapper_session_lifecycle(mock_build, mock_predictor):
    mock_build.return_value = mock_predictor
    wrapper = SAM21Wrapper(checkpoint_path="dummy.pt")

    # First init
    session_id = wrapper.init_video("video_dir")
    assert session_id == id("test_state")
    assert wrapper._state == "test_state"
    assert mock_predictor.init_state.called

    # Re-init (should call close_session)
    with patch.object(wrapper, "close_session", wraps=wrapper.close_session) as mock_close:
        wrapper.init_video("new_video_dir")
        assert mock_close.called

    wrapper.close_session()
    assert wrapper._state is None
    assert mock_predictor.reset_state.called


@patch("core.managers.sam21.build_sam2_video_predictor")
def test_sam21_wrapper_add_bbox_prompt(mock_build, mock_predictor):
    mock_build.return_value = mock_predictor
    wrapper = SAM21Wrapper(checkpoint_path="dummy.pt")
    wrapper._state = "test_state"

    # Test with 3D mask output (ndim == 3)
    mock_predictor.add_new_points_or_box.return_value = (None, None, torch.zeros((1, 1, 100, 100)))
    mask = wrapper.add_bbox_prompt(frame_idx=0, obj_id=1, bbox_xywh=[10, 10, 50, 50], img_size=(100, 100))
    assert mask.shape == (100, 100)

    # Test with 2D mask output (ndim == 2) - though SAM2 usually returns 3D or 4D
    mock_predictor.add_new_points_or_box.return_value = (None, None, torch.zeros((1, 100, 100)))
    mask = wrapper.add_bbox_prompt(frame_idx=0, obj_id=1, bbox_xywh=[10, 10, 50, 50], img_size=(100, 100))
    assert mask.shape == (100, 100)

    assert mock_predictor.add_new_points_or_box.called
    # Check that box was converted to xyxy correctly
    args, kwargs = mock_predictor.add_new_points_or_box.call_args
    assert np.array_equal(kwargs["box"], np.array([10, 10, 60, 60], dtype=np.float32))


@patch("core.managers.sam21.build_sam2_video_predictor")
def test_sam21_wrapper_propagate(mock_build, mock_predictor):
    mock_build.return_value = mock_predictor
    wrapper = SAM21Wrapper(checkpoint_path="dummy.pt")
    wrapper._state = "test_state"

    # Mock 3D masks in propagate
    mock_predictor.propagate_in_video.return_value = [
        (0, [1], torch.zeros((1, 1, 100, 100))),
    ]
    results = list(wrapper.propagate(start_idx=0, max_frames=None, direction="forward"))
    assert len(results) == 1
    assert results[0][2].shape == (100, 100)

    # Check reverse parameter for backward direction
    list(wrapper.propagate(start_idx=5, direction="backward"))
    args, kwargs = mock_predictor.propagate_in_video.call_args
    assert kwargs["reverse"] is True


@patch("core.managers.sam21.build_sam2_video_predictor")
def test_sam21_wrapper_add_point_prompt(mock_build, mock_predictor):
    mock_build.return_value = mock_predictor
    wrapper = SAM21Wrapper(checkpoint_path="dummy.pt")
    wrapper._state = "test_state"

    mock_predictor.add_new_points_or_box.return_value = (None, None, torch.zeros((1, 1, 100, 100)))
    mask = wrapper.add_point_prompt(0, 1, [[50, 50]], [1], (100, 100))

    assert mask.shape == (100, 100)
    assert mock_predictor.add_new_points_or_box.called


@patch("core.managers.sam21.build_sam2_video_predictor")
def test_sam21_wrapper_stubs_and_utility(mock_build, mock_predictor):
    mock_build.return_value = mock_predictor
    wrapper = SAM21Wrapper(checkpoint_path="dummy.pt")
    wrapper._state = "test_state"

    # Test stubs
    assert wrapper.detect_objects(np.zeros((10, 10, 3)), "person") == []
    wrapper.remove_object(1)  # Should not raise

    with pytest.raises(NotImplementedError):
        wrapper.add_text_prompt(0, "text")

    # Test utility
    wrapper.clear_prompts()
    assert mock_predictor.reset_state.called

    wrapper.reset_session()
    assert wrapper._state is None


@patch("core.managers.sam21.build_sam2_video_predictor")
@patch("core.managers.sam21.torch.cuda.is_available", return_value=True)
@patch("core.managers.sam21.torch.cuda.empty_cache")
def test_sam21_wrapper_shutdown(mock_empty_cache, mock_cuda, mock_build, mock_predictor):
    mock_build.return_value = mock_predictor
    wrapper = SAM21Wrapper(checkpoint_path="dummy.pt")
    wrapper._state = "test_state"

    wrapper.shutdown()

    assert wrapper._state is None
    assert wrapper.predictor is None
    assert mock_empty_cache.called
