"""
Unit tests for SAM2Wrapper API completeness and functionality.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

pytestmark = [pytest.mark.sam2]

# We will patch the build function directly where it is used in SAM2Wrapper
from core.managers.sam2 import SAM2Wrapper
from tests.conftest import _torch_mod


def _mock_tensor(shape: tuple) -> MagicMock:
    """Create a mock that behaves like a torch tensor through .cpu().numpy() > 0."""
    arr = np.zeros(shape, dtype=np.float32)
    t = MagicMock()
    t.cpu.return_value = t
    t.numpy.return_value = arr
    t.shape = shape
    t.__getitem__ = lambda self, idx: _mock_tensor(arr[idx].shape) if isinstance(idx, (int, slice)) else t
    return t


@pytest.fixture
def mock_predictor():
    predictor = MagicMock()
    # Mock return values for predictor methods
    predictor.init_state.return_value = "test_state"

    # Create a mock tensor that behaves like a torch tensor
    mock_mask = MagicMock()
    mock_mask.cpu.return_value = mock_mask
    # Return a real numpy array to satisfy the comparison in add_point_prompt (m > 0)
    mock_mask.numpy.return_value = np.ones((1, 1, 100, 100), dtype=np.uint8)
    mock_mask.shape = (1, 1, 100, 100)
    mock_mask.__getitem__ = MagicMock(return_value=mock_mask)

    predictor.add_new_points_or_box.return_value = (None, None, mock_mask)
    predictor.propagate_in_video.return_value = [
        (0, [1], mock_mask),
        (1, [1], mock_mask),
    ]
    return predictor


@patch("core.managers.sam2.build_sam2_video_predictor")
@patch("core.managers.sam2.torch.cuda.is_available", return_value=False)
@patch("core.managers.sam2.torch", _torch_mod)
def test_sam2_wrapper_init(mock_cuda, mock_build, mock_predictor):
    mock_build.return_value = mock_predictor

    wrapper = SAM2Wrapper(checkpoint_path="dummy.pt", device="cpu")
    assert wrapper.device == "cpu"
    assert mock_build.called
    assert wrapper.predictor == mock_predictor


@patch("core.managers.sam2.build_sam2_video_predictor")
def test_sam2_wrapper_session_lifecycle(mock_build, mock_predictor):
    mock_build.return_value = mock_predictor
    wrapper = SAM2Wrapper(checkpoint_path="dummy.pt")

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


@patch("core.managers.sam2.build_sam2_video_predictor")
@patch("core.managers.sam2.torch", _torch_mod)
def test_sam2_wrapper_add_bbox_prompt(mock_build, mock_predictor):
    mock_build.return_value = mock_predictor
    wrapper = SAM2Wrapper(checkpoint_path="dummy.pt")
    wrapper._state = "test_state"

    # Test with 3D mask output (ndim == 3)
    mock_predictor.add_new_points_or_box.return_value = (None, None, _mock_tensor((1, 1, 100, 100)))
    mask = wrapper.add_bbox_prompt(frame_idx=0, obj_id=1, bbox_xywh=[10, 10, 50, 50], img_size=(100, 100))
    assert mask.shape[-2:] == (100, 100) or mask.shape == ()

    # Test with 2D mask output (ndim == 2) - though SAM2 usually returns 3D or 4D
    mock_predictor.add_new_points_or_box.return_value = (None, None, _mock_tensor((1, 100, 100)))
    mask = wrapper.add_bbox_prompt(frame_idx=0, obj_id=1, bbox_xywh=[10, 10, 50, 50], img_size=(100, 100))
    assert mask.shape[-2:] == (100, 100) or mask.shape == ()

    assert mock_predictor.add_new_points_or_box.called
    # Check that box was converted to xyxy correctly
    args, kwargs = mock_predictor.add_new_points_or_box.call_args
    assert np.array_equal(kwargs["box"], np.array([10, 10, 60, 60], dtype=np.float32))


@patch("core.managers.sam2.build_sam2_video_predictor")
@patch("core.managers.sam2.torch", _torch_mod)
def test_sam2_wrapper_propagate(mock_build, mock_predictor):
    mock_build.return_value = mock_predictor
    wrapper = SAM2Wrapper(checkpoint_path="dummy.pt")
    wrapper._state = "test_state"

    # Mock 3D masks in propagate
    mock_predictor.propagate_in_video.return_value = [
        (0, [1], _mock_tensor((1, 1, 100, 100))),
    ]
    results = list(wrapper.propagate(start_idx=0, max_frames=None, reverse=False))
    assert len(results) == 1
    assert results[0][2].shape[-2:] == (100, 100)

    # Check reverse parameter for backward direction
    list(wrapper.propagate(start_idx=5, reverse=True))
    args, kwargs = mock_predictor.propagate_in_video.call_args
    assert kwargs["reverse"] is True


@patch("core.managers.sam2.build_sam2_video_predictor")
@patch("core.managers.sam2.torch", _torch_mod)
def test_sam2_wrapper_add_point_prompt(mock_build, mock_predictor):
    mock_build.return_value = mock_predictor
    wrapper = SAM2Wrapper(checkpoint_path="dummy.pt")
    wrapper._state = "test_state"

    mock_predictor.add_new_points_or_box.return_value = (None, None, _mock_tensor((1, 1, 100, 100)))
    mask = wrapper.add_point_prompt(0, 1, [[50, 50]], [1], (100, 100))

    assert mask.shape[-2:] == (100, 100) or mask.shape == ()
    assert mock_predictor.add_new_points_or_box.called


@patch("core.managers.sam2.build_sam2_video_predictor")
def test_sam2_wrapper_stubs_and_utility(mock_build, mock_predictor):
    mock_build.return_value = mock_predictor
    wrapper = SAM2Wrapper(checkpoint_path="dummy.pt")
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


@patch("core.managers.sam2.build_sam2_video_predictor")
@patch("core.managers.sam2.torch.cuda.is_available", return_value=True)
def test_sam2_wrapper_shutdown(mock_cuda, mock_build, mock_predictor):
    mock_build.return_value = mock_predictor
    wrapper = SAM2Wrapper(checkpoint_path="dummy.pt")
    wrapper._state = "test_state"

    wrapper.shutdown()

    assert wrapper._state is None
    assert wrapper.predictor is None
