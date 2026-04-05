"""
Unit tests for SAM2Wrapper API completeness and functionality.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

pytestmark = [pytest.mark.sam2]


@pytest.fixture
def mock_predictor():
    """Provides a mocked SAM2 predictor with deterministic mask outputs."""
    predictor = MagicMock()
    predictor.init_state.return_value = MagicMock(name="test_state")

    # Use the robust mock tensor from conftest
    from tests.conftest import createmocktensor

    mask_3d = createmocktensor("mask_3d", (1, 1, 100, 100))

    # Ensure predictor methods return our MockTensor for deterministic mask comparisons
    predictor.add_new_points_or_box.return_value = (None, None, mask_3d)
    predictor.propagate_in_video.return_value = [
        (0, [1], mask_3d),
        (1, [1], mask_3d),
    ]
    return predictor


def test_sam2_wrapper_init(mock_predictor):
    """Test initialization logic and lazy loading."""
    from core.managers.sam2 import SAM2Wrapper

    # Patch the real sam2 build function
    with patch("sam2.build_sam.build_sam2_video_predictor", return_value=mock_predictor):
        wrapper = SAM2Wrapper("dummy_ckpt")
        assert wrapper.predictor == mock_predictor


def test_sam2_wrapper_init_video(mock_predictor):
    """Test that init_video calls the underlying predictor."""
    from core.managers.sam2 import SAM2Wrapper

    with patch("sam2.build_sam.build_sam2_video_predictor", return_value=mock_predictor):
        wrapper = SAM2Wrapper("dummy_ckpt")
        sid = wrapper.init_video("dummy_path")
        # sid should be the id of the state object
        assert sid is not None
        mock_predictor.init_state.assert_called_once_with(video_path="dummy_path")


def test_sam2_wrapper_add_bbox_prompt(mock_predictor):
    """Test that add_bbox_prompt returns a boolean mask."""
    from core.managers.sam2 import SAM2Wrapper

    with patch("sam2.build_sam.build_sam2_video_predictor", return_value=mock_predictor):
        wrapper = SAM2Wrapper("dummy_ckpt")
        wrapper.init_video("dummy_path")

        with patch("torch.inference_mode"):
            mask = wrapper.add_bbox_prompt(frame_idx=0, obj_id=1, bbox_xywh=[10, 10, 50, 50], img_size=(100, 100))

        # mask should be a boolean numpy array
        assert isinstance(mask, np.ndarray)
        assert mask.dtype == bool
        assert mask.shape == (100, 100)
        mock_predictor.add_new_points_or_box.assert_called_once()


def test_sam2_wrapper_propagate(mock_predictor):
    """Test the propagate generator logic."""
    from core.managers.sam2 import SAM2Wrapper

    with patch("sam2.build_sam.build_sam2_video_predictor", return_value=mock_predictor):
        wrapper = SAM2Wrapper("dummy_ckpt")
        wrapper.init_video("dummy_path")

        with patch("torch.inference_mode"):
            results = list(wrapper.propagate(start_idx=0, max_frames=None, reverse=False))

        # Generator yields (frame_idx, obj_ids, masks)
        assert len(results) == 2
        assert results[0][0] == 0
        assert results[1][0] == 1
        assert isinstance(results[0][2], np.ndarray)


def test_sam2_wrapper_close_session(mock_predictor):
    """Test the cleanup logic in close_session."""
    from core.managers.sam2 import SAM2Wrapper

    with patch("sam2.build_sam.build_sam2_video_predictor", return_value=mock_predictor):
        wrapper = SAM2Wrapper("dummy_ckpt")
        wrapper.init_video("dummy_path")
        wrapper.close_session()
        assert wrapper._state is None


def test_sam2_wrapper_shutdown(mock_predictor):
    """Test the shutdown and GC logic."""
    from core.managers.sam2 import SAM2Wrapper

    with patch("sam2.build_sam.build_sam2_video_predictor", return_value=mock_predictor):
        wrapper = SAM2Wrapper("dummy_ckpt")
        wrapper.shutdown()
        assert wrapper.predictor is None
