from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# We need to mock build_sam3_video_predictor BEFORE importing SAM3Wrapper if possible,
# but sam3_unit fixture in conftest already does some mocking.
# Let's use patch.dict on sys.modules to control the mock.


@pytest.fixture
def sam3_unit_fresh():
    mock_predictor = MagicMock()
    mock_predictor.handle_request.return_value = {
        "session_id": "test_session",
        "outputs": {"out_binary_masks": np.ones((1, 1, 10, 10))},
    }
    mock_predictor.handle_stream_request.return_value = [
        {"frame_index": 0, "outputs": {"out_binary_masks": np.ones((1, 1, 10, 10)), "out_obj_ids": [1]}}
    ]

    mock_build = MagicMock(return_value=mock_predictor)

    with patch.dict("sys.modules", {"sam3.model_builder": MagicMock(build_sam3_video_predictor=mock_build)}):
        from core.managers.sam3 import SAM3Wrapper

        wrapper = SAM3Wrapper(checkpoint_path="fake.pt", device="cpu")
        return wrapper


def test_sam3_wrapper_session_lifecycle_extended(sam3_unit_fresh):
    wrapper = sam3_unit_fresh
    # 1. init_video
    session_id = wrapper.init_video("video.mp4")
    assert session_id == "test_session"

    # 2. add_bbox_prompt
    mask = wrapper.add_bbox_prompt(0, 1, [0, 0, 5, 5], (10, 10))
    assert isinstance(mask, np.ndarray)

    # 3. propagate
    results = list(wrapper.propagate(0))
    assert len(results) == 1
    assert results[0][0] == 0

    # 4. close_session
    wrapper.close_session()
    assert wrapper.session_id is None


def test_sam3_wrapper_reset_session(sam3_unit_fresh):
    wrapper = sam3_unit_fresh
    wrapper.session_id = "old_session"

    with patch.object(wrapper, "close_session") as mock_close:
        wrapper.reset_session()
        mock_close.assert_called_once()


def test_sam3_wrapper_uninitialized_raises(sam3_unit_fresh):
    wrapper = sam3_unit_fresh
    wrapper.session_id = None

    with pytest.raises(RuntimeError, match="init_video must be called"):
        wrapper.add_bbox_prompt(0, 1, [0, 0, 5, 5], (10, 10))

    with pytest.raises(RuntimeError, match="init_video must be called"):
        list(wrapper.propagate(0))


def test_sam3_wrapper_import_error_degradation():
    """Test that SAM3Wrapper raises RuntimeError if predictor fails to load."""
    mock_build = MagicMock(return_value=None)

    with patch.dict("sys.modules", {"sam3.model_builder": MagicMock(build_sam3_video_predictor=mock_build)}):
        from core.managers.sam3 import SAM3Wrapper

        with pytest.raises(RuntimeError, match="SAM3 model failed to load"):
            SAM3Wrapper(checkpoint_path="fake.pt", device="cpu")


def test_sam3_wrapper_add_point_prompt(sam3_unit_fresh):
    wrapper = sam3_unit_fresh
    wrapper.session_id = "test_session"
    mask = wrapper.add_point_prompt(0, 1, [[5, 5]], [1], (10, 10))
    assert isinstance(mask, np.ndarray)
    wrapper.predictor.handle_request.assert_called()


def test_sam3_wrapper_remove_object(sam3_unit_fresh):
    wrapper = sam3_unit_fresh
    wrapper.session_id = "test_session"
    wrapper.remove_object(1)
    wrapper.predictor.handle_request.assert_called()


def test_sam3_wrapper_clear_prompts(sam3_unit_fresh):
    wrapper = sam3_unit_fresh
    wrapper.session_id = "test_session"
    with patch.object(wrapper, "reset_session") as mock_reset:
        wrapper.clear_prompts()
        mock_reset.assert_called_once()
