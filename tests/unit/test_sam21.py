from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def mock_sam2_predictor():
    with patch("core.managers.sam21.build_sam2_video_predictor") as mock_build:
        predictor = MagicMock()
        mock_build.return_value = predictor
        yield predictor


@pytest.fixture
def mock_cuda():
    with patch("core.managers.sam21.torch.cuda") as mock:
        yield mock


def test_sam21_wrapper_init(mock_sam2_predictor):
    """Test initialization logic and lazy loading."""
    from core.managers.sam21 import SAM21Wrapper

    wrapper = SAM21Wrapper("/tmp/model.pt", "cuda")

    assert wrapper.device == "cuda"
    assert wrapper.predictor is mock_sam2_predictor
    import core.managers.sam21

    core.managers.sam21.build_sam2_video_predictor.assert_called_once_with(
        config_file="configs/sam2.1/sam2.1_hiera_t.yaml", ckpt_path="/tmp/model.pt", device="cuda"
    )


@patch("core.managers.sam21.torch.inference_mode")
def test_sam21_init_video(mock_inference_mode, mock_sam2_predictor):
    """Test that init_video calls the underlying predictor."""
    from core.managers.sam21 import SAM21Wrapper

    wrapper = SAM21Wrapper("/tmp/model.pt", "cuda")
    wrapper.init_video("/tmp/frames")

    mock_sam2_predictor.init_state.assert_called_once_with(video_path="/tmp/frames")
    assert wrapper._state == mock_sam2_predictor.init_state.return_value


@patch("core.managers.sam21.torch.inference_mode")
def test_sam21_propagate_in_video(mock_inference_mode, mock_sam2_predictor):
    """Test the propagate generator logic."""
    from core.managers.sam21 import SAM21Wrapper

    wrapper = SAM21Wrapper("/tmp/model.pt", "cuda")
    wrapper._state = MagicMock()

    import numpy as np

    mock_mask = MagicMock()
    mock_mask.cpu.return_value.numpy.return_value = np.array([[[1, 1], [0, 0]]])

    # Mock the generator
    mock_sam2_predictor.propagate_in_video.return_value = [
        (0, [1], mock_mask),
    ]

    results = list(wrapper.propagate(start_idx=0, max_frames=10, direction="forward"))

    assert len(results) == 1
    mock_sam2_predictor.propagate_in_video.assert_called_once_with(
        wrapper._state, start_frame_idx=0, max_frame_num_to_track=10, reverse=False
    )


def test_sam21_close_session(mock_sam2_predictor, mock_cuda):
    """Test the cleanup logic in close_session."""
    from core.managers.sam21 import SAM21Wrapper

    wrapper = SAM21Wrapper("/tmp/model.pt", "cuda")
    original_state = MagicMock()
    wrapper._state = original_state

    wrapper.close_session()

    mock_sam2_predictor.reset_state.assert_called_once_with(original_state)
    assert wrapper._state is None
    mock_cuda.empty_cache.assert_called_once()
