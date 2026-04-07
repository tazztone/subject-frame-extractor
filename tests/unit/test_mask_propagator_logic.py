import threading
from queue import Queue
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from core.models import AnalysisParameters
from core.scene_utils.mask_propagator import MaskPropagator


@pytest.fixture
def mock_logger():
    return MagicMock()


@pytest.fixture
def mock_sam3_wrapper():
    mock = MagicMock()
    mock.init_video.return_value = None
    mock.add_bbox_prompt.return_value = np.ones((10, 10), dtype=bool)
    return mock


@pytest.fixture
def mock_config():
    config = MagicMock()
    config.sam2_model_path = "model.pt"
    config.sam3_model_path = "model.pt"
    config.device = "cpu"
    config.utility_mask_quality_threshold = 0.5
    config.min_mask_area_pct = 0.1
    return config


@pytest.fixture
def mask_propagator(mock_logger, mock_sam3_wrapper, mock_config):
    params = AnalysisParameters(source_path="video.mp4")
    params.min_mask_area_pct = 0.1
    propagator = MaskPropagator(
        params=params,
        dam_tracker=mock_sam3_wrapper,
        cancel_event=threading.Event(),
        progress_queue=Queue(),
        config=mock_config,
        logger=mock_logger,
    )
    return propagator


class TestMaskPropagatorLogic:
    @patch("core.scene_utils.mask_propagator.postprocess_mask", side_effect=lambda x, **k: x)
    def test_propagate_video_success(self, mock_post, mask_propagator, mock_sam3_wrapper):
        """Test successful video propagation."""
        mock_mask = np.ones((10, 10), dtype=bool)
        mock_sam3_wrapper.propagate.return_value = iter([(1, 1, mock_mask)])

        masks, areas, empties, errors = mask_propagator.propagate_video(
            video_path="video.mp4",
            frame_numbers=[0, 1],
            prompts=[{"frame": 0, "bbox": [0, 0, 10, 10]}],
            frame_size=(10, 10),
            frame_map={0: "a.webp", 1: "b.webp"},
        )

        assert 0 in masks
        assert 1 in masks
        assert areas[0] == 100.0
        assert areas[1] == 100.0
        mock_sam3_wrapper.init_video.assert_called_once()

    def test_propagate_video_outer_exception(self, mask_propagator, mock_sam3_wrapper):
        """Test unexpected exception during propagation."""
        # Use a specific message that we can reliably check
        mock_sam3_wrapper.init_video.side_effect = Exception("Unexpected error")

        masks, areas, empties, errors = mask_propagator.propagate_video(
            video_path="v.mp4",
            frame_numbers=[0],
            prompts=[{"frame": 0, "bbox": [0, 0, 10, 10]}],
            frame_size=(100, 100),
            frame_map={0: "a.webp"},
        )
        # str(errors[0]) should contain "Unexpected error"
        assert "Unexpected error" in str(errors[0])
