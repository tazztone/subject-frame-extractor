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
def mask_propagator(mock_logger, mock_sam3_wrapper):
    config = MagicMock()
    config.device = "cuda"
    config.utility_mask_quality_threshold = 0.5
    config.min_mask_area_pct = 0.1

    params = AnalysisParameters(source_path="v.mp4")
    params.min_mask_area_pct = 0.1

    return MaskPropagator(
        params=params,
        dam_tracker=mock_sam3_wrapper,
        cancel_event=threading.Event(),
        progress_queue=Queue(),
        config=config,
        logger=mock_logger,
    )


class TestMaskPropagatorOOM:
    def test_propagate_video_oom_recovery(self, mask_propagator, mock_sam3_wrapper):
        """Test handling of CUDA OOM error and cache clearing."""
        # 1. Force the sampler to raise "out of memory"
        mock_sam3_wrapper.init_video.side_effect = RuntimeError("out of memory")

        # 2. Mock device checks
        with patch("core.scene_utils.mask_propagator.is_cuda_available", return_value=True):
            with patch("core.scene_utils.mask_propagator.empty_cache") as mock_empty_cache:
                mask_propagator.device = "cuda"  # Explicitly set device to cuda for test

                masks, areas, empties, errors = mask_propagator.propagate_video(
                    video_path="v.mp4",
                    frame_numbers=[0],
                    prompts=[{"frame": 0, "bbox": [0, 0, 10, 10]}],
                    frame_size=(100, 100),
                    frame_map={0: "a.webp"},
                )

                # Verify empty_cache was called
                mock_empty_cache.assert_called_once()
                # Verify error was captured
                assert "GPU error" in str(errors[0])
                assert empties[0] is True
