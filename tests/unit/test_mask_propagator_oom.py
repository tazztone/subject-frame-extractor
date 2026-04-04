import threading
from queue import Queue
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from core.scene_utils.mask_propagator import MaskPropagator
from tests.conftest import OutOfMemoryError


class TestMaskPropagatorOOM:
    @pytest.fixture
    def mock_dam_tracker(self):
        return MagicMock()

    @pytest.fixture
    def propagator(self, mock_config, mock_logger, mock_dam_tracker):
        params = MagicMock()
        cancel_event = threading.Event()
        progress_queue = Queue()
        return MaskPropagator(
            params=params,
            dam_tracker=mock_dam_tracker,
            cancel_event=cancel_event,
            progress_queue=progress_queue,
            config=mock_config,
            logger=mock_logger,
        )

    @patch("core.scene_utils.mask_propagator.torch.cuda.empty_cache")
    @patch("core.scene_utils.mask_propagator.torch.cuda.is_available", return_value=True)
    def test_propagate_video_oom_recovery(self, mock_cuda_avail, mock_empty_cache, propagator, mock_dam_tracker):
        """Test that OutOfMemoryError is caught and handled during propagate_video."""
        # Setup tracker to raise OOM
        # propagate_video calls self.dam_tracker.propagate(...)
        mock_dam_tracker.propagate.side_effect = OutOfMemoryError("CUDA out of memory")

        frame_numbers = [1, 2]
        # start_frame_idx is derived from prompts[0]["frame"]
        prompts = [{"frame": 1, "bbox": [10, 10, 50, 50], "obj_id": 1}]

        # Run propagation
        masks, areas, empty, errors = propagator.propagate_video(
            video_path="test.mp4",
            frame_numbers=frame_numbers,
            prompts=prompts,
            frame_size=(500, 500),
            frame_map={1: "f1.png", 2: "f2.png"},
        )

        # Verify recovery
        # masks[fn] = np.zeros((h, w), dtype=np.uint8)
        for m in masks.values():
            assert isinstance(m, np.ndarray)
            assert np.sum(m) == 0

        assert all(v is True for v in empty.values())
        assert any("GPU error" in str(err) for err in errors.values())
        mock_empty_cache.assert_called()
        propagator.logger.error.assert_called()
