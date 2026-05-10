import threading
from queue import Queue
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from core.models import AnalysisParameters

# We need to ensure we're using the mocked torch from conftest if applicable,
# or we just need to be careful about what we import.
# Since this is a unit test (no 'integration' in name), conftest mocks are active.
from core.scene_utils.mask_propagator import MaskPropagator


@pytest.fixture
def mock_sam3_wrapper():
    """Mocks the SAM3Wrapper class."""
    wrapper = MagicMock()
    # Mock init_video to return inference state (dummy)
    wrapper.init_video.return_value = {"state": "ready"}
    # Mock add_bbox_prompt to return a mask (H, W)
    wrapper.add_bbox_prompt.return_value = np.ones((100, 100), dtype=bool)
    # Mock propagate to return a generator of (frame_idx, obj_id, mask)
    # Default behavior: empty generator
    wrapper.propagate.return_value = iter([])
    return wrapper


@pytest.fixture
def mask_propagator(mock_config, mock_logger, mock_sam3_wrapper):
    """Creates a MaskPropagator instance with mocks."""
    params = AnalysisParameters(
        source_path="test.mp4",
        output_folder="/tmp",
        min_mask_area_pct=0.1,  # 0.1%
    )
    cancel_event = threading.Event()
    progress_queue = Queue()

    return MaskPropagator(
        params=params,
        dam_tracker=mock_sam3_wrapper,
        cancel_event=cancel_event,
        progress_queue=progress_queue,
        config=mock_config,
        logger=mock_logger,
        device="cpu",
    )


class TestMaskPropagatorLogic:
    """
    Tests the logic of MaskPropagator without requiring real SAM3 models.
    """

    def test_initialization(self, mask_propagator, mock_sam3_wrapper):
        """Test that the propagator initializes correctly."""
        assert mask_propagator.dam_tracker == mock_sam3_wrapper
        assert not mask_propagator.cancel_event.is_set()

    def test_propagate_video_success(self, mask_propagator, mock_sam3_wrapper):
        """Test successful video propagation flow."""
        # Setup mocks
        # Forward: frame 2
        forward_gen = iter([(2, 1, np.ones((100, 100), dtype=bool))])
        # Backward: frame 0
        backward_gen = iter([(0, 1, np.ones((100, 100), dtype=bool))])

        mock_sam3_wrapper.propagate.side_effect = [forward_gen, backward_gen]

        frame_numbers = [0, 1, 2]
        seed_frame = 1
        bbox = [10, 10, 50, 50]
        frame_size = (100, 100)
        prompts = [{"frame": seed_frame, "bbox": bbox, "obj_id": 1}]
        frame_map = {0: "f0.webp", 1: "f1.webp", 2: "f2.webp"}

        masks, areas, empties, errors = mask_propagator.propagate_video(
            video_path="video.mp4",
            frame_numbers=frame_numbers,
            prompts=prompts,
            frame_size=frame_size,
            frame_map=frame_map,
        )

        # Verify SAM3 calls
        mock_sam3_wrapper.init_video.assert_called_once_with("video.mp4")
        mock_sam3_wrapper.add_bbox_prompt.assert_called_once_with(
            frame_idx=seed_frame, obj_id=1, bbox_xywh=bbox, img_size=frame_size, text="person"
        )

        # Verify propagation calls
        assert mock_sam3_wrapper.propagate.call_count == 2
        mock_sam3_wrapper.propagate.assert_any_call(start_idx=seed_frame, reverse=False, max_frames=1)
        mock_sam3_wrapper.propagate.assert_any_call(start_idx=seed_frame, reverse=True, max_frames=1)

        # Verify results
        assert len(masks) == 3
        assert 1 in masks
        assert np.any(masks[1] > 0)  # Seed mask (processed)

        assert areas[0] > 0  # Mask was processed
        # Use == instead of is because numpy bools are not singletons
        assert empties[0] == False
        assert errors[0] is None

    def test_propagate_video_handles_empty_masks(self, mask_propagator, mock_sam3_wrapper):
        """Test handling of empty/missing masks from SAM3."""
        # Seed frame returns None (failed prompt)
        mock_sam3_wrapper.add_bbox_prompt.return_value = None

        # Propagation returns None masks
        mock_sam3_wrapper.propagate.return_value = iter([(1, 1, None)])

        masks, areas, empties, errors = mask_propagator.propagate_video(
            video_path="video.mp4",
            frame_numbers=[0, 1],
            prompts=[{"frame": 0, "bbox": [0, 0, 10, 10]}],
            frame_size=(100, 100),
            frame_map={0: "a.webp", 1: "b.webp"},
        )

        # Should return blank masks, not None
        assert np.all(masks[0] == 0)
        assert np.all(masks[1] == 0)
        assert empties[0] == True
        assert errors[0] == "Empty mask"

    def test_propagate_video_cancellation(self, mask_propagator, mock_sam3_wrapper):
        """Test that propagation stops when cancel event is set."""

        # Setup infinite generator to simulate long process
        def infinite_gen():
            i = 1
            while True:
                yield (i, 1, np.ones((100, 100), dtype=bool))
                i += 1

        mock_sam3_wrapper.propagate.side_effect = [infinite_gen(), iter([])]

        # Cancel immediately
        mask_propagator.cancel_event.set()

        masks, areas, empties, errors = mask_propagator.propagate_video(
            video_path="video.mp4",
            frame_numbers=[0, 1, 2],
            prompts=[{"frame": 0, "bbox": [0, 0, 10, 10]}],
            frame_size=(100, 100),
            frame_map={0: "a.webp", 1: "b.webp", 2: "c.webp"},
        )

        # Verify it didn't crash
        assert 0 in masks

    def test_propagate_video_mid_batch_cancellation(self, mask_propagator, mock_sam3_wrapper):
        """Test cancellation within the frame processing loop."""

        # Mock propagate to return some frames then we cancel
        def gen():
            yield (1, 1, np.ones((10, 10), dtype=bool))
            mask_propagator.cancel_event.set()
            yield (2, 1, np.ones((10, 10), dtype=bool))

        mock_sam3_wrapper.propagate.return_value = gen()

        with patch("core.scene_utils.mask_propagator.postprocess_mask", side_effect=lambda x, **k: x):
            mask_propagator.propagate_video(
                video_path="v.mp4",
                frame_numbers=[0, 1, 2],
                prompts=[{"frame": 0, "bbox": [0, 0, 5, 5]}],
                frame_size=(10, 10),
                frame_map={0: "0.webp", 1: "1.webp", 2: "2.webp"},
            )

        assert mask_propagator.cancel_event.is_set()

    def test_propagate_video_outer_exception(self, mask_propagator, mock_sam3_wrapper):
        """Test outer error handler in propagate_video."""
        mock_sam3_wrapper.init_video.side_effect = Exception("Unexpected error")

        masks, areas, empties, errors = mask_propagator.propagate_video(
            video_path="v.mp4",
            frame_numbers=[0],
            prompts=[{"frame": 0, "bbox": [0, 0, 5, 5]}],
            frame_size=(10, 10),
            frame_map={0: "0.webp"},
        )
        assert "Unexpected error" in errors[0]

    def test_close_with_none_tracker(self, mask_propagator):
        """Test close() when dam_tracker is None."""
        mask_propagator.dam_tracker = None
        # Should not raise
        mask_propagator.close()

    def test_error_handling_gpu_oom(self, mask_propagator, mock_sam3_wrapper):
        """Test handling of CUDA OOM error."""
        with patch("core.scene_utils.mask_propagator.torch.cuda.OutOfMemoryError", RuntimeError):
            mock_sam3_wrapper.init_video.side_effect = RuntimeError("out of memory")

            with patch("core.scene_utils.mask_propagator.torch.cuda.is_available", return_value=True, create=True):
                with patch("core.scene_utils.mask_propagator.torch.cuda.empty_cache") as mock_empty_cache:
                    masks, areas, empties, errors = mask_propagator.propagate_video(
                        video_path="video.mp4",
                        frame_numbers=[0],
                        prompts=[{"frame": 0, "bbox": [0, 0, 10, 10]}],
                        frame_size=(100, 100),
                        frame_map={0: "a.webp"},
                    )

                    mock_empty_cache.assert_called()
                    assert empties[0] is True
                    assert "GPU error" in errors[0]

    def test_executor_cleanup_on_failure(self, mask_propagator, mock_sam3_wrapper):
        """Test ThreadPoolExecutor cleanup on failure."""
        mock_sam3_wrapper.propagate.return_value = iter([(1, 1, np.ones((10, 10), dtype=bool))])

        # Mock postprocess_mask to raise error for one frame
        with patch("core.scene_utils.mask_propagator.postprocess_mask", side_effect=Exception("post failure")):
            mask_propagator.propagate_video(
                video_path="v.mp4",
                frame_numbers=[0, 1],
                prompts=[{"frame": 0, "bbox": [0, 0, 5, 5]}],
                frame_size=(10, 10),
                frame_map={0: "0.webp", 1: "1.webp"},
            )
            # Should continue and log error for that frame
            mask_propagator.logger.error.assert_any_call("Parallel mask post-processing failed: post failure")

    def test_propagate_video_no_tracker(self, mask_propagator):
        """Test propagate_video with no tracker initialized."""
        mask_propagator.dam_tracker = None
        masks, areas, empties, errors = mask_propagator.propagate_video(
            video_path="v.mp4", frame_numbers=[0, 1], prompts=[], frame_size=(10, 10), frame_map={}
        )
        assert masks[0] is None
        assert errors[0] == "Tracker not initialized"

    def test_propagate_video_heartbeats(self, mask_propagator, mock_sam3_wrapper):
        """Test forward and backward propagation heartbeats."""
        # Setup many frames to trigger frame_idx % 50 == 0
        frames = [0, 50, 100]
        # Forward: frame 50
        forward_gen = iter([(50, 1, np.ones((10, 10), dtype=bool))])
        # Backward: frame 0
        backward_gen = iter([(0, 1, np.ones((10, 10), dtype=bool))])

        mock_sam3_wrapper.propagate.side_effect = [forward_gen, backward_gen]

        mask_propagator.propagate_video(
            video_path="v.mp4",
            frame_numbers=frames,
            prompts=[{"frame": 50, "bbox": [0, 0, 5, 5]}],
            frame_size=(10, 10),
            frame_map={f: f"{f}.webp" for f in frames},
        )

        # Verify heartbeats logged
        heartbeat_calls = [c for c in mask_propagator.logger.info.call_args_list if "heartbeat" in c[0][0]]
        assert len(heartbeat_calls) >= 2

    def test_propagate_video_finally_failure(self, mask_propagator, mock_sam3_wrapper):
        """Test failure in close_session during finally block."""
        mock_sam3_wrapper.close_session.side_effect = Exception("Cleanup Fail")

        # This should not raise but log debug
        mask_propagator.propagate_video(
            video_path="v.mp4", frame_numbers=[0], prompts=[], frame_size=(10, 10), frame_map={}
        )
        mask_propagator.logger.debug.assert_any_call(
            "Error during SAM3 session cleanup: Cleanup Fail", component="propagator"
        )

    def test_propagate_legacy_basic(self, mask_propagator, mock_sam3_wrapper):
        """Test legacy propagate method success path."""
        frames = [np.zeros((10, 10, 3), dtype=np.uint8)] * 3
        mock_sam3_wrapper.propagate.side_effect = [
            iter([(1, 1, np.ones((10, 10), dtype=bool))]),  # Forward from 0
            iter([]),  # Backward
        ]

        masks, areas, empties, errors = mask_propagator.propagate(
            shot_frames_rgb=frames, seed_idx=0, bbox_xywh=[0, 0, 5, 5]
        )
        assert len(masks) == 3
        assert areas[0] > 0

    def test_propagate_legacy_no_tracker(self, mask_propagator):
        """Test legacy propagate with no tracker."""
        mask_propagator.dam_tracker = None
        frames = [np.zeros((10, 10, 3), dtype=np.uint8)]
        masks, areas, empties, errors = mask_propagator.propagate(frames, 0, [0, 0, 5, 5])
        assert errors[0] == "Tracker not initialized"

    def test_propagate_legacy_error_paths(self, mask_propagator, mock_sam3_wrapper):
        """Test legacy propagate error handling."""
        frames = [np.zeros((10, 10, 3), dtype=np.uint8)]
        # GPU OOM
        with patch("core.scene_utils.mask_propagator.torch.cuda.OutOfMemoryError", RuntimeError):
            mock_sam3_wrapper.init_video.side_effect = RuntimeError("out of memory")
            mask_propagator.propagate(frames, 0, [0, 0, 5, 5])
            mask_propagator.logger.error.assert_any_call(
                "GPU error in propagation: out of memory", component="propagator"
            )

        # Generic Exception
        mock_sam3_wrapper.init_video.side_effect = Exception("Generic Fail")
        mask_propagator.propagate(frames, 0, [0, 0, 5, 5])
        mask_propagator.logger.error.assert_any_call(
            "Propagation error: Generic Fail", component="propagator", exc_info=True
        )
