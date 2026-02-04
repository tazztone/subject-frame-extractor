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
        # Forward: frame 1
        forward_gen = iter([(1, 1, np.ones((100, 100), dtype=bool))])
        # Backward: frame -1 (not requested)
        backward_gen = iter([])

        mock_sam3_wrapper.propagate.side_effect = [forward_gen, backward_gen]

        frame_numbers = [0, 1]
        seed_frame = 0
        bbox = [10, 10, 50, 50]
        frame_size = (100, 100)

        masks, areas, empties, errors = mask_propagator.propagate_video(
            video_path="video.mp4",
            frame_numbers=frame_numbers,
            seed_frame_num=seed_frame,
            bbox_xywh=bbox,
            frame_size=frame_size,
        )

        # Verify SAM3 calls
        mock_sam3_wrapper.init_video.assert_called_once_with("video.mp4")
        mock_sam3_wrapper.add_bbox_prompt.assert_called_once_with(
            frame_idx=seed_frame, obj_id=1, bbox_xywh=bbox, img_size=frame_size
        )

        # Verify propagation calls
        assert mock_sam3_wrapper.propagate.call_count == 2
        mock_sam3_wrapper.propagate.assert_any_call(start_idx=seed_frame, reverse=False)
        mock_sam3_wrapper.propagate.assert_any_call(start_idx=seed_frame, reverse=True)

        # Verify results
        assert len(masks) == 2
        assert 0 in masks
        assert 1 in masks
        assert np.all(masks[0] == 255)  # Seed mask (from ones)
        assert np.all(masks[1] == 255)  # Propagated mask

        assert areas[0] == 100.0  # Full mask
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
            seed_frame_num=0,
            bbox_xywh=[0, 0, 10, 10],
            frame_size=(100, 100),
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
            seed_frame_num=0,
            bbox_xywh=[0, 0, 10, 10],
            frame_size=(100, 100),
        )

        # Verify it didn't crash
        assert 0 in masks

    def test_propagate_legacy_success(self, mask_propagator, mock_sam3_wrapper):
        """Test the legacy propagate() method with temp files."""
        frames = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(3)]

        # Setup mocks
        mock_sam3_wrapper.propagate.side_effect = [
            iter([(1, 1, np.ones((100, 100), dtype=bool))]),  # Forward
            iter([(2, 1, np.ones((100, 100), dtype=bool))]),  # Backward
        ]

        # Patch core.utils.rgb_to_pil since it is imported inside the method
        with patch("core.utils.rgb_to_pil") as mock_rgb_to_pil:
            mock_pil_img = MagicMock()
            mock_rgb_to_pil.return_value = mock_pil_img

            masks, areas, empties, errors = mask_propagator.propagate(
                shot_frames_rgb=frames, seed_idx=0, bbox_xywh=[0, 0, 10, 10]
            )

            # Verify it saved images (assuming rgb_to_pil is used)
            # Note: logic inside propagate does `from core.utils import rgb_to_pil`
            # Since we patched core.utils.rgb_to_pil, the import should get the mock
            assert mock_pil_img.save.call_count == 3

            # Verify SAM3 init with a path
            mock_sam3_wrapper.init_video.assert_called_once()
            args, _ = mock_sam3_wrapper.init_video.call_args
            assert isinstance(args[0], str)  # path

            # Verify propagation
            assert len(masks) == 3
            assert masks[0] is not None  # Seed

    def test_error_handling_gpu_oom(self, mask_propagator, mock_sam3_wrapper):
        """Test handling of CUDA OOM error."""

        # The issue with TypeError: catching classes that do not inherit from BaseException
        # is likely because conftest.py mocks torch.cuda.OutOfMemoryError but assigns it to something that isn't a class
        # inheriting from BaseException, OR because we are importing torch inside the test
        # but the module under test (mask_propagator.py) imported it when it was mocked by conftest.

        # In conftest.py:
        # mock_torch = MagicMock(name='torch')
        # ...
        # This means torch.cuda.OutOfMemoryError is a MagicMock object, which is not a type.

        # The fix is to ensure torch.cuda.OutOfMemoryError is a real class (or mock class) that inherits from BaseException.
        # But we cannot easily change conftest.py without affecting other tests.

        # However, MaskPropagator also catches RuntimeError.
        # If we raise RuntimeError, it should be caught.
        # The traceback showed:
        # except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        # TypeError: catching classes that do not inherit from BaseException is not allowed

        # This confirms that torch.cuda.OutOfMemoryError is NOT a valid exception class in the context of mask_propagator.py
        # because of the mock.

        # To test this, we must ensure that mask_propagator uses a valid exception class.
        # Since we can't change the import in mask_propagator.py easily (it's already imported),
        # We can try to patch `core.scene_utils.mask_propagator.torch.cuda.OutOfMemoryError` to be `RuntimeError`.

        with patch("core.scene_utils.mask_propagator.torch.cuda.OutOfMemoryError", RuntimeError):
            # Now the except clause is effectively `except (RuntimeError, RuntimeError)` which is valid.

            # We raise RuntimeError
            mock_sam3_wrapper.init_video.side_effect = RuntimeError("GPU OOM Simulated")

            # Also ensure torch.cuda.is_available is True
            with patch("core.scene_utils.mask_propagator.torch.cuda.is_available", return_value=True):
                with patch("core.scene_utils.mask_propagator.torch.cuda.empty_cache") as mock_empty_cache:
                    masks, areas, empties, errors = mask_propagator.propagate_video(
                        video_path="video.mp4",
                        frame_numbers=[0],
                        seed_frame_num=0,
                        bbox_xywh=[0, 0, 10, 10],
                        frame_size=(100, 100),
                    )

                    mock_empty_cache.assert_called()
                    assert empties[0] is True
                    assert "GPU error" in errors[0]

    def test_tracker_progress(self, mask_propagator, mock_sam3_wrapper):
        """Test that progress tracker is updated."""
        tracker = MagicMock()

        mock_sam3_wrapper.propagate.return_value = iter([(1, 1, np.ones((100, 100), dtype=bool))])

        mask_propagator.propagate_video(
            video_path="video.mp4",
            frame_numbers=[0, 1],
            seed_frame_num=0,
            bbox_xywh=[0, 0, 10, 10],
            frame_size=(100, 100),
            tracker=tracker,
        )

        assert tracker.set_stage.call_count >= 1
        assert tracker.step.call_count >= 1
