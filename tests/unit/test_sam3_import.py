from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from core.managers.sam3 import SAM3Wrapper


class TestSAM3Import:
    def test_sam3_wrapper_raises_on_missing_checkpoint(self):
        """Test SAM3Wrapper raises RuntimeError if checkpoint is missing/fail to load."""
        # SAM3Wrapper imports build_sam3_predictor FROM sam3.model_builder
        # We need to patch it in sys.modules or the module itself
        with patch("sam3.model_builder.build_sam3_predictor") as mock_build:
            mock_build.return_value = None  # Simulates load failure

            with pytest.raises(RuntimeError, match="failed to load"):
                SAM3Wrapper(checkpoint_path="/nonexistent/sam3.pt", device="cpu")

    def test_sam3_wrapper_init_success(self):
        """Test SAM3Wrapper initializes correctly when build succeeds."""
        mock_predictor = MagicMock()
        mock_predictor.model = MagicMock()

        with patch("sam3.model_builder.build_sam3_predictor", return_value=mock_predictor):
            wrapper = SAM3Wrapper(checkpoint_path="dummy.pt", device="cpu")
            assert wrapper.predictor == mock_predictor
            # Check if overrides were applied
            assert mock_predictor.model.hotstart_delay == 0
            assert mock_predictor.model.masklet_confirmation_enable is False

    def test_sam3_wrapper_session_and_prompts(self):
        """Test init_video, add_bbox_prompt, and close_session."""
        mock_predictor = MagicMock()
        mock_predictor.model = MagicMock()
        mock_predictor.handle_request.return_value = {"session_id": "sess_123"}

        with patch("sam3.model_builder.build_sam3_predictor", return_value=mock_predictor):
            wrapper = SAM3Wrapper(checkpoint_path="dummy.pt", device="cpu")

            # 1. Init video
            sid = wrapper.init_video("video.mp4")
            assert sid == "sess_123"
            assert wrapper.session_id == "sess_123"

            # 2. Add bbox prompt
            mock_predictor.handle_request.return_value = {"outputs": {"out_binary_masks": np.ones((1, 100, 100))}}
            mask = wrapper.add_bbox_prompt(0, 1, [10, 10, 50, 50], (100, 100))
            assert mask.shape == (100, 100)
            assert np.all(mask)

            # 3. Propagate
            mock_predictor.handle_stream_request.return_value = [
                {"frame_index": 1, "outputs": {"out_binary_masks": np.ones((1, 100, 100)), "out_obj_ids": [1]}}
            ]
            results = list(wrapper.propagate(start_idx=0, max_frames=5))
            assert len(results) == 1
            assert results[0][0] == 1  # frame_idx

            # 4. Close session
            wrapper.close_session()
            assert wrapper.session_id is None
