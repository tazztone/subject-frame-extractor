"""
Unit tests for SAM3Wrapper API completeness and functionality.

These tests ensure all required methods exist on SAM3Wrapper and verify
basic functionality with mocking to avoid GPU requirements.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class TestSAM3WrapperAPICompleteness:
    """
    Tests to verify SAM3Wrapper has all required methods.

    These tests would have caught the missing detect_objects() issue.
    """

    @pytest.fixture
    def mock_wrapper_class(self):
        """Create a mock-free SAM3Wrapper class reference."""
        # Use build_sam3_video_predictor as per the new refactor
        with patch("core.managers.build_sam3_video_predictor") as mock_build:
            mock_predictor = MagicMock()
            mock_build.return_value = mock_predictor

            with patch("torch.cuda.is_available", return_value=False):
                from core.managers import SAM3Wrapper

                yield SAM3Wrapper

    @pytest.fixture
    def mock_wrapper(self, mock_wrapper_class):
        """Create a mocked SAM3Wrapper instance."""
        return mock_wrapper_class(device="cpu")

    def test_all_required_methods_exist(self, mock_wrapper):
        """
        Verify all API methods expected by SeedSelector exist on SAM3Wrapper.

        Now includes remove_object and shutdown.
        """
        required_methods = [
            "init_video",
            "add_bbox_prompt",
            "propagate",
            "clear_prompts",
            # These were missing before:
            "detect_objects",
            "add_text_prompt",
            "add_point_prompt",
            "reset_session",
            "close_session",
            "remove_object",
            "shutdown",
        ]

        for method in required_methods:
            assert hasattr(mock_wrapper, method), f"SAM3Wrapper missing required method: {method}"
            assert callable(getattr(mock_wrapper, method)), f"SAM3Wrapper.{method} is not callable"

    def test_detect_objects_signature(self, mock_wrapper):
        """Verify detect_objects has correct signature."""
        import inspect

        sig = inspect.signature(mock_wrapper.detect_objects)
        params = list(sig.parameters.keys())

        assert "frame_rgb" in params, "detect_objects missing frame_rgb parameter"
        assert "prompt" in params, "detect_objects missing prompt parameter"

    def test_add_text_prompt_signature(self, mock_wrapper):
        """Verify add_text_prompt has correct signature."""
        import inspect

        sig = inspect.signature(mock_wrapper.add_text_prompt)
        params = list(sig.parameters.keys())

        assert "frame_idx" in params, "add_text_prompt missing frame_idx parameter"
        assert "text" in params, "add_text_prompt missing text parameter"

    def test_add_point_prompt_signature(self, mock_wrapper):
        """Verify add_point_prompt has correct signature."""
        import inspect

        sig = inspect.signature(mock_wrapper.add_point_prompt)
        params = list(sig.parameters.keys())

        assert "frame_idx" in params
        assert "obj_id" in params
        assert "points" in params
        assert "labels" in params
        assert "img_size" in params


class TestSAM3WrapperMethodBehavior:
    """Tests for SAM3Wrapper method behavior with mocking."""

    @pytest.fixture
    def mock_wrapper(self):
        """Create a fully mocked SAM3Wrapper."""
        with patch("core.managers.build_sam3_video_predictor") as mock_build:
            mock_predictor = MagicMock()
            mock_predictor.handle_request = MagicMock(return_value={})
            mock_predictor.handle_stream_request = MagicMock(return_value=[])

            mock_build.return_value = mock_predictor

            with patch("torch.cuda.is_available", return_value=False):
                from core.managers import SAM3Wrapper

                wrapper = SAM3Wrapper(device="cpu")
                # wrapper.predictor is already set by __init__ due to mock_build
                yield wrapper

    def test_detect_objects_returns_empty_on_empty_prompt(self, mock_wrapper):
        """detect_objects should return empty list for empty prompt."""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        assert mock_wrapper.detect_objects(frame, "") == []
        assert mock_wrapper.detect_objects(frame, "   ") == []
        assert mock_wrapper.detect_objects(frame, None) == []

    def test_add_text_prompt_requires_init_video(self, mock_wrapper):
        """add_text_prompt should raise if init_video not called."""
        mock_wrapper.session_id = None

        with pytest.raises(RuntimeError) as exc_info:
            mock_wrapper.add_text_prompt(0, "person")

        assert "init_video" in str(exc_info.value)

    def test_add_point_prompt_requires_init_video(self, mock_wrapper):
        """add_point_prompt should raise if init_video not called."""
        mock_wrapper.session_id = None

        with pytest.raises(RuntimeError) as exc_info:
            mock_wrapper.add_point_prompt(0, 1, [(50, 50)], [1], (100, 100))

        assert "init_video" in str(exc_info.value)

    def test_reset_session_with_no_state_is_safe(self, mock_wrapper):
        """reset_session should not raise when no session is active."""
        mock_wrapper.session_id = None
        mock_wrapper.reset_session()  # Should not raise

    def test_close_session_with_no_state_is_safe(self, mock_wrapper):
        """close_session should not raise when no session is active."""
        mock_wrapper.session_id = None
        mock_wrapper.close_session()  # Should not raise

    def test_close_session_clears_session_id(self, mock_wrapper):
        """close_session should clear session_id."""
        mock_wrapper.session_id = "test_session"
        mock_wrapper.predictor = MagicMock()

        mock_wrapper.close_session()

        assert mock_wrapper.session_id is None

    def test_remove_object_calls_predictor(self, mock_wrapper):
        """remove_object should call predictor handle_request."""
        mock_wrapper.session_id = "test_session"

        mock_wrapper.remove_object(1)

        mock_wrapper.predictor.handle_request.assert_called_with(
            request={
                "type": "remove_object",
                "session_id": "test_session",
                "obj_id": 1,
            }
        )

    def test_shutdown_calls_predictor(self, mock_wrapper):
        """shutdown should call predictor shutdown."""
        mock_wrapper.shutdown()
        mock_wrapper.predictor.shutdown.assert_called_once()


class TestSeedSelectorTrackerInterface:
    """
    Tests that verify SeedSelector's expected tracker interface.

    This catches cases where SeedSelector calls methods that don't exist.
    """

    def test_seed_selector_tracker_methods_match_wrapper(self):
        """
        Verify all tracker methods called by SeedSelector exist on SAM3Wrapper.

        Parse SeedSelector code to find tracker method calls.
        """
        import re
        from pathlib import Path

        seed_selector_path = Path(__file__).parent.parent / "core" / "scene_utils" / "seed_selector.py"
        content = seed_selector_path.read_text(encoding="utf-8")

        # Find all self.tracker.method_name() calls
        tracker_calls = re.findall(r"self\.tracker\.(\w+)\s*\(", content)

        # These are the methods SeedSelector expects to exist
        expected_methods = set(tracker_calls)

        # Verify SAM3Wrapper has all of them
        with patch("core.managers.build_sam3_video_predictor") as mock_build:
            mock_predictor = MagicMock()
            mock_build.return_value = mock_predictor

            with patch("torch.cuda.is_available", return_value=False):
                from core.managers import SAM3Wrapper

                wrapper = SAM3Wrapper(device="cpu")

                for method in expected_methods:
                    assert hasattr(wrapper, method), (
                        f"SeedSelector calls tracker.{method}() but SAM3Wrapper doesn't have it!"
                    )
