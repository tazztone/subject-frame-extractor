"""
Unit tests for SAM3Wrapper API completeness and functionality.

These tests ensure all required methods exist on SAM3Wrapper and verify
basic functionality with mocking to avoid GPU requirements.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch, PropertyMock


class TestSAM3WrapperAPICompleteness:
    """
    Tests to verify SAM3Wrapper has all required methods.
    
    These tests would have caught the missing detect_objects() issue.
    """

    @pytest.fixture
    def mock_wrapper_class(self):
        """Create a mock-free SAM3Wrapper class reference."""
        with patch('core.managers.build_sam3_video_model') as mock_build:
            mock_model = MagicMock()
            mock_model.tracker = MagicMock()
            mock_model.detector = MagicMock()
            mock_model.detector.backbone = MagicMock()
            mock_build.return_value = mock_model
            
            with patch('torch.cuda.is_available', return_value=False):
                from core.managers import SAM3Wrapper
                yield SAM3Wrapper

    @pytest.fixture
    def mock_wrapper(self, mock_wrapper_class):
        """Create a mocked SAM3Wrapper instance."""
        return mock_wrapper_class(device='cpu')

    def test_all_required_methods_exist(self, mock_wrapper):
        """
        Verify all API methods expected by SeedSelector exist on SAM3Wrapper.
        
        This is the key test that would have caught detect_objects() missing.
        """
        required_methods = [
            'init_video',
            'add_bbox_prompt',
            'propagate',
            'clear_prompts',
            # These were missing before:
            'detect_objects',
            'add_text_prompt',
            'add_point_prompt',
            'reset_session',
            'close_session',
        ]
        
        for method in required_methods:
            assert hasattr(mock_wrapper, method), f"SAM3Wrapper missing required method: {method}"
            assert callable(getattr(mock_wrapper, method)), f"SAM3Wrapper.{method} is not callable"

    def test_detect_objects_signature(self, mock_wrapper):
        """Verify detect_objects has correct signature."""
        import inspect
        sig = inspect.signature(mock_wrapper.detect_objects)
        params = list(sig.parameters.keys())
        
        assert 'frame_rgb' in params, "detect_objects missing frame_rgb parameter"
        assert 'prompt' in params, "detect_objects missing prompt parameter"

    def test_add_text_prompt_signature(self, mock_wrapper):
        """Verify add_text_prompt has correct signature."""
        import inspect
        sig = inspect.signature(mock_wrapper.add_text_prompt)
        params = list(sig.parameters.keys())
        
        assert 'frame_idx' in params, "add_text_prompt missing frame_idx parameter"
        assert 'text' in params, "add_text_prompt missing text parameter"

    def test_add_point_prompt_signature(self, mock_wrapper):
        """Verify add_point_prompt has correct signature."""
        import inspect
        sig = inspect.signature(mock_wrapper.add_point_prompt)
        params = list(sig.parameters.keys())
        
        assert 'frame_idx' in params
        assert 'obj_id' in params
        assert 'points' in params
        assert 'labels' in params
        assert 'img_size' in params


class TestSAM3WrapperMethodBehavior:
    """Tests for SAM3Wrapper method behavior with mocking."""

    @pytest.fixture
    def mock_wrapper(self):
        """Create a fully mocked SAM3Wrapper."""
        with patch('core.managers.build_sam3_video_model') as mock_build:
            mock_model = MagicMock()
            mock_tracker = MagicMock()
            mock_detector = MagicMock()
            mock_detector.backbone = MagicMock()
            
            mock_model.tracker = mock_tracker
            mock_model.detector = mock_detector
            mock_build.return_value = mock_model
            
            with patch('torch.cuda.is_available', return_value=False):
                from core.managers import SAM3Wrapper
                wrapper = SAM3Wrapper(device='cpu')
                wrapper.sam3_model = mock_model
                yield wrapper

    def test_detect_objects_returns_empty_on_empty_prompt(self, mock_wrapper):
        """detect_objects should return empty list for empty prompt."""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        
        assert mock_wrapper.detect_objects(frame, "") == []
        assert mock_wrapper.detect_objects(frame, "   ") == []
        assert mock_wrapper.detect_objects(frame, None) == []

    def test_add_text_prompt_requires_init_video(self, mock_wrapper):
        """add_text_prompt should raise if init_video not called."""
        mock_wrapper.inference_state = None
        
        with pytest.raises(RuntimeError) as exc_info:
            mock_wrapper.add_text_prompt(0, "person")
        
        assert "init_video" in str(exc_info.value)

    def test_add_point_prompt_requires_init_video(self, mock_wrapper):
        """add_point_prompt should raise if init_video not called."""
        mock_wrapper.inference_state = None
        
        with pytest.raises(RuntimeError) as exc_info:
            mock_wrapper.add_point_prompt(0, 1, [(50, 50)], [1], (100, 100))
        
        assert "init_video" in str(exc_info.value)

    def test_reset_session_with_no_state_is_safe(self, mock_wrapper):
        """reset_session should not raise when no session is active."""
        mock_wrapper.inference_state = None
        mock_wrapper.reset_session()  # Should not raise

    def test_close_session_with_no_state_is_safe(self, mock_wrapper):
        """close_session should not raise when no session is active."""
        mock_wrapper.inference_state = None
        mock_wrapper.close_session()  # Should not raise

    def test_close_session_clears_inference_state(self, mock_wrapper):
        """close_session should clear inference_state."""
        mock_wrapper.inference_state = {"some": "state"}
        mock_wrapper.predictor = MagicMock()
        
        mock_wrapper.close_session()
        
        assert mock_wrapper.inference_state is None


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
        from pathlib import Path
        import re
        
        seed_selector_path = Path(__file__).parent.parent / 'core' / 'scene_utils' / 'seed_selector.py'
        content = seed_selector_path.read_text(encoding='utf-8')
        
        # Find all self.tracker.method_name() calls
        tracker_calls = re.findall(r'self\.tracker\.(\w+)\s*\(', content)
        
        # These are the methods SeedSelector expects to exist
        expected_methods = set(tracker_calls)
        
        # Verify SAM3Wrapper has all of them
        with patch('core.managers.build_sam3_video_model') as mock_build:
            mock_model = MagicMock()
            mock_model.tracker = MagicMock()
            mock_model.detector = MagicMock()
            mock_model.detector.backbone = MagicMock()
            mock_build.return_value = mock_model
            
            with patch('torch.cuda.is_available', return_value=False):
                from core.managers import SAM3Wrapper
                wrapper = SAM3Wrapper(device='cpu')
                
                for method in expected_methods:
                    assert hasattr(wrapper, method), \
                        f"SeedSelector calls tracker.{method}() but SAM3Wrapper doesn't have it!"
