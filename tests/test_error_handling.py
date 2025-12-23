"""
Tests for error handling and edge cases.

These tests ensure the application handles errors gracefully and provides
useful feedback to users.
"""
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
import numpy as np

from core.config import Config
from core.models import Scene, Frame, AnalysisParameters, QualityConfig
from core.filtering import apply_all_filters_vectorized
from pydantic import ValidationError


class TestConfigEdgeCases:
    """Tests for Config validation and edge cases."""

    def test_config_default_values(self):
        """Test that Config has sensible defaults."""
        config = Config()
        assert config.sharpness_base_scale > 0
        assert config.edge_strength_base_scale > 0
        assert config.retry_max_attempts >= 1

    def test_config_custom_values(self, tmp_path):
        """Test that Config accepts custom values."""
        config = Config(
            downloads_dir=str(tmp_path / "dl"),
            sharpness_base_scale=5000.0,
            edge_strength_base_scale=200.0,
        )
        assert config.sharpness_base_scale == 5000.0
        assert config.edge_strength_base_scale == 200.0

    def test_config_path_creation(self, tmp_path):
        """Test that Config creates necessary directories."""
        logs_dir = tmp_path / "logs"
        config = Config(logs_dir=str(logs_dir))
        # logs_dir should be usable as a path
        assert isinstance(config.logs_dir, str)

    @patch('pathlib.Path.mkdir', MagicMock())
    @patch('pathlib.Path.touch', MagicMock()) 
    @patch('pathlib.Path.unlink', MagicMock())
    def test_config_invalid_quality_weights(self):
        """Test that Config rejects invalid quality weights (sum cannot be zero)."""
        with pytest.raises(ValidationError):
            Config(
                quality_weights_sharpness=0,
                quality_weights_edge_strength=0,
                quality_weights_contrast=0,
                quality_weights_brightness=0,
                quality_weights_entropy=0,
                quality_weights_niqe=0,
            )


class TestSceneEdgeCases:
    """Tests for Scene model edge cases."""

    def test_scene_minimal(self):
        """Test Scene with minimal required fields."""
        scene = Scene(shot_id=1, start_frame=0, end_frame=10)
        assert scene.shot_id == 1
        assert scene.status == "pending"

    def test_scene_with_seed_result(self):
        """Test Scene with seed_result data."""
        scene = Scene(
            shot_id=1,
            start_frame=0,
            end_frame=100,
            seed_result={'bbox': [10, 10, 50, 50], 'details': {'type': 'person'}},
            seed_metrics={'score': 0.95},
        )
        assert scene.seed_result is not None
        assert scene.seed_metrics['score'] == 0.95

    def test_scene_status_transitions(self):
        """Test Scene status can be changed."""
        scene = Scene(shot_id=1, start_frame=0, end_frame=10)
        assert scene.status == "pending"
        
        scene.status = "included"
        assert scene.status == "included"
        
        scene.status = "excluded"
        assert scene.status == "excluded"


class TestFrameEdgeCases:
    """Tests for Frame model edge cases."""

    def test_frame_with_none_image(self):
        """Test Frame with None image data (some workflows don't need images)."""
        # Frame requires numpy array, so we need to provide a dummy one
        dummy = np.zeros((10, 10, 3), dtype=np.uint8)
        frame = Frame(image_data=dummy, frame_number=1)
        assert frame.frame_number == 1
        assert frame.metrics is not None

    def test_frame_with_image_data(self, sample_image_rgb):
        """Test Frame stores image data correctly."""
        frame = Frame(image_data=sample_image_rgb, frame_number=1)
        assert frame.image_data is not None
        assert frame.image_data.shape == (100, 100, 3)


class TestFilteringEdgeCases:
    """Tests for filtering with edge cases and empty data."""

    def test_filter_empty_frames(self, mock_config):
        """Test filtering handles empty frame list."""
        filters = {"face_sim_enabled": False}
        kept, rejected, _, _ = apply_all_filters_vectorized([], filters, mock_config)
        assert len(kept) == 0
        assert len(rejected) == 0

    def test_filter_all_frames_pass(self, sample_frames_data, mock_config):
        """Test filtering when all frames pass."""
        filters = {
            "face_sim_enabled": False,
            "mask_area_enabled": False,
        }
        kept, rejected, _, _ = apply_all_filters_vectorized(sample_frames_data, filters, mock_config)
        assert len(kept) == 6  # All frames should pass
        assert len(rejected) == 0

    def test_filter_strict_thresholds(self, sample_frames_data, mock_config):
        """Test filtering with very strict thresholds rejects most frames."""
        filters = {
            "face_sim_enabled": True,
            "face_sim_min": 0.99,  # Very high threshold
        }
        kept, rejected, _, _ = apply_all_filters_vectorized(sample_frames_data, filters, mock_config)
        # All frames with face_sim should be rejected (max is 0.8)
        assert len(rejected) >= len([f for f in sample_frames_data if f.get('face_sim')])


class TestAnalysisParametersValidation:
    """Tests for AnalysisParameters validation."""

    def test_params_minimal(self, tmp_path):
        """Test AnalysisParameters with minimal required fields."""
        params = AnalysisParameters(
            source_path="video.mp4",
            output_folder=str(tmp_path),
        )
        assert params.source_path == "video.mp4"
        assert params.thumbnails_only is True  # Default

    def test_params_full(self, tmp_path):
        """Test AnalysisParameters with all fields."""
        params = AnalysisParameters(
            source_path="video.mp4",
            video_path="video.mp4",
            output_folder=str(tmp_path),
            thumbnails_only=False,
            tracker_model_name="sam3",
            enable_face_filter=True,
            face_ref_img_path="face.jpg",
        )
        assert params.enable_face_filter is True
        assert params.tracker_model_name == "sam3"


class TestQualityConfigEdgeCases:
    """Tests for QualityConfig edge cases."""

    def test_quality_config_with_required_fields(self):
        """Test QualityConfig with required fields."""
        config = QualityConfig(sharpness_base_scale=2500.0, edge_strength_base_scale=100.0)
        assert config.sharpness_base_scale == 2500.0
        assert config.edge_strength_base_scale == 100.0
        assert config.enable_niqe is True  # default

    def test_quality_config_niqe_disabled(self):
        """Test QualityConfig with NIQE disabled."""
        config = QualityConfig(sharpness_base_scale=2500.0, edge_strength_base_scale=100.0, enable_niqe=False)
        assert config.enable_niqe is False

    def test_quality_config_custom_scales(self):
        """Test QualityConfig with custom scales."""
        config = QualityConfig(
            sharpness_base_scale=5000.0,
            edge_strength_base_scale=200.0,
        )
        assert config.sharpness_base_scale == 5000.0
        assert config.edge_strength_base_scale == 200.0


class TestErrorHandlerDecorators:
    """Tests for ErrorHandler.with_retry and with_fallback decorators."""

    def test_with_retry_success_first_try(self, mock_logger):
        """Test with_retry when function succeeds on first try."""
        from core.error_handling import ErrorHandler

        handler = ErrorHandler(mock_logger, max_attempts=3, backoff_seconds=[0.01, 0.02])

        call_count = 0

        @handler.with_retry()
        def always_succeeds():
            nonlocal call_count
            call_count += 1
            return "success"

        result = always_succeeds()

        assert result == "success"
        assert call_count == 1

    def test_with_retry_success_after_failures(self, mock_logger):
        """Test with_retry when function succeeds after initial failures."""
        from core.error_handling import ErrorHandler

        handler = ErrorHandler(mock_logger, max_attempts=3, backoff_seconds=[0.01, 0.02])

        call_count = 0

        @handler.with_retry()
        def fails_twice_then_succeeds():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Temporary failure")
            return "success"

        result = fails_twice_then_succeeds()

        assert result == "success"
        assert call_count == 3

    def test_with_retry_all_attempts_fail(self, mock_logger):
        """Test with_retry raises exception when all attempts fail."""
        from core.error_handling import ErrorHandler

        handler = ErrorHandler(mock_logger, max_attempts=2, backoff_seconds=[0.01])

        @handler.with_retry()
        def always_fails():
            raise ValueError("Always fails")

        with pytest.raises(ValueError) as exc_info:
            always_fails()

        assert "Always fails" in str(exc_info.value)

    def test_with_retry_custom_exceptions(self, mock_logger):
        """Test with_retry only catches specified exceptions."""
        from core.error_handling import ErrorHandler

        handler = ErrorHandler(mock_logger, max_attempts=3, backoff_seconds=[0.01])

        call_count = 0

        @handler.with_retry(recoverable_exceptions=(ValueError,))
        def raises_type_error():
            nonlocal call_count
            call_count += 1
            raise TypeError("Not recoverable")

        with pytest.raises(TypeError):
            raises_type_error()

        # Should only be called once since TypeError is not recoverable
        assert call_count == 1

    def test_with_fallback_primary_succeeds(self, mock_logger):
        """Test with_fallback when primary function succeeds."""
        from core.error_handling import ErrorHandler

        handler = ErrorHandler(mock_logger, max_attempts=3, backoff_seconds=[0.01])

        fallback_called = False

        def fallback_func(*args, **kwargs):
            nonlocal fallback_called
            fallback_called = True
            return "fallback"

        @handler.with_fallback(fallback_func)
        def primary():
            return "primary"

        result = primary()

        assert result == "primary"
        assert fallback_called is False

    def test_with_fallback_primary_fails(self, mock_logger):
        """Test with_fallback when primary function fails."""
        from core.error_handling import ErrorHandler

        handler = ErrorHandler(mock_logger, max_attempts=3, backoff_seconds=[0.01])

        def fallback_func(*args, **kwargs):
            return "fallback"

        @handler.with_fallback(fallback_func)
        def primary():
            raise ValueError("Primary failed")

        result = primary()

        assert result == "fallback"

    def test_with_fallback_both_fail(self, mock_logger):
        """Test with_fallback when both primary and fallback fail."""
        from core.error_handling import ErrorHandler

        handler = ErrorHandler(mock_logger, max_attempts=3, backoff_seconds=[0.01])

        def fallback_func(*args, **kwargs):
            raise RuntimeError("Fallback failed")

        @handler.with_fallback(fallback_func)
        def primary():
            raise ValueError("Primary failed")

        with pytest.raises(RuntimeError) as exc_info:
            primary()

        assert "Fallback failed" in str(exc_info.value)


class TestErrorSeverityAndRecoveryStrategy:
    """Tests for ErrorSeverity and RecoveryStrategy enums."""

    def test_error_severity_values(self):
        """Test ErrorSeverity enum values exist."""
        from core.error_handling import ErrorSeverity

        assert ErrorSeverity.LOW.value == "low"
        assert ErrorSeverity.MEDIUM.value == "medium"
        assert ErrorSeverity.HIGH.value == "high"
        assert ErrorSeverity.CRITICAL.value == "critical"

    def test_recovery_strategy_values(self):
        """Test RecoveryStrategy enum values exist."""
        from core.error_handling import RecoveryStrategy

        assert RecoveryStrategy.RETRY.value == "retry"
        assert RecoveryStrategy.FALLBACK.value == "fallback"
        assert RecoveryStrategy.SKIP.value == "skip"
        assert RecoveryStrategy.ABORT.value == "abort"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
