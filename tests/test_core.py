"""
Tests for core functionality - Config, Logger, Filtering, and Event validation.

Uses fixtures from conftest.py for mock setup.
"""
import pytest
from pydantic import ValidationError
from unittest.mock import MagicMock, patch
from pathlib import Path
import numpy as np

# Imports from refactored modules (mocks are applied in conftest.py)
from core.config import Config
from core.database import Database
from core.logger import AppLogger
from core.models import Scene, Frame, QualityConfig, _coerce
from core.filtering import apply_all_filters_vectorized
from ui.gallery_utils import auto_set_thresholds
from core.events import PreAnalysisEvent


class TestUtils:
    @pytest.mark.parametrize("value, to_type, expected", [
        ("True", bool, True),
        ("false", bool, False),
        ("1", bool, True),
        ("0", bool, False),
        ("yes", bool, True),
        ("no", bool, False),
        (True, bool, True),
        (False, bool, False),
        ("123", int, 123),
        (123, int, 123),
        ("123.45", float, 123.45),
        (123.45, float, 123.45),
        ("string", str, "string"),
    ])
    def test_coerce(self, value, to_type, expected):
        assert _coerce(value, to_type) == expected

    def test_coerce_invalid_raises(self):
        with pytest.raises(ValueError):
            _coerce("not-a-number", int)
        with pytest.raises(ValueError):
            _coerce("not-a-float", float)

    def test_config_init(self):
        mock_config_data = {}
        with patch('core.config.json_config_settings_source', return_value=mock_config_data):
            config = Config(logs_dir="init_logs")
        assert config.logs_dir == "init_logs"

    @patch('pathlib.Path.mkdir', MagicMock())
    @patch('pathlib.Path.touch', MagicMock())
    @patch('pathlib.Path.unlink', MagicMock())
    def test_validation_error(self):
        """Test that a validation error is raised for invalid config."""
        with pytest.raises(ValidationError):
            # quality_weights sum cannot be zero
            Config(
                quality_weights_sharpness=0, quality_weights_edge_strength=0,
                quality_weights_contrast=0, quality_weights_brightness=0,
                quality_weights_entropy=0, quality_weights_niqe=0
            )


class TestAppLogger:
    def test_app_logger_instantiation(self, mock_config):
        """Tests that the logger can be instantiated with a valid config."""
        try:
            AppLogger(config=mock_config, log_to_console=False, log_to_file=False)
        except Exception as e:
            pytest.fail(f"Logger instantiation with a config object failed: {e}")

    def test_auto_set_thresholds(self):
        per_metric_values = {'sharpness': list(range(10, 101, 10)), 'contrast': [1, 2, 3, 4, 5]}
        slider_keys = ['sharpness_min', 'sharpness_max', 'contrast_min']
        selected_metrics = list(per_metric_values.keys())
        updates = auto_set_thresholds(per_metric_values, 75, slider_keys, selected_metrics)
        assert updates['slider_sharpness_min']['value'] == 77.5
        assert updates['slider_contrast_min']['value'] == 4.0

    def test_apply_all_filters_with_face_and_mask(self, sample_frames_data, mock_config):
        """Verify filtering by face similarity and mask area."""
        filters = {
            "face_sim_enabled": True,
            "face_sim_min": 0.5,
            "mask_area_enabled": True,
            "mask_area_pct_min": 10.0,
        }
        kept, rejected, _, _ = apply_all_filters_vectorized(sample_frames_data, filters, mock_config)

        kept_filenames = {f['filename'] for f in kept}
        rejected_filenames = {f['filename'] for f in rejected}

        assert 'frame_01.png' in kept_filenames
        assert 'frame_04.png' in rejected_filenames  # face_sim too low
        assert 'frame_05.png' in rejected_filenames  # mask_area_pct too low

    def test_calculate_quality_metrics_with_niqe(self, mock_config):
        """Test quality metrics calculation including NIQE."""
        mock_niqe_metric = MagicMock()
        mock_niqe_metric.device.type = 'cpu'
        mock_niqe_metric.return_value = 5.0

        image_data = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        frame = Frame(image_data=image_data, frame_number=1)

        quality_config = QualityConfig(
            sharpness_base_scale=mock_config.sharpness_base_scale,
            edge_strength_base_scale=mock_config.edge_strength_base_scale,
            enable_niqe=True
        )

        with patch('core.models.torch.from_numpy') as mock_torch_from_numpy:
            mock_tensor = MagicMock()
            mock_tensor.to.return_value = mock_tensor
            mock_torch_from_numpy.return_value.float.return_value.permute.return_value.unsqueeze.return_value = mock_tensor
            frame.calculate_quality_metrics(image_data, quality_config, MagicMock(), niqe_metric=mock_niqe_metric, main_config=mock_config)

        assert frame.metrics.niqe_score > 0
        mock_niqe_metric.assert_called_once()
        assert frame.error is None


class TestPreAnalysisEvent:
    def test_face_ref_validation(self, tmp_path, mock_ui_state):
        """Test the custom validator for face_ref_img_path."""
        video_path = tmp_path / "video.mp4"
        video_path.touch()
        mock_ui_state['video_path'] = str(video_path)

        # Valid image file
        valid_img = tmp_path / "face.jpg"
        valid_img.touch()
        mock_ui_state['face_ref_img_path'] = str(valid_img)
        event = PreAnalysisEvent.model_validate(mock_ui_state)
        assert event.face_ref_img_path == str(valid_img)

        # Path is the same as the video
        mock_ui_state['face_ref_img_path'] = str(video_path)
        event = PreAnalysisEvent.model_validate(mock_ui_state)
        assert event.face_ref_img_path == ""

        # Path does not exist
        mock_ui_state['face_ref_img_path'] = "/non/existent.png"
        event = PreAnalysisEvent.model_validate(mock_ui_state)
        assert event.face_ref_img_path == ""

        # Path has invalid extension
        invalid_ext = tmp_path / "face.txt"
        invalid_ext.touch()
        mock_ui_state['face_ref_img_path'] = str(invalid_ext)
        event = PreAnalysisEvent.model_validate(mock_ui_state)
        assert event.face_ref_img_path == ""

        # Path is empty
        mock_ui_state['face_ref_img_path'] = ""
        event = PreAnalysisEvent.model_validate(mock_ui_state)
        assert event.face_ref_img_path == ""


if __name__ == "__main__":
    pytest.main([__file__])
