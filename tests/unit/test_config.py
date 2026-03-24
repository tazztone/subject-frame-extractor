import os
from unittest.mock import MagicMock, patch

import pytest

from core.config import Config


def test_config_defaults():
    """Test that Config initializes with expected default values."""
    config = Config()
    assert config.logs_dir == "logs"
    assert config.ffmpeg_thumbnail_quality == 80
    assert config.quality_weights_sharpness == 25
    assert "sharpness" in config.quality_weights
    assert config.quality_weights["sharpness"] == 25
    # SAM2 Migration
    assert config.default_tracker_model_name == "sam3"
    assert "sam2.1" in config.sam2_checkpoint_url


def test_config_env_overrides():
    """Test that environment variables correctly override default values."""
    with patch.dict(os.environ, {"APP_LOGS_DIR": "custom_logs", "APP_FFMPEG_THUMBNAIL_QUALITY": "95"}):
        config = Config()
        assert config.logs_dir == "custom_logs"
        assert config.ffmpeg_thumbnail_quality == 95


def test_config_has_sam2_checkpoint_url():
    """Test that sam2_checkpoint_url is present and points to a SAM2.1 model."""
    config = Config()
    assert hasattr(config, "sam2_checkpoint_url")
    assert "sam2.1" in config.sam2_checkpoint_url  # points to hiera-tiny


def test_config_default_tracker_is_sam3():
    """Test that the default tracker is now SAM3."""
    config = Config()
    assert config.default_tracker_model_name == "sam3"


def test_config_invalid_quality_weights():
    """Test that Config raises a validation error if all quality weights are zero."""
    with pytest.raises(ValueError, match="The sum of quality_weights cannot be zero"):
        Config(
            quality_weights_sharpness=0,
            quality_weights_edge_strength=0,
            quality_weights_contrast=0,
            quality_weights_brightness=0,
            quality_weights_entropy=0,
            quality_weights_niqe=0,
        )


def test_config_boundary_quality_weights():
    """Test extreme boundary values for quality weights."""
    # Single weight at 1.0, others at 0.0
    config = Config(
        quality_weights_sharpness=1.0,
        quality_weights_edge_strength=0,
        quality_weights_contrast=0,
        quality_weights_brightness=0,
        quality_weights_entropy=0,
        quality_weights_niqe=0,
    )
    assert config.quality_weights["sharpness"] == 1.0
    assert config.quality_weights["edge_strength"] == 0

    # Very large weight
    config = Config(quality_weights_sharpness=1000000)
    assert config.quality_weights["sharpness"] == 1000000

    # Negative weight should fail
    with pytest.raises(ValueError, match="Quality weights cannot be negative"):
        Config(quality_weights_sharpness=-1)


@patch("core.config.Path.mkdir")
@patch("core.config.Path.touch")
@patch("core.config.Path.unlink")
def test_config_path_validation(mock_unlink, mock_touch, mock_mkdir):
    """Test that Config correctly validates and creates directories on initialization."""
    cfg = Config(logs_dir="test_logs", models_dir="test_models", downloads_dir="test_downloads")
    cfg.validate()

    # Check that mkdir was called for each directory
    assert mock_mkdir.call_count >= 3
    # Check that touch was called to test writability
    assert mock_touch.call_count >= 3
    # Check that unlink was called to clean up the test file
    assert mock_unlink.call_count >= 3


@patch("core.config.Path.mkdir")
@patch("core.config.Path.touch")
def test_config_path_validation_failure(mock_touch, mock_mkdir):
    """Test that Config handles directory writability failures gracefully (prints warning)."""
    mock_touch.side_effect = PermissionError("No permission")

    with patch("builtins.print") as mock_print:
        cfg = Config(logs_dir="readonly_dir")
        cfg.validate()
        # Should have printed a warning
        assert mock_print.called
        # Verify the specific message
        mock_print.assert_any_call("WARNING: Directory readonly_dir is not writable.")


def test_config_json_source():
    """Test loading configuration from a JSON file."""
    import json

    from core.config import json_config_settings_source

    mock_data = {"ffmpeg_thumbnail_quality": 88}

    # Test file exists and valid JSON
    with patch("core.config.Path.is_file", return_value=True):
        with patch("builtins.open", MagicMock()):
            with patch("json.load", return_value=mock_data):
                result = json_config_settings_source()
                assert result == mock_data

    # Test file does not exist
    with patch("core.config.Path.is_file", return_value=False):
        result = json_config_settings_source()
        assert result == {}

    # Test invalid JSON
    with patch("core.config.Path.is_file", return_value=True):
        with patch("builtins.open", MagicMock()):
            with patch("json.load", side_effect=json.JSONDecodeError("msg", "doc", 0)):
                result = json_config_settings_source()
                assert result == {}


def test_quality_weights_property():
    """Test the quality_weights property helper."""
    config = Config(quality_weights_sharpness=10, quality_weights_niqe=50)
    weights = config.quality_weights
    assert weights["sharpness"] == 10
    assert weights["niqe"] == 50
    assert weights["contrast"] == 15  # default
