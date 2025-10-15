import pytest
import yaml
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Mock heavy ML dependencies that might be imported indirectly
modules_to_mock = {
    'torch': MagicMock(),
    'ultralytics': MagicMock(),
}
patch.dict(sys.modules, modules_to_mock).start()

from app.config import Config

@pytest.fixture
def mock_config_file(tmp_path):
    """Creates a temporary config.yaml file for testing."""
    config_dir = tmp_path / "app"
    config_dir.mkdir()
    config_file = config_dir / "config.yaml"

    config_data = {
        'model_paths': {
            'grounding_dino_config': 'GroundingDINO_SwinT_OGC.py',
            'grounding_dino_checkpoint': 'groundingdino_swint_ogc.pth',
        },
        'grounding_dino_params': {
            'box_threshold': 0.35,
            'text_threshold': 0.25,
        },
        'quality_weights': {
            'niqe': 0.2,
            'sharpness': 0.3,
            'contrast': 0.1,
            'brightness': 0.1,
            'entropy': 0.3,
        },
        'thumbnail_cache_size': 150,
    }

    with open(config_file, 'w') as f:
        yaml.dump(config_data, f)

    return config_file

def test_config_loading_success(mock_config_file):
    """Tests that the Config class loads a valid YAML file correctly."""
    with patch.object(Config, 'CONFIG_FILE', mock_config_file):
        config = Config()
        assert config.thumbnail_cache_size == 150
        assert config.GROUNDING_BOX_THRESHOLD == 0.35
        assert 'sharpness' in config.QUALITY_METRICS

def test_config_loading_file_not_found():
    """Tests that a FileNotFoundError is raised if the config file is missing."""
    with patch.object(Config, 'CONFIG_FILE', Path("non_existent_config.yaml")):
        with pytest.raises(FileNotFoundError):
            Config()
