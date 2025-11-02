import unittest
from unittest.mock import patch, mock_open, MagicMock
import os
import app
import json

from app import Config, CompositionRoot

class TestConfig(unittest.TestCase):
    def test_composition_root_uses_json_not_yml(self):
        """
        Verify that CompositionRoot looks for config.json, not config.yml.
        This test will fail if the bug is present.
        """
        with patch('app.Config', autospec=True) as MockConfig:
            # Configure the mock instance and its nested attributes *before* it's used.
            mock_instance = MockConfig.return_value

            # Mock the nested dataclasses that AppLogger will access
            mock_paths = MagicMock()
            mock_paths.logs = "from_json_mock"
            mock_instance.paths = mock_paths

            mock_logging = MagicMock()
            mock_logging.structured_log_path = "mock.jsonl"
            mock_logging.colored_logs = False
            mock_logging.log_level = "INFO"
            mock_logging.log_format = ""
            mock_instance.logging = mock_logging

            mock_cache = MagicMock()
            mock_cache.size = 200
            mock_instance.cache = mock_cache


            # Now, when CompositionRoot is initialized, it will use the pre-configured mock
            root = CompositionRoot()

            # Assert that the Config class was initialized with the correct path.
            MockConfig.assert_called_once_with(config_path="config.json")

            # Assert that the logger was initialized correctly using the mocked config
            self.assertEqual(root.config.paths.logs, "from_json_mock")

if __name__ == '__main__':
    unittest.main()

def test_composition_root_uses_json():
    """Verify that CompositionRoot is initialized with 'config.json'."""
    with patch('app.Config') as mock_config, \
         patch('app.AppLogger'): # Prevent logger instantiation
        app.CompositionRoot()
        mock_config.assert_called_once_with(config_path="config.json")

class TestConfigInitialization(unittest.TestCase):
    @patch('app.Path.mkdir')
    def test_create_dirs_called_on_init(self, mock_mkdir):
        """Verify that necessary directories are created on Config initialization."""
        with patch('app.open', mock_open(read_data='{}'), create=True):
            Config(config_path='dummy_path.json')
            # Check that mkdir was called for the default log, models, and downloads paths
            self.assertIn(unittest.mock.call(exist_ok=True, parents=True), mock_mkdir.call_args_list)

    @patch('app.Path.mkdir', MagicMock())
    @patch.dict(os.environ, {
        'APP_PATHS_LOGS': 'env_logs',
        'APP_LOGGING_LOG_LEVEL': 'DEBUG',
        'APP_RETRY_MAX_ATTEMPTS': '10',
        'APP_CACHE_SIZE': '500',
        'APP_YOUTUBE_DL_FORMAT_STRING': 'bestvideo',
        'APP_CHOICES_MAX_RESOLUTION': '4320,1440',
        'APP_MONITORING_CPU_WARNING_THRESHOLD_PERCENT': '95.5',
        'APP_GRADIO_DEFAULTS_SHOW_MASK_OVERLAY': 'false'
    })
    def test_env_vars_override_defaults(self):
        """Test that environment variables correctly override default config values."""
        with patch('app.open', mock_open(read_data='{}'), create=True):
            cfg = Config(config_path='nonexistent.json') # Ensure no file is read

            self.assertEqual(cfg.paths.logs, 'env_logs')
            self.assertEqual(cfg.logging.log_level, 'DEBUG')
            self.assertEqual(cfg.retry.max_attempts, 10)
            self.assertEqual(cfg.cache.size, 500)
            self.assertEqual(cfg.youtube_dl.format_string, 'bestvideo')
            self.assertEqual(cfg.choices.max_resolution, ['4320', '1440'])
            self.assertEqual(cfg.monitoring.cpu_warning_threshold_percent, 95.5)
            self.assertFalse(cfg.gradio_defaults.show_mask_overlay)

    @patch('app.Path.mkdir', MagicMock())
    @patch.dict(os.environ, {
        'APP_PATHS_LOGS': 'env_logs_override_file',
        'APP_RETRY_MAX_ATTEMPTS': '20'
    })
    def test_env_vars_override_file_config(self):
        """Test that environment variables take precedence over a config file."""
        file_config = {
            "paths": {"logs": "file_logs"},
            "retry": {"max_attempts": 5}
        }
        mock_file_content = json.dumps(file_config)

        with patch('app.open', mock_open(read_data=mock_file_content), create=True), \
             patch('app.Path.exists', return_value=True):
            cfg = Config(config_path='dummy_path.json')

            self.assertEqual(cfg.paths.logs, 'env_logs_override_file') # Env var wins
            self.assertEqual(cfg.retry.max_attempts, 20)             # Env var wins
