import unittest
from unittest.mock import patch, mock_open, MagicMock
import os
import app

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
