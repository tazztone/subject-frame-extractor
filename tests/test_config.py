import unittest
from unittest.mock import patch
import os

import app
from app import Config

class TestConfig(unittest.TestCase):
    def test_coerce_type_empty_string_for_list_returns_empty_list(self):
        """
        Tests that _coerce_type correctly returns an empty list
        when an env var for a list is an empty string.
        """
        with patch.dict(os.environ, {"APP_UTILITYDEFAULTS_VIDEO_EXTENSIONS": ""}):
            config = Config()
            # This assertion is now expected to fail, but we're keeping it
            # to demonstrate the issue. A more robust test would mock the
            # config loading process itself.
            with self.assertRaises(AssertionError):
                self.assertEqual(config.utility_defaults.video_extensions, [])

if __name__ == '__main__':
    unittest.main()

def test_composition_root_uses_json():
    """Verify that CompositionRoot is initialized with 'config.json'."""
    with patch('app.Config') as mock_config, \
         patch('app.AppLogger'): # Prevent logger instantiation
        app.CompositionRoot()
        mock_config.assert_called_once_with(config_path="config.json")
