import unittest
from unittest.mock import patch
import os

from app import Config

class TestConfig(unittest.TestCase):
    @patch.dict(os.environ, {"APP_UTILITYDEFAULTS_VIDEO_EXTENSIONS": ""})
    def test_coerce_type_empty_string_for_list_returns_empty_list(self):
        """
        Tests that _coerce_type correctly returns an empty list
        when an env var for a list is an empty string.
        """
        config = Config()
        self.assertEqual(config.utility_defaults.video_extensions, [])

if __name__ == '__main__':
    unittest.main()
