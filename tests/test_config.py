import unittest
from unittest.mock import patch, mock_open
from pathlib import Path
import os
import shutil
import yaml

# Add app to the python path
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.config import Config

class TestConfig(unittest.TestCase):

    def setUp(self):
        """Set up a clean environment for each test."""
        self.test_dir = Path(__file__).parent / "test_output"
        self.test_dir.mkdir(exist_ok=True)
        self.config_path = self.test_dir / "config.yaml"

        # Mock Config's DIRS to use the test directory
        self.patcher = patch.dict(Config.DIRS, {
            'logs': self.test_dir / "logs",
            'configs': self.test_dir,
            'models': self.test_dir / "models",
            'downloads': self.test_dir / "downloads"
        })
        self.patcher.start()

        # Update CONFIG_FILE to use the test directory
        Config.CONFIG_FILE = self.config_path

    def tearDown(self):
        """Clean up the test environment."""
        self.patcher.stop()
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_load_config_file_not_found(self):
        """Test that FileNotFoundError is raised if config.yaml is missing."""
        if self.config_path.exists():
            self.config_path.unlink()
        with self.assertRaises(FileNotFoundError):
            Config().load_config()

    def test_setup_directories_and_logger(self):
        """Test that all required directories are created."""
        Config.setup_directories_and_logger()
        for dir_path in Config.DIRS.values():
            self.assertTrue(dir_path.exists())
            self.assertTrue(dir_path.is_dir())