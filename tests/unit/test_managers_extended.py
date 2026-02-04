
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from core.managers import ModelRegistry, ThumbnailManager
from core.config import Config

class TestManagersExtended:

    def test_model_registry_basic_load(self):
        """Test basic ModelRegistry load (retry logic is in get_tracker/etc, not get_or_load)."""
        logger = MagicMock() # Use mock logger that accepts success
        registry = ModelRegistry(logger=logger)
        mock_loader = MagicMock(return_value="Success")

        # get_or_load only handles locking and caching, not retries itself
        result = registry.get_or_load("test_model", mock_loader)

        assert result == "Success"
        mock_loader.assert_called_once()

        # Second call should use cache
        result2 = registry.get_or_load("test_model", mock_loader)
        assert result2 == "Success"
        mock_loader.assert_called_once()

    def test_thumbnail_manager_eviction_logic(self, tmp_path):
        """Test LRU eviction in ThumbnailManager."""
        config = Config()
        config.cache_size = 2 # Small size for testing
        manager = ThumbnailManager(MagicMock(), config)

        # Fake images
        import numpy as np
        img1 = np.zeros((10,10,3), dtype=np.uint8)

        # Patch PIL.Image.open to return context manager that yields fake image
        with patch("PIL.Image.open") as mock_open:
            mock_img = MagicMock()
            mock_img.convert.return_value = img1
            mock_open.return_value.__enter__.return_value = mock_img

            (tmp_path / "1.jpg").touch()
            (tmp_path / "2.jpg").touch()
            (tmp_path / "3.jpg").touch()

            manager.get(tmp_path / "1.jpg")
            manager.get(tmp_path / "2.jpg")

            assert len(manager.cache) == 2

            # Access 1 to make it recently used
            manager.get(tmp_path / "1.jpg")

            # Add 3, should evict 2 (LRU)
            manager.get(tmp_path / "3.jpg")

            assert len(manager.cache) == 2
            assert Path(tmp_path / "1.jpg") in manager.cache
            assert Path(tmp_path / "3.jpg") in manager.cache
            assert Path(tmp_path / "2.jpg") not in manager.cache
