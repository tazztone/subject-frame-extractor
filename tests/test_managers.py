import pytest
from unittest.mock import MagicMock, patch, ANY
from pathlib import Path
import numpy as np
from core.managers import ThumbnailManager, ModelRegistry, VideoManager
from PIL import Image

class TestThumbnailManager:
    @pytest.fixture
    def manager(self, mock_logger, mock_config, tmp_path):
        mock_config.cache_size = 10
        mock_config.cache_cleanup_threshold = 0.8
        mock_config.cache_eviction_factor = 0.5
        return ThumbnailManager(mock_logger, mock_config)

    @patch('PIL.Image.open')
    def test_get_existing(self, mock_open_img, manager, tmp_path):
        p = tmp_path / "thumb.webp"
        p.touch()

        real_img = Image.new('RGB', (10, 10))
        mock_open_img.return_value.__enter__.return_value = real_img

        res = manager.get(p)
        assert res is not None
        assert p in manager.cache

        res2 = manager.get(p)
        assert res2 is res

    def test_get_missing(self, manager, tmp_path):
        p = tmp_path / "missing.webp"
        assert manager.get(p) is None

    def test_clear_cache(self, manager):
        manager.cache["test"] = "data"
        manager.clear_cache()
        assert len(manager.cache) == 0

    @patch('PIL.Image.open')
    def test_cleanup_old_entries(self, mock_open_img, manager, tmp_path):
        manager.max_size = 2

        p1 = tmp_path / "1.webp"
        p2 = tmp_path / "2.webp"
        p3 = tmp_path / "3.webp"
        p1.touch(); p2.touch(); p3.touch()

        real_img = Image.new('RGB', (10, 10))
        mock_open_img.return_value.__enter__.return_value = real_img

        manager.get(p1)
        manager.get(p2)
        manager.get(p3)

        assert len(manager.cache) <= 2
        assert p3 in manager.cache

class TestModelRegistry:
    @pytest.fixture
    def registry(self, mock_logger):
        return ModelRegistry(mock_logger)

    def test_get_or_load(self, registry):
        loader = MagicMock(return_value="model")
        assert registry.get_or_load("key", loader) == "model"
        loader.assert_called_once()
        assert registry.get_or_load("key", loader) == "model"
        loader.assert_called_once()

    def test_clear(self, registry):
        loader = MagicMock(return_value="model")
        registry.get_or_load("key", loader)
        registry.clear()
        # Access private attribute directly or check logging if mock_logger used
        assert len(registry._models) == 0

class TestVideoManager:
    @patch('cv2.VideoCapture')
    def test_get_video_info(self, mock_cap_cls):
        mock_cap = mock_cap_cls.return_value
        mock_cap.isOpened.return_value = True

        # Order: FPS, WIDTH, HEIGHT, COUNT
        mock_cap.get.side_effect = [30.0, 1920.0, 1080.0, 100.0]

        info = VideoManager.get_video_info("video.mp4")
        assert info['fps'] == 30.0
        assert info['frame_count'] == 100
        assert info['width'] == 1920
        assert info['height'] == 1080

    @patch('cv2.VideoCapture')
    def test_get_video_info_fail(self, mock_cap_cls):
        mock_cap = mock_cap_cls.return_value
        mock_cap.isOpened.return_value = False
        with pytest.raises(IOError):
            VideoManager.get_video_info("video.mp4")

    @patch('core.managers.validate_video_file')
    def test_prepare_video(self, mock_validate, mock_config):
        mock_config.default_max_resolution = "1080"
        vm = VideoManager("video.mp4", mock_config)
        path = vm.prepare_video(MagicMock())
        assert path == "video.mp4"
