import pytest
from unittest.mock import MagicMock, patch, ANY
from pathlib import Path
import numpy as np
import torch
import shutil
from core.managers import ThumbnailManager, ModelRegistry, VideoManager, SAM3Wrapper
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

    @patch('core.managers.download_model')
    def test_get_tracker_success(self, mock_download, registry, mock_config, tmp_path):
        # Create dummy model file so it doesn't try to download
        models_dir = tmp_path / "models"
        models_dir.mkdir(exist_ok=True)
        (models_dir / "sam3.pt").touch()

        with patch('core.managers.SAM3Wrapper') as MockWrapper:
            tracker = registry.get_tracker(
                model_name="sam3",
                models_path=str(models_dir),
                user_agent="agent",
                retry_params=(3, (0.1,)), # Correct format for retry_params
                config=mock_config
            )
            assert tracker is not None
            MockWrapper.assert_called()
            mock_download.assert_not_called()

    @patch('core.managers.download_model')
    def test_get_tracker_download(self, mock_download, registry, mock_config, tmp_path):
        models_dir = tmp_path / "models"
        models_dir.mkdir(exist_ok=True)

        with patch('core.managers.SAM3Wrapper') as MockWrapper:
             registry.get_tracker(
                model_name="sam3",
                models_path=str(models_dir),
                user_agent="agent",
                retry_params=(3, (0.1,)),
                config=mock_config
            )
             mock_download.assert_called()


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

class TestSAM3Wrapper:
    @pytest.fixture
    def wrapper(self):
        # We must patch build_sam3_video_model in the module where it is imported
        with patch('sam3.model_builder.build_sam3_video_model') as mock_build:
            # Setup mock model structure
            mock_model = MagicMock()
            mock_model.tracker = MagicMock()
            mock_model.detector.backbone = "backbone"
            mock_build.return_value = mock_model

            # The wrapper instantiates immediately
            return SAM3Wrapper(checkpoint_path="dummy.pt")

    def test_init_video(self, wrapper):
        mock_state = MagicMock()
        wrapper.predictor.init_state.return_value = mock_state

        state = wrapper.init_video("video.mp4")

        assert state is mock_state
        wrapper.predictor.init_state.assert_called_with(video_path="video.mp4")
        assert wrapper.inference_state is mock_state

    def test_add_bbox_prompt(self, wrapper):
        # Setup mocks
        wrapper.inference_state = MagicMock()

        # Helper to create a proper mock that handles > 0.0 comparison
        mock_bool_tensor = MagicMock()
        mock_bool_tensor.cpu.return_value.numpy.return_value = np.ones((1, 100, 100), dtype=bool)

        mock_mask_tensor = MagicMock()
        # Define comparisons explicitly
        mock_mask_tensor.__gt__.return_value = mock_bool_tensor

        wrapper.predictor.add_new_points_or_box.return_value = (
            None, [1], None, [mock_mask_tensor]
        )

        mask = wrapper.add_bbox_prompt(0, 1, [0, 0, 10, 10], (100, 100))

        assert mask is not None
        assert isinstance(mask, np.ndarray)
        assert mask.shape == (100, 100)

    def test_add_bbox_prompt_no_init(self, wrapper):
        with pytest.raises(RuntimeError):
            wrapper.add_bbox_prompt(0, 1, [0,0,10,10], (100,100))

    def test_propagate(self, wrapper):
        wrapper.inference_state = MagicMock()

        # Helper to create a proper mock that handles > 0.0 comparison
        mock_bool_tensor = MagicMock()
        mock_bool_tensor.cpu.return_value.numpy.return_value = np.ones((1, 100, 100), dtype=bool)

        mock_mask_tensor = MagicMock()
        mock_mask_tensor.__gt__.return_value = mock_bool_tensor

        wrapper.predictor.propagate_in_video.return_value = iter([
            (0, [1], None, [mock_mask_tensor], None),
            (1, [1], None, [mock_mask_tensor], None)
        ])

        results = list(wrapper.propagate(start_idx=0))
        assert len(results) == 2
        assert results[0][0] == 0 # frame_idx
        assert results[1][0] == 1 # frame_idx

    def test_detect_objects(self, wrapper):
        wrapper.predictor.init_state.return_value = MagicMock()

        # Create a mask with a white box at 10,10 10x10
        mask_np = np.zeros((1, 100, 100), dtype=bool)
        mask_np[0, 10:20, 10:20] = True

        mock_bool_tensor = MagicMock()
        mock_bool_tensor.cpu.return_value.numpy.return_value = mask_np

        mock_mask_tensor = MagicMock()
        mock_mask_tensor.__gt__.return_value = mock_bool_tensor

        wrapper.predictor.add_new_points.return_value = (
            None, [1], None, [mock_mask_tensor]
        )

        img = np.zeros((100, 100, 3), dtype=np.uint8)

        with patch('tempfile.mkdtemp', return_value="/tmp/test"):
             with patch('PIL.Image.fromarray') as mock_img:
                with patch('shutil.rmtree'): # prevent actual deletion error
                    results = wrapper.detect_objects(img, "person")

        assert len(results) == 1
        assert results[0]['label'] == "person"
        # Bbox should be approx [10, 10, 20, 20]
        # cv2.boundingRect returns x, y, w, h. We return x, y, x+w, y+h
        assert results[0]['bbox'] == [10, 10, 20, 20]
