
import pytest
import numpy as np
import threading
import torch
from unittest.mock import MagicMock, patch, ANY, call
from pathlib import Path
from core.managers import (
    ThumbnailManager,
    ModelRegistry,
    VideoManager,
    get_face_landmarker,
    get_face_analyzer,
    initialize_analysis_models
)

class TestManagers:

    @pytest.fixture
    def mock_logger(self):
        return MagicMock()

    @pytest.fixture
    def mock_config(self):
        config = MagicMock()
        config.cache_size = 10
        config.cache_cleanup_threshold = 0.8
        config.cache_eviction_factor = 0.5
        config.default_max_resolution = "1080"
        config.ytdl_output_template = "%(title)s.%(ext)s"
        config.ytdl_format_string = "bestvideo[height<={max_res}]+bestaudio/best[height<={max_res}]"
        config.models_dir = "/tmp/models"
        config.face_landmarker_url = "http://example.com/face_landmarker.task"
        config.face_landmarker_sha256 = "dummy_sha"
        config.retry_max_attempts = 1
        config.retry_backoff_seconds = (0.1,)
        config.user_agent = "test-agent"
        config.model_face_analyzer_det_size = (640, 640)
        config.sam3_checkpoint_url = "http://example.com/sam3.pt"
        config.sam3_checkpoint_sha256 = "dummy_sha_sam3"
        config.huggingface_token = "token"
        return config

    # --- ThumbnailManager Tests ---

    def test_thumbnail_manager_init(self, mock_logger, mock_config):
        tm = ThumbnailManager(mock_logger, mock_config)
        assert tm.max_size == 10
        assert isinstance(tm.cache, dict)

    @patch('core.managers.Image.open')
    @patch('pathlib.Path.exists', return_value=True)
    def test_thumbnail_manager_get_miss(self, mock_exists, mock_open, mock_logger, mock_config):
        tm = ThumbnailManager(mock_logger, mock_config)

        # Setup mock image
        mock_img = MagicMock()
        mock_img.convert.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_open.return_value.__enter__.return_value = mock_img

        path = Path("test.webp")
        result = tm.get(path)

        assert result is not None
        assert isinstance(result, np.ndarray)
        assert path in tm.cache
        mock_open.assert_called_once_with(path)

    def test_thumbnail_manager_get_hit(self, mock_logger, mock_config):
        tm = ThumbnailManager(mock_logger, mock_config)
        path = Path("test.webp")
        data = np.zeros((10, 10, 3), dtype=np.uint8)
        tm.cache[path] = data

        result = tm.get(path)
        assert result is data
        # Ensure path moved to end (LRU)
        assert list(tm.cache.keys())[-1] == path

    @patch('pathlib.Path.exists', return_value=False)
    def test_thumbnail_manager_get_not_exist(self, mock_exists, mock_logger, mock_config):
        tm = ThumbnailManager(mock_logger, mock_config)
        assert tm.get(Path("nonexistent.webp")) is None

    def test_thumbnail_manager_cleanup(self, mock_logger, mock_config):
        mock_config.cache_size = 3
        mock_config.cache_cleanup_threshold = 0.5 # Cleanup when > 1.5 items
        mock_config.cache_eviction_factor = 0.5

        tm = ThumbnailManager(mock_logger, mock_config)

        # Fill cache
        p1 = Path("1.webp")
        p2 = Path("2.webp")
        p3 = Path("3.webp")

        tm.cache[p1] = np.zeros((1,1))
        tm.cache[p2] = np.zeros((1,1))

        # Add 3rd item, should trigger cleanup logic check?
        # Threshold is max_size * cleanup_threshold = 3 * 0.5 = 1.5
        # Current size is 2. > 1.5. So next get should cleanup?
        # get() calls _cleanup_old_entries IF len > threshold.

        # Let's mock _cleanup_old_entries to verify call
        with patch.object(tm, '_cleanup_old_entries') as mock_clean:
             with patch('core.managers.Image.open'), patch('pathlib.Path.exists', return_value=True):
                 tm.get(p3)
                 mock_clean.assert_called_once()

    def test_thumbnail_manager_eviction(self, mock_logger, mock_config):
        mock_config.cache_size = 2
        tm = ThumbnailManager(mock_logger, mock_config)

        p1, p2, p3 = Path("1"), Path("2"), Path("3")

        # Manually populate to avoid IO
        tm.cache[p1] = 1
        tm.cache[p2] = 2

        # Add 3rd via manual insertion logic simulation or force add
        # The get method enforces max_size at the end.

        with patch('core.managers.Image.open'), patch('pathlib.Path.exists', return_value=True):
             # This will add p3. size becomes 3.
             # Then loop `while len > max_size: popitem(last=False)`

             # Mock Image return
             mock_img = MagicMock()
             mock_img.convert.return_value = 3
             with patch('PIL.Image.open', return_value=MagicMock(__enter__=lambda x: mock_img, __exit__=lambda *args: None)):
                tm.get(p3)

        assert len(tm.cache) == 2
        assert p3 in tm.cache
        assert p1 not in tm.cache # LRU (p1 was inserted first)

    # --- ModelRegistry Tests ---

    def test_model_registry_get_or_load(self, mock_logger):
        registry = ModelRegistry(mock_logger)
        loader = MagicMock(return_value="model")

        # First call
        res1 = registry.get_or_load("key", loader)
        assert res1 == "model"
        loader.assert_called_once()

        # Second call
        res2 = registry.get_or_load("key", loader)
        assert res2 == "model"
        loader.assert_called_once() # Should not be called again

    def test_model_registry_get_or_load_error(self, mock_logger):
        registry = ModelRegistry(mock_logger)
        loader = MagicMock(side_effect=RuntimeError("Fail"))

        with pytest.raises(RuntimeError):
            registry.get_or_load("key", loader)

        assert "key" not in registry._models

    def test_model_registry_clear(self, mock_logger):
        registry = ModelRegistry(mock_logger)
        registry._models["key"] = "val"
        registry.clear()
        assert len(registry._models) == 0

    @patch('core.managers.download_model')
    @patch('core.managers.SAM3Wrapper')
    @patch('torch.cuda.is_available', return_value=True)
    def test_get_tracker_success(self, mock_cuda, mock_wrapper, mock_download, mock_logger, mock_config):
        registry = ModelRegistry(mock_logger)

        # Mock Path.exists to return False, forcing download
        with patch('pathlib.Path.exists', return_value=False):
            tracker = registry.get_tracker("sam3", "/tmp/models", "agent", (1,), mock_config)

        mock_download.assert_called_once()
        mock_wrapper.assert_called_once_with(str(Path("/tmp/models/sam3.pt")), device='cuda')
        assert tracker == mock_wrapper.return_value

    @patch('core.managers.SAM3Wrapper')
    @patch('torch.cuda.is_available', return_value=True)
    def test_get_tracker_oom_fallback(self, mock_cuda, mock_wrapper, mock_logger, mock_config):
        registry = ModelRegistry(mock_logger)

        # First call raises OOM
        mock_wrapper.side_effect = [RuntimeError("out of memory"), MagicMock()]

        with patch('pathlib.Path.exists', return_value=True):
            tracker = registry.get_tracker("sam3", "/tmp/models", "agent", (1,), mock_config)

        # Should have tried twice: once cuda, once cpu
        assert mock_wrapper.call_count == 2
        mock_wrapper.assert_has_calls([
            call(str(Path("/tmp/models/sam3.pt")), device='cuda'),
            call(str(Path("/tmp/models/sam3.pt")), device='cpu')
        ])
        assert registry.runtime_device_override == 'cpu'

    # --- VideoManager Tests ---

    def test_video_manager_prepare_local(self, mock_config):
        vm = VideoManager("test.mp4", mock_config)
        with patch('core.managers.validate_video_file') as mock_val:
            path = vm.prepare_video(MagicMock())
            mock_val.assert_called_once()
            assert path == "test.mp4"

    @patch('core.managers.ytdlp.YoutubeDL')
    def test_video_manager_prepare_youtube(self, mock_ytdl, mock_config, mock_logger):
        vm = VideoManager("https://youtube.com/watch?v=123", mock_config)

        mock_instance = mock_ytdl.return_value.__enter__.return_value
        mock_instance.extract_info.return_value = {}
        mock_instance.prepare_filename.return_value = "downloaded.mp4"

        path = vm.prepare_video(mock_logger)

        assert path == "downloaded.mp4"
        mock_instance.extract_info.assert_called_once()

    @patch('cv2.VideoCapture')
    def test_get_video_info(self, mock_cap):
        instance = mock_cap.return_value
        instance.isOpened.return_value = True
        instance.get.side_effect = [30.0, 1920, 1080, 100] # FPS, W, H, Count

        info = VideoManager.get_video_info("test.mp4")

        assert info['fps'] == 30.0
        assert info['width'] == 1920
        assert info['height'] == 1080
        assert info['frame_count'] == 100

    # --- Face Model Tests ---

    @patch('core.managers.vision.FaceLandmarker')
    @patch('core.managers.python.BaseOptions')
    @patch('core.managers.vision.FaceLandmarkerOptions')
    def test_get_face_landmarker(self, mock_opts, mock_base, mock_cls, mock_logger):
        # Reset thread local if necessary (using a fresh thread is cleaner but let's try direct)
        # We need to access the thread_local object in the module.
        # But we can just test that it calls create_from_options

        detector = get_face_landmarker("model.task", mock_logger)
        mock_cls.create_from_options.assert_called()
        assert detector == mock_cls.create_from_options.return_value

    @patch('core.managers.get_face_analyzer')
    @patch('core.managers.download_model')
    @patch('pathlib.Path.exists', return_value=True)
    @patch('pathlib.Path.is_file', return_value=True)
    @patch('cv2.imread', return_value=np.zeros((100,100,3)))
    def test_initialize_analysis_models(self, mock_imread, mock_isfile, mock_exists, mock_download, mock_get_analyzer, mock_config, mock_logger):
        params = MagicMock()
        params.enable_face_filter = True
        params.face_model_name = "buffalo_l"
        params.face_ref_img_path = "ref.jpg"

        model_registry = MagicMock()

        mock_analyzer = MagicMock()
        mock_get_analyzer.return_value = mock_analyzer

        # Mock face detection result
        mock_face = MagicMock()
        mock_face.det_score = 0.9
        mock_face.normed_embedding = np.zeros(512)
        mock_analyzer.get.return_value = [mock_face]

        with patch('torch.cuda.is_available', return_value=False):
             models = initialize_analysis_models(params, mock_config, mock_logger, model_registry)

        assert models['face_analyzer'] == mock_analyzer
        assert models['ref_emb'] is not None
        mock_download.assert_called() # Landmarker download

