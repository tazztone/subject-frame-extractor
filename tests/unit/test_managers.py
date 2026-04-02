from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from core.managers import (
    ModelRegistry,
    ThumbnailManager,
    VideoManager,
    get_face_analyzer,
    get_face_landmarker,
    initialize_analysis_models,
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
        config.thumbnail_cache_max_mb = 10
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
        config.downloads_dir = "/tmp/downloads"
        config.monitoring_memory_critical_threshold_mb = 16384
        config.monitoring_gpu_memory_critical_threshold_mb = 8192
        config.monitoring_cpu_warning_threshold_percent = 90.0
        config.quality_weights_sharpness = 25
        config.quality_weights_edge_strength = 15
        config.quality_weights_contrast = 15
        config.quality_weights_brightness = 10
        config.quality_weights_entropy = 15
        config.quality_weights_niqe = 20
        config.quality_niqe_offset = 10.0
        config.quality_niqe_scale_factor = 10.0
        config.quality_contrast_clamp = 2.0
        return config

    # --- ThumbnailManager Tests ---

    def test_thumbnail_manager_init(self, mock_logger, mock_config):
        tm = ThumbnailManager(mock_logger, mock_config)
        assert tm.max_bytes == 10 * 1024 * 1024
        assert isinstance(tm.cache, dict)

    @patch("core.managers.thumbnails.Image.open", create=True)
    @patch("pathlib.Path.exists", return_value=True)
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

    @patch("pathlib.Path.exists", return_value=False)
    def test_thumbnail_manager_get_not_exist(self, mock_exists, mock_logger, mock_config):
        tm = ThumbnailManager(mock_logger, mock_config)
        assert tm.get(Path("nonexistent.webp")) is None

    def test_thumbnail_manager_cleanup(self, mock_logger, mock_config):
        mock_config.cache_size = 3
        mock_config.thumbnail_cache_max_mb = 1.0  # 1MB
        mock_config.cache_cleanup_threshold = 0.9  # 0.9MB threshold
        tm = ThumbnailManager(mock_logger, mock_config)

        # Fill cache
        p1 = Path("1.webp")
        _ = Path("2.webp")
        p3 = Path("3.webp")

        # 600KB each, total 1.2MB > 0.9MB threshold
        data = np.zeros((1000, 200, 3), dtype=np.uint8)  # 600,000 bytes
        tm.cache[p1] = data
        tm.current_bytes = data.nbytes

        # Threshold check: current_bytes > max_bytes * cleanup_threshold

        # Threshold check: current_bytes > max_bytes * cleanup_threshold
        # Let's mock _cleanup_old_entries to verify call
        with patch.object(tm, "_cleanup_old_entries") as mock_clean:
            with (
                patch("PIL.Image.open", create=True) as mock_open_pil,
                patch("pathlib.Path.exists", return_value=True),
            ):
                mock_img = MagicMock()
                mock_img.convert.return_value = data  # 600KB
                mock_open_pil.return_value.__enter__.return_value = mock_img

                # Use a low max_count to ensure COUNT-based trigger for _cleanup_old_entries
                tm.max_count = 1
                tm.cleanup_threshold = 0.5
                # Adding p3 will make len(cache)=1, which is > 1*0.5
                tm.get(p3)
                mock_clean.assert_called()

    def test_thumbnail_manager_eviction(self, mock_logger, mock_config):
        mock_config.thumbnail_cache_max_mb = 0.0001  # Very small ~100 bytes
        tm = ThumbnailManager(mock_logger, mock_config)

        p1, _, p3 = Path("1"), Path("2"), Path("3")
        data = np.zeros((50, 50, 3), dtype=np.uint8)  # 7500 bytes

        # Manually populate
        tm.cache[p1] = data
        tm.current_bytes = data.nbytes

        with patch("core.managers.thumbnails.Image.open", create=True), patch("pathlib.Path.exists", return_value=True):
            mock_img = MagicMock()
            mock_img.convert.return_value = data
            with patch(
                "PIL.Image.open", return_value=MagicMock(__enter__=lambda x: mock_img, __exit__=lambda *args: None)
            ):
                # This will add p3, which is 7500 bytes. Total 15000.
                # Threshold is 100 bytes. So it MUST evict p1.
                tm.get(p3)

        assert p3 in tm.cache
        assert p1 not in tm.cache

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
        loader.assert_called_once()  # Should not be called again

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

    @patch("core.managers.registry.ModelRegistry._load_tracker_impl")
    def test_get_tracker_success(self, mock_load, mock_logger, mock_config):
        mock_load.return_value = "mock_tracker"
        registry = ModelRegistry(mock_logger)
        registry._models = {}

        tracker = registry.get_tracker("sam3", "/tmp/models", "agent", (1,), mock_config)

        mock_load.assert_called_once()
        assert tracker == "mock_tracker"

    @patch("core.managers.registry.ModelRegistry._load_tracker_impl")
    @patch("core.managers.registry.torch.cuda.empty_cache", create=True)
    @patch("core.managers.registry.torch.cuda.is_available", return_value=True, create=True)
    def test_get_tracker_oom_fallback(self, mock_cuda, mock_empty, mock_load, mock_logger, mock_config):
        registry = ModelRegistry(mock_logger)

        # First call raises OOM
        mock_load.side_effect = [RuntimeError("out of memory"), "mock_tracker_cpu"]

        tracker = registry.get_tracker("sam3", "/tmp/models", "agent", (1,), mock_config)

        # Should have tried twice: once cuda, once cpu
        assert mock_load.call_count == 2
        assert mock_load.call_args_list[0][0][4] == "cuda"
        assert mock_load.call_args_list[1][0][4] == "cpu"

        assert tracker == "mock_tracker_cpu"
        assert registry.runtime_device_override == "cpu"

    # --- VideoManager Tests ---

    def test_video_manager_prepare_local(self, mock_config):
        vm = VideoManager("test.mp4", mock_config)
        with patch("core.managers.video.validate_video_file", return_value="test.mp4") as mock_val:
            path = vm.prepare_video(MagicMock())
            mock_val.assert_called_once()
            assert path == "test.mp4"

    @patch("core.managers.video.ytdlp.YoutubeDL")
    def test_video_manager_prepare_youtube(self, mock_ytdl, mock_config, mock_logger):
        vm = VideoManager("https://youtube.com/watch?v=123", mock_config)

        mock_instance = mock_ytdl.return_value.__enter__.return_value
        mock_instance.extract_info.return_value = {}
        mock_instance.prepare_filename.return_value = "downloaded.mp4"

        with patch("core.managers.video.validate_video_file", return_value="downloaded.mp4"):
            path = vm.prepare_video(mock_logger)

        assert path == "downloaded.mp4"
        mock_instance.extract_info.assert_called_once()

    def test_video_manager_invalid_inputs(self, mock_config, mock_logger):
        # Invalid URL/File
        vm = VideoManager("invalid_file.mp4", mock_config)
        # Assuming validate_video_file raises FileNotFoundError
        with patch("core.managers.video.validate_video_file", side_effect=FileNotFoundError):
            with pytest.raises(FileNotFoundError):
                vm.prepare_video(mock_logger)

    @patch("core.managers.video.DownloadError", new_callable=lambda: type("DownloadError", (Exception,), {}))
    @patch("core.managers.video.ytdlp")
    def test_video_manager_youtube_error(self, mock_ytdlp_module, mock_download_error_cls, mock_config, mock_logger):
        # Setup the YoutubeDL context manager mock
        mock_ctx = mock_ytdlp_module.YoutubeDL.return_value.__enter__.return_value
        mock_ctx.extract_info.side_effect = mock_download_error_cls("Failed")

        vm = VideoManager("https://youtube.com/watch?v=bad", mock_config)

        with pytest.raises(RuntimeError) as excinfo:
            vm.prepare_video(mock_logger)

        assert "Download failed" in str(excinfo.value)

    @patch("cv2.VideoCapture")
    def test_get_video_info(self, mock_cap):
        instance = mock_cap.return_value
        instance.isOpened.return_value = True

        import cv2

        def get_side_effect(prop):
            if prop == cv2.CAP_PROP_FRAME_WIDTH:
                return 1920
            if prop == cv2.CAP_PROP_FRAME_HEIGHT:
                return 1080
            if prop == cv2.CAP_PROP_FPS:
                return 30.0
            if prop == cv2.CAP_PROP_FRAME_COUNT:
                return 100
            return 0

        instance.get.side_effect = get_side_effect

        info = VideoManager.get_video_info("test.mp4")

        assert info["fps"] == 30.0
        assert info["width"] == 1920
        assert info["height"] == 1080
        assert info["frame_count"] == 100

    # --- Face Model Tests ---

    @patch("core.managers.face.vision.FaceLandmarker")
    @patch("core.managers.face.python.BaseOptions")
    @patch("core.managers.face.vision.FaceLandmarkerOptions")
    def test_get_face_landmarker(self, mock_opts, mock_base, mock_cls, mock_logger):
        # Reset thread local if necessary (using a fresh thread is cleaner but let's try direct)
        # We need to access the thread_local object in the module.
        # But we can just test that it calls create_from_options

        detector = get_face_landmarker("model.task", mock_logger)
        mock_cls.create_from_options.assert_called()
        assert detector == mock_cls.create_from_options.return_value

    @patch("core.managers.model_loader.PersonDetector")
    @patch("core.managers.model_loader.get_face_analyzer")
    @patch("core.managers.model_loader.download_model")
    @patch("pathlib.Path.exists", return_value=True)
    @patch("pathlib.Path.is_file", return_value=True)
    @patch("core.managers.model_loader.get_face_landmarker")
    @patch("cv2.imread", return_value=np.zeros((100, 100, 3)))
    def test_initialize_analysis_models(
        self,
        mock_imread,
        mock_get_landmarker,
        mock_isfile,
        mock_exists,
        mock_download,
        mock_get_analyzer,
        mock_person_detector_cls,
        mock_config,
        mock_logger,
    ):
        params = MagicMock()
        params.compute_face_sim = True
        params.face_model_name = "buffalo_l"
        params.face_ref_img_path = "ref.jpg"
        params.person_detector_model = "YOLO26n"

        model_registry = MagicMock()

        mock_analyzer = MagicMock()
        mock_get_analyzer.return_value = mock_analyzer

        # Mock face detection result
        mock_face = MagicMock()
        mock_face.det_score = 0.9
        mock_face.normed_embedding = np.zeros(512)
        mock_analyzer.get.return_value = [mock_face]

        # Use a scoped patch instead of manual mutation to avoid cross-test pollution
        with patch("torch.cuda.is_available", return_value=False, create=True):
            models = initialize_analysis_models(params, mock_config, mock_logger, model_registry)

        assert models["face_analyzer"] == mock_analyzer
        assert models["ref_emb"] is not None
        assert "person_detector" in models
        mock_download.assert_called()  # Landmarker download

    @patch("insightface.app.FaceAnalysis")
    @patch("time.sleep", return_value=None)
    def test_get_face_analyzer_retry_logic(self, mock_sleep, mock_face_analysis_cls, mock_logger):
        # First attempt raises OOM, second attempt succeeds with CPU
        mock_instance_gpu = MagicMock()
        mock_instance_gpu.prepare.side_effect = RuntimeError("out of memory")

        mock_instance_cpu = MagicMock()

        mock_face_analysis_cls.side_effect = [mock_instance_gpu, mock_instance_cpu]

        registry = ModelRegistry(mock_logger)

        result = get_face_analyzer("buffalo_l", "/tmp", (640, 640), mock_logger, registry, device="cuda")

        assert result == mock_instance_cpu
        # Check that we tried to load GPU first, then CPU
        assert mock_instance_gpu.prepare.called
        assert mock_instance_cpu.prepare.called

    @patch("insightface.app.FaceAnalysis")
    @patch("time.sleep", return_value=None)
    def test_get_face_analyzer_cpu_fallback_failure(self, mock_sleep, mock_face_analysis_cls, mock_logger):
        # GPU fails with OOM, then CPU also fails
        mock_instance_gpu = MagicMock()
        mock_instance_gpu.prepare.side_effect = RuntimeError("out of memory")

        mock_instance_cpu = MagicMock()
        mock_instance_cpu.prepare.side_effect = RuntimeError("CPU failure")

        mock_face_analysis_cls.side_effect = [mock_instance_gpu, mock_instance_cpu]

        registry = ModelRegistry(mock_logger)

        with pytest.raises(RuntimeError, match="CPU fallback also failed"):
            get_face_analyzer("buffalo_l", "/tmp", (640, 640), mock_logger, registry, device="cuda")

    @patch("core.managers.face.vision.FaceLandmarker")
    def test_get_face_landmarker_failure(self, mock_landmarker, mock_logger):
        mock_landmarker.create_from_options.side_effect = Exception("MediaPipe fail")
        # Ensure we don't use the thread local if it exists
        import core.managers.face

        if hasattr(core.managers.face.thread_local, "face_landmarker_instance"):
            del core.managers.face.thread_local.face_landmarker_instance

        with pytest.raises(RuntimeError, match="Could not initialize MediaPipe face landmarker model"):
            get_face_landmarker("model.task", mock_logger)

    @patch("core.managers.thumbnails.Image.open")
    @patch("pathlib.Path.exists", return_value=True)
    def test_thumbnail_manager_corrupt_file(self, mock_exists, mock_open, mock_logger, mock_config):
        tm = ThumbnailManager(mock_logger, mock_config)
        mock_open.side_effect = IOError("Corrupt file")

        result = tm.get("test.webp")
        assert result is None
        # Should verify warning log
        mock_logger.warning.assert_called()

    @patch("core.managers.model_loader.lpips.LPIPS")
    def test_get_lpips_metric(self, mock_lpips):
        mock_instance = mock_lpips.return_value
        mock_instance.to.return_value = mock_instance

        from core.managers.model_loader import get_lpips_metric

        metric = get_lpips_metric(model_name="alex", device="cpu")

        mock_lpips.assert_called_once_with(net="alex")
        mock_instance.to.assert_called_once_with("cpu")
        assert metric == mock_instance

    def test_model_registry_basic_load(self, mock_logger):
        """Test basic ModelRegistry load (retry logic is in get_tracker/etc, not get_or_load)."""
        registry = ModelRegistry(logger=mock_logger)
        mock_loader = MagicMock(return_value="Success")

        # get_or_load only handles locking and caching, not retries itself
        result = registry.get_or_load("test_model", mock_loader)

        assert result == "Success"
        mock_loader.assert_called_once()

        # Second call should use cache
        result2 = registry.get_or_load("test_model", mock_loader)
        assert result2 == "Success"
        mock_loader.assert_called_once()

    def test_thumbnail_manager_eviction_logic(self, tmp_path, mock_logger):
        """Test LRU eviction in ThumbnailManager."""
        from core.config import Config

        config = Config()
        config.cache_size = 2  # Small size for testing
        manager = ThumbnailManager(mock_logger, config)

        # Fake images
        img1 = np.zeros((10, 10, 3), dtype=np.uint8)

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
