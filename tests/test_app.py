import pytest
import sys
import unittest
from unittest.mock import MagicMock, patch, mock_open
from pathlib import Path
import os
import yaml
import time
import numpy as np
import gradio as gr
import cv2
import datetime
import cv2
import datetime

# Mock heavy dependencies before they are imported by the app
mock_torch = MagicMock(name='torch')
mock_torch.__version__ = "2.0.0"
mock_torch.hub = MagicMock(name='torch.hub')
mock_torch.cuda = MagicMock(name='torch.cuda')
mock_torch.cuda.is_available.return_value = False

mock_torchvision = MagicMock(name='torchvision')
mock_torchvision.ops = MagicMock(name='torchvision.ops')
mock_torchvision.transforms = MagicMock(name='torchvision.transforms')
mock_torchvision.transforms.functional = MagicMock(name='torchvision.transforms.functional')

mock_insightface = MagicMock(name='insightface')
mock_insightface.app = MagicMock(name='insightface.app')

mock_imagehash = MagicMock()
mock_psutil = MagicMock(name='psutil')
mock_psutil.cpu_percent.return_value = 50.0
mock_psutil.virtual_memory.return_value = MagicMock(percent=50.0, available=1024*1024*1024)
mock_psutil.disk_usage.return_value = MagicMock(percent=50.0)
mock_process = mock_psutil.Process.return_value
mock_process.memory_info.return_value.rss = 100 * 1024 * 1024
mock_process.cpu_percent.return_value = 10.0


modules_to_mock = {
    'torch': mock_torch,
    'torch.hub': mock_torch.hub,
    'torchvision': mock_torchvision,
    'torchvision.ops': mock_torchvision.ops,
    'torchvision.transforms': mock_torchvision.transforms,
    'torchvision.transforms.functional': mock_torchvision.transforms.functional,
    # 'cv2': MagicMock(name='cv2'), # cv2 is now used in tests, so we don't mock it globally
    'insightface': mock_insightface,
    'insightface.app': mock_insightface.app,
    'onnxruntime': MagicMock(name='onnxruntime'),
    'grounding_dino': MagicMock(name='grounding_dino'),
    'grounding_dino.groundingdino': MagicMock(),
    'grounding_dino.groundingdino.util': MagicMock(),
    'grounding_dino.groundingdino.util.inference': MagicMock(),
    'DAM4SAM': MagicMock(name='DAM4SAM'),
    'DAM4SAM.dam4sam_tracker': MagicMock(name='DAM4SAM.dam4sam_tracker'),
    'ultralytics': MagicMock(name='ultralytics'),
    'GPUtil': MagicMock(getGPUs=lambda: [MagicMock(memoryUtil=0.5)]),
    'imagehash': mock_imagehash,
    'psutil': mock_psutil,
    'matplotlib': MagicMock(),
    'matplotlib.pyplot': MagicMock(),
    'scenedetect': MagicMock(),
    'yt_dlp': MagicMock(),
}

patch.dict(sys.modules, modules_to_mock).start()

# Now import the monolithic app
import app

# --- Mocks for Tests ---

@pytest.fixture
def test_config():
    """Provides a clean, default Config object for each test."""
    # We patch `_create_dirs` to avoid creating directories during tests
    with patch('app.Config._create_dirs'):
        yield app.Config(config_path=None)

@pytest.fixture
def sample_frames_data():
    return [
        {'filename': 'frame_01.png', 'phash': 'a'*16, 'metrics': {'sharpness_score': 50, 'contrast_score': 50}, 'face_sim': 0.8, 'mask_area_pct': 20},
        {'filename': 'frame_02.png', 'phash': 'a'*16, 'metrics': {'sharpness_score': 50, 'contrast_score': 50}, 'face_sim': 0.8, 'mask_area_pct': 20},
        {'filename': 'frame_03.png', 'phash': 'b'*16, 'metrics': {'sharpness_score': 5, 'contrast_score': 50}, 'face_sim': 0.8, 'mask_area_pct': 20},
        {'filename': 'frame_04.png', 'phash': 'c'*16, 'metrics': {'sharpness_score': 50, 'contrast_score': 50}, 'face_sim': 0.2, 'mask_area_pct': 20},
        {'filename': 'frame_05.png', 'phash': 'd'*16, 'metrics': {'sharpness_score': 50, 'contrast_score': 50}, 'face_sim': 0.8, 'mask_area_pct': 2},
        {'filename': 'frame_06.png', 'phash': 'e'*16, 'metrics': {'sharpness_score': 50, 'contrast_score': 50}, 'mask_area_pct': 20},
    ]

@pytest.fixture
def sample_scenes():
    # Add start_frame and end_frame to match the Scene dataclass structure
    return [
        {'shot_id': 1, 'start_frame': 0, 'end_frame': 100, 'status': 'pending', 'seed_result': {'details': {'mask_area_pct': 50}}, 'seed_metrics': {'best_face_sim': 0.9, 'score': 0.95}},
        {'shot_id': 2, 'start_frame': 101, 'end_frame': 200, 'status': 'pending', 'seed_result': {'details': {'mask_area_pct': 5}}, 'seed_metrics': {'best_face_sim': 0.8, 'score': 0.9}},
        {'shot_id': 3, 'start_frame': 201, 'end_frame': 300, 'status': 'pending', 'seed_result': {'details': {'mask_area_pct': 60}}, 'seed_metrics': {'best_face_sim': 0.4, 'score': 0.8}},
        {'shot_id': 4, 'start_frame': 301, 'end_frame': 400, 'status': 'pending', 'seed_result': {'details': {'mask_area_pct': 70}}, 'seed_metrics': {'score': 0.7}},
    ]

# --- Test Classes ---

class TestUtils:
    @pytest.mark.parametrize("value, to_type, expected", [
        ("True", bool, True),
        ("false", bool, False),
        ("1", bool, True),
        ("0", bool, False),
        ("yes", bool, True),
        ("no", bool, False),
        (True, bool, True),
        (False, bool, False),
        ("123", int, 123),
        (123, int, 123),
        ("123.45", float, 123.45),
        (123.45, float, 123.45),
        ("string", str, "string"),
    ])
    def test_coerce(self, value, to_type, expected):
        assert app._coerce(value, to_type) == expected

    def test_coerce_invalid_raises(self):
        with pytest.raises(ValueError):
            app._coerce("not-a-number", int)
        with pytest.raises(ValueError):
            app._coerce("not-a-float", float)

class TestConfig:
    def test_default_config_loading(self, test_config):
        """Verify that the Config class correctly loads default values."""
        assert test_config.paths.logs == "logs"
        assert test_config.quality_weights.sharpness == 25
        assert not test_config.ui_defaults.disable_parallel

    @patch('builtins.open', new_callable=mock_open, read_data="paths:\n  logs: 'custom_logs'\nquality_weights:\n  sharpness: 50")
    @patch('pathlib.Path.exists', return_value=True)
    def test_file_override(self, mock_exists, mock_file, test_config):
        """Verify that a config file overrides defaults."""
        with patch('app.Config._create_dirs'):
            config = app.Config(config_path="dummy_path.yml")
            assert config.paths.logs == "custom_logs"
            assert config.quality_weights.sharpness == 50
            # Check that a non-overridden value remains default
            assert config.quality_weights.contrast == 15

    @patch.dict(os.environ, {"APP_PATHS_LOGS": "env_logs", "APP_QUALITY_WEIGHTS_SHARPNESS": "75"})
    def test_env_var_override(self, test_config):
        """Verify that environment variables override defaults and file configs."""
        with patch('app.Config._create_dirs'):
            config = app.Config(config_path=None) # Ensure no file is loaded
            assert config.paths.logs == "env_logs"
            assert config.quality_weights.sharpness == 75
            assert isinstance(config.quality_weights.sharpness, int) # Type coercion
            assert config.quality_weights.contrast == 15 # Default value

    @patch('builtins.open', new_callable=mock_open, read_data="paths:\n  logs: 'file_logs'")
    @patch('pathlib.Path.exists', return_value=True)
    @patch.dict(os.environ, {"APP_PATHS_LOGS": "env_logs"})
    def test_precedence_env_over_file(self, mock_exists, mock_file, test_config):
        """Verify that environment variables have precedence over config files."""
        with patch('app.Config._create_dirs'):
            config = app.Config(config_path="dummy_path.yml")
            assert config.paths.logs == "env_logs"

class TestEnhancedLogging:
    def test_enhanced_logger_instantiation(self, test_config):
        """Tests that the logger can be instantiated with a valid config."""
        try:
            # The logger now requires a config object
            app.EnhancedLogger(config=test_config, log_to_console=False, log_to_file=False)
        except Exception as e:
            pytest.fail(f"Logger instantiation with a config object failed: {e}")

    def test_operation_context_timing(self, test_config):
        logger = app.EnhancedLogger(config=test_config, log_to_console=False, log_to_file=False)
        logger.logger.log = MagicMock()
        with patch('builtins.open', mock_open()):
            with logger.operation("test_operation", "test_component"):
                time.sleep(0.01)
            # Check that start and done messages were logged
            assert logger.logger.log.call_count == 2

class TestFilterLogic:
    def test_apply_all_filters_no_filters(self, sample_frames_data, test_config):
        kept, rejected, _, _ = app.apply_all_filters_vectorized(sample_frames_data, {}, test_config)
        assert len(kept) == len(sample_frames_data)

    def test_auto_set_thresholds(self):
        per_metric_values = {'sharpness': list(range(10, 101, 10)), 'contrast': [1, 2, 3, 4, 5]}
        slider_keys = ['sharpness_min', 'sharpness_max', 'contrast_min']
        updates = app.auto_set_thresholds(per_metric_values, 75, slider_keys)
        assert updates['slider_sharpness_min']['value'] == 77.5
        assert updates['slider_contrast_min']['value'] == 4.0

    def test_apply_all_filters_with_face_and_mask(self, sample_frames_data, test_config):
        """Verify filtering by face similarity and mask area."""
        filters = {
            "face_sim_enabled": True,
            "face_sim_min": 0.5,
            "mask_area_enabled": True,
            "mask_area_pct_min": 10.0,
        }
        kept, rejected, _, _ = app.apply_all_filters_vectorized(sample_frames_data, filters, test_config)

        kept_filenames = {f['filename'] for f in kept}
        rejected_filenames = {f['filename'] for f in rejected}

        assert 'frame_01.png' in kept_filenames
        assert 'frame_04.png' in rejected_filenames # face_sim too low
        assert 'frame_05.png' in rejected_filenames # mask_area_pct too low

    @patch('app._update_gallery')
    def test_on_filters_changed(self, mock_update_gallery, sample_frames_data, test_config):
        """Verify that on_filters_changed correctly calls the gallery update function."""
        mock_update_gallery.return_value = ("Status", gr.update(value=[]))
        slider_values = {'sharpness_min': 10.0}
        event = app.FilterEvent(
            all_frames_data=sample_frames_data,
            per_metric_values={'face_sim': [0.8], 'mask_area_pct': [20.0]},
            output_dir="/fake/dir",
            gallery_view="Kept Frames",
            show_overlay=True,
            overlay_alpha=0.5,
            require_face_match=False,
            dedup_thresh=-1,
            slider_values=slider_values
        )

        result = app.on_filters_changed(event, MagicMock(), test_config)

        mock_update_gallery.assert_called_once()
        assert "filter_status_text" in result
        assert "results_gallery" in result

    @patch('app.on_filters_changed')
    def test_reset_filters(self, mock_on_filters_changed, sample_frames_data, test_config):
        """Verify that resetting filters restores default values and updates the UI."""
        mock_on_filters_changed.return_value = {"filter_status_text": "Reset", "results_gallery": []}
        slider_keys = ['sharpness_min', 'sharpness_max']

        # This function is now part of the UI class, but we test the core logic
        result = app.reset_filters(
            all_frames_data=sample_frames_data,
            per_metric_values={},
            output_dir="/fake/dir",
            config=test_config,
            slider_keys=slider_keys,
            thumbnail_manager=MagicMock()
        )

        assert result['slider_sharpness_min']['value'] == test_config.filter_defaults.sharpness['default_min']
        assert result['slider_sharpness_max']['value'] == test_config.filter_defaults.sharpness['default_max']
        assert result['require_face_match_input']['value'] == test_config.ui_defaults.require_face_match
        mock_on_filters_changed.assert_called_once()


class TestQuality:
    def test_compute_entropy(self):
        # A uniform distribution histogram should have max entropy (8.0)
        # We normalize it in the function, so it should be close to 1.0
        hist = np.ones(256, dtype=np.uint64)
        assert np.isclose(app.compute_entropy(hist, 8.0), 1.0)

    def test_quality_metrics_small_mask_fallback(self, test_config):
        # Create a deterministic checkerboard pattern to guarantee non-zero sharpness
        c100 = np.zeros((100, 100), dtype=np.uint8)
        c100[::2, ::2] = 255; c100[1::2, 1::2] = 255
        image_data = np.stack([c100]*3, axis=-1)
        frame = app.Frame(image_data=image_data, frame_number=1)

        c50 = np.zeros((50, 50), dtype=np.uint8)
        c50[::2, ::2] = 255; c50[1::2, 1::2] = 255
        thumb_image_rgb = np.stack([c50]*3, axis=-1)
        small_mask = np.zeros((50, 50), dtype=np.uint8)
        small_mask[25, 25] = 255 # Mask is too small (1 pixel)

        mock_quality_config = app.QualityConfig(
            sharpness_base_scale=1.0,
            edge_strength_base_scale=1.0,
            enable_niqe=False
        )
        mock_logger = MagicMock()

        # This should not raise an exception, but fallback to full-frame analysis
        try:
            frame.calculate_quality_metrics(
                thumb_image_rgb, mock_quality_config, mock_logger,
                mask=small_mask, main_config=test_config
            )
        except ValueError:
            pytest.fail("calculate_quality_metrics raised ValueError on small mask, but should have fallen back.")

        # Ensure metrics were still calculated (not all zero)
        assert frame.metrics.sharpness_score != 0.0
        assert frame.metrics.quality_score != 0.0

class TestSceneLogic:
    def test_get_scene_status_text(self):
        assert app.get_scene_status_text([]) == "No scenes loaded."
        assert app.get_scene_status_text([{'status': 'included'}, {'status': 'excluded'}]) == "1/2 scenes included for propagation."

    @patch('app.save_scene_seeds')
    def test_toggle_scene_status(self, mock_save, sample_scenes):
        scenes, _, _ = app.toggle_scene_status(sample_scenes, 2, 'included', '/fake/dir', MagicMock())
        assert scenes[1]['status'] == 'included'
        mock_save.assert_called_once()

    @patch('app.save_scene_seeds')
    def test_apply_bulk_scene_filters(self, mock_save, sample_scenes):
        """Verify that bulk filters correctly include/exclude scenes."""
        scenes, _ = app.apply_bulk_scene_filters(
            scenes=sample_scenes,
            min_mask_area=10.0,
            min_face_sim=0.5,
            min_confidence=0.85,
            enable_face_filter=True,
            output_folder='/fake/dir',
            logger=MagicMock()
        )

        status_map = {s['shot_id']: s['status'] for s in scenes}
        assert status_map[1] == 'included'
        assert status_map[2] == 'excluded' # Mask area too low
        assert status_map[3] == 'excluded' # Face sim too low
        assert status_map[4] == 'excluded' # Confidence too low
        mock_save.assert_called_once()

    @patch('app.save_scene_seeds')
    @patch('app.get_person_detector')
    @patch('app.get_face_analyzer')
    @patch('app.SubjectMasker')
    @patch('app.ThumbnailManager')
    def test_apply_scene_overrides(self, mock_thumbnail_manager, mock_subject_masker, mock_get_face_analyzer, mock_get_person_detector, mock_save, sample_scenes, test_config):
        """Verify that applying a scene override re-computes the seed."""
        mock_masker_instance = mock_subject_masker.return_value
        mock_masker_instance.get_seed_for_frame.return_value = ([10, 10, 20, 20], {'type': 'recomputed'})
        # Set the frame_map directly on the mock instance
        mock_masker_instance.frame_map = {s['shot_id']: f"frame_{s['shot_id']}.webp" for s in sample_scenes}
        mock_get_person_detector.return_value = MagicMock()
        mock_get_face_analyzer.return_value = MagicMock()
        mock_thumbnail_manager.get.return_value = np.zeros((64, 64, 3), dtype=np.uint8)

        _, updated_scenes, msg = app.apply_scene_overrides(
            scenes_list=sample_scenes,
            selected_shot_id=1,
            prompt="new prompt",
            box_th=0.5,
            text_th=0.5,
            output_folder='/fake/dir',
            ana_ui_map_keys=['output_folder'], # Need to provide at least one key
            ana_input_components=['/fake/dir'],
            cuda_available=False,
            thumbnail_manager=mock_thumbnail_manager,
            config=test_config,
            logger=MagicMock()
        )

        assert "updated and saved" in msg
        assert updated_scenes[0]['seed_result']['details']['type'] == 'recomputed'
        mock_save.assert_called_once()


class TestUtils:
    def test_sanitize_filename(self, test_config):
        # The function now depends on a config object
        assert app.sanitize_filename("a/b\\c:d*e?f\"g<h>i|j.txt", config=test_config) == "a_b_c_d_e_f_g_h_i_j.txt"

    @patch('app.gc.collect')
    @patch('app.torch')
    def test_safe_resource_cleanup(self, mock_torch, mock_gc):
        mock_torch.cuda.is_available.return_value = True
        with app.safe_resource_cleanup(): pass
        mock_gc.assert_called_once()
        mock_torch.cuda.empty_cache.assert_called_once()

class TestVideo:
    @patch('app.VideoManager.get_video_info')
    @patch('app.run_ffmpeg_extraction')
    @patch('pathlib.Path.is_file', return_value=True)
    def test_extraction_pipeline_run(self, mock_is_file, mock_ffmpeg, mock_info, test_config):
        params = app.AnalysisParameters.from_ui(MagicMock(), test_config, source_path='/fake.mp4')
        logger = MagicMock()
        
        # Instantiate with all required arguments
        pipeline = app.EnhancedExtractionPipeline(
            config=test_config,
            logger=logger,
            params=params,
            progress_queue=MagicMock(),
            cancel_event=MagicMock()
        )
        pipeline.run()
        mock_ffmpeg.assert_called()

class TestModels:
    @patch('app.download_model')
    @patch('app.gdino_load_model')
    def test_get_grounding_dino_model_path_resolution(self, mock_gdino_load_model, mock_download):
        """
        Tests that get_grounding_dino_model correctly handles relative, absolute, and empty paths.
        """
        # Case 1: Relative path
        app.get_grounding_dino_model.cache_clear()
        relative_path = "Grounded-SAM-2/grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        app.get_grounding_dino_model(
            gdino_config_path=relative_path,
            gdino_checkpoint_path="models/groundingdino_swint_ogc.pth",
            models_path="models",
            grounding_dino_url="http://fake.url/model.pth",
            device="cpu"
        )
        mock_gdino_load_model.assert_called_once()
        passed_config_path = mock_gdino_load_model.call_args.kwargs['model_config_path']
        expected_path = app.project_root / relative_path
        assert Path(passed_config_path).is_absolute(), "Should resolve relative paths to absolute"
        assert Path(passed_config_path) == expected_path, "Should correctly join relative path with project root"

        # Case 2: Empty path (should use default from Config)
        mock_gdino_load_model.reset_mock()
        app.get_grounding_dino_model.cache_clear()
        app.get_grounding_dino_model(
            gdino_config_path="",  # Empty path
            gdino_checkpoint_path="models/groundingdino_swint_ogc.pth",
            models_path="models",
            grounding_dino_url="http://fake.url/model.pth",
            device="cpu"
        )
        mock_gdino_load_model.assert_called_once()
        passed_config_path_default = mock_gdino_load_model.call_args.kwargs['model_config_path']
        expected_default_path = app.project_root / app.Config.Paths.grounding_dino_config
        assert Path(passed_config_path_default).is_absolute(), "Should use an absolute path for the default"
        assert Path(passed_config_path_default) == expected_default_path, "Should fall back to the default config path"

class TestVideoManager:
    @patch('app.ytdlp')
    def test_prepare_video_youtube(self, mock_ytdlp, test_config):
        """Verify YouTube download logic is triggered for YouTube URLs."""
        source_path = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        video_manager = app.VideoManager(source_path, test_config)
        mock_yt_downloader = MagicMock()
        mock_ytdlp.YoutubeDL.return_value.__enter__.return_value = mock_yt_downloader
        mock_yt_downloader.extract_info.return_value = {}
        mock_yt_downloader.prepare_filename.return_value = "/fake/path/video.mp4"

        result = video_manager.prepare_video(MagicMock())

        mock_ytdlp.YoutubeDL.assert_called_once()
        mock_yt_downloader.extract_info.assert_called_once_with(source_path, download=True)
        assert result == "/fake/path/video.mp4"

    @patch('pathlib.Path.is_file', return_value=True)
    def test_prepare_video_local_file(self, mock_is_file, test_config):
        """Verify local file path is returned directly."""
        source_path = "/path/to/local/video.mp4"
        video_manager = app.VideoManager(source_path, test_config)

        result = video_manager.prepare_video(MagicMock())

        assert result == source_path

    @patch('pathlib.Path.is_file', return_value=False)
    def test_prepare_video_local_file_not_found(self, mock_is_file, test_config):
        """Verify FileNotFoundError is raised for non-existent local files."""
        source_path = "/path/to/nonexistent/video.mp4"
        video_manager = app.VideoManager(source_path, test_config)

        with pytest.raises(FileNotFoundError):
            video_manager.prepare_video(MagicMock())

    @patch('app.ytdlp')
    def test_prepare_video_youtube_download_failure(self, mock_ytdlp, test_config):
        """Verify that a download error from yt-dlp raises a RuntimeError."""
        source_path = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        video_manager = app.VideoManager(source_path, test_config)

        mock_yt_downloader = MagicMock()
        mock_ytdlp.YoutubeDL.return_value.__enter__.return_value = mock_yt_downloader
        # Simulate a download error
        mock_ytdlp.utils.DownloadError = Exception # Mock the exception class
        mock_yt_downloader.extract_info.side_effect = mock_ytdlp.utils.DownloadError("Video unavailable")

        with pytest.raises(RuntimeError, match="Download failed"):
            video_manager.prepare_video(MagicMock())


class TestFrame:
    def test_calculate_quality_metrics_no_mask(self, test_config):
        """Verify quality metrics are calculated correctly without a mask."""
        # Create a simple gradient image for predictable metrics
        image_data = np.zeros((100, 100, 3), dtype=np.uint8)
        image_data[:, :, 0] = np.tile(np.arange(100), (100, 1))  # Gradient
        frame = app.Frame(image_data=image_data, frame_number=1)

        quality_config = app.QualityConfig(
            sharpness_base_scale=test_config.sharpness_base_scale,
            edge_strength_base_scale=test_config.edge_strength_base_scale,
            enable_niqe=False
        )

        frame.calculate_quality_metrics(image_data, quality_config, MagicMock(), main_config=test_config)

        # Check that metrics are calculated and have plausible values
        assert frame.metrics.sharpness_score > 0
        assert frame.metrics.contrast_score > 0
        assert 0 <= frame.metrics.brightness_score <= 100
        assert frame.error is None

    @patch('app.pyiqa')
    @patch('app.torch.tensor')
    def test_calculate_quality_metrics_with_niqe(self, mock_torch_tensor, mock_pyiqa, test_config):
        """Verify that NIQE score is calculated when enabled."""
        mock_niqe_metric = MagicMock()
        mock_niqe_metric.return_value = 5.0  # Mock NIQE score
        mock_torch_tensor.return_value = 5.0
        mock_pyiqa.create_metric.return_value = mock_niqe_metric

        image_data = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        frame = app.Frame(image_data=image_data, frame_number=1)

        quality_config = app.QualityConfig(
            sharpness_base_scale=test_config.sharpness_base_scale,
            edge_strength_base_scale=test_config.edge_strength_base_scale,
            enable_niqe=True
        )

        frame.calculate_quality_metrics(image_data, quality_config, MagicMock(), niqe_metric=mock_niqe_metric, main_config=test_config)

        assert frame.metrics.niqe_score > 0
        mock_niqe_metric.assert_called_once()
        assert frame.error is None


class TestSeedSelector:
    @pytest.fixture
    def mock_seed_selector(self, test_config):
        """Provides a SeedSelector instance with mocked dependencies."""
        mock_face_analyzer = MagicMock()
        mock_person_detector = MagicMock()
        mock_gdino_model = MagicMock()
        mock_tracker = MagicMock()
        mock_logger = MagicMock()
        params = app.AnalysisParameters.from_ui(mock_logger, test_config)
        selector = app.SeedSelector(
            params=params,
            config=test_config,
            face_analyzer=mock_face_analyzer,
            reference_embedding=np.random.rand(512),
            person_detector=mock_person_detector,
            tracker=mock_tracker,
            gdino_model=mock_gdino_model,
            logger=mock_logger
        )
        # Attach mocks to the selector instance for easy access in tests
        selector.mock_face_analyzer = mock_face_analyzer
        selector.mock_person_detector = mock_person_detector
        selector.mock_gdino_model = mock_gdino_model
        return selector

    def test_select_seed_identity_first_success(self, mock_seed_selector):
        """Test 'Identity-First' when a matching face and body are found."""
        selector = mock_seed_selector
        selector.params.primary_seed_strategy = "👤 By Face"
        frame_rgb = np.zeros((200, 200, 3), dtype=np.uint8)

        with patch.object(selector, '_find_target_face', return_value=({'bbox': [50, 50, 70, 70]}, {'type': 'face_match'})) as mock_find_face, \
             patch.object(selector, '_get_yolo_boxes', return_value=[{'bbox': [40, 40, 100, 180], 'conf': 0.9, 'type': 'yolo'}]) as mock_get_yolo, \
             patch.object(selector, '_get_dino_boxes', return_value=([], {})) as mock_get_dino:

            box, details = selector.select_seed(frame_rgb)

            assert box is not None
            assert details['type'] == 'evidence_based_selection'
            mock_find_face.assert_called_once_with(frame_rgb)
            mock_get_yolo.assert_called_once()

    def test_select_seed_object_first_success(self, mock_seed_selector):
        """Test 'Object-First' when a DINO box is found."""
        selector = mock_seed_selector
        selector.params.primary_seed_strategy = "📝 By Text"
        selector.params.text_prompt = "a person"
        frame_rgb = np.zeros((200, 200, 3), dtype=np.uint8)

        dino_result = [{'bbox': [10, 10, 50, 50], 'conf': 0.9, 'label': 'person', 'type': 'dino'}]
        with patch.object(selector, '_get_dino_boxes', return_value=(dino_result, {'type': 'dino'})) as mock_get_dino, \
             patch.object(selector, '_get_yolo_boxes', return_value=[]) as mock_get_yolo:

            box, details = selector.select_seed(frame_rgb)

            assert box == [10, 10, 40, 40]  # xywh format
            assert details['type'] == 'dino'
            mock_get_dino.assert_called_once()

    @pytest.mark.parametrize("strategy, expected_box", [
        ("Largest Person", [0, 0, 180, 180]),  # The larger box
        ("Center-most Person", [80, 80, 40, 40]) # The center-most box
    ])
    def test_choose_person_by_strategy(self, mock_seed_selector, strategy, expected_box):
        """Test prominent person selection strategies."""
        selector = mock_seed_selector
        selector.params.primary_seed_strategy = "🧑‍🤝‍🧑 Find Prominent Person"
        selector.params.seed_strategy = strategy
        frame_rgb = np.zeros((200, 200, 3), dtype=np.uint8)

        yolo_boxes = [
            {'bbox': [0, 0, 180, 180], 'conf': 0.9, 'type': 'yolo'}, # Large, off-center
            {'bbox': [80, 80, 120, 120], 'conf': 0.8, 'type': 'yolo'}  # Smaller, center-most
        ]
        with patch.object(selector, '_get_yolo_boxes', return_value=yolo_boxes):
            box, details = selector.select_seed(frame_rgb)
            assert box == expected_box

    def test_select_seed_face_fallback_to_text(self, mock_seed_selector):
        """Test 'Face + Text Fallback' when no face is found."""
        selector = mock_seed_selector
        selector.params.primary_seed_strategy = "🔄 Face + Text Fallback"
        selector.params.text_prompt = "a dog"
        frame_rgb = np.zeros((200, 200, 3), dtype=np.uint8)

        dino_result = [{'bbox': [20, 20, 60, 60], 'conf': 0.8, 'label': 'dog', 'type': 'dino'}]
        # No face is found, but a DINO box is.
        with patch.object(selector, '_find_target_face', return_value=(None, {'error': 'no_matching_face'})) as mock_find_face, \
             patch.object(selector, '_get_dino_boxes', return_value=(dino_result, {'type': 'dino'})) as mock_get_dino:

            box, details = selector.select_seed(frame_rgb)

            assert box == [20, 20, 40, 40]
            assert details['type'] == 'dino'
            mock_find_face.assert_called_once()
            mock_get_dino.assert_called_once()

    def test_identity_first_fallback_to_expanded_box(self, mock_seed_selector):
        """Test _identity_first_seed fallback to an expanded face box."""
        selector = mock_seed_selector
        frame_rgb = np.zeros((200, 200, 3), dtype=np.uint8)

        # A face is found, but no corresponding body boxes.
        with patch.object(selector, '_find_target_face', return_value=({'bbox': [90, 90, 110, 110]}, {'type': 'face_match'})) as mock_find_face, \
             patch.object(selector, '_get_yolo_boxes', return_value=[]) as mock_get_yolo, \
             patch.object(selector, '_get_dino_boxes', return_value=([], {})) as mock_get_dino, \
             patch.object(selector, '_expand_face_to_body', return_value=[50, 50, 100, 100]) as mock_expand:

            box, details = selector._identity_first_seed(frame_rgb, selector.params)

            assert box == [50, 50, 100, 100]
            assert details['type'] == 'expanded_box_from_face'
            mock_expand.assert_called_once()


class TestAnalysisPipeline:
    @pytest.fixture
    def mock_analysis_pipeline(self, test_config, tmp_path):
        """Provides an AnalysisPipeline instance with mocked dependencies."""
        params = app.AnalysisParameters.from_ui(MagicMock(), test_config, output_folder=str(tmp_path))
        progress_queue = MagicMock()
        cancel_event = MagicMock()
        cancel_event.is_set.return_value = False  # Ensure the pipeline doesn't exit prematurely
        thumbnail_manager = MagicMock()
        logger = MagicMock()

        pipeline = app.AnalysisPipeline(
            config=test_config,
            logger=logger,
            params=params,
            progress_queue=progress_queue,
            cancel_event=cancel_event,
            thumbnail_manager=thumbnail_manager
        )

        # Create dummy files and directories
        (tmp_path / "thumbs").mkdir()
        (tmp_path / "frame_map.json").write_text("[1, 2, 3]")
        (tmp_path / "thumbs" / "frame_1.webp").touch()

        return pipeline

    @patch('app.get_face_analyzer')
    @patch('app.get_person_detector')
    @patch('app.SubjectMasker')
    @patch('app.create_frame_map', return_value={1: "frame_1.webp"})
    def test_run_full_analysis(self, mock_create_frame_map, mock_subject_masker, mock_get_person_detector, mock_get_face_analyzer, mock_analysis_pipeline):
        """Test the full analysis pipeline run."""
        pipeline = mock_analysis_pipeline
        pipeline.params.disable_parallel = True # Easier to test without threading
        scenes_to_process = [app.Scene(shot_id=0, start_frame=0, end_frame=2)]

        mock_masker_instance = mock_subject_masker.return_value
        mock_masker_instance.run_propagation.return_value = {} # No masks

        # Mock the loop that processes frames to avoid dealing with threads
        with patch.object(pipeline, '_run_analysis_loop', return_value=None) as mock_run_loop:
            # Create a dummy file to be found by the pipeline
            (pipeline.output_dir / "frame_map.json").write_text("[1, 2, 3]")

            result = pipeline.run_full_analysis(scenes_to_process)

            assert result.get('done', False), f"Pipeline failed, result: {result}"
            assert Path(result['metadata_path']).exists()
            mock_run_loop.assert_called_once_with(scenes_to_process)


class TestEnhancedAppUI:
    @pytest.fixture
    def mock_app_ui(self, test_config):
        """Provides an EnhancedAppUI instance with mocked dependencies."""
        progress_queue = MagicMock()
        cancel_event = MagicMock()
        cancel_event.is_set.return_value = False
        thumbnail_manager = MagicMock()
        logger = MagicMock()
        ui = app.EnhancedAppUI(
            config=test_config,
            logger=logger,
            progress_queue=progress_queue,
            cancel_event=cancel_event,
            thumbnail_manager=thumbnail_manager
        )
        return ui

    @patch('app.execute_extraction')
    def test_run_extraction_wrapper(self, mock_execute_extraction, mock_app_ui):
        """Test the extraction wrapper in the UI class."""
        mock_execute_extraction.return_value = iter([{"done": True, "log": "Success"}])
        args = [None] * len(mock_app_ui.ext_ui_map_keys)
        result = mock_app_ui.run_extraction_wrapper(*args)
        assert result['unified_log'] == "Success"
        mock_execute_extraction.assert_called_once()

    @patch('app.execute_pre_analysis')
    def test_run_pre_analysis_wrapper(self, mock_execute_pre_analysis, mock_app_ui):
        """Test the pre-analysis wrapper in the UI class."""
        mock_execute_pre_analysis.return_value = iter([{"done": True, "log": "Success", "scenes": []}])
        args = [None] * len(mock_app_ui.ana_ui_map_keys)
        result = mock_app_ui.run_pre_analysis_wrapper(*args)
        assert result['unified_log'] == "Success"
        mock_execute_pre_analysis.assert_called_once()

    @patch('app.gr.Button')
    def test_create_component(self, MockButton, mock_app_ui):
        """Tests that _create_component correctly creates a Gradio component."""
        MockButton.return_value = "mock_button"
        name = "test_button"
        comp_type = "button"
        kwargs = {'value': "Test"}
        component = mock_app_ui._create_component(name, comp_type, kwargs)
        assert mock_app_ui.components[name] == "mock_button"
        MockButton.assert_called_once_with(**kwargs)

    def test_on_select_for_edit(self, mock_app_ui, sample_scenes):
        """Tests the on_select_for_edit function for correct UI updates."""
        index_map = list(range(len(sample_scenes)))
        select_data = MagicMock(spec=gr.SelectData)
        select_data.index = 0
        outputs = mock_app_ui.on_select_for_edit(select_data, sample_scenes, "Kept", index_map, "/fake/dir")
        (scenes, status_text, gallery_update, new_index_map, selected_id,
         editor_status, prompt, box_thresh, text_thresh, accordion_update) = outputs
        assert selected_id == sample_scenes[0]['shot_id']
        assert "Editing Scene 1" in editor_status['value']
        assert accordion_update['open'] is True

class TestCompositionRoot:
    @patch('app.Config')
    @patch('app.EnhancedLogger')
    @patch('app.ThumbnailManager')
    @patch('app.Queue')
    @patch('app.threading.Event')
    def test_initialization(self, mock_event, mock_queue, mock_thumbnail_manager, mock_logger, mock_config):
        """Tests that CompositionRoot initializes its components correctly."""
        root = app.CompositionRoot()
        assert isinstance(root.get_config(), MagicMock)
        assert isinstance(root.get_logger(), MagicMock)
        assert isinstance(root.get_thumbnail_manager(), MagicMock)
        mock_logger.return_value.set_progress_queue.assert_called_once()

    @patch('app.CompositionRoot.get_app_ui')
    def test_cleanup(self, mock_get_app_ui):
        """Tests that the cleanup method calls necessary cleanup functions."""
        root = app.CompositionRoot()
        root.thumbnail_manager.cleanup = MagicMock()
        root.cancel_event.set = MagicMock()
        root.cleanup()
        root.thumbnail_manager.cleanup.assert_called_once()
        root.cancel_event.set.assert_called_once()

class TestErrorHandler:
    @pytest.fixture
    def error_handler(self, test_config):
        """Provides an ErrorHandler instance with a mock logger."""
        logger = MagicMock()
        return app.ErrorHandler(logger, test_config)

    def test_with_retry_success(self, error_handler):
        """Tests that the retry decorator returns the function's result on success."""
        @error_handler.with_retry()
        def successful_func():
            return "success"
        assert successful_func() == "success"
        error_handler.logger.warning.assert_not_called()

    def test_with_retry_failure(self, error_handler):
        """Tests that the retry decorator retries and then raises the exception."""
        mock_func = MagicMock(side_effect=ValueError("test error"))
        @error_handler.with_retry(max_attempts=2, backoff_seconds=[0.01])
        def failing_func():
            return mock_func()
        with pytest.raises(ValueError, match="test error"):
            failing_func()
        assert mock_func.call_count == 2
        error_handler.logger.warning.assert_called_once()
        error_handler.logger.error.assert_called_once()

    def test_with_fallback_success(self, error_handler):
        """Tests that the fallback decorator returns the primary function's result on success."""
        @error_handler.with_fallback(fallback_func=lambda: "fallback")
        def successful_func():
            return "primary"
        assert successful_func() == "primary"
        error_handler.logger.warning.assert_not_called()

    def test_with_fallback_failure(self, error_handler):
        """Tests that the fallback decorator returns the fallback function's result on failure."""
        @error_handler.with_fallback(fallback_func=lambda: "fallback")
        def failing_func():
            raise ValueError("primary failed")
        assert failing_func() == "fallback"
        error_handler.logger.warning.assert_called_once()

class TestSessionManagement:
    @pytest.fixture
    def mock_ui(self, test_config):
        """Provides a mock UI object for session management tests."""
        ui = MagicMock()
        ui.config = test_config
        ui.logger = MagicMock()
        ui.thumbnail_manager = MagicMock()
        return ui

    @patch('app.validate_session_dir', return_value=(Path('/fake/session'), None))
    @patch('pathlib.Path.exists', autospec=True)
    def test_execute_session_load_success(self, mock_exists, mock_validate, mock_ui):
        """Tests a successful session load."""
        mock_files = {
            str(Path('/fake/session/run_config.json')): '{"source_path": "/fake/video.mp4", "output_folder": "/fake/session"}',
            str(Path('/fake/session/scenes.json')): '[[0, 100], [101, 200]]'
        }
        def open_side_effect(path, *args, **kwargs):
            path_str = str(path)
            return mock_open(read_data=mock_files.get(path_str, '')).return_value
        def exists_side_effect(self):
            s = str(self)
            return s in mock_files or s == str(Path('/fake/session'))
        mock_exists.side_effect = exists_side_effect
        with patch('builtins.open', side_effect=open_side_effect):
            event = app.SessionLoadEvent(session_path='/fake/session')
            result = next(app.execute_session_load(mock_ui, event, mock_ui.logger, mock_ui.config, mock_ui.thumbnail_manager))
        assert "Successfully loaded session" in result['log']
        assert result['status'] == "... Session loaded. You can now proceed from where you left off."
        assert result['source_input']['value'] == "/fake/video.mp4"

class TestExport:
    @patch('app.subprocess.run')
    @patch('app.datetime')
    def test_export_cropping_logic(self, mock_datetime, mock_subprocess, tmp_path, test_config):
        # 1. Setup mock UI and paths
        mock_ui = MagicMock()
        mock_ui.logger = MagicMock()
        mock_ui.cancel_event = MagicMock()
        mock_ui.cancel_event.is_set.return_value = False
        mock_ui.config = test_config # Add the config attribute to the mock

        # Mock datetime to control the output folder name
        mock_datetime.now.return_value = datetime.datetime(2023, 1, 1, 12, 0, 0)

        # 2. Create dummy files and directories
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        masks_dir = output_dir / "masks"
        masks_dir.mkdir()
        video_path = tmp_path / "video.mp4"
        video_path.touch()

        # Create dummy image and mask data in memory
        dummy_frame_img = np.zeros((1080, 1920, 3), dtype=np.uint8)
        dummy_mask_img = np.zeros((1080, 1920), dtype=np.uint8)
        # Subject bbox: x=760, y=390, w=400, h=300 (4:3 AR)
        cv2.rectangle(dummy_mask_img, (760, 390), (1160, 690), 255, -1)

        # Save the necessary input files that the function reads
        cv2.imwrite(str(masks_dir / "frame_000001.png"), dummy_mask_img)
        (output_dir / "frame_map.json").write_text('[1]')

        # 3. Mock subprocess.run to simulate FFmpeg creating the exported frame
        def simulate_ffmpeg_export(*args, **kwargs):
            export_dir = tmp_path / "output_exported_20230101_120000"
            export_dir.mkdir(exist_ok=True)
            # The export function first calls ffmpeg to extract the full frame,
            # then renames it. The cropping logic reads this renamed file.
            exported_frame_path = export_dir / "frame_000001.png"
            cv2.imwrite(str(exported_frame_path), dummy_frame_img)
            return MagicMock(returncode=0, stderr="")

        mock_subprocess.side_effect = simulate_ffmpeg_export

        # 4. Setup frame metadata for the export event
        all_frames_data = [{'filename': 'frame_000001.png', 'mask_path': 'frame_000001.png'}]

        event = app.ExportEvent(
            all_frames_data=all_frames_data,
            output_dir=str(output_dir),
            video_path=str(video_path),
            enable_crop=True,
            crop_ars="16:9,1:1",
            crop_padding=10, # 10% padding
            filter_args={}
        )

        # 5. Call the export function
        app.EnhancedAppUI.export_kept_frames(mock_ui, event)

        # 6. Assertions
        mock_subprocess.assert_called_once()
        crop_dir = tmp_path / "output_exported_20230101_120000" / "cropped"
        assert crop_dir.exists()

        # Logic Check:
        # Bbox: w=400, h=300 (AR=1.33). Padding: 10% -> padded_w=440, padded_h=330.
        # AR 16:9 (r=1.77): h_r=max(330, 440/r)=330, w_r=330*r=586.6. Area=193600. Diff=0.44
        # AR 1:1 (r=1.0): h_r=max(330, 440/r)=440, w_r=440*r=440.0. Area=193600. Diff=0.33
        # Areas are identical, tie-breaker chooses AR closest to subject AR (1.33), which is 1:1.
        expected_crop_file = crop_dir / "frame_000001_crop_1x1.png"

        files_in_crop_dir = list(crop_dir.glob('*'))
        assert expected_crop_file.exists(), (
            f"Expected crop file not found: {expected_crop_file}. "
            f"Files found in crop dir: {[f.name for f in files_in_crop_dir]}"
        )

        cropped_image = cv2.imread(str(expected_crop_file))
        assert cropped_image is not None, "Cropped image could not be read from disk."
        h, w, _ = cropped_image.shape
        assert w / h == pytest.approx(1/1, rel=0.01), f"Cropped image AR is {w/h}, expected ~{1/1}"


if __name__ == "__main__":
    pytest.main([__file__])
