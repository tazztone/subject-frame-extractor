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
    'mobile_sam': MagicMock(name='mobile_sam'),
    'segment_anything': MagicMock(name='segment_anything'),
    'skimage': MagicMock(name='skimage'),
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
    return [
        {'shot_id': 1, 'status': 'pending', 'seed_result': {'details': {'mask_area_pct': 50}}, 'seed_metrics': {'best_face_sim': 0.9, 'score': 0.95}},
        {'shot_id': 2, 'status': 'pending', 'seed_result': {'details': {'mask_area_pct': 5}}, 'seed_metrics': {'best_face_sim': 0.8, 'score': 0.9}},
        {'shot_id': 3, 'status': 'pending', 'seed_result': {'details': {'mask_area_pct': 60}}, 'seed_metrics': {'best_face_sim': 0.4, 'score': 0.8}},
        {'shot_id': 4, 'status': 'pending', 'seed_result': {'details': {'mask_area_pct': 70}}, 'seed_metrics': {'score': 0.7}},
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

class TestPerformanceOptimizer:
    @pytest.fixture
    def resource_manager(self, test_config):
        test_config.monitoring.memory_warning_threshold_mb = 100
        test_config.monitoring.memory_critical_threshold_mb = 200
        test_config.monitoring.cpu_warning_threshold_percent = 90
        logger = MagicMock()
        return app.AdaptiveResourceManager(logger=logger, config=test_config)

    @patch('app.time.sleep')
    @patch('psutil.Process')
    @patch('psutil.cpu_percent')
    def test_adjust_parameters(self, mock_cpu_percent, mock_process, mock_sleep, resource_manager):
        mock_cpu_percent.return_value = 95
        mock_process.return_value.memory_info.return_value.rss = 150 * 1024 * 1024
        resource_manager.current_limits['batch_size'] = 32
        resource_manager.current_limits['num_workers'] = 4
        resource_manager._adjust_parameters(resource_manager._get_resource_metrics())
        assert resource_manager.current_limits['batch_size'] == 22
        assert resource_manager.current_limits['num_workers'] == 2

    @patch('time.sleep')
    @patch('psutil.Process')
    @patch('psutil.cpu_percent')
    def test_adjust_parameters_critical_memory(self, mock_cpu_percent, mock_process, mock_sleep, resource_manager):
        mock_cpu_percent.return_value = 50
        mock_process.return_value.memory_info.return_value.rss = 250 * 1024 * 1024 # Above critical threshold
        resource_manager.current_limits['batch_size'] = 32
        resource_manager._adjust_parameters(resource_manager._get_resource_metrics())
        assert resource_manager.current_limits['batch_size'] == 1
        resource_manager.logger.critical.assert_called_once()

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

if __name__ == "__main__":
    pytest.main([__file__])