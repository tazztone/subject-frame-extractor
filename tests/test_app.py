import pytest
import sys
import unittest
from unittest.mock import MagicMock, patch, mock_open
from pathlib import Path
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

class MockConfig:
    def __init__(self):
        self.ui_defaults = {
            'max_resolution': '1080', 'thumbnails_only': True, 'thumb_megapixels': 0.5,
            'scene_detect': True, 'method': 'keyframes', 'interval': '2', 'nth_frame': '10',
            'use_png': False, 'text_prompt': 'test', 'pre_analysis_enabled': True,
            'pre_sample_nth': 1, 'person_detector_model': 'yolo11s.pt', 'face_model_name': 'buffalo_l',
            'dam4sam_model_name': 'sam21pp-T', 'enable_dedup': False, 'require_face_match': False,
        }
        self.thumbnail_cache_size = 100
        self.min_mask_area_pct = 5.0
        self.sharpness_base_scale = 1.0
        self.edge_strength_base_scale = 1.0
        self.grounding_dino_params = {'box_threshold': 0.3, 'text_threshold': 0.3}
        self.DIRS = {'logs': Path('logs'), 'downloads': Path('/tmp/downloads')}
        self.filter_defaults = {
            'dedup_thresh': {'min': 0, 'max': 10, 'default': 5, 'step': 1},
            'niqe': {'min': 0, 'max': 20, 'default_min': 0, 'step': 0.1},
            'sharpness': {'min': 0, 'max': 20, 'default_min': 0, 'step': 0.1},
            'edge_strength': {'min': 0, 'max': 20, 'default_min': 0, 'step': 0.1},
            'contrast': {'min': 0, 'max': 255, 'default_min': 0, 'step': 1},
            'brightness': {'min': 0, 'max': 255, 'default_min': 0, 'default_max': 255, 'step': 1},
            'entropy': {'min': 0, 'max': 8, 'default_min': 0, 'step': 0.1},
            'face_sim': {'min': 0, 'max': 1, 'default_min': 0, 'step': 0.01},
            'mask_area_pct': {'min': 0, 'max': 100, 'default_min': 0, 'step': 0.1},
        }
        self.QUALITY_METRICS = ['niqe', 'sharpness', 'edge_strength', 'contrast', 'brightness', 'entropy']
        self.GROUNDING_DINO_CONFIG = "mock_config.py"
        self.GROUNDING_DINO_CKPT = "mock_ckpt.pth"
        self.GROUNDING_BOX_THRESHOLD = 0.35
        self.GROUNDING_TEXT_THRESHOLD = 0.25
        self.settings = {
            'monitoring': {
                'cpu_threshold': 80, 'mem_threshold': 80, 'high_threshold_duration': 10,
                'cooldown_duration': 30, 'memory_warning_threshold_mb': 8192,
                'cpu_warning_threshold_percent': 90, 'gpu_memory_warning_threshold_percent': 90
            }
        }

@pytest.fixture
def mock_app_ui():
    with patch('app.Config', MockConfig):
        app_ui = app.EnhancedAppUI()
        return app_ui


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

class TestAppUI:
    def test_build_ui_does_not_crash(self, mock_app_ui):
        try:
            demo = mock_app_ui.build_ui()
            assert isinstance(demo, gr.Blocks)
        except Exception as e:
            pytest.fail(f"build_ui() raised an exception: {e}")

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
    def test_config_loads_from_default_string(self):
        """Verify that the Config class correctly loads settings from the hardcoded DEFAULT_CONFIG string."""
        config = app.Config()
        # Check a few values to ensure the YAML string was parsed correctly
        assert 'ui_defaults' in config.settings
        assert config.settings['ui_defaults']['max_resolution'] == "maximum available"
        assert config.settings['quality_weights']['sharpness'] == 25
        assert config.GROUNDING_BOX_THRESHOLD == 0.35


class TestEnhancedLogging(unittest.TestCase):
    def test_enhanced_logger_basic_functionality(self):
        """
        Tests that the logger can be instantiated and basic log messages can be sent
        without crashing. It also checks that the custom SUCCESS level works.
        """
        logger = app.EnhancedLogger(log_to_console=False, log_to_file=False)

        # We use mock_open to prevent actual file I/O
        with patch('builtins.open', mock_open()) as mock_file:
            # Wrap in a try-except block to provide more context on failure
            try:
                logger.info("Test info message", component="test")
                logger.warning("Test warning message", component="test")
                logger.success("Test success message", component="test")
            except Exception as e:
                self.fail(f"Logger unexpectedly raised an exception: {e}")

            # Verify that the manual file write was called for each log message
            self.assertEqual(mock_file().write.call_count, 3)

    def test_operation_context_timing(self):
        logger = app.EnhancedLogger(log_to_console=False, log_to_file=False)
        with patch('builtins.open', mock_open()):
            with logger.operation_context("test_operation", "test_component") as ctx:
                time.sleep(0.01)
                self.assertEqual(ctx['operation'], "test_operation")

class TestFilterLogic:
    @pytest.fixture
    def mock_config_instance(self):
        with patch('app.Config') as MockConfig:
            mock_instance = MockConfig.return_value
            mock_instance.QUALITY_METRICS = ['sharpness', 'contrast']
            mock_instance.filter_defaults = {'sharpness': {'default_min': 10, 'default_max': 90}, 'contrast': {'default_min': 20, 'default_max': 80}, 'dedup_thresh': {'default': 5}}
            mock_instance.ui_defaults = {'require_face_match': False}
            yield mock_instance

    def test_apply_all_filters_no_filters(self, sample_frames_data, mock_config_instance):
        kept, rejected, _, _ = app.apply_all_filters_vectorized(sample_frames_data, {}, mock_config_instance)
        assert len(kept) == len(sample_frames_data)

    def test_auto_set_thresholds(self):
        per_metric_values = {'sharpness': list(range(10, 101, 10)), 'contrast': [1, 2, 3, 4, 5]}
        slider_keys = ['sharpness_min', 'sharpness_max', 'contrast_min']
        updates = app.auto_set_thresholds(per_metric_values, 75, slider_keys)
        assert updates['slider_sharpness_min']['value'] == 77.5
        assert updates['slider_contrast_min']['value'] == 4.0

class TestPerformanceOptimizer(unittest.TestCase):
    @patch('app.Config')
    @patch('app.EnhancedLogger')
    def setUp(self, mock_logger, mock_config):
        self.mock_config = mock_config
        self.mock_config.settings = {'monitoring': {
            'memory_warning_threshold_mb': 100,
            'memory_critical_threshold_mb': 200,
            'cpu_warning_threshold_percent': 90
        }}
        self.mock_logger = mock_logger
        self.resource_manager = app.AdaptiveResourceManager(logger=self.mock_logger, config=self.mock_config)

    @patch('app.time.sleep')
    @patch('psutil.Process')
    @patch('psutil.cpu_percent')
    def test_adjust_parameters(self, mock_cpu_percent, mock_process, mock_sleep):
        mock_cpu_percent.return_value = 95
        mock_process.return_value.memory_info.return_value.rss = 150 * 1024 * 1024
        self.resource_manager.current_limits['batch_size'] = 32
        self.resource_manager.current_limits['num_workers'] = 4
        self.resource_manager._adjust_parameters(self.resource_manager._get_resource_metrics())
        self.assertEqual(self.resource_manager.current_limits['batch_size'], 22)
        self.assertEqual(self.resource_manager.current_limits['num_workers'], 3)

    @patch('time.sleep')
    @patch('psutil.Process')
    @patch('psutil.cpu_percent')
    def test_adjust_parameters_critical_memory(self, mock_cpu_percent, mock_process, mock_sleep):
        mock_cpu_percent.return_value = 50
        mock_process.return_value.memory_info.return_value.rss = 250 * 1024 * 1024 # Above critical threshold
        self.resource_manager.current_limits['batch_size'] = 32
        self.resource_manager._adjust_parameters(self.resource_manager._get_resource_metrics())
        self.assertEqual(self.resource_manager.current_limits['batch_size'], 1)
        self.mock_logger.critical.assert_called_once()

class TestQuality(unittest.TestCase):
    def test_compute_entropy(self):
        hist = np.ones(256, dtype=np.uint64)
        self.assertAlmostEqual(app.compute_entropy(hist), 1.0, places=5)

    def test_quality_metrics_small_mask_fallback(self):
        frame = app.Frame(image_data=np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8), frame_number=1)
        thumb_image_rgb = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        small_mask = np.zeros((50, 50), dtype=np.uint8)
        small_mask[25, 25] = 255 # Mask is too small (1 pixel)
        mock_config = MockConfig()
        mock_logger = MagicMock()

        # This should not raise an exception, but fallback to full-frame analysis
        try:
            frame.calculate_quality_metrics(thumb_image_rgb, mock_config, mock_logger, mask=small_mask)
        except ValueError:
            pytest.fail("calculate_quality_metrics raised ValueError on small mask, but should have fallen back.")

        # Ensure metrics were still calculated (not all zero)
        self.assertNotEqual(frame.metrics.sharpness_score, 0.0)

class TestSceneLogic:
    def test_get_scene_status_text(self):
        assert app.get_scene_status_text([]) == "No scenes loaded."
        assert app.get_scene_status_text([{'status': 'included'}, {'status': 'excluded'}]) == "1/2 scenes included for propagation."

    @patch('app.save_scene_seeds')
    def test_toggle_scene_status(self, mock_save, sample_scenes):
        scenes, _, _ = app.toggle_scene_status(sample_scenes, 2, 'included', '/fake/dir', MagicMock())
        assert scenes[1]['status'] == 'included'
        mock_save.assert_called_once()

class TestUtils(unittest.TestCase):
    def test_sanitize_filename(self):
        self.assertEqual(app.sanitize_filename("a/b\\c:d*e?f\"g<h>i|j.txt"), "a_b_c_d_e_f_g_h_i_j.txt")

    @patch('app.gc.collect')
    @patch('app.torch')
    def test_safe_resource_cleanup(self, mock_torch, mock_gc):
        mock_torch.cuda.is_available.return_value = True
        with app.safe_resource_cleanup(): pass
        mock_gc.assert_called_once()
        mock_torch.cuda.empty_cache.assert_called_once()

class TestVideo(unittest.TestCase):
    @patch('app.VideoManager.get_video_info')
    @patch('app.run_ffmpeg_extraction')
    @patch('pathlib.Path.is_file', return_value=True)
    def test_extraction_pipeline_run(self, mock_is_file, mock_ffmpeg, mock_info):
        with patch('app.Config', MockConfig):
            params = app.AnalysisParameters(source_path='/fake.mp4')
            pipeline = app.EnhancedExtractionPipeline(params, MagicMock(), MagicMock(), app.Config(), MagicMock(), MagicMock())
            pipeline.run()
            mock_ffmpeg.assert_called()

if __name__ == "__main__":
    pytest.main([__file__])