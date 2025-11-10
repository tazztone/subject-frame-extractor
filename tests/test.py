import pytest
from pydantic import ValidationError
import sys
import unittest
from unittest.mock import MagicMock, patch, mock_open
from pathlib import Path
import os
import json
import time
import numpy as np
import gradio as gr
import cv2
import datetime
from collections import deque

# Add project root to the Python path to allow for submodule imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Add submodule paths for direct import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Grounded-SAM-2', 'grounding_dino')))

# Mock heavy dependencies before they are imported by the app
mock_torch = MagicMock(name='torch')
mock_torch.__version__ = "2.0.0"
mock_torch.__path__ = ['fake'] # Make it a package
mock_torch.__spec__ = MagicMock()
mock_torch.hub = MagicMock(name='torch.hub')
mock_torch.cuda = MagicMock(name='torch.cuda')
mock_torch.cuda.is_available.return_value = False
mock_torch.distributed = MagicMock(name='torch.distributed')
mock_torch.multiprocessing = MagicMock(name='torch.multiprocessing')
mock_torch.amp = MagicMock(name='torch.amp')


mock_torch_autograd = MagicMock(name='torch.autograd')
mock_torch_autograd.Variable = MagicMock(name='torch.autograd.Variable')


mock_torch_nn = MagicMock(name='torch.nn')
mock_torch_nn.__path__ = ['fake'] # Make it a package
# Create a dummy class to act as torch.nn.Module to allow class inheritance in dependencies
class MockNNModule:
    def __init__(self, *args, **kwargs): pass
    def __call__(self, *args, **kwargs): return MagicMock()
mock_torch_nn.Module = MockNNModule

mock_torch_nn_init = MagicMock(name='torch.nn.init')
mock_torch_nn_functional = MagicMock(name='torch.nn.functional')
mock_torch_optim = MagicMock(name='torch.optim')
mock_torch_utils = MagicMock(name='torch.utils')
mock_torch_utils.__path__ = ['fake']
mock_torch_utils_data = MagicMock(name='torch.utils.data')


mock_torchvision = MagicMock(name='torchvision')
mock_torchvision.ops = MagicMock(name='torchvision.ops')
mock_torchvision.transforms = MagicMock(name='torchvision.transforms')
mock_torchvision.transforms.functional = MagicMock(name='torchvision.transforms.functional')
mock_torchvision.utils = MagicMock(name='torchvision.utils')

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


mock_matplotlib = MagicMock(name='matplotlib')
mock_matplotlib.__path__ = ['fake']
mock_matplotlib.ticker = MagicMock(name='matplotlib.ticker')
mock_matplotlib.figure = MagicMock(name='matplotlib.figure')
mock_matplotlib.backends = MagicMock(name='matplotlib.backends')
mock_matplotlib.backends.backend_agg = MagicMock(name='matplotlib.backends.backend_agg')


modules_to_mock = {
    'torch': mock_torch,
    'torch.hub': mock_torch.hub,
    'torch.distributed': mock_torch.distributed,
    'torch.multiprocessing': mock_torch.multiprocessing,
    'torch.autograd': mock_torch_autograd,
    'torch.nn': mock_torch_nn,
    'torch.nn.init': mock_torch_nn_init,
    'torch.nn.functional': mock_torch_nn_functional,
    'torch.optim': mock_torch_optim,
    'torch.utils': mock_torch_utils,
    'torch.utils.data': mock_torch_utils_data,
    'torchvision': mock_torchvision,
    'torchvision.ops': mock_torchvision.ops,
    'torchvision.transforms': mock_torchvision.transforms,
    'torchvision.transforms.functional': mock_torchvision.transforms.functional,
    'torchvision.utils': mock_torchvision.utils,
    # 'cv2': MagicMock(name='cv2'), # cv2 is now used in tests, so we don't mock it globally
    'insightface': mock_insightface,
    'insightface.app': mock_insightface.app,
    'onnxruntime': MagicMock(name='onnxruntime'),
    'groundingdino': MagicMock(name='groundingdino'),
    'groundingdino.util': MagicMock(name='groundingdino.util'),
    'groundingdino.util.inference': MagicMock(name='groundingdino.util.inference'),
    'groundingdino.config': MagicMock(name='groundingdino.config'),
    'DAM4SAM': MagicMock(name='DAM4SAM'),
    'DAM4SAM.utils': MagicMock(name='DAM4SAM.utils'),
    'DAM4SAM.dam4sam_tracker': MagicMock(name='DAM4SAM.dam4sam_tracker'),
    'ultralytics': MagicMock(name='ultralytics'),
    'GPUtil': MagicMock(getGPUs=lambda: [MagicMock(memoryUtil=0.5)]),
    'imagehash': mock_imagehash,
    'psutil': mock_psutil,
    'matplotlib': mock_matplotlib,
    'matplotlib.ticker': mock_matplotlib.ticker,
    'matplotlib.figure': mock_matplotlib.figure,
    'matplotlib.backends': mock_matplotlib.backends,
    'matplotlib.backends.backend_agg': mock_matplotlib.backends.backend_agg,
    'matplotlib.pyplot': MagicMock(),
    'scenedetect': MagicMock(),
    'yt_dlp': MagicMock(),
    'pyiqa': MagicMock(name='pyiqa'),
    'mediapipe': MagicMock(),
    'mediapipe.tasks': MagicMock(),
    'mediapipe.tasks.python': MagicMock(),
    'mediapipe.tasks.python.vision': MagicMock(),
}

patch.dict(sys.modules, modules_to_mock).start()

# Now import the monolithic app
import app
from app import Config, CompositionRoot

# --- Mocks for Tests ---
@pytest.fixture
def mock_ui_state():
    """Provides a dictionary with default values for UI-related event models."""
    return {
        'source_path': 'test.mp4',
        'upload_video': None,
        'method': 'interval',
        'interval': '1.0',
        'nth_frame': '5',
        'max_resolution': "720",
        'thumbnails_only': True,
        'thumb_megapixels': 0.2,
        'scene_detect': True,
        'output_folder': '/fake/output',
        'video_path': '/fake/video.mp4',
        'resume': False,
        'enable_face_filter': False,
        'face_ref_img_path': '',
        'face_ref_img_upload': None,
        'face_model_name': 'buffalo_l',
        'enable_subject_mask': False,
        'dam4sam_model_name': 'sam21pp-T',
        'person_detector_model': 'yolo11x.pt',
        'best_frame_strategy': 'Largest Person',
        'text_prompt': '',
        'box_threshold': 0.35,
        'text_threshold': 0.25,
        'min_mask_area_pct': 1.0,
        'sharpness_base_scale': 2500.0,
        'edge_strength_base_scale': 100.0,
        'gdino_config_path': 'GroundingDINO_SwinT_OGC.py',
        'gdino_checkpoint_path': 'models/groundingdino_swint_ogc.pth',
        'pre_analysis_enabled': True,
        'pre_sample_nth': 1,
        'primary_seed_strategy': '🧑‍🤝‍🧑 Find Prominent Person',
    }

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
    scenes_data = [
        {'shot_id': 1, 'start_frame': 0, 'end_frame': 100, 'status': 'pending', 'seed_result': {'details': {'mask_area_pct': 50}}, 'seed_metrics': {'best_face_sim': 0.9, 'score': 0.95}},
        {'shot_id': 2, 'start_frame': 101, 'end_frame': 200, 'status': 'pending', 'seed_result': {'details': {'mask_area_pct': 5}}, 'seed_metrics': {'best_face_sim': 0.8, 'score': 0.9}},
        {'shot_id': 3, 'start_frame': 201, 'end_frame': 300, 'status': 'pending', 'seed_result': {'details': {'mask_area_pct': 60}}, 'seed_metrics': {'best_face_sim': 0.4, 'score': 0.8}},
        {'shot_id': 4, 'start_frame': 301, 'end_frame': 400, 'status': 'pending', 'seed_result': {'details': {'mask_area_pct': 70}}, 'seed_metrics': {'score': 0.7}},
    ]
    return [app.Scene(**data) for data in scenes_data]

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
    @patch('app.Path.mkdir', MagicMock())
    @patch('os.access', return_value=True)
    def test_default_config_loading(self, mock_access):
        """Verify that the Config class correctly loads default values."""
        # Prevent file loading by ensuring no config file exists
        with patch('app.json_config_settings_source', return_value={}):
             config = app.Config()
        assert config.logs_dir == "logs"
        assert config.quality_weights_sharpness == 25
        assert not config.default_disable_parallel

    @patch('app.Path.mkdir', MagicMock())
    @patch('os.access', return_value=True)
    def test_file_override(self, mock_access):
        """Verify that a config file overrides defaults."""
        mock_config_data = {"logs_dir": "custom_logs", "quality_weights_sharpness": 50}
        with patch('app.json_config_settings_source', return_value=mock_config_data):
            config = app.Config()

        assert config.logs_dir == "custom_logs"
        assert config.quality_weights_sharpness == 50
        assert config.quality_weights_contrast == 15 # Check that a non-overridden value remains default

    @patch('app.Path.mkdir', MagicMock())
    @patch('os.access', return_value=True)
    def test_env_var_override(self, mock_access, monkeypatch):
        """Verify that environment variables override defaults."""
        monkeypatch.setenv("APP_LOGS_DIR", "env_logs")
        monkeypatch.setenv("APP_QUALITY_WEIGHTS_SHARPNESS", "75")

        with patch('app.json_config_settings_source', return_value={}):
            config = app.Config()

        assert config.logs_dir == "env_logs"
        assert config.quality_weights_sharpness == 75
        assert isinstance(config.quality_weights_sharpness, int) # Type coercion
        assert config.quality_weights_contrast == 15 # Default value

    @patch('app.Path.mkdir', MagicMock())
    @patch('os.access', return_value=True)
    def test_precedence_env_over_file(self, mock_access, monkeypatch):
        """Verify that environment variables have precedence over config files."""
        monkeypatch.setenv("APP_LOGS_DIR", "env_logs")
        mock_config_data = {"logs_dir": "file_logs"}

        with patch('app.json_config_settings_source', return_value=mock_config_data):
            config = app.Config()

        assert config.logs_dir == "env_logs"

    @patch('app.Path.mkdir', MagicMock())
    @patch('os.access', return_value=True)
    def test_init_arg_override(self, mock_access, monkeypatch):
        """Verify that arguments passed to the constructor have the highest precedence."""
        monkeypatch.setenv("APP_LOGS_DIR", "env_logs")
        mock_config_data = {"logs_dir": "file_logs"}

        with patch('app.json_config_settings_source', return_value=mock_config_data):
            # Pass an argument to the constructor
            config = app.Config(logs_dir="init_logs")

        assert config.logs_dir == "init_logs"

    @patch('app.Path.mkdir', MagicMock())
    @patch('os.access', return_value=True)
    def test_validation_error(self, mock_access):
        """Test that a validation error is raised for invalid config."""
        with pytest.raises(ValidationError):
            # quality_weights sum cannot be zero
            app.Config(quality_weights_sharpness=0, quality_weights_edge_strength=0, quality_weights_contrast=0, quality_weights_brightness=0, quality_weights_entropy=0, quality_weights_niqe=0)

class TestAppLogger:
    def test_app_logger_instantiation(self):
        """Tests that the logger can be instantiated with a valid config."""
        try:
            config = Config()
            app.AppLogger(config=config, log_to_console=False, log_to_file=False)
        except Exception as e:
            pytest.fail(f"Logger instantiation with a config object failed: {e}")

    def test_operation_context_timing(self):
        config = Config()
        logger = app.AppLogger(config=config, log_to_console=False, log_to_file=False)
        logger.logger.log = MagicMock()
        with patch('builtins.open', mock_open()):
            with logger.operation("test_operation", "test_component"):
                time.sleep(0.01)
            # Check that start and done messages were logged
            assert logger.logger.log.call_count == 2

class TestFilterLogic:
    def test_apply_all_filters_no_filters(self, sample_frames_data):
        kept, rejected, _, _ = app.apply_all_filters_vectorized(sample_frames_data, {}, Config())
        assert len(kept) == len(sample_frames_data)

    def test_auto_set_thresholds(self):
        per_metric_values = {'sharpness': list(range(10, 101, 10)), 'contrast': [1, 2, 3, 4, 5]}
        slider_keys = ['sharpness_min', 'sharpness_max', 'contrast_min']
        selected_metrics = list(per_metric_values.keys())
        updates = app.auto_set_thresholds(per_metric_values, 75, slider_keys, selected_metrics)
        assert updates['slider_sharpness_min']['value'] == 77.5
        assert updates['slider_contrast_min']['value'] == 4.0

    def test_apply_all_filters_with_face_and_mask(self, sample_frames_data):
        """Verify filtering by face similarity and mask area."""
        filters = {
            "face_sim_enabled": True,
            "face_sim_min": 0.5,
            "mask_area_enabled": True,
            "mask_area_pct_min": 10.0,
        }
        kept, rejected, _, _ = app.apply_all_filters_vectorized(sample_frames_data, filters, Config())

        kept_filenames = {f['filename'] for f in kept}
        rejected_filenames = {f['filename'] for f in rejected}

        assert 'frame_01.png' in kept_filenames
        assert 'frame_04.png' in rejected_filenames # face_sim too low
        assert 'frame_05.png' in rejected_filenames # mask_area_pct too low

    @patch('app._update_gallery')
    def test_on_filters_changed(self, mock_update_gallery, sample_frames_data):
        """Verify that on_filters_changed correctly calls the gallery update function."""
        mock_update_gallery.return_value = ("Status", gr.update(value=[]))
        slider_values = {'sharpness_min': 10.0}
        event = app.FilterEvent(
            all_frames_data=sample_frames_data,
            per_metric_values={'face_sim': [0.8], 'mask_area_pct': [20.0]},
            output_dir="/fake/dir",
            gallery_view="Kept",
            show_overlay=True,
            overlay_alpha=0.5,
            require_face_match=False,
            dedup_thresh=-1,
            slider_values=slider_values,
            dedup_method="phash",
        )

        result = app.on_filters_changed(event, MagicMock(), Config())

        mock_update_gallery.assert_called_once()
        assert "filter_status_text" in result
        assert "results_gallery" in result

    @patch('app.on_filters_changed', return_value={"filter_status_text": "Reset", "results_gallery": gr.update(value=[])})
    def test_reset_filters(self, mock_on_filters_changed, sample_frames_data):
        """Verify that resetting filters restores default values."""
        test_config = Config()
        mock_ui = MagicMock(spec=app.EnhancedAppUI)
        mock_ui.config = test_config
        mock_ui.thumbnail_manager = MagicMock()
        # Simplify the mock to isolate the slider logic
        mock_ui.components = {
            'metric_sliders': {
                'sharpness_min': MagicMock(),
                'sharpness_max': MagicMock(),
            },
            'metric_accs': {}, # Keep it simple
            'dedup_method_input': MagicMock(value="phash"),
        }

        # The method is part of the class, so we call it from an instance
        result_tuple = app.EnhancedAppUI.on_reset_filters(
            mock_ui,
            all_frames_data=sample_frames_data,
            per_metric_values={'sharpness': [1,2,3]},
            output_dir="/fake/dir"
        )

        # The order of slider updates is determined by sorted keys: ['sharpness_max', 'sharpness_min']
        # Unpack accordingly to fix the assertion error.
        slider_max_update, slider_min_update = result_tuple[0], result_tuple[1]

        assert slider_min_update['value'] == test_config.filter_default_sharpness['default_min']
        assert slider_max_update['value'] == test_config.filter_default_sharpness['default_max']

        # The original test asserted this was called, let's keep it to ensure behavior is preserved
        mock_on_filters_changed.assert_called_once()


    def test_load_and_prep_filter_data_uses_config_for_hist_ranges(self, tmp_path):
        """
        Verify that histogram ranges for yaw and pitch are taken from the config.
        """
        metadata_content = (
            json.dumps({"params": {}}) + '\n' +
            json.dumps({"filename": "f1", "metrics": {"yaw": -150, "pitch": 150}}) + '\n'
        )
        metadata_path = tmp_path / "metadata.jsonl"
        metadata_path.write_text(metadata_content)

        config = Config() # Use default config which has wide ranges
        _, metric_values = app.load_and_prep_filter_data(
            str(metadata_path),
            get_all_filter_keys=lambda: ['yaw', 'pitch'],
            config=config
        )

        yaw_hist_bins = metric_values.get('yaw_hist', ([], []))[1]
        assert yaw_hist_bins[0] == config.filter_default_yaw['min']
        assert yaw_hist_bins[-1] == config.filter_default_yaw['max']

        pitch_hist_bins = metric_values.get('pitch_hist', ([], []))[1]
        assert pitch_hist_bins[0] == config.filter_default_pitch['min']
        assert pitch_hist_bins[-1] == config.filter_default_pitch['max']


    def test_deduplication_filter(self, sample_frames_data):
        """Verify that the deduplication filter removes similar frames."""
        # Frame 1 and 2 have identical phash
        filters = {"enable_dedup": True, "dedup_thresh": 0}

        # A more realistic mock for imagehash that returns different objects for different hashes
        hash_a = MagicMock(name="hash_a")
        hash_b = MagicMock(name="hash_b")
        hash_c = MagicMock(name="hash_c")
        hash_d = MagicMock(name="hash_d")
        hash_e = MagicMock(name="hash_e")

        # The lambda needs to accept two arguments (self and other) because of how it's called on the mock instance.
        hash_a.__sub__ = lambda s, o: 0 if o is hash_a else 1
        hash_b.__sub__ = lambda s, o: 0 if o is hash_b else 1
        hash_c.__sub__ = lambda s, o: 0 if o is hash_c else 1
        hash_d.__sub__ = lambda s, o: 0 if o is hash_d else 1
        hash_e.__sub__ = lambda s, o: 0 if o is hash_e else 1

        def side_effect(h_str):
            if h_str == 'a'*16: return hash_a
            if h_str == 'b'*16: return hash_b
            if h_str == 'c'*16: return hash_c
            if h_str == 'd'*16: return hash_d
            if h_str == 'e'*16: return hash_e
            return MagicMock() # Should not happen in this test

        mock_imagehash.hex_to_hash.side_effect = side_effect

        kept, rejected, _, _ = app.apply_all_filters_vectorized(sample_frames_data, filters, Config())

        kept_filenames = {f['filename'] for f in kept}
        assert 'frame_01.png' in kept_filenames
        assert 'frame_02.png' not in kept_filenames # Should be removed as a duplicate
        assert len(kept) == len(sample_frames_data) - 1

    def test_face_similarity_filter_require_match(self, sample_frames_data):
        """Verify face similarity filter rejects frames with low similarity AND frames with no face when required."""
        filters = {
            "face_sim_enabled": True,
            "face_sim_min": 0.5,
            "require_face_match": True # Explicitly require a face
        }
        kept, rejected, _, _ = app.apply_all_filters_vectorized(sample_frames_data, filters, Config())

        kept_filenames = {f['filename'] for f in kept}
        rejected_filenames = {f['filename'] for f in rejected}

        assert 'frame_01.png' in kept_filenames # High similarity
        assert 'frame_04.png' in rejected_filenames # Low similarity
        assert 'frame_06.png' in rejected_filenames # No face_sim value and require_face_match is on


class TestQuality:
    def test_compute_entropy(self):
        # A uniform distribution histogram should have max entropy (8.0)
        # We normalize it in the function, so it should be close to 1.0
        hist = np.ones(256, dtype=np.uint64)
        assert np.isclose(app.compute_entropy(hist, 8.0), 1.0)

    def test_quality_metrics_small_mask_fallback(self):
        test_config = Config()
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
        assert app.get_scene_status_text([])[0] == "No scenes loaded."
        assert app.get_scene_status_text([app.Scene(shot_id=1, start_frame=0, end_frame=0, status='included'), app.Scene(shot_id=2, start_frame=0, end_frame=0, status='excluded')])[0] == "1/2 scenes included for propagation."

    @patch('app.save_scene_seeds')
    def test_toggle_scene_status(self, mock_save, sample_scenes):
        scenes, _, _, _ = app.toggle_scene_status(sample_scenes, 2, 'included', '/fake/dir', MagicMock())
        assert scenes[1].status == 'included'
        mock_save.assert_called_once()

    @patch('app.save_scene_seeds')
    def test_apply_bulk_scene_filters(self, mock_save, sample_scenes, tmp_path):
        """Verify that bulk filters correctly include/exclude scenes."""
        test_config = Config()
        # Create a dummy preview file for the scene_thumb function to find
        (tmp_path / "previews").mkdir()
        for scene in sample_scenes:
            scene.preview_path = str(tmp_path / "previews" / f"scene_{scene.shot_id}.jpg")
            with open(scene.preview_path, 'w') as f:
                f.write('') # Create an empty file

        ui = app.EnhancedAppUI(config=test_config, logger=MagicMock(), progress_queue=MagicMock(), cancel_event=MagicMock(), thumbnail_manager=MagicMock())
        scenes, _, _, _, _, _, _ = ui.on_apply_bulk_scene_filters_extended(
            scenes=sample_scenes,
            min_mask_area=10.0,
            min_face_sim=0.5,
            min_confidence=0.85,
            enable_face_filter=True,
            output_folder=str(tmp_path),
            view="Kept"
        )

        status_map = {s.shot_id: s.status for s in scenes}
        assert status_map[1] == 'included'
        assert status_map[2] == 'excluded' # Mask area too low
        assert status_map[3] == 'excluded' # Face sim too low
        assert status_map[4] == 'excluded' # Confidence too low
        mock_save.assert_called_once()

class TestUtils:
    def test_sanitize_filename(self):
        # The function now depends on a config object
        assert app.sanitize_filename("a/b\\c:d*e?f\"g<h>i|j.txt", config=Config()) == "a_b_c_d_e_f_g_h_i_j.txt"

    @patch('app.gc.collect')
    @patch('app.torch')
    def test_safe_resource_cleanup(self, mock_torch, mock_gc):
        mock_torch.cuda.is_available.return_value = True
        with app.safe_resource_cleanup(device='cuda'): pass
        mock_gc.assert_called_once()
        mock_torch.cuda.empty_cache.assert_called_once()

class TestVideo:
    @patch('app.VideoManager.get_video_info')
    @patch('app.run_ffmpeg_extraction')
    @patch('pathlib.Path.is_file', return_value=True)
    @patch('pathlib.Path.exists', return_value=True)
    @patch('pathlib.Path.stat', return_value=MagicMock(st_size=1, st_mode=0))
    @patch('cv2.VideoCapture')
    @patch('pathlib.Path.mkdir')
    def test_extraction_pipeline_run(self, mock_mkdir, mock_videocapture, mock_stat, mock_exists, mock_is_file, mock_ffmpeg, mock_info, tmp_path, mock_ui_state):
        # Use tmp_path for a valid output directory
        output_dir = tmp_path / "downloads" / "fake"
        output_dir.mkdir(parents=True)
        mock_ui_state['output_folder'] = str(output_dir)

        params = app.ExtractionEvent.model_validate(mock_ui_state)
        logger = MagicMock()
        
        # Instantiate with all required arguments
        pipeline = app.EnhancedExtractionPipeline(
            config=Config(),
            logger=logger,
            params=params,
            progress_queue=MagicMock(),
            cancel_event=MagicMock()
        )
        pipeline.run()
        mock_ffmpeg.assert_called()

    @patch('app.is_image_folder', return_value=True)
    @patch('app.list_images')
    @patch('app.make_photo_thumbs')
    def test_extraction_pipeline_image_folder(self, mock_make_thumbs, mock_list_images, mock_is_folder, mock_ui_state, tmp_path):
        # Arrange
        output_dir = tmp_path / "image_folder_output"
        # No need to mkdir, the pipeline should do it.

        mock_list_images.return_value = [Path('/fake/dir/img1.jpg')]
        mock_ui_state['source_path'] = '/fake/dir'
        mock_ui_state['output_folder'] = str(output_dir)

        params = app.ExtractionEvent.model_validate(mock_ui_state)
        logger = MagicMock()
        pipeline = app.EnhancedExtractionPipeline(
            config=Config(),
            logger=logger,
            params=params,
            progress_queue=MagicMock(),
            cancel_event=MagicMock()
        )

        # Act
        result = pipeline.run()

        # Assert
        mock_is_folder.assert_called_once_with(Path('/fake/dir'))
        mock_list_images.assert_called_once()
        mock_make_thumbs.assert_called_once()

        assert (output_dir / "run_config.json").exists()
        assert (output_dir / "scenes.json").exists()

        assert result['done']
        assert not result['video_path']


class TestModels:
    def setup_method(self):
        self.logger = MagicMock()
        # Create a mock registry for testing purposes.
        # In a real scenario, the CompositionRoot would create this.
        self.mock_registry = app.ModelRegistry(logger=self.logger)
        app.model_registry = self.mock_registry # Inject mock registry

    @patch('app.download_model')
    @patch('app.gdino_load_model')
    @patch('app.resolve_grounding_dino_config')
    def test_get_grounding_dino_model_with_registry(self, mock_resolve_config, mock_gdino_load_model, mock_download):
        """
        Tests that get_grounding_dino_model uses the model registry and the path resolver.
        """
        fake_resolved_path = "/resolved/path/to/config.py"
        mock_resolve_config.return_value = fake_resolved_path
        mock_model = MagicMock()
        mock_gdino_load_model.return_value = mock_model

        # Spy on the registry's get_or_load method
        with patch.object(self.mock_registry, 'get_or_load', side_effect=self.mock_registry.get_or_load) as mock_get_or_load:
            # First call - should trigger loading
            model1 = app.get_grounding_dino_model(
                gdino_config_path="some_config.py",
                gdino_checkpoint_path="models/groundingdino_swint_ogc.pth",
                models_path="models",
                grounding_dino_url="http://fake.url/model.pth",
                user_agent="test-agent",
                retry_params=(3, (1, 2, 3)),
                device="cpu",
                logger=self.logger
            )
            # Second call - should be cached
            model2 = app.get_grounding_dino_model(
                gdino_config_path="some_config.py",
                gdino_checkpoint_path="models/groundingdino_swint_ogc.pth",
                models_path="models",
                grounding_dino_url="http://fake.url/model.pth",
                user_agent="test-agent",
                retry_params=(3, (1, 2, 3)),
                device="cpu",
                logger=self.logger
            )

            assert model1 is mock_model
            assert model2 is mock_model # Should be the same instance
            # get_or_load should be called, but the loader inside it should only run once.
            assert mock_get_or_load.call_count == 2
            mock_resolve_config.assert_called_once_with("some_config.py")
            mock_gdino_load_model.assert_called_once() # Loader function called only once
            passed_config_path = mock_gdino_load_model.call_args.kwargs['model_config_path']
            assert passed_config_path == fake_resolved_path

    def test_get_person_detector_with_registry(self):
        """
        Tests that get_person_detector uses the model registry for caching.
        """
        mock_detector_instance = MagicMock(spec=app.PersonDetector)

        with patch('app.PersonDetector', return_value=mock_detector_instance) as mock_person_detector_class:
            with patch.object(self.mock_registry, 'get_or_load', side_effect=self.mock_registry.get_or_load) as mock_get_or_load:
                # First call
                detector1 = app.get_person_detector("yolo.pt", "cpu", 640, 0.5, self.logger)
                # Second call
                detector2 = app.get_person_detector("yolo.pt", "cpu", 640, 0.5, self.logger)

                assert detector1 is mock_detector_instance
                assert detector2 is mock_detector_instance
                # The loader function (and thus the class constructor) should only be called once.
                mock_person_detector_class.assert_called_once()
                # get_or_load is called twice, but the loader fn within it only executes on the first call.
                assert mock_get_or_load.call_count == 2

class TestVideoManager:
    @patch('app.ytdlp')
    def test_prepare_video_youtube(self, mock_ytdlp):
        """Verify YouTube download logic is triggered for YouTube URLs."""
        source_path = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        with patch('app.json_config_settings_source', return_value={}):
            video_manager = app.VideoManager(source_path, Config())
        mock_yt_downloader = MagicMock()
        mock_ytdlp.YoutubeDL.return_value.__enter__.return_value = mock_yt_downloader
        mock_yt_downloader.extract_info.return_value = {}
        mock_yt_downloader.prepare_filename.return_value = "/fake/path/video.mp4"

        result = video_manager.prepare_video(MagicMock())

        mock_ytdlp.YoutubeDL.assert_called_once()
        mock_yt_downloader.extract_info.assert_called_once_with(source_path, download=True)
        assert result == "/fake/path/video.mp4"

    @patch('pathlib.Path.is_file', return_value=True)
    @patch('pathlib.Path.exists', return_value=True)
    @patch('pathlib.Path.stat', return_value=MagicMock(st_size=1))
    @patch('cv2.VideoCapture')
    def test_prepare_video_local_file(self, mock_videocapture, mock_stat, mock_exists, mock_is_file):
        """Verify local file path is returned directly."""
        source_path = "/path/to/local/video.mp4"
        with patch('app.json_config_settings_source', return_value={}):
             video_manager = app.VideoManager(source_path, Config())

        result = video_manager.prepare_video(MagicMock())

        assert result == source_path

    @patch('pathlib.Path.exists', return_value=False)
    def test_prepare_video_local_file_not_found(self, mock_exists):
        """Verify FileNotFoundError is raised for non-existent local files."""
        source_path = "/path/to/nonexistent/video.mp4"
        with patch('app.json_config_settings_source', return_value={}):
            video_manager = app.VideoManager(source_path, Config())

        with pytest.raises(FileNotFoundError):
            video_manager.prepare_video(MagicMock())

    @patch('app.ytdlp')
    def test_prepare_video_youtube_download_failure(self, mock_ytdlp):
        """Verify that a download error from yt-dlp raises a RuntimeError."""
        source_path = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        with patch('app.json_config_settings_source', return_value={}):
            video_manager = app.VideoManager(source_path, Config())

        mock_yt_downloader = MagicMock()
        mock_ytdlp.YoutubeDL.return_value.__enter__.return_value = mock_yt_downloader
        # Simulate a download error
        mock_ytdlp.utils.DownloadError = Exception # Mock the exception class
        mock_yt_downloader.extract_info.side_effect = mock_ytdlp.utils.DownloadError("Video unavailable")

        with pytest.raises(RuntimeError, match="Download failed"):
            video_manager.prepare_video(MagicMock())


class TestFrame:
    def test_calculate_quality_metrics_no_mask(self):
        """Verify quality metrics are calculated correctly without a mask."""
        test_config = Config()
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
    @patch('app.torch.from_numpy')
    def test_calculate_quality_metrics_with_niqe(self, mock_torch_from_numpy, mock_pyiqa):
        """Verify that NIQE score is calculated when enabled."""
        test_config = Config()
        mock_niqe_metric = MagicMock()
        mock_niqe_metric.return_value = 5.0  # Mock NIQE score
        # The tensor needs to be mock'd to allow chaining
        mock_tensor = MagicMock()
        mock_tensor.to.return_value = mock_tensor
        mock_torch_from_numpy.return_value.float.return_value.permute.return_value.unsqueeze.return_value = mock_tensor

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

class TestPreAnalysisEvent:
    def test_face_ref_validation(self, tmp_path, mock_ui_state):
        """Test the custom validator for face_ref_img_path."""
        video_path = tmp_path / "video.mp4"
        video_path.touch()
        mock_ui_state['video_path'] = str(video_path)

        # Valid image file
        valid_img = tmp_path / "face.jpg"
        valid_img.touch()
        mock_ui_state['face_ref_img_path'] = str(valid_img)
        event = app.PreAnalysisEvent.model_validate(mock_ui_state)
        assert event.face_ref_img_path == str(valid_img)

        # Path is the same as the video
        mock_ui_state['face_ref_img_path'] = str(video_path)
        event = app.PreAnalysisEvent.model_validate(mock_ui_state)
        assert event.face_ref_img_path == ""

        # Path does not exist
        mock_ui_state['face_ref_img_path'] = "/non/existent.png"
        event = app.PreAnalysisEvent.model_validate(mock_ui_state)
        assert event.face_ref_img_path == ""

        # Path has invalid extension
        invalid_ext = tmp_path / "face.txt"
        invalid_ext.touch()
        mock_ui_state['face_ref_img_path'] = str(invalid_ext)
        event = app.PreAnalysisEvent.model_validate(mock_ui_state)
        assert event.face_ref_img_path == ""

        # Path is empty
        mock_ui_state['face_ref_img_path'] = ""
        event = app.PreAnalysisEvent.model_validate(mock_ui_state)
        assert event.face_ref_img_path == ""

class TestCompositionRoot:
    def test_initialization(self):
        """Tests that CompositionRoot initializes its components correctly."""
        with patch('app.json_config_settings_source', return_value={}):
            root = app.CompositionRoot()
        assert isinstance(root.get_config(), Config)
        assert isinstance(root.get_logger(), app.AppLogger)
        assert isinstance(root.get_thumbnail_manager(), app.ThumbnailManager)
        # Check that the logger got the config
        assert root.get_logger().config is root.get_config()

    @patch('app.cleanup_models')
    def test_cleanup(self, mock_cleanup_models):
        """Tests that the cleanup method calls necessary cleanup functions."""
        with patch('app.json_config_settings_source', return_value={}):
            root = app.CompositionRoot()
        root.thumbnail_manager.clear_cache = MagicMock()
        root.cancel_event.set = MagicMock()
        root.cleanup()
        mock_cleanup_models.assert_called_once()
        root.thumbnail_manager.clear_cache.assert_called_once()
        root.cancel_event.set.assert_called_once()

if __name__ == "__main__":
    pytest.main([__file__])
