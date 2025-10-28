import pytest
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
import logging

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
    'mediapipe': MagicMock(),
    'mediapipe.tasks': MagicMock(),
    'mediapipe.tasks.python': MagicMock(),
    'mediapipe.tasks.python.vision': MagicMock(),
}

patch.dict(sys.modules, modules_to_mock).start()

# Now import the monolithic app
import app
from app import AppUI

# --- Mocks for Tests ---

@pytest.fixture
def test_config():
    """Provides a clean, default Config object for each test."""
    with patch('app.Config._create_dirs'):
        yield app.Config(config_path=None)

@pytest.fixture
def app_ui(test_config):
    """Provides a patched AppUI instance for testing UI-decoupled methods."""
    with patch.object(AppUI, '_create_ui'), \
         patch.object(AppUI, '_setup_event_handlers'):
        ui = AppUI(config=test_config, logger=MagicMock())
        yield ui

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
        {'shot_id': 1, 'start_frame': 0, 'end_frame': 100, 'status': 'pending', 'seed_result': {'details': {'mask_area_pct': 50}}, 'seed_metrics': {'best_face_sim': 0.9, 'score': 0.95}},
        {'shot_id': 2, 'start_frame': 101, 'end_frame': 200, 'status': 'pending', 'seed_result': {'details': {'mask_area_pct': 5}}, 'seed_metrics': {'best_face_sim': 0.8, 'score': 0.9}},
        {'shot_id': 3, 'start_frame': 201, 'end_frame': 300, 'status': 'pending', 'seed_result': {'details': {'mask_area_pct': 60}}, 'seed_metrics': {'best_face_sim': 0.4, 'score': 0.8}},
        {'shot_id': 4, 'start_frame': 301, 'end_frame': 400, 'status': 'pending', 'seed_result': {'details': {'mask_area_pct': 70}}, 'seed_metrics': {'score': 0.7}},
    ]

# --- Test Classes ---

class TestUtils:
    @pytest.mark.parametrize("value, to_type, expected", [
        ("True", bool, True), ("false", bool, False), ("1", bool, True), ("0", bool, False),
        ("yes", bool, True), ("no", bool, False), (True, bool, True), (False, bool, False),
        ("123", int, 123), (123, int, 123), ("123.45", float, 123.45), (123.45, float, 123.45),
        ("string", str, "string"),
    ])
    def test_coerce(self, value, to_type, expected):
        assert app._coerce(value, to_type) == expected

    def test_coerce_invalid_raises(self):
        with pytest.raises(ValueError):
            app._coerce("not-a-number", int)
        with pytest.raises(ValueError):
            app._coerce("not-a-float", float)

    def test_sanitize_filename(self, test_config):
        assert app.sanitize_filename("a/b\\c:d*e?f\"g<h>i|j.txt", config=test_config) == "a_b_c_d_e_f_g_h_i_j.txt"

    @patch('app.gc.collect')
    @patch('app.torch')
    def test_safe_resource_cleanup(self, mock_torch, mock_gc):
        mock_torch.cuda.is_available.return_value = True
        with app.safe_resource_cleanup():
            pass
        mock_gc.assert_called_once()
        mock_torch.cuda.empty_cache.assert_called_once()

class TestConfig:
    def test_default_config_loading(self, test_config):
        assert test_config.paths.logs == "logs"
        assert test_config.quality_weights.sharpness == 25
        assert not test_config.ui_defaults.disable_parallel

    @patch('builtins.open', new_callable=mock_open, read_data='{"paths": {"logs": "custom_logs"}, "quality_weights": {"sharpness": 50}}')
    @patch('pathlib.Path.exists', return_value=True)
    def test_file_override(self, mock_exists, mock_file):
        with patch('app.Config._create_dirs'):
            config = app.Config(config_path="dummy_path.json")
            assert config.paths.logs == "custom_logs"
            assert config.quality_weights.sharpness == 50
            assert config.quality_weights.contrast == 15

    @patch.dict(os.environ, {"APP_PATHS_LOGS": "env_logs", "APP_QUALITY_WEIGHTS_SHARPNESS": "75"})
    def test_env_var_override(self):
        with patch('app.Config._create_dirs'):
            config = app.Config(config_path=None)
            assert config.paths.logs == "env_logs"
            assert config.quality_weights.sharpness == 75
            assert isinstance(config.quality_weights.sharpness, int)
            assert config.quality_weights.contrast == 15

class TestFilterLogic:
    def test_apply_all_filters_no_filters(self, sample_frames_data, test_config, app_ui):
        kept, _, _, _ = app_ui.apply_all_filters_vectorized(sample_frames_data, {}, test_config)
        assert len(kept) == len(sample_frames_data)

    def test_apply_all_filters_with_face_and_mask(self, sample_frames_data, test_config, app_ui):
        filters = {"face_sim_enabled": True, "face_sim_min": 0.5, "mask_area_enabled": True, "mask_area_pct_min": 10.0}
        kept, rejected, _, _ = app_ui.apply_all_filters_vectorized(sample_frames_data, filters, test_config)
        kept_filenames = {f['filename'] for f in kept}
        rejected_filenames = {f['filename'] for f in rejected}
        assert 'frame_01.png' in kept_filenames
        assert 'frame_04.png' in rejected_filenames
        assert 'frame_05.png' in rejected_filenames

class TestQuality:
    def test_compute_entropy(self):
        hist = np.ones(256, dtype=np.uint64)
        assert np.isclose(app.compute_entropy(hist, 8.0), 1.0)

class TestSceneLogic:
    def test_get_scene_status_text(self, app_ui):
        assert app_ui.get_scene_status_text([])[0] == "No scenes loaded."
        assert "1/2 scenes included" in app_ui.get_scene_status_text([{'status': 'included'}, {'status': 'excluded'}])[0]

class TestModels:
    def setup_method(self):
        self.logger = MagicMock()

    @patch.object(AppUI, 'download_model')
    @patch('app.gdino_load_model')
    def test_get_grounding_dino_model_path_resolution(self, mock_gdino_load_model, mock_download, app_ui):
        app._dino_model_cache = None
        relative_path = "Grounded-SAM-2/grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        app_ui._get_grounding_dino_model(gdino_config_path=relative_path, gdino_checkpoint_path="models/groundingdino_swint_ogc.pth")
        mock_gdino_load_model.assert_called_once()
        passed_config_path = mock_gdino_load_model.call_args.kwargs['model_config_path']
        expected_path = app.project_root / relative_path
        assert Path(passed_config_path).is_absolute() and Path(passed_config_path) == expected_path

class TestFrame:
    def test_calculate_quality_metrics_no_mask(self, test_config):
        image_data = np.zeros((100, 100, 3), dtype=np.uint8)
        image_data[:, :, 0] = np.tile(np.arange(100), (100, 1))
        frame = app.Frame(image_data=image_data, frame_number=1)
        quality_config = app.QualityConfig(sharpness_base_scale=test_config.sharpness_base_scale, edge_strength_base_scale=test_config.edge_strength_base_scale, enable_niqe=False)
        frame.calculate_quality_metrics(image_data, quality_config, MagicMock(), main_config=test_config)
        assert frame.metrics.sharpness_score > 0
        assert frame.metrics.contrast_score > 0
        assert 0 <= frame.metrics.brightness_score <= 100
        assert frame.error is None

class TestImageFolderUtils:
    @patch('pathlib.Path.is_dir')
    def test_is_image_folder(self, mock_is_dir):
        mock_is_dir.return_value = True
        assert app.is_image_folder('/fake/dir') is True
        mock_is_dir.return_value = False
        assert app.is_image_folder('/fake/file.txt') is False
        assert app.is_image_folder(None) is False

    @patch('pathlib.Path.iterdir')
    def test_list_images(self, mock_iterdir, test_config):
        def create_mock_path(name, is_file_val):
            p = MagicMock(spec=Path)
            p.name = name
            p.suffix = Path(name).suffix
            p.is_file.return_value = is_file_val
            p.__lt__ = lambda self, other: self.name < other.name
            return p
        mock_files = [create_mock_path('z.jpg', True), create_mock_path('a.png', True), create_mock_path('doc.txt', True)]
        mock_iterdir.return_value = mock_files
        result = app.list_images(Path('/fake/dir'), cfg=test_config)
        assert [r.name for r in result] == ['a.png', 'z.jpg']
