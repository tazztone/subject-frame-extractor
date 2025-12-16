import pytest
from pydantic import ValidationError
import sys
import unittest
from unittest.mock import MagicMock, patch, mock_open
from pathlib import Path
import json
import time
import numpy as np
import gradio as gr
import cv2
import datetime
from collections import deque
import pydantic

# Add project root to the Python path to allow for submodule imports
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

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
mock_torch_nn.attention = MagicMock(name='torch.nn.attention')

mock_torch_nn_init = MagicMock(name='torch.nn.init')
mock_torch_nn_functional = MagicMock(name='torch.nn.functional')
mock_torch_optim = MagicMock(name='torch.optim')
mock_torch_utils = MagicMock(name='torch.utils')
mock_torch_utils.__path__ = ['fake']
mock_torch_utils_data = MagicMock(name='torch.utils.data')
mock_torch_utils_checkpoint = MagicMock(name='torch.utils.checkpoint')
mock_torch_utils_pytree = MagicMock(name='torch.utils._pytree')


mock_torchvision = MagicMock(name='torchvision')
mock_torchvision.ops = MagicMock(name='torchvision.ops')
mock_torchvision.ops.roi_align = MagicMock(name='torchvision.ops.roi_align')
mock_torchvision.ops.misc = MagicMock(name='torchvision.ops.misc')
mock_torchvision.datasets = MagicMock(name='torchvision.datasets')
mock_torchvision.datasets.vision = MagicMock(name='torchvision.datasets.vision')
mock_torchvision.transforms = MagicMock(name='torchvision.transforms')
mock_torchvision.transforms.functional = MagicMock(name='torchvision.transforms.functional')
mock_torchvision.utils = MagicMock(name='torchvision.utils')

mock_insightface = MagicMock(name='insightface')
mock_insightface.app = MagicMock(name='insightface.app')

mock_timm = MagicMock(name='timm')
mock_timm.models = MagicMock(name='timm.models')
mock_timm.models.layers = MagicMock(name='timm.models.layers')

# mock_imagehash = MagicMock() # Don't mock imagehash as it's a light dependency and needed for dedup tests
mock_pycocotools = MagicMock(name='pycocotools')
mock_pycocotools.mask = MagicMock(name='pycocotools.mask')

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
    'torch.nn.attention': mock_torch_nn.attention,
    'torch.nn.init': mock_torch_nn.init,
    'torch.nn.functional': mock_torch_nn_functional,
    'torch.optim': mock_torch_optim,
    'torch.utils': mock_torch_utils,
    'torch.utils.data': mock_torch_utils_data,
    'torch.utils.checkpoint': mock_torch_utils_checkpoint,
    'torch.utils._pytree': mock_torch_utils_pytree,
    'torchvision': mock_torchvision,
    'torchvision.ops': mock_torchvision.ops,
    'torchvision.ops.roi_align': mock_torchvision.ops.roi_align,
    'torchvision.ops.misc': mock_torchvision.ops.misc,
    'torchvision.datasets': mock_torchvision.datasets,
    'torchvision.datasets.vision': mock_torchvision.datasets.vision,
    'torchvision.transforms': mock_torchvision.transforms,
    'torchvision.transforms.functional': mock_torchvision.transforms.functional,
    'torchvision.utils': mock_torchvision.utils,
    # 'cv2': MagicMock(name='cv2'), # cv2 is now used in tests, so we don't mock it globally
    'insightface': mock_insightface,
    'insightface.app': mock_insightface.app,
    'timm': mock_timm,
    'timm.models': mock_timm.models,
    'timm.models.layers': mock_timm.models.layers,
    'onnxruntime': MagicMock(name='onnxruntime'),
    'DAM4SAM': MagicMock(name='DAM4SAM'),
    'DAM4SAM.utils': MagicMock(name='DAM4SAM.utils'),
    'DAM4SAM.dam4sam_tracker': MagicMock(name='DAM4SAM.dam4sam_tracker'),
    'GPUtil': MagicMock(getGPUs=lambda: [MagicMock(memoryUtil=0.5)]),
    # 'imagehash': mock_imagehash,
    'pycocotools': mock_pycocotools,
    'pycocotools.mask': mock_pycocotools.mask,
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
    'lpips': MagicMock(name='lpips'),
    'numba': MagicMock(name='numba'),
    'skimage': MagicMock(name='skimage'),
    'skimage.metrics': MagicMock(name='skimage.metrics'),
}

# Mock pydantic_settings if not available
mock_pydantic_settings = MagicMock(name='pydantic_settings')
mock_pydantic_settings.BaseSettings = pydantic.BaseModel
mock_pydantic_settings.SettingsConfigDict = dict
modules_to_mock['pydantic_settings'] = mock_pydantic_settings

patch.dict(sys.modules, modules_to_mock).start()

# Imports from refactored modules
from core.config import Config
from core.database import Database
from core.logger import AppLogger
from core.models import Scene, Frame, QualityConfig, _coerce
from core.filtering import apply_all_filters_vectorized
from ui.gallery_utils import auto_set_thresholds
from core.events import PreAnalysisEvent

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
        'tracker_model_name': 'sam3',
        'best_frame_strategy': 'Largest Person',
        'text_prompt': '',
        'min_mask_area_pct': 1.0,
        'sharpness_base_scale': 2500.0,
        'edge_strength_base_scale': 100.0,
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
    return [Scene(**data) for data in scenes_data]

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
        assert _coerce(value, to_type) == expected

    def test_coerce_invalid_raises(self):
        with pytest.raises(ValueError):
            _coerce("not-a-number", int)
        with pytest.raises(ValueError):
            _coerce("not-a-float", float)

    def test_config_init(self):
        mock_config_data = {}
        with patch('core.config.json_config_settings_source', return_value=mock_config_data):
            # Pass an argument to the constructor
            config = Config(logs_dir="init_logs")

        assert config.logs_dir == "init_logs"

    @patch('pathlib.Path.mkdir', MagicMock())
    @patch('pathlib.Path.touch', MagicMock())
    @patch('pathlib.Path.unlink', MagicMock())
    def test_validation_error(self):
        """Test that a validation error is raised for invalid config."""
        with pytest.raises(ValidationError):
            # quality_weights sum cannot be zero
            Config(quality_weights_sharpness=0, quality_weights_edge_strength=0, quality_weights_contrast=0, quality_weights_brightness=0, quality_weights_entropy=0, quality_weights_niqe=0)

class TestAppLogger:
    def test_app_logger_instantiation(self):
        """Tests that the logger can be instantiated with a valid config."""
        try:
            config = Config()
            AppLogger(config=config, log_to_console=False, log_to_file=False)
        except Exception as e:
            pytest.fail(f"Logger instantiation with a config object failed: {e}")
    def test_auto_set_thresholds(self):
        per_metric_values = {'sharpness': list(range(10, 101, 10)), 'contrast': [1, 2, 3, 4, 5]}
        slider_keys = ['sharpness_min', 'sharpness_max', 'contrast_min']
        selected_metrics = list(per_metric_values.keys())
        updates = auto_set_thresholds(per_metric_values, 75, slider_keys, selected_metrics)
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
        kept, rejected, _, _ = apply_all_filters_vectorized(sample_frames_data, filters, Config())

        kept_filenames = {f['filename'] for f in kept}
        rejected_filenames = {f['filename'] for f in rejected}

        assert 'frame_01.png' in kept_filenames
        assert 'frame_04.png' in rejected_filenames # face_sim too low
        assert 'frame_05.png' in rejected_filenames # mask_area_pct too low

    def test_calculate_quality_metrics_with_niqe(self):
        """Test quality metrics calculation including NIQE."""
        test_config = Config()
        mock_niqe_metric = MagicMock()
        mock_niqe_metric.device.type = 'cpu'
        mock_niqe_metric.return_value = 5.0 # Raw NIQE score (float is fine for mock)

        image_data = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        frame = Frame(image_data=image_data, frame_number=1)

        quality_config = QualityConfig(
            sharpness_base_scale=test_config.sharpness_base_scale,
            edge_strength_base_scale=test_config.edge_strength_base_scale,
            enable_niqe=True
        )
        
        # Mock torch.from_numpy chain
        with patch('core.models.torch.from_numpy') as mock_torch_from_numpy:
            mock_tensor = MagicMock()
            mock_tensor.to.return_value = mock_tensor
            mock_torch_from_numpy.return_value.float.return_value.permute.return_value.unsqueeze.return_value = mock_tensor
            
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
        event = PreAnalysisEvent.model_validate(mock_ui_state)
        assert event.face_ref_img_path == str(valid_img)

        # Path is the same as the video
        mock_ui_state['face_ref_img_path'] = str(video_path)
        event = PreAnalysisEvent.model_validate(mock_ui_state)
        assert event.face_ref_img_path == ""

        # Path does not exist
        mock_ui_state['face_ref_img_path'] = "/non/existent.png"
        event = PreAnalysisEvent.model_validate(mock_ui_state)
        assert event.face_ref_img_path == ""

        # Path has invalid extension
        invalid_ext = tmp_path / "face.txt"
        invalid_ext.touch()
        mock_ui_state['face_ref_img_path'] = str(invalid_ext)
        event = PreAnalysisEvent.model_validate(mock_ui_state)
        assert event.face_ref_img_path == ""

        # Path is empty
        mock_ui_state['face_ref_img_path'] = ""
        event = PreAnalysisEvent.model_validate(mock_ui_state)
        assert event.face_ref_img_path == ""



if __name__ == "__main__":
    pytest.main([__file__])
