import sys
from unittest.mock import MagicMock, patch

# Create sophisticated mocks for packages that have submodules.
mock_torch = MagicMock(name='torch')
mock_torch.__version__ = "2.0.0"  # Satisfy ultralytics import
mock_torch.hub = MagicMock(name='torch.hub')
mock_torch.cuda = MagicMock(name='torch.cuda')
mock_torch.cuda.is_available.return_value = False

mock_torchvision = MagicMock(name='torchvision')
mock_torchvision.ops = MagicMock(name='torchvision.ops')
mock_torchvision.transforms = MagicMock(name='torchvision.transforms')
mock_torchvision.transforms.functional = MagicMock(name='torchvision.transforms.functional')

mock_insightface = MagicMock(name='insightface')
mock_insightface.app = MagicMock(name='insightface.app')

# Mock other heavy ML dependencies to avoid installing them.
modules_to_mock = {
    'torch': mock_torch,
    'torch.hub': mock_torch.hub,
    'torchvision': mock_torchvision,
    'torchvision.ops': mock_torchvision.ops,
    'torchvision.transforms': mock_torchvision.transforms,
    'torchvision.transforms.functional': mock_torchvision.transforms.functional,
    'cv2': MagicMock(name='cv2'),
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
}

# Apply the patch to sys.modules BEFORE any application code is imported.
patch.dict(sys.modules, modules_to_mock).start()

# Now, with mocks in place, we can safely import the rest of the modules.
import pytest
import gradio as gr

# Since we are not testing the full application, we can use a simplified config
class MockConfig:
    def __init__(self):
        self.ui_defaults = {
            'max_resolution': '1080',
            'thumbnails_only': True,
            'thumb_megapixels': 0.5,
            'scene_detect': True,
            'method': 'keyframes',
            'interval': '2',
            'nth_frame': '10',
            'use_png': False,
            'text_prompt': 'test',
            'pre_analysis_enabled': True,
            'pre_sample_nth': 1,
            'person_detector_model': 'yolo11s.pt',
            'face_model_name': 'buffalo_l',
            'dam4sam_model_name': 'sam21pp-T',
            'enable_dedup': False,
            'require_face_match': False,
        }
        self.thumbnail_cache_size = 100
        self.min_mask_area_pct = 5.0
        self.sharpness_base_scale = 1.0
        self.edge_strength_base_scale = 1.0
        self.grounding_dino_params = {
            'box_threshold': 0.3,
            'text_threshold': 0.3,
        }
        self.DIRS = {'logs': 'logs'}
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

# We must import the application code AFTER the mocks are in place.
from app.app_ui import EnhancedAppUI

@pytest.fixture
def mock_app_ui():
    """Fixture to create an AppUI instance with a mock config."""
    # The torch mock is already active from the global patch.
    app_ui = EnhancedAppUI(config=MockConfig())
    return app_ui

def test_build_ui_does_not_crash(mock_app_ui):
    """
    Tests that the Gradio UI can be built without raising an exception.
    This is a basic smoke test to catch initialization errors.
    """
    try:
        demo = mock_app_ui.build_ui()
        assert isinstance(demo, gr.Blocks)
    except Exception as e:
        pytest.fail(f"build_ui() raised an exception: {e}")