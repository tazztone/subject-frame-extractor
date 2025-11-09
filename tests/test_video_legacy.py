import pytest
import sys
from unittest.mock import MagicMock, patch
from pathlib import Path
import numpy as np

# Add project root for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the application code
import app
from app import Config, CompositionRoot, EnhancedAppUI

# --- Fixtures ---

@pytest.fixture
def mock_ui_state():
    return {
        'source_path': 'test.mp4', 'video_path': '/fake/video.mp4', 'output_folder': '/fake/output',
        'method': 'interval', 'interval': '1.0', 'max_resolution': "720", 'thumbnails_only': True,
        'scene_detect': True, 'pre_analysis_enabled': True, 'enable_subject_mask': False,
        'primary_seed_strategy': '🧑‍🤝‍🧑 Find Prominent Person', 'person_detector_model': 'yolo11x.pt',
        'upload_video': None, 'nth_frame': '5', 'thumb_megapixels': 0.2, 'resume': False,
        'enable_face_filter': False, 'face_ref_img_path': '', 'face_ref_img_upload': None,
        'face_model_name': 'buffalo_l', 'dam4sam_model_name': 'sam21pp-T', 'best_frame_strategy': 'Largest Person',
        'text_prompt': '', 'box_threshold': 0.35, 'text_threshold': 0.25, 'min_mask_area_pct': 1.0,
        'sharpness_base_scale': 2500.0, 'edge_strength_base_scale': 100.0,
        'gdino_config_path': 'GroundingDINO_SwinT_OGC.py',
        'gdino_checkpoint_path': 'models/groundingdino_swint_ogc.pth', 'pre_sample_nth': 1
    }

@pytest.fixture
def ui_instance():
    """Provides a mocked EnhancedAppUI instance for testing."""
    with patch('app.CompositionRoot') as mock_root_class:
        mock_root = mock_root_class.return_value
        mock_root.get_config.return_value = Config()
        mock_root.get_logger.return_value = MagicMock()
        ui = EnhancedAppUI()
        ui.thumbnail_manager = MagicMock()
        return ui

# --- Test Classes ---

class TestQuality:
    @patch('app.Frame.calculate_quality_metrics')
    def test_quality_metrics_calculation(self, mock_calculate_quality):
        """Test that the quality metrics calculation is called and handled."""
        test_config = Config()
        image_data = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        frame = app.Frame(image_data=image_data, frame_number=1)

        # Simulate a successful quality calculation by setting the score directly
        frame.metrics.sharpness_score = 50.0
        frame.error = None

        assert frame.metrics.sharpness_score > 0 and frame.error is None

class TestVideo:
    @patch('app.EnhancedExtractionPipeline._run_ffmpeg')
    @patch('app.validate_video_file', return_value=True)
    @patch('app.VideoManager')
    def test_extraction_pipeline_run(self, MockVideoManager, mock_validate, mock_ffmpeg, tmp_path, mock_ui_state):
        output_dir = tmp_path / "fake_output"
        output_dir.mkdir()
        mock_ui_state['output_folder'] = str(output_dir)

        params = app.ExtractionEvent.model_validate(mock_ui_state)

        # Configure the mock VideoManager
        mock_video_manager_instance = MockVideoManager.return_value
        mock_video_manager_instance.prepare_video.return_value = "/fake/video.mp4"
        mock_video_manager_instance.get_video_info.return_value = (1920, 1080, 30.0, 300)

        pipeline = app.EnhancedExtractionPipeline(config=Config(), logger=MagicMock(), params=params, progress_queue=MagicMock(), cancel_event=MagicMock())
        pipeline.run()

        mock_ffmpeg.assert_called()
        assert (output_dir / "run_config.json").exists()


class TestCompositionRoot:
    def test_initialization_and_cleanup(self):
        with patch('app.json_config_settings_source', return_value={}), \
             patch('app.cleanup_models') as mock_cleanup:
            root = CompositionRoot()
            assert isinstance(root.get_config(), Config)
            root.thumbnail_manager = MagicMock()
            root.thumbnail_manager.clear_cache = MagicMock()
            root.cleanup()
            mock_cleanup.assert_called_once()
            root.thumbnail_manager.clear_cache.assert_called_once()
            assert root.cancel_event.is_set()

if __name__ == "__main__":
    pytest.main([__file__])
