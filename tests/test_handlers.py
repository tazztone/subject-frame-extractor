"""
Tests for UI handlers (analysis, extraction, filtering).

These tests verify the handler classes work correctly in isolation,
using mocked dependencies to avoid GPU requirements.
"""
import pytest
from unittest.mock import MagicMock, patch, PropertyMock
import gradio as gr


class TestAnalysisHandler:
    """Tests for AnalysisHandler class."""
    
    @pytest.fixture
    def mock_app_ui(self, mock_config, mock_logger, mock_thumbnail_manager, mock_model_registry):
        """Create a mock AppUI instance."""
        mock_app = MagicMock()
        mock_app.config = mock_config
        mock_app.logger = mock_logger
        mock_app.thumbnail_manager = mock_thumbnail_manager
        mock_app.model_registry = mock_model_registry
        mock_app.progress_queue = MagicMock()
        mock_app.cancel_event = MagicMock()
        mock_app.ana_ui_map_keys = ['output_folder', 'seed_strategy']
        mock_app._create_pre_analysis_event = MagicMock()
        return mock_app
    
    @pytest.fixture
    def handler(self, mock_app_ui, mock_config, mock_logger, mock_thumbnail_manager, mock_model_registry):
        """Create an AnalysisHandler instance."""
        from ui.handlers.analysis_handler import AnalysisHandler
        return AnalysisHandler(
            app=mock_app_ui,
            config=mock_config,
            logger=mock_logger,
            thumbnail_manager=mock_thumbnail_manager,
            model_registry=mock_model_registry
        )
    
    def test_init(self, handler, mock_app_ui, mock_config):
        """Test AnalysisHandler initialization."""
        assert handler.app == mock_app_ui
        assert handler.config == mock_config
        assert handler.logger is not None
        assert handler.thumbnail_manager is not None
        assert handler.model_registry is not None
    
    def test_on_pre_analysis_success_basic(self, handler):
        """Test on_pre_analysis_success returns correct updates."""
        result = {
            "unified_log": "Pre-analysis done",
            "scenes": [{"shot_id": 1}],
            "output_dir": "/test/output",
        }
        
        updates = handler.on_pre_analysis_success(result)
        
        assert updates["unified_log"] == "Pre-analysis done"
        assert updates["scenes_state"] == [{"shot_id": 1}]
        assert updates["analysis_output_dir_state"] == "/test/output"
        assert "main_tabs" in updates
    
    def test_on_pre_analysis_success_with_face_ref(self, handler):
        """Test on_pre_analysis_success includes face reference when present."""
        result = {
            "unified_log": "Done",
            "scenes": [],
            "output_dir": "/test",
            "final_face_ref_path": "/path/to/face.png",
        }
        
        updates = handler.on_pre_analysis_success(result)
        
        assert updates["final_face_ref_path_state"] == "/path/to/face.png"
    
    def test_on_pre_analysis_success_default_log(self, handler):
        """Test on_pre_analysis_success uses default log message."""
        result = {}
        
        updates = handler.on_pre_analysis_success(result)
        
        assert updates["unified_log"] == "Pre-analysis complete."
    
    def test_on_propagation_success(self, handler):
        """Test on_propagation_success returns correct updates."""
        result = {
            "unified_log": "Propagation done",
            "output_dir": "/test/output",
        }
        
        updates = handler.on_propagation_success(result)
        
        assert updates["unified_log"] == "Propagation done"
        assert updates["analysis_output_dir_state"] == "/test/output"
        assert "filtering_tab" in updates
        assert "main_tabs" in updates
    
    def test_on_analysis_success(self, handler):
        """Test on_analysis_success returns correct updates."""
        result = {
            "unified_log": "Analysis done",
            "output_dir": "/test/output",
        }
        
        updates = handler.on_analysis_success(result)
        
        assert updates["unified_log"] == "Analysis done"
        assert updates["analysis_output_dir_state"] == "/test/output"
        assert "filtering_tab" in updates
        assert "main_tabs" in updates


class TestExtractionHandler:
    """Tests for ExtractionHandler class."""
    
    @pytest.fixture
    def mock_app_ui(self, mock_config, mock_logger, mock_thumbnail_manager, mock_model_registry):
        """Create a mock AppUI instance."""
        mock_app = MagicMock()
        mock_app.config = mock_config
        mock_app.logger = mock_logger
        mock_app.thumbnail_manager = mock_thumbnail_manager
        mock_app.model_registry = mock_model_registry
        mock_app.progress_queue = MagicMock()
        mock_app.cancel_event = MagicMock()
        mock_app.ext_ui_map_keys = ['source_path', 'method', 'interval']
        return mock_app
    
    @pytest.fixture
    def handler(self, mock_app_ui, mock_config, mock_logger, mock_thumbnail_manager, mock_model_registry):
        """Create an ExtractionHandler instance."""
        from ui.handlers.extraction_handler import ExtractionHandler
        return ExtractionHandler(
            app=mock_app_ui,
            config=mock_config,
            logger=mock_logger,
            thumbnail_manager=mock_thumbnail_manager,
            model_registry=mock_model_registry
        )
    
    def test_init(self, handler, mock_app_ui, mock_config):
        """Test ExtractionHandler initialization."""
        assert handler.app == mock_app_ui
        assert handler.config == mock_config
        assert handler.logger is not None
    
    def test_on_extraction_success(self, handler):
        """Test on_extraction_success returns correct updates."""
        result = {
            "unified_log": "Extraction done",
            "extracted_video_path_state": "/path/to/video.mp4",
            "extracted_frames_dir_state": "/path/to/frames",
        }
        
        updates = handler.on_extraction_success(result)
        
        assert updates["unified_log"] == "Extraction done"
        assert updates["extracted_video_path_state"] == "/path/to/video.mp4"
        assert updates["extracted_frames_dir_state"] == "/path/to/frames"
        assert "main_tabs" in updates
    
    def test_on_extraction_success_default_values(self, handler):
        """Test on_extraction_success uses defaults for missing values."""
        result = {}
        
        updates = handler.on_extraction_success(result)
        
        assert updates["unified_log"] == "Extraction complete."
        assert updates["extracted_video_path_state"] == ""
        assert updates["extracted_frames_dir_state"] == ""


class TestFilteringHandler:
    """Tests for FilteringHandler class."""
    
    @pytest.fixture
    def mock_app_ui(self, mock_config, mock_logger, mock_thumbnail_manager):
        """Create a mock AppUI instance."""
        mock_app = MagicMock()
        mock_app.config = mock_config
        mock_app.logger = mock_logger
        mock_app.thumbnail_manager = mock_thumbnail_manager
        mock_app.get_all_filter_keys = MagicMock(return_value=[
            'sharpness_min', 'contrast_min', 'face_sim_min'
        ])
        return mock_app
    
    @pytest.fixture
    def handler(self, mock_app_ui, mock_config, mock_logger, mock_thumbnail_manager):
        """Create a FilteringHandler instance."""
        from ui.handlers.filtering_handler import FilteringHandler
        return FilteringHandler(
            app=mock_app_ui,
            config=mock_config,
            logger=mock_logger,
            thumbnail_manager=mock_thumbnail_manager
        )
    
    def test_init(self, handler, mock_app_ui, mock_config):
        """Test FilteringHandler initialization."""
        assert handler.app == mock_app_ui
        assert handler.config == mock_config
        assert handler.logger is not None
    
    def test_on_preset_changed_no_filters(self, handler):
        """Test on_preset_changed with 'No Filters' preset."""
        updates = handler.on_preset_changed("No Filters")
        
        # All sliders should be set to 0
        for key in ['sharpness_min', 'contrast_min', 'face_sim_min']:
            assert f"slider_{key}" in updates
    
    def test_on_preset_changed_quality_focus(self, handler):
        """Test on_preset_changed with 'Quality Focus' preset."""
        updates = handler.on_preset_changed("Quality Focus")
        
        # Quality Focus preset has specific values
        assert "slider_sharpness_min" in updates
        assert "slider_contrast_min" in updates
    
    def test_on_preset_changed_face_priority(self, handler):
        """Test on_preset_changed with 'Face Priority' preset."""
        updates = handler.on_preset_changed("Face Priority")
        
        assert "slider_face_sim_min" in updates
    
    def test_on_preset_changed_balanced(self, handler):
        """Test on_preset_changed with 'Balanced' preset."""
        updates = handler.on_preset_changed("Balanced")
        
        assert "slider_sharpness_min" in updates
        assert "slider_face_sim_min" in updates
    
    def test_on_preset_changed_unknown_preset(self, handler):
        """Test on_preset_changed with unknown preset uses defaults."""
        updates = handler.on_preset_changed("Unknown Preset")
        
        # Should return updates with default values (0)
        for key in ['sharpness_min', 'contrast_min', 'face_sim_min']:
            assert f"slider_{key}" in updates
    
    @patch('ui.gallery_utils.on_filters_changed')
    def test_on_reset_filters(self, mock_on_filters, handler):
        """Test on_reset_filters resets all sliders."""
        mock_on_filters.return_value = {"gallery": []}
        
        updates = handler.on_reset_filters(
            all_frames_data=[],
            per_metric_values={},
            output_dir="/test"
        )
        
        # All sliders should be reset
        for key in ['sharpness_min', 'contrast_min', 'face_sim_min']:
            assert f"slider_{key}" in updates
    
    @patch('ui.gallery_utils.auto_set_thresholds')
    def test_on_auto_set_thresholds(self, mock_auto_set, handler):
        """Test on_auto_set_thresholds calls utility correctly."""
        mock_auto_set.return_value = {"slider_sharpness_min": 10}
        
        result = handler.on_auto_set_thresholds(
            {"sharpness": [10, 20, 30]},  # per_metric_values
            25,  # percentile
            True, False, True  # checkbox values
        )
        
        mock_auto_set.assert_called_once()
        assert result == {"slider_sharpness_min": 10}
