"""
Tests for AppUI handlers and state management.

These tests verify that AppUI methods correctly handle the consolidated
ApplicationState and perform UI updates as expected.
"""

from unittest.mock import MagicMock, patch
import pytest
from ui.app_ui import AppUI, ApplicationState


class TestAppUIHandlers:
    """Tests for AppUI event handlers."""

    @pytest.fixture
    def app_ui(self, mock_config, mock_logger, mock_thumbnail_manager, mock_model_registry, mock_progress_queue, mock_cancel_event):
        """Create an AppUI instance for testing."""
        app = AppUI(
            config=mock_config,
            logger=mock_logger,
            progress_queue=mock_progress_queue,
            cancel_event=mock_cancel_event,
            thumbnail_manager=mock_thumbnail_manager,
            model_registry=mock_model_registry
        )
        # Manually initialize critical components for testing
        app.components["application_state"] = MagicMock()
        app.components["unified_log"] = MagicMock()
        app.components["unified_status"] = MagicMock()
        app.components["main_tabs"] = MagicMock()
        app.components["main_tabs"] = MagicMock()
        # app.components["stepper"] = MagicMock() # Removed from UI
        app.components["extracted_video_path_state"] = MagicMock()
        app.components["extracted_frames_dir_state"] = MagicMock()
        app.components["scenes_state"] = MagicMock()
        app.components["analysis_output_dir_state"] = MagicMock()
        app.components["seeding_results_column"] = MagicMock()
        app.components["propagation_group"] = MagicMock()
        app.components["propagate_masks_button"] = MagicMock()
        app.components["scene_filter_status"] = MagicMock()
        app.components["metric_sliders"] = {}
        app.components["metric_accs"] = {}
        return app

    def test_on_extraction_success(self, app_ui):
        """Test _on_extraction_success updates ApplicationState correctly."""
        current_state = ApplicationState()
        result = {
            "extracted_video_path_state": "/path/to/video.mp4",
            "extracted_frames_dir_state": "/path/to/frames",
        }

        updates = app_ui._on_extraction_success(result, current_state)
        
        new_state = updates[app_ui.components["application_state"]]
        assert new_state.extracted_video_path == "/path/to/video.mp4"
        assert new_state.extracted_frames_dir == "/path/to/frames"

    def test_on_pre_analysis_success(self, app_ui):
        """Test _on_pre_analysis_success updates ApplicationState correctly."""
        current_state = ApplicationState(extracted_video_path="/path/to/video.mp4")
        result = {
            "scenes": [{"shot_id": 1, "start_frame": 0, "end_frame": 100}],
            "output_dir": "/test/output",
        }

        updates = app_ui._on_pre_analysis_success(result, current_state)
        
        new_state = updates[app_ui.components["application_state"]]
        assert len(new_state.scenes) == 1
        assert new_state.analysis_output_dir == "/test/output"

    def test_on_propagation_success(self, app_ui):
        """Test _on_propagation_success returns the state."""
        current_state = ApplicationState(analysis_output_dir="/test")
        result = {"output_dir": "/test"}

        updates = app_ui._on_propagation_success(result, current_state)
        
        assert updates[app_ui.components["application_state"]] == current_state

    def test_on_analysis_success(self, app_ui):
        """Test _on_analysis_success updates ApplicationState correctly."""
        current_state = ApplicationState()
        result = {
            "metadata_path": "/test/metadata.db",
        }

        updates = app_ui._on_analysis_success(result, current_state)
        
        new_state = updates[app_ui.components["application_state"]]
        assert new_state.analysis_metadata_path == "/test/metadata.db"

    def test_on_reset_filters(self, app_ui):
        """Test on_reset_filters resets ApplicationState."""
        current_state = ApplicationState(smart_filter_enabled=True)
        
        # Note: on_reset_filters is decorated with @safe_ui_callback, 
        # but we can test the underlying logic if needed or the wrapped call
        updates = app_ui.on_reset_filters(current_state)
        
        new_state = updates[0] # Returns a tuple
        assert new_state.smart_filter_enabled is False

    @patch("ui.app_ui.on_filters_changed")
    def test_on_filters_changed_wrapper(self, mock_on_filters, app_ui):
        """Test on_filters_changed_wrapper uses ApplicationState data."""
        mock_on_filters.return_value = {"filter_status_text": "OK", "results_gallery": []}
        state = ApplicationState(all_frames_data=[{"f": 1}], analysis_output_dir="/test")
        
        # Zip slider values - assuming default 3 metrics for test
        slider_vals = [0.0, 0.0, 0.0] 
        
        # We need to ensure sliders are in components for zip to work if it uses them
        # In current impl it uses sorted(self.components['metric_sliders'].keys())
        app_ui.components['metric_sliders'] = {} # Empty for simple test
        
        status, gallery = app_ui.on_filters_changed_wrapper(
            state, "Kept", False, 0.6, False, 5, "pHash", *slider_vals
        )
        
        assert status == "OK"
        mock_on_filters.assert_called_once()
