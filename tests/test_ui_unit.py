import pytest
from unittest.mock import MagicMock, patch, ANY
import gradio as gr
from ui.app_ui import AppUI
from core.events import ExtractionEvent
from core.pipelines import execute_extraction

@pytest.fixture
def app_ui(mock_config, mock_logger, mock_progress_queue, mock_cancel_event, mock_thumbnail_manager, mock_model_registry):
    with patch('ui.app_ui.AppUI.preload_models'):
        app = AppUI(mock_config, mock_logger, mock_progress_queue, mock_cancel_event, mock_thumbnail_manager, mock_model_registry)
        app.components = {
            'face_seeding_group': MagicMock(),
            'text_seeding_group': MagicMock(),
            'auto_seeding_group': MagicMock(),
            'enable_face_filter_input': MagicMock()
        }
        app.ana_input_components = []
        return app

def test_stepper_html(app_ui):
    html = app_ui._get_stepper_html(0)
    assert "Source" in html
    assert "Export" in html

def test_run_extraction_wrapper(app_ui):
    with patch.object(app_ui, '_run_pipeline', return_value=iter([])) as mock_run_pipeline:
        # Args matching ext_ui_map_keys:
        # ['source_path', 'upload_video', 'method', 'interval', 'nth_frame', 'max_resolution', 'thumb_megapixels', 'scene_detect']
        args = ["video.mp4", None, "interval", 1.0, 1, "1080", 0.5, True]

        gen = app_ui.run_extraction_wrapper(*args)
        list(gen)

        mock_run_pipeline.assert_called_once()
        call_args = mock_run_pipeline.call_args[0]
        assert call_args[0] == execute_extraction
        event = call_args[1]
        assert isinstance(event, ExtractionEvent)
        assert event.source_path == "video.mp4"
        assert event.method == "interval"

def test_fix_strategy_visibility_face_ref(app_ui):
    res = app_ui._fix_strategy_visibility("Face (Reference)")
    assert isinstance(res, dict)
    assert len(res) > 0

def test_fix_strategy_visibility_text(app_ui):
    res = app_ui._fix_strategy_visibility("Text Prompt")
    assert isinstance(res, dict)

def test_get_metric_description(app_ui):
    desc = app_ui.get_metric_description("quality_score")
    assert "score" in desc
