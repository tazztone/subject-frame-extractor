import pytest
import gradio as gr
from unittest.mock import MagicMock, patch
from ui.app_ui import AppUI
from core.application_state import ApplicationState
from core.models import Scene
import torch

@pytest.fixture
def mock_app():
    config = MagicMock()
    config.default_tracker_model_name = "sam3"
    config.models_dir = "/tmp/models"
    config.user_agent = "test"
    config.retry_max_attempts = 1
    config.retry_backoff_seconds = [1]
    
    logger = MagicMock()
    progress_queue = MagicMock()
    cancel_event = MagicMock()
    thumbnail_manager = MagicMock()
    model_registry = MagicMock()
    
    with patch("torch.cuda.is_available", return_value=False):
        app = AppUI(config, logger, progress_queue, cancel_event, thumbnail_manager, model_registry)
        with gr.Blocks() as demo:
            app.build_ui()
            app._create_event_handlers()
        app.demo = demo
    return app

def test_pagination_contract(mock_app):
    """Verify that pagination handlers return exactly 4 values as wired."""
    # Find dependencies for next_page_button.click
    next_btn_id = mock_app.components["next_page_button"]._id
    
    # In Gradio 6, introspection is via demo.config["dependencies"] and demo.fns
    dep_config = next(d for d in mock_app.demo.config["dependencies"] 
                      if any(t[0] == next_btn_id and t[1] == "click" for t in d["targets"]))
    
    # Sanity check: verify wiring says 4 outputs
    assert len(dep_config["outputs"]) == 4
    
    # Get the function object
    dep_id = dep_config["id"]
    fn_obj = mock_app.demo.fns[dep_id]
    fn = fn_obj.fn
    
    # Inputs: [app_state, view, page_num]
    state = ApplicationState()
    result = fn(state, "All", "1")
    
    # Assert return count (app_state, items, pages_text, dropdown_update)
    assert isinstance(result, tuple)
    assert len(result) == 4
    assert isinstance(result[0], ApplicationState)

def test_pipeline_wrappers_yield_component_keys(mock_app):
    """Verify that pipeline wrappers yield dictionaries with component-object keys."""
    # We'll test run_extraction_wrapper as a representative
    # It takes (state, *args)
    state = ApplicationState()
    # We need to mock the pipeline call inside to return immediately
    with patch("ui.app_ui.execute_extraction") as mock_exec:
        # Mock it yielding an error to test the error path specifically
        mock_exec.side_effect = Exception("Test Error")
        
        # The wrapper catch-block yields the error dict
        gen = mock_app.run_extraction_wrapper(state)
        first_yield = next(gen)
        
        assert isinstance(first_yield, dict)
        # Verify keys are component objects, not strings
        for key in first_yield.keys():
            assert hasattr(key, "_id"), f"Key {key} should be a Gradio component object"
            assert key in mock_app.components.values()
