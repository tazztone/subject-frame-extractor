"""
Smoke test for UI initialization.
Ensures that AppUI.build_ui() completes without KeyError or crashing.
"""

import threading
from queue import Queue
from unittest.mock import MagicMock, patch

import gradio as gr
import pytest

from ui.app_ui import AppUI

# Mark as smoke test
pytestmark = pytest.mark.smoke


@patch("torch.cuda.is_available", return_value=False)
def test_ui_initialization_smoke(mock_cuda):
    """
    Verify that the UI can be built with mocked dependencies.
    This catches KeyErrors in component registration or tab building.
    """
    config = MagicMock()
    # Provide necessary config attributes for building the UI
    config.default_tracker_model_name = "sam2"
    config.debug_mode = True
    config.gradio_auto_pctl_input = 50
    config.gradio_show_mask_overlay = True
    config.gradio_overlay_alpha = 0.5
    config.export_enable_crop = True
    config.export_crop_padding = 10
    config.export_crop_ars = "1:1, 9:16"
    config.filter_default_quality_score = {"min": 0, "max": 100, "step": 1, "default_min": 50}
    config.models_dir = "/tmp/models"
    config.user_agent = "test"
    config.retry_max_attempts = 1
    config.retry_backoff_seconds = [1]
    config.default_face_model_name = "buffalo_l"
    config.default_primary_seed_strategy = "Largest Person"

    logger = MagicMock()
    progress_queue = Queue()
    cancel_event = threading.Event()
    thumbnail_manager = MagicMock()
    model_registry = MagicMock()
    database = MagicMock()
    # Mock registry names for dropdown choices
    model_registry.get_tracker_names.return_value = ["sam2"]
    model_registry.get_tracker_vram_requirement.return_value = 4000
    model_registry.get_detector_names.return_value = ["YOLO12l-Seg"]

    app = AppUI(config, logger, progress_queue, cancel_event, thumbnail_manager, model_registry, database)

    # Avoid thread startup
    with patch.object(app, "preload_models"):
        with gr.Blocks():
            app.build_ui()
            # If we reach here without exception, the build was successful
            assert "application_state" in app.components
            assert isinstance(app.components["application_state"], gr.State)

    print("UI Initialization Successful")
