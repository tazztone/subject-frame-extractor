from unittest.mock import MagicMock

import pytest

from core.managers.sam3 import SAM3Wrapper


@pytest.fixture
def mock_database():
    """Provides a thread-safe mock Database instance."""
    db = MagicMock()
    db.set_db_path = MagicMock()
    # Mock context manager if needed
    db.__enter__ = MagicMock(return_value=db)
    db.__exit__ = MagicMock()
    return db


@pytest.fixture
def mock_app_ui(mock_config, mock_logger, mock_database, mock_thumbnail_manager, mock_model_registry):
    """Provides a mocked AppUI instance with all sub-handlers/managers."""
    import threading
    from queue import Queue
    from unittest.mock import patch

    from ui.app_ui import AppUI

    q = Queue()
    evt = threading.Event()

    with patch("ui.app_ui.is_cuda_available", return_value=False):
        ui = AppUI(mock_config, mock_logger, q, evt, mock_thumbnail_manager, mock_model_registry, mock_database)
        ui.components = MagicMock()  # Will be populated as needed by tests
        return ui


@pytest.fixture
def sam3_unit():
    """
    A SAM3Wrapper instance for unit tests.
    Bypasses __init__ entirely — no SAM3 package import, no torch dependency.
    Unit tests verify method behaviour only.
    """
    wrapper = object.__new__(SAM3Wrapper)  # allocates instance, skips __init__
    wrapper.predictor = MagicMock()

    def handle_request(*args, **kwargs):
        request = args[0] if args else kwargs.get("request")
        if request.get("type") == "start_session":
            return {"session_id": "test_session"}
        elif request.get("type") == "add_prompt":
            import torch

            return {"outputs": {"out_binary_masks": torch.ones((1, 100, 100)), "out_obj_ids": [1]}}
        elif request.get("type") == "detect_objects":
            return {"outputs": [{"bbox": [0, 0, 10, 10], "label": "person"}]}
        elif request.get("type") in ("close_session", "remove_object", "clear_prompts"):
            return {"status": "ok"}
        return {}

    wrapper.predictor.handle_request.side_effect = handle_request

    import torch

    wrapper.predictor.handle_stream_request.return_value = [
        {"frame_index": 0, "outputs": {"out_binary_masks": torch.ones((1, 100, 100)), "out_obj_ids": [1]}}
    ]

    wrapper.predictor.model = MagicMock()
    wrapper.session_id = None
    wrapper.device = "cpu"
    return wrapper
