from unittest.mock import MagicMock

import pytest

from core.managers.sam3 import SAM3Wrapper


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

            return {"outputs": {"out_binary_masks": torch.zeros((1, 100, 100)), "out_obj_ids": [1]}}
        elif request.get("type") == "detect_objects":
            return {"outputs": [{"bbox": [0, 0, 10, 10], "label": "person"}]}
        elif request.get("type") in ("close_session", "remove_object", "clear_prompts"):
            return {"status": "ok"}
        return {}

    wrapper.predictor.handle_request.side_effect = handle_request

    import torch

    wrapper.predictor.handle_stream_request.return_value = [
        {"frame_index": 0, "outputs": {"out_binary_masks": torch.zeros((1, 100, 100)), "out_obj_ids": [1]}}
    ]

    wrapper.predictor.model = MagicMock()
    wrapper.session_id = None
    wrapper.device = "cpu"
    return wrapper
