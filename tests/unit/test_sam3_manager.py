import numpy as np
import pytest


def test_sam3_wrapper_session_lifecycle(sam3_unit):
    # First init
    session_id = sam3_unit.init_video("video.mp4")
    assert session_id == "test_session"
    assert sam3_unit.session_id == "test_session"

    # Re-init (should call close_session)
    from unittest.mock import patch

    with patch.object(sam3_unit, "close_session", wraps=sam3_unit.close_session) as mock_close:
        sam3_unit.init_video("new_video.mp4")
        assert mock_close.called

    sam3_unit.close_session()
    assert sam3_unit.session_id is None
    assert sam3_unit.predictor.handle_request.called


def test_sam3_wrapper_add_bbox_prompt(sam3_unit):
    sam3_unit.session_id = "test_session"

    # Test with text and 3D mask
    mask = sam3_unit.add_bbox_prompt(
        frame_idx=0, obj_id=1, bbox_xywh=[10, 10, 50, 50], img_size=(100, 100), text="person"
    )
    assert isinstance(mask, np.ndarray)
    assert mask.shape == (100, 100)

    # Test with masks=None fallback
    sam3_unit.predictor.handle_request.return_value = {"outputs": {"out_binary_masks": None}}
    mask = sam3_unit.add_bbox_prompt(0, 1, [0, 0, 10, 10], (100, 100))
    assert mask.sum() == 0
    assert sam3_unit.predictor.handle_request.called


def test_sam3_wrapper_propagate(sam3_unit):
    sam3_unit.session_id = "test_session"

    # Test normal propagation
    results = list(sam3_unit.propagate(start_idx=0, max_frames=1))
    assert len(results) == 1

    # Test propagation with missing masks/ids
    sam3_unit.predictor.handle_stream_request.return_value = [{"frame_index": 1, "outputs": {}}]
    results = list(sam3_unit.propagate(start_idx=0, max_frames=1))
    assert len(results) == 0


def test_sam3_wrapper_detect_objects(sam3_unit):
    # Normal case
    results = sam3_unit.detect_objects(np.zeros((100, 100, 3), dtype=np.uint8), "person")
    assert len(results) == 1

    # Empty prompt
    assert sam3_unit.detect_objects(None, "") == []
    assert sam3_unit.detect_objects(None, "  ") == []


def test_sam3_wrapper_utility_methods(sam3_unit):
    # RuntimeError when no session
    with pytest.raises(RuntimeError):
        sam3_unit.add_text_prompt(0, "test")
    with pytest.raises(RuntimeError):
        sam3_unit.add_point_prompt(0, 1, [[10, 10]], [1], (100, 100))

    # Safe returns/no-ops when no session
    assert sam3_unit.remove_object(1) is None
    assert sam3_unit.clear_prompts() is None

    sam3_unit.session_id = "test_session"
    sam3_unit.add_text_prompt(0, "test")
    sam3_unit.add_point_prompt(0, 1, [[10, 10]], [1], (100, 100))
    sam3_unit.remove_object(1)
    sam3_unit.clear_prompts()
    sam3_unit.reset_session()

    assert sam3_unit.predictor.handle_request.called


def test_sam3_wrapper_shutdown(sam3_unit):
    from unittest.mock import MagicMock, patch

    sam3_unit.session_id = "test_session"
    sam3_unit.predictor.shutdown = MagicMock()

    with (
        patch("core.managers.sam3.torch.cuda.is_available", return_value=True, create=True),
        patch("core.managers.sam3.torch.cuda.empty_cache") as mock_empty,
    ):
        sam3_unit.shutdown()
        assert mock_empty.called

    assert sam3_unit.session_id is None
    assert sam3_unit.predictor is None


def test_triton_mocking():
    import sys

    from core.utils import _setup_triton_mock

    # Temporarily remove triton from sys.modules
    if "triton" in sys.modules:
        del sys.modules["triton"]

    # Directly test setup_triton_mock logic by ensuring it populates sys.modules
    _setup_triton_mock()
    assert "triton" in sys.modules
    assert hasattr(sys.modules["triton"], "jit")
    assert sys.modules["triton"].jit is not None
