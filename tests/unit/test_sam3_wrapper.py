import numpy as np
import pytest


class TestSAM3WrapperAPICompleteness:
    def test_all_required_methods_exist(self, sam3_unit):
        required_methods = [
            "init_video",
            "add_bbox_prompt",
            "propagate",
            "clear_prompts",
            "detect_objects",
            "add_text_prompt",
            "add_point_prompt",
            "reset_session",
            "close_session",
            "remove_object",
            "shutdown",
        ]

        for method in required_methods:
            assert hasattr(sam3_unit, method), f"SAM3Wrapper missing required method: {method}"
            assert callable(getattr(sam3_unit, method)), f"SAM3Wrapper.{method} is not callable"

    def test_detect_objects_signature(self, sam3_unit):
        import inspect

        method = sam3_unit.detect_objects
        sig = inspect.signature(method)
        params = list(sig.parameters.keys())
        assert "frame_rgb" in params
        assert "prompt" in params

    def test_add_text_prompt_signature(self, sam3_unit):
        import inspect

        method = sam3_unit.add_text_prompt
        sig = inspect.signature(method)
        params = list(sig.parameters.keys())
        assert "frame_idx" in params
        assert "text" in params

    def test_add_point_prompt_signature(self, sam3_unit):
        import inspect

        sig = inspect.signature(sam3_unit.add_point_prompt)
        params = list(sig.parameters.keys())
        assert "frame_idx" in params
        assert "obj_id" in params
        assert "points" in params
        assert "labels" in params
        assert "img_size" in params


class TestSAM3WrapperMethodBehavior:
    def test_detect_objects_returns_empty_on_empty_prompt(self, sam3_unit):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        assert sam3_unit.detect_objects(frame, "") == []
        assert sam3_unit.detect_objects(frame, "   ") == []
        assert sam3_unit.detect_objects(frame, None) == []

    def test_add_text_prompt_requires_init_video(self, sam3_unit):
        sam3_unit.session_id = None
        with pytest.raises(RuntimeError) as exc_info:
            sam3_unit.add_text_prompt(0, "person")
        assert "init_video" in str(exc_info.value)

    def test_add_point_prompt_requires_init_video(self, sam3_unit):
        sam3_unit.session_id = None
        with pytest.raises(RuntimeError) as exc_info:
            sam3_unit.add_point_prompt(0, 1, [(50, 50)], [1], (100, 100))
        assert "init_video" in str(exc_info.value)

    def test_reset_session_with_no_state_is_safe(self, sam3_unit):
        sam3_unit.session_id = None
        sam3_unit.reset_session()

    def test_close_session_with_no_state_is_safe(self, sam3_unit):
        sam3_unit.session_id = None
        sam3_unit.close_session()

    def test_close_session_clears_session_id(self, sam3_unit):
        sam3_unit.session_id = "test_session"
        sam3_unit.close_session()
        assert sam3_unit.session_id is None

    def test_remove_object_calls_predictor(self, sam3_unit):
        sam3_unit.session_id = "test_session"
        sam3_unit.remove_object(1)
        sam3_unit.predictor.handle_request.assert_called_with(
            request={
                "type": "remove_object",
                "session_id": "test_session",
                "obj_id": 1,
            }
        )

    def test_shutdown_cleans_up_resources(self, sam3_unit):
        from unittest.mock import MagicMock, patch

        import torch

        original_predictor = sam3_unit.predictor
        original_predictor.shutdown = MagicMock()

        with (
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(torch.cuda, "empty_cache") as mock_empty_cache,
            patch.object(torch.cuda, "synchronize", create=True),
            patch.object(sam3_unit, "close_session", wraps=sam3_unit.close_session) as mock_close_session,
        ):
            sam3_unit.shutdown()
            mock_close_session.assert_called_once()
            original_predictor.shutdown.assert_called_once()
            assert sam3_unit.predictor is None
            mock_empty_cache.assert_called()


class TestSeedSelectorTrackerInterface:
    def test_seed_selector_tracker_methods_match_wrapper(self, sam3_unit):
        import re
        from pathlib import Path

        seed_selector_path = Path(__file__).resolve().parents[2] / "core" / "scene_utils" / "seed_selector.py"
        content = seed_selector_path.read_text(encoding="utf-8")
        tracker_calls = re.findall(r"self\.tracker\.(\w+)\s*\(", content)
        expected_methods = set(tracker_calls)

        for method in expected_methods:
            assert hasattr(sam3_unit, method), f"SeedSelector calls tracker.{method}() but SAM3Wrapper doesn't have it!"
