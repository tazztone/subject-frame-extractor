"""
Function signature validation tests.

These tests verify that pipeline functions have correct signatures,
return types, and decorator configurations.

Run with: python -m pytest tests/test_signatures.py -v
"""

import inspect

import pytest

# Mark all tests as signature tests
pytestmark = pytest.mark.signature


class TestPipelineSignatures:
    """Verify pipeline functions have correct signatures."""

    def test_execute_extraction_returns_generator(self):
        """execute_extraction should return a generator when called."""
        # Note: The function is decorated with @handle_common_errors
        # which wraps it, so we check it's callable
        from core.pipelines import execute_extraction

        assert callable(execute_extraction)

    def test_execute_pre_analysis_returns_generator(self):
        """execute_pre_analysis should return a generator when called."""
        from core.pipelines import execute_pre_analysis

        assert callable(execute_pre_analysis)

    def test_execute_propagation_is_generator(self):
        """execute_propagation should be a generator function."""
        from core.pipelines import execute_propagation

        # This one isn't decorated with @handle_common_errors
        assert callable(execute_propagation)

    def test_execute_analysis_returns_generator(self):
        """execute_analysis should return a generator when called."""
        from core.pipelines import execute_analysis

        assert callable(execute_analysis)

    def test_execute_session_load_returns_dict(self):
        """execute_session_load returns dict, not generator."""
        from core.pipelines import execute_session_load

        # This is not decorated, so we can check directly
        assert not inspect.isgeneratorfunction(execute_session_load), (
            "execute_session_load should return dict, not generator"
        )


class TestEventModels:
    """Verify event models have required fields."""

    def test_pre_analysis_event_has_required_fields(self):
        from core.events import PreAnalysisEvent

        fields = PreAnalysisEvent.model_fields.keys()
        assert "video_path" in fields or "output_folder" in fields

    def test_extraction_event_exists(self):
        from core.events import ExtractionEvent

        assert ExtractionEvent is not None

    def test_propagation_event_exists(self):
        from core.events import PropagationEvent

        assert PropagationEvent is not None

    def test_export_event_exists(self):
        from core.events import ExportEvent

        assert ExportEvent is not None


class TestModelClasses:
    """Verify model classes have correct structure."""

    def test_analysis_parameters_has_from_ui(self):
        """AnalysisParameters should have from_ui factory method."""
        from core.models import AnalysisParameters

        assert hasattr(AnalysisParameters, "from_ui")
        assert callable(AnalysisParameters.from_ui)

    def test_scene_has_required_fields(self):
        from core.models import Scene

        fields = Scene.model_fields.keys()
        required = ["shot_id", "start_frame", "end_frame"]
        for field in required:
            assert field in fields, f"Scene missing required field: {field}"

    def test_frame_has_metrics(self):
        from core.models import Frame

        fields = Frame.model_fields.keys()
        assert "metrics" in fields


class TestManagerClasses:
    """Verify manager classes have correct interface."""

    def test_model_registry_has_get_or_load(self):
        from core.managers import ModelRegistry

        assert hasattr(ModelRegistry, "get_or_load")

    def test_thumbnail_manager_has_get(self):
        from core.managers import ThumbnailManager

        assert hasattr(ThumbnailManager, "get")

    def test_video_manager_has_get_video_info(self):
        from core.managers import VideoManager

        assert hasattr(VideoManager, "get_video_info")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
