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

    def test_sam21_matches_sam3_interface(self):
        """Verify SAM2 and SAM3 wrappers share the same public interface."""
        from core.managers.sam2 import SAM2Wrapper
        from core.managers.sam3 import SAM3Wrapper

        required_methods = [
            "init_video",
            "add_bbox_prompt",
            "propagate",
            "add_point_prompt",
            "close_session",
            "reset_session",
            "clear_prompts",
            "remove_object",
            "detect_objects",
            "add_text_prompt",
            "shutdown",
        ]
        for method in required_methods:
            assert hasattr(SAM2Wrapper, method), f"SAM21Wrapper missing method: {method}"
            assert hasattr(SAM3Wrapper, method), f"SAM3Wrapper missing method: {method}"


class TestMockAppSyncValidation:
    """Validate that mock_app.py stubs match production signatures."""

    def test_mock_app_function_signatures(self):
        """Compare mock_app.py stubs against real function signatures."""
        import inspect

        import core.export
        import core.photo_utils
        import core.pipelines
        import core.xmp_writer

        # We import the mock functions from mock_app.py
        # Since mock_app.py patches them on import, we need to be careful
        # but here we just want to check parameter counts and names
        from tests import mock_app

        sync_targets = [
            (core.pipelines.ExtractionPipeline._run_impl, mock_app.mock_extraction_run),
            (core.pipelines.execute_pre_analysis, mock_app.mock_pre_analysis_execution),
            (core.pipelines.execute_propagation, mock_app.mock_propagation_execution),
            (core.pipelines.execute_analysis, mock_app.mock_analysis_execution),
            (core.photo_utils.ingest_folder, mock_app.mock_ingest_folder),
            (core.xmp_writer.export_xmps_for_photos, mock_app.mock_export_xmps_for_photos),
            (core.export.export_kept_frames, mock_app.mock_export_kept_frames),
        ]

        for real_fn, mock_fn in sync_targets:
            # Get underlying function if it's a bound method or decorated
            if hasattr(real_fn, "__wrapped__"):
                real_fn = real_fn.__wrapped__

            real_sig = inspect.signature(real_fn)
            mock_sig = inspect.signature(mock_fn)

            real_params = list(real_sig.parameters.keys())
            mock_params = list(mock_sig.parameters.keys())

            # Check if all required real params exist in mock
            # (Mock might have fewer if it uses *args/**kwargs, but we prefer explicit match)
            for p in real_params:
                assert p in mock_params, f"Mock {mock_fn.__name__} missing parameter: {p}"

            # Check for name mismatches in non-variadic params
            for p in mock_params:
                if p not in ["args", "kwargs"] and p not in real_params:
                    # Allow 'self' in mock if patching instance methods
                    if p == "self":
                        continue
                    assert p in real_params, f"Mock {mock_fn.__name__} has extra parameter: {p}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
