"""
Smoke tests for import validation.

These tests verify that all modules can be imported without mocks,
catching missing imports like Image, gradio, etc. before runtime.

Run with: python -m pytest tests/test_smoke.py -v
"""

import pytest

# Mark all tests as smoke tests (fast, no GPU needed)
pytestmark = pytest.mark.smoke


class TestImportSmoke:
    """Verify all modules import correctly without mocks."""

    def test_import_core_config(self):
        from core import config

        assert config.Config is not None

    def test_import_core_database(self):
        from core import database

        assert database.Database is not None

    def test_import_core_events(self):
        from core import events

        assert events.PreAnalysisEvent is not None

    def test_import_core_export(self):
        from core import export

        assert export.export_kept_frames is not None

    def test_import_core_filtering(self):
        from core import filtering

        assert filtering.apply_all_filters_vectorized is not None

    def test_import_core_logger(self):
        from core import logger

        assert logger.AppLogger is not None

    def test_import_core_models(self):
        from core import models

        assert models.AnalysisParameters is not None

    def test_import_core_pipelines(self):
        from core import pipelines

        assert pipelines.execute_extraction is not None

    def test_import_core_managers(self):
        from core import managers

        assert managers.ModelRegistry is not None

    def test_import_core_utils(self):
        from core import utils

        assert utils.create_frame_map is not None

    def test_import_core_error_handling(self):
        from core import error_handling

        assert error_handling.ErrorHandler is not None

    def test_import_core_progress(self):
        from core import progress

        assert progress.AdvancedProgressTracker is not None

    def test_import_core_batch_manager(self):
        from core import batch_manager

        assert batch_manager.BatchManager is not None

    def test_import_core_scene_utils(self):
        from core import scene_utils

        assert scene_utils.SubjectMasker is not None

    def test_import_ui_app_ui(self):
        from ui import app_ui

        assert app_ui.AppUI is not None

    def test_import_ui_gallery_utils(self):
        from ui import gallery_utils

        assert gallery_utils is not None


class TestCriticalSymbols:
    """Verify critical symbols exist in modules (catches missing imports)."""

    def test_pipelines_has_image(self):
        """PIL.Image was missing - caused NameError."""
        from core import pipelines

        assert hasattr(pipelines, "Image"), "PIL.Image not imported in pipelines.py"

    def test_pipelines_has_gradio(self):
        """gradio was missing - caused NameError."""
        from core import pipelines

        assert hasattr(pipelines, "gr"), "gradio not imported in pipelines.py"

    def test_pipelines_has_torch(self):
        from core import pipelines

        assert hasattr(pipelines, "torch"), "torch not imported in pipelines.py"

    def test_pipelines_has_json(self):
        from core import pipelines

        assert hasattr(pipelines, "json"), "json not imported in pipelines.py"

    def test_export_has_subprocess(self):
        from core import export

        assert hasattr(export, "subprocess"), "subprocess not imported in export.py"


class TestDependencyImports:
    """Verify external dependencies are available."""

    def test_pil_available(self):
        from PIL import Image

        assert Image is not None

    def test_cv2_available(self):
        import cv2

        assert cv2 is not None

    def test_numpy_available(self):
        import numpy as np

        assert np is not None

    def test_gradio_available(self):
        import gradio as gr

        assert gr is not None

    def test_pydantic_available(self):
        import pydantic

        assert pydantic is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
