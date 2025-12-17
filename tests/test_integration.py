"""
Integration tests for local GPU hardware.

These tests run WITHOUT mocks to catch real integration issues like missing imports,
model loading failures, and GPU compatibility problems.

Run with: python -m pytest tests/test_integration.py -v -s --integration
Skip with: python -m pytest tests/ --ignore=tests/test_integration.py

Requirements:
- CUDA-capable GPU
- All models downloaded
- Full dependencies installed
"""
import pytest
import sys
from pathlib import Path

# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration


class TestImportSmoke:
    """Smoke tests that verify all modules import without mocking."""

    def test_import_core_modules(self):
        """Test that all core modules can be imported."""
        from core import config, database, events, export, filtering, logger, models, pipelines
        from core import managers, utils, error_handling, progress, batch_manager
        from core import scene_utils

    def test_import_ui_modules(self):
        """Test that all UI modules can be imported."""
        from ui import app_ui, gallery_utils

    def test_import_pil(self):
        """Test PIL is available (was missing in pipelines.py)."""
        from PIL import Image
        assert Image is not None

    def test_import_cv2(self):
        """Test OpenCV is available."""
        import cv2
        assert cv2 is not None

    def test_import_torch(self):
        """Test PyTorch is available."""
        import torch
        assert torch is not None

    def test_import_gradio(self):
        """Test Gradio is available (was missing in pipelines.py)."""
        import gradio as gr
        assert gr is not None

    def test_pipelines_has_all_imports(self):
        """Verify pipelines.py has all required imports at module level."""
        from core import pipelines
        # These were missing and caused runtime errors
        assert hasattr(pipelines, 'Image')  # PIL.Image
        assert hasattr(pipelines, 'gr')     # gradio


class TestGPUIntegration:
    """Tests that require GPU hardware."""

    def test_cuda_available(self):
        """Verify CUDA is available and working."""
        import torch
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        assert torch.cuda.device_count() > 0

    def test_sam3_import(self):
        """Test SAM3 can be imported (requires: pip install -e SAM3_repo)."""
        import torch
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        try:
            from sam3.model_builder import build_sam3_video_predictor
            assert build_sam3_video_predictor is not None
        except ImportError as e:
            pytest.skip(f"SAM3 not installed. Run: pip install -e SAM3_repo. Error: {e}")

    def test_insightface_import(self):
        """Test InsightFace can be imported."""
        try:
            import insightface
            assert insightface is not None
        except ImportError as e:
            pytest.skip(f"InsightFace not installed: {e}")

    def test_pyiqa_import(self):
        """Test PyIQA (NIQE) can be imported."""
        try:
            import pyiqa
            assert pyiqa is not None
        except ImportError as e:
            pytest.skip(f"PyIQA not installed: {e}")


class TestConfigIntegration:
    """Tests Config without mocks."""

    def test_config_loads(self, tmp_path):
        """Test Config loads and creates directories."""
        from core.config import Config
        config = Config(logs_dir=str(tmp_path / "logs"))
        assert Path(config.logs_dir).exists()

    def test_config_quality_weights(self):
        """Test quality weights are valid."""
        from core.config import Config
        config = Config()
        weights = config.quality_weights
        assert sum(weights.values()) > 0


class TestModelLoadingIntegration:
    """Tests that model loading works correctly."""

    def test_model_registry_initialization(self):
        """Test ModelRegistry can be initialized."""
        import torch
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        from core.managers import ModelRegistry
        registry = ModelRegistry()
        assert registry is not None

    def test_thumbnail_manager_initialization(self, tmp_path):
        """Test ThumbnailManager can be initialized."""
        from core.config import Config
        from core.logger import AppLogger
        from core.managers import ThumbnailManager

        config = Config(logs_dir=str(tmp_path / "logs"))
        logger = AppLogger(config, log_to_console=False, log_to_file=False)
        tm = ThumbnailManager(logger, config)
        assert tm is not None


class TestPipelineIntegration:
    """Tests pipeline classes without full execution."""

    def test_extraction_pipeline_init(self, tmp_path):
        """Test ExtractionPipeline can be initialized."""
        import threading
        from queue import Queue
        from core.config import Config
        from core.logger import AppLogger
        from core.models import AnalysisParameters
        from core.pipelines import ExtractionPipeline

        config = Config(logs_dir=str(tmp_path / "logs"))
        logger = AppLogger(config, log_to_console=False, log_to_file=False)
        params = AnalysisParameters(source_path="test.mp4", output_folder=str(tmp_path))
        
        pipeline = ExtractionPipeline(config, logger, params, Queue(), threading.Event())
        assert pipeline is not None

    def test_analysis_pipeline_init(self, tmp_path):
        """Test AnalysisPipeline can be initialized."""
        import torch
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        import threading
        from queue import Queue
        from core.config import Config
        from core.logger import AppLogger
        from core.models import AnalysisParameters
        from core.pipelines import AnalysisPipeline
        from core.managers import ThumbnailManager, ModelRegistry

        config = Config(logs_dir=str(tmp_path / "logs"))
        logger = AppLogger(config, log_to_console=False, log_to_file=False)
        
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        
        params = AnalysisParameters(source_path="test.mp4", output_folder=str(output_dir))
        tm = ThumbnailManager(logger, config)
        registry = ModelRegistry(logger)

        pipeline = AnalysisPipeline(config, logger, params, Queue(), threading.Event(), tm, registry)
        assert pipeline is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--integration"])
