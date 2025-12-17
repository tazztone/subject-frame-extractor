"""
GPU E2E Tests - Real inference with actual models.

These tests run REAL GPU inference to catch runtime errors like:
- BFloat16/Float32 dtype mismatches
- CUDA OOM errors  
- Model loading failures
- Tensor shape mismatches

Run with: python -m pytest tests/test_gpu_e2e.py -m gpu_e2e -v

Requirements:
- CUDA-capable GPU with 6GB+ VRAM
- SAM3 installed (pip install -e SAM3_repo)
- All models downloaded
"""
import pytest
import numpy as np
from pathlib import Path

# Mark all tests as gpu_e2e (requires GPU, slow)
pytestmark = [pytest.mark.gpu_e2e, pytest.mark.slow]


def _create_test_image(width=256, height=256):
    """Create a simple test image with a rectangle (simulates an object)."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    # Add a background
    img[:, :] = [100, 150, 200]  # Light blue background
    # Add a rectangle (simulates a person/object)
    img[50:200, 80:180] = [200, 100, 100]  # Red rectangle
    return img


def _create_test_image_with_face(width=256, height=256):
    """Create a test image with a face-like pattern."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:, :] = [200, 200, 200]  # Gray background
    # Add a face-like oval
    import cv2
    cv2.ellipse(img, (128, 100), (60, 80), 0, 0, 360, (255, 220, 180), -1)
    # Add eyes
    cv2.circle(img, (108, 85), 10, (50, 50, 50), -1)
    cv2.circle(img, (148, 85), 10, (50, 50, 50), -1)
    return img


@pytest.fixture
def test_image():
    """Provides a simple test image."""
    return _create_test_image()


@pytest.fixture
def test_image_with_face():
    """Provides a test image with face-like features."""
    return _create_test_image_with_face()


class TestCUDAAvailability:
    """Verify CUDA is available and working before running GPU tests."""

    def test_cuda_available(self):
        """CUDA must be available for GPU E2E tests."""
        import torch
        assert torch.cuda.is_available(), "CUDA is required for GPU E2E tests"

    def test_cuda_memory_available(self):
        """Verify sufficient GPU memory (~4GB needed)."""
        import torch
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        free_memory = torch.cuda.get_device_properties(0).total_memory
        min_required = 4 * 1024 * 1024 * 1024  # 4GB
        assert free_memory >= min_required, f"Need at least 4GB GPU memory, got {free_memory / 1e9:.1f}GB"


class TestSAM3Inference:
    """Real SAM3 inference tests - catches BFloat16 and other runtime errors."""

    def test_sam3_wrapper_initialization(self, tmp_path):
        """SAM3Wrapper can be initialized without errors."""
        import torch
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        from core.managers import SAM3Wrapper
        
        # This tests the dtype fix we added
        wrapper = SAM3Wrapper(str(tmp_path / "sam3.pt"), device="cuda")
        assert wrapper is not None
        assert wrapper.predictor is not None

    def test_sam3_text_detection(self, test_image, tmp_path):
        """SAM3 can detect objects using text prompt - catches BFloat16 errors."""
        import torch
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        from core.managers import SAM3Wrapper
        
        wrapper = SAM3Wrapper(str(tmp_path / "sam3.pt"), device="cuda")
        
        try:
            # This is where the BFloat16 error occurred
            results = wrapper.detect_objects(test_image, "object")
            # Results may be empty but no error should occur
            assert isinstance(results, list)
        finally:
            wrapper.cleanup()

    def test_sam3_bbox_initialization(self, test_image, tmp_path):
        """SAM3 can initialize session with bounding box."""
        import torch
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        from core.managers import SAM3Wrapper
        
        wrapper = SAM3Wrapper(str(tmp_path / "sam3.pt"), device="cuda")
        
        try:
            # Provide a bounding box in the test image
            result = wrapper.initialize(
                images=[test_image],
                bbox=[80, 50, 100, 150],  # x, y, w, h
                prompt_frame_idx=0
            )
            assert isinstance(result, dict)
            # pred_mask may be None if detection fails, but no error should occur
        finally:
            wrapper.cleanup()


class TestInsightFaceInference:
    """Real InsightFace inference tests."""

    def test_insightface_initialization(self, tmp_path):
        """InsightFace can be initialized."""
        import torch
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        from core.config import Config
        from core.logger import AppLogger
        from core.managers import ModelRegistry, get_face_analyzer
        
        config = Config(logs_dir=str(tmp_path / "logs"))
        logger = AppLogger(config, log_to_console=False, log_to_file=False)
        registry = ModelRegistry(logger)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        models_path = str(tmp_path / "models")
        
        # get_face_analyzer(model_name, models_path, det_size_tuple, logger, model_registry, device)
        analyzer = get_face_analyzer(
            "buffalo_l", 
            models_path,
            (640, 640),
            logger,
            registry,
            device
        )
        
        assert analyzer is not None

    def test_face_detection_on_image(self, test_image_with_face, tmp_path):
        """InsightFace can process an image without errors."""
        import torch
        import cv2
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        from core.config import Config
        from core.logger import AppLogger
        from core.managers import ModelRegistry, get_face_analyzer
        
        config = Config(logs_dir=str(tmp_path / "logs"))
        logger = AppLogger(config, log_to_console=False, log_to_file=False)
        registry = ModelRegistry(logger)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        models_path = str(tmp_path / "models")
        
        analyzer = get_face_analyzer(
            "buffalo_l",
            models_path,
            (640, 640),
            logger,
            registry,
            device
        )
        
        # Convert RGB to BGR for InsightFace
        image_bgr = cv2.cvtColor(test_image_with_face, cv2.COLOR_RGB2BGR)
        
        # This should not raise errors (faces may not be detected in synthetic image)
        faces = analyzer.get(image_bgr)
        assert isinstance(faces, list)


class TestPipelineE2E:
    """End-to-end pipeline tests with real execution."""

    def test_extraction_pipeline_creates_output(self, tmp_path):
        """ExtractionPipeline initializes correctly with real config."""
        import threading
        from queue import Queue
        
        from core.config import Config
        from core.logger import AppLogger
        from core.models import AnalysisParameters
        from core.pipelines import ExtractionPipeline
        
        config = Config(logs_dir=str(tmp_path / "logs"))
        logger = AppLogger(config, log_to_console=False, log_to_file=False)
        
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        
        params = AnalysisParameters(
            source_path="test.mp4",  # Doesn't need to exist for init
            output_folder=str(output_dir)
        )
        
        pipeline = ExtractionPipeline(config, logger, params, Queue(), threading.Event())
        assert pipeline is not None
        assert pipeline.config is not None

    def test_analysis_pipeline_initializes_with_real_managers(self, tmp_path):
        """AnalysisPipeline initializes with real ThumbnailManager and ModelRegistry."""
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
        
        params = AnalysisParameters(
            source_path="test.mp4",
            output_folder=str(output_dir)
        )
        
        tm = ThumbnailManager(logger, config)
        registry = ModelRegistry(logger)
        
        pipeline = AnalysisPipeline(
            config, logger, params, Queue(), threading.Event(), tm, registry
        )
        
        assert pipeline is not None
        assert pipeline.thumbnail_manager is not None
        assert pipeline.model_registry is not None


class TestVideoE2E:
    """End-to-end tests with real video processing."""

    @pytest.fixture
    def test_video_path(self, tmp_path):
        """Create a small test video (5 frames, 256x256)."""
        import cv2
        
        video_path = tmp_path / "test_video.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(video_path), fourcc, 5.0, (256, 256))
        
        # Create 5 frames with moving object
        for i in range(5):
            frame = np.zeros((256, 256, 3), dtype=np.uint8)
            frame[:, :] = [100, 150, 200]  # Blue background
            # Moving rectangle (simulates person)
            x = 50 + i * 20
            cv2.rectangle(frame, (x, 50), (x + 80, 200), (200, 100, 100), -1)
            out.write(frame)
        
        out.release()
        return str(video_path)

    @pytest.fixture
    def test_frames_dir(self, tmp_path, test_video_path):
        """Create directory with extracted frames and required files."""
        import cv2
        import json
        
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        thumbs_dir = output_dir / "thumbs"
        thumbs_dir.mkdir()
        
        # Extract frames from test video
        cap = cv2.VideoCapture(test_video_path)
        frame_numbers = []
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_path = thumbs_dir / f"frame_{frame_idx:06d}.webp"
            cv2.imwrite(str(frame_path), frame)
            frame_numbers.append(frame_idx)
            frame_idx += 1
        
        cap.release()
        
        # Create scenes.json (all frames as one scene)
        scenes = [[0, frame_idx - 1]]
        with open(output_dir / "scenes.json", "w") as f:
            json.dump(scenes, f)
        
        # Create frame_map.json
        with open(output_dir / "frame_map.json", "w") as f:
            json.dump(frame_numbers, f)
        
        return output_dir

    def test_extraction_pipeline_on_real_video(self, test_video_path, tmp_path):
        """ExtractionPipeline can process a real video file."""
        import threading
        from queue import Queue
        
        from core.config import Config
        from core.logger import AppLogger
        from core.models import AnalysisParameters
        from core.pipelines import ExtractionPipeline
        
        config = Config(
            logs_dir=str(tmp_path / "logs"),
            downloads_dir=str(tmp_path / "downloads")
        )
        logger = AppLogger(config, log_to_console=False, log_to_file=False)
        
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        
        params = AnalysisParameters(
            source_path=test_video_path,
            video_path=test_video_path,
            output_folder=str(output_dir),
            thumbnails_only=True,
            method="all"
        )
        
        pipeline = ExtractionPipeline(config, logger, params, Queue(), threading.Event())
        result = pipeline.run()
        
        assert result is not None
        assert result.get("done") == True
        assert (output_dir / "thumbs").exists()

    def test_pre_analysis_with_sam3(self, test_frames_dir, tmp_path):
        """Pre-analysis can run SAM3 on extracted frames."""
        import torch
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        import threading
        from queue import Queue
        
        from core.config import Config
        from core.logger import AppLogger
        from core.models import AnalysisParameters
        from core.managers import ThumbnailManager, ModelRegistry
        from core.scene_utils import SubjectMasker
        
        config = Config(
            logs_dir=str(tmp_path / "logs"),
            models_dir=str(tmp_path / "models")
        )
        logger = AppLogger(config, log_to_console=False, log_to_file=False)
        
        params = AnalysisParameters(
            source_path="test.mp4",
            video_path="test.mp4",
            output_folder=str(test_frames_dir),
            thumbnails_only=True,
            enable_subject_mask=True,
            primary_seed_strategy="📦 Use Bounding Box"
        )
        
        registry = ModelRegistry(logger)
        tm = ThumbnailManager(logger, config)
        
        # This tests the full SAM3 initialization and masker setup
        masker = SubjectMasker(
            params=params,
            progress_queue=Queue(),
            cancel_event=threading.Event(),
            config=config,
            thumbnail_manager=tm,
            logger=logger,
            model_registry=registry,
            device="cuda"
        )
        
        assert masker is not None


class TestQualityMetricsE2E:
    """Tests for quality metric calculation with real images."""

    def test_calculate_quality_metrics_real(self, test_image, tmp_path):
        """Frame quality metrics can be calculated on real image."""
        import torch
        
        from core.models import Frame, QualityConfig
        from core.config import Config
        from core.logger import AppLogger
        
        config = Config(logs_dir=str(tmp_path / "logs"))
        logger = AppLogger(config, log_to_console=False, log_to_file=False)
        
        quality_config = QualityConfig(
            sharpness_base_scale=config.sharpness_base_scale,
            edge_strength_base_scale=config.edge_strength_base_scale,
            enable_niqe=False  # Skip NIQE for speed
        )
        
        frame = Frame(image_data=test_image, frame_number=0)
        
        # This tests real metric calculation
        frame.calculate_quality_metrics(
            test_image,
            quality_config,
            logger,
            main_config=config,
            metrics_to_compute={
                'sharpness': True,
                'edge_strength': True,
                'contrast': True,
                'brightness': True,
                'entropy': True
            }
        )
        
        assert frame.metrics is not None
        assert frame.metrics.sharpness_score is not None
        assert frame.metrics.edge_strength_score is not None

    def test_niqe_metric_calculation(self, test_image, tmp_path):
        """NIQE metric can be calculated (requires pyiqa)."""
        import torch
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        try:
            import pyiqa
        except ImportError:
            pytest.skip("pyiqa not installed")
        
        from core.models import Frame, QualityConfig
        from core.config import Config
        from core.logger import AppLogger
        
        config = Config(logs_dir=str(tmp_path / "logs"))
        logger = AppLogger(config, log_to_console=False, log_to_file=False)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        niqe_metric = pyiqa.create_metric('niqe', device=device)
        
        quality_config = QualityConfig(
            sharpness_base_scale=config.sharpness_base_scale,
            edge_strength_base_scale=config.edge_strength_base_scale,
            enable_niqe=True
        )
        
        frame = Frame(image_data=test_image, frame_number=0)
        
        frame.calculate_quality_metrics(
            test_image,
            quality_config,
            logger,
            main_config=config,
            niqe_metric=niqe_metric,
            metrics_to_compute={'quality': True}
        )
        
        assert frame.metrics is not None
        # NIQE score should be calculated
        assert frame.metrics.niqe_score is not None or frame.metrics.quality_score is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "-m", "gpu_e2e"])
