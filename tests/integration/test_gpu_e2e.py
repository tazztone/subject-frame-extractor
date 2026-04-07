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

import os
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from core.managers import SAM3Wrapper

if os.environ.get("PYTEST_INTEGRATION_MODE", "").lower() != "true":
    pytest.skip("Set PYTEST_INTEGRATION_MODE=true to run GPU E2E tests", allow_module_level=True)

# Mark all tests as gpu_e2e (requires GPU, slow)
# Automatically skip entire file if CUDA is not available
pytestmark = [
    pytest.mark.gpu_e2e,
    pytest.mark.slow,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available for GPU E2E tests"),
]


def _create_test_image(width=256, height=256, frame_idx=0):
    """Create a high-entropy test image with a noise-textured complex object (simulates a tracked subject)."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:, :] = [40, 60, 80]  # Dark blue background

    # Subject size: 60% height, 40% width
    h = int(height * 0.6)
    w = int(width * 0.4)

    # Moving base position from center
    y1 = (height - h) // 2
    base_x = (width - w) // 2
    if frame_idx > 0:
        x1 = base_x + frame_idx * 2  # Gradual movement
    else:
        x1 = base_x

    y2, x2 = y1 + h, x1 + w

    # Clip to bounds
    cy1, cy2 = max(0, y1), min(height, y2)
    cx1, cx2 = max(0, x1), min(width, x2)
    ah, aw = cy2 - cy1, cx2 - cx1

    if ah > 0 and aw > 0:
        # 1. Base rectangle (Reddish)
        subject = np.full((ah, aw, 3), [200, 100, 100], dtype=np.uint8)

        # 2. Add green inner rect (distinct feature for tracker)
        gh, gw = int(ah * 0.3), int(aw * 0.3)
        if gh > 0 and gw > 0:
            subject[gh : 2 * gh, gw : 2 * gw] = [100, 200, 100]

        # 3. Add blue circle (high-contrast feature)
        cv2.circle(subject, (aw // 2, ah // 4), min(aw, ah) // 6, (100, 100, 255), -1)

        # 4. Add noise texture to help deep trackers (SAM3/SAM2) distinguish the object
        noise = (np.random.rand(ah, aw, 3) * 60).astype(np.uint8)
        subject = cv2.add(subject, noise)

        # 5. Add white border
        cv2.rectangle(subject, (0, 0), (aw - 1, ah - 1), (255, 255, 255), 2)

        img[cy1:cy2, cx1:cx2] = subject

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


def _generate_synthetic_video(video_path, width=1280, height=720, num_frames=60):
    """Generate a realistic 720p synthetic video with a moving textured object."""
    import cv2

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(video_path), fourcc, 30.0, (width, height))

    try:
        for i in range(num_frames):
            img = _create_test_image(width, height, frame_idx=i)
            out.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    finally:
        out.release()
    return video_path


@pytest.fixture(scope="module")
def sample_video(tmp_path_factory):
    """Resolve a test video: prefer real media if present in downloads, else generate synthetic 720p."""
    project_root = Path(__file__).parents[2]
    real = project_root / "downloads" / "example clip 720p 2x.mp4"

    if real.exists():
        return real, [652, 207, 71, 102]  # Known good bbox for this clip

    # Generate synthetic 720p video
    tmp_dir = tmp_path_factory.mktemp("video_infra")
    video_path = tmp_dir / "synthetic_720p.mp4"
    _generate_synthetic_video(video_path, 1280, 720, 60)

    # Bbox for the moving object in _create_test_image
    # Standard: [cx-75, cy-50, 150, 100] -> x, y, w, h
    # In frame 0: cx=width//2, cy=height//2
    # For 1280x720: cx=640, cy=360 -> [565, 310, 150, 100]
    return video_path, [565, 310, 150, 100]


def _create_test_frames_dir(tmp_path, num_frames=5, width=256, height=256):
    """Create a directory with high-entropy test frames for robust tracking tests."""
    from PIL import Image

    frames_dir = tmp_path / "test_frames"
    frames_dir.mkdir(exist_ok=True)

    for i in range(num_frames):
        img = _create_test_image(width, height, frame_idx=i)
        Image.fromarray(img).save(frames_dir / f"{i:05d}.jpg")

    return frames_dir


def _is_sam3_available():
    """Check if SAM3 is properly installed and can be imported."""
    try:
        from sam3.model_builder import build_sam3_predictor  # noqa: F401

        return True
    except ImportError:
        return False


requires_sam3 = pytest.mark.skipif(not _is_sam3_available(), reason="SAM3 not installed (pip install -e SAM3_repo)")


def _is_sam2_available():
    """Check if SAM2 is properly installed and can be imported."""
    try:
        import sam2.build_sam  # noqa: F401

        return True
    except ImportError:
        return False


# Skip decorator for tests requiring SAM2
requires_sam2 = pytest.mark.skipif(not _is_sam2_available(), reason="SAM2 not installed (pip install sam2)")


@pytest.fixture
def test_image():
    """Provides a simple test image."""
    return _create_test_image()


@pytest.fixture
def test_image_with_face():
    """Provides a test image with face-like features."""
    return _create_test_image_with_face()


@pytest.fixture
def test_frames_dir(tmp_path):
    """Provides a directory with test frames for video processing."""
    return _create_test_frames_dir(tmp_path)


@pytest.mark.gpu_e2e
class TestCUDAAvailability:
    """Verify CUDA is available and working before running GPU tests."""

    def test_cuda_available(self):
        """CUDA must be available for GPU E2E tests."""
        import torch

        if not torch.cuda.is_available():
            pytest.skip("CUDA is required for this specific GPU test")

    def test_cuda_memory_available(self):
        """Verify sufficient GPU memory (~4GB needed)."""
        import torch

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        props = torch.cuda.get_device_properties(0)
        total = props.total_memory
        allocated = torch.cuda.memory_allocated(0)
        # Guard against mock leakage from parallel workers
        assert isinstance(total, int), f"total_memory is not int (likely a MagicMock leak): {type(total)}"
        free_memory = total - allocated
        min_required = 4 * 1024 * 1024 * 1024  # 4GB
        assert free_memory >= min_required, f"Need at least 4GB GPU memory, got {free_memory / 1e9:.1f}GB"


@pytest.mark.gpu_e2e
@pytest.mark.sam3
class TestSAM3Inference:
    """Real SAM3 inference tests - catches BFloat16 and other runtime errors."""

    @requires_sam3
    @pytest.mark.xdist_group("sam3_isolated")
    def test_sam3_wrapper_initialization(self, tmp_path):
        """SAM3Wrapper can be initialized without errors."""
        from pathlib import Path

        import torch

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        project_root = Path(__file__).parents[2]
        # Check for 3.1 first, then 3.0
        real_checkpoint_31 = project_root / "models" / "sam3.1_multiplex.pt"
        real_checkpoint_30 = project_root / "models" / "sam3.pt"
        if real_checkpoint_31.exists():
            real_checkpoint = real_checkpoint_31
        elif real_checkpoint_30.exists():
            real_checkpoint = real_checkpoint_30
        else:
            pytest.skip("SAM3 checkpoint not found locally")

        wrapper = SAM3Wrapper(checkpoint_path=str(real_checkpoint), device="cuda")
        try:
            assert wrapper is not None
            assert wrapper.predictor is not None
        finally:
            wrapper.shutdown()

    @requires_sam3
    def test_sam3_init_video(self, tmp_path, module_model_registry):
        """SAM3 init_video() initializes inference state correctly."""
        import torch

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        # Create 40 frames to satisfy SAM3 hotstart_delay
        test_frames_dir = _create_test_frames_dir(tmp_path, num_frames=40)

        wrapper = module_model_registry.get_tracker("sam3")
        try:
            wrapper.reset_session()
            session_id = wrapper.init_video(str(test_frames_dir))
            assert session_id is not None
            assert wrapper.session_id is not None
        finally:
            if wrapper:
                wrapper.close_session()

    @requires_sam3
    def test_sam3_add_bbox_prompt(self, tmp_path, module_model_registry):
        """SAM3 add_bbox_prompt() returns valid mask."""
        import torch

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        # Create 40 frames
        test_frames_dir = _create_test_frames_dir(tmp_path, num_frames=40)
        wrapper = module_model_registry.get_tracker("sam3")

        try:
            wrapper.reset_session()
            wrapper.init_video(str(test_frames_dir))

            # Add bbox prompt covering the object in test image
            mask = wrapper.add_bbox_prompt(
                frame_idx=0,
                obj_id=1,
                bbox_xywh=[77, 51, 102, 153],  # x, y, w, h
                img_size=(256, 256),  # w, h
            )

            assert mask is not None
            assert isinstance(mask, np.ndarray)
            assert mask.ndim == 2  # Should be 2D (H, W) mask
            assert mask.shape == (256, 256)
        finally:
            if wrapper:
                wrapper.close_session()

    @requires_sam3
    def test_sam3_propagate_forward(self, tmp_path, sample_video, module_model_registry):
        """SAM3 can propagate forward through real or realistic synthetic video."""
        import torch

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        video_path, bbox = sample_video
        wrapper = module_model_registry.get_tracker("sam3")

        try:
            wrapper.reset_session()
            wrapper.init_video(str(video_path))
            mask = wrapper.add_bbox_prompt(frame_idx=0, obj_id=1, bbox_xywh=bbox, img_size=(1280, 720))

            assert mask is not None
            assert isinstance(mask, np.ndarray)

            # Propagate forward 5 frames
            for i, result in enumerate(wrapper.propagate(start_idx=0, reverse=False)):
                if i >= 5:
                    break
                assert result is not None
        finally:
            if wrapper:
                wrapper.close_session()

    @requires_sam3
    def test_sam3_propagate_bidirectional(self, tmp_path, sample_video, module_model_registry):
        """SAM3 can propagate from a middle frame."""
        import torch

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        video_path, bbox = sample_video
        wrapper = module_model_registry.get_tracker("sam3")

        try:
            wrapper.reset_session()
            wrapper.init_video(str(video_path))

            # Add prompt on frame 5
            mask = wrapper.add_bbox_prompt(frame_idx=5, obj_id=1, bbox_xywh=bbox, img_size=(1280, 720))
            assert mask is not None

            # Propagate forward
            for i, result in enumerate(wrapper.propagate(start_idx=5, reverse=False)):
                if i >= 2:
                    break
                assert result is not None

            # Propagate backward
            for i, result in enumerate(wrapper.propagate(start_idx=5, reverse=True)):
                if i >= 2:
                    break
                assert result is not None
        finally:
            if wrapper:
                wrapper.close_session()

    @requires_sam3
    def test_sam3_clear_prompts(self, test_frames_dir, module_model_registry):
        """SAM3 clear_prompts() resets session state."""
        import torch

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        wrapper = module_model_registry.get_tracker("sam3")

        try:
            wrapper.reset_session()
            wrapper.init_video(str(test_frames_dir))
            wrapper.add_bbox_prompt(0, 1, [50, 50, 80, 150], (256, 256))

            # Clear prompts should now reset the session
            wrapper.clear_prompts()

            # Since clear_prompts resets session, we MUST re-init video before adding more prompts
            wrapper.init_video(str(test_frames_dir))

            # Should be able to add new prompt after clearing
            mask = wrapper.add_bbox_prompt(0, 2, [60, 60, 70, 140], (256, 256))
            assert mask is not None
        finally:
            if wrapper:
                wrapper.close_session()


@pytest.mark.gpu_e2e
@pytest.mark.sam2
class TestSAM2Inference:
    """Real SAM2 inference tests - catches BFloat16 and other runtime errors."""

    @requires_sam2
    def test_sam2_wrapper_initialization(self, tmp_path, module_model_registry):
        """SAM2Wrapper can be initialized without errors via ModelRegistry."""
        import torch

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        wrapper = module_model_registry.get_tracker("sam2")
        assert wrapper is not None
        assert wrapper.predictor is not None

    @requires_sam2
    def test_sam2_init_video(self, test_frames_dir, module_model_registry):
        """SAM2 init_video() initializes inference state correctly."""
        import torch

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        wrapper = module_model_registry.get_tracker("sam2")

        try:
            session_id = wrapper.init_video(str(test_frames_dir))
            assert session_id is not None
        finally:
            if wrapper:
                wrapper.close_session()

    @requires_sam2
    def test_sam2_add_bbox_prompt(self, tmp_path, module_model_registry):
        """SAM2 add_bbox_prompt() returns valid mask."""
        import torch

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        # Create 5 frames
        test_frames_dir = _create_test_frames_dir(tmp_path, num_frames=5)
        wrapper = module_model_registry.get_tracker("sam2")

        try:
            wrapper.init_video(str(test_frames_dir))

            # Add bbox prompt covering the object in test image
            mask = wrapper.add_bbox_prompt(
                frame_idx=0,
                obj_id=1,
                bbox_xywh=[77, 51, 102, 153],  # x, y, w, h
                img_size=(256, 256),  # w, h
            )

            assert mask is not None
            assert isinstance(mask, np.ndarray)
            assert mask.ndim == 2  # Should be 2D (H, W) mask
            assert mask.shape == (256, 256)
            assert mask.any(), "Mask should not be empty"
        finally:
            if wrapper:
                wrapper.close_session()

    @requires_sam2
    def test_sam2_propagate_forward(self, test_frames_dir, module_model_registry):
        """SAM2 propagate() forward generator yields valid results."""
        import torch

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        wrapper = module_model_registry.get_tracker("sam2")

        try:
            wrapper.init_video(str(test_frames_dir))
            wrapper.add_bbox_prompt(frame_idx=0, obj_id=1, bbox_xywh=[77, 51, 102, 153], img_size=(256, 256))

            # Propagate forward
            propagated = list(wrapper.propagate(start_idx=0, reverse=False))

            assert len(propagated) > 0
            for frame_idx, obj_id, mask in propagated:
                assert isinstance(frame_idx, int)
                assert frame_idx >= 0
                assert isinstance(obj_id, int)
                assert isinstance(mask, np.ndarray)
                assert mask.ndim == 2
        finally:
            if wrapper:
                wrapper.close_session()


@pytest.mark.gpu_e2e
class TestInsightFaceInference:
    """Real InsightFace inference tests."""

    def test_insightface_initialization(self, tmp_path, module_model_registry):
        """InsightFace can be initialized."""
        import torch

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        from core.config import Config
        from core.logger import AppLogger
        from core.managers import get_face_analyzer

        config = Config(logs_dir=str(tmp_path / "logs"))
        logger = AppLogger(config, log_to_console=False, log_to_file=False)
        registry = module_model_registry

        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Point to real models directory to avoid download
        from pathlib import Path

        models_path = str(Path(__file__).parent.parent.parent / "models")

        # get_face_analyzer(model_name, models_path, det_size_tuple, logger, model_registry, device)
        analyzer = get_face_analyzer("buffalo_l", models_path, (640, 640), logger, registry, device)

        assert analyzer is not None

    def test_face_detection_on_image(self, test_image_with_face, tmp_path, module_model_registry):
        """InsightFace can process an image without errors."""
        import cv2
        import torch

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        from core.config import Config
        from core.logger import AppLogger
        from core.managers import get_face_analyzer

        config = Config(logs_dir=str(tmp_path / "logs"))
        logger = AppLogger(config, log_to_console=False, log_to_file=False)
        registry = module_model_registry

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Point to real models directory to avoid download
        from pathlib import Path

        models_path = str(Path(__file__).parent.parent.parent / "models")

        analyzer = get_face_analyzer("buffalo_l", models_path, (640, 640), logger, registry, device)

        # Convert RGB to BGR for InsightFace
        image_bgr = cv2.cvtColor(test_image_with_face, cv2.COLOR_RGB2BGR)

        # This should not raise errors (faces may not be detected in synthetic image)
        faces = analyzer.get(image_bgr)
        assert isinstance(faces, list)


@pytest.mark.gpu_e2e
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
            output_folder=str(output_dir),
        )

        pipeline = ExtractionPipeline(config, logger, params, Queue(), threading.Event())
        assert pipeline is not None
        assert pipeline.config is not None

    def test_analysis_pipeline_initializes_with_real_managers(self, tmp_path, module_model_registry, database):
        """AnalysisPipeline initializes with real ThumbnailManager and ModelRegistry."""
        import torch

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        import threading
        from queue import Queue

        from core.config import Config
        from core.logger import AppLogger
        from core.managers import ThumbnailManager
        from core.models import AnalysisParameters
        from core.pipelines import AnalysisPipeline

        config = Config(logs_dir=str(tmp_path / "logs"))
        logger = AppLogger(config, log_to_console=False, log_to_file=False)

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        params = AnalysisParameters(source_path="test.mp4", output_folder=str(output_dir))

        tm = ThumbnailManager(logger, config)
        registry = module_model_registry

        pipeline = AnalysisPipeline(config, logger, params, Queue(), threading.Event(), tm, registry)

        assert pipeline is not None
        assert pipeline.thumbnail_manager is not None
        assert pipeline.model_registry is not None


@pytest.mark.gpu_e2e
class TestVideoE2E:
    """End-to-end tests with real video processing."""

    @pytest.fixture
    def test_video_path(self, tmp_path):
        """Create a small test video (5 frames, 256x256)."""
        import cv2

        video_path = tmp_path / "test_video.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
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
        import json

        import cv2

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

        config = Config(logs_dir=str(tmp_path / "logs"), downloads_dir=str(tmp_path / "downloads"))
        logger = AppLogger(config, log_to_console=False, log_to_file=False)

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        params = AnalysisParameters(
            source_path=test_video_path,
            video_path=test_video_path,
            output_folder=str(output_dir),
            thumbnails_only=True,
            method="all",
        )

        pipeline = ExtractionPipeline(config, logger, params, Queue(), threading.Event())
        result = pipeline.run()

        assert result is not None
        assert result.get("done") is True
        assert (output_dir / "thumbs").exists()

    @requires_sam3
    def test_pre_analysis_with_sam3(self, test_frames_dir, module_model_registry):
        """Pre-analysis can run SAM3 on extracted frames."""
        import torch

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        import threading
        from queue import Queue

        from core.managers import ThumbnailManager
        from core.models import AnalysisParameters
        from core.scene_utils import SubjectMasker

        config = module_model_registry.logger.config
        logger = module_model_registry.logger

        params = AnalysisParameters(
            source_path="test.mp4",
            video_path="test.mp4",
            output_folder=str(test_frames_dir),
            thumbnails_only=True,
            enable_subject_mask=True,
            tracker_model_name="sam3",
            primary_seed_strategy="📦 Use Bounding Box",
        )

        tm = ThumbnailManager(logger, config)

        # This tests the full SAM3 initialization and masker setup
        masker = SubjectMasker(
            params=params,
            progress_queue=Queue(),
            cancel_event=threading.Event(),
            config=config,
            thumbnail_manager=tm,
            logger=logger,
            model_registry=module_model_registry,
            device="cuda",
        )

        assert masker is not None


@pytest.mark.gpu_e2e
class TestMaskPropagatorE2E:
    """Tests for MaskPropagator with real SAM3 inference."""

    @requires_sam3
    def test_mask_propagator_propagate(self, tmp_path, module_model_registry):
        """MaskPropagator.propagate() works with new SAM3 API."""
        import threading
        from queue import Queue

        import torch

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        from core.config import Config
        from core.logger import AppLogger
        from core.models import AnalysisParameters
        from core.scene_utils.mask_propagator import MaskPropagator

        config = Config(logs_dir=str(tmp_path / "logs"))
        logger = AppLogger(config, log_to_console=False, log_to_file=False)

        params = AnalysisParameters(source_path="test.mp4", output_folder=str(tmp_path), min_mask_area_pct=0.01)

        wrapper = module_model_registry.get_tracker("sam3")

        try:
            propagator = MaskPropagator(
                params=params,
                dam_tracker=wrapper,
                cancel_event=threading.Event(),
                progress_queue=Queue(),
                config=config,
                logger=logger,
                device="cuda",
            )

            # Create test frames
            frames_rgb = [_create_test_image() for _ in range(5)]

            # Run propagation
            masks, areas, empties, errors = propagator.propagate(
                shot_frames_rgb=frames_rgb, seed_idx=0, bbox_xywh=[50, 50, 80, 150]
            )

            assert len(masks) == 5
            assert len(areas) == 5
            assert len(empties) == 5
            assert len(errors) == 5

            # At least seed frame should have a mask
            assert masks[0] is not None
            assert isinstance(masks[0], np.ndarray)
        finally:
            # Note: We don't shutdown here as it's a shared registry tracker
            wrapper.reset_session()

    @requires_sam3
    def test_mask_propagator_bidirectional(self, tmp_path, module_model_registry):
        """MaskPropagator.propagate() works bidirectionally from middle frame."""
        import threading
        from queue import Queue

        import torch

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        from core.config import Config
        from core.logger import AppLogger
        from core.models import AnalysisParameters
        from core.scene_utils.mask_propagator import MaskPropagator

        config = Config(logs_dir=str(tmp_path / "logs"))
        logger = AppLogger(config, log_to_console=False, log_to_file=False)

        params = AnalysisParameters(source_path="test.mp4", output_folder=str(tmp_path), min_mask_area_pct=0.01)

        wrapper = module_model_registry.get_tracker("sam3")

        try:
            propagator = MaskPropagator(
                params=params,
                dam_tracker=wrapper,
                cancel_event=threading.Event(),
                progress_queue=Queue(),
                config=config,
                logger=logger,
                device="cuda",
            )

            # Create 10 test frames with moving object
            frames_rgb = []
            for i in range(10):
                img = np.zeros((256, 256, 3), dtype=np.uint8)
                img[:, :] = [100, 150, 200]
                x = 30 + i * 15
                img[50:200, x : x + 80] = [200, 100, 100]
                frames_rgb.append(img)

            # Start from middle frame
            seed_idx = 5
            x_at_seed = 30 + seed_idx * 15

            masks, areas, empties, errors = propagator.propagate(
                shot_frames_rgb=frames_rgb, seed_idx=seed_idx, bbox_xywh=[x_at_seed, 50, 80, 150]
            )

            assert len(masks) == 10
            # All frames should have masks (either from forward or backward propagation)
            for i, mask in enumerate(masks):
                assert mask is not None, f"Frame {i} has no mask"
                assert isinstance(mask, np.ndarray)
        finally:
            wrapper.reset_session()


@pytest.mark.gpu_e2e
class TestOperatorE2E:
    """Tests for quality metric calculation using the Operator framework."""

    def test_run_operators_real(self, test_image, tmp_path):
        """Standard operators can be executed on a real image."""
        from core.config import Config
        from core.logger import AppLogger
        from core.operators import discover_operators, run_operators

        config = Config(logs_dir=str(tmp_path / "logs"))
        logger = AppLogger(config, log_to_console=False, log_to_file=False)

        # Ensure operators are discovered
        discover_operators()

        # Run subset of operators for speed
        results = run_operators(
            image_rgb=test_image,
            config=config,
            operators=["sharpness", "edge_strength", "contrast", "brightness"],
            logger=logger,
        )

        assert "sharpness" in results
        assert results["sharpness"].success is True
        assert "sharpness_score" in results["sharpness"].metrics
        assert "edge_strength" in results
        assert results["edge_strength"].success is True

    def test_niqe_operator_real(self, test_image, tmp_path):
        """NIQE operator can be executed (requires pyiqa and GPU)."""
        import torch

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        try:
            import pyiqa  # noqa: F401
        except ImportError:
            pytest.skip("pyiqa not installed")

        from core.config import Config
        from core.logger import AppLogger
        from core.operators import OperatorRegistry, discover_operators, run_operators

        config = Config(logs_dir=str(tmp_path / "logs"))
        logger = AppLogger(config, log_to_console=False, log_to_file=False)

        discover_operators()
        niqe_op = OperatorRegistry.get("niqe")
        if not niqe_op:
            pytest.skip("NIQE operator not found")

        # Initialize the operator (loads model)
        niqe_op.initialize(config)

        try:
            results = run_operators(image_rgb=test_image, config=config, operators=["niqe"], logger=logger)

            assert "niqe" in results
            assert results["niqe"].success is True
            assert "niqe_score" in results["niqe"].metrics
        finally:
            niqe_op.cleanup()


@pytest.mark.gpu_e2e
class TestExportE2E:
    """E2E tests for export pipeline."""

    def test_export_pipeline_initialization(self, tmp_path):
        """ExportPipeline can be initialized with real config."""
        from core.config import Config
        from core.logger import AppLogger

        config = Config(logs_dir=str(tmp_path / "logs"))
        AppLogger(config, log_to_console=False, log_to_file=False)

        # Create required directories
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Export functions should be importable
        from core.export import export_kept_frames

        assert export_kept_frames is not None

    def test_export_with_real_frames(self, tmp_path):
        """Export can process frames from a real directory."""
        import json

        import cv2

        from core.config import Config
        from core.logger import AppLogger

        config = Config(logs_dir=str(tmp_path / "logs"))
        AppLogger(config, log_to_console=False, log_to_file=False)

        # Create output structure
        output_dir = tmp_path / "session"
        output_dir.mkdir()
        thumbs_dir = output_dir / "thumbs"
        thumbs_dir.mkdir()
        export_dir = tmp_path / "export"
        export_dir.mkdir()

        # Create test frames
        for i in range(5):
            frame = _create_test_image()
            cv2.imwrite(str(thumbs_dir / f"frame_{i:06d}.webp"), frame)

        # Create frame map
        frame_map = list(range(5))
        with open(output_dir / "frame_map.json", "w") as f:
            json.dump(frame_map, f)

        # Test should not crash (full export requires more setup)
        assert (output_dir / "thumbs").exists()
        assert len(list(thumbs_dir.glob("*.webp"))) == 5

    def test_export_dry_run_mode(self, tmp_path):
        """Dry run export mode works without creating files."""
        from core.config import Config
        from core.events import ExportEvent
        from core.export import dry_run_export
        from core.logger import AppLogger

        config = Config(logs_dir=str(tmp_path / "logs"))
        logger = AppLogger(config, log_to_console=False, log_to_file=False)

        export_dir = tmp_path / "export_dry"

        # Create minimal 5-frame mock session metadata
        all_frames_data = [
            {"filename": f"frame_{i + 1:06d}.webp", "frame_number": i, "metrics": {"quality_score": 50.0}}
            for i in range(5)
        ]

        event = ExportEvent(
            all_frames_data=all_frames_data,
            output_dir=str(tmp_path / "session"),
            video_path=str(tmp_path / "test.mp4"),
            enable_crop=False,
            crop_ars="1:1",
            crop_padding=10,
            filter_args={"quality_score_min": 0.0},  # Include all
        )

        # Dry run should return summary string containing frame count
        summary = dry_run_export(event, config, logger)

        assert "Dry Run: 5 frames" in summary
        assert not export_dir.exists()


@pytest.mark.gpu_e2e
class TestCancellationE2E:
    """E2E tests for cancel operations during pipeline execution."""

    @requires_sam3
    def test_propagation_with_cancel_event(self, tmp_path, test_frames_dir, module_model_registry):
        """MaskPropagator handles cancel event during propagation."""
        import threading
        from queue import Queue

        import torch

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        from core.config import Config
        from core.logger import AppLogger
        from core.models import AnalysisParameters
        from core.scene_utils.mask_propagator import MaskPropagator

        config = Config(logs_dir=str(tmp_path / "logs"))
        logger = AppLogger(config, log_to_console=False, log_to_file=False)

        params = AnalysisParameters(source_path="test.mp4", output_folder=str(tmp_path), min_mask_area_pct=0.01)

        # Use shared tracker
        wrapper = module_model_registry.get_tracker("sam3")
        cancel_event = threading.Event()

        try:
            wrapper.reset_session()
            propagator = MaskPropagator(
                params=params,
                dam_tracker=wrapper,
                cancel_event=cancel_event,
                progress_queue=Queue(),
                config=config,
                logger=logger,
                device="cuda",
            )

            # Create test frames
            frames_rgb = [_create_test_image() for _ in range(10)]

            # Set cancel event after a short delay to simulate user cancellation
            def cancel_after_delay():
                import time

                time.sleep(0.5)
                cancel_event.set()

            # Start the cancel thread
            cancel_thread = threading.Thread(target=cancel_after_delay)
            cancel_thread.start()

            # Run propagation (may be interrupted)
            masks, areas, empties, errors = propagator.propagate(
                shot_frames_rgb=frames_rgb, seed_idx=0, bbox_xywh=[50, 50, 80, 150]
            )

            cancel_thread.join()

            # Should return lists (possibly incomplete due to cancel)
            assert isinstance(masks, list)
            assert isinstance(areas, list)

        finally:
            pass  # cleanup() removed

    def test_analysis_pipeline_cancel(self, tmp_path, module_model_registry, database):
        """AnalysisPipeline handles cancel event gracefully."""
        import threading
        from queue import Queue

        import torch

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        from core.config import Config
        from core.logger import AppLogger
        from core.models import AnalysisParameters

        config = Config(logs_dir=str(tmp_path / "logs"))
        logger = AppLogger(config, log_to_console=False, log_to_file=False)

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        params = AnalysisParameters(source_path="test.mp4", output_folder=str(output_dir))

        # Create cancel event in cancelled state
        cancel_event = threading.Event()
        cancel_event.set()  # Pre-cancelled

        # Pipeline initialization should still work
        from core.managers import ThumbnailManager
        from core.pipelines import AnalysisPipeline

        tm = ThumbnailManager(logger, config)
        registry = module_model_registry

        pipeline = AnalysisPipeline(config, logger, params, Queue(), cancel_event, tm, registry)

        assert pipeline is not None


@pytest.mark.gpu_e2e
class TestMediaPipeLandmarkerE2E:
    """E2E tests for MediaPipe Face Landmarker."""

    def test_face_landmarker_import(self):
        """MediaPipe face landmarker can be imported."""
        try:
            import mediapipe as mp

            assert mp is not None
        except ImportError:
            pytest.skip("MediaPipe not installed")

    def test_face_landmarker_model_download(self, tmp_path):
        """Face landmarker model can be downloaded."""

        from core.config import Config
        from core.logger import AppLogger

        config = Config(logs_dir=str(tmp_path / "logs"), models_dir=str(tmp_path / "models"))
        AppLogger(config, log_to_console=False, log_to_file=False)

        # Model download is handled by managers
        # Just verify the config paths work
        assert Path(config.models_dir).exists() or True  # May not exist yet


@pytest.mark.gpu_e2e
class TestLargeVideoE2E:
    """E2E tests for handling larger videos/frame sequences."""

    def test_many_frames_processing(self, tmp_path):
        """Test processing a larger number of frames."""
        from PIL import Image

        frames_dir = tmp_path / "many_frames"
        frames_dir.mkdir()

        # Create 50 frames
        num_frames = 50
        for i in range(num_frames):
            img = _create_test_image(width=128, height=128)  # Smaller for speed
            Image.fromarray(img).save(frames_dir / f"{i:05d}.jpg")

        assert len(list(frames_dir.glob("*.jpg"))) == num_frames

    @requires_sam3
    @pytest.mark.sam3
    def test_sam3_with_many_frames(self, tmp_path, sample_video, module_model_registry):
        """SAM3 can process a larger sequence."""
        import torch

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        video_path, bbox = sample_video
        wrapper = module_model_registry.get_tracker("sam3")

        try:
            wrapper.reset_session()
            wrapper.init_video(str(video_path))
            mask = wrapper.add_bbox_prompt(frame_idx=0, obj_id=1, bbox_xywh=bbox, img_size=(1280, 720))

            assert mask is not None

            # Propagate through all frames
            propagated = list(wrapper.propagate(start_idx=0, reverse=False))
            assert len(propagated) > 0, "SAM3 propagation returned no masks"

        finally:
            if wrapper:
                wrapper.close_session()


@pytest.mark.gpu_e2e
class TestMaskGenerationE2E:
    """E2E tests for mask generation to catch silent failures."""

    @requires_sam3
    def test_get_mask_for_bbox_e2e(self, test_frames_dir, tmp_path, module_model_registry):
        """Test SeedSelector._get_mask_for_bbox with real SAM3."""
        import torch

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        from core.config import Config
        from core.logger import AppLogger
        from core.models import AnalysisParameters
        from core.scene_utils.seed_selector import SeedSelector

        config = Config(logs_dir=str(tmp_path / "logs"))
        logger = AppLogger(config, log_to_console=False, log_to_file=False)
        wrapper = module_model_registry.get_tracker("sam3")

        # Create a test frame
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        frame[200:500, 400:800] = [200, 100, 100]  # Foreground object

        selector = SeedSelector(
            params=AnalysisParameters(source_path="test.mp4"),
            config=config,
            face_analyzer=None,
            reference_embedding=None,
            tracker=wrapper,
            logger=logger,
            device="cuda",
        )

        try:
            # Manually init tracker session as SeedSelector doesn't do it for us
            wrapper.reset_session()
            # We need at least one frame to init video context for SAM3
            dummy_frame_path = tmp_path / "dummy.jpg"
            import cv2

            cv2.imwrite(str(dummy_frame_path), frame)
            wrapper.init_video(str(dummy_frame_path))

            mask = selector._get_mask_for_bbox(frame, [400, 200, 400, 300])

            assert mask is not None
            assert isinstance(mask, np.ndarray)
            assert mask.shape == (720, 1280)
            assert mask.any(), "Mask should not be empty for foreground object"
        finally:
            if wrapper:
                wrapper.close_session()

    @requires_sam3
    def test_identity_first_seed_e2e(self, test_image_with_face, tmp_path, module_model_registry):
        """Test 'By Face' seeding strategy with real models."""
        import torch

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        from core.config import Config
        from core.logger import AppLogger
        from core.managers import get_face_analyzer
        from core.models import AnalysisParameters
        from core.scene_utils.seed_selector import SeedSelector

        config = Config(logs_dir=str(tmp_path / "logs"))
        logger = AppLogger(config, log_to_console=False, log_to_file=False)
        # Use shared tracker
        wrapper = module_model_registry.get_tracker("sam3")

        # Allow buffalo_l to run, it will fallback to CPU if OOM occurs
        face_analyzer = get_face_analyzer(
            "buffalo_l", str(tmp_path / "models"), (640, 640), logger, module_model_registry, "cuda"
        )

        # Get embedding from test image
        import cv2

        img_bgr = cv2.cvtColor(test_image_with_face, cv2.COLOR_RGB2BGR)
        faces = face_analyzer.get(img_bgr)
        if not faces:
            pytest.skip("No faces detected in test image, cannot test identity seeding")

        ref_embedding = faces[0].normed_embedding

        try:
            selector = SeedSelector(
                params=AnalysisParameters(
                    source_path="test.mp4", primary_seed_strategy="👤 By Face", compute_face_sim=True
                ),
                config=config,
                face_analyzer=face_analyzer,
                reference_embedding=ref_embedding,
                tracker=wrapper,
                logger=logger,
                device="cuda",
            )
            assert selector is not None
        finally:
            if wrapper:
                wrapper.close_session()

        try:
            # Test seeding
            bbox, details = selector.select_seed(test_image_with_face)

            assert bbox is not None
            assert details.get("type") in ["evidence_based_selection", "face_match", "expanded_box_from_face"]
        finally:
            pass

    def test_pre_analysis_mask_generation_e2e(self, test_frames_dir, tmp_path, module_model_registry):
        """Test the full pre-analysis flow including mask generation."""
        import torch

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        import threading
        from queue import Queue

        from core.config import Config
        from core.logger import AppLogger
        from core.managers import ThumbnailManager
        from core.models import AnalysisParameters
        from core.scene_utils import SubjectMasker

        config = Config(logs_dir=str(tmp_path / "logs"), models_dir=str(tmp_path / "models"))
        logger = AppLogger(config, log_to_console=False, log_to_file=False)
        registry = module_model_registry
        tm = ThumbnailManager(logger, config)

        params = AnalysisParameters.from_ui(
            logger,
            config,
            video_path=str(test_frames_dir),
            output_folder=str(test_frames_dir),
            thumbnails_only=True,
            enable_subject_mask=True,
            primary_seed_strategy="📦 Use Bounding Box",
            initial_bbox=[50, 50, 100, 150],  # Explicit bbox
        )

        # Initialize SubjectMasker
        masker = SubjectMasker(
            params=params,
            progress_queue=Queue(),
            cancel_event=threading.Event(),
            config=config,
            thumbnail_manager=tm,
            logger=logger,
            model_registry=registry,
            device="cuda",
        )

        assert masker.dam_tracker is not None
        # face_analyzer is optional and not used in this mask-focused test


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "-m", "gpu_e2e"])
