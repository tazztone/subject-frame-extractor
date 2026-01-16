from __future__ import annotations

import gc
import logging
import threading
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

import cv2
import lpips
import numpy as np
import torch
import yt_dlp as ytdlp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image

# SAM3 imports
# On Windows, Triton is not available. We need to mock it BEFORE importing SAM3
# because sam3/model/edt.py imports triton at module level.
build_sam3_video_predictor = None
Sam3VideoPredictor = None


def _setup_triton_mock():
    """Create a mock triton module if triton is not available (Windows)."""
    import sys

    try:
        import triton  # noqa: F401

        return False  # Triton is available, no mock needed
    except ImportError:
        pass

    # Create mock triton module with proper __spec__ for PyTorch compatibility
    import types
    from importlib.machinery import ModuleSpec
    from unittest.mock import MagicMock

    # Create a proper module object (not just MagicMock)
    mock_triton = types.ModuleType("triton")
    mock_triton.__spec__ = ModuleSpec("triton", None)
    mock_triton.__path__ = []
    mock_triton.__file__ = "<mock>"
    mock_triton.language = types.ModuleType("triton.language")
    mock_triton.language.__spec__ = ModuleSpec("triton.language", None)
    mock_triton.jit = lambda fn: fn  # Decorator that returns function unchanged

    # Create a mock for tl (triton.language) attributes
    class MockTL:
        constexpr = lambda x: x
        program_id = MagicMock(return_value=0)
        load = MagicMock(return_value=0)
        store = MagicMock()

    for attr in dir(MockTL):
        if not attr.startswith("_"):
            setattr(mock_triton.language, attr, getattr(MockTL, attr))

    sys.modules["triton"] = mock_triton
    sys.modules["triton.language"] = mock_triton.language
    return True


# Apply triton mock before SAM3 import
_triton_mocked = _setup_triton_mock()

try:
    from sam3.model_builder import build_sam3_video_predictor  # noqa: F401

    from core import sam3_patches

    # Apply patches to replace triton functions with CPU fallbacks
    if _triton_mocked:
        sam3_patches.apply_patches()
except ImportError as e:
    # This might fail if run in isolation without path setup or missing dependencies
    logging.getLogger(__name__).warning(f"Failed to import SAM3 dependencies: {e}")

if TYPE_CHECKING:
    from insightface.app import FaceAnalysis

    from core.config import Config
    from core.logger import AppLogger
    from core.models import AnalysisParameters

from core.error_handling import ErrorHandler
from core.utils import download_model, validate_video_file


class ThumbnailManager:
    """Manages an in-memory LRU cache for image thumbnails."""

    # TODO: Add memory-based eviction (keep simple: LRU or similar)
    def __init__(self, logger: "AppLogger", config: "Config"):
        """Initializes the manager with a configurable cache size."""
        self.logger = logger
        self.config = config
        self.cache = OrderedDict()
        self.max_size = self.config.cache_size
        self.logger.info(f"ThumbnailManager initialized with cache size {self.max_size}")

    def get(self, thumb_path: Path) -> Optional[np.ndarray]:
        """Retrieves a thumbnail from cache or loads it from disk."""
        if not isinstance(thumb_path, Path):
            thumb_path = Path(thumb_path)
        if thumb_path in self.cache:
            self.cache.move_to_end(thumb_path)
            return self.cache[thumb_path]
        if not thumb_path.exists():
            return None
        if len(self.cache) > self.max_size * self.config.cache_cleanup_threshold:
            self._cleanup_old_entries()
        try:
            with Image.open(thumb_path) as pil_thumb:
                thumb_img = np.array(pil_thumb.convert("RGB"))
            self.cache[thumb_path] = thumb_img
            while len(self.cache) > self.max_size:
                self.cache.popitem(last=False)
            return thumb_img
        except Exception as e:
            self.logger.warning("Failed to load thumbnail with Pillow", extra={"path": str(thumb_path), "error": e})
            return None

    def clear_cache(self):
        """Clears the thumbnail cache and triggers garbage collection."""
        self.cache.clear()
        gc.collect()

    def _cleanup_old_entries(self):
        num_to_remove = int(self.max_size * self.config.cache_eviction_factor)
        for _ in range(num_to_remove):
            if not self.cache:
                break
            self.cache.popitem(last=False)


class ModelRegistry:
    """
    Thread-safe registry for lazy loading and caching of heavy ML models.
    """

    # TODO: Add model unloading for memory-constrained environments
    def __init__(self, logger: Optional["AppLogger"] = None):
        self._models: Dict[str, Any] = {}
        self._locks: Dict[str, threading.Lock] = defaultdict(threading.Lock)
        self.logger = logger or logging.getLogger(__name__)
        self.runtime_device_override: Optional[str] = None

    def get_or_load(self, key: str, loader_fn: Callable[[], Any]) -> Any:
        """Retrieves a model by key, loading it via loader_fn if not present."""
        if key not in self._models:
            with self._locks[key]:
                if key not in self._models:
                    if self.logger:
                        self.logger.info(f"Loading model '{key}' for the first time...")
                    try:
                        val = loader_fn()
                        # print(f"DEBUG: ModelRegistry loaded {key} -> {val}")
                        self._models[key] = val
                    except Exception as e:
                        # print(f"DEBUG: ModelRegistry failed to load {key}: {e}")
                        raise e
                    if self.logger:
                        self.logger.success(f"Model '{key}' loaded successfully.")
        return self._models[key]

    def clear(self):
        """Clears all loaded models from the registry."""
        if self.logger:
            self.logger.info("Clearing all models from the registry.")
        self._models.clear()

    def get_tracker(
        self, model_name: str, models_path: str, user_agent: str, retry_params: tuple, config: "Config"
    ) -> Optional["SAM3Wrapper"]:
        """
        Gets or loads the SAM3 tracker, handling CPU fallback on CUDA OOM.
        """
        key = f"tracker_{model_name}"

        def _loader():
            device = self.runtime_device_override or ("cuda" if torch.cuda.is_available() else "cpu")
            try:
                return self._load_tracker_impl(model_name, models_path, user_agent, retry_params, device, config)
            except RuntimeError as e:
                if "out of memory" in str(e) and device == "cuda":
                    self.logger.warning("CUDA OOM during tracker init. Switching to CPU for this session.")
                    torch.cuda.empty_cache()
                    self.runtime_device_override = "cpu"
                    return self._load_tracker_impl(model_name, models_path, user_agent, retry_params, "cpu", config)
                raise e

        try:
            return self.get_or_load(key, _loader)
        except Exception as e:
            self.logger.error(f"Failed to initialize tracker: {e}", exc_info=True)
            return None

    def _load_tracker_impl(
        self, model_name: str, models_path: str, user_agent: str, retry_params: tuple, device: str, config: "Config"
    ) -> "SAM3Wrapper":
        if device == "cuda" and not torch.cuda.is_available():
            self.logger.warning(
                "CUDA not available, SAM3 requires CUDA. Attempting to run on CPU (might be slow/fail).",
                component="tracker",
            )

        checkpoint_path = Path(models_path) / "sam3.pt"
        if not checkpoint_path.exists():
            self.logger.info(f"Downloading SAM3 model to {checkpoint_path}...", component="tracker")
            download_model(
                url=config.sam3_checkpoint_url,
                dest_path=checkpoint_path,
                description="SAM3 Model",
                logger=self.logger,
                error_handler=ErrorHandler(self.logger, *retry_params),
                user_agent=user_agent,
                expected_sha256=config.sam3_checkpoint_sha256,
                token=config.huggingface_token,
            )

        self.logger.info(f"Loading SAM3 model on {device}...", component="tracker")
        return SAM3Wrapper(str(checkpoint_path), device=device)


class SAM3Wrapper:
    """
    SAM3 Tracker using official Sam3VideoPredictor API.

    Refactored to use the high-level request/response API.
    """

    def __init__(self, checkpoint_path=None, device="cuda"):
        """
        Initialize SAM3 wrapper using build_sam3_video_predictor.

        Args:
            checkpoint_path: Optional path to checkpoint (auto-downloads from HF if None)
            device: Device to run on ('cuda' or 'cpu')
        """
        from sam3.model_builder import build_sam3_video_predictor

        self.device = device
        self._checkpoint_path = checkpoint_path

        # Setup optimal precision for Ampere GPUs
        if torch.cuda.is_available():
            torch.set_float32_matmul_precision("high")
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

        # Build predictor (handles multi-GPU internally if configured)
        # Note: We currently default to using all available GPUs or CPU based on env
        gpus_to_use = range(torch.cuda.device_count()) if device == "cuda" else None
        self.predictor = build_sam3_video_predictor(gpus_to_use=gpus_to_use)

        # Session state
        self.session_id = None
        self._temp_dir = None

    def init_video(self, video_path: str):
        """
        Initialize inference session with video or frame directory.

        Args:
            video_path: Path to video file or directory of JPEG frames

        Returns:
            session_id
        """
        if self.session_id is not None:
            try:
                self.close_session()
            except Exception:
                pass

        response = self.predictor.handle_request(
            request=dict(
                type="start_session",
                resource_path=str(video_path),
            )
        )
        self.session_id = response["session_id"]
        return self.session_id

    def add_bbox_prompt(self, frame_idx: int, obj_id: int, bbox_xywh: list, img_size: tuple) -> np.ndarray:
        """
        Add bounding box prompt at specified frame.

        Args:
            frame_idx: Frame index to add prompt
            obj_id: Unique object ID (any integer)
            bbox_xywh: Bounding box as [x, y, width, height]
            img_size: Image dimensions as (width, height)

        Returns:
            Initial mask as numpy array (H, W)
        """
        if self.session_id is None:
            raise RuntimeError("Must call init_video() before adding prompts")

        w, h = img_size
        x, y, bw, bh = bbox_xywh

        # Convert to relative coordinates (0-1) in xywh format
        rel_box = [x / w, y / h, bw / w, bh / h]

        response = self.predictor.handle_request(
            request=dict(
                type="add_prompt",
                session_id=self.session_id,
                frame_index=frame_idx,
                obj_id=obj_id,
                bounding_boxes=np.array([rel_box], dtype=np.float32),
                bounding_box_labels=np.array([1], dtype=np.int32),
            )
        )

        # Response contains 'outputs': {'out_binary_masks': ..., 'out_obj_ids': ...}
        outputs = response.get("outputs", {})
        masks = outputs.get("out_binary_masks")
        obj_ids = outputs.get("out_obj_ids")

        # Find mask for our obj_id
        if masks is not None and len(masks) > 0:
            if hasattr(masks, "cpu"):
                masks = masks.cpu().numpy()

            if obj_ids is not None:
                if hasattr(obj_ids, "cpu"):
                    obj_ids = obj_ids.cpu().numpy()
                try:
                    idx = list(obj_ids).index(obj_id)
                    mask = masks[idx]
                    if mask.ndim == 3:
                        mask = mask[0]
                    return mask > 0
                except ValueError:
                    pass

            mask = masks[0]
            if mask.ndim == 3:
                mask = mask[0]
            return mask > 0

        return np.zeros((h, w), dtype=bool)

    def propagate(self, start_idx: int = 0, max_frames: int = None, reverse: bool = False):
        """
        Generator yielding masks for each frame during propagation.

        Args:
            start_idx: Frame index to start propagation from
            max_frames: Maximum frames to propagate
            reverse: If True, propagate backward from start_idx

        Yields:
            Tuple of (frame_idx, obj_id, mask) where mask is numpy array
        """
        if self.session_id is None:
            return

        for response in self.predictor.handle_stream_request(
            request=dict(
                type="propagate_in_video",
                session_id=self.session_id,
                start_frame_index=start_idx,
                max_frame_num_to_track=max_frames or 9999,
                reverse=reverse,
            )
        ):
            frame_idx = response.get("frame_index")
            outputs = response.get("outputs", {})
            masks = outputs.get("out_binary_masks")
            obj_ids = outputs.get("out_obj_ids")

            if masks is None or obj_ids is None:
                continue

            if hasattr(masks, "cpu"):
                masks = masks.cpu().numpy()
            if hasattr(obj_ids, "cpu"):
                obj_ids = obj_ids.cpu().numpy()

            for i, obj_id in enumerate(obj_ids):
                mask = masks[i]
                if mask.ndim == 3:
                    mask = mask[0]
                yield frame_idx, obj_id, mask > 0

    def clear_prompts(self):
        """Reset all prompts in current session."""
        if self.session_id:
            self.predictor.handle_request(
                request=dict(
                    type="reset_session",
                    session_id=self.session_id,
                )
            )

    def detect_objects(self, frame_rgb: np.ndarray, prompt: str) -> list:
        """
        Detect objects in a single frame using text prompt.
        """
        if not prompt or not prompt.strip():
            return []

        try:
            from sam3.model.sam3_image_processor import Sam3Processor

            model = getattr(self.predictor, "model", None)
            detector = getattr(model, "detector", None) if model else getattr(self.predictor, "detector", None)

            processor = Sam3Processor(model=detector, resolution=1008, device=self.device, confidence_threshold=0.3)

            state = processor.set_image(frame_rgb)
            state = processor.set_text_prompt(prompt, state)

            results = []
            if "boxes" in state and "scores" in state:
                boxes = state["boxes"].cpu().numpy()
                scores = state["scores"].cpu().numpy()

                for i, (box, score) in enumerate(zip(boxes, scores)):
                    results.append({"bbox": box.tolist(), "conf": float(score), "type": "text_prompt"})

            return results

        except Exception as e:
            logging.getLogger(__name__).warning(f"detect_objects failed: {e}")
            return []

    def add_text_prompt(self, frame_idx: int, text: str) -> dict:
        """
        Add text prompt for video object detection using new API.
        """
        if self.session_id is None:
            raise RuntimeError("Must call init_video() before add_text_prompt()")

        if not text or not text.strip():
            return {"obj_ids": [], "masks": [], "boxes": []}

        try:
            response = self.predictor.handle_request(
                request=dict(
                    type="add_prompt",
                    session_id=self.session_id,
                    frame_index=frame_idx,
                    text=text,
                )
            )

            outputs = response.get("outputs", {})
            return {
                "obj_ids": outputs.get("out_obj_ids", []),
                "masks": outputs.get("out_binary_masks", []),
                "boxes": outputs.get("out_boxes_xywh", []),
                "probs": outputs.get("out_probs", []),
            }
        except Exception as e:
            logging.getLogger(__name__).warning(f"add_text_prompt failed: {e}")
            return {"obj_ids": [], "masks": [], "boxes": []}

    def add_point_prompt(self, frame_idx: int, obj_id: int, points: list, labels: list, img_size: tuple) -> np.ndarray:
        """
        Add point prompts for mask refinement using new API.
        """
        if self.session_id is None:
            raise RuntimeError("Must call init_video() before add_point_prompt()")

        w, h = img_size

        rel_points = np.array([[x / w, y / h] for x, y in points], dtype=np.float32)
        point_labels = np.array(labels, dtype=np.int32)

        try:
            response = self.predictor.handle_request(
                request=dict(
                    type="add_prompt",
                    session_id=self.session_id,
                    frame_index=frame_idx,
                    obj_id=obj_id,
                    points=rel_points,
                    point_labels=point_labels,
                )
            )

            outputs = response.get("outputs", {})
            masks = outputs.get("out_binary_masks")
            obj_ids = outputs.get("out_obj_ids")

            if masks is not None and len(masks) > 0:
                if hasattr(masks, "cpu"):
                    masks = masks.cpu().numpy()

                if obj_ids is not None:
                    if hasattr(obj_ids, "cpu"):
                        obj_ids = obj_ids.cpu().numpy()
                    try:
                        idx = list(obj_ids).index(obj_id)
                        mask = masks[idx]
                        if mask.ndim == 3:
                            mask = mask[0]
                        return mask > 0
                    except ValueError:
                        pass

                mask = masks[0]
                if mask.ndim == 3:
                    mask = mask[0]
                return mask > 0

            return np.zeros((h, w), dtype=bool)

        except Exception as e:
            logging.getLogger(__name__).warning(f"add_point_prompt failed: {e}")
            return np.zeros((h, w), dtype=bool)

    def remove_object(self, obj_id: int):
        """
        Remove an object from the tracking session.
        """
        if self.session_id is None:
            return

        try:
            self.predictor.handle_request(
                request=dict(
                    type="remove_object",
                    session_id=self.session_id,
                    obj_id=obj_id,
                )
            )
        except Exception as e:
            logging.getLogger(__name__).warning(f"remove_object failed: {e}")

    def reset_session(self):
        """
        Reset all prompts and results (clears session state).
        """
        if self.session_id is not None:
            try:
                self.predictor.handle_request(
                    request=dict(
                        type="reset_session",
                        session_id=self.session_id,
                    )
                )
            except Exception as e:
                logging.getLogger(__name__).warning(f"reset_session failed: {e}")

    def close_session(self):
        """
        Close the inference session and free GPU resources.
        """
        if self.session_id is not None:
            try:
                self.predictor.handle_request(
                    request=dict(
                        type="close_session",
                        session_id=self.session_id,
                    )
                )
            except Exception:
                pass
            self.session_id = None

        # Clean up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def shutdown(self):
        """Shutdown the predictor and free multi-GPU resources."""
        # TODO: Add proper cleanup of all GPU resources
        if hasattr(self.predictor, "shutdown"):
            self.predictor.shutdown()


thread_local = threading.local()


def get_face_landmarker(model_path: str, logger: "AppLogger") -> vision.FaceLandmarker:
    """
    Returns a thread-local MediaPipe FaceLandmarker instance.
    """

    if hasattr(thread_local, "face_landmarker_instance"):
        return thread_local.face_landmarker_instance
    logger.info("Initializing MediaPipe FaceLandmarker for new thread.", component="face_landmarker")
    try:
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1,
            min_face_detection_confidence=0.3,
            min_face_presence_confidence=0.3,
        )
        detector = vision.FaceLandmarker.create_from_options(options)
        thread_local.face_landmarker_instance = detector
        logger.success("Face landmarker model initialized successfully for this thread.")
        return detector
    except Exception as e:
        logger.error(f"Could not initialize MediaPipe face landmarker model. Error: {e}", component="face_landmarker")
        raise RuntimeError("Could not initialize MediaPipe face landmarker model.") from e


# We need model_registry to be accessible.
# In app.py it was a global. Here it will be passed or instantiated.
# But get_face_analyzer used global model_registry in app.py.
# I should change get_face_analyzer to accept model_registry.


def get_face_analyzer(
    model_name: str,
    models_path: str,
    det_size_tuple: tuple,
    logger: "AppLogger",
    model_registry: "ModelRegistry",
    device: str = "cpu",
) -> "FaceAnalysis":
    """
    Gets or loads the InsightFace FaceAnalysis app, with OOM handling.
    """
    from insightface.app import FaceAnalysis

    model_key = f"face_analyzer_{model_name}_{device}_{det_size_tuple}"

    def _loader():
        logger.info(f"Loading face model: {model_name} on device: {device}")
        try:
            is_cuda = device == "cuda"
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if is_cuda else ["CPUExecutionProvider"]
            analyzer = FaceAnalysis(name=model_name, root=models_path, providers=providers)
            analyzer.prepare(ctx_id=0 if is_cuda else -1, det_size=det_size_tuple)
            logger.success(f"Face model loaded with {'CUDA' if is_cuda else 'CPU'}.")
            return analyzer
        except Exception as e:
            if "out of memory" in str(e) and device == "cuda":
                torch.cuda.empty_cache()
                logger.warning("CUDA OOM, retrying with CPU...")
                try:
                    analyzer = FaceAnalysis(name=model_name, root=models_path, providers=["CPUExecutionProvider"])
                    analyzer.prepare(ctx_id=-1, det_size=det_size_tuple)
                    return analyzer
                except Exception as cpu_e:
                    logger.error(f"CPU fallback also failed: {cpu_e}")
            raise RuntimeError(f"Could not initialize face analysis model. Error: {e}") from e

    return model_registry.get_or_load(model_key, _loader)


def get_lpips_metric(model_name: str = "alex", device: str = "cpu") -> torch.nn.Module:
    """Returns the LPIPS metric model."""
    return lpips.LPIPS(net=model_name).to(device)


def initialize_analysis_models(
    params: "AnalysisParameters", config: "Config", logger: "AppLogger", model_registry: "ModelRegistry"
) -> dict:
    """
    Initializes all necessary analysis models based on parameters.

    Returns:
        Dictionary of initialized models (face_analyzer, ref_emb, etc.).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    face_analyzer, ref_emb, face_landmarker = None, None, None

    if params.enable_face_filter:
        face_analyzer = get_face_analyzer(
            model_name=params.face_model_name,
            models_path=str(config.models_dir),
            det_size_tuple=tuple(config.model_face_analyzer_det_size),
            logger=logger,
            model_registry=model_registry,
            device=device,
        )
        if face_analyzer and params.face_ref_img_path:
            ref_path = Path(params.face_ref_img_path)
            if ref_path.exists() and ref_path.is_file():
                try:
                    ref_img = cv2.imread(str(ref_path))
                    if ref_img is not None:
                        faces = face_analyzer.get(ref_img)
                        if faces:
                            ref_emb = max(faces, key=lambda x: x.det_score).normed_embedding
                            logger.info("Reference face embedding created successfully.")
                        else:
                            logger.warning("No face found in reference image.", extra={"path": ref_path})
                    else:
                        logger.warning("Could not read reference face image.", extra={"path": ref_path})
                except Exception:
                    logger.error("Failed to process reference face image.", exc_info=True)
            else:
                logger.warning("Reference face image path does not exist.", extra={"path": ref_path})

    landmarker_path = Path(config.models_dir) / Path(config.face_landmarker_url).name
    error_handler = ErrorHandler(logger, config.retry_max_attempts, config.retry_backoff_seconds)
    download_model(
        config.face_landmarker_url,
        landmarker_path,
        "MediaPipe Face Landmarker",
        logger,
        error_handler,
        config.user_agent,
        expected_sha256=config.face_landmarker_sha256,
    )
    if landmarker_path.exists():
        face_landmarker = get_face_landmarker(str(landmarker_path), logger)

    # We might need to return person_detector if it is used elsewhere.
    # In app.py initialize_analysis_models returned dict.
    # But where is person_detector?
    # Ah, "person_detector = models['person_detector']" in app.py line 1856.
    # But initialize_analysis_models in app.py line 792 DOES NOT return person_detector?
    # Wait, let's check app.py again.

    return {"face_analyzer": face_analyzer, "ref_emb": ref_emb, "face_landmarker": face_landmarker, "device": device}


class VideoManager:
    """Handles video preparation and metadata extraction."""

    def __init__(self, source_path: str, config: "Config", max_resolution: Optional[str] = None):
        self.source_path = source_path
        self.config = config
        self.max_resolution = max_resolution or self.config.default_max_resolution
        self.is_youtube = "youtube.com/" in source_path or "youtu.be/" in source_path

    def prepare_video(self, logger: "AppLogger") -> str:
        """
        Prepares the video for processing.

        Downloads it if it's a YouTube URL, or validates the local path.
        """
        if self.is_youtube:
            logger.info("Downloading video", component="video", user_context={"source": self.source_path})
            tmpl = self.config.ytdl_output_template
            max_h = None if self.max_resolution == "maximum available" else int(self.max_resolution)
            ydl_opts = {
                "outtmpl": str(Path(self.config.downloads_dir) / tmpl),
                "format": self.config.ytdl_format_string.format(max_res=max_h)
                if max_h
                else "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best",
                "merge_output_format": "mp4",
                "noprogress": True,
                "quiet": True,
            }
            try:
                with ytdlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(self.source_path, download=True)
                    return str(Path(ydl.prepare_filename(info)))
            except ytdlp.utils.DownloadError as e:
                raise RuntimeError(f"Download failed. Resolution may not be available. Details: {e}") from e
        local_path = Path(self.source_path)
        validate_video_file(local_path)
        return str(local_path)

    @staticmethod
    def get_video_info(video_path: str) -> dict:
        """Extracts metadata (FPS, dimensions, frame count) from the video file."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(f"Could not open video: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        if not np.isfinite(fps) or fps <= 0:
            fps = 30.0
        info = {
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": fps,
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        }
        cap.release()
        return info
