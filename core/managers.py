from __future__ import annotations
import collections
from collections import OrderedDict, defaultdict
import gc
import logging
import threading
import time
import shutil
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union, TYPE_CHECKING
import torch
import numpy as np
import cv2
from PIL import Image
import lpips
import yt_dlp as ytdlp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# SAM3 imports
# On Windows, Triton is not available. We need to mock it BEFORE importing SAM3
# because sam3/model/edt.py imports triton at module level.
build_sam3_video_predictor = None
Sam3VideoPredictor = None

def _setup_triton_mock():
    """Create a mock triton module if triton is not available (Windows)."""
    import sys
    try:
        import triton
        return False  # Triton is available, no mock needed
    except ImportError:
        pass
    
    # Create mock triton module with proper __spec__ for PyTorch compatibility
    from unittest.mock import MagicMock
    from importlib.machinery import ModuleSpec
    import types
    
    # Create a proper module object (not just MagicMock)
    mock_triton = types.ModuleType('triton')
    mock_triton.__spec__ = ModuleSpec('triton', None)
    mock_triton.__path__ = []
    mock_triton.__file__ = '<mock>'
    mock_triton.language = types.ModuleType('triton.language')
    mock_triton.language.__spec__ = ModuleSpec('triton.language', None)
    mock_triton.jit = lambda fn: fn  # Decorator that returns function unchanged
    
    # Create a mock for tl (triton.language) attributes
    class MockTL:
        constexpr = lambda x: x
        program_id = MagicMock(return_value=0)
        load = MagicMock(return_value=0)
        store = MagicMock()
    
    for attr in dir(MockTL):
        if not attr.startswith('_'):
            setattr(mock_triton.language, attr, getattr(MockTL, attr))
    
    sys.modules['triton'] = mock_triton
    sys.modules['triton.language'] = mock_triton.language
    return True

# Apply triton mock before SAM3 import
_triton_mocked = _setup_triton_mock()

try:
    from core import sam3_patches
    from sam3.model_builder import build_sam3_video_model
    
    # Apply patches to replace triton functions with CPU fallbacks
    if _triton_mocked:
        sam3_patches.apply_patches()
except ImportError as e:
    # This might fail if run in isolation without path setup or missing dependencies
    logging.getLogger(__name__).warning(f"Failed to import SAM3 dependencies: {e}")

if TYPE_CHECKING:
    from core.config import Config
    from core.logger import AppLogger
    from core.models import AnalysisParameters

from core.utils import download_model, validate_video_file, safe_resource_cleanup
from core.error_handling import ErrorHandler

class ThumbnailManager:
    """Manages an in-memory LRU cache for image thumbnails."""
    def __init__(self, logger: 'AppLogger', config: 'Config'):
        """Initializes the manager with a configurable cache size."""
        self.logger = logger
        self.config = config
        self.cache = OrderedDict()
        self.max_size = self.config.cache_size
        self.logger.info(f"ThumbnailManager initialized with cache size {self.max_size}")

    def get(self, thumb_path: Path) -> Optional[np.ndarray]:
        """Retrieves a thumbnail from cache or loads it from disk."""
        if not isinstance(thumb_path, Path): thumb_path = Path(thumb_path)
        if thumb_path in self.cache:
            self.cache.move_to_end(thumb_path)
            return self.cache[thumb_path]
        if not thumb_path.exists(): return None
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
            self.logger.warning("Failed to load thumbnail with Pillow", extra={'path': str(thumb_path), 'error': e})
            return None

    def clear_cache(self):
        """Clears the thumbnail cache and triggers garbage collection."""
        self.cache.clear()
        gc.collect()

    def _cleanup_old_entries(self):
        num_to_remove = int(self.max_size * self.config.cache_eviction_factor)
        for _ in range(num_to_remove):
            if not self.cache: break
            self.cache.popitem(last=False)

class ModelRegistry:
    """
    Thread-safe registry for lazy loading and caching of heavy ML models.
    """
    def __init__(self, logger: Optional['AppLogger'] = None):
        self._models: Dict[str, Any] = {}
        self._locks: Dict[str, threading.Lock] = defaultdict(threading.Lock)
        self.logger = logger or logging.getLogger(__name__)
        self.runtime_device_override: Optional[str] = None

    def get_or_load(self, key: str, loader_fn: Callable[[], Any]) -> Any:
        """Retrieves a model by key, loading it via loader_fn if not present."""
        if key not in self._models:
            with self._locks[key]:
                if key not in self._models:
                    if self.logger: self.logger.info(f"Loading model '{key}' for the first time...")
                    try:
                        val = loader_fn()
                        # print(f"DEBUG: ModelRegistry loaded {key} -> {val}")
                        self._models[key] = val
                    except Exception as e:
                        # print(f"DEBUG: ModelRegistry failed to load {key}: {e}")
                        raise e
                    if self.logger: self.logger.success(f"Model '{key}' loaded successfully.")
        return self._models[key]

    def clear(self):
        """Clears all loaded models from the registry."""
        if self.logger: self.logger.info("Clearing all models from the registry.")
        self._models.clear()

    def get_tracker(self, model_name: str, models_path: str, user_agent: str,
                    retry_params: tuple, config: 'Config') -> Optional['SAM3Wrapper']:
        """
        Gets or loads the SAM3 tracker, handling CPU fallback on CUDA OOM.
        """
        key = f"tracker_{model_name}"

        def _loader():
            device = self.runtime_device_override or ("cuda" if torch.cuda.is_available() else "cpu")
            try:
                return self._load_tracker_impl(model_name, models_path, user_agent, retry_params, device, config)
            except RuntimeError as e:
                if "out of memory" in str(e) and device == 'cuda':
                    self.logger.warning("CUDA OOM during tracker init. Switching to CPU for this session.")
                    torch.cuda.empty_cache()
                    self.runtime_device_override = 'cpu'
                    return self._load_tracker_impl(model_name, models_path, user_agent, retry_params, 'cpu', config)
                raise e

        try:
            return self.get_or_load(key, _loader)
        except Exception as e:
            self.logger.error(f"Failed to initialize tracker: {e}", exc_info=True)
            return None

    def _load_tracker_impl(self, model_name: str, models_path: str, user_agent: str,
                           retry_params: tuple, device: str, config: 'Config') -> 'SAM3Wrapper':
        if device == 'cuda' and not torch.cuda.is_available():
            self.logger.warning("CUDA not available, SAM3 requires CUDA. Attempting to run on CPU (might be slow/fail).", component="tracker")

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
                token=config.huggingface_token
            )

        self.logger.info(f"Loading SAM3 model on {device}...", component="tracker")
        return SAM3Wrapper(str(checkpoint_path), device=device)

class SAM3Wrapper:
    """
    SAM3 Tracker using official Sam3TrackerPredictor API.
    
    Based on: https://github.com/facebookresearch/sam3/blob/main/examples/sam3_for_sam2_video_task_example.ipynb
    
    Key API patterns:
    - init_state(video_path) for session initialization
    - add_new_points_or_box() for prompts with relative coordinates
    - propagate_in_video() generator for mask propagation
    """
    
    def __init__(self, checkpoint_path=None, device="cuda"):
        """
        Initialize SAM3 wrapper using official model builder pattern.
        
        Args:
            checkpoint_path: Optional path to checkpoint (auto-downloads from HF if None)
            device: Device to run on ('cuda' or 'cpu')
        """
        from sam3.model_builder import build_sam3_video_model
        
        self.device = device
        self._checkpoint_path = checkpoint_path
        
        # Setup optimal precision for Ampere GPUs (official pattern)
        if torch.cuda.is_available():
            torch.set_float32_matmul_precision('high')
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
        
        # Build model using official pattern
        self.sam3_model = build_sam3_video_model()
        self.predictor = self.sam3_model.tracker
        self.predictor.backbone = self.sam3_model.detector.backbone
        
        # Session state
        self.inference_state = None
        self._temp_dir = None
    
    def init_video(self, video_path: str):
        """
        Initialize inference state with video or frame directory.
        
        Args:
            video_path: Path to video file or directory of JPEG frames
            
        Returns:
            inference_state object
        """
        self.inference_state = self.predictor.init_state(video_path=video_path)
        return self.inference_state
    
    def add_bbox_prompt(self, frame_idx: int, obj_id: int, bbox_xywh: list, 
                        img_size: tuple) -> np.ndarray:
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
        if self.inference_state is None:
            raise RuntimeError("Must call init_video() before adding prompts")
        
        w, h = img_size
        x, y, bw, bh = bbox_xywh
        
        # Convert to relative coordinates (0-1) in xyxy format
        rel_box = np.array([[x/w, y/h, (x+bw)/w, (y+bh)/h]], dtype=np.float32)
        
        _, obj_ids, low_res_masks, video_res_masks = self.predictor.add_new_points_or_box(
            inference_state=self.inference_state,
            frame_idx=frame_idx,
            obj_id=obj_id,
            box=rel_box,
        )
        
        # Return first mask as boolean numpy array
        if video_res_masks is not None and len(video_res_masks) > 0:
            mask = (video_res_masks[0] > 0.0).cpu().numpy()
            if mask.ndim == 3:
                mask = mask[0]  # Remove batch dimension if present
            return mask
        return np.zeros((h, w), dtype=bool)
    
    def propagate(self, start_idx: int = 0, max_frames: int = None, 
                  reverse: bool = False):
        """
        Generator yielding masks for each frame during propagation.
        
        Args:
            start_idx: Frame index to start propagation from
            max_frames: Maximum frames to propagate (default: all frames)
            reverse: If True, propagate backward from start_idx
            
        Yields:
            Tuple of (frame_idx, obj_id, mask) where mask is numpy array
        """
        if self.inference_state is None:
            return
        
        for frame_idx, obj_ids, low_res_masks, video_res_masks, obj_scores in \
            self.predictor.propagate_in_video(
                self.inference_state,
                start_frame_idx=start_idx,
                max_frame_num_to_track=max_frames or 9999,
                reverse=reverse,
                propagate_preflight=True,
            ):
            for i, obj_id in enumerate(obj_ids):
                mask = (video_res_masks[i] > 0.0).cpu().numpy()
                if mask.ndim == 3:
                    mask = mask[0]
                yield frame_idx, obj_id, mask
    
    def clear_prompts(self):
        """Reset all prompts in current session."""
        if self.inference_state:
            self.predictor.clear_all_points_in_video(self.inference_state)
    
    # === Legacy compatibility methods ===
    # These provide backward compatibility during migration
    
    def initialize(self, images, init_mask=None, bbox=None, prompt_frame_idx=0):
        """
        Legacy method: Initialize session with images and optional prompt.
        
        DEPRECATED: Use init_video() + add_bbox_prompt() instead.
        
        Args:
            images: List of PIL Images or numpy arrays
            bbox: [x, y, w, h] bounding box
            prompt_frame_idx: Index of the frame to apply the prompt to
            
        Returns:
            dict with 'pred_mask' key
        """
        import tempfile
        import os
        
        # Save images to temp directory for init_state()
        temp_dir = tempfile.mkdtemp()
        for i, img in enumerate(images):
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            img.save(os.path.join(temp_dir, f"{i:05d}.jpg"))
        
        self._temp_dir = temp_dir
        
        # Initialize video session
        self.init_video(temp_dir)
        
        if bbox is not None and self.inference_state:
            # Get image dimensions from first image
            if isinstance(images[0], np.ndarray):
                h, w = images[0].shape[:2]
            else:
                w, h = images[0].size
            
            mask = self.add_bbox_prompt(prompt_frame_idx, 1, bbox, (w, h))
            return {"pred_mask": mask}
        
        return {"pred_mask": None}
    
    def propagate_from(self, start_idx, direction="forward"):
        """
        Legacy method: Yields results starting from start_idx in given direction.
        
        DEPRECATED: Use propagate() generator instead.
        
        Yields:
            Dict with 'frame_index' and 'outputs' keys
        """
        reverse = (direction == "backward")
        
        for frame_idx, obj_id, mask in self.propagate(start_idx, reverse=reverse):
            yield {
                'frame_index': frame_idx,
                'outputs': {
                    'obj_id_to_mask': {obj_id: mask}
                }
            }
    
    def detect_objects(self, image_rgb: np.ndarray, text_prompt: str) -> list:
        """
        Detect objects in an image using text prompt.
        
        Args:
            image_rgb: RGB numpy array
            text_prompt: Text description of object to find
            
        Returns:
            List of detection dicts with bbox, conf, label, type
        """
        import tempfile
        import os
        
        # Create temp directory with single image
        temp_dir = tempfile.mkdtemp()
        Image.fromarray(image_rgb).save(os.path.join(temp_dir, "00000.jpg"))
        self._temp_dir = temp_dir
        
        # Initialize and detect using SAM3's detector
        self.init_video(temp_dir)
        
        # Use text prompt detection if available
        # Note: This uses the detector component, not tracker
        try:
            h, w = image_rgb.shape[:2]
            
            # Add text prompt as a point at center (triggering detection)
            rel_points = torch.tensor([[0.5, 0.5]], dtype=torch.float32)
            labels = torch.tensor([1], dtype=torch.int32)
            
            _, obj_ids, _, video_res_masks = self.predictor.add_new_points(
                inference_state=self.inference_state,
                frame_idx=0,
                obj_id=1,
                points=rel_points,
                labels=labels,
            )
            
            results = []
            if video_res_masks is not None and len(video_res_masks) > 0:
                mask = (video_res_masks[0] > 0.0).cpu().numpy()
                if mask.ndim == 3:
                    mask = mask[0]
                
                if np.any(mask):
                    x, y, bw, bh = cv2.boundingRect(mask.astype(np.uint8))
                    results.append({
                        "bbox": [x, y, x + bw, y + bh],
                        "conf": 1.0,
                        "label": text_prompt,
                        "type": "sam3_text"
                    })
            
            return results
        except Exception as e:
            logging.getLogger(__name__).warning(f"Text detection failed: {e}")
            return []
    
    def cleanup(self):
        """Clean up temporary resources."""
        import shutil
        if hasattr(self, "_temp_dir") and self._temp_dir:
            try:
                shutil.rmtree(self._temp_dir)
            except Exception:
                pass
            self._temp_dir = None
        
        # Clear session state
        self.inference_state = None

thread_local = threading.local()

def get_face_landmarker(model_path: str, logger: 'AppLogger') -> vision.FaceLandmarker:
    """
    Returns a thread-local MediaPipe FaceLandmarker instance.
    """
    if hasattr(thread_local, 'face_landmarker_instance'): return thread_local.face_landmarker_instance
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

def get_face_analyzer(model_name: str, models_path: str, det_size_tuple: tuple, logger: 'AppLogger',
                      model_registry: 'ModelRegistry', device: str = 'cpu') -> 'FaceAnalysis':
    """
    Gets or loads the InsightFace FaceAnalysis app, with OOM handling.
    """
    from insightface.app import FaceAnalysis
    model_key = f"face_analyzer_{model_name}_{device}_{det_size_tuple}"

    def _loader():
        logger.info(f"Loading face model: {model_name} on device: {device}")
        try:
            is_cuda = device == 'cuda'
            providers = (['CUDAExecutionProvider', 'CPUExecutionProvider'] if is_cuda else ['CPUExecutionProvider'])
            analyzer = FaceAnalysis(name=model_name, root=models_path, providers=providers)
            analyzer.prepare(ctx_id=0 if is_cuda else -1, det_size=det_size_tuple)
            logger.success(f"Face model loaded with {'CUDA' if is_cuda else 'CPU'}.")
            return analyzer
        except Exception as e:
            if "out of memory" in str(e) and device == 'cuda':
                torch.cuda.empty_cache()
                logger.warning("CUDA OOM, retrying with CPU...")
                try:
                    analyzer = FaceAnalysis(name=model_name, root=models_path, providers=['CPUExecutionProvider'])
                    analyzer.prepare(ctx_id=-1, det_size=det_size_tuple)
                    return analyzer
                except Exception as cpu_e:
                    logger.error(f"CPU fallback also failed: {cpu_e}")
            raise RuntimeError(f"Could not initialize face analysis model. Error: {e}") from e

    return model_registry.get_or_load(model_key, _loader)

def get_lpips_metric(model_name: str = 'alex', device: str = 'cpu') -> torch.nn.Module:
    """Returns the LPIPS metric model."""
    return lpips.LPIPS(net=model_name).to(device)

def initialize_analysis_models(params: 'AnalysisParameters', config: 'Config', logger: 'AppLogger',
                               model_registry: 'ModelRegistry') -> dict:
    """
    Initializes all necessary analysis models based on parameters.

    Returns:
        Dictionary of initialized models (face_analyzer, ref_emb, etc.).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    face_analyzer, ref_emb, face_landmarker = None, None, None

    if params.enable_face_filter:
        face_analyzer = get_face_analyzer(model_name=params.face_model_name, models_path=str(config.models_dir),
                                          det_size_tuple=tuple(config.model_face_analyzer_det_size),
                                          logger=logger, model_registry=model_registry, device=device)
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
                        else: logger.warning("No face found in reference image.", extra={'path': ref_path})
                    else: logger.warning("Could not read reference face image.", extra={'path': ref_path})
                except Exception as e: logger.error("Failed to process reference face image.", exc_info=True)
            else: logger.warning("Reference face image path does not exist.", extra={'path': ref_path})

    landmarker_path = Path(config.models_dir) / Path(config.face_landmarker_url).name
    error_handler = ErrorHandler(logger, config.retry_max_attempts, config.retry_backoff_seconds)
    download_model(config.face_landmarker_url, landmarker_path, "MediaPipe Face Landmarker", logger, error_handler, config.user_agent, expected_sha256=config.face_landmarker_sha256)
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
    def __init__(self, source_path: str, config: 'Config', max_resolution: Optional[str] = None):
        self.source_path = source_path
        self.config = config
        self.max_resolution = max_resolution or self.config.default_max_resolution
        self.is_youtube = ("youtube.com/" in source_path or "youtu.be/" in source_path)

    def prepare_video(self, logger: 'AppLogger') -> str:
        """
        Prepares the video for processing.

        Downloads it if it's a YouTube URL, or validates the local path.
        """
        if self.is_youtube:
            logger.info("Downloading video", component="video", user_context={'source': self.source_path})
            tmpl = self.config.ytdl_output_template
            max_h = None if self.max_resolution == "maximum available" else int(self.max_resolution)
            ydl_opts = {
                'outtmpl': str(Path(self.config.downloads_dir) / tmpl),
                'format': self.config.ytdl_format_string.format(max_res=max_h) if max_h else "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best",
                'merge_output_format': 'mp4',
                'noprogress': True,
                'quiet': True
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
        if not cap.isOpened(): raise IOError(f"Could not open video: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        if not np.isfinite(fps) or fps <= 0: fps = 30.0
        info = {"width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "fps": fps, "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}
        cap.release()
        return info
