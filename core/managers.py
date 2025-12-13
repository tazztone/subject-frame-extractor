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
# Ensure project root is in path or SAM3_repo is in path.
# Since we are in core/, we might need to adjust path if not already handled by app entry point.
# Assuming app.py handles sys.path setup for SAM3_repo.
try:
    from core import sam3_patches
    from sam3.model_builder import build_sam3_video_predictor
    from sam3.model.sam3_video_predictor import Sam3VideoPredictor
    sam3_patches.apply_patches()
except ImportError:
    # This might fail if run in isolation without path setup
    pass

if TYPE_CHECKING:
    from core.config import Config
    from core.logger import AppLogger
    from core.models import AnalysisParameters

from core.utils import download_model, validate_video_file, safe_resource_cleanup
from core.error_handling import ErrorHandler

class ThumbnailManager:
    def __init__(self, logger: 'AppLogger', config: 'Config'):
        self.logger = logger
        self.config = config
        self.cache = OrderedDict()
        self.max_size = self.config.cache_size
        self.logger.info(f"ThumbnailManager initialized with cache size {self.max_size}")

    def get(self, thumb_path: Path) -> Optional[np.ndarray]:
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
        self.cache.clear()
        gc.collect()

    def _cleanup_old_entries(self):
        num_to_remove = int(self.max_size * self.config.cache_eviction_factor)
        for _ in range(num_to_remove):
            if not self.cache: break
            self.cache.popitem(last=False)

class ModelRegistry:
    def __init__(self, logger: Optional['AppLogger'] = None):
        self._models: Dict[str, Any] = {}
        self._locks: Dict[str, threading.Lock] = defaultdict(threading.Lock)
        self.logger = logger or logging.getLogger(__name__)
        self.runtime_device_override: Optional[str] = None

    def get_or_load(self, key: str, loader_fn: Callable[[], Any]) -> Any:
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
        if self.logger: self.logger.info("Clearing all models from the registry.")
        self._models.clear()

    def get_tracker(self, model_name: str, models_path: str, user_agent: str,
                    retry_params: tuple, config: 'Config') -> Optional['SAM3Wrapper']:
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
                expected_sha256=config.sam3_checkpoint_sha256
            )

        self.logger.info(f"Loading SAM3 model on {device}...", component="tracker")
        return SAM3Wrapper(str(checkpoint_path), device=device)

class SAM3Wrapper:
    def __init__(self, checkpoint_path, device="cuda"):
        self.device = device
        self.predictor = build_sam3_video_predictor(
            ckpt_path=checkpoint_path,
            device=device
        )
        self.session_id = None

    def initialize(self, images, init_mask=None, bbox=None, prompt_frame_idx=0):
        """
        Initialize session with images and optional prompt.
        images: List of PIL Images.
        bbox: [x, y, w, h]
        prompt_frame_idx: Index of the frame to apply the prompt to.
        """
        if self.session_id is not None:
            try:
                self.predictor.close_session(self.session_id)
            except Exception:
                pass

        self.session_id = self.predictor.start_session(images)

        if bbox is not None:
            # Convert xywh to xyxy
            x, y, w, h = bbox
            xyxy = [x, y, x + w, y + h]
            self.predictor.add_prompt(self.session_id, frame_idx=prompt_frame_idx, bounding_boxes=[xyxy])

        # Return mask for the prompt frame
        gen = self.predictor.propagate_in_video(self.session_id, start_frame_idx=prompt_frame_idx, max_frame_num_to_track=1)
        try:
            _, out = next(gen)
            if out and 'obj_id_to_mask' in out and len(out['obj_id_to_mask']) > 0:
                pred_mask = list(out['obj_id_to_mask'].values())[0]
                if isinstance(pred_mask, torch.Tensor):
                    pred_mask = pred_mask.cpu().numpy().astype(bool)
                    if pred_mask.ndim == 3: pred_mask = pred_mask[0]
                return {'pred_mask': pred_mask}
        except StopIteration:
            pass
        return {'pred_mask': None}

    def propagate_from(self, start_idx, direction="forward"):
        """
        Yields results starting from start_idx in the given direction.
        """
        return self.predictor.propagate_in_video(self.session_id, start_frame_idx=start_idx, propagation_direction=direction)

    def detect_objects(self, image_rgb: np.ndarray, text_prompt: str) -> List[dict]:
        if self.session_id is not None:
            try: self.predictor.close_session(self.session_id)
            except Exception: pass

        pil_img = Image.fromarray(image_rgb)
        self.session_id = self.predictor.start_session([pil_img])

        res = self.predictor.add_prompt(self.session_id, frame_idx=0, text=text_prompt)
        outputs = res.get('outputs', {})

        results = []
        if outputs and 'obj_id_to_mask' in outputs:
            scores = outputs.get('obj_id_to_score', {})
            for obj_id, mask in outputs['obj_id_to_mask'].items():
                if isinstance(mask, torch.Tensor):
                    mask = mask.cpu().numpy()

                mask_bool = mask > 0
                if mask_bool.ndim == 3: mask_bool = mask_bool[0]

                if not np.any(mask_bool): continue

                x, y, w, h = cv2.boundingRect(mask_bool.astype(np.uint8))
                score = float(scores.get(obj_id, 1.0))
                if hasattr(score, 'item'): score = score.item()

                results.append({
                    'bbox': [x, y, x + w, y + h],
                    'conf': score,
                    'label': text_prompt,
                    'type': 'sam3_text'
                })

        results.sort(key=lambda x: x['conf'], reverse=True)
        return results

thread_local = threading.local()

def get_face_landmarker(model_path: str, logger: 'AppLogger') -> vision.FaceLandmarker:
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
    return lpips.LPIPS(net=model_name).to(device)

def initialize_analysis_models(params: 'AnalysisParameters', config: 'Config', logger: 'AppLogger',
                               model_registry: 'ModelRegistry') -> dict:
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
    def __init__(self, source_path: str, config: 'Config', max_resolution: Optional[str] = None):
        self.source_path = source_path
        self.config = config
        self.max_resolution = max_resolution or self.config.default_max_resolution
        self.is_youtube = ("youtube.com/" in source_path or "youtu.be/" in source_path)

    def prepare_video(self, logger: 'AppLogger') -> str:
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
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened(): raise IOError(f"Could not open video: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        if not np.isfinite(fps) or fps <= 0: fps = 30.0
        info = {"width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "fps": fps, "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}
        cap.release()
        return info
