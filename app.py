# keep this app Monolithic. 
import gradio as gr
import cv2
import numpy as np
import os
import json
import re
import shutil
import logging
from logging.handlers import RotatingFileHandler
import threading
import time
import subprocess
import gc
import math
from collections import Counter, defaultdict
from pathlib import Path
from datetime import datetime
from queue import Queue, Empty
from dataclasses import dataclass, asdict, field
from concurrent.futures import ThreadPoolExecutor
import hashlib
from contextlib import contextmanager
import urllib.request
import yt_dlp as ytdlp
from scenedetect import detect, ContentDetector
from PIL import Image
import torch
from torchvision.ops import box_convert
from torchvision import transforms
from ultralytics import YOLO
from DAM4SAM.dam4sam_tracker import DAM4SAMTracker
from insightface.app import FaceAnalysis
from numba import njit
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io
import imagehash
import pyiqa
from grounding_dino.groundingdino.util.inference import (
    load_model as gdino_load_model,
    load_image as gdino_load_image,
    predict as gdino_predict,
)
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# --- Unified Logging & Configuration ---

# Add custom SUCCESS log level for more semantic logging
SUCCESS_LEVEL = 25
logging.addLevelName(SUCCESS_LEVEL, "SUCCESS")

def success_log_method(self, message, *args, **kwargs):
    if self.isEnabledFor(SUCCESS_LEVEL):
        self._log(SUCCESS_LEVEL, message, args, **kwargs)
logging.Logger.success = success_log_method


class StructuredFormatter(logging.Formatter):
    """Custom formatter to include extra context in log messages."""
    def format(self, record):
        # Find all keys in the record that are not standard LogRecord attributes
        extra_items = {k: v for k, v in record.__dict__.items() if k not in logging.LogRecord.__dict__ and k != 'args'}
        if extra_items:
            # Append the key-value pairs to the original message
            record.msg = f"{record.msg} [{', '.join(f'{k}={v}' for k, v in extra_items.items())}]"
        return super().format(record)


class UnifiedLogger:
    def __init__(self, log_file_path=None):
        self.progress_queue = None
        self.logger = logging.getLogger('unified_logger')
        if not self.logger.handlers:  # Prevent adding handlers multiple times on hot-reload
            self.logger.setLevel(logging.INFO)
            self.logger.propagate = False
            formatter = StructuredFormatter('%(asctime)s - %(levelname)s - %(message)s')
            
            if log_file_path:
                # Use a rotating file handler for the main application log to prevent it from growing too large
                fh = RotatingFileHandler(log_file_path, maxBytes=5*1024*1024, backupCount=3, encoding='utf-8')
                fh.setFormatter(formatter)
                self.logger.addHandler(fh)

            ch = logging.StreamHandler()
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)

    def set_progress_queue(self, queue):
        """Dynamically set the queue for UI log updates."""
        self.progress_queue = queue

    def add_handler(self, handler):
        """Add a new handler, e.g., for a per-run log file."""
        self.logger.addHandler(handler)
        return handler

    def remove_handler(self, handler):
        """Remove a handler and ensure it's closed properly."""
        if handler:
            self.logger.removeHandler(handler)
            handler.close()

    def _log(self, level_name, message, exc_info=False, extra=None):
        level = logging.getLevelName(level_name.upper())
        self.logger.log(level, message, exc_info=exc_info, extra=extra)
        if self.progress_queue:
            extra_str = f" [{', '.join(f'{k}={v}' for k, v in extra.items())}]" if extra else ""
            self.progress_queue.put({"log": f"[{level_name.upper()}] {message}{extra_str}"})

    def info(self, message, extra=None, **kwargs): self._log('INFO', message, extra=extra or kwargs)
    def warning(self, message, extra=None, **kwargs): self._log('WARNING', message, extra=extra or kwargs)
    def error(self, message, exc_info=False, extra=None, **kwargs): self._log('ERROR', message, exc_info=exc_info, extra=extra or kwargs)
    def critical(self, message, exc_info=False, extra=None, **kwargs): self._log('CRITICAL', message, exc_info=exc_info, extra=extra or kwargs)
    def success(self, message, extra=None, **kwargs): self._log('SUCCESS', message, extra=extra or kwargs)
    
    def pipeline_error(self, operation, e, extra=None, **kwargs):
        full_context = {'error': str(e), **(extra or kwargs)}
        self.error(f"{operation} failed", exc_info=True, extra=full_context)
        return {"error": str(e)}

class Config:
    BASE_DIR = Path(__file__).parent
    DIRS = {'logs': BASE_DIR / "logs", 'configs': BASE_DIR / "configs", 'models': BASE_DIR / "models", 'downloads': BASE_DIR / "downloads"}
    LOG_FILE = DIRS['logs'] / "frame_extractor.log"
    CONFIG_FILE = DIRS['configs'] / "config.yaml"

    def __init__(self):
        self.settings = self.load_config()
        # Dynamically set attributes from the loaded config for easy access
        for key, value in self.settings.items():
            setattr(self, key, value)
        
        # Backward compatibility for hardcoded paths if needed
        self.GROUNDING_DINO_CONFIG = self.BASE_DIR / self.model_paths['grounding_dino_config']
        self.GROUNDING_DINO_CKPT = self.DIRS['models'] / Path(self.model_paths['grounding_dino_checkpoint']).name
        self.GROUNDING_BOX_THRESHOLD = self.grounding_dino_params['box_threshold']
        self.GROUNDING_TEXT_THRESHOLD = self.grounding_dino_params['text_threshold']
        self.QUALITY_METRICS = list(self.quality_weights.keys())

    def load_config(self):
        self.DIRS['configs'].mkdir(exist_ok=True)
        if not self.CONFIG_FILE.exists():
            raise FileNotFoundError(f"Configuration file not found at {self.CONFIG_FILE}. Please ensure it exists.")
        with open(self.CONFIG_FILE, 'r') as f:
            return yaml.safe_load(f)

    @classmethod
    def setup_directories_and_logger(cls):
        for dir_path in cls.DIRS.values():
            dir_path.mkdir(exist_ok=True)
        # Pass the main log file path to the logger
        return UnifiedLogger(log_file_path=cls.LOG_FILE)


# --- Global Initialization ---
try:
    config = Config()
    logger = config.setup_directories_and_logger()
except FileNotFoundError as e:
    # Handle missing config file gracefully
    print(f"FATAL: {e}")
    # Create a dummy logger to output the error
    logging.basicConfig()
    logger = logging.getLogger()
    logger.error(f"FATAL: {e}")
    # A simple Gradio app to show the error if the main one can't start
    with gr.Blocks() as error_app:
        gr.Markdown(f"# Configuration Error\n\n**Could not start the application.**\n\nReason: `{e}`\n\nPlease create a `configs/config.yaml` file and restart.")
    error_app.launch()
    exit() # Exit the script

# --- Utility Functions ---
def check_dependencies():
    if not shutil.which("ffmpeg"):
        logger.error("FFMPEG is not installed or not in PATH.")
        raise RuntimeError("FFMPEG is not installed. Please install it to continue.")

def sanitize_filename(name, max_length=50):
    return re.sub(r'[^\w\-_.]', '_', name)[:max_length]

def safe_execute_with_retry(func, max_retries=3, delay=1.0, backoff=2.0):
    last_exception = None
    for attempt in range(max_retries + 1):
        try:
            return func()
        except Exception as e:
            last_exception = e
            if attempt < max_retries:
                logger.warning(f"Attempt {attempt + 1} failed. Retrying in {delay}s...", extra={'error': e})
                time.sleep(delay)
                delay *= backoff
    raise last_exception if last_exception else RuntimeError("Function failed after retries.")

def download_model(url, dest_path, description, min_size=1_000_000):
    if dest_path.is_file() and dest_path.stat().st_size >= min_size:
        return
    def download_func():
        logger.info(f"Downloading {description}", extra={'url': url, 'dest': dest_path})
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as resp, open(dest_path, "wb") as out:
            shutil.copyfileobj(resp, out)
        if not dest_path.exists() or dest_path.stat().st_size < min_size:
            raise RuntimeError(f"Downloaded {description} seems incomplete")
        logger.success(f"{description} downloaded successfully.")
    try:
        safe_execute_with_retry(download_func)
    except Exception as e:
        logger.error(f"Failed to download {description}", exc_info=True, extra={'url': url})
        raise RuntimeError(f"Failed to download required model: {description}") from e

def render_mask_overlay(frame_bgr: np.ndarray, mask_gray: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    if mask_gray is None:
        return frame_bgr
    h, w = frame_bgr.shape[:2]
    if mask_gray.shape[:2] != (h, w):
        mask_gray = cv2.resize(mask_gray, (w, h), interpolation=cv2.INTER_NEAREST)
    m = (mask_gray > 128)
    red_layer = np.zeros_like(frame_bgr, dtype=np.uint8)
    red_layer[..., 2] = 255
    blended = cv2.addWeighted(frame_bgr, 1.0 - alpha, red_layer, alpha, 0.0)
    if m.ndim == 2: m = m[..., np.newaxis]
    elif m.ndim == 3 and m.shape[2] != 1:
        logger.warning(f"Unexpected mask shape. Skipping overlay.", extra={'shape': m.shape})
        return frame_bgr
    out = np.where(m, blended, frame_bgr)
    return out

@contextmanager
def safe_resource_cleanup():
    try:
        yield
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def _to_json_safe(obj):
    if isinstance(obj, dict):
        return {k: _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        return _to_json_safe(obj.item())
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, float):
        return round(obj, 4)
    return obj
    
def bgr_to_pil(image_bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))

def pil_to_bgr(image_pil: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

# --- Person Detector Wrapper ---
class PersonDetector:
    def __init__(self, model="yolo11x.pt", imgsz=640, conf=0.3, device='cuda'):
        if YOLO is None:
            raise ImportError("Ultralytics YOLO not installed.")
        model_path = config.DIRS['models'] / model
        model_path.parent.mkdir(exist_ok=True)
        download_model(f"https://huggingface.co/Ultralytics/YOLO11/resolve/main/{model}", model_path, "YOLO person detector")
        
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model = YOLO(str(model_path))
        self.model.to(self.device)
        self.imgsz = imgsz
        self.conf = conf
        logger.info(f"YOLO person detector loaded", extra={'device': self.device, 'model': model})

    def detect_boxes(self, img_bgr):
        # The predict method handles device placement internally, but we ensure the model is on the correct device.
        res = self.model.predict(img_bgr, imgsz=self.imgsz, conf=self.conf, classes=[0], verbose=False, device=self.device)
        boxes = []
        for r in res:
            if getattr(r, "boxes", None) is None: continue
            # Move results to CPU for post-processing
            cpu_boxes = r.boxes.cpu()
            for b in cpu_boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                score = float(b.conf[0])
                boxes.append((x1, y1, x2, y2, score))
        return boxes

# --- Numba Optimized Image Processing ---
@njit
def compute_entropy(hist):
    prob = hist / (np.sum(hist) + 1e-7)
    entropy = -np.sum(prob[prob > 0] * np.log2(prob[prob > 0]))
    return min(max(entropy / 8.0, 0), 1.0)

# --- Core Data Classes ---
@dataclass
class FrameMetrics:
    quality_score: float = 0.0; sharpness_score: float = 0.0; edge_strength_score: float = 0.0;
    contrast_score: float = 0.0; brightness_score: float = 0.0; entropy_score: float = 0.0
    niqe_score: float = 0.0

@dataclass
class Frame:
    image_data: np.ndarray; frame_number: int
    metrics: FrameMetrics = field(default_factory=FrameMetrics)
    face_similarity_score: float | None = None; max_face_confidence: float | None = None
    error: str | None = None

    def calculate_quality_metrics(self, thumb_image: np.ndarray, mask: np.ndarray | None = None, niqe_metric=None):
        try:
            gray = cv2.cvtColor(thumb_image, cv2.COLOR_BGR2GRAY)
            
            # The mask is now expected to be at the thumbnail resolution
            active_mask = (mask > 128) if mask is not None and mask.ndim == 2 else None
            if active_mask is not None and np.sum(active_mask) < 100:
                raise ValueError("Mask too small.")

            # --- Sharpness and Edges (on thumbnail) ---
            lap = cv2.Laplacian(gray, cv2.CV_64F)
            masked_lap = lap[active_mask] if active_mask is not None else lap
            # Refactored sharpness: no zero filtering, scaled by resolution
            sharpness = np.var(masked_lap) if masked_lap.size > 0 else 0
            sharpness_scaled = sharpness / (config.sharpness_base_scale * (gray.size / 500_000))

            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            edge_strength = np.mean(np.sqrt(sobelx**2 + sobely**2))
            # NEW: Size-aware scaling for edge strength
            edge_strength_scaled = edge_strength / (config.edge_strength_base_scale * (gray.size / 500_000))

            # --- Brightness, Contrast, Entropy (on thumbnail) ---
            pixels = gray[active_mask] if active_mask is not None else gray
            mean_br, std_br = (np.mean(pixels), np.std(pixels)) if pixels.size > 0 else (0,0)
            brightness = mean_br / 255.0
            contrast = std_br / (mean_br + 1e-7)
            
            # For entropy, use the original resolution gray image for better histogram accuracy
            gray_full = cv2.cvtColor(self.image_data, cv2.COLOR_BGR2GRAY)
            mask_full = cv2.resize(mask, (gray_full.shape[1], gray_full.shape[0]), interpolation=cv2.INTER_NEAREST) if mask is not None else None
            active_mask_full = (mask_full > 128).astype(np.uint8) if mask_full is not None else None
            hist = cv2.calcHist([gray_full], [0], active_mask_full, [256], [0, 256]).flatten()
            entropy = compute_entropy(hist)
            
            # NIQE calculation (on original resolution image)
            niqe_score = 0.0
            if niqe_metric is not None:
                try:
                    rgb_image = cv2.cvtColor(self.image_data, cv2.COLOR_BGR2RGB)
                    if active_mask_full is not None:
                        mask_3ch = np.stack([active_mask_full] * 3, axis=-1) > 0
                        rgb_image = np.where(mask_3ch, rgb_image, 0)
                    
                    img_tensor = torch.from_numpy(rgb_image).float().permute(2, 0, 1).unsqueeze(0) / 255.0
                    with torch.no_grad(), torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                        niqe_raw = float(niqe_metric(img_tensor.to(niqe_metric.device)))
                        niqe_score = max(0, min(100, (10 - niqe_raw) * 10))
                except Exception as e:
                    logger.warning(f"NIQE calculation failed", extra={'frame': self.frame_number, 'error': e})

            scores_norm = {
                "sharpness": min(sharpness_scaled, 1.0),
                "edge_strength": min(edge_strength_scaled, 1.0),
                "contrast": min(contrast, 2.0) / 2.0, "brightness": brightness, "entropy": entropy,
                "niqe": niqe_score / 100.0
            }
            self.metrics = FrameMetrics(**{f"{k}_score": float(v * 100) for k, v in scores_norm.items()})
            self.metrics.quality_score = float(sum(scores_norm[k] * (config.quality_weights[k] / 100.0) for k in config.QUALITY_METRICS) * 100)
        except Exception as e:
            self.error = f"Quality calc failed: {e}"
            logger.error(f"Frame quality calculation failed", exc_info=True, extra={'frame': self.frame_number})


@dataclass
class AnalysisParameters:
    source_path: str = ""; method: str = ""; interval: float = 0.0
    max_resolution: str = ""; fast_scene: bool = False
    use_png: bool = True; output_folder: str = ""; video_path: str = ""
    disable_parallel: bool = False; resume: bool = False
    enable_face_filter: bool = False; face_ref_img_path: str = ""
    face_model_name: str = ""
    enable_subject_mask: bool = False
    dam4sam_model_name: str = ""
    person_detector_model: str = ""
    seed_strategy: str = ""
    scene_detect: bool = False
    nth_frame: int = 0
    require_face_match: bool = False
    enable_dedup: bool = False
    dedup_thresh: int = 0
    text_prompt: str = ""
    prompt_type_for_video: str = "box"
    
    # These can now be overridden by the UI
    gdino_config_path: str = str(config.GROUNDING_DINO_CONFIG)
    gdino_checkpoint_path: str = str(config.GROUNDING_DINO_CKPT)
    box_threshold: float = config.GROUNDING_BOX_THRESHOLD
    text_threshold: float = config.GROUNDING_TEXT_THRESHOLD
    min_mask_area_pct: float = config.min_mask_area_pct
    sharpness_base_scale: float = config.sharpness_base_scale
    edge_strength_base_scale: float = config.edge_strength_base_scale

    @classmethod
    def from_ui(cls, **kwargs):
        # Create an instance with defaults from config
        instance = cls(**config.ui_defaults)
        # Update with values from UI
        for key, value in kwargs.items():
            if hasattr(instance, key):
                # Coerce types if necessary
                target_type = type(getattr(instance, key))
                try:
                    if value is not None and value != '':
                        setattr(instance, key, target_type(value))
                except (ValueError, TypeError):
                    logger.warning(f"Could not coerce UI value for '{key}' to {target_type}. Using default.", extra={'key': key, 'value': value})
        return instance

# --- Subject Masking Logic ---
@dataclass
class MaskingResult:
    mask_path: str | None = None; shot_id: int | None = None; seed_type: str | None = None
    seed_face_sim: float | None = None; mask_area_pct: float | None = None
    mask_empty: bool = True; error: str | None = None

class SubjectMasker:
    def __init__(self, params, progress_queue, cancel_event, frame_map=None, face_analyzer=None, reference_embedding=None, person_detector=None):
        self.params = params; self.progress_queue = progress_queue; self.cancel_event = cancel_event
        self.frame_map = frame_map; self.face_analyzer = face_analyzer
        self.reference_embedding = reference_embedding; self.person_detector = person_detector
        self.tracker = None; self.mask_dir = None; self.shots = []
        self._gdino = None
        self._sam2_img = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    def _load_image_from_array(self, image_bgr: np.ndarray):
        """Load image from numpy array instead of file path"""
        # Convert BGR to RGB (GroundingDINO expects RGB)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        # Mirror the preprocessing from gdino_load_image
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # The gdino_predict function adds the batch dimension, so we should not add one here.
        image_tensor = transform(image_rgb)
        return image_rgb, image_tensor

    def _init_grounder(self):
        if self._gdino is not None:
            return True
        try:
            # Use paths from params which may have been overridden by UI
            ckpt_path = Path(self.params.gdino_checkpoint_path)
            if not ckpt_path.is_absolute():
                ckpt_path = config.DIRS['models'] / ckpt_path.name
            
            download_model(
                "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth",
                ckpt_path, "GroundingDINO Swin-T model", min_size=500_000_000
            )
            self._gdino = gdino_load_model(
                model_config_path=self.params.gdino_config_path,
                model_checkpoint_path=str(ckpt_path),
                device=self._device,
            )
            logger.info("Grounding DINO model loaded.", extra={'model_path': str(ckpt_path)})
            return True
        except Exception as e:
            logger.warning(f"Grounding DINO unavailable.", exc_info=True)
            self._gdino = None
            return False

    def _ground_first_frame_xywh(self, frame_bgr_small: np.ndarray, text: str):
        if not self._init_grounder():
            return None, {}

        # Load image directly from numpy array, bypassing temp files
        image_source, image_tensor = self._load_image_from_array(frame_bgr_small)
        h, w = image_source.shape[:2]

        with torch.no_grad(), torch.cuda.amp.autocast(enabled=self._device=='cuda'):
            boxes, confidences, labels = gdino_predict(
                model=self._gdino,
                image=image_tensor.to(self._device),
                caption=text,
                box_threshold=float(self.params.box_threshold),
                text_threshold=float(self.params.text_threshold),
            )
        
        if boxes is None or len(boxes) == 0:
            return None, {"type": "text_prompt", "error": "no_boxes"}
            
        scale = torch.tensor([w, h, w, h], device=boxes.device, dtype=boxes.dtype)
        boxes_abs = (boxes * scale).cpu()
        xyxy = box_convert(boxes=boxes_abs, in_fmt="cxcywh", out_fmt="xyxy").numpy()
        conf = confidences.cpu().numpy().tolist()
        
        idx = int(np.argmax(conf))
        x1, y1, x2, y2 = map(float, xyxy[idx])
        xywh = [int(max(0, x1)), int(max(0, y1)), int(max(1, x2 - x1)), int(max(1, y2 - y1))]
        details = {"type": "text_prompt", "label": labels[idx] if labels else "", "conf": float(conf[idx])}
        
        return xywh, details

    def _ground_first_frame_mask_xywh(self, frame_bgr_small: np.ndarray, text: str):
        xywh, details = self._ground_first_frame_xywh(frame_bgr_small, text)
        if xywh is None:
            return None, details

        # Use DAM4SAM directly instead of separate SAM2 predictor
        mask = self._sam2_mask_for_bbox(frame_bgr_small, xywh)
        if mask is None:
            logger.warning("SAM2 mask generation failed. Falling back to box prompt.")
            return xywh, details

        # Find tight bounding box around the mask
        ys, xs = np.where(mask > 128)
        if ys.size == 0:
            return xywh, details
        x1, x2, y1, y2 = xs.min(), xs.max()+1, ys.min(), ys.max()+1
        return [int(x1), int(y1), int(x2 - x1), int(y2 - y1)], {**details, "type": "text_prompt_mask"}

    def run(self, video_path: str, frames_dir: str) -> dict[str, dict]:
        self.mask_dir = Path(frames_dir) / "masks"
        self.mask_dir.mkdir(exist_ok=True)
        logger.info("Starting subject masking...")

        if self.params.scene_detect: self._detect_scenes(video_path, frames_dir)
        
        if not self._initialize_tracker():
            logger.error("Could not initialize tracker; skipping masking.")
            return {}
        
        if not self.shots:
            if self.frame_map:
                self.shots = [(0, max(self.frame_map.keys()) + 1 if self.frame_map else 0)]
            else:
                image_files = list(Path(frames_dir).glob("frame_*.*"))
                self.shots = [(0, len(image_files))]

        mask_metadata = {}
        for shot_id, (start_frame, end_frame) in enumerate(self.shots):
            with safe_resource_cleanup():
                if self.cancel_event.is_set(): break
                self.progress_queue.put({"stage": f"Masking Shot {shot_id+1}/{len(self.shots)}"})
                shot_context = {'shot_id': shot_id, 'start_frame': start_frame, 'end_frame': end_frame}
                logger.info(f"Masking shot", extra=shot_context)
                shot_frames_data = self._load_shot_frames(frames_dir, start_frame, end_frame)
                if not shot_frames_data: continue

                small_images = [f[1] for f in shot_frames_data]
                seed_idx, bbox, seed_details = self._seed_identity(small_images)
                
                if bbox is None:
                    for fn, _, _ in shot_frames_data:
                        if (fname := self.frame_map.get(fn)):
                            mask_metadata[fname] = asdict(MaskingResult(error="Subject not found", shot_id=shot_id))
                    continue
                
                masks, areas, empties, errors = self._propagate_masks(small_images, seed_idx, bbox)

                for i, (original_fn, _, (h, w)) in enumerate(shot_frames_data):
                    if not (frame_fname := self.frame_map.get(original_fn)): continue
                    mask_path = self.mask_dir / f"{Path(frame_fname).stem}.png"
                    result_args = {
                        "shot_id": shot_id, "seed_type": seed_details.get('type'),
                        "seed_face_sim": seed_details.get('seed_face_sim'), "mask_area_pct": areas[i],
                        "mask_empty": empties[i], "error": errors[i]
                    }
                    if masks[i] is not None and np.any(masks[i]):
                        mask_full_res = cv2.resize(masks[i], (w, h), interpolation=cv2.INTER_NEAREST)
                        if mask_full_res.ndim == 3:
                            mask_full_res = mask_full_res[:, :, 0]
                        cv2.imwrite(str(mask_path), mask_full_res)
                        mask_metadata[frame_fname] = asdict(MaskingResult(mask_path=str(mask_path), **result_args))
                    else:
                        mask_metadata[frame_fname] = asdict(MaskingResult(mask_path=None, **result_args))
        logger.success("Subject masking complete.")
        return mask_metadata

    def _initialize_tracker(self):
        if not all([DAM4SAMTracker, torch, torch.cuda.is_available()]):
            logger.error("DAM4SAM dependencies or CUDA not available.")
            return False
        try:
            model_name = self.params.dam4sam_model_name
            logger.info(f"Initializing DAM4SAM tracker", extra={'model': model_name})
            model_urls = {
                "sam21pp-T": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt",
                "sam21pp-S": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt",
                "sam21pp-B+": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt",
                "sam21pp-L": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
            }
            checkpoint_path = config.DIRS['models'] / Path(model_urls[model_name]).name
            download_model(model_urls[model_name], checkpoint_path, f"{model_name} model", 100_000_000)
            
            from DAM4SAM.utils.utils import determine_tracker
            actual_path, _ = determine_tracker(model_name)
            if not Path(actual_path).exists():
                Path(actual_path).parent.mkdir(exist_ok=True, parents=True)
                shutil.copy(checkpoint_path, actual_path)

            self.tracker = DAM4SAMTracker(model_name)
            logger.success("DAM4SAM tracker initialized.")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize DAM4SAM tracker", exc_info=True)
            return False

    def _detect_scenes(self, video_path: str, frames_dir: str):
        if not detect:
            raise ImportError("PySceneDetect is required.")
        try:
            logger.info("Detecting scene cuts...")
            scene_list = detect(video_path, ContentDetector())
            self.shots = [(s.frame_num, e.frame_num) for s, e in scene_list] if scene_list else []
            logger.info(f"Found {len(self.shots)} shots.")
        except Exception as e:
            logger.critical(f"Scene detection failed", exc_info=True)
            self.shots = []

    def _load_shot_frames(self, frames_dir, start, end):
        frames = []
        if not self.frame_map: 
            # Fallback for preview mode where frame_map might not be initialized
            image_files = sorted(list(Path(frames_dir).glob("frame_*.png")) + list(Path(frames_dir).glob("frame_*.jpg")),
                                key=lambda p: int(re.search(r'frame_(\d+)', p.name).group(1)))
            self.frame_map = {int(re.search(r'frame_(\d+)', f.name).group(1)): f.name for f in image_files}

        thumb_dir = Path(frames_dir) / "thumbs"

        for fn in sorted(fn for fn in self.frame_map if start <= fn < end):
            original_p = Path(frames_dir) / self.frame_map[fn]
            thumb_p = thumb_dir / f"{original_p.stem}.jpg"
            
            if not thumb_p.exists():
                continue
            
            thumb_img = cv2.imread(str(thumb_p))
            if thumb_img is None:
                continue

            # Get original dimensions efficiently without loading the full image
            try:
                with Image.open(original_p) as img_pil:
                    w, h = img_pil.size
            except Exception as e:
                logger.warning(f"Could not read dimensions for {original_p.name}, skipping.", extra={'error': e})
                continue
                
            frames.append((fn, thumb_img, (h, w)))
        return frames

    def _seed_identity(self, shot_frames):
        if not shot_frames:
            return None, None, {}

        if getattr(self.params, "text_prompt", ""):
            if getattr(self.params, "prompt_type_for_video", "box") == "mask":
                xywh, details = self._ground_first_frame_mask_xywh(shot_frames[0], self.params.text_prompt)
            else:
                xywh, details = self._ground_first_frame_xywh(shot_frames[0], self.params.text_prompt)
            if xywh is not None:
                logger.info(f"Text-prompt seed found", extra=details)
                return 0, xywh, details
            else:
                logger.warning("Text-prompt grounding returned no boxes; falling back to existing strategy.")
        
        # Combined logic for choosing a seed bbox from person/face
        return self._choose_seed_bbox(shot_frames)
    
    def _choose_seed_bbox(self, shot_frames):
        """Refactored logic to select the best seed bounding box."""
        if self.face_analyzer and self.reference_embedding is not None and self.params.enable_face_filter:
            logger.info("Searching for reference face...")
            best_face, best_dist, seed_idx = None, float('inf'), -1
            for i, frame in enumerate(shot_frames[:5]):
                faces = self.face_analyzer.get(frame) if frame is not None else []
                for face in faces:
                    dist = 1 - np.dot(face.normed_embedding, self.reference_embedding)
                    if dist < best_dist:
                        best_dist, best_face, seed_idx = dist, face, i
            
            if best_face and best_dist < 0.6:
                logger.info(f"Found reference face", extra={'frame_index': seed_idx, 'distance': f"{best_dist:.2f}"})
                details = {'type': 'face_match', 'seed_face_sim': 1 - best_dist}
                face_bbox = best_face.bbox.astype(int)
                final_bbox = self._get_body_box_for_face(shot_frames[seed_idx], face_bbox, details)
                return seed_idx, final_bbox, details

        # Fallback strategies if no reference face match
        logger.info("No matching reference face found. Applying fallback seeding strategy.", extra={'strategy': self.params.seed_strategy})
        seed_idx = 0
        first_frame = shot_frames[0]
        
        if self.params.seed_strategy in ["Largest Person", "Center-most Person"] and self.person_detector:
            boxes = self.person_detector.detect_boxes(first_frame)
            if boxes:
                h, w = first_frame.shape[:2]; cx, cy = w / 2, h / 2
                strategy_map = {
                    "Largest Person": lambda b: (b[2] - b[0]) * (b[3] - b[1]),
                    "Center-most Person": lambda b: -math.hypot((b[0] + b[2]) / 2 - cx, (b[1] + b[3]) / 2 - cy)
                }
                score_func = strategy_map[self.params.seed_strategy]
                x1, y1, x2, y2, _ = sorted(boxes, key=score_func, reverse=True)[0]
                return seed_idx, [x1, y1, x2 - x1, y2 - y1], {'type': f'person_{self.params.seed_strategy.lower().replace(" ", "_")}'}

        # Fallback to largest face if other strategies fail or aren't selected
        if self.face_analyzer:
            faces = self.face_analyzer.get(first_frame)
            if faces:
                largest_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
                details = {'type': 'face_largest'}
                face_bbox = largest_face.bbox.astype(int)
                final_bbox = self._get_body_box_for_face(first_frame, face_bbox, details)
                return seed_idx, final_bbox, details

        logger.warning("No faces or persons found to seed shot. Using fallback rectangle.")
        h, w, _ = first_frame.shape
        return 0, [w // 4, h // 4, w // 2, h // 2], {'type': 'fallback_rect'}

    def _get_body_box_for_face(self, frame_img, face_bbox, details_dict):
        """Helper to find a person box for a face or expand the face box."""
        x1, y1, x2, y2 = face_bbox
        person_bbox = self._pick_person_box_for_face(frame_img, [x1, y1, x2-x1, y2-y1])
        if person_bbox:
            logger.info(f"Seeding with person box for face.", extra={'box': person_bbox})
            details_dict['type'] = f'person_box_from_{details_dict["type"]}'
            return person_bbox
        else:
            expanded_box = self._expand_face_to_body([x1, y1, x2-x1, y2-y1], frame_img.shape)
            logger.info(f"Seeding with heuristic expansion for face.", extra={'box': expanded_box})
            details_dict['type'] = f'expanded_box_from_{details_dict["type"]}'
            return expanded_box
        
    def _pick_person_box_for_face(self, frame_img, face_bbox):
        if not self.person_detector: return None
        px1, py1, pw, ph = face_bbox
        fx, fy = px1 + pw / 2.0, py1 + ph / 2.0
        try:
            candidates = self.person_detector.detect_boxes(frame_img)
        except Exception as e:
            logger.warning(f"Person detector failed on frame.", extra={'error': e})
            return None
        if not candidates: return None
        
        def iou(b):
            ix1, iy1 = max(b[0], px1), max(b[1], py1)
            ix2, iy2 = min(b[2], px1 + pw), min(b[3], py1 + ph)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            union = (b[2]-b[0])*(b[3]-b[1]) + pw*ph - inter + 1e-6
            return inter / union

        pool = sorted(candidates, key=lambda b: ((b[0]<=fx<=b[2] and b[1]<=fy<=b[3]), iou(b), b[4]), reverse=True)
        best_box = pool[0]
        if not (best_box[0] <= fx <= best_box[2] and best_box[1] <= fy <= best_box[3]) and iou(best_box) < 0.1:
            return None
        return [best_box[0], best_box[1], best_box[2] - best_box[0], best_box[3] - best_box[1]]

    def _expand_face_to_body(self, face_bbox, img_shape):
        H, W = img_shape[:2]; x, y, w, h = face_bbox; cx = x + w / 2
        new_w = min(W, w * 4.0); new_h = min(H, h * 7.0)
        new_x = max(0, cx - new_w / 2); new_y = max(0, y - h * 0.75)
        return [int(v) for v in [new_x, new_y, min(new_w, W-new_x), min(new_h, H-new_y)]]
    
    def _propagate_masks(self, shot_frames, seed_idx, bbox_xywh):
        if not self.tracker or not shot_frames:
            err_msg = "Tracker not initialized" if not self.tracker else "No frames"
            shape = shot_frames[0].shape[:2] if shot_frames else (100, 100)
            return ([np.zeros(shape, np.uint8)] * len(shot_frames), [0.0] * len(shot_frames), [True] * len(shot_frames), [err_msg] * len(shot_frames))

        logger.info(f"Propagating masks", extra={'num_frames': len(shot_frames), 'seed_index': seed_idx})
        self.progress_queue.put({"stage": "Masking", "total": len(shot_frames)})
        masks = [None] * len(shot_frames)

        def _propagate_direction(start_idx, end_idx, step):
            for i in range(start_idx, end_idx, step):
                if self.cancel_event.is_set(): break
                frame_pil = bgr_to_pil(shot_frames[i])
                outputs = self.tracker.track(frame_pil)
                mask = outputs.get('pred_mask')
                masks[i] = (mask * 255).astype(np.uint8) if mask is not None else np.zeros_like(shot_frames[i], dtype=np.uint8)[:, :, 0]
                self.progress_queue.put({"progress": 1})

        try:
            with torch.cuda.amp.autocast(enabled=self._device == 'cuda'):
                seed_frame_pil = bgr_to_pil(shot_frames[seed_idx])
                
                # Initialize for forward pass
                outputs = self.tracker.initialize(seed_frame_pil, None, bbox=bbox_xywh)
                mask = outputs.get('pred_mask')
                masks[seed_idx] = (mask * 255).astype(np.uint8) if mask is not None else np.zeros_like(shot_frames[seed_idx], dtype=np.uint8)[:, :, 0]
                self.progress_queue.put({"progress": 1})
                
                _propagate_direction(seed_idx + 1, len(shot_frames), 1)

                # Re-initialize for backward pass
                self.tracker.initialize(seed_frame_pil, None, bbox=bbox_xywh)
                _propagate_direction(seed_idx - 1, -1, -1)

            # Finalization step
            h, w = shot_frames[0].shape[:2]
            final_results = []
            for i, mask in enumerate(masks):
                if self.cancel_event.is_set() or mask is None:
                    mask = np.zeros((h, w), dtype=np.uint8)
                
                img_area = h * w
                area_pct = (np.sum(mask > 0) / img_area) * 100 if img_area > 0 else 0.0
                is_empty = area_pct < self.params.min_mask_area_pct
                error = "Empty mask" if is_empty else None
                final_results.append((mask, float(area_pct), bool(is_empty), error))
            
            return tuple(zip(*final_results)) if final_results else ([], [], [], [])

        except Exception as e:
            logger.critical(f"DAM4SAM propagation failed", exc_info=True)
            h, w = shot_frames[0].shape[:2]
            error_msg = f"Propagation failed: {e}"
            return ([np.zeros((h, w), np.uint8)] * len(shot_frames), [0.0] * len(shot_frames), [True] * len(shot_frames), [error_msg] * len(shot_frames))

    def _draw_bbox(self, img_bgr, xywh, color=(0, 0, 255), thickness=2):
        x, y, w, h = map(int, xywh or [0, 0, 0, 0])
        img_out = img_bgr.copy()
        cv2.rectangle(img_out, (x, y), (x + w, y + h), color, thickness)
        return img_out

    def _sam2_mask_for_bbox(self, frame_bgr_small, bbox_xywh):
        """Generate mask using DAM4SAM's tracker directly"""
        if not self.tracker or bbox_xywh is None:
            return None
        
        # Use DAM4SAM's initialize method which handles SAM2 internally
        try:
            outputs = self.tracker.initialize(bgr_to_pil(frame_bgr_small), None, bbox=bbox_xywh)
            mask = outputs.get('pred_mask')
            return (mask * 255).astype(np.uint8) if mask is not None else None
        except Exception as e:
            logger.warning(f"DAM4SAM mask generation failed.", extra={'error': e})
            return None

    def preview_seeds(self, video_path: str, frames_dir: str, max_scenes: int = 100):
        # Initialize tracker if needed for mask previews
        if not self.tracker:
            self._initialize_tracker()
            
        previews = []
        if not video_path or not Path(video_path).exists():
            self.progress_queue.put({"log": "[WARNING] Video path missing for seeding preview; enable scene detection with a valid video file."})
            return previews
        self._detect_scenes(video_path, frames_dir)
        if not self.shots:
            self.progress_queue.put({"log": "[WARNING] No scenes found; cannot preview seeding."})
            return previews
        shots = self.shots[:max_scenes]
        for shot_id, (start_frame, end_frame) in enumerate(shots):
            data = self._load_shot_frames(frames_dir, start_frame, end_frame)
            if not data:
                continue
            # first entry: (orig_fn, thumb_img, (h,w))
            _, thumb_bgr, _ = data[0]
            seed_idx, bbox, details = self._seed_identity([thumb_bgr])
            caption = f"Shot {shot_id+1}: {details.get('type','?')}"
            if 'conf' in details:
                caption += f" conf={details['conf']:.2f}"
            # choose preview mode
            mask = None
            if self.params.prompt_type_for_video == 'mask' and bbox is not None:
                mask = self._sam2_mask_for_bbox(thumb_bgr, bbox)
            if mask is not None:
                overlay = render_mask_overlay(thumb_bgr, mask, alpha=0.5)
                previews.append((cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), caption))
            else:
                with_box = self._draw_bbox(thumb_bgr, bbox) if bbox is not None else thumb_bgr
                previews.append((cv2.cvtColor(with_box, cv2.COLOR_BGR2RGB), caption))
        return previews


# --- Backend Analysis Pipeline ---
class VideoManager:
    def __init__(self, source_path, max_resolution="maximum available"):
        self.source_path = source_path; self.max_resolution = max_resolution
        self.is_youtube = "youtube.com/" in source_path or "youtu.be/" in source_path

    def prepare_video(self):
        if self.is_youtube:
            if not ytdlp: raise ImportError("yt-dlp not installed.")
            logger.info(f"Downloading video", extra={'source': self.source_path})
            res_filter = f"[height<={self.max_resolution}]" if self.max_resolution != "maximum available" else ""
            ydl_opts = {
                'outtmpl': str(config.DIRS['downloads'] / '%(id)s_%(title).40s_%(height)sp.%(ext)s'),
                'format': f'bestvideo{res_filter}[ext=mp4]+bestaudio[ext=m4a]/best{res_filter}[ext=mp4]/best',
                'merge_output_format': 'mp4', 'noprogress': True, 'quiet': True
            }
            try:
                with ytdlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(self.source_path, download=True)
                    return str(Path(ydl.prepare_filename(info)))
            except ytdlp.utils.DownloadError as e:
                raise RuntimeError(f"Download failed. Resolution may not be available. Details: {e}") from e
        
        local_path = Path(self.source_path)
        if not local_path.is_file(): raise FileNotFoundError(f"Video file not found: {local_path}")
        return str(local_path)

    @staticmethod
    def get_video_info(video_path):
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened(): raise IOError(f"Could not open video: {video_path}")
        info = {"width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "fps": cap.get(cv2.CAP_PROP_FPS), "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}
        cap.release()
        return info

class Pipeline:
    def __init__(self, params: AnalysisParameters, progress_queue: Queue, cancel_event: threading.Event):
        self.params = params; self.progress_queue = progress_queue
        self.cancel_event = cancel_event; self.logger = logger

class ExtractionPipeline(Pipeline):
    def run(self):
        try:
            logger.info("Preparing video source...")
            vid_manager = VideoManager(self.params.source_path, self.params.max_resolution)
            video_path = Path(vid_manager.prepare_video())
            logger.info(f"Video ready", extra={'path': sanitize_filename(video_path.name)})

            video_info = VideoManager.get_video_info(video_path)
            output_dir = config.DIRS['downloads'] / video_path.stem
            output_dir.mkdir(exist_ok=True)
            self._run_ffmpeg(video_path, output_dir, video_info)

            if self.cancel_event.is_set(): return {"log": "Extraction cancelled."}
            logger.success("Extraction complete.")
            return {"done": True, "output_dir": str(output_dir), "video_path": str(video_path)}
        except Exception as e:
            return logger.pipeline_error("extraction", e)

    def _run_ffmpeg(self, video_path, output_dir, video_info):
        use_showinfo = self.params.method != 'all'
        select_filter = {
            'interval': f"fps=1/{max(0.1, float(self.params.interval))}", 'keyframes': "select='eq(pict_type,I)'",
            'scene': f"select='gt(scene,{0.5 if self.params.fast_scene else 0.4})'", 'all': f"fps={video_info.get('fps', 30)}",
            'every_nth_frame': f"select='not(mod(n,{max(1, int(self.params.nth_frame))}))'"
        }.get(self.params.method)
        
        cmd = ['ffmpeg', '-y', '-i', str(video_path), '-hide_banner', '-loglevel', 'info' if use_showinfo else 'error', '-progress', 'pipe:1']
        filter_str = (select_filter + ",showinfo") if use_showinfo and select_filter else "showinfo" if use_showinfo else select_filter
        if filter_str: cmd.extend(['-vf', filter_str, '-vsync', 'vfr'])
        cmd.extend(['-f', 'image2', str(output_dir / f"frame_%06d.{'png' if self.params.use_png else 'jpg'}")])
        
        log_file_path = output_dir / "ffmpeg_log.txt"
        stderr_handle = open(log_file_path, 'w') if use_showinfo else subprocess.DEVNULL
        
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=stderr_handle, text=True, encoding='utf-8', bufsize=1)
        
        def monitor_progress():
            while process.poll() is None:
                if (line := process.stdout.readline()) and (frame_match := re.search(r'frame=\s*(\d+)', line)):
                    self.progress_queue.put({"progress_abs": int(frame_match.group(1))})
                else: time.sleep(0.1)

        progress_thread = threading.Thread(target=monitor_progress, daemon=True)
        progress_thread.start()
        self.progress_queue.put({"total": video_info.get('frame_count', 1), "stage": "Extraction"})
        
        while process.poll() is None:
            if self.cancel_event.is_set(): process.terminate(); break
            time.sleep(0.1)
        process.wait()
        
        if stderr_handle != subprocess.DEVNULL: stderr_handle.close()
        
        if use_showinfo and log_file_path.exists():
            try:
                with open(log_file_path, 'r') as f:
                    frame_map_list = [int(m.group(1)) for l in f if (m := re.search(r' n:\s*(\d+)', l))]
                with open(output_dir / "frame_map.json", 'w') as f:
                    json.dump(frame_map_list, f)
            finally:
                log_file_path.unlink(missing_ok=True)
        
        if process.returncode != 0 and not self.cancel_event.is_set():
            raise RuntimeError(f"FFmpeg failed with code {process.returncode}.")

class AnalysisPipeline(Pipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_dir = Path(self.params.output_folder)
        self.metadata_path = self.output_dir / "metadata.jsonl"
        self.write_lock = threading.Lock(); self.gpu_lock = threading.Lock()
        self.face_analyzer = None; self.reference_embedding = None; self.mask_metadata = {}
        self.niqe_metric = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Shared resources
        self.shared_analyzers = {}

    def _get_shared_analyzer(self, key, factory):
        if key not in self.shared_analyzers:
            self.shared_analyzers[key] = factory()
        return self.shared_analyzers[key]
        
    def _initialize_niqe_metric(self):
        if self.niqe_metric is None:
            try:
                self.niqe_metric = pyiqa.create_metric('niqe', device=self.device)
                logger.info("NIQE metric initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize NIQE metric", extra={'error': e})

    def run(self):
        run_log_handler = None
        try:
            if not self.output_dir.is_dir(): raise ValueError("Output folder is required.")
            
            # Create and add a log handler for this specific run
            run_log_path = self.output_dir / "analysis_run.log"
            run_log_handler = logging.FileHandler(run_log_path, mode='w', encoding='utf-8')
            run_log_handler.setFormatter(StructuredFormatter('%(asctime)s - %(levelname)s - %(message)s'))
            self.logger.add_handler(run_log_handler)

            config_hash = self._get_config_hash()
            if self.params.resume and self._check_resume(config_hash):
                logger.success("Resuming previous analysis.", extra={'output_dir': self.output_dir})
                return {"done": True, "metadata_path": str(self.metadata_path), "output_dir": str(self.output_dir)}
            
            if (self.output_dir / "masks").exists(): shutil.rmtree(self.output_dir / "masks")
            self.metadata_path.unlink(missing_ok=True)
            with self.metadata_path.open('w') as f:
                header_params = {k:v for k,v in asdict(self.params).items() if k not in ['source_path', 'output_folder', 'video_path']}
                header_params['sharpness_base_scale'] = self.params.sharpness_base_scale
                header_params['edge_strength_base_scale'] = self.params.edge_strength_base_scale
                header = {"config_hash": config_hash, "params": header_params}
                f.write(json.dumps(header) + '\n')

            # Prepare thumbnails first to be used by all subsequent steps
            self._prepare_thumbnails()

            needs_face_analyzer = self.params.enable_face_filter or \
                                  (self.params.enable_subject_mask and "Reference Face" in self.params.seed_strategy and not self.params.text_prompt)
            if needs_face_analyzer: self._initialize_face_analyzer()
            if self.params.enable_face_filter and self.params.face_ref_img_path: self._process_reference_face()
            
            person_detector = None
            if self.params.enable_subject_mask:
                person_detector = self._get_shared_analyzer(
                    f"person_{self.params.person_detector_model}",
                    lambda: PersonDetector(model=self.params.person_detector_model, device=self.device)
                )

            if self.params.enable_subject_mask:
                is_video_path_valid = self.params.video_path and Path(self.params.video_path).exists()
                if self.params.scene_detect and not is_video_path_valid:
                    logger.warning("Valid video path not provided; scene detection for masking disabled.")
                masker = SubjectMasker(self.params, self.progress_queue, self.cancel_event, self._create_frame_map(),
                                       self.face_analyzer, self.reference_embedding, person_detector)
                self.mask_metadata = masker.run(self.params.video_path if is_video_path_valid else "", str(self.output_dir))
            
            self._run_analysis_loop()
            if self.cancel_event.is_set(): return {"log": "Analysis cancelled."}
            logger.success("Analysis complete.", extra={'output_dir': self.output_dir})
            return {"done": True, "metadata_path": str(self.metadata_path), "output_dir": str(self.output_dir)}
        except Exception as e:
            return logger.pipeline_error("analysis", e)
        finally:
            self.logger.remove_handler(run_log_handler)

    def _get_config_hash(self):
        d = asdict(self.params)
        params_to_hash = {k: d.get(k) for k in ['enable_subject_mask', 'scene_detect', 'enable_face_filter',
                                                'face_model_name', 'dam4sam_model_name', 'min_mask_area_pct',
                                                'sharpness_base_scale', 'edge_strength_base_scale']}
        params_to_hash['quality_weights'] = config.quality_weights
        return hashlib.sha1(json.dumps(params_to_hash, sort_keys=True).encode()).hexdigest()

    def _check_resume(self, current_hash):
        if not self.metadata_path.is_file() or self.metadata_path.stat().st_size == 0: return False
        with self.metadata_path.open('r') as f:
            try:
                header = json.loads(f.readline())
                if header.get("config_hash") == current_hash:
                    logger.info("Resuming with compatible metadata.")
                    return True
            except (json.JSONDecodeError, IndexError): pass
        logger.warning("Config changed or metadata invalid. Re-running analysis.")
        return False
    
    def _create_frame_map(self):
        logger.info("Loading frame map...")
        frame_map_path = self.output_dir / "frame_map.json"
        image_files = sorted(list(self.output_dir.glob("frame_*.png")) + list(self.output_dir.glob("frame_*.jpg")),
                             key=lambda p: int(re.search(r'frame_(\d+)', p.name).group(1)))
        if not frame_map_path.exists():
            logger.warning("frame_map.json not found. Assuming sequential mapping.")
            return {int(re.search(r'frame_(\d+)', f.name).group(1)): f.name for f in image_files}
        try:
            with open(frame_map_path, 'r') as f: frame_map_list = json.load(f)
            return {orig_num: image_files[i].name for i, orig_num in enumerate(sorted(frame_map_list)) if i < len(image_files)}
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse frame_map.json. Using filename-based mapping.", exc_info=True)
            return {int(re.search(r'frame_(\d+)', f.name).group(1)): f.name for f in image_files}

    def _initialize_face_analyzer(self):
        if not FaceAnalysis: raise ImportError("insightface library not installed.")
        self.face_analyzer = self._get_shared_analyzer(
            self.params.face_model_name,
            lambda: self._create_face_analysis_instance()
        )
    
    def _create_face_analysis_instance(self):
        logger.info(f"Loading face model", extra={'model': self.params.face_model_name})
        try:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.device == 'cuda' else ['CPUExecutionProvider']
            analyzer = FaceAnalysis(name=self.params.face_model_name, root=str(config.DIRS['models']), providers=providers)
            analyzer.prepare(ctx_id=0 if self.device == 'cuda' else -1, det_size=(640, 640))
            logger.success(f"Face model loaded", extra={'device': 'CUDA' if self.device == 'cuda' else 'CPU'})
            return analyzer
        except Exception as e:
            raise RuntimeError(f"Could not initialize face analysis model. Error: {e}") from e


    def _process_reference_face(self):
        if not self.face_analyzer: return
        ref_path = Path(self.params.face_ref_img_path)
        if not ref_path.is_file(): raise FileNotFoundError(f"Reference face image not found: {ref_path}")
        logger.info("Processing reference face...")
        ref_img = cv2.imread(str(ref_path))
        if ref_img is None: raise ValueError("Could not read reference image.")
        ref_faces = self.face_analyzer.get(ref_img)
        if not ref_faces: raise ValueError("No face found in reference image.")
        self.reference_embedding = max(ref_faces, key=lambda x: x.det_score).normed_embedding
        logger.success("Reference face processed.")

    def _prepare_thumbnails(self):
        image_files = sorted(list(self.output_dir.glob("frame_*.png")) + list(self.output_dir.glob("frame_*.jpg")))
        thumb_dir = self.output_dir / "thumbs"
        thumb_dir.mkdir(exist_ok=True)
        logger.info("Generating thumbnails...")
        self.progress_queue.put({"total": len(image_files), "stage": "Thumbnails"})
        
        for img_path in image_files:
            if self.cancel_event.is_set(): break
            thumb_path = thumb_dir / f"{img_path.stem}.jpg"
            if thumb_path.exists():
                self.progress_queue.put({"progress": 1})
                continue
            try:
                img = cv2.imread(str(img_path))
                if img is None: continue
                h, w = img.shape[:2]
                if (h*w) == 0: continue
                scale = math.sqrt(500_000 / (h * w)) if (h*w) > 500_000 else 1.0
                thumb = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA) if scale < 1.0 else img
                cv2.imwrite(str(thumb_path), thumb, [cv2.IMWRITE_JPEG_QUALITY, 85])
                self.progress_queue.put({"progress": 1})
            except Exception as e:
                logger.error(f"Failed to create thumbnail", extra={'file': img_path.name, 'error': e})

    def _run_analysis_loop(self):
        image_files = sorted(list(self.output_dir.glob("frame_*.png")) + list(self.output_dir.glob("frame_*.jpg")))
        
        self.progress_queue.put({"total": len(image_files), "stage": "Analysis"})
        num_workers = 1 if self.params.disable_parallel or self.params.enable_face_filter else min(os.cpu_count() or 4, 8)
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            list(executor.map(self._process_single_frame, image_files))

    def _process_single_frame(self, image_path):
        if self.cancel_event.is_set(): return
        
        frame_num_match = re.search(r'frame_(\d+)', image_path.name)
        frame_num = int(frame_num_match.group(1)) if frame_num_match else -1
        log_context = {'file': image_path.name, 'frame_num': frame_num}

        try:
            self._initialize_niqe_metric()

            image_data = cv2.imread(str(image_path))
            if image_data is None: raise ValueError("Could not read image.")

            thumb_path = self.output_dir / "thumbs" / f"{image_path.stem}.jpg"
            thumb_image = cv2.imread(str(thumb_path))
            if thumb_image is None: raise ValueError("Could not read thumbnail.")
            
            frame = Frame(image_data, frame_num)
            mask_meta = self.mask_metadata.get(image_path.name, {})
            
            mask_thumb = None
            if mask_meta.get("mask_path"):
                mask_full = cv2.imread(mask_meta["mask_path"], cv2.IMREAD_GRAYSCALE)
                if mask_full is not None:
                    mask_thumb = cv2.resize(mask_full, (thumb_image.shape[1], thumb_image.shape[0]), interpolation=cv2.INTER_NEAREST)

            frame.calculate_quality_metrics(thumb_image=thumb_image, mask=mask_thumb, niqe_metric=self.niqe_metric)

            if self.params.enable_face_filter and self.reference_embedding is not None and self.face_analyzer:
                self._analyze_face_similarity(frame)
            
            meta = {"filename": image_path.name, "metrics": asdict(frame.metrics)}
            if frame.face_similarity_score is not None: meta["face_sim"] = frame.face_similarity_score
            if frame.max_face_confidence is not None: meta["face_conf"] = frame.max_face_confidence
            meta.update(mask_meta)

            if self.params.enable_dedup:
                pil_thumb = bgr_to_pil(thumb_image)
                meta['phash'] = str(imagehash.phash(pil_thumb))

            if frame.error: meta["error"] = frame.error
            if meta.get("mask_path"): meta["mask_path"] = Path(meta["mask_path"]).name
            
            meta = _to_json_safe(meta)
            with self.write_lock, self.metadata_path.open('a') as f:
                json.dump(meta, f)
                f.write('\n')
            self.progress_queue.put({"progress": 1})
        except Exception as e:
            logger.critical(f"Error processing frame", exc_info=True, extra={**log_context, 'error': e})
            meta = {"filename": image_path.name, "error": f"processing_failed: {e}"}
            with self.write_lock, self.metadata_path.open('a') as f:
                json.dump(meta, f)
                f.write('\n')
            self.progress_queue.put({"progress": 1})

    def _analyze_face_similarity(self, frame):
        try:
            with self.gpu_lock: faces = self.face_analyzer.get(frame.image_data)
            if faces:
                best_face = max(faces, key=lambda x: x.det_score)
                distance = 1 - np.dot(best_face.normed_embedding, self.reference_embedding)
                frame.face_similarity_score = 1.0 - float(distance)
                frame.max_face_confidence = float(best_face.det_score)
        except Exception as e:
            frame.error = f"Face similarity failed: {e}"

# --- Gradio UI & Event Handlers ---
class AppUI:
    def __init__(self):
        self.components = {}
        self.cancel_event = threading.Event()
        self.last_task_result = {}
        self.cuda_available = torch.cuda.is_available()
        # Centralized place for heavy, reusable models
        self.shared_analyzers = {}
        self.ext_ui_map_keys = [
            'source_path', 'upload_video', 'method', 'interval', 'nth_frame',
            'fast_scene', 'max_resolution', 'use_png'
        ]
        self.ana_ui_map_keys = [
            'output_folder', 'video_path', 'resume', 'enable_face_filter',
            'face_ref_img_path', 'face_ref_img_upload', 'face_model_name', 'enable_subject_mask',
            'dam4sam_model_name', 'person_detector_model', 'seed_strategy', 'scene_detect',
            'enable_dedup', 'text_prompt', 'prompt_type_for_video', 'box_threshold',
            'text_threshold', 'min_mask_area_pct', 'sharpness_base_scale',
            'edge_strength_base_scale', 'gdino_config_path', 'gdino_checkpoint_path'
        ]

    def build_ui(self):
        css = """
        .plot-and-slider-column {
            max-width: 560px !important;
            margin: auto;
        }
        """
        with gr.Blocks(theme=gr.themes.Default(), css=css) as demo:
            gr.Markdown("# 🎬 Frame Extractor & Analyzer")
            if not self.cuda_available:
                gr.Markdown("⚠️ **CPU Mode** — GPU-dependent features are disabled or will be slow.")
            
            with gr.Tabs():
                with gr.Tab("📹 1. Frame Extraction"): self._create_extraction_tab()
                with gr.Tab("🔍 2. Frame Analysis") as self.components['analysis_tab']: self._create_analysis_tab()
                with gr.Tab("🎯 3. Filtering & Export") as self.components['filtering_tab']: self._create_filtering_tab()

            with gr.Row():
                with gr.Column(scale=2):
                    self._create_component('unified_log', 'textbox', {'label': "📋 Processing Log", 'lines': 10, 'interactive': False, 'autoscroll': True})
                with gr.Column(scale=1):
                    self._create_component('unified_status', 'textbox', {'label': "📊 Status Summary", 'lines': 2, 'interactive': False})
            
            self._create_event_handlers()
        return demo

    def _create_component(self, name, comp_type, kwargs):
        comp_map = {'button': gr.Button, 'textbox': gr.Textbox, 'dropdown': gr.Dropdown, 'slider': gr.Slider,
                    'checkbox': gr.Checkbox, 'file': gr.File, 'radio': gr.Radio, 'gallery': gr.Gallery,
                    'plot': gr.Plot, 'markdown': gr.Markdown, 'html': gr.HTML, 'number': gr.Number}
        self.components[name] = comp_map[comp_type](**kwargs)
        return self.components[name]

    def _create_extraction_tab(self):
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 📹 Video Source")
                self._create_component('source_input', 'textbox', {'label': "Video URL or Local Path", 'placeholder': "Enter YouTube URL or local video file path"})
                self._create_component('upload_video_input', 'file', {'label': "Or Upload Video", 'file_types': ["video"], 'type': "filepath"})
            with gr.Column():
                gr.Markdown("### ⚙️ Extraction Settings")
                method_choices = ["keyframes", "interval", "every_nth_frame", "all", "scene"]
                self._create_component('method_input', 'dropdown', {'choices': method_choices, 'value': config.ui_defaults['method'], 'label': "Method"})
                self._create_component('interval_input', 'textbox', {'label': "Interval (s)", 'value': config.ui_defaults['interval'], 'visible': False})
                self._create_component('nth_frame_input', 'textbox', {'label': "N-th Frame Value", 'value': config.ui_defaults["nth_frame"], 'visible': False})
                self._create_component('fast_scene_input', 'checkbox', {'label': "Fast Scene Detect", 'visible': False})
                self._create_component('max_resolution', 'dropdown', {'choices': ["maximum available", "2160", "1080", "720"], 'value': config.ui_defaults['max_resolution'], 'label': "DL Res"})
                self._create_component('use_png_input', 'checkbox', {'label': "Save as PNG", 'value': config.ui_defaults['use_png']})
        
        start_btn = gr.Button("🚀 Start Extraction", variant="primary")
        stop_btn = gr.Button("⏹️ Stop", variant="stop", interactive=False)
        self.components.update({'start_extraction_button': start_btn, 'stop_extraction_button': stop_btn})

    def _create_analysis_tab(self):
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📁 Input")
                self._create_component('frames_folder_input', 'textbox', {'label': "📂 Extracted Frames Folder"})
                self._create_component('analysis_video_path_input', 'textbox', {'label': "🎥 Original Video Path (for Scene Detection)"})
                self._create_component('resume_input', 'checkbox', {'label': "💾 Skip re-analysis if config is unchanged", 'value': config.ui_defaults["resume"]})
            
            with gr.Column(scale=1):
                gr.Markdown("### 🌱 Seeding")
                with gr.Group():
                    self._create_component('enable_face_filter_input', 'checkbox', {'label': "Enable Face Similarity", 'value': config.ui_defaults['enable_face_filter'], 'interactive': True})
                    self._create_component('face_model_name_input', 'dropdown', {'choices': ["buffalo_l", "buffalo_s"], 'value': config.ui_defaults['face_model_name'], 'label': "Face Model"})
                    self._create_component('face_ref_img_path_input', 'textbox', {'label': "📸 Reference Image Path"})
                    self._create_component('face_ref_img_upload_input', 'file', {'label': "📤 Or Upload", 'type': "filepath"})
                with gr.Group():
                    self._create_component('text_prompt_input', 'textbox', {'label': "Ground with text (overrides other strategies)", 'placeholder': "e.g., 'a woman in a red dress'", 'value': config.ui_defaults['text_prompt'], 'interactive': True})
                    with gr.Row():
                        self._create_component('gdino_box_thresh_input', 'slider', {'label': "Box Thresh", 'minimum': 0.0, 'maximum': 1.0, 'step': 0.05, 'value': config.grounding_dino_params['box_threshold']})
                        self._create_component('gdino_text_thresh_input', 'slider', {'label': "Text Thresh", 'minimum': 0.0, 'maximum': 1.0, 'step': 0.05, 'value': config.grounding_dino_params['text_threshold']})
                    self._create_component('prompt_type_for_video_input', 'dropdown', {'choices': ['box', 'mask'], 'value': config.ui_defaults['prompt_type_for_video'], 'label': 'Prompt Type', 'interactive': True})
                    self._create_component('seed_strategy_input', 'dropdown', {'choices': ["Reference Face / Largest", "Largest Person", "Center-most Person"], 'value': config.ui_defaults['seed_strategy'], 'label': "Fallback Seed Strategy"})
                    self._create_component('person_detector_model_input', 'dropdown', {'choices': ['yolo11x.pt', 'yolo11s.pt'], 'value': config.ui_defaults['person_detector_model'], 'label': "Person Detector"})
                
                self._create_component('preview_seeds_button', 'button', {'value': '👀 Preview Seeding on Scenes'})
                self._create_component('seeding_preview_gallery', 'gallery', {
                    'columns': [4, 6, 8], 'rows': 2, 'height': 'auto',
                    'preview': True, 'allow_preview': True, 'object_fit': 'contain', 'visible': True
                })

            with gr.Column(scale=1):
                gr.Markdown("### 🧠 SAM2 & Metrics")
                self._create_component('enable_subject_mask_input', 'checkbox', {'label': "Enable Subject-Only Metrics", 'value': config.ui_defaults['enable_subject_mask'], 'interactive': True})
                with gr.Row():
                    self._create_component('dam4sam_model_name_input', 'dropdown', {'choices': ['sam21pp-T', 'sam21pp-S', 'sam21pp-B+', 'sam21pp-L'], 'value': config.ui_defaults['dam4sam_model_name'], 'label': "DAM4SAM Model"})
                    self._create_component('min_mask_area_pct_input', 'slider', {'label': "Min Mask %", 'minimum': 0.0, 'maximum': 10.0, 'step': 0.1, 'value': config.min_mask_area_pct})
                self._create_component('scene_detect_input', 'checkbox', {'label': "Use Scene Detection", 'value': config.ui_defaults['scene_detect'], 'interactive': True})
                self._create_component('enable_dedup_input', 'checkbox', {'label': "Enable Near-Duplicate Filtering", 'value': config.ui_defaults['enable_dedup'], 'interactive': True})
                
                with gr.Accordion("Advanced Settings", open=False):
                    self._create_component('sharpness_base_scale_input', 'number', {'label': "Sharpness Base Scale", 'value': config.sharpness_base_scale})
                    self._create_component('edge_strength_base_scale_input', 'number', {'label': "Edge Strength Base Scale", 'value': config.edge_strength_base_scale})
                    self._create_component('gdino_config_path_input', 'file', {'label': "GroundingDINO Config Override", 'type': 'filepath'})
                    self._create_component('gdino_checkpoint_path_input', 'file', {'label': "GroundingDINO Checkpoint Override", 'type': 'filepath'})
        
        start_btn = gr.Button("🔬 Start Analysis", variant="primary")
        stop_btn = gr.Button("⏹️ Stop", variant="stop", interactive=False)
        self.components.update({'start_analysis_button': start_btn, 'stop_analysis_button': stop_btn})

    def _create_filtering_tab(self):
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 🎛️ Filter Controls")
                with gr.Row():
                    self._create_component('existing_analysis_dir_input', 'textbox', {
                        'label': "📁 Existing Analysis Folder",
                        'placeholder': "Path to folder containing metadata.jsonl"
                    })
                    self._create_component('load_existing_button', 'button', {'value': "Load Analysis"})
                self._create_component('auto_pctl_input', 'slider', {'label': 'Auto-Threshold Percentile', 'minimum': 1, 'maximum': 99, 'value': 75, 'step': 1})
                with gr.Row():
                    self._create_component('apply_auto_button', 'button', {'value': 'Apply Percentile to Mins'})
                    self._create_component('reset_filters_button', 'button', {'value': "Reset Filters"})
                self._create_component('filter_status_text', 'markdown', {'value': "Load an analysis to begin."})

                self.components['metric_plots'] = {}
                self.components['metric_sliders'] = {}
                
                with gr.Accordion("Deduplication", open=True, visible=True):
                    f_def = config.filter_defaults['dedup_thresh']
                    self._create_component('dedup_thresh_input', 'slider', {'label': "Similarity Threshold", 'minimum': f_def['min'], 'maximum': f_def['max'], 'value': f_def['default'], 'step': f_def['step']})

                all_metrics = self.get_all_filter_keys()
                # Reorder to prioritize NIQE
                ordered_metrics = sorted(all_metrics, key=lambda m: m == 'niqe', reverse=True)
                
                for k in ordered_metrics:
                    if k not in config.filter_defaults: continue
                    f_def = config.filter_defaults[k]
                    accordion_label = k.replace('_', ' ').title()
                    is_open = k in config.QUALITY_METRICS
                    with gr.Accordion(accordion_label, open=is_open):
                        with gr.Column(elem_classes="plot-and-slider-column"):
                            self.components['metric_plots'][k] = self._create_component(f'plot_{k}', 'html', {'visible': False})
                            # Removed the gr.Row wrapper to stack the sliders vertically.
                            self.components['metric_sliders'][f"{k}_min"] = self._create_component(f'slider_{k}_min', 'slider', {'label': "Min", 'minimum': f_def['min'], 'maximum': f_def['max'], 'value': f_def['default_min'], 'step': f_def['step'], 'interactive': True, 'visible': False})
                            if 'default_max' in f_def:
                                self.components['metric_sliders'][f"{k}_max"] = self._create_component(f'slider_{k}_max', 'slider', {'label': "Max", 'minimum': f_def['min'], 'maximum': f_def['max'], 'value': f_def['default_max'], 'step': f_def['step'], 'interactive': True, 'visible': False})
                            if k == "face_sim":
                                self._create_component('require_face_match_input', 'checkbox', {'label': "Reject if no face", 'value': config.ui_defaults['require_face_match'], 'visible': False})

            with gr.Column(scale=2):
                gr.Markdown("### 🖼️ Results Gallery")
                with gr.Row():
                    self._create_component('gallery_view_toggle', 'radio', {'choices': ["Kept Frames", "Rejected Frames"], 'value': "Kept Frames", 'label': "Show in Gallery"})
                    self._create_component('show_mask_overlay_input', 'checkbox', {'label': "Show Mask Overlay", 'value': True})
                    self._create_component('overlay_alpha_slider', 'slider', {'label': "Overlay Alpha", 'minimum': 0.0, 'maximum': 1.0, 'value': 0.6, 'step': 0.1})
                
                self._create_component('results_gallery', 'gallery', {'columns': [4, 6, 8], 'rows': 2, 'height': 'auto', 'preview': True, 'allow_preview': True, 'object_fit': 'contain'})
                self._create_component('export_button', 'button', {'value': "📤 Export Kept Frames", 'variant': "primary"})
                with gr.Row():
                    self._create_component('enable_crop_input', 'checkbox', {'label': "✂️ Crop to Subject", 'value': True})
                    self._create_component('crop_ar_input', 'textbox', {'label': "ARs", 'value': "16:9,1:1,9:16"})
                    self._create_component('crop_padding_input', 'slider', {'label': "Padding %", 'value': 1})

    def get_all_filter_keys(self):
        return config.QUALITY_METRICS + ["face_sim", "mask_area_pct"]

    def _create_event_handlers(self):
        self.components.update({
            'extracted_video_path_state': gr.State(""), 'extracted_frames_dir_state': gr.State(""),
            'analysis_output_dir_state': gr.State(""), 'analysis_metadata_path_state': gr.State(""),
            'all_frames_data_state': gr.State([]), 'per_metric_values_state': gr.State({})
        })
        self._setup_visibility_toggles()
        self._setup_pipeline_handlers()
        self._setup_filtering_handlers()

    def run_extraction_wrapper(self, *args):
        ui_args = dict(zip(self.ext_ui_map_keys, args))
        yield from self._run_pipeline("extraction", ui_args)

    def run_analysis_wrapper(self, *args):
        ui_args = dict(zip(self.ana_ui_map_keys, args))
        yield from self._run_pipeline("analysis", ui_args)

    def _setup_visibility_toggles(self):
        c = self.components
        c['method_input'].change(
            lambda m: (gr.update(visible=m=='interval'), gr.update(visible=m=='scene'), gr.update(visible=m=='every_nth_frame')),
            c['method_input'],
            [c['interval_input'], c['fast_scene_input'], c['nth_frame_input']]
        )

    def _setup_pipeline_handlers(self):
        c = self.components
        ext_comp_map = {
            'source_path': 'source_input', 'upload_video': 'upload_video_input', 'method': 'method_input',
            'interval': 'interval_input', 'nth_frame': 'nth_frame_input', 'fast_scene': 'fast_scene_input',
            'max_resolution': 'max_resolution', 'use_png': 'use_png_input'
        }
        ext_inputs = [c[ext_comp_map[k]] for k in self.ext_ui_map_keys]
        ext_outputs = [c['start_extraction_button'], c['stop_extraction_button'], c['unified_log'], c['unified_status'], 
                       c['extracted_video_path_state'], c['extracted_frames_dir_state'], c['frames_folder_input'], c['analysis_video_path_input']]
        c['start_extraction_button'].click(
            self.run_extraction_wrapper, 
            ext_inputs, 
            ext_outputs
        )
        c['stop_extraction_button'].click(lambda: self.cancel_event.set())

        ana_comp_map = {
            'output_folder': 'frames_folder_input', 'video_path': 'analysis_video_path_input', 
            'resume': 'resume_input',
            'enable_face_filter': 'enable_face_filter_input', 'face_ref_img_path': 'face_ref_img_path_input', 
            'face_ref_img_upload': 'face_ref_img_upload_input', 'face_model_name': 'face_model_name_input',
            'enable_subject_mask': 'enable_subject_mask_input', 'dam4sam_model_name': 'dam4sam_model_name_input', 
            'person_detector_model': 'person_detector_model_input', 'seed_strategy': 'seed_strategy_input', 
            'scene_detect': 'scene_detect_input', 'enable_dedup': 'enable_dedup_input', 'text_prompt': 'text_prompt_input', 
            'prompt_type_for_video': 'prompt_type_for_video_input',
            # New UI controls
            'box_threshold': 'gdino_box_thresh_input', 'text_threshold': 'gdino_text_thresh_input',
            'min_mask_area_pct': 'min_mask_area_pct_input', 'sharpness_base_scale': 'sharpness_base_scale_input',
            'edge_strength_base_scale': 'edge_strength_base_scale_input', 'gdino_config_path': 'gdino_config_path_input',
            'gdino_checkpoint_path': 'gdino_checkpoint_path_input'
        }
        ana_inputs = [c[ana_comp_map[k]] for k in self.ana_ui_map_keys]
        ana_outputs = [c['start_analysis_button'], c['stop_analysis_button'], c['unified_log'], c['unified_status'], 
                       c['analysis_output_dir_state'], c['analysis_metadata_path_state'], c['filtering_tab']]
        c['start_analysis_button'].click(
            self.run_analysis_wrapper, 
            ana_inputs, 
            ana_outputs
        )
        c['stop_analysis_button'].click(lambda: self.cancel_event.set())
        
        preview_input_keys = [
            'enable_face_filter', 'face_ref_img_path', 'face_ref_img_upload', 'face_model_name',
            'enable_subject_mask', 'dam4sam_model_name', 'person_detector_model', 'seed_strategy',
            'scene_detect', 'text_prompt', 'prompt_type_for_video', 'box_threshold', 'text_threshold'
        ]
        preview_inputs = [c['frames_folder_input'], c['analysis_video_path_input']] + [c[ana_comp_map[k]] for k in preview_input_keys]
        
        c['preview_seeds_button'].click(
            self.preview_seeds_wrapper,
            preview_inputs,
            [c['seeding_preview_gallery'], c['unified_log']]
        )

    def _get_shared_analyzer(self, key, factory):
        with threading.Lock():
            if key not in self.shared_analyzers:
                self.shared_analyzers[key] = factory()
            return self.shared_analyzers[key]

    def _create_face_analysis_instance(self, model_name):
        logger.info(f"Loading face model: {model_name}")
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider']
            analyzer = FaceAnalysis(name=model_name, root=str(config.DIRS['models']), providers=providers)
            analyzer.prepare(ctx_id=0 if device == 'cuda' else -1, det_size=(640, 640))
            logger.success(f"Face model loaded with {'CUDA' if device == 'cuda' else 'CPU'}.")
            return analyzer
        except Exception as e:
            raise RuntimeError(f"Could not initialize face analysis model. Error: {e}") from e

    def preview_seeds_wrapper(self, frames_folder, video_path, *args):
        # Gate: need scenes and a valid video
        if not args[8] or not video_path or not Path(video_path).exists(): # args[8] is scene_detect
            return gr.update(value=[], visible=True), "[INFO] Enable scene detection and provide a valid video path to preview."

        self.cancel_event.clear()
        q = Queue()
        logger.set_progress_queue(q)
        
        # Reuse thumbnail prep logic from AnalysisPipeline
        temp_params = AnalysisParameters.from_ui(output_folder=frames_folder)
        temp_pipeline = AnalysisPipeline(temp_params, q, self.cancel_event)
        logger.info("Generating thumbnails for preview...")
        temp_pipeline._prepare_thumbnails()
        logger.info("Thumbnails ready for preview.")

        # Map args to param names
        param_names = ['enable_face_filter', 'face_ref_img_path', 'face_ref_img_upload', 'face_model_name',
                       'enable_subject_mask', 'dam4sam_model_name', 'person_detector_model', 'seed_strategy',
                       'scene_detect', 'text_prompt', 'prompt_type_for_video', 'box_threshold', 'text_threshold']
        ui_args = dict(zip(param_names, args))

        # Handle file upload
        if ui_args.get('face_ref_img_upload'):
            dest = config.DIRS['downloads'] / Path(ui_args['face_ref_img_upload']).name
            shutil.copy2(ui_args['face_ref_img_upload'], dest)
            ui_args['face_ref_img_path'] = str(dest)
        
        params = AnalysisParameters.from_ui(output_folder=frames_folder, video_path=video_path, **ui_args)
        
        # Lazily initialize and share analyzers/detectors
        person_detector = None
        if ui_args['enable_subject_mask']:
            person_detector = self._get_shared_analyzer(
                f"person_{ui_args['person_detector_model']}",
                lambda: PersonDetector(model=ui_args['person_detector_model'])
            )

        face_analyzer, ref_emb = None, None
        if ui_args['enable_face_filter'] and ui_args['face_ref_img_path']:
            face_analyzer = self._get_shared_analyzer(
                ui_args['face_model_name'],
                lambda: self._create_face_analysis_instance(ui_args['face_model_name'])
            )
            ref_img = cv2.imread(ui_args['face_ref_img_path'])
            if ref_img is not None:
                faces = face_analyzer.get(ref_img)
                if faces:
                    ref_emb = max(faces, key=lambda x: x.det_score).normed_embedding

        masker = SubjectMasker(params, q, threading.Event(), frame_map=None, face_analyzer=face_analyzer,
                               reference_embedding=ref_emb, person_detector=person_detector)
        previews = masker.preview_seeds(video_path, frames_folder)
        return gr.update(value=previews, visible=True), f"[INFO] Generated {len(previews)} previews."

    def _setup_filtering_handlers(self):
        c = self.components
        slider_keys = sorted(c['metric_sliders'].keys())
        slider_comps = [c['metric_sliders'][k] for k in slider_keys]
        fast_filter_inputs = [c['all_frames_data_state'], c['per_metric_values_state'], c['analysis_output_dir_state'],
                              c['gallery_view_toggle'], c['show_mask_overlay_input'], c['overlay_alpha_slider'],
                              c['require_face_match_input'], c['dedup_thresh_input']] + slider_comps
        fast_filter_outputs = [c['filter_status_text'], c['results_gallery']]
        
        for control in slider_comps + [c['dedup_thresh_input'], c['gallery_view_toggle'], c['show_mask_overlay_input'], c['overlay_alpha_slider'], c['require_face_match_input']]:
            handler = control.release if hasattr(control, 'release') else control.input if hasattr(control, 'input') else control.change
            handler(self.on_filters_changed, fast_filter_inputs, fast_filter_outputs)

        def load_and_trigger_update(metadata_path, output_dir, *current_filter_values):
            all_frames, metric_values = self.load_and_prep_filter_data(metadata_path)
            svgs = self.build_all_metric_svgs(metric_values)
            
            updates = {c['all_frames_data_state']: all_frames, c['per_metric_values_state']: metric_values}
            
            plot_keys = self.get_all_filter_keys()
            for k in plot_keys:
                has_data = k in metric_values and len(metric_values.get(k, [])) > 0
                updates[c['metric_plots'][k]] = gr.update(visible=has_data, value=svgs.get(k, ""))
                if f"{k}_min" in c['metric_sliders']: updates[c['metric_sliders'][f"{k}_min"]] = gr.update(visible=has_data)
                if f"{k}_max" in c['metric_sliders']: updates[c['metric_sliders'][f"{k}_max"]] = gr.update(visible=has_data)
                if k == "face_sim" and 'require_face_match_input' in c: updates[c['require_face_match_input']] = gr.update(visible=has_data)

            status, gallery = self.on_filters_changed(all_frames, metric_values, output_dir, "Kept Frames", True, 0.6, *current_filter_values)
            updates[c['filter_status_text']] = status
            updates[c['results_gallery']] = gallery
            
            # This is complex because Gradio needs a flat list of return values
            all_output_comps = [c['all_frames_data_state'], c['per_metric_values_state'], c['filter_status_text'], c['results_gallery']] + \
                               [c['metric_plots'][k] for k in plot_keys] + \
                               [c['metric_sliders'][k] for k in slider_keys] + [c['require_face_match_input']]
            return [updates.get(comp, gr.update()) for comp in all_output_comps]
        
        load_inputs = [c['analysis_metadata_path_state'], c['analysis_output_dir_state'], c['require_face_match_input'], c['dedup_thresh_input']] + slider_comps
        load_outputs = [c['all_frames_data_state'], c['per_metric_values_state'], c['filter_status_text'], c['results_gallery']] + \
                       [c['metric_plots'][k] for k in self.get_all_filter_keys()] + \
                       slider_comps + [c['require_face_match_input']]
        
        c['filtering_tab'].select(load_and_trigger_update, load_inputs, load_outputs)
        c['analysis_metadata_path_state'].change(load_and_trigger_update, load_inputs, load_outputs)
        
        def load_existing_analysis(dir_path):
            try:
                p = Path(dir_path or "")
                if not p.is_dir():
                    raise ValueError("Folder does not exist")
                md = p / "metadata.jsonl"
                if not md.exists():
                    raise ValueError("metadata.jsonl not found in folder")
                # Optional sanity: check first line header exists
                try:
                    with md.open("r") as f:
                        first = f.readline()
                        json.loads(first)
                except Exception:
                    logger.warning("metadata.jsonl header missing or unreadable; continuing")
                # Optional: check thumbs, but not strictly required for filtering
                return str(p), str(md), f"[SUCCESS] Loaded analysis from: {p}"
            except Exception as e:
                return None, None, f"[ERROR] {e}"

        self.components['load_existing_button'].click(
            load_existing_analysis,
            [c['existing_analysis_dir_input']],
            [c['analysis_output_dir_state'], c['analysis_metadata_path_state'], c['unified_log']]
        ).then(
            load_and_trigger_update,
            [c['analysis_metadata_path_state'], c['analysis_output_dir_state'], c['require_face_match_input'], c['dedup_thresh_input']] + [c['metric_sliders'][k] for k in sorted(c['metric_sliders'].keys())],
            [c['all_frames_data_state'], c['per_metric_values_state'], c['filter_status_text'], c['results_gallery']] + [c['metric_plots'][k] for k in self.get_all_filter_keys()] + [c['metric_sliders'][k] for k in sorted(c['metric_sliders'].keys())] + [c['require_face_match_input']]
        )

        export_inputs = [c['all_frames_data_state'], c['analysis_output_dir_state'], c['enable_crop_input'], 
                         c['crop_ar_input'], c['crop_padding_input'], c['require_face_match_input'], c['dedup_thresh_input']] + slider_comps
        c['export_button'].click(self.export_kept_frames, export_inputs, c['unified_log'])
        
        reset_outputs = slider_comps + [c['require_face_match_input'], c['dedup_thresh_input'], c['filter_status_text'], c['results_gallery']]
        c['reset_filters_button'].click(self.reset_filters, [c['all_frames_data_state'], c['per_metric_values_state'], c['analysis_output_dir_state']], reset_outputs)
        
        c['apply_auto_button'].click(self.auto_set_thresholds, [c['per_metric_values_state'], c['auto_pctl_input']], slider_comps).then(
            self.on_filters_changed, fast_filter_inputs, fast_filter_outputs)

    def _set_ui_state(self, buttons, state, status_msg=""):
        start_btn, stop_btn = buttons
        log_comp, status_comp = self.components['unified_log'], self.components['unified_status']
        if state == "loading":
            return {start_btn: gr.update(interactive=False), stop_btn: gr.update(interactive=True), log_comp: "", status_comp: status_msg}
        elif state == "ready":
            return {start_btn: gr.update(interactive=True), stop_btn: gr.update(interactive=False)}
        else: # success or error
            prefix = "[SUCCESS]" if state == "success" else "[ERROR]"
            return {start_btn: gr.update(interactive=True), stop_btn: gr.update(interactive=False), log_comp: f"{prefix} {status_msg}", status_comp: f"{prefix} {status_msg}"}

    def _run_pipeline(self, pipeline_type, ui_args):
        if pipeline_type == "extraction":
            buttons = (self.components['start_extraction_button'], self.components['stop_extraction_button'])
            pipeline_class = ExtractionPipeline
            param_map = {'upload_video': 'source_path'}
            required_key = 'source_path'
            success_msg = lambda res: f"Extraction complete. Output: {res['output_dir']}"
            output_updates = lambda res: {
                self.components['extracted_video_path_state']: res.get("video_path", ""),
                self.components['extracted_frames_dir_state']: res["output_dir"],
                self.components['frames_folder_input']: res["output_dir"],
                self.components['analysis_video_path_input']: res.get("video_path", "")
            }
        elif pipeline_type == "analysis":
            buttons = (self.components['start_analysis_button'], self.components['stop_analysis_button'])
            pipeline_class = AnalysisPipeline
            param_map = {'face_ref_img_upload': 'face_ref_img_path'}
            required_key = 'output_folder'
            success_msg = lambda res: f"Analysis complete. Output: {res['output_dir']}"
            output_updates = lambda res: {
                self.components['analysis_output_dir_state']: res["output_dir"],
                self.components['analysis_metadata_path_state']: res["metadata_path"],
                self.components['filtering_tab']: gr.update(interactive=True)
            }
        else:
            raise ValueError(f"Unknown pipeline type: {pipeline_type}")

        yield self._set_ui_state(buttons, "loading", f"Starting {pipeline_type}...")
        self.cancel_event.clear()

        try:
            # Handle file uploads and remap keys
            if 'upload_video' in ui_args and ui_args['upload_video']:
                source = ui_args.pop('upload_video')
                dest = str(config.DIRS['downloads'] / Path(source).name)
                shutil.copy2(source, dest)
                ui_args['source_path'] = dest
            
            if 'face_ref_img_upload' in ui_args and ui_args['face_ref_img_upload']:
                ref_upload = ui_args.pop('face_ref_img_upload')
                dest = config.DIRS['downloads'] / Path(ref_upload).name
                shutil.copy2(ref_upload, dest)
                ui_args['face_ref_img_path'] = str(dest)

            if not ui_args.get(required_key):
                raise ValueError(f"'{required_key}' is required.")

            params = AnalysisParameters.from_ui(**ui_args)
            
            q = Queue()
            logger.set_progress_queue(q)
            yield from self._run_task(pipeline_class(params, q, self.cancel_event).run)
            
            result = self.last_task_result
            if result.get("done") and not self.cancel_event.is_set():
                final_state = self._set_ui_state(buttons, "success", success_msg(result))
                final_state.update(output_updates(result))
                yield final_state
            else:
                yield self._set_ui_state(buttons, "ready")
        except Exception as e:
            logger.error(f"{pipeline_type.capitalize()} setup failed", exc_info=True)
            yield self._set_ui_state(buttons, "error", str(e))
            
    def _run_task(self, task_func):
        progress_queue = task_func.__self__.progress_queue
        log_buffer, processed, total, stage = [], 0, 1, "Initializing"
        start_time, last_yield = time.time(), 0.0

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(task_func)
            while future.running():
                if self.cancel_event.is_set(): break
                try:
                    msg = progress_queue.get(timeout=0.1)
                    if "log" in msg: log_buffer.append(msg["log"])
                    if "stage" in msg: stage, processed, start_time = msg["stage"], 0, time.time()
                    if "total" in msg: total = msg["total"] or 1
                    if "progress" in msg: processed += msg["progress"]
                    if "progress_abs" in msg: processed = msg["progress_abs"]
                    
                    if time.time() - last_yield > 0.25:
                        elapsed = time.time() - start_time
                        rate = processed / elapsed if elapsed > 0 else 0
                        eta = (total - processed) / rate if rate > 0 else 0
                        status = f"**{stage}:** {processed}/{total} ({processed/total:.1%}) | {rate:.1f} items/s | ETA: {int(eta//60):02d}:{int(eta%60):02d}"
                        yield {self.components['unified_log']: "\n".join(log_buffer), self.components['unified_status']: status}
                        last_yield = time.time()
                except Empty:
                    pass
        
        self.last_task_result = future.result() or {}
        if "log" in self.last_task_result: log_buffer.append(self.last_task_result["log"])
        if "error" in self.last_task_result: log_buffer.append(f"[ERROR] {self.last_task_result['error']}")
        
        status_text = "⏹️ Cancelled." if self.cancel_event.is_set() else f"❌ Error: {self.last_task_result.get('error')}" if 'error' in self.last_task_result else "✅ Complete."
        yield {self.components['unified_log']: "\n".join(log_buffer), self.components['unified_status']: status_text}

    def histogram_svg(self, hist_data, title=""):
        if not hist_data: return ""
        try:
            # Lazy import to reduce initial load time
            import matplotlib.pyplot as plt
            import matplotlib.ticker as mticker
            import io
            
            counts, bins = hist_data
            if not isinstance(counts, list) or not isinstance(bins, list) or len(bins) != len(counts) + 1:
                return ""

            with plt.style.context("dark_background"):
                fig, ax = plt.subplots(figsize=(4.6, 2.2), dpi=120)
                ax.bar(bins[:-1], counts, width=np.diff(bins), color="#7aa2ff", alpha=0.85, align="edge")
                ax.grid(axis="y", alpha=0.2); ax.margins(x=0)
                ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
                for side in ("top", "right"): ax.spines[side].set_visible(False)
                ax.tick_params(labelsize=8); ax.set_title(title)
                buf = io.StringIO()
                fig.savefig(buf, format="svg", bbox_inches="tight")
                plt.close(fig)
            return buf.getvalue()
        except ImportError:
            logger.warning("Matplotlib not found, cannot generate histogram SVGs.")
            return "<p style='color:red;'>Matplotlib not installed. Cannot render plot.</p>"
        except Exception as e:
            logger.error(f"Failed to generate histogram SVG.", exc_info=True)
            return ""

    def build_all_metric_svgs(self, per_metric_values):
        svgs = {}
        for k in self.get_all_filter_keys():
            if (h := per_metric_values.get(f"{k}_hist")):
                svgs[k] = self.histogram_svg(h, title="")
        return svgs

    @staticmethod
    def _apply_all_filters_vectorized(all_frames_data, filters):
        if not all_frames_data: return [], [], Counter(), {}

        num_frames = len(all_frames_data)
        filenames = [f['filename'] for f in all_frames_data]
        
        metric_arrays = {}
        for k in config.QUALITY_METRICS: metric_arrays[k] = np.array([f.get("metrics", {}).get(f"{k}_score", np.nan) for f in all_frames_data], dtype=np.float32)
        metric_arrays["face_sim"] = np.array([f.get("face_sim", np.nan) for f in all_frames_data], dtype=np.float32)
        metric_arrays["mask_area_pct"] = np.array([f.get("mask_area_pct", np.nan) for f in all_frames_data], dtype=np.float32)

        kept_mask = np.ones(num_frames, dtype=bool)
        reasons = defaultdict(list)

        for k in config.QUALITY_METRICS:
            min_val, max_val = filters.get(f"{k}_min", 0), filters.get(f"{k}_max", 100)
            low_mask, high_mask = metric_arrays[k] < min_val, metric_arrays[k] > max_val
            for i in np.where(low_mask)[0]: reasons[filenames[i]].append(f"{k}_low")
            for i in np.where(high_mask)[0]: reasons[filenames[i]].append(f"{k}_high")
            kept_mask &= ~low_mask & ~high_mask

        if filters.get("face_sim_enabled"):
            valid = ~np.isnan(metric_arrays["face_sim"])
            low_mask = valid & (metric_arrays["face_sim"] < filters.get("face_sim_min", 0.5))
            for i in np.where(low_mask)[0]: reasons[filenames[i]].append("face_sim_low")
            kept_mask &= ~low_mask
            if filters.get("require_face_match"):
                missing_mask = ~valid
                for i in np.where(missing_mask)[0]: reasons[filenames[i]].append("face_missing")
                kept_mask &= ~missing_mask
        
        if filters.get("mask_area_enabled"):
            combined_mask = metric_arrays["mask_area_pct"] < filters.get("mask_area_pct_min", 1.0)
            for i in np.where(combined_mask)[0]: reasons[filenames[i]].append("mask_too_small")
            kept_mask &= ~combined_mask

        dedup_thresh_val = filters.get("dedup_thresh", 5)
        if filters.get("enable_dedup") and dedup_thresh_val != -1:
            kept_indices = np.where(kept_mask)[0]
            if len(kept_indices) > 1:
                sorted_kept_indices = sorted(kept_indices, key=lambda i: filenames[i])
                hashes = {i: imagehash.hex_to_hash(all_frames_data[i]['phash']) for i in sorted_kept_indices if 'phash' in all_frames_data[i]}
                
                last_hash_idx = sorted_kept_indices[0]
                for i in range(1, len(sorted_kept_indices)):
                    current_idx = sorted_kept_indices[i]
                    if last_hash_idx in hashes and current_idx in hashes:
                        if (hashes[last_hash_idx] - hashes[current_idx]) <= dedup_thresh_val:
                            kept_mask[current_idx] = False
                            reasons[filenames[current_idx]].append("duplicate")
                        else:
                            last_hash_idx = current_idx
                    else: last_hash_idx = current_idx

        kept_indices, rejected_indices = np.where(kept_mask)[0], np.where(~kept_mask)[0]
        kept = [all_frames_data[i] for i in kept_indices]
        rejected = [all_frames_data[i] for i in rejected_indices]
        counts = Counter(r for r_list in reasons.values() for r in r_list)
        return kept, rejected, counts, reasons

    def load_and_prep_filter_data(self, metadata_path):
        if not metadata_path or not Path(metadata_path).exists(): return [], {}
        with Path(metadata_path).open('r') as f:
            next(f) # skip header
            all_frames = [json.loads(line) for line in f if line.strip()]

        metric_values = {}
        for k in self.get_all_filter_keys():
            is_pct = k == 'mask_area_pct'
            is_face_sim = k == 'face_sim'
            values = np.asarray([f.get("metrics" if not is_face_sim else "", {}).get(f"{k}_score" if not is_face_sim else "face_sim", f.get(k)) for f in all_frames if f.get(k) is not None or f.get("metrics", {}).get(f"{k}_score") is not None], dtype=float)
            if values.size > 0:
                hist_range = (0, 1) if is_face_sim else (0, 100)
                counts, bins = np.histogram(values, bins=50, range=hist_range)
                metric_values[k] = values.tolist()
                metric_values[f"{k}_hist"] = (counts.tolist(), bins.tolist())
        return all_frames, metric_values

    def _update_gallery(self, all_frames_data, filters, output_dir, gallery_view, show_overlay, overlay_alpha):
        kept, rejected, counts, per_frame_reasons = self._apply_all_filters_vectorized(all_frames_data, filters or {})
        
        status_parts = [f"**Kept:** {len(kept)}/{len(all_frames_data)}"]
        if counts: status_parts.append(f"**Rejections:** {', '.join([f'{k}: {v}' for k,v in counts.most_common(3)])}")
        status_text = " | ".join(status_parts)

        frames_to_show = rejected if gallery_view == "Rejected Frames" else kept
        preview_images = []
        if output_dir:
            output_path = Path(output_dir)
            thumb_dir = output_path / "thumbs"
            for f_meta in frames_to_show[:100]:
                thumb_path = thumb_dir / f"{Path(f_meta['filename']).stem}.jpg"
                if not thumb_path.exists(): continue
                caption = f"Reasons: {', '.join(per_frame_reasons.get(f_meta['filename'], []))}" if gallery_view == "Rejected Frames" else ""

                if show_overlay and not f_meta.get("mask_empty", True) and (mask_name := f_meta.get("mask_path")):
                    mask_path = output_path / "masks" / mask_name
                    thumb = cv2.imread(str(thumb_path))
                    mask_gray = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) if mask_path.exists() else None
                    if thumb is not None and mask_gray is not None:
                        thumb_overlay = render_mask_overlay(thumb, mask_gray, float(overlay_alpha))
                        preview_images.append((cv2.cvtColor(thumb_overlay, cv2.COLOR_BGR2RGB), caption))
                    else:
                        preview_images.append((str(thumb_path), caption))
                else:
                    preview_images.append((str(thumb_path), caption))
        
        gallery_update = gr.update(value=preview_images, rows=(1 if gallery_view == "Rejected Frames" else 2))
        return status_text, gallery_update

    def on_filters_changed(self, all_frames_data, per_metric_values, output_dir, gallery_view, show_overlay, overlay_alpha, require_face_match, dedup_thresh, *slider_values):
        if not all_frames_data: return "Run analysis to see results.", []
        
        slider_keys = sorted(self.components['metric_sliders'].keys())
        filters = {key: val for key, val in zip(slider_keys, slider_values)}
        filters.update({"require_face_match": require_face_match, "dedup_thresh": dedup_thresh,
                        "face_sim_enabled": bool(per_metric_values.get("face_sim")),
                        "mask_area_enabled": bool(per_metric_values.get("mask_area_pct")),
                        "enable_dedup": any('phash' in f for f in all_frames_data) if all_frames_data else False})
        
        return self._update_gallery(all_frames_data, filters, output_dir, gallery_view, show_overlay, overlay_alpha)

    def export_kept_frames(self, all_frames_data, output_dir, enable_crop, crop_ars, crop_padding, *filter_args):
        if not all_frames_data: return "No metadata to export."
        try:
            slider_keys = sorted(self.components['metric_sliders'].keys())
            require_face_match, dedup_thresh, *slider_values = filter_args
            filters = {key: val for key, val in zip(slider_keys, slider_values)}
            filters.update({"require_face_match": require_face_match, "dedup_thresh": dedup_thresh,
                            "face_sim_enabled": any("face_sim" in f for f in all_frames_data),
                            "mask_area_enabled": any("mask_area_pct" in f for f in all_frames_data),
                            "enable_dedup": any('phash' in f for f in all_frames_data) if all_frames_data else False})
            
            kept, _, _, _ = self._apply_all_filters_vectorized(all_frames_data, filters)
            n_kept, total = len(kept), len(all_frames_data)
            if total == 0: return "Exported 0/0 frames: no metadata."

            out_root = Path(output_dir)
            export_dir = out_root.parent / f"{out_root.name}_exported_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            export_dir.mkdir(exist_ok=True, parents=True)
            
            ok = 0
            for frame_meta in sorted(kept, key=lambda x: x["filename"]):
                try:
                    src_path = out_root / frame_meta["filename"]
                    if not src_path.exists(): continue
                    dst_path = export_dir / frame_meta["filename"]
                    if enable_crop and not frame_meta.get("mask_empty", True) and (mask_name := frame_meta.get("mask_path")):
                        mask_path = out_root / "masks" / mask_name
                        img = cv2.imread(str(src_path))
                        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) if mask_path.exists() else None
                        if img is not None and mask is not None:
                            cropped = self._crop_frame(img, mask, crop_ars, crop_padding)
                            cv2.imwrite(str(dst_path), cropped)
                        else: shutil.copy2(src_path, dst_path)
                    else: shutil.copy2(src_path, dst_path)
                    ok += 1
                except Exception as e:
                    logger.warning(f"Export failed for {frame_meta.get('filename', 'unknown')}: {e}")
            return f"Exported {ok}/{n_kept} kept frames to {export_dir.name}. Total frames in metadata: {total}"
        except Exception as e:
            logger.error(f"Error during export process", exc_info=True)
            return f"Error during export: {e}"

    def reset_filters(self, all_frames_data, per_metric_values, output_dir):
        # This function must return updates in the *exact* same order as reset_outputs is defined in _setup_filtering_handlers
        output_values = []
        slider_default_values = []

        # 1. Get slider updates in the correct sorted order
        slider_keys = sorted(self.components['metric_sliders'].keys())
        for key in slider_keys:
            metric_key = re.sub(r'_(min|max)$', '', key)
            default_key = 'default_max' if key.endswith('_max') else 'default_min'
            default_val = config.filter_defaults[metric_key][default_key]
            
            output_values.append(gr.update(value=default_val))
            slider_default_values.append(default_val)

        # 2. Get other filter control updates
        face_match_default = config.ui_defaults['require_face_match']
        dedup_default = config.filter_defaults['dedup_thresh']['default']

        output_values.append(gr.update(value=face_match_default))
        output_values.append(gr.update(value=dedup_default))
        
        # 3. Get gallery and status updates
        if all_frames_data:
            status_text, gallery_update = self.on_filters_changed(
                all_frames_data, per_metric_values, output_dir, "Kept Frames", True, 0.6,
                face_match_default, dedup_default, *slider_default_values
            )
            output_values.append(status_text)
            output_values.append(gallery_update)
        else:
            output_values.append("Load an analysis to begin.")
            output_values.append([])
        
        return output_values
    
    def auto_set_thresholds(self, per_metric_values, p=75):
        slider_keys = sorted(self.components['metric_sliders'].keys())
        updates = [gr.update() for _ in slider_keys]
        if not per_metric_values: return updates
        
        pmap = {k: float(np.percentile(np.asarray(vals, dtype=np.float32), p))
                for k, vals in per_metric_values.items() if not k.endswith('_hist') and vals}
        
        for i, key in enumerate(slider_keys):
            if key.endswith('_min'):
                metric = key[:-4]
                if metric in pmap:
                    updates[i] = gr.update(value=round(pmap[metric], 2))
        return updates

    def _parse_ar(self, s: str) -> tuple[int, int]:
        try:
            if isinstance(s, str) and ":" in s:
                w_str, h_str = s.split(":", 1)
                return max(int(w_str), 1), max(int(h_str), 1)
        except Exception: pass
        return 1, 1

    def _crop_frame(self, img: np.ndarray, mask: np.ndarray, crop_ars: str, padding: int) -> np.ndarray:
        h, w = img.shape[:2]
        if mask is None:
            return img
        
        # Robustly reduce to 2D
        if mask.ndim == 3:
            if mask.shape[2] == 1:
                mask = mask[:, :, 0]
            else:
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        mask = (mask > 128).astype(np.uint8)
        
        ys, xs = np.where(mask > 0)
        if ys.size == 0: return img
        x1, x2, y1, y2 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
        
        bw, bh = x2 - x1, y2 - y1
        pad_x, pad_y = int(round(bw * padding/100.0)), int(round(bh * padding/100.0))
        x1, y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
        x2, y2 = min(w, x2 + pad_x), min(h, y2 + pad_y)
        
        bw, bh = x2 - x1, y2 - y1
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        
        ars = [self._parse_ar(s.strip()) for s in str(crop_ars).split(',') if s.strip()]
        if not ars: return img[y1:y2, x1:x2]

        def expand_to_ar(r):
            if bw / (bh + 1e-9) < r: new_w, new_h = int(np.ceil(bh * r)), bh
            else: new_w, new_h = bw, int(np.ceil(bw / r))
            if new_w > w or new_h > h: return None
            
            x1n, y1n = int(round(cx - new_w / 2)), int(round(cy - new_h / 2))
            if x1n < 0: x1n = 0
            if y1n < 0: y1n = 0
            if x1n + new_w > w: x1n = w - new_w
            if y1n + new_h > h: y1n = h - new_h
            x2n, y2n = x1n + new_w, y1n + new_h
            
            if x1n > x1 or y1n > y1 or x2n < x2 or y2n < y2: return None
            return (x1n, y1n, x2n, y2n, (new_w * new_h) / max(1, bw * bh))

        candidates = []
        for ar in ars:
            r_w, r_h = (ar if isinstance(ar, (tuple, list)) and len(ar) == 2 else (1, 1))
            if r_h > 0:
                res = expand_to_ar(r_w / r_h)
                if res:
                    candidates.append(res)
                    
        if candidates:
            x1n, y1n, x2n, y2n, _ = sorted(candidates, key=lambda t: t[4])[0]
            return img[y1n:y2n, x1n:x2n]
        return img[y1:y2, x1:x2]


if __name__ == "__main__":
    check_dependencies()
    AppUI().build_ui().launch()

