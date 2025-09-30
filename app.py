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
from collections import Counter, defaultdict, OrderedDict
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
from functools import lru_cache

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
        # Start with the default formatted message
        s = super().format(record)
        # Find and append extra context
        extra_items = {k: v for k, v in record.__dict__.items() if k not in logging.LogRecord.__dict__ and k != 'args'}
        if extra_items:
            s += f" [{', '.join(f'{k}={v}' for k, v in extra_items.items())}]"
        return s


class UnifiedLogger:
    def __init__(self):
        self.progress_queue = None
        self.logger = logging.getLogger('unified_logger')
        if not self.logger.handlers:  # Prevent adding handlers multiple times on hot-reload
            self.logger.setLevel(logging.INFO)
            self.logger.propagate = False
            
            # Use a simple formatter for the console to keep it clean
            console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            ch = logging.StreamHandler()
            ch.setFormatter(console_formatter)
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
    CONFIG_FILE = DIRS['configs'] / "config.yaml"

    def __init__(self):
        self.settings = self.load_config()
        for key, value in self.settings.items():
            setattr(self, key, value)
        
        self.thumbnail_cache_size = self.settings.get('thumbnail_cache_size', 200)
        
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
        return UnifiedLogger()

# --- Global Initialization ---
try:
    config = Config()
    logger = config.setup_directories_and_logger()
except FileNotFoundError as e:
    print(f"FATAL: {e}")
    logging.basicConfig()
    logger = logging.getLogger()
    logger.error(f"FATAL: {e}")
    with gr.Blocks() as error_app:
        gr.Markdown(f"# Configuration Error\n\n**Could not start the application.**\n\nReason: `{e}`\n\nPlease create a `configs/config.yaml` file and restart.")
    error_app.launch()
    exit()

# --- Utility & Model Loading Functions ---
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

def render_mask_overlay(frame_rgb: np.ndarray, mask_gray: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    if mask_gray is None:
        return frame_rgb
    h, w = frame_rgb.shape[:2]
    if mask_gray.shape[:2] != (h, w):
        mask_gray = cv2.resize(mask_gray, (w, h), interpolation=cv2.INTER_NEAREST)
    m = (mask_gray > 128)
    
    red_layer = np.zeros_like(frame_rgb, dtype=np.uint8)
    red_layer[..., 0] = 255  # Red channel for RGB
    
    blended = cv2.addWeighted(frame_rgb, 1.0 - alpha, red_layer, alpha, 0.0)
    if m.ndim == 2: m = m[..., np.newaxis]
    elif m.ndim == 3 and m.shape[2] != 1:
        logger.warning(f"Unexpected mask shape. Skipping overlay.", extra={'shape': m.shape})
        return frame_rgb
    out = np.where(m, blended, frame_rgb)
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
    if hasattr(obj, '__dict__'): # For dataclasses
        return asdict(obj)
    return obj
    
def rgb_to_pil(image_rgb: np.ndarray) -> Image.Image:
    return Image.fromarray(image_rgb)

def create_frame_map(output_dir: Path):
    """Loads or creates a map from original frame number to sequential filename."""
    logger.info("Loading frame map...")
    frame_map_path = output_dir / "frame_map.json"
    if not frame_map_path.exists():
        thumb_files = sorted(list((output_dir / "thumbs").glob("frame_*.webp")),
                             key=lambda p: int(re.search(r'frame_(\d+)', p.name).group(1)))
        return {int(re.search(r'frame_(\d+)', f.name).group(1)): f.name for f in thumb_files}
    try:
        with open(frame_map_path, 'r') as f:
            frame_map_list = json.load(f)
        return {orig_num: f"frame_{i+1:06d}.webp" for i, orig_num in enumerate(sorted(frame_map_list))}
    except Exception as e:
        logger.error(f"Failed to parse frame_map.json. Using fallback.", exc_info=True)
        return {}

@lru_cache(maxsize=None)
def get_face_analyzer(model_name):
    logger.info(f"Loading or getting cached face model: {model_name}")
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider']
        analyzer = FaceAnalysis(name=model_name, root=str(config.DIRS['models']), providers=providers)
        analyzer.prepare(ctx_id=0 if device == 'cuda' else -1, det_size=(640, 640))
        logger.success(f"Face model loaded with {'CUDA' if device == 'cuda' else 'CPU'}.")
        return analyzer
    except Exception as e:
        raise RuntimeError(f"Could not initialize face analysis model. Error: {e}") from e

@lru_cache(maxsize=None)
def get_person_detector(model_name, device):
    logger.info(f"Loading or getting cached person detector: {model_name}")
    return PersonDetector(model=model_name, device=device)

class ThumbnailManager:
    """Manages loading and caching of thumbnails with an LRU policy."""
    def __init__(self, max_size=200):
        self.cache = OrderedDict()
        self.max_size = max_size
        logger.info(f"ThumbnailManager initialized with cache size {max_size}")

    def get(self, thumb_path: Path):
        """Retrieves a thumbnail from cache or loads it from disk as an RGB numpy array."""
        if not isinstance(thumb_path, Path):
            thumb_path = Path(thumb_path)
            
        if thumb_path in self.cache:
            self.cache.move_to_end(thumb_path)
            return self.cache[thumb_path]
        
        if not thumb_path.exists():
            return None

        try:
            with Image.open(thumb_path) as pil_thumb:
                thumb_rgb_pil = pil_thumb.convert("RGB")
                thumb_img = np.array(thumb_rgb_pil)
            
            self.cache[thumb_path] = thumb_img
            if len(self.cache) > self.max_size:
                self.cache.popitem(last=False)
            
            return thumb_img
        except Exception as e:
            logger.warning(f"Failed to load thumbnail", extra={'path': str(thumb_path), 'error': e})
            return None

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

    def detect_boxes(self, img_rgb):
        res = self.model.predict(img_rgb, imgsz=self.imgsz, conf=self.conf, classes=[0], verbose=False, device=self.device)
        boxes = []
        for r in res:
            if getattr(r, "boxes", None) is None: continue
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

    def calculate_quality_metrics(self, thumb_image_rgb: np.ndarray, mask: np.ndarray | None = None, niqe_metric=None):
        try:
            gray = cv2.cvtColor(thumb_image_rgb, cv2.COLOR_RGB2GRAY)
            active_mask = (mask > 128) if mask is not None and mask.ndim == 2 else None
            if active_mask is not None and np.sum(active_mask) < 100:
                raise ValueError("Mask too small.")
            
            lap = cv2.Laplacian(gray, cv2.CV_64F)
            masked_lap = lap[active_mask] if active_mask is not None else lap
            sharpness = np.var(masked_lap) if masked_lap.size > 0 else 0
            sharpness_scaled = sharpness / (config.sharpness_base_scale * (gray.size / 500_000))
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3); sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            edge_strength = np.mean(np.sqrt(sobelx**2 + sobely**2))
            edge_strength_scaled = edge_strength / (config.edge_strength_base_scale * (gray.size / 500_000))
            pixels = gray[active_mask] if active_mask is not None else gray
            mean_br, std_br = (np.mean(pixels), np.std(pixels)) if pixels.size > 0 else (0,0)
            brightness = mean_br / 255.0; contrast = std_br / (mean_br + 1e-7)
            
            gray_full = cv2.cvtColor(self.image_data, cv2.COLOR_RGB2GRAY)
            mask_full = cv2.resize(mask, (gray_full.shape[1], gray_full.shape[0]), interpolation=cv2.INTER_NEAREST) if mask is not None else None
            active_mask_full = (mask_full > 128).astype(np.uint8) if mask_full is not None else None
            hist = cv2.calcHist([gray_full], [0], active_mask_full, [256], [0, 256]).flatten()
            entropy = compute_entropy(hist)
            
            niqe_score = 0.0
            if niqe_metric is not None:
                try:
                    rgb_image = self.image_data
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
                "sharpness": min(sharpness_scaled, 1.0), "edge_strength": min(edge_strength_scaled, 1.0),
                "contrast": min(contrast, 2.0) / 2.0, "brightness": brightness, "entropy": entropy,
                "niqe": niqe_score / 100.0
            }
            self.metrics = FrameMetrics(**{f"{k}_score": float(v * 100) for k, v in scores_norm.items()})
            self.metrics.quality_score = float(sum(scores_norm[k] * (config.quality_weights[k] / 100.0) for k in config.QUALITY_METRICS) * 100)
        except Exception as e:
            self.error = f"Quality calc failed: {e}"
            logger.error(f"Frame quality calculation failed", exc_info=True, extra={'frame': self.frame_number})

@dataclass
class Scene:
    shot_id: int
    start_frame: int
    end_frame: int
    status: str = "pending"  # pending, included, excluded
    best_seed_frame: int | None = None
    seed_metrics: dict = field(default_factory=dict)
    seed_config: dict = field(default_factory=dict) # User overrides for this scene
    seed_result: dict = field(default_factory=dict) # Result of seeding (bbox, type, etc)
    preview_path: str | None = None # Path to the preview image for the UI gallery
    manual_status_change: bool = False

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
    
    thumbnails_only: bool = True
    thumb_megapixels: float = 0.5
    pre_analysis_enabled: bool = False
    pre_sample_nth: int = 1
    
    gdino_config_path: str = str(config.GROUNDING_DINO_CONFIG)
    gdino_checkpoint_path: str = str(config.GROUNDING_DINO_CKPT)
    box_threshold: float = config.GROUNDING_BOX_THRESHOLD
    text_threshold: float = config.GROUNDING_TEXT_THRESHOLD
    min_mask_area_pct: float = config.min_mask_area_pct
    sharpness_base_scale: float = config.sharpness_base_scale
    edge_strength_base_scale: float = config.edge_strength_base_scale

    @classmethod
    def from_ui(cls, **kwargs):
        instance = cls(**config.ui_defaults)
        for key, value in kwargs.items():
            if hasattr(instance, key):
                target_type = type(getattr(instance, key))
                try:
                    if value is not None and value != '':
                        setattr(instance, key, target_type(value))
                except (ValueError, TypeError):
                    logger.warning(f"Could not coerce UI value for '{key}' to {target_type}. Using default.", extra={'key': key, 'value': value})
        return instance

    def _get_config_hash(self, output_dir: Path) -> str:
        """
        Creates a hash of parameters and scene seeds to detect changes for resume logic.
        """
        data_to_hash = json.dumps(_to_json_safe(asdict(self)), sort_keys=True)
        scene_seeds_path = output_dir / "scene_seeds.json"
        if scene_seeds_path.exists():
            data_to_hash += scene_seeds_path.read_text()
        return hashlib.sha256(data_to_hash.encode()).hexdigest()

# --- Subject Masking Logic ---
@dataclass
class MaskingResult:
    mask_path: str | None = None; shot_id: int | None = None; seed_type: str | None = None
    seed_face_sim: float | None = None; mask_area_pct: float | None = None
    mask_empty: bool = True; error: str | None = None


class SeedSelector:
    """Handles the logic for selecting the initial seed (bounding box) for a scene."""
    def __init__(self, params, face_analyzer, reference_embedding, person_detector, tracker, gdino_model):
        self.params = params
        self.face_analyzer = face_analyzer
        self.reference_embedding = reference_embedding
        self.person_detector = person_detector
        self.tracker = tracker
        self._gdino = gdino_model
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    def _load_image_from_array(self, image_rgb: np.ndarray):
        transform = transforms.Compose([
            transforms.ToPILImage(), transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image_tensor = transform(image_rgb)
        return image_rgb, image_tensor

    def _ground_first_frame_xywh(self, frame_rgb_small: np.ndarray, text: str, box_th: float, text_th: float):
        if not self._gdino: return None, {}
        image_source, image_tensor = self._load_image_from_array(frame_rgb_small)
        h, w = image_source.shape[:2]

        with torch.no_grad(), torch.cuda.amp.autocast(enabled=self._device=='cuda'):
            boxes, confidences, labels = gdino_predict(
                model=self._gdino, image=image_tensor.to(self._device), caption=text,
                box_threshold=float(box_th), text_threshold=float(text_th),
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

    def _sam2_mask_for_bbox(self, frame_rgb_small, bbox_xywh):
        if not self.tracker or bbox_xywh is None: return None
        try:
            outputs = self.tracker.initialize(rgb_to_pil(frame_rgb_small), None, bbox=bbox_xywh)
            mask = outputs.get('pred_mask')
            return (mask * 255).astype(np.uint8) if mask is not None else None
        except Exception as e:
            logger.warning(f"DAM4SAM mask generation failed.", extra={'error': e})
            return None

    def _ground_first_frame_mask_xywh(self, frame_rgb_small: np.ndarray, text: str, box_th: float, text_th: float):
        xywh, details = self._ground_first_frame_xywh(frame_rgb_small, text, box_th, text_th)
        if xywh is None: return None, details
        mask = self._sam2_mask_for_bbox(frame_rgb_small, xywh)
        if mask is None:
            logger.warning("SAM2 mask generation failed. Falling back to box prompt.")
            return xywh, details
        ys, xs = np.where(mask > 128)
        if ys.size == 0: return xywh, details
        x1, x2, y1, y2 = xs.min(), xs.max()+1, ys.min(), ys.max()+1
        return [int(x1), int(y1), int(x2 - x1), int(y2 - y1)], {**details, "type": "text_prompt_mask"}

    def _seed_identity(self, frame_rgb, current_params=None):
        p = self.params if current_params is None else current_params
        
        prompt_text = getattr(p, "text_prompt", "")
        if isinstance(current_params, dict):
            prompt_text = current_params.get('text_prompt', prompt_text)

        if prompt_text:
            prompt_type = getattr(p, "prompt_type_for_video", "box")
            box_th = getattr(p, "box_threshold", self.params.box_threshold)
            text_th = getattr(p, "text_threshold", self.params.text_threshold)
            if isinstance(current_params, dict):
                prompt_type = current_params.get('prompt_type_for_video', prompt_type)
                box_th = current_params.get('box_threshold', box_th)
                text_th = current_params.get('text_threshold', text_th)

            if prompt_type == "mask":
                xywh, details = self._ground_first_frame_mask_xywh(frame_rgb, prompt_text, box_th, text_th)
            else:
                xywh, details = self._ground_first_frame_xywh(frame_rgb, prompt_text, box_th, text_th)
            if xywh is not None:
                logger.info(f"Text-prompt seed found", extra=details)
                return xywh, details
            else:
                logger.warning("Text-prompt grounding returned no boxes; falling back.")
        
        return self._choose_seed_bbox(frame_rgb, p)
    
    def _choose_seed_bbox(self, frame_rgb, current_params):
        frame_bgr_for_face = None
        if self.face_analyzer:
            frame_bgr_for_face = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            
        if self.face_analyzer and self.reference_embedding is not None and current_params.enable_face_filter:
            faces = self.face_analyzer.get(frame_bgr_for_face) if frame_bgr_for_face is not None else []
            if faces:
                best_face, best_dist = None, float('inf')
                for face in faces:
                    dist = 1 - np.dot(face.normed_embedding, self.reference_embedding)
                    if dist < best_dist: best_dist, best_face = dist, face
                
                if best_face and best_dist < 0.6:
                    details = {'type': 'face_match', 'seed_face_sim': 1 - best_dist}
                    face_bbox = best_face.bbox.astype(int)
                    final_bbox = self._get_body_box_for_face(frame_rgb, face_bbox, details)
                    return final_bbox, details

        logger.info("No matching face. Applying fallback seeding.", extra={'strategy': current_params.seed_strategy})
        
        if current_params.seed_strategy in ["Largest Person", "Center-most Person"] and self.person_detector:
            boxes = self.person_detector.detect_boxes(frame_rgb)
            if boxes:
                h, w = frame_rgb.shape[:2]; cx, cy = w / 2, h / 2
                strategy_map = {
                    "Largest Person": lambda b: (b[2] - b[0]) * (b[3] - b[1]),
                    "Center-most Person": lambda b: -math.hypot((b[0] + b[2]) / 2 - cx, (b[1] + b[3]) / 2 - cy)
                }
                score_func = strategy_map[current_params.seed_strategy]
                x1, y1, x2, y2, _ = sorted(boxes, key=score_func, reverse=True)[0]
                return [x1, y1, x2 - x1, y2 - y1], {'type': f'person_{current_params.seed_strategy.lower().replace(" ", "_")}'}

        if self.face_analyzer:
            faces = self.face_analyzer.get(frame_bgr_for_face)
            if faces:
                largest_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
                details = {'type': 'face_largest'}
                face_bbox = largest_face.bbox.astype(int)
                final_bbox = self._get_body_box_for_face(frame_rgb, face_bbox, details)
                return final_bbox, details

        logger.warning("No faces or persons found to seed shot. Using fallback rectangle.")
        h, w, _ = frame_rgb.shape
        return [w // 4, h // 4, w // 2, h // 2], {'type': 'fallback_rect'}

    def _get_body_box_for_face(self, frame_rgb, face_bbox, details_dict):
        x1, y1, x2, y2 = face_bbox
        person_bbox = self._pick_person_box_for_face(frame_rgb, [x1, y1, x2-x1, y2-y1])
        if person_bbox:
            details_dict['type'] = f'person_box_from_{details_dict["type"]}'
            return person_bbox
        else:
            expanded_box = self._expand_face_to_body([x1, y1, x2-x1, y2-y1], frame_rgb.shape)
            details_dict['type'] = f'expanded_box_from_{details_dict["type"]}'
            return expanded_box
        
    def _pick_person_box_for_face(self, frame_rgb, face_bbox):
        if not self.person_detector: return None
        px1, py1, pw, ph = face_bbox; fx, fy = px1 + pw / 2.0, py1 + ph / 2.0
        try:
            candidates = self.person_detector.detect_boxes(frame_rgb)
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
        if not (best_box[0] <= fx <= best_box[2] and best_box[1] <= fy <= best_box[3]) and iou(best_box) < 0.1: return None
        return [best_box[0], best_box[1], best_box[2] - best_box[0], best_box[3] - best_box[1]]

    def _expand_face_to_body(self, face_bbox, img_shape):
        H, W = img_shape[:2]; x, y, w, h = face_bbox; cx = x + w / 2
        new_w = min(W, w * 4.0); new_h = min(H, h * 7.0)
        new_x = max(0, cx - new_w / 2); new_y = max(0, y - h * 0.75)
        return [int(v) for v in [new_x, new_y, min(new_w, W-new_x), min(new_h, H-new_y)]]

class MaskPropagator:
    """Handles propagating a mask from a seed frame throughout a scene."""
    def __init__(self, params, tracker, cancel_event, progress_queue):
        self.params = params
        self.tracker = tracker
        self.cancel_event = cancel_event
        self.progress_queue = progress_queue
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def propagate(self, shot_frames_rgb, seed_idx, bbox_xywh):
        if not self.tracker or not shot_frames_rgb:
            err_msg = "Tracker not initialized" if not self.tracker else "No frames"
            shape = shot_frames_rgb[0].shape[:2] if shot_frames_rgb else (100, 100)
            return ([np.zeros(shape, np.uint8)] * len(shot_frames_rgb), [0.0] * len(shot_frames_rgb), [True] * len(shot_frames_rgb), [err_msg] * len(shot_frames_rgb))

        logger.info(f"Propagating masks", extra={'num_frames': len(shot_frames_rgb), 'seed_index': seed_idx})
        self.progress_queue.put({"stage": "Masking", "total": len(shot_frames_rgb)})
        masks = [None] * len(shot_frames_rgb)

        def _propagate_direction(start_idx, end_idx, step):
            for i in range(start_idx, end_idx, step):
                if self.cancel_event.is_set(): break
                outputs = self.tracker.track(rgb_to_pil(shot_frames_rgb[i]))
                mask = outputs.get('pred_mask')
                masks[i] = (mask * 255).astype(np.uint8) if mask is not None else np.zeros_like(shot_frames_rgb[i], dtype=np.uint8)[:, :, 0]
                self.progress_queue.put({"progress": 1})

        try:
            with torch.cuda.amp.autocast(enabled=self._device == 'cuda'):
                outputs = self.tracker.initialize(rgb_to_pil(shot_frames_rgb[seed_idx]), None, bbox=bbox_xywh)
                mask = outputs.get('pred_mask')
                masks[seed_idx] = (mask * 255).astype(np.uint8) if mask is not None else np.zeros_like(shot_frames_rgb[seed_idx], dtype=np.uint8)[:, :, 0]
                self.progress_queue.put({"progress": 1})
                _propagate_direction(seed_idx + 1, len(shot_frames_rgb), 1)
                self.tracker.initialize(rgb_to_pil(shot_frames_rgb[seed_idx]), None, bbox=bbox_xywh)
                _propagate_direction(seed_idx - 1, -1, -1)
            h, w = shot_frames_rgb[0].shape[:2]
            final_results = []
            for i, mask in enumerate(masks):
                if self.cancel_event.is_set() or mask is None: mask = np.zeros((h, w), dtype=np.uint8)
                img_area = h * w
                area_pct = (np.sum(mask > 0) / img_area) * 100 if img_area > 0 else 0.0
                is_empty = area_pct < self.params.min_mask_area_pct
                error = "Empty mask" if is_empty else None
                final_results.append((mask, float(area_pct), bool(is_empty), error))
            return tuple(zip(*final_results)) if final_results else ([], [], [], [])
        except Exception as e:
            logger.critical(f"DAM4SAM propagation failed", exc_info=True)
            h, w = shot_frames_rgb[0].shape[:2]
            error_msg = f"Propagation failed: {e}"
            return ([np.zeros((h, w), np.uint8)] * len(shot_frames_rgb), [0.0] * len(shot_frames_rgb), [True] * len(shot_frames_rgb), [error_msg] * len(shot_frames_rgb))

class SubjectMasker:
    """Orchestrates subject seeding and mask propagation for video analysis."""
    def __init__(self, params, progress_queue, cancel_event, frame_map=None, face_analyzer=None, reference_embedding=None, person_detector=None, thumbnail_manager=None, niqe_metric=None):
        self.params = params; self.progress_queue = progress_queue; self.cancel_event = cancel_event
        self.frame_map = frame_map; self.face_analyzer = face_analyzer
        self.reference_embedding = reference_embedding; self.person_detector = person_detector
        self.tracker = None; self.mask_dir = None; self.shots = []
        self._gdino = None
        self._sam2_img = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self.thumbnail_manager = thumbnail_manager if thumbnail_manager is not None else ThumbnailManager()
        self.niqe_metric = niqe_metric
        
        # Initialize sub-components
        self._initialize_models()
        self.seed_selector = SeedSelector(params, face_analyzer, reference_embedding, person_detector, self.tracker, self._gdino)
        self.mask_propagator = MaskPropagator(params, self.tracker, cancel_event, progress_queue)

    def _initialize_models(self):
        self._init_grounder()
        self._initialize_tracker()

    def _init_grounder(self):
        if self._gdino is not None: return True
        try:
            ckpt_path = Path(self.params.gdino_checkpoint_path)
            if not ckpt_path.is_absolute():
                ckpt_path = config.DIRS['models'] / ckpt_path.name
            download_model(
                "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth",
                ckpt_path, "GroundingDINO Swin-T model", min_size=500_000_000
            )
            self._gdino = gdino_load_model(
                model_config_path=self.params.gdino_config_path, model_checkpoint_path=str(ckpt_path), device=self._device,
            )
            logger.info("Grounding DINO model loaded.", extra={'model_path': str(ckpt_path)})
            return True
        except Exception as e:
            logger.warning(f"Grounding DINO unavailable.", exc_info=True)
            self._gdino = None
            return False

    def _initialize_tracker(self):
        if not all([DAM4SAMTracker, torch, torch.cuda.is_available()]):
            logger.error("DAM4SAM dependencies or CUDA not available.")
            return False
        if self.tracker: return True
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

    def run_propagation(self, frames_dir: str, scenes_to_process: list[Scene]) -> dict[str, dict]:
        self.mask_dir = Path(frames_dir) / "masks"
        self.mask_dir.mkdir(exist_ok=True)
        logger.info("Starting subject mask propagation...")
        
        if not self.tracker:
            logger.error("Tracker not initialized; skipping masking.")
            return {}
        
        self.frame_map = self.frame_map or self._create_frame_map(frames_dir)

        mask_metadata = {}
        total_scenes = len(scenes_to_process)
        for i, scene in enumerate(scenes_to_process):
            with safe_resource_cleanup():
                if self.cancel_event.is_set(): break
                self.progress_queue.put({"stage": f"Masking Scene {i+1}/{total_scenes}"})
                shot_context = {'shot_id': scene.shot_id, 'start_frame': scene.start_frame, 'end_frame': scene.end_frame}
                logger.info(f"Masking shot", extra=shot_context)
                
                seed_frame_num = scene.best_seed_frame
                shot_frames_data = self._load_shot_frames(frames_dir, scene.start_frame, scene.end_frame)
                if not shot_frames_data: continue

                frame_numbers, small_images, dims = zip(*shot_frames_data)
                
                try:
                    seed_idx_in_shot = frame_numbers.index(seed_frame_num)
                except ValueError:
                    logger.warning(f"Seed frame {seed_frame_num} not found in loaded shot frames for {scene.shot_id}, skipping.")
                    continue
                
                bbox = scene.seed_result.get('bbox')
                seed_details = scene.seed_result.get('details', {})

                if bbox is None:
                    for fn in frame_numbers:
                        if (fname := self.frame_map.get(fn)):
                            mask_metadata[fname] = asdict(MaskingResult(error="Subject not found", shot_id=scene.shot_id))
                    continue
                
                masks, areas, empties, errors = self.mask_propagator.propagate(small_images, seed_idx_in_shot, bbox)

                for i, (original_fn, _, (h, w)) in enumerate(shot_frames_data):
                    if not (frame_fname := self.frame_map.get(original_fn)): continue
                    mask_path = self.mask_dir / f"{Path(frame_fname).stem}.png"
                    result_args = {
                        "shot_id": scene.shot_id, "seed_type": seed_details.get('type'),
                        "seed_face_sim": seed_details.get('seed_face_sim'), "mask_area_pct": areas[i],
                        "mask_empty": empties[i], "error": errors[i]
                    }
                    if masks[i] is not None and np.any(masks[i]):
                        mask_full_res = cv2.resize(masks[i], (w, h), interpolation=cv2.INTER_NEAREST)
                        if mask_full_res.ndim == 3: mask_full_res = mask_full_res[:, :, 0]
                        cv2.imwrite(str(mask_path), mask_full_res)
                        mask_metadata[frame_fname] = asdict(MaskingResult(mask_path=str(mask_path), **result_args))
                    else:
                        mask_metadata[frame_fname] = asdict(MaskingResult(mask_path=None, **result_args))
        logger.success("Subject masking complete.")
        return mask_metadata

    def _create_frame_map(self, frames_dir):
        return create_frame_map(Path(frames_dir))

    def _load_shot_frames(self, frames_dir, start, end):
        frames = []
        if not self.frame_map: self.frame_map = self._create_frame_map(frames_dir)

        thumb_dir = Path(frames_dir) / "thumbs"
        for fn in sorted(fn for fn in self.frame_map if start <= fn < end):
            thumb_p = thumb_dir / f"{Path(self.frame_map[fn]).stem}.webp"
            thumb_img = self.thumbnail_manager.get(thumb_p)
            if thumb_img is None: continue
            
            h, w = thumb_img.shape[:2]
            frames.append((fn, thumb_img, (h, w)))
        return frames

    def _select_best_seed_frame_in_scene(self, scene: Scene, frames_dir: str):
        if not self.params.pre_analysis_enabled:
            scene.best_seed_frame = scene.start_frame
            scene.seed_metrics = {'reason': 'pre-analysis disabled'}
            return

        shot_frames = self._load_shot_frames(frames_dir, scene.start_frame, scene.end_frame)
        if not shot_frames:
            scene.best_seed_frame = scene.start_frame
            scene.seed_metrics = {'reason': 'no frames loaded'}
            return
        
        step = max(1, self.params.pre_sample_nth)
        candidates = shot_frames[::step]
        scores = []
        
        for frame_num, thumb_rgb, _ in candidates:
            niqe_score = 10.0
            if self.niqe_metric:
                img_tensor = torch.from_numpy(thumb_rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
                with torch.no_grad(), torch.cuda.amp.autocast(enabled=self._device=='cuda'):
                    niqe_score = float(self.niqe_metric(img_tensor.to(self.niqe_metric.device)))
            
            face_sim = 0.0
            if self.face_analyzer and self.reference_embedding is not None:
                thumb_bgr = cv2.cvtColor(thumb_rgb, cv2.COLOR_RGB2BGR)
                faces = self.face_analyzer.get(thumb_bgr)
                if faces:
                    best_face = max(faces, key=lambda x: x.det_score)
                    face_sim = 1.0 - (1 - np.dot(best_face.normed_embedding, self.reference_embedding))
            
            combined_score = (10 - niqe_score) + (face_sim * 10)
            scores.append(combined_score)
        
        best_local_idx = int(np.argmax(scores)) if scores else 0
        best_frame_num, _, _ = candidates[best_local_idx]
        scene.best_seed_frame = best_frame_num
        scene.seed_metrics = {'reason': 'pre-analysis complete', 'score': max(scores) if scores else 0, 'best_niqe': niqe_score, 'best_face_sim': face_sim}

    def get_seed_for_frame(self, frame_rgb: np.ndarray, seed_config: dict):
        """Public method to get a seed for a given frame with optional overrides."""
        return self.seed_selector._seed_identity(frame_rgb, current_params=seed_config)

    def get_mask_for_bbox(self, frame_rgb_small, bbox_xywh):
        """Public method to get a SAM mask for a bounding box."""
        return self.seed_selector._sam2_mask_for_bbox(frame_rgb_small, bbox_xywh)
        
    def draw_bbox(self, img_rgb, xywh, color=(255, 0, 0), thickness=2):
        x, y, w, h = map(int, xywh or [0, 0, 0, 0])
        img_out = img_rgb.copy()
        cv2.rectangle(img_out, (x, y), (x + w, y + h), color, thickness)
        return img_out

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
        run_log_handler = None
        try:
            logger.info("Preparing video source...")
            vid_manager = VideoManager(self.params.source_path, self.params.max_resolution)
            video_path = Path(vid_manager.prepare_video())
            
            output_dir = config.DIRS['downloads'] / video_path.stem
            output_dir.mkdir(exist_ok=True)
            
            run_log_path = output_dir / "extraction_run.log"
            run_log_handler = logging.FileHandler(run_log_path, mode='w', encoding='utf-8')
            run_log_handler.setFormatter(StructuredFormatter('%(asctime)s - %(levelname)s - %(message)s'))
            self.logger.add_handler(run_log_handler)
            logger.info(f"Video ready", extra={'path': sanitize_filename(video_path.name)})

            video_info = VideoManager.get_video_info(video_path)
            
            if self.params.scene_detect:
                self._run_scene_detection(video_path, output_dir)
            
            self._run_ffmpeg(video_path, output_dir, video_info)

            if self.cancel_event.is_set(): return {"log": "Extraction cancelled."}
            logger.success("Extraction complete.")
            return {"done": True, "output_dir": str(output_dir), "video_path": str(video_path)}
        except Exception as e:
            return logger.pipeline_error("extraction", e)
        finally:
            self.logger.remove_handler(run_log_handler)
            
    def _run_scene_detection(self, video_path, output_dir):
        logger.info("Detecting scenes...")
        try:
            scene_list = detect(str(video_path), ContentDetector())
            shots = [(s.frame_num, e.frame_num) for s, e in scene_list] if scene_list else []
            with (output_dir / "scenes.json").open('w') as f:
                json.dump(shots, f)
            logger.success(f"Found {len(shots)} scenes.")
        except Exception as e:
            logger.error("Scene detection failed.", exc_info=True)


    def _run_ffmpeg(self, video_path, output_dir, video_info):
        log_file_path = output_dir / "ffmpeg_log.txt"
        
        cmd_base = ['ffmpeg', '-y', '-i', str(video_path), '-hide_banner', '-loglevel', 'info']
        
        if self.params.thumbnails_only:
            thumb_dir = output_dir / "thumbs"
            thumb_dir.mkdir(exist_ok=True)
            
            target_area = self.params.thumb_megapixels * 1_000_000
            w, h = video_info.get('width', 1920), video_info.get('height', 1080)
            scale_factor = math.sqrt(target_area / (w * h))
            vf_scale = f"scale=w=trunc(iw*{scale_factor}/2)*2:h=trunc(ih*{scale_factor}/2)*2"
            
            fps = video_info.get('fps', 30)
            vf_filter = f"fps={fps}," + vf_scale + ",showinfo"
            cmd = cmd_base + [
                '-vf', vf_filter,
                '-c:v', 'libwebp',
                '-lossless', '0',
                '-quality', '80',
                '-vsync', 'vfr',
                str(thumb_dir / "frame_%06d.webp")
            ]
        else: # Legacy full-res extraction
            select_filter = {
                'interval': f"fps=1/{max(0.1, float(self.params.interval))}", 'keyframes': "select='eq(pict_type,I)'",
                'scene': f"select='gt(scene,{0.5 if self.params.fast_scene else 0.4})'", 'all': f"fps={video_info.get('fps', 30)}",
                'every_nth_frame': f"select='not(mod(n,{max(1, int(self.params.nth_frame))}))'"
            }.get(self.params.method)
            vf_filter = (select_filter + ",showinfo") if select_filter else "showinfo"
            cmd = cmd_base + ['-vf', vf_filter, '-vsync', 'vfr', '-f', 'image2', str(output_dir / f"frame_%06d.{'png' if self.params.use_png else 'jpg'}")]
        
        with open(log_file_path, 'w') as stderr_handle:
            process = subprocess.Popen(cmd, stderr=stderr_handle, text=True, encoding='utf-8', bufsize=1)
            self.progress_queue.put({"total": video_info.get('frame_count', 1), "stage": "Extraction"})
            
            while process.poll() is None:
                if self.cancel_event.is_set(): process.terminate(); break
                time.sleep(0.1)
            process.wait()
        
        try:
            with open(log_file_path, 'r') as f:
                log_content = f.read()
            frame_map_list = [int(m.group(1)) for m in re.finditer(r' n:\s*(\d+)', log_content)]
            with open(output_dir / "frame_map.json", 'w') as f:
                json.dump(frame_map_list, f)
        finally:
            log_file_path.unlink(missing_ok=True)
        
        if process.returncode != 0 and not self.cancel_event.is_set():
            raise RuntimeError(f"FFmpeg failed with code {process.returncode}.")

class AnalysisPipeline(Pipeline):
    def __init__(self, params: AnalysisParameters, progress_queue: Queue, cancel_event: threading.Event, thumbnail_manager=None):
        super().__init__(params, progress_queue, cancel_event)
        self.output_dir = Path(self.params.output_folder)
        self.thumb_dir = self.output_dir / "thumbs"
        self.masks_dir = self.output_dir / "masks"
        self.frame_map_path = self.output_dir / "frame_map.json"
        self.metadata_path = self.output_dir / "metadata.jsonl"
        self.write_lock = threading.Lock(); self.gpu_lock = threading.Lock()
        self.face_analyzer = None; self.reference_embedding = None; self.mask_metadata = {}
        self.niqe_metric = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.thumbnail_manager = thumbnail_manager if thumbnail_manager is not None else ThumbnailManager()
        
    def _initialize_niqe_metric(self):
        if self.niqe_metric is None:
            try:
                self.niqe_metric = pyiqa.create_metric('niqe', device=self.device)
                logger.info("NIQE metric initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize NIQE metric", extra={'error': e})

    def run_full_analysis(self, scenes_to_process: list[Scene]):
        run_log_handler = None
        try:
            run_log_path = self.output_dir / "analysis_run.log"
            run_log_handler = logging.FileHandler(run_log_path, mode='w', encoding='utf-8')
            run_log_handler.setFormatter(StructuredFormatter('%(asctime)s - %(levelname)s - %(message)s'))
            self.logger.add_handler(run_log_handler)
            
            self.metadata_path.unlink(missing_ok=True)
            with self.metadata_path.open('w') as f:
                header = {"params": asdict(self.params)}
                f.write(json.dumps(_to_json_safe(header)) + '\n')

            if self.params.enable_face_filter:
                self.face_analyzer = get_face_analyzer(self.params.face_model_name)
                if self.params.face_ref_img_path: self._process_reference_face()
            
            person_detector = get_person_detector(self.params.person_detector_model, self.device)

            masker = SubjectMasker(self.params, self.progress_queue, self.cancel_event, self._create_frame_map(),
                                   self.face_analyzer, self.reference_embedding, person_detector,
                                   thumbnail_manager=self.thumbnail_manager, niqe_metric=self.niqe_metric)
            self.mask_metadata = masker.run_propagation(str(self.output_dir), scenes_to_process)
            
            self._run_analysis_loop(scenes_to_process)
            
            if self.cancel_event.is_set(): return {"log": "Analysis cancelled."}
            logger.success("Analysis complete.", extra={'output_dir': self.output_dir})
            return {"done": True, "metadata_path": str(self.metadata_path), "output_dir": str(self.output_dir)}
        except Exception as e:
            return logger.pipeline_error("analysis", e)
        finally:
            self.logger.remove_handler(run_log_handler)
    
    def _create_frame_map(self):
        return create_frame_map(self.output_dir)

    def _process_reference_face(self):
        if not self.face_analyzer: return
        ref_path = Path(self.params.face_ref_img_path)
        if not ref_path.is_file(): raise FileNotFoundError(f"Reference face image not found: {ref_path}")
        logger.info("Processing reference face...")
        ref_img = cv2.imread(str(ref_path)) # Reads as BGR
        if ref_img is None: raise ValueError("Could not read reference image.")
        ref_faces = self.face_analyzer.get(ref_img)
        if not ref_faces: raise ValueError("No face found in reference image.")
        self.reference_embedding = max(ref_faces, key=lambda x: x.det_score).normed_embedding
        logger.success("Reference face processed.")
    
    def _run_analysis_loop(self, scenes_to_process: list[Scene]):
        frame_map = self._create_frame_map()
        all_frame_nums_to_process = {
            fn for scene in scenes_to_process for fn in range(scene.start_frame, scene.end_frame) if fn in frame_map
        }
        
        image_files_to_process = [
            self.thumb_dir / f"{Path(frame_map[fn]).stem}.webp" for fn in sorted(list(all_frame_nums_to_process))
        ]
        
        self.progress_queue.put({"total": len(image_files_to_process), "stage": "Analysis"})
        num_workers = 1 if self.params.disable_parallel else min(os.cpu_count() or 4, 8)
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            list(executor.map(self._process_single_frame, image_files_to_process))

    def _process_single_frame(self, thumb_path):
        if self.cancel_event.is_set(): return
        
        frame_num_match = re.search(r'frame_(\d+)', thumb_path.name)
        if not frame_num_match: return
        log_context = {'file': thumb_path.name}

        try:
            self._initialize_niqe_metric()
            
            thumb_image_rgb = self.thumbnail_manager.get(thumb_path)
            if thumb_image_rgb is None: raise ValueError("Could not read thumbnail.")
            
            frame = Frame(thumb_image_rgb, -1)
            
            base_filename = thumb_path.name.replace('.webp', '.png')
            mask_meta = self.mask_metadata.get(base_filename, {})
            
            mask_thumb = None
            if mask_meta.get("mask_path"):
                mask_full_path = Path(mask_meta["mask_path"])
                if not mask_full_path.is_absolute():
                     mask_full_path = self.masks_dir / mask_full_path.name
                if mask_full_path.exists():
                    mask_full = cv2.imread(str(mask_full_path), cv2.IMREAD_GRAYSCALE)
                    if mask_full is not None:
                        mask_thumb = cv2.resize(mask_full, (thumb_image_rgb.shape[1], thumb_image_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)

            frame.calculate_quality_metrics(thumb_image_rgb=thumb_image_rgb, mask=mask_thumb, niqe_metric=self.niqe_metric)

            if self.params.enable_face_filter and self.reference_embedding is not None and self.face_analyzer:
                self._analyze_face_similarity(frame, thumb_image_rgb)
            
            meta = {"filename": base_filename, "metrics": asdict(frame.metrics)}
            if frame.face_similarity_score is not None: meta["face_sim"] = frame.face_similarity_score
            if frame.max_face_confidence is not None: meta["face_conf"] = frame.max_face_confidence
            meta.update(mask_meta)

            if self.params.enable_dedup:
                pil_thumb = rgb_to_pil(thumb_image_rgb)
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
            meta = {"filename": thumb_path.name, "error": f"processing_failed: {e}"}
            with self.write_lock, self.metadata_path.open('a') as f:
                json.dump(meta, f); f.write('\n')
            self.progress_queue.put({"progress": 1})

    def _analyze_face_similarity(self, frame, image_rgb):
        try:
            # insightface expects BGR
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            with self.gpu_lock: faces = self.face_analyzer.get(image_bgr)
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
        self.thumbnail_manager = ThumbnailManager(max_size=config.thumbnail_cache_size)
        self.ext_ui_map_keys = [
            'source_path', 'upload_video', 'method', 'interval', 'nth_frame',
            'fast_scene', 'max_resolution', 'use_png', 'thumbnails_only',
            'thumb_megapixels', 'scene_detect'
        ]
        self.ana_ui_map_keys = [
            'output_folder', 'video_path', 'resume', 'enable_face_filter',
            'face_ref_img_path', 'face_ref_img_upload', 'face_model_name', 'enable_subject_mask',
            'dam4sam_model_name', 'person_detector_model', 'seed_strategy', 'scene_detect',
            'enable_dedup', 'text_prompt', 'prompt_type_for_video', 'box_threshold',
            'text_threshold', 'min_mask_area_pct', 'sharpness_base_scale',
            'edge_strength_base_scale', 'gdino_config_path', 'gdino_checkpoint_path',
            'pre_analysis_enabled', 'pre_sample_nth'
        ]

    def build_ui(self):
        css = """.plot-and-slider-column { max-width: 560px !important; margin: auto; } .scene-editor { border: 1px solid #444; padding: 10px; border-radius: 5px; }"""
        with gr.Blocks(theme=gr.themes.Default(), css=css) as demo:
            gr.Markdown("# 🎬 Frame Extractor & Analyzer v2.0")
            if not self.cuda_available: gr.Markdown("⚠️ **CPU Mode** — GPU-dependent features are disabled or will be slow.")
            
            with gr.Tabs():
                with gr.Tab("📹 1. Frame Extraction"): self._create_extraction_tab()
                with gr.Tab("🎯 2. Seeding & Scene Selection") as self.components['analysis_tab']: self._create_analysis_tab()
                with gr.Tab("📊 3. Filtering & Export") as self.components['filtering_tab']: self._create_filtering_tab()

            with gr.Row():
                with gr.Column(scale=2): self._create_component('unified_log', 'textbox', {'label': "📋 Processing Log", 'lines': 10, 'interactive': False, 'autoscroll': True})
                with gr.Column(scale=1): self._create_component('unified_status', 'textbox', {'label': "📊 Status Summary", 'lines': 2, 'interactive': False})
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
                with gr.Accordion("Thumbnail Extraction (Recommended)", open=True):
                    self._create_component('thumbnails_only_input', 'checkbox', {'label': "Extract Thumbnails Only", 'value': config.ui_defaults['thumbnails_only']})
                    self._create_component('thumb_megapixels_input', 'slider', {'label': "Thumbnail Size (MP)", 'minimum': 0.1, 'maximum': 2.0, 'step': 0.1, 'value': config.ui_defaults['thumb_megapixels']})
                    self._create_component('ext_scene_detect_input', 'checkbox', {'label': "Use Scene Detection", 'value': config.ui_defaults['scene_detect']})
                with gr.Accordion("Legacy Full-Frame Extraction", open=False):
                    method_choices = ["keyframes", "interval", "every_nth_frame", "all", "scene"]
                    self._create_component('method_input', 'dropdown', {'choices': method_choices, 'value': config.ui_defaults['method'], 'label': "Method"})
                    self._create_component('interval_input', 'textbox', {'label': "Interval (s)", 'value': config.ui_defaults['interval'], 'visible': False})
                    self._create_component('nth_frame_input', 'textbox', {'label': "N-th Frame Value", 'value': config.ui_defaults["nth_frame"], 'visible': False})
                    self._create_component('fast_scene_input', 'checkbox', {'label': "Fast Scene Detect", 'visible': False})
                    self._create_component('use_png_input', 'checkbox', {'label': "Save as PNG", 'value': config.ui_defaults['use_png']})
                self._create_component('max_resolution', 'dropdown', {'choices': ["maximum available", "2160", "1080", "720"], 'value': config.ui_defaults['max_resolution'], 'label': "DL Res"})
        
        start_btn = gr.Button("🚀 Start Extraction", variant="primary")
        self.components.update({'start_extraction_button': start_btn})

    def _create_analysis_tab(self):
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📁 Input & Pre-Analysis")
                self._create_component('frames_folder_input', 'textbox', {'label': "📂 Extracted Frames Folder"})
                self._create_component('analysis_video_path_input', 'textbox', {'label': "🎥 Original Video Path (for Export)"})
                with gr.Group():
                    self._create_component('pre_analysis_enabled_input', 'checkbox', {'label': 'Enable Pre-Analysis to find best seed frame', 'value': config.ui_defaults['pre_analysis_enabled']})
                    self._create_component('pre_sample_nth_input', 'number', {'label': 'Sample every Nth thumbnail for pre-analysis', 'value': config.ui_defaults['pre_sample_nth'], 'interactive': True})
                with gr.Accordion("Global Seeding Settings", open=True):
                    self._create_component('enable_face_filter_input', 'checkbox', {'label': "Enable Face Similarity", 'value': config.ui_defaults['enable_face_filter']})
                    self._create_component('face_model_name_input', 'dropdown', {'choices': ["buffalo_l", "buffalo_s"], 'value': config.ui_defaults['face_model_name'], 'label': "Face Model"})
                    self._create_component('face_ref_img_path_input', 'textbox', {'label': "📸 Reference Image Path"})
                    self._create_component('face_ref_img_upload_input', 'file', {'label': "📤 Or Upload", 'type': "filepath"})
                    self._create_component('text_prompt_input', 'textbox', {'label': "Ground with text", 'placeholder': "e.g., 'a woman in a red dress'", 'value': config.ui_defaults['text_prompt']})
                    self._create_component('seed_strategy_input', 'dropdown', {'choices': ["Reference Face / Largest", "Largest Person", "Center-most Person"], 'value': config.ui_defaults['seed_strategy'], 'label': "Fallback Seed Strategy"})
                    self._create_component('person_detector_model_input', 'dropdown', {'choices': ['yolo11x.pt', 'yolo11s.pt'], 'value': config.ui_defaults['person_detector_model'], 'label': "Person Detector"})
                    self._create_component('dam4sam_model_name_input', 'dropdown', {'choices': ["sam21pp-T", "sam21pp-S", "sam21pp-B+", "sam21pp-L"], 'value': config.ui_defaults['dam4sam_model_name'], 'label': "SAM Tracker Model"})
                with gr.Accordion("Advanced Analysis Settings", open=True):
                    self._create_component('enable_dedup_input', 'checkbox', {'label': "Enable Deduplication (pHash)", 'value': config.ui_defaults.get('enable_dedup', False)})

                self._create_component('start_pre_analysis_button', 'button', {'value': '🌱 Start Pre-Analysis & Seeding Preview', 'variant': 'primary'})
                self._create_component('propagate_masks_button', 'button', {'value': '🔬 Propagate Masks on Kept Scenes', 'variant': 'primary', 'interactive': False})
            
            with gr.Column(scale=2):
                gr.Markdown("### 🎭 Seeding Preview & Scene Filtering")
                self._create_component('seeding_preview_gallery', 'gallery', {'label': 'Scene Seed Previews', 'columns': [4, 6, 8], 'rows': 2, 'height': 'auto', 'preview': True, 'allow_preview': True, 'object_fit': 'contain'})
                with gr.Accordion("Scene Editor", open=False, elem_classes="scene-editor") as self.components['scene_editor_accordion']:
                    self._create_component('scene_editor_status_md', 'markdown', {'value': "Select a scene to edit."})
                    with gr.Row():
                        self._create_component('scene_editor_prompt_input', 'textbox', {'label': 'Per-Scene Text Prompt'})
                        self._create_component('scene_editor_prompt_type_input', 'dropdown', {'choices': ['box', 'mask'], 'value': 'box', 'label': 'Prompt Type'})
                    with gr.Row():
                        self._create_component('scene_editor_box_thresh_input', 'slider', {'label': "Box Thresh", 'minimum': 0.0, 'maximum': 1.0, 'step': 0.05, 'value': config.grounding_dino_params['box_threshold']})
                        self._create_component('scene_editor_text_thresh_input', 'slider', {'label': "Text Thresh", 'minimum': 0.0, 'maximum': 1.0, 'step': 0.05, 'value': config.grounding_dino_params['text_threshold']})
                    with gr.Row():
                        self._create_component('scene_recompute_button', 'button', {'value': '🔄 Recompute Preview'})
                        self._create_component('scene_include_button', 'button', {'value': '👍 Include'})
                        self._create_component('scene_exclude_button', 'button', {'value': '👎 Exclude'})
                with gr.Accordion("Bulk Scene Actions & Filters", open=True):
                    self._create_component('scene_filter_status', 'markdown', {'value': 'No scenes loaded.'})
                    self._create_component('scene_mask_area_min_input', 'slider', {'label': "Min Seed Mask Area %", 'minimum': 0.0, 'maximum': 100.0, 'value': config.min_mask_area_pct, 'step': 0.1})
                    self._create_component('scene_face_sim_min_input', 'slider', {'label': "Min Seed Face Sim", 'minimum': 0.0, 'maximum': 1.0, 'value': 0.5, 'step': 0.05})
                    self._create_component('apply_bulk_filters_button', 'button', {'value': 'Apply Bulk Filters'})
                    with gr.Row():
                        self._create_component('bulk_include_all_button', 'button', {'value': 'Include All'})
                        self._create_component('bulk_exclude_all_button', 'button', {'value': 'Exclude All'})

    def _create_filtering_tab(self):
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 🎛️ Filter Controls")
                self._create_component('load_analysis_for_filtering_button', 'button', {'value': "🔄 Load/Refresh Analysis Results"})
                self._create_component('auto_pctl_input', 'slider', {'label': 'Auto-Threshold Percentile', 'minimum': 1, 'maximum': 99, 'value': 75, 'step': 1})
                with gr.Row():
                    self._create_component('apply_auto_button', 'button', {'value': 'Apply Percentile to Mins'})
                    self._create_component('reset_filters_button', 'button', {'value': "Reset Filters"})
                self._create_component('filter_status_text', 'markdown', {'value': "Load an analysis to begin."})

                self.components['metric_plots'] = {}; self.components['metric_sliders'] = {}
                with gr.Accordion("Deduplication", open=True, visible=True):
                    f_def = config.filter_defaults['dedup_thresh']
                    self._create_component('dedup_thresh_input', 'slider', {'label': "Similarity Threshold", 'minimum': f_def['min'], 'maximum': f_def['max'], 'value': f_def['default'], 'step': f_def['step']})
                all_metrics = self.get_all_filter_keys()
                ordered_metrics = sorted(all_metrics, key=lambda m: m == 'niqe', reverse=True)
                for k in ordered_metrics:
                    if k not in config.filter_defaults: continue
                    f_def = config.filter_defaults[k]
                    with gr.Accordion(k.replace('_', ' ').title(), open=k in config.QUALITY_METRICS):
                        with gr.Column(elem_classes="plot-and-slider-column"):
                            self.components['metric_plots'][k] = self._create_component(f'plot_{k}', 'html', {'visible': False})
                            self.components['metric_sliders'][f"{k}_min"] = self._create_component(f'slider_{k}_min', 'slider', {'label': "Min", 'minimum': f_def['min'], 'maximum': f_def['max'], 'value': f_def['default_min'], 'step': f_def['step'], 'interactive': True, 'visible': False})
                            if 'default_max' in f_def: self.components['metric_sliders'][f"{k}_max"] = self._create_component(f'slider_{k}_max', 'slider', {'label': "Max", 'minimum': f_def['min'], 'maximum': f_def['max'], 'value': f_def['default_max'], 'step': f_def['step'], 'interactive': True, 'visible': False})
                            if k == "face_sim": self._create_component('require_face_match_input', 'checkbox', {'label': "Reject if no face", 'value': config.ui_defaults['require_face_match'], 'visible': False})

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
            'all_frames_data_state': gr.State([]), 'per_metric_values_state': gr.State({}),
            'scenes_state': gr.State([]), 'selected_scene_id_state': gr.State(None)
        })
        self._setup_visibility_toggles()
        self._setup_pipeline_handlers()
        self._setup_filtering_handlers()
        self._setup_scene_editor_handlers()
        self._setup_bulk_scene_handlers()

    def run_extraction_wrapper(self, *args):
        ui_args = dict(zip(self.ext_ui_map_keys, args))
        yield from self._run_pipeline("extraction", ui_args)

    def run_pre_analysis_wrapper(self, *args):
        ui_args = dict(zip(self.ana_ui_map_keys, args))
        yield from self._run_pipeline("pre_analysis", ui_args)
        
    def run_propagation_wrapper(self, output_folder, video_path, scenes, *args):
        ui_args = dict(zip(self.ana_ui_map_keys, args))
        ui_args['output_folder'] = output_folder
        ui_args['video_path'] = video_path
        yield from self._run_pipeline("propagation", ui_args, scenes=scenes)

    def _setup_visibility_toggles(self):
        c = self.components
        c['method_input'].change(
            lambda m: (gr.update(visible=m=='interval'), gr.update(visible=m=='scene'), gr.update(visible=m=='every_nth_frame')),
            c['method_input'], [c['interval_input'], c['fast_scene_input'], c['nth_frame_input']]
        )
        c['thumbnails_only_input'].change(
            lambda x: (gr.update(interactive=x), gr.update(interactive=x), gr.update(interactive=not x)),
            c['thumbnails_only_input'],
            [c['thumb_megapixels_input'], c['ext_scene_detect_input'], c['method_input']]
        )
    
    def _setup_pipeline_handlers(self):
        c = self.components
        ext_comp_map = {k: f"{k}_input" for k in self.ext_ui_map_keys}
        ext_comp_map.update({'source_path': 'source_input', 'upload_video': 'upload_video_input',
                             'max_resolution': 'max_resolution', 'scene_detect': 'ext_scene_detect_input'})
        ext_inputs = [c[ext_comp_map[k]] for k in self.ext_ui_map_keys]
        ext_outputs = [c['unified_log'], c['unified_status'], c['extracted_video_path_state'], c['extracted_frames_dir_state'],
                       c['frames_folder_input'], c['analysis_video_path_input']]
        c['start_extraction_button'].click(self.run_extraction_wrapper, ext_inputs, ext_outputs)
        
        ana_comp_map = {
            'output_folder': 'frames_folder_input', 'video_path': 'analysis_video_path_input', 'resume': gr.State(False),
            'enable_face_filter': 'enable_face_filter_input', 'face_ref_img_path': 'face_ref_img_path_input', 
            'face_ref_img_upload': 'face_ref_img_upload_input', 'face_model_name': 'face_model_name_input',
            'enable_subject_mask': gr.State(True), 'dam4sam_model_name': 'dam4sam_model_name_input', 
            'person_detector_model': 'person_detector_model_input', 'seed_strategy': 'seed_strategy_input', 
            'scene_detect': 'ext_scene_detect_input', 'enable_dedup': 'enable_dedup_input', 'text_prompt': 'text_prompt_input', 
            'prompt_type_for_video': gr.State('box'),
            'box_threshold': 'scene_editor_box_thresh_input', 'text_threshold': 'scene_editor_text_thresh_input',
            'min_mask_area_pct': gr.State(config.min_mask_area_pct),
            'sharpness_base_scale': gr.State(config.sharpness_base_scale),
            'edge_strength_base_scale': gr.State(config.edge_strength_base_scale),
            'gdino_config_path': gr.State(str(config.GROUNDING_DINO_CONFIG)),
            'gdino_checkpoint_path': gr.State(str(config.GROUNDING_DINO_CKPT)),
            'pre_analysis_enabled': 'pre_analysis_enabled_input', 'pre_sample_nth': 'pre_sample_nth_input'
        }
        self.ana_input_components = [c.get(ana_comp_map[k], ana_comp_map[k]) for k in self.ana_ui_map_keys]
        
        pre_ana_outputs = [c['unified_log'], c['unified_status'], c['seeding_preview_gallery'], c['scenes_state'],
                           c['propagate_masks_button'], c['scene_filter_status']]
        c['start_pre_analysis_button'].click(self.run_pre_analysis_wrapper, self.ana_input_components, pre_ana_outputs)
        
        prop_inputs = [c['frames_folder_input'], c['analysis_video_path_input'], c['scenes_state']] + self.ana_input_components
        prop_outputs = [c['unified_log'], c['unified_status'], c['analysis_output_dir_state'],
                        c['analysis_metadata_path_state'], c['filtering_tab']]
        c['propagate_masks_button'].click(self.run_propagation_wrapper, prop_inputs, prop_outputs)
    
    def _setup_scene_editor_handlers(self):
        c = self.components
        
        def on_select_scene(scenes, evt: gr.SelectData):
            if not scenes or evt.index is None: 
                return gr.update(open=False), None, "", "box", config.GROUNDING_BOX_THRESHOLD, config.GROUNDING_TEXT_THRESHOLD, gr.update()
            scene = scenes[evt.index]
            cfg = scene.get('seed_config', {})
            status_md = f"**Editing Scene {scene['shot_id']}** (Frames {scene['start_frame']}-{scene['end_frame']})"
            prompt = cfg.get('text_prompt', '') if cfg else ''
            
            return (gr.update(open=True, value=status_md), scene['shot_id'],
                    prompt, cfg.get('prompt_type_for_video', 'box'),
                    cfg.get('box_threshold', config.GROUNDING_BOX_THRESHOLD),
                    cfg.get('text_threshold', config.GROUNDING_TEXT_THRESHOLD))

        c['seeding_preview_gallery'].select(
            on_select_scene, [c['scenes_state']],
            [c['scene_editor_accordion'], c['selected_scene_id_state'],
             c['scene_editor_prompt_input'], c['scene_editor_prompt_type_input'],
             c['scene_editor_box_thresh_input'], c['scene_editor_text_thresh_input']]
        )
        
        recompute_inputs = [c['scenes_state'], c['selected_scene_id_state'],
                            c['scene_editor_prompt_input'], c['scene_editor_prompt_type_input'],
                            c['scene_editor_box_thresh_input'], c['scene_editor_text_thresh_input'],
                            c['frames_folder_input']] + self.ana_input_components
        c['scene_recompute_button'].click(
            self.apply_scene_overrides, 
            inputs=recompute_inputs,
            outputs=[c['seeding_preview_gallery'], c['scenes_state'], c['unified_status']]
        )
        
        include_exclude_inputs = [c['scenes_state'], c['selected_scene_id_state'], c['frames_folder_input']]
        include_exclude_outputs = [c['scenes_state'], c['scene_filter_status'], c['unified_status']]
        c['scene_include_button'].click(
            lambda s, sid, folder: self._toggle_scene_status(s, sid, 'included', folder), 
            include_exclude_inputs, include_exclude_outputs
        )
        c['scene_exclude_button'].click(
            lambda s, sid, folder: self._toggle_scene_status(s, sid, 'excluded', folder), 
            include_exclude_inputs, include_exclude_outputs
        )

    def _setup_bulk_scene_handlers(self):
        c = self.components
        
        def bulk_toggle(scenes, new_status, output_folder):
            if not scenes: return [], "No scenes to update."
            for s in scenes: 
                s['status'] = new_status
                s['manual_status_change'] = True
            self._save_scene_seeds(scenes, output_folder)
            status_text = self._get_scene_status_text(scenes)
            return scenes, status_text

        c['bulk_include_all_button'].click(
            lambda s, folder: bulk_toggle(s, 'included', folder),
            [c['scenes_state'], c['frames_folder_input']], [c['scenes_state'], c['scene_filter_status']]
        )
        c['bulk_exclude_all_button'].click(
            lambda s, folder: bulk_toggle(s, 'excluded', folder),
            [c['scenes_state'], c['frames_folder_input']], [c['scenes_state'], c['scene_filter_status']]
        )
        c['apply_bulk_filters_button'].click(
            self.apply_bulk_scene_filters,
            [c['scenes_state'], c['scene_mask_area_min_input'], c['scene_face_sim_min_input'], c['enable_face_filter_input'], c['frames_folder_input']],
            [c['scenes_state'], c['scene_filter_status']]
        )

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

        def load_and_trigger_update(metadata_path, output_dir):
            if not metadata_path or not output_dir: return [gr.update()]*len(load_outputs)
            all_frames, metric_values = self.load_and_prep_filter_data(metadata_path)
            svgs = self.build_all_metric_svgs(metric_values)
            updates = {c['all_frames_data_state']: all_frames, c['per_metric_values_state']: metric_values}
            
            for k in self.get_all_filter_keys():
                has_data = k in metric_values and len(metric_values.get(k, [])) > 0
                updates[c['metric_plots'][k]] = gr.update(visible=has_data, value=svgs.get(k, ""))
                if f"{k}_min" in c['metric_sliders']: updates[c['metric_sliders'][f"{k}_min"]] = gr.update(visible=has_data)
                if f"{k}_max" in c['metric_sliders']: updates[c['metric_sliders'][f"{k}_max"]] = gr.update(visible=has_data)
                if k == "face_sim": updates[c['require_face_match_input']] = gr.update(visible=has_data)

            default_filters = [c['metric_sliders'][k].value for k in slider_keys]
            status, gallery = self.on_filters_changed(all_frames, metric_values, output_dir, "Kept Frames", True, 0.6,
                                                      c['require_face_match_input'].value, c['dedup_thresh_input'].value,
                                                      *default_filters)
            updates[c['filter_status_text']] = status; updates[c['results_gallery']] = gallery
            return [updates.get(comp, gr.update()) for comp in load_outputs]

        load_outputs = [c['all_frames_data_state'], c['per_metric_values_state'], c['filter_status_text'], c['results_gallery']] + \
                       [c['metric_plots'][k] for k in self.get_all_filter_keys()] + slider_comps + [c['require_face_match_input']]
        c['filtering_tab'].select(load_and_trigger_update, [c['analysis_metadata_path_state'], c['analysis_output_dir_state']], load_outputs)
        c['load_analysis_for_filtering_button'].click(load_and_trigger_update, [c['analysis_metadata_path_state'], c['analysis_output_dir_state']], load_outputs)

        export_inputs = [c['all_frames_data_state'], c['analysis_output_dir_state'], c['extracted_video_path_state'],
                         c['enable_crop_input'], c['crop_ar_input'], c['crop_padding_input'],
                         c['require_face_match_input'], c['dedup_thresh_input']] + slider_comps
        c['export_button'].click(self.export_kept_frames, export_inputs, c['unified_log'])
        
        reset_outputs = slider_comps + [c['require_face_match_input'], c['dedup_thresh_input'], c['filter_status_text'], c['results_gallery']]
        c['reset_filters_button'].click(self.reset_filters, [c['all_frames_data_state'], c['per_metric_values_state'], c['analysis_output_dir_state']], reset_outputs)
        c['apply_auto_button'].click(self.auto_set_thresholds, [c['per_metric_values_state'], c['auto_pctl_input']], slider_comps).then(
            self.on_filters_changed, fast_filter_inputs, fast_filter_outputs)

    def _run_pipeline(self, pipeline_type, ui_args, scenes=None):
        self.cancel_event.clear()
        q = Queue()
        logger.set_progress_queue(q)
        
        try:
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

            params = AnalysisParameters.from_ui(**ui_args)
            
            if pipeline_type == "extraction":
                yield from self.execute_extraction(params, q)
            elif pipeline_type == "pre_analysis":
                yield from self.execute_pre_analysis(params, q)
            elif pipeline_type == "propagation":
                yield from self.execute_propagation(params, q, scenes)
        except Exception as e:
            logger.error(f"{pipeline_type} setup failed", exc_info=True)
            yield {self.components['unified_log']: str(e), self.components['unified_status']: f"[ERROR] {e}"}

    def execute_extraction(self, params, q):
        yield {self.components['unified_log']: "", self.components['unified_status']: "Starting extraction..."}
        pipeline = ExtractionPipeline(params, q, self.cancel_event)
        yield from self._run_task(pipeline.run, q)
        result = self.last_task_result
        if result.get("done"):
            yield {
                self.components['unified_log']: "Extraction complete.", self.components['unified_status']: f"Output: {result['output_dir']}",
                self.components['extracted_video_path_state']: result.get("video_path", ""),
                self.components['extracted_frames_dir_state']: result["output_dir"],
                self.components['frames_folder_input']: result["output_dir"],
                self.components['analysis_video_path_input']: result.get("video_path", "")
            }

    def _save_scene_seeds(self, scenes_list, output_dir_str):
        if not scenes_list or not output_dir_str: return
        output_dir = Path(output_dir_str)
        scene_seeds = {
            str(s['shot_id']): {
                'seed_frame_idx': s.get('best_seed_frame'),
                'seed_type': s.get('seed_result', {}).get('details', {}).get('type'),
                'seed_config': s.get('seed_config', {}),
                'status': s.get('status', 'pending'),
                'seed_metrics': s.get('seed_metrics', {})
            }
            for s in scenes_list
        }
        try:
            (output_dir / "scene_seeds.json").write_text(json.dumps(_to_json_safe(scene_seeds), indent=2))
            logger.info("Saved scene_seeds.json")
        except Exception as e:
            logger.error("Failed to save scene_seeds.json", exc_info=True)

    def execute_pre_analysis(self, params, q):
        yield {self.components['unified_log']: "", self.components['unified_status']: "Starting Pre-Analysis..."}
        
        output_dir = Path(params.output_folder)
        scenes_path = output_dir / "scenes.json"
        if not scenes_path.exists():
            yield {self.components['unified_log']: "[ERROR] scenes.json not found. Run extraction with scene detection."}
            return
        
        with scenes_path.open('r') as f: shots = json.load(f)
        scenes = [Scene(shot_id=i, start_frame=s, end_frame=e) for i, (s, e) in enumerate(shots)]
        
        scene_seeds_path = output_dir / "scene_seeds.json"
        if scene_seeds_path.exists() and params.resume:
            logger.info("Loading existing scene_seeds.json")
            with scene_seeds_path.open('r') as f: loaded_seeds = json.load(f)
            for scene in scenes:
                seed_data = loaded_seeds.get(str(scene.shot_id))
                if seed_data:
                    scene.best_seed_frame = seed_data.get('seed_frame_idx')
                    scene.seed_config = seed_data.get('seed_config', {})
                    scene.status = seed_data.get('status', 'pending')
                    scene.seed_metrics = seed_data.get('seed_metrics', {})

        niqe_metric, face_analyzer, ref_emb, person_detector = None, None, None, None
        device = "cuda" if self.cuda_available else "cpu"
        if params.pre_analysis_enabled:
            niqe_metric = pyiqa.create_metric('niqe', device=device)
        if params.enable_face_filter:
            face_analyzer = get_face_analyzer(params.face_model_name)
            if params.face_ref_img_path:
                ref_img = cv2.imread(params.face_ref_img_path)
                if ref_img is not None:
                    faces = face_analyzer.get(ref_img)
                    if faces: ref_emb = max(faces, key=lambda x: x.det_score).normed_embedding
        person_detector = get_person_detector(params.person_detector_model, device)

        masker = SubjectMasker(params, q, self.cancel_event, face_analyzer=face_analyzer,
                               reference_embedding=ref_emb, person_detector=person_detector,
                               niqe_metric=niqe_metric, thumbnail_manager=self.thumbnail_manager)
        masker.frame_map = masker._create_frame_map(str(output_dir))
        
        previews = []
        for i, scene in enumerate(scenes):
            q.put({"stage": f"Pre-analyzing scene {i+1}/{len(scenes)}", "total": len(scenes), "progress": i})
            
            if not scene.best_seed_frame:
                masker._select_best_seed_frame_in_scene(scene, str(output_dir))
            
            fname = masker.frame_map.get(scene.best_seed_frame)
            if not fname:
                logger.warning(f"Could not find best_seed_frame {scene.best_seed_frame} in frame_map for scene {scene.shot_id}")
                continue
            
            thumb_path = output_dir / "thumbs" / f"{Path(fname).stem}.webp"
            thumb_rgb = self.thumbnail_manager.get(thumb_path)
            
            if thumb_rgb is None:
                logger.warning(f"Could not load thumbnail for best_seed_frame {scene.best_seed_frame} at path {thumb_path}")
                continue

            bbox, details = masker.get_seed_for_frame(thumb_rgb, seed_config=scene.seed_config or params)
            scene.seed_result = {'bbox': bbox, 'details': details}
            
            mask = masker.get_mask_for_bbox(thumb_rgb, bbox) if bbox else None
            if mask is not None:
                h, w = mask.shape[:2]
                area_pct = (np.sum(mask > 0) / (h * w)) * 100 if (h*w) > 0 else 0.0
                scene.seed_result['details']['mask_area_pct'] = area_pct

            overlay_rgb = render_mask_overlay(thumb_rgb, mask) if mask is not None else masker.draw_bbox(thumb_rgb, bbox)
            
            caption = f"Scene {scene.shot_id} (Seed: {scene.best_seed_frame}) | {details.get('type', 'N/A')}"
            previews.append((overlay_rgb, caption))
            scene.preview_path = "dummy" 
            if scene.status == 'pending': scene.status = 'included'

        scenes_as_dict = [asdict(s) for s in scenes]
        self._save_scene_seeds(scenes_as_dict, str(output_dir))
        q.put({"stage": "Pre-analysis complete", "progress": len(scenes)})

        yield {
            self.components['unified_log']: "Pre-analysis complete.", self.components['unified_status']: f"{len(scenes)} scenes found.",
            self.components['seeding_preview_gallery']: gr.update(value=previews),
            self.components['scenes_state']: scenes_as_dict,
            self.components['propagate_masks_button']: gr.update(interactive=True),
            self.components['scene_filter_status']: self._get_scene_status_text(scenes_as_dict)
        }
        
    def execute_propagation(self, params, q, scenes_dict):
        scenes_to_process = [Scene(**s) for s in scenes_dict if s['status'] == 'included']
        if not scenes_to_process:
            yield {self.components['unified_log']: "No scenes were included for propagation.", self.components['unified_status']: "Propagation skipped."}
            return
        
        yield {self.components['unified_log']: "", self.components['unified_status']: f"Starting propagation on {len(scenes_to_process)} scenes..."}
        
        pipeline = AnalysisPipeline(params, q, self.cancel_event, thumbnail_manager=self.thumbnail_manager)
        yield from self._run_task(lambda: pipeline.run_full_analysis(scenes_to_process), q)
        
        result = self.last_task_result
        if result.get("done"):
            yield {
                self.components['unified_log']: "Propagation and analysis complete.",
                self.components['unified_status']: f"Metadata saved to {result['metadata_path']}",
                self.components['analysis_output_dir_state']: result['output_dir'],
                self.components['analysis_metadata_path_state']: result['metadata_path'],
                self.components['filtering_tab']: gr.update(interactive=True)
            }

    def _run_task(self, task_func, progress_queue):
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
                    
                    if time.time() - last_yield > 0.25:
                        elapsed = time.time() - start_time
                        rate = processed / elapsed if elapsed > 0 else 0
                        eta = (total - processed) / rate if rate > 0 else 0
                        status = f"**{stage}:** {processed}/{total} ({processed/total:.1%}) | {rate:.1f} items/s | ETA: {int(eta//60):02d}:{int(eta%60):02d}"
                        yield {self.components['unified_log']: "\n".join(log_buffer), self.components['unified_status']: status}
                        last_yield = time.time()
                except Empty: pass
        
        self.last_task_result = future.result() or {}
        if "log" in self.last_task_result: log_buffer.append(self.last_task_result["log"])
        if "error" in self.last_task_result: log_buffer.append(f"[ERROR] {self.last_task_result['error']}")
        status_text = "⏹️ Cancelled." if self.cancel_event.is_set() else f"❌ Error: {self.last_task_result.get('error')}" if 'error' in self.last_task_result else "✅ Complete."
        yield {self.components['unified_log']: "\n".join(log_buffer), self.components['unified_status']: status_text}

    def histogram_svg(self, hist_data, title=""):
        if not hist_data: return ""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.ticker as mticker
            import io
            counts, bins = hist_data
            if not isinstance(counts, list) or not isinstance(bins, list) or len(bins) != len(counts) + 1: return ""
            with plt.style.context("dark_background"):
                fig, ax = plt.subplots(figsize=(4.6, 2.2), dpi=120)
                ax.bar(bins[:-1], counts, width=np.diff(bins), color="#7aa2ff", alpha=0.85, align="edge")
                ax.grid(axis="y", alpha=0.2); ax.margins(x=0)
                ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
                for side in ("top", "right"): ax.spines[side].set_visible(False)
                ax.tick_params(labelsize=8); ax.set_title(title)
                buf = io.StringIO(); fig.savefig(buf, format="svg", bbox_inches="tight"); plt.close(fig)
            return buf.getvalue()
        except Exception as e:
            logger.error(f"Failed to generate histogram SVG.", exc_info=True)
            return ""

    def build_all_metric_svgs(self, per_metric_values):
        svgs = {}
        for k in self.get_all_filter_keys():
            if (h := per_metric_values.get(f"{k}_hist")): svgs[k] = self.histogram_svg(h, title="")
        return svgs

    @staticmethod
    def _apply_all_filters_vectorized(all_frames_data, filters):
        if not all_frames_data: return [], [], Counter(), {}
        num_frames = len(all_frames_data); filenames = [f['filename'] for f in all_frames_data]
        metric_arrays = {}
        for k in config.QUALITY_METRICS: metric_arrays[k] = np.array([f.get("metrics", {}).get(f"{k}_score", np.nan) for f in all_frames_data], dtype=np.float32)
        metric_arrays["face_sim"] = np.array([f.get("face_sim", np.nan) for f in all_frames_data], dtype=np.float32)
        metric_arrays["mask_area_pct"] = np.array([f.get("mask_area_pct", np.nan) for f in all_frames_data], dtype=np.float32)
        kept_mask = np.ones(num_frames, dtype=bool); reasons = defaultdict(list)
        
        # --- 1. Deduplication (run first on all frames) ---
        dedup_thresh_val = filters.get("dedup_thresh", 5)
        if filters.get("enable_dedup") and dedup_thresh_val != -1:
            all_indices = list(range(num_frames))
            sorted_indices = sorted(all_indices, key=lambda i: filenames[i])
            hashes = {i: imagehash.hex_to_hash(all_frames_data[i]['phash']) for i in sorted_indices if 'phash' in all_frames_data[i]}
            
            for i in range(1, len(sorted_indices)):
                current_idx = sorted_indices[i]
                prev_idx = sorted_indices[i-1]
                if prev_idx in hashes and current_idx in hashes:
                    if (hashes[prev_idx] - hashes[current_idx]) <= dedup_thresh_val:
                        kept_mask[current_idx] = False
                        reasons[filenames[current_idx]].append('duplicate')

        # --- 2. Quality & Metric Filters (run on remaining frames) ---
        for k in config.QUALITY_METRICS:
            min_val, max_val = filters.get(f"{k}_min", 0), filters.get(f"{k}_max", 100)
            # Only apply filter to frames that are still candidates
            current_kept_indices = np.where(kept_mask)[0]
            values_to_check = metric_arrays[k][current_kept_indices]
            
            low_mask_rel = values_to_check < min_val
            high_mask_rel = values_to_check > max_val
            
            low_indices_abs = current_kept_indices[low_mask_rel]
            high_indices_abs = current_kept_indices[high_mask_rel]
            
            for i in low_indices_abs: reasons[filenames[i]].append(f"{k}_low")
            for i in high_indices_abs: reasons[filenames[i]].append(f"{k}_high")
            
            kept_mask[low_indices_abs] = False
            kept_mask[high_indices_abs] = False

        if filters.get("face_sim_enabled"):
            current_kept_indices = np.where(kept_mask)[0]
            face_sim_values = metric_arrays["face_sim"][current_kept_indices]
            
            valid = ~np.isnan(face_sim_values)
            low_mask_rel = valid & (face_sim_values < filters.get("face_sim_min", 0.5))
            low_indices_abs = current_kept_indices[low_mask_rel]
            for i in low_indices_abs: reasons[filenames[i]].append("face_sim_low")
            kept_mask[low_indices_abs] = False

            if filters.get("require_face_match"):
                missing_mask_rel = ~valid
                missing_indices_abs = current_kept_indices[missing_mask_rel]
                for i in missing_indices_abs: reasons[filenames[i]].append("face_missing")
                kept_mask[missing_indices_abs] = False

        if filters.get("mask_area_enabled"):
            current_kept_indices = np.where(kept_mask)[0]
            mask_area_values = metric_arrays["mask_area_pct"][current_kept_indices]

            small_mask_rel = mask_area_values < filters.get("mask_area_pct_min", 1.0)
            small_indices_abs = current_kept_indices[small_mask_rel]
            for i in small_indices_abs: reasons[filenames[i]].append("mask_too_small")
            kept_mask[small_indices_abs] = False

        kept = [all_frames_data[i] for i in np.where(kept_mask)[0]]
        rejected = [all_frames_data[i] for i in np.where(~kept_mask)[0]]
        counts = Counter(r for r_list in reasons.values() for r in r_list)
        return kept, rejected, counts, reasons

    def load_and_prep_filter_data(self, metadata_path):
        if not metadata_path or not Path(metadata_path).exists(): return [], {}
        with Path(metadata_path).open('r') as f:
            try: next(f) # skip header
            except StopIteration: return [], {}
            all_frames = [json.loads(line) for line in f if line.strip()]

        metric_values = {}
        for k in self.get_all_filter_keys():
            is_face_sim = k == 'face_sim'
            values = np.asarray([f.get(k, f.get("metrics", {}).get(f"{k}_score")) for f in all_frames if f.get(k) is not None or f.get("metrics", {}).get(f"{k}_score") is not None], dtype=float)
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
            masks_dir = output_path / "masks"
            for f_meta in frames_to_show[:100]:
                thumb_path = thumb_dir / f"{Path(f_meta['filename']).stem}.webp"
                caption = f"Reasons: {', '.join(per_frame_reasons.get(f_meta['filename'], []))}" if gallery_view == "Rejected Frames" else ""
                
                thumb_rgb_np = self.thumbnail_manager.get(thumb_path)
                if thumb_rgb_np is None: continue

                if show_overlay and not f_meta.get("mask_empty", True) and (mask_name := f_meta.get("mask_path")):
                    mask_path = masks_dir / mask_name
                    if mask_path.exists():
                        mask_gray = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                        thumb_overlay_rgb = render_mask_overlay(thumb_rgb_np, mask_gray, float(overlay_alpha))
                        preview_images.append((thumb_overlay_rgb, caption))
                    else: preview_images.append((thumb_rgb_np, caption))
                else: preview_images.append((thumb_rgb_np, caption))

        return status_text, gr.update(value=preview_images, rows=(1 if gallery_view == "Rejected Frames" else 2))

    def on_filters_changed(self, all_frames_data, per_metric_values, output_dir, gallery_view, show_overlay, overlay_alpha, require_face_match, dedup_thresh, *slider_values):
        if not all_frames_data: return "Run analysis to see results.", []
        slider_keys = sorted(self.components['metric_sliders'].keys())
        filters = {key: val for key, val in zip(slider_keys, slider_values)}
        filters.update({"require_face_match": require_face_match, "dedup_thresh": dedup_thresh,
                        "face_sim_enabled": bool(per_metric_values.get("face_sim")),
                        "mask_area_enabled": bool(per_metric_values.get("mask_area_pct")),
                        "enable_dedup": any('phash' in f for f in all_frames_data) if all_frames_data else False})
        return self._update_gallery(all_frames_data, filters, output_dir, gallery_view, show_overlay, overlay_alpha)

    def export_kept_frames(self, all_frames_data, output_dir, video_path, enable_crop, crop_ars, crop_padding, *filter_args):
        if not all_frames_data: return "No metadata to export."
        if not video_path or not Path(video_path).exists(): return "[ERROR] Original video path is required for export."
        try:
            slider_keys = sorted(self.components['metric_sliders'].keys())
            require_face_match, dedup_thresh, *slider_values = filter_args
            filters = {key: val for key, val in zip(slider_keys, slider_values)}
            filters.update({"require_face_match": require_face_match, "dedup_thresh": dedup_thresh,
                            "face_sim_enabled": any("face_sim" in f for f in all_frames_data),
                            "mask_area_enabled": any("mask_area_pct" in f for f in all_frames_data),
                            "enable_dedup": any('phash' in f for f in all_frames_data)})
            
            kept, _, _, _ = self._apply_all_filters_vectorized(all_frames_data, filters)
            if not kept: return "No frames kept after filtering. Nothing to export."

            out_root = Path(output_dir)
            frame_map_path = out_root / "frame_map.json"
            if not frame_map_path.exists(): return "[ERROR] frame_map.json not found. Cannot export."
            with frame_map_path.open('r') as f: frame_map_list = json.load(f)
            
            fn_to_orig_map = {f"frame_{i+1:06d}.png": orig for i, orig in enumerate(sorted(frame_map_list))}

            frames_to_extract = sorted([fn_to_orig_map[f['filename']] for f in kept if f['filename'] in fn_to_orig_map])
            if not frames_to_extract: return "No valid frames found to extract."
            
            export_dir = out_root.parent / f"{out_root.name}_exported_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            export_dir.mkdir(exist_ok=True, parents=True)
            
            select_filter = f"select='in(n,{','.join(map(str, frames_to_extract))})'"
            cmd = ['ffmpeg', '-y', '-i', str(video_path), '-vf', select_filter, '-vsync', 'vfr', str(export_dir / "frame_%06d.png")]
            
            logger.info("Starting final export extraction...", extra={'command': ' '.join(cmd)})
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            
            return f"Exported {len(frames_to_extract)} frames to {export_dir.name}."
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg export failed", exc_info=True, extra={'stderr': e.stderr})
            return f"Error during export: FFmpeg failed. Check logs."
        except Exception as e:
            logger.error(f"Error during export process", exc_info=True)
            return f"Error during export: {e}"

    def reset_filters(self, all_frames_data, per_metric_values, output_dir):
        output_values = []; slider_default_values = []
        slider_keys = sorted(self.components['metric_sliders'].keys())
        for key in slider_keys:
            metric_key = re.sub(r'_(min|max)$', '', key)
            default_key = 'default_max' if key.endswith('_max') else 'default_min'
            default_val = config.filter_defaults[metric_key][default_key]
            output_values.append(gr.update(value=default_val)); slider_default_values.append(default_val)
        
        face_match_default = config.ui_defaults['require_face_match']
        dedup_default = config.filter_defaults['dedup_thresh']['default']
        output_values.append(gr.update(value=face_match_default))
        output_values.append(gr.update(value=dedup_default))
        
        if all_frames_data:
            status_text, gallery_update = self.on_filters_changed(all_frames_data, per_metric_values, output_dir, "Kept Frames", True, 0.6, face_match_default, dedup_default, *slider_default_values)
            output_values.extend([status_text, gallery_update])
        else: output_values.extend(["Load an analysis to begin.", []])
        return output_values
    
    def auto_set_thresholds(self, per_metric_values, p=75):
        slider_keys = sorted(self.components['metric_sliders'].keys())
        updates = [gr.update() for _ in slider_keys]
        if not per_metric_values: return updates
        pmap = {k: float(np.percentile(np.asarray(vals, dtype=np.float32), p)) for k, vals in per_metric_values.items() if not k.endswith('_hist') and vals}
        for i, key in enumerate(slider_keys):
            if key.endswith('_min'):
                metric = key[:-4]
                if metric in pmap: updates[i] = gr.update(value=round(pmap[metric], 2))
        return updates

    def _get_scene_status_text(self, scenes_list):
        if not scenes_list: return "No scenes loaded."
        num_included = sum(1 for s in scenes_list if s['status'] == 'included')
        return f"{num_included}/{len(scenes_list)} scenes included for propagation."

    def _toggle_scene_status(self, scenes_list, selected_shot_id, new_status, output_folder):
        if selected_shot_id is None or not scenes_list:
            return scenes_list, self._get_scene_status_text(scenes_list), "No scene selected."
        
        scene_found = False
        for s in scenes_list:
            if s['shot_id'] == selected_shot_id:
                s['status'] = new_status
                s['manual_status_change'] = True
                scene_found = True
                break
        
        if scene_found:
            self._save_scene_seeds(scenes_list, output_folder)
            return scenes_list, self._get_scene_status_text(scenes_list), f"Scene {selected_shot_id} status set to {new_status}."
        else:
            return scenes_list, self._get_scene_status_text(scenes_list), f"Could not find scene {selected_shot_id}."

    def apply_bulk_scene_filters(self, scenes, min_mask_area, min_face_sim, enable_face_filter, output_folder):
        if not scenes:
            return [], "No scenes to filter."

        for scene in scenes:
            if scene.get('manual_status_change'):
                continue
            
            is_excluded = False
            seed_result = scene.get('seed_result', {})
            details = seed_result.get('details', {})
            
            mask_area = details.get('mask_area_pct', 101)
            if mask_area < min_mask_area:
                is_excluded = True

            if enable_face_filter and not is_excluded:
                face_sim = details.get('seed_face_sim', 1.01)
                if face_sim < min_face_sim:
                    is_excluded = True
            
            scene['status'] = 'excluded' if is_excluded else 'included'

        self._save_scene_seeds(scenes, output_folder)
        return scenes, self._get_scene_status_text(scenes)

    def apply_scene_overrides(self, scenes_list, selected_shot_id, prompt, prompt_type, box_th, text_th, output_folder, *ana_args):
        if selected_shot_id is None or not scenes_list:
            return gr.update(), scenes_list, "No scene selected to apply overrides."
        
        scene_idx, scene_dict = next(((i, s) for i, s in enumerate(scenes_list) if s['shot_id'] == selected_shot_id), (None, None))
        if scene_dict is None:
            return gr.update(), scenes_list, "Error: Selected scene not found in state."

        try:
            scene_dict['seed_config'] = {
                'text_prompt': prompt, 'prompt_type_for_video': prompt_type,
                'box_threshold': box_th, 'text_threshold': text_th,
            }

            ui_args = dict(zip(self.ana_ui_map_keys, ana_args))
            ui_args['output_folder'] = output_folder
            params = AnalysisParameters.from_ui(**ui_args)
            
            face_analyzer, ref_emb, person_detector = None, None, None
            device = "cuda" if self.cuda_available else "cpu"
            if params.enable_face_filter:
                face_analyzer = get_face_analyzer(params.face_model_name)
                if params.face_ref_img_path:
                    ref_img = cv2.imread(params.face_ref_img_path)
                    if ref_img is not None:
                        faces = face_analyzer.get(ref_img)
                        if faces: ref_emb = max(faces, key=lambda x: x.det_score).normed_embedding
            person_detector = get_person_detector(params.person_detector_model, device)

            masker = SubjectMasker(params, Queue(), threading.Event(), face_analyzer=face_analyzer,
                                   reference_embedding=ref_emb, person_detector=person_detector,
                                   thumbnail_manager=self.thumbnail_manager)
            masker.frame_map = masker._create_frame_map(output_folder)
            
            fname = masker.frame_map.get(scene_dict['best_seed_frame'])
            if not fname: raise ValueError("Framemap lookup failed for re-seeding.")
            
            thumb_path = Path(output_folder) / "thumbs" / f"{Path(fname).stem}.webp"
            thumb_rgb = self.thumbnail_manager.get(thumb_path)
            
            bbox, details = masker.get_seed_for_frame(thumb_rgb, scene_dict['seed_config'])
            scene_dict['seed_result'] = {'bbox': bbox, 'details': details}

            self._save_scene_seeds(scenes_list, output_folder)
            
            updated_gallery_previews = self._regenerate_all_previews(scenes_list, output_folder, masker)
            
            return updated_gallery_previews, scenes_list, f"Scene {selected_shot_id} updated and saved."
            
        except Exception as e:
            logger.error("Failed to apply scene overrides", exc_info=True)
            return gr.update(), scenes_list, f"[ERROR] {e}"

    def _regenerate_all_previews(self, scenes_list, output_folder, masker):
        previews = []
        output_dir = Path(output_folder)
        for scene_dict in scenes_list:
            fname = masker.frame_map.get(scene_dict['best_seed_frame'])
            if not fname: continue
            
            thumb_path = output_dir / "thumbs" / f"{Path(fname).stem}.webp"
            thumb_rgb = self.thumbnail_manager.get(thumb_path)
            if thumb_rgb is None: continue

            bbox = scene_dict.get('seed_result', {}).get('bbox')
            details = scene_dict.get('seed_result', {}).get('details', {})
            
            mask = masker.get_mask_for_bbox(thumb_rgb, bbox) if bbox else None
            overlay_rgb = render_mask_overlay(thumb_rgb, mask) if mask is not None else masker.draw_bbox(thumb_rgb, bbox)
            
            caption = f"Scene {scene_dict['shot_id']} (Seed: {scene_dict['best_seed_frame']}) | {details.get('type', 'N/A')}"
            previews.append((overlay_rgb, caption))
        return previews

    def _parse_ar(self, s: str) -> tuple[int, int]:
        try:
            if isinstance(s, str) and ":" in s:
                w_str, h_str = s.split(":", 1)
                return max(int(w_str), 1), max(int(h_str), 1)
        except Exception: pass
        return 1, 1

    def _crop_frame(self, img: np.ndarray, mask: np.ndarray, crop_ars: str, padding: int) -> np.ndarray:
        h, w = img.shape[:2]
        if mask is None: return img
        if mask.ndim == 3: mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask = (mask > 128).astype(np.uint8)
        ys, xs = np.where(mask > 0)
        if ys.size == 0: return img
        x1, x2, y1, y2 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
        bw, bh = x2 - x1, y2 - y1
        pad_x, pad_y = int(round(bw * padding/100.0)), int(round(bh * padding/100.0))
        x1, y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
        x2, y2 = min(w, x2 + pad_x), min(h, y2 + pad_y)
        bw, bh = x2 - x1, y2 - y1; cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
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
                if res: candidates.append(res)
        if candidates:
            x1n, y1n, x2n, y2n, _ = sorted(candidates, key=lambda t: t[4])[0]
            return img[y1n:y2n, x1n:x2n]
        return img[y1:y2, x1:x2]

if __name__ == "__main__":
    if not shutil.which("ffmpeg"):
        raise RuntimeError("FFMPEG is not installed or not in the system's PATH.")
    AppUI().build_ui().launch()

