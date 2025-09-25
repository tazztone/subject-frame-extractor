# keep this app Monolithic.
import gradio as gr
import cv2
import numpy as np
import os
import json
import re
import shutil
import logging
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
from ultralytics import YOLO
from DAM4SAM.dam4sam_tracker import DAM4SAMTracker
from insightface.app import FaceAnalysis
from numba import njit, prange
import yaml
import plotly.graph_objects as go
try:
    import imagehash
except ImportError:
    imagehash = None

# --- Unified Logging & Configuration ---
class Config:
    BASE_DIR = Path(__file__).parent
    DIRS = {'logs': BASE_DIR / "logs", 'configs': BASE_DIR / "configs", 'models': BASE_DIR / "models", 'downloads': BASE_DIR / "downloads"}
    LOG_FILE = DIRS['logs'] / "frame_extractor.log"
    QUALITY_METRICS = ["sharpness", "edge_strength", "contrast", "brightness", "entropy"]
    QUALITY_WEIGHTS = {"sharpness": 30, "edge_strength": 20, "contrast": 20, "brightness": 10, "entropy": 20}
    NORMALIZATION_CONSTANTS = {"sharpness": 1000, "edge_strength": 100}
    QUALITY_DOWNSCALE_FACTOR = 0.25
    UI_DEFAULTS = {
        "method": "all", "interval": 5.0, "max_resolution": "maximum available", "fast_scene": False,
        "resume": False, "use_png": True, "disable_parallel": False, "enable_face_filter": True,
        "face_model_name": "buffalo_l", "quality_thresh": 12.0, "face_thresh": 0.5,
        "enable_subject_mask": True, "scene_detect": True, "dam4sam_model_name": "sam21pp-L",
        "person_detector_model": "yolo11x.pt",
        "seed_strategy": "Reference Face / Largest",
        "nth_frame": 10,
        "require_face_match": False, "enable_dedup": True, "dedup_thresh": 5,
    }
    MIN_MASK_AREA_PCT = 1.0

    @classmethod
    def setup_directories_and_logger(cls):
        for dir_path in cls.DIRS.values():
            dir_path.mkdir(exist_ok=True)
        return UnifiedLogger(log_file_path=cls.LOG_FILE)

class UnifiedLogger:
    def __init__(self, progress_queue=None, log_file_path=None):
        self.progress_queue = progress_queue
        self.logger = logging.getLogger('unified_logger')
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        self.logger.handlers.clear()
        if log_file_path:
            fh = logging.FileHandler(log_file_path)
            fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(fh)
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(ch)

    def _log(self, level, message, ui_only=False, exc_info=False):
        log_func = getattr(self.logger, level.lower(), self.logger.info)
        if not ui_only:
            log_func(message, exc_info=exc_info)
        if self.progress_queue:
            self.progress_queue.put({"log": f"[{level.upper()}] {message}"})

    def info(self, message, **kwargs): self._log('INFO', message, **kwargs)
    def warning(self, message, **kwargs): self._log('WARNING', message, **kwargs)
    def error(self, message, **kwargs): self._log('ERROR', message, **kwargs)
    def critical(self, message, **kwargs): self._log('CRITICAL', message, **kwargs)
    def success(self, message, **kwargs): self._log('SUCCESS', message, **kwargs)
    def pipeline_error(self, operation, e):
        self.error(f"{operation} failed: {e}", exc_info=True)
        return {"error": str(e)}

# --- Global Initialization ---
config = Config()
logger = config.setup_directories_and_logger()

# --- Utility Functions ---
def get_feature_status():
    cuda_available = torch.cuda.is_available()
    masking_libs_ok = all([torch, DAM4SAMTracker, Image, yaml])
    return {
        'face_analysis': True, 'youtube_dl': True,
        'scene_detection': True, 'masking': masking_libs_ok and cuda_available,
        'masking_libs_installed': masking_libs_ok, 'cuda_available': cuda_available,
        'numba_acceleration': True, 'person_detection': True,
        'perceptual_hashing': imagehash is not None,
    }

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
                logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                time.sleep(delay)
                delay *= backoff
    raise last_exception if last_exception else RuntimeError("Function failed after retries.")

def download_model(url, dest_path, description, min_size=1_000_000):
    if dest_path.is_file() and dest_path.stat().st_size >= min_size:
        return
    def download_func():
        logger.info(f"Downloading {description} from {url} to {dest_path}")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as resp, open(dest_path, "wb") as out:
            shutil.copyfileobj(resp, out)
        if not dest_path.exists() or dest_path.stat().st_size < min_size:
            raise RuntimeError(f"Downloaded {description} seems incomplete")
        logger.success(f"{description} downloaded successfully.")
    try:
        safe_execute_with_retry(download_func)
    except Exception as e:
        logger.error(f"Failed to download {description}: {e}", exc_info=True)
        raise RuntimeError(f"Failed to download required model: {description}") from e

def render_mask_overlay(frame_bgr: np.ndarray, mask_gray: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Applies a red overlay to a frame based on a mask."""
    if mask_gray is None:
        return frame_bgr
    h, w = frame_bgr.shape[:2]
    if mask_gray.shape[:2] != (h, w):
        mask_gray = cv2.resize(mask_gray, (w, h), interpolation=cv2.INTER_NEAREST)

    # Create a binary mask where the mask is active
    m = (mask_gray > 128)

    # Create a solid red layer
    red_layer = np.zeros_like(frame_bgr, dtype=np.uint8)
    red_layer[..., 2] = 255  # BGR format for red

    # Blend the original frame and the red layer
    blended = cv2.addWeighted(frame_bgr, 1.0 - alpha, red_layer, alpha, 0.0)
    
    # Prepare the boolean mask for broadcasting with a 3-channel image.
    if m.ndim == 2: m = m[..., np.newaxis]
    elif m.ndim == 3 and m.shape[2] != 1:
        logger.warning(f"Unexpected mask shape: {m.shape}. Skipping overlay.")
        return frame_bgr

    out = np.where(m, blended, frame_bgr)
    return out

@contextmanager
def safe_resource_cleanup():
    try:
        yield
    finally:
        gc.collect()
        if torch and torch.cuda.is_available():
            torch.cuda.empty_cache()

def _to_json_safe(obj):
    if isinstance(obj, dict):
        return {k: _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    # numpy scalars/arrays
    if isinstance(obj, np.generic):
        return _to_json_safe(obj.item())
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    # round floats like existing roundfloats did
    if isinstance(obj, float):
        return round(obj, 4)
    return obj

# --- Person Detector Wrapper ---
class PersonDetector:
    def __init__(self, model="yolo11x.pt", imgsz=640, conf=0.3):
        if YOLO is None:
            raise ImportError("Ultralytics YOLO not installed.")
        
        model_path = config.DIRS['models'] / model
        model_path.parent.mkdir(exist_ok=True)
        download_model(f"https://huggingface.co/Ultralytics/YOLO11/resolve/main/{model}", model_path, "YOLO person detector")
        
        self.model = YOLO(str(model_path))
        self.imgsz = imgsz
        self.conf = conf

    def detect_boxes(self, img_bgr):
        res = self.model.predict(img_bgr[..., ::-1], imgsz=self.imgsz, conf=self.conf, classes=[0], verbose=False)
        boxes = []
        for r in res:
            if getattr(r, "boxes", None) is None: continue
            for b in r.boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                score = float(b.conf[0])
                boxes.append((x1, y1, x2, y2, score))
        return boxes

# --- Numba Optimized Image Processing ---
@njit(parallel=True)
def compute_edge_strength(sobelx, sobely):
    total_mag = 0.0
    for i in prange(sobelx.shape[0]):
        for j in range(sobelx.shape[1]):
            total_mag += np.sqrt(sobelx[i, j]**2 + sobely[i, j]**2)
    return total_mag / (sobelx.size or 1)
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

@dataclass
class Frame:
    image_data: np.ndarray; frame_number: int
    metrics: FrameMetrics = field(default_factory=FrameMetrics)
    face_similarity_score: float | None = None; max_face_confidence: float | None = None
    error: str | None = None

    def calculate_quality_metrics(self, mask: np.ndarray | None = None):
        try:
            gray = cv2.cvtColor(self.image_data, cv2.COLOR_BGR2GRAY)
            active_mask = (mask > 128).astype(np.uint8) if mask is not None and mask.ndim == 2 else None
            if active_mask is not None and np.sum(active_mask) < 100:
                raise ValueError("Mask too small.")

            dsf = config.QUALITY_DOWNSCALE_FACTOR
            preview = cv2.resize(gray, (0, 0), fx=dsf, fy=dsf, interpolation=cv2.INTER_AREA)
            preview_mask = cv2.resize(active_mask, preview.shape[::-1], interpolation=cv2.INTER_NEAREST) if active_mask is not None else None
            
            lap = cv2.Laplacian(preview, cv2.CV_64F)
            masked_lap = lap[preview_mask > 0] if preview_mask is not None else lap
            sharpness = np.var(masked_lap[masked_lap != 0]) if masked_lap.size > 0 else 0
            
            sobelx = cv2.Sobel(preview, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(preview, cv2.CV_64F, 0, 1, ksize=3)
            edge_strength = compute_edge_strength(sobelx, sobely)
            
            pixels = gray[active_mask > 0] if active_mask is not None else gray
            mean_br, std_br = (np.mean(pixels), np.std(pixels)) if pixels.size > 0 else (0,0)
            brightness = mean_br / 255.0
            contrast = std_br / (mean_br + 1e-7)
            
            hist = cv2.calcHist([gray], [0], active_mask, [256], [0, 256]).flatten()
            entropy = compute_entropy(hist)
            
            scores_norm = {
                "sharpness": min(sharpness / config.NORMALIZATION_CONSTANTS["sharpness"], 1.0),
                "edge_strength": min(edge_strength / config.NORMALIZATION_CONSTANTS["edge_strength"], 1.0),
                "contrast": min(contrast, 2.0) / 2.0, "brightness": brightness, "entropy": entropy
            }
            self.metrics = FrameMetrics(**{f"{k}_score": float(v * 100) for k, v in scores_norm.items()})
            self.metrics.quality_score = float(sum(scores_norm[k] * (config.QUALITY_WEIGHTS[k] / 100.0) for k in config.QUALITY_METRICS) * 100)
        except Exception as e:
            self.error = f"Quality calc failed: {e}"
            logger.error(f"Frame {self.frame_number}: {self.error}")

@dataclass
class AnalysisParameters:
    source_path: str = ""; method: str = config.UI_DEFAULTS["method"]; interval: float = config.UI_DEFAULTS["interval"]
    max_resolution: str = config.UI_DEFAULTS["max_resolution"]; fast_scene: bool = config.UI_DEFAULTS["fast_scene"]
    use_png: bool = config.UI_DEFAULTS["use_png"]; output_folder: str = ""; video_path: str = ""
    disable_parallel: bool = config.UI_DEFAULTS["disable_parallel"]; resume: bool = config.UI_DEFAULTS["resume"]
    enable_face_filter: bool = config.UI_DEFAULTS["enable_face_filter"]; face_ref_img_path: str = ""
    face_model_name: str = config.UI_DEFAULTS["face_model_name"]
    enable_subject_mask: bool = config.UI_DEFAULTS["enable_subject_mask"]
    dam4sam_model_name: str = config.UI_DEFAULTS["dam4sam_model_name"]
    person_detector_model: str = config.UI_DEFAULTS["person_detector_model"]
    seed_strategy: str = config.UI_DEFAULTS["seed_strategy"]
    scene_detect: bool = config.UI_DEFAULTS["scene_detect"]
    nth_frame: int = config.UI_DEFAULTS["nth_frame"]
    quality_weights: dict = field(default_factory=lambda: config.QUALITY_WEIGHTS.copy())
    thresholds: dict = field(default_factory=lambda: {k: config.UI_DEFAULTS.get(f"{k}_thresh", 50.0) for k in config.QUALITY_METRICS})
    require_face_match: bool = config.UI_DEFAULTS["require_face_match"]
    enable_dedup: bool = config.UI_DEFAULTS["enable_dedup"]
    dedup_thresh: int = config.UI_DEFAULTS["dedup_thresh"]

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

    def run(self, video_path: str, frames_dir: str) -> dict[str, dict]:
        self.mask_dir = Path(frames_dir) / "masks"
        self.mask_dir.mkdir(exist_ok=True)
        logger.info("Starting subject masking...")

        if self.params.scene_detect: self._detect_scenes(video_path, frames_dir)
        
        if not self._initialize_tracker():
            logger.error("Could not initialize tracker; skipping masking.")
            return {}
        
        if not self.shots: # Fallback if scene detection failed or was disabled
            if self.frame_map:
                self.shots = [(0, max(self.frame_map.keys()) + 1 if self.frame_map else 0)]
            else: # Fallback if frame_map is also missing
                image_files = list(Path(frames_dir).glob("frame_*.*"))
                self.shots = [(0, len(image_files))]

        mask_metadata = {}
        for shot_id, (start_frame, end_frame) in enumerate(self.shots):
            with safe_resource_cleanup():
                if self.cancel_event.is_set(): break
                self.progress_queue.put({"stage": f"Masking Shot {shot_id+1}/{len(self.shots)}"})
                logger.info(f"Masking shot {shot_id+1}/{len(self.shots)} (Frames {start_frame}-{end_frame})")
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
                        # Upsample mask to original resolution before saving
                        mask_full_res = cv2.resize(masks[i], (w, h), interpolation=cv2.INTER_NEAREST)
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
            logger.info(f"Initializing DAM4SAM tracker with model '{model_name}'...")
            model_urls = {
                "sam21pp-T": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt",
                "sam21pp-S": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt",
                "sam21pp-B+": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt",
                "sam21pp-L": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
            }
            checkpoint_path = config.DIRS['models'] / Path(model_urls[model_name]).name
            checkpoint_path.parent.mkdir(exist_ok=True)
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
            logger.error(f"Failed to initialize DAM4SAM tracker: {e}", exc_info=True)
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
            logger.critical(f"Scene detection failed: {e}", exc_info=True)
            self.shots = [] # Let the main run method handle the fallback

    def _load_shot_frames(self, frames_dir, start, end, max_side=640):
        frames = []
        if not self.frame_map: return []
        for fn in sorted(fn for fn in self.frame_map if start <= fn < end):
            p = Path(frames_dir) / self.frame_map[fn]
            img = cv2.imread(str(p))
            if img is None: continue
            
            h, w = img.shape[:2]
            scale = min(1.0, max_side / max(h, w))
            
            if scale < 1.0:
                img_small = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            else:
                img_small = img
                
            frames.append((fn, img_small, (h, w)))
        return frames

    def _seed_identity(self, shot_frames):
        if not shot_frames: return None, None, None
        
        # If face filter is disabled, prefer YOLO people
        if not self.params.enable_face_filter and self.person_detector:
            logger.info("Face filter off. Seeding with largest/central person box.")
            boxes = self.person_detector.detect_boxes(shot_frames[0])
            if boxes:
                h, w = shot_frames[0].shape[:2]
                cx, cy = w/2, h/2
                
                strategy_map = {
                    "Largest Person": lambda b: (b[2]-b[0])*(b[3]-b[1]),
                    "Center-most Person": lambda b: -math.hypot( (b[0]+b[2])/2 - cx, (b[1]+b[3])/2 - cy )
                }
                score_func = strategy_map.get(self.params.seed_strategy, strategy_map["Largest Person"])
                
                x1,y1,x2,y2,_ = sorted(boxes, key=score_func, reverse=True)[0]
                return 0, [x1, y1, x2-x1, y2-y1], {'type': 'person_auto'}

        # Existing reference-face path (only if enabled and embedding present)
        seed_details, matched_face, seed_idx = {}, None, -1
        if self.face_analyzer and self.reference_embedding is not None and self.params.enable_face_filter:
            logger.info("Searching for reference face...")
            min_dist = float('inf')
            for i, frame in enumerate(shot_frames[:5]):
                faces = self.face_analyzer.get(frame) if frame is not None else []
                for face in faces:
                    dist = 1 - np.dot(face.normed_embedding, self.reference_embedding)
                    if dist < min_dist:
                        min_dist, matched_face, seed_idx = dist, face, i
            if matched_face and min_dist < 0.6:
                logger.info(f"Found reference face in frame {seed_idx} (dist: {min_dist:.2f}).")
                seed_details = {'type': 'face_match', 'seed_face_sim': 1 - min_dist}
            else:
                matched_face = None

        if not matched_face:
            logger.info("No face match. Seeding with largest face in first frame.")
            seed_idx = 0
            faces = self.face_analyzer.get(shot_frames[0]) if self.face_analyzer else []
            if not faces:
                logger.warning("No faces found to seed shot.")
                h, w, _ = shot_frames[0].shape
                return 0, [w//4, h//4, w//2, h//2], {'type': 'fallback_rect'}
            matched_face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
            seed_details = {'type': 'face_largest'}

        x1, y1, x2, y2 = matched_face.bbox.astype(int)
        face_bbox = [x1, y1, x2 - x1, y2 - y1]
        
        person_bbox = self._pick_person_box_for_face(shot_frames[seed_idx], face_bbox)
        if person_bbox:
            final_bbox = person_bbox
            logger.info(f"Found person box {final_bbox}. Seeding.")
        else:
            final_bbox = self._expand_face_to_body(face_bbox, shot_frames[seed_idx].shape)
            logger.info(f"No person box. Using heuristic expansion {final_bbox}.")
        seed_details['type'] = ('person_box_from_' if person_bbox else 'expanded_box_from_') + seed_details['type']
        return seed_idx, final_bbox, seed_details
        
    def _pick_person_box_for_face(self, frame_img, face_bbox):
        if not self.person_detector: return None
        px1, py1, pw, ph = face_bbox
        fx, fy = px1 + pw / 2.0, py1 + ph / 2.0
        try:
            candidates = self.person_detector.detect_boxes(frame_img)
        except Exception as e:
            logger.warning(f"Person detector failed on frame: {e}")
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
        if not (best_box[0]<=fx<=best_box[2] and best_box[1]<=fy<=best_box[3]) and iou(best_box) < 0.1:
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
            return ([np.zeros(shape, np.uint8)]*len(shot_frames), [0.0]*len(shot_frames), [True]*len(shot_frames), [err_msg]*len(shot_frames))
        
        logger.info(f"Propagating masks for {len(shot_frames)} frames from seed {seed_idx}...")
        self.progress_queue.put({"stage": "Masking", "total": len(shot_frames)})
        masks = [None] * len(shot_frames)
        
        try:
            seed_frame_pil = Image.fromarray(cv2.cvtColor(shot_frames[seed_idx], cv2.COLOR_BGR2RGB))
            outputs = self.tracker.initialize(seed_frame_pil, None, bbox=bbox_xywh)
            mask = (outputs.get('pred_mask') * 255).astype(np.uint8) if outputs.get('pred_mask') is not None else np.zeros_like(shot_frames[seed_idx], dtype=np.uint8)[:,:,0]
            masks[seed_idx] = mask
            self.progress_queue.put({"progress": 1})

            # Forward pass
            for i in range(seed_idx + 1, len(shot_frames)):
                if self.cancel_event.is_set(): break
                frame_pil = Image.fromarray(cv2.cvtColor(shot_frames[i], cv2.COLOR_BGR2RGB))
                outputs = self.tracker.track(frame_pil)
                masks[i] = (outputs.get('pred_mask') * 255).astype(np.uint8) if outputs.get('pred_mask') is not None else np.zeros_like(shot_frames[i], dtype=np.uint8)[:,:,0]
                self.progress_queue.put({"progress": 1})

            self.tracker.initialize(seed_frame_pil, None, bbox=bbox_xywh) # Re-initialize

            # Backward pass
            for i in range(seed_idx - 1, -1, -1):
                if self.cancel_event.is_set(): break
                frame_pil = Image.fromarray(cv2.cvtColor(shot_frames[i], cv2.COLOR_BGR2RGB))
                outputs = self.tracker.track(frame_pil)
                masks[i] = (outputs.get('pred_mask') * 255).astype(np.uint8) if outputs.get('pred_mask') is not None else np.zeros_like(shot_frames[i], dtype=np.uint8)[:,:,0]
                self.progress_queue.put({"progress": 1})

            h, w = shot_frames[0].shape[:2]
            for i in range(len(masks)):
                if self.cancel_event.is_set(): masks[i] = None
                if masks[i] is None: masks[i] = np.zeros((h, w), dtype=np.uint8)
                
                if self.person_detector and np.any(masks[i]):
                    boxes = self.person_detector.detect_boxes(shot_frames[i])
                    if boxes:
                        ys, xs = np.where(masks[i] > 0)
                        if ys.size > 0:
                            x1m, x2m, y1m, y2m = xs.min(), xs.max()+1, ys.min(), ys.max()+1
                            def iou(b):
                                ix1, iy1 = max(b[0], x1m), max(b[1], y1m)
                                ix2, iy2 = min(b[2], x2m), min(b[3], y2m)
                                inter = max(0, ix2-ix1)*max(0, iy2-iy1)
                                union = (b[2]-b[0])*(b[3]-b[1]) + (x2m-x1m)*(y2m-y1m) - inter + 1e-6
                                return inter/union
                            
                            bx1, by1, bx2, by2, _ = max(boxes, key=iou)
                            clip = np.zeros_like(masks[i])
                            clip[by1:by2, bx1:bx2] = 255
                            masks[i] = cv2.bitwise_and(masks[i], clip)

            img_area = h * w
            final_results = []
            for mask in masks:
                area_pct = (np.sum(mask > 0) / img_area) * 100 if img_area > 0 else 0.0
                is_empty = area_pct < config.MIN_MASK_AREA_PCT
                final_results.append((mask, float(area_pct), bool(is_empty), "Empty mask" if is_empty else None))
            return zip(*final_results)

        except Exception as e:
            logger.critical(f"DAM4SAM propagation failed: {e}", exc_info=True)
            h, w = shot_frames[0].shape[:2]
            return ([np.zeros((h,w), np.uint8)]*len(shot_frames), [0.0]*len(shot_frames), [True]*len(shot_frames), [f"Propagation failed: {e}"]*len(shot_frames))

# --- Backend Analysis Pipeline ---
class VideoManager:
    def __init__(self, source_path, max_resolution="maximum available"):
        self.source_path = source_path; self.max_resolution = max_resolution
        self.is_youtube = "youtube.com/" in source_path or "youtu.be/" in source_path

    def prepare_video(self):
        if self.is_youtube:
            if not ytdlp: raise ImportError("yt-dlp not installed.")
            logger.info(f"Downloading video: {self.source_path}")
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
            logger.info(f"Video ready: {sanitize_filename(video_path.name)}")

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
        self.features = get_feature_status()

    def run(self):
        try:
            if not self.output_dir.is_dir(): raise ValueError("Output folder is required.")
            config_hash = self._get_config_hash()
            if self.params.resume and self._check_resume(config_hash):
                return {"done": True, "metadata_path": str(self.metadata_path), "output_dir": str(self.output_dir)}
            
            if (self.output_dir / "masks").exists(): shutil.rmtree(self.output_dir / "masks")
            self.metadata_path.unlink(missing_ok=True)
            with self.metadata_path.open('w') as f:
                header = {"config_hash": config_hash, "params": {k:v for k,v in asdict(self.params).items() if k not in ['source_path', 'output_folder', 'video_path']}}
                f.write(json.dumps(header) + '\n')

            needs_face_analyzer = self.params.enable_face_filter or \
                                  (self.params.enable_subject_mask and "Reference Face" in self.params.seed_strategy)
            if needs_face_analyzer:
                self._initialize_face_analyzer()
            if self.params.enable_face_filter and self.params.face_ref_img_path:
                self._process_reference_face()
            
            person_detector = None
            if self.params.enable_subject_mask and self.features['person_detection']:
                try:
                    person_detector = PersonDetector(model=self.params.person_detector_model)
                    logger.info(f"Person detector ({self.params.person_detector_model}) initialized.")
                except Exception as e:
                    logger.warning(f"Person detector unavailable: {e}")

            if self.params.enable_subject_mask:
                if not self.features['masking']:
                    logger.warning("Subject masking unavailable (missing dependencies or no CUDA).")
                else:
                    is_video_path_valid = self.params.video_path and Path(self.params.video_path).exists()
                    if self.params.scene_detect and not is_video_path_valid:
                        logger.warning("Valid video path not provided; scene detection for masking disabled.")
                    masker = SubjectMasker(self.params, self.progress_queue, self.cancel_event, self._create_frame_map(),
                                           self.face_analyzer, self.reference_embedding, person_detector)
                    self.mask_metadata = masker.run(self.params.video_path if is_video_path_valid else "", str(self.output_dir))
            
            self._run_frame_processing()
            if self.cancel_event.is_set(): return {"log": "Analysis cancelled."}
            logger.success("Analysis complete. Go to 'Filtering & Export' tab.")
            return {"done": True, "metadata_path": str(self.metadata_path), "output_dir": str(self.output_dir)}
        except Exception as e:
            return logger.pipeline_error("analysis", e)

    def _get_config_hash(self):
        d = asdict(self.params)
        params_to_hash = {k: d.get(k) for k in ['enable_subject_mask', 'scene_detect', 'enable_face_filter',
                                                'face_model_name', 'quality_weights', 'dam4sam_model_name']}
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
            logger.error(f"Failed to parse frame_map.json: {e}. Using filename-based mapping.")
            return {int(re.search(r'frame_(\d+)', f.name).group(1)): f.name for f in image_files}

    def _initialize_face_analyzer(self):
        if not FaceAnalysis: raise ImportError("insightface library not installed.")
        if self.face_analyzer: return
        logger.info(f"Loading face model: {self.params.face_model_name}")
        try:
            self.face_analyzer = FaceAnalysis(
                name=self.params.face_model_name,
                root=str(config.DIRS['models']),
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            try:
                self.face_analyzer.prepare(ctx_id=0, det_size=(640, 640))
                logger.success("Face model loaded with CUDA.")
            except Exception as e_cuda:
                logger.warning(f"CUDA init failed: {e_cuda}. Falling back to CPU...")
                self.face_analyzer.prepare(ctx_id=-1, det_size=(640, 640))
                logger.success("Face model loaded with CPU.")
        except Exception as e_cpu:
            raise RuntimeError(f"Could not initialize face analysis model. Error: {e_cpu}") from e_cpu

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

    def _run_frame_processing(self):
        image_files = sorted(list(self.output_dir.glob("frame_*.png")) + list(self.output_dir.glob("frame_*.jpg")))
        
        thumb_dir = self.output_dir / "thumbs"
        thumb_dir.mkdir(exist_ok=True)
        for img_path in image_files:
            thumb_path = thumb_dir / f"{img_path.stem}.jpg"
            if not thumb_path.exists():
                img = cv2.imread(str(img_path))
                if img is not None:
                    h, w = img.shape[:2]
                    scale = math.sqrt(500_000 / (h * w)) if (h*w) > 500_000 else 1.0
                    if scale < 1.0:
                        thumb = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
                        cv2.imwrite(str(thumb_path), thumb, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    else:
                        shutil.copy2(img_path, thumb_path)

        self.progress_queue.put({"total": len(image_files), "stage": "Analysis"})
        num_workers = 1 if self.params.disable_parallel or self.params.enable_face_filter else min(os.cpu_count() or 4, 8)
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            list(executor.map(self._process_single_frame, image_files))

    def _process_single_frame(self, image_path):
        if self.cancel_event.is_set(): return
        try:
            image_data = cv2.imread(str(image_path))
            if image_data is None: raise ValueError("Could not read image.")
            
            frame_num_match = re.search(r'frame_(\d+)', image_path.name)
            frame_num = int(frame_num_match.group(1)) if frame_num_match else -1

            frame = Frame(image_data, frame_num)
            mask_meta = self.mask_metadata.get(image_path.name, {})
            mask = cv2.imread(mask_meta["mask_path"], cv2.IMREAD_GRAYSCALE) if mask_meta.get("mask_path") else None
            frame.calculate_quality_metrics(mask=mask)

            if self.params.enable_face_filter and self.reference_embedding is not None and self.face_analyzer:
                self._analyze_face_similarity(frame)
            
            meta = {"filename": image_path.name, "metrics": asdict(frame.metrics)}
            if frame.face_similarity_score is not None:
                meta["face_sim"] = frame.face_similarity_score
            if frame.max_face_confidence is not None:
                meta["face_conf"] = frame.max_face_confidence
            meta.update(mask_meta)
            
            if imagehash and self.params.enable_dedup:
                pil_img = Image.fromarray(cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB))
                meta['phash'] = str(imagehash.phash(pil_img))

            if frame.error: meta["error"] = frame.error
            if meta.get("mask_path"): meta["mask_path"] = Path(meta["mask_path"]).name
            
            meta = _to_json_safe(meta)
            with self.write_lock, self.metadata_path.open('a') as f:
                json.dump(meta, f)
                f.write('\n')
            self.progress_queue.put({"progress": 1})
        except Exception as e:
            logger.critical(f"Error processing frame {image_path.name}: {e}", exc_info=True)
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
        self.feature_status = get_feature_status()
        self.config_manager = self.ConfigurationManager(config.DIRS['configs'])

    class ConfigurationManager:
        def __init__(self, config_dir: Path): self.config_dir = config_dir
        def list_configs(self): return [f.stem for f in self.config_dir.glob("*.json")]
        def save_config(self, name, settings):
            if not name: raise ValueError("Config name required.")
            config_path = self.config_dir / f"{sanitize_filename(name)}.json"
            if config_path.exists(): shutil.copy2(config_path, config_path.with_suffix('.json.bak'))
            with open(config_path, 'w') as f: json.dump(settings, f, indent=2)
        def load_config(self, name):
            if not name: raise ValueError("Config name required.")
            with open(self.config_dir / f"{sanitize_filename(name)}.json", 'r') as f: return json.load(f)
        def delete_config(self, name):
            if name: (self.config_dir / f"{sanitize_filename(name)}.json").unlink(missing_ok=True)

    def build_ui(self):
        with gr.Blocks(theme=gr.themes.Default()) as demo:
            gr.Markdown("# 🎬 Frame Extractor & Analyzer")
            if not self.feature_status['cuda_available']:
                gr.Markdown("⚠️ **CPU Mode** — GPU-dependent features (Face Analysis, Subject Masking) are disabled.")
            
            with gr.Tabs():
                with gr.Tab("📹 1. Frame Extraction"): self._create_extraction_tab()
                with gr.Tab("🔍 2. Frame Analysis") as self.components['analysis_tab']: self._create_analysis_tab()
                with gr.Tab("🎯 3. Filtering & Export") as self.components['filtering_tab']: self._create_filtering_tab()

            with gr.Row():
                with gr.Column(scale=2):
                    self._create_component('unified_log', 'textbox', {'label': "📋 Processing Log", 'lines': 10, 'interactive': False, 'autoscroll': True})
                with gr.Column(scale=1):
                    self._create_component('unified_status', 'textbox', {'label': "📊 Status Summary", 'lines': 2, 'interactive': False})

            with gr.Accordion("⚙️ Config", open=False):
                with gr.Row():
                    with gr.Column(): self._create_config_presets_ui()
                    with gr.Column():
                        self._create_component('disable_parallel_input', 'checkbox', {'label': "🐌 Disable Parallelism", 'value': config.UI_DEFAULTS["disable_parallel"]})
            
            self._create_event_handlers()
        return demo

    def _create_component(self, name, comp_type, kwargs):
        comp_map = {'button': gr.Button, 'textbox': gr.Textbox, 'dropdown': gr.Dropdown, 'slider': gr.Slider,
                    'checkbox': gr.Checkbox, 'file': gr.File, 'radio': gr.Radio, 'gallery': gr.Gallery,
                    'plot': gr.Plot, 'markdown': gr.Markdown}
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
                method_choices = ["keyframes", "interval", "every_nth_frame", "all"] + (["scene"] if self.feature_status['scene_detection'] else [])
                self._create_component('method_input', 'dropdown', {'choices': method_choices, 'value': "all", 'label': "Method"})
                self._create_component('interval_input', 'textbox', {'label': "Interval (s)", 'value': 5.0, 'visible': False})
                self._create_component('nth_frame_input', 'textbox', {'label': "N-th Frame Value", 'value': config.UI_DEFAULTS["nth_frame"], 'visible': False})
                self._create_component('fast_scene_input', 'checkbox', {'label': "Fast Scene Detect", 'visible': False})
                self._create_component('max_resolution', 'dropdown', {'choices': ["maximum available", "2160", "1080", "720"], 'value': "maximum available", 'label': "DL Res"})
                self._create_component('use_png_input', 'checkbox', {'label': "Save as PNG", 'value': True})
        
        start_btn = gr.Button("🚀 Start Extraction", variant="primary")
        stop_btn = gr.Button("⏹️ Stop", variant="stop", interactive=False)
        self.components.update({'start_extraction_button': start_btn, 'stop_extraction_button': stop_btn})

    def _create_analysis_tab(self):
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📁 Input")
                self._create_component('frames_folder_input', 'textbox', {'label': "📂 Extracted Frames Folder"})
                self._create_component('analysis_video_path_input', 'textbox', {'label': "🎥 Original Video Path (Optional)"})
                self._create_component('resume_input', 'checkbox', {'label': "💾 Skip re-analysis", 'value': config.UI_DEFAULTS["resume"]})
            with gr.Column(scale=2):
                gr.Markdown("### ⚙️ Analysis Settings")
                with gr.Group():
                    self._create_component('enable_face_filter_input', 'checkbox', {'label': "Enable Face Similarity", 'value': config.UI_DEFAULTS['enable_face_filter'], 'interactive': self.feature_status['face_analysis']})
                    self._create_component('face_model_name_input', 'dropdown', {'choices': ["buffalo_l", "buffalo_s"], 'value': "buffalo_l", 'label': "Face Model"})
                    self._create_component('face_ref_img_path_input', 'textbox', {'label': "📸 Reference Image Path"})
                    self._create_component('face_ref_img_upload_input', 'file', {'label': "📤 Or Upload", 'type': "filepath"})
                with gr.Group():
                    self._create_component('enable_subject_mask_input', 'checkbox', {'label': "Enable Subject-Only Metrics", 'value': self.feature_status['masking'], 'interactive': self.feature_status['masking']})
                    self._create_component('dam4sam_model_name_input', 'dropdown', {'choices': ['sam21pp-T', 'sam21pp-S', 'sam21pp-B+', 'sam21pp-L'], 'value': 'sam21pp-L', 'label': "DAM4SAM Model"})
                    self._create_component('person_detector_model_input', 'dropdown', {'choices': ['yolo11x.pt', 'yolo11s.pt'], 'value': 'yolo11x.pt', 'label': "Person Detector"})
                    self._create_component('seed_strategy_input', 'dropdown', {'choices': ["Reference Face / Largest", "Largest Person", "Center-most Person"], 'value': config.UI_DEFAULTS['seed_strategy'], 'label': "Seed Strategy"})
                    self._create_component('scene_detect_input', 'checkbox', {'label': "Use Scene Detection", 'value': self.feature_status['scene_detection'], 'interactive': self.feature_status['scene_detection']})
                with gr.Group():
                    self._create_component('enable_dedup_input', 'checkbox', {'label': "Enable Near-Duplicate Filtering", 'value': config.UI_DEFAULTS['enable_dedup'], 'interactive': self.feature_status['perceptual_hashing']})
                    
        start_btn = gr.Button("🔬 Start Analysis", variant="primary")
        stop_btn = gr.Button("⏹️ Stop", variant="stop", interactive=False)
        self.components.update({'start_analysis_button': start_btn, 'stop_analysis_button': stop_btn})

    def _create_filtering_tab(self):
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 🎛️ Filter Controls")
                self._create_component('auto_pctl_input', 'slider', {
                    'label': 'Auto-Threshold Percentile', 'minimum': 50, 'maximum': 99, 'value': 75, 'step': 1
                })
                with gr.Row():
                    self._create_component('apply_auto_button', 'button', {'value': 'Apply Percentile to Mins'})
                    self._create_component('reset_filters_button', 'button', {'value': "Reset Filters"})
                
                self._create_component('filter_status_text', 'markdown', {'value': "Load an analysis to begin."})

                self.components['metric_plots'] = {}
                self.components['metric_sliders'] = {}
                
                with gr.Accordion("Deduplication", open=True, visible=self.feature_status['perceptual_hashing']):
                    self._create_component('dedup_thresh_input', 'slider', {'label': "Similarity Threshold", 'minimum': 0, 'maximum': 32, 'value': config.UI_DEFAULTS['dedup_thresh'], 'step': 1})

                all_metrics = config.QUALITY_METRICS + ["face_sim", "mask_area_pct"]
                for k in all_metrics:
                    with gr.Accordion(k.replace('_', ' ').title(), open=k in config.QUALITY_METRICS):
                        self.components['metric_plots'][k] = self._create_component(f'plot_{k}', 'plot', {'label': f"{k} Distribution", 'visible': False, 'config': {'staticPlot': True}})
                        with gr.Row():
                            if k == "face_sim":
                                self.components['metric_sliders'][f"{k}_min"] = self._create_component(f'slider_{k}_min', 'slider', {'label': "Min", 'minimum': 0.0, 'maximum': 1.0, 'value': 0.5, 'step': 0.01, 'interactive': True, 'visible': False})
                                self._create_component('require_face_match_input', 'checkbox', {'label': "Reject if no face", 'value': config.UI_DEFAULTS['require_face_match'], 'visible': False})
                            elif k == "mask_area_pct":
                                self.components['metric_sliders'][f"{k}_min"] = self._create_component(f'slider_{k}_min', 'slider', {'label': "Min %", 'minimum': 0.0, 'maximum': 100.0, 'value': config.MIN_MASK_AREA_PCT, 'step': 0.1, 'interactive': True, 'visible': False})
                            else:
                                self.components['metric_sliders'][f"{k}_min"] = self._create_component(f'slider_{k}_min', 'slider', {'label': "Min", 'minimum': 0.0, 'maximum': 100.0, 'value': 0.0, 'step': 0.5, 'interactive': True, 'visible': False})
                                self.components['metric_sliders'][f"{k}_max"] = self._create_component(f'slider_{k}_max', 'slider', {'label': "Max", 'minimum': 0.0, 'maximum': 100.0, 'value': 100.0, 'step': 0.5, 'interactive': True, 'visible': False})

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

    def _create_config_presets_ui(self):
        with gr.Group():
            self._create_component('config_status', 'textbox', {'label': "Status", 'interactive': False, 'lines': 1})
            with gr.Row():
                self._create_component('config_dropdown', 'dropdown', {'label': "Select Config", 'choices': self.config_manager.list_configs()})
                self._create_component('load_button', 'button', {'value': "Load"})
                self._create_component('delete_button', 'button', {'value': "Delete", 'variant': "stop"})
            with gr.Row():
                self._create_component('config_name_input', 'textbox', {'label': "New Config Name"})
                self._create_component('save_button', 'button', {'value': "Save"})

    def _create_event_handlers(self):
        self.components.update({
            'extracted_video_path_state': gr.State(""), 'extracted_frames_dir_state': gr.State(""),
            'analysis_output_dir_state': gr.State(""), 'analysis_metadata_path_state': gr.State(""),
            'all_frames_data_state': gr.State([]), 'per_metric_values_state': gr.State({})
        })
        self._setup_visibility_toggles()
        self._setup_pipeline_handlers()
        self._setup_filtering_handlers()
        self._setup_config_handlers()

    def _setup_visibility_toggles(self):
        c = self.components
        c['method_input'].change(
            lambda m: (gr.update(visible=m=='interval'), gr.update(visible=m=='scene'), gr.update(visible=m=='every_nth_frame')),
            c['method_input'],
            [c['interval_input'], c['fast_scene_input'], c['nth_frame_input']]
        )

    def _setup_pipeline_handlers(self):
        ext_inputs = [
            self.components['source_input'], self.components['upload_video_input'], self.components['method_input'],
            self.components['interval_input'], self.components['nth_frame_input'], self.components['fast_scene_input'],
            self.components['max_resolution'], self.components['use_png_input']
        ]
        ext_outputs = [c for name, c in self.components.items() if name in ['start_extraction_button', 'stop_extraction_button', 'unified_log', 'unified_status', 'extracted_video_path_state', 'extracted_frames_dir_state', 'frames_folder_input', 'analysis_video_path_input']]
        self.components['start_extraction_button'].click(self.run_extraction_wrapper, ext_inputs, ext_outputs)
        self.components['stop_extraction_button'].click(lambda: self.cancel_event.set())

        ana_inputs = [
            self.components['frames_folder_input'], self.components['analysis_video_path_input'],
            self.components['disable_parallel_input'], self.components['resume_input'],
            self.components['enable_face_filter_input'], self.components['face_ref_img_path_input'],
            self.components['face_ref_img_upload_input'], self.components['face_model_name_input'],
            self.components['enable_subject_mask_input'], self.components['dam4sam_model_name_input'],
            self.components['person_detector_model_input'], self.components['seed_strategy_input'], 
            self.components['scene_detect_input'], self.components['enable_dedup_input']
        ]
        
        ana_outputs = [c for name, c in self.components.items() if name in ['start_analysis_button', 'stop_analysis_button', 'unified_log', 'unified_status', 'analysis_output_dir_state', 'analysis_metadata_path_state', 'filtering_tab']]
        self.components['start_analysis_button'].click(self.run_analysis_wrapper, ana_inputs, ana_outputs)
        self.components['stop_analysis_button'].click(lambda: self.cancel_event.set())

    def _setup_filtering_handlers(self):
        c = self.components
        
        slider_keys = sorted(c['metric_sliders'].keys())
        slider_comps = [c['metric_sliders'][k] for k in slider_keys]
        
        # Consolidate all filter inputs
        filter_inputs = [
            c['all_frames_data_state'], c['per_metric_values_state'], c['analysis_output_dir_state'],
            c['gallery_view_toggle'], c['show_mask_overlay_input'], c['overlay_alpha_slider'],
            c['require_face_match_input'], c['dedup_thresh_input']
        ] + slider_comps

        fast_filter_outputs = [c['filter_status_text'], c['results_gallery']]
        
        # Fast updates on slider release
        for control in slider_comps + [c['dedup_thresh_input']]:
            control.release(self.on_filters_changed_fast, filter_inputs, fast_filter_outputs)
        
        plot_keys = config.QUALITY_METRICS + ["face_sim", "mask_area_pct"]
        plot_comps = [c['metric_plots'][k] for k in plot_keys]
        full_filter_outputs = plot_comps + fast_filter_outputs
        
        # Slower updates for other controls
        for control in [c['gallery_view_toggle'], c['show_mask_overlay_input'], c['overlay_alpha_slider'], c['require_face_match_input']]:
            control.input(self.on_filters_changed, filter_inputs, full_filter_outputs)

        # Tab selection logic
        def load_and_trigger_update(metadata_path, *current_slider_values):
            all_frames, metric_values = self.load_and_prep_filter_data(metadata_path)
            
            # Create a dict of all default filter values for the initial run
            # Note: The *current_slider_values passed here might not be defaults if user fiddled before loading
            # It's better to construct the filter dict manually from defaults.
            default_filters = { f"slider_{k}_min": s.value for k,s in c['metric_sliders'].items() if k.endswith('_min')}
            default_filters.update({ f"slider_{k}_max": s.value for k,s in c['metric_sliders'].items() if k.endswith('_max')})
            default_filters['require_face_match_input'] = c['require_face_match_input'].value
            default_filters['dedup_thresh_input'] = c['dedup_thresh_input'].value

            # This is complex because Gradio's state management is tricky. We'll simplify.
            # We will pass all filter components to the load function.
            
            visibility_updates = {}
            for k in plot_keys:
                has_data = k in metric_values and len(metric_values.get(k, [])) > 0
                visibility_updates[c['metric_plots'][k]] = gr.update(visible=has_data)
                if f"{k}_min" in c['metric_sliders']:
                    visibility_updates[c['metric_sliders'][f"{k}_min"]] = gr.update(visible=has_data)
                if k == "face_sim" and 'require_face_match_input' in c:
                     visibility_updates[c['require_face_match_input']] = gr.update(visible=has_data)
                if f"{k}_max" in c['metric_sliders']:
                    visibility_updates[c['metric_sliders'][f"{k}_max"]] = gr.update(visible=has_data)

            # Manually get all current filter values to run the first filter
            all_filter_vals = [s.value for s in slider_comps]
            all_filter_vals.insert(0, c['dedup_thresh_input'].value)
            all_filter_vals.insert(0, c['require_face_match_input'].value)

            filter_updates = self.on_filters_changed(all_frames, metric_values, c['analysis_output_dir_state'].value, "Kept Frames", True, 0.6, *all_filter_vals)

            final_updates = { c['all_frames_data_state']: all_frames, c['per_metric_values_state']: metric_values }
            final_updates.update({comp: val for comp, val in zip(full_filter_outputs, filter_updates)})
            final_updates.update(visibility_updates)
            return final_updates

        load_inputs = [c['analysis_metadata_path_state']] + [c['require_face_match_input'], c['dedup_thresh_input']] + slider_comps
        load_outputs = [c['all_frames_data_state'], c['per_metric_values_state']] + full_filter_outputs + list(c['metric_sliders'].values()) + [c.get('require_face_match_input')]
        c['filtering_tab'].select(load_and_trigger_update, load_inputs, [item for item in load_outputs if item is not None])

        # Also run when analysis metadata becomes available
        c['analysis_metadata_path_state'].change(load_and_trigger_update, load_inputs, [item for item in load_outputs if item is not None])

        # Export and other buttons
        export_inputs = [
            c['all_frames_data_state'], c['analysis_output_dir_state'], c['enable_crop_input'], 
            c['crop_ar_input'], c['crop_padding_input'], c['require_face_match_input'], c['dedup_thresh_input']
        ] + slider_comps
        c['export_button'].click(self.export_kept_frames, export_inputs, c['unified_log'])
        
        reset_outputs = list(c['metric_sliders'].values()) + [c['require_face_match_input'], c['dedup_thresh_input'], c['filter_status_text'], c['results_gallery']]
        c['reset_filters_button'].click(self.reset_filters, [c['all_frames_data_state'], c['per_metric_values_state'], c['analysis_output_dir_state']], reset_outputs)
        
        c['apply_auto_button'].click(
            self.auto_set_thresholds,
            [c['per_metric_values_state'], c['auto_pctl_input']],
            list(c['metric_sliders'].values())
        ).then(
            self.on_filters_changed_fast,
            filter_inputs,
            fast_filter_outputs
        )

    def _setup_config_handlers(self):
        c, cm = self.components, self.config_manager
        ordered_comp_ids = ['method_input', 'interval_input', 'max_resolution', 'fast_scene_input', 'use_png_input', 
                            'disable_parallel_input', 'resume_input', 'enable_face_filter_input', 
                            'face_model_name_input', 'enable_subject_mask_input', 'dam4sam_model_name_input', 
                            'person_detector_model_input', 'scene_detect_input', 'seed_strategy_input', 'enable_dedup_input']
        config_controls = [c[comp_id] for comp_id in ordered_comp_ids]
        
        c['save_button'].click(lambda name, *v: cm.save_config(name, {ordered_comp_ids[i]:val for i,val in enumerate(v)}) or (f"Saved '{name}'", gr.update(choices=cm.list_configs())), [c['config_name_input']] + config_controls, [c['config_status'], c['config_dropdown']])
        c['load_button'].click(lambda name: [v for k in ordered_comp_ids for v in [cm.load_config(name).get(k)]] + [f"Loaded '{name}'"], c['config_dropdown'], config_controls + [c['config_status']])
        c['delete_button'].click(lambda name: (cm.delete_config(name) or (f"Deleted '{name}'", gr.update(choices=cm.list_configs(), value=None))), c['config_dropdown'], [c['config_status'], c['config_dropdown']])

    def _set_ui_state(self, buttons, state, status_msg=""):
        start_btn, stop_btn = buttons
        log_comp, status_comp = self.components['unified_log'], self.components['unified_status']
        if state == "loading":
            return {start_btn: gr.update(interactive=False), stop_btn: gr.update(interactive=True), log_comp: "", status_comp: status_msg}
        elif state == "ready":
            return {start_btn: gr.update(interactive=True), stop_btn: gr.update(interactive=False)}
        elif state == "error":
            return {start_btn: gr.update(interactive=True), stop_btn: gr.update(interactive=False), log_comp: f"[ERROR] {status_msg}", status_comp: f"[ERROR] {status_msg}"}
        elif state == "success":
            return {start_btn: gr.update(interactive=True), stop_btn: gr.update(interactive=False), log_comp: f"[SUCCESS] {status_msg}", status_comp: f"[SUCCESS] {status_msg}"}
        return {}

    def run_extraction_wrapper(self, source_path, upload_video, method, interval, nth_frame, fast_scene, max_resolution, use_png):
        buttons = (self.components['start_extraction_button'], self.components['stop_extraction_button'])
        yield self._set_ui_state(buttons, "loading", "Starting extraction...")
        self.cancel_event.clear()
        
        source = upload_video if upload_video else source_path
        if not source:
            yield self._set_ui_state(buttons, "error", "Video source is required.")
            return

        try:
            if upload_video:
                source = str(config.DIRS['downloads'] / Path(source).name)
                shutil.copy2(upload_video, source)
            elif not re.match(r'https?://|youtu\.be', str(source)):
                if not Path(source).is_file(): raise ValueError("Local file not found.")
        except Exception as e:
            yield self._set_ui_state(buttons, "error", f"Invalid video source: {e}")
            return

        try:
            params = AnalysisParameters(
                source_path=source, method=method, interval=float(interval), nth_frame=int(nth_frame),
                fast_scene=fast_scene, max_resolution=max_resolution, use_png=use_png
            )
        except ValueError as e:
            yield self._set_ui_state(buttons, "error", f"Invalid numeric input for interval or N-th frame: {e}")
            return

        yield from self._run_task(ExtractionPipeline(params, Queue(), self.cancel_event).run)
        
        result = self.last_task_result
        if result.get("done") and not self.cancel_event.is_set():
            final_state = self._set_ui_state(buttons, "success", f"Extraction complete. Output: {result['output_dir']}")
            final_state.update({
                self.components['extracted_video_path_state']: result.get("video_path", ""),
                self.components['extracted_frames_dir_state']: result["output_dir"],
                self.components['frames_folder_input']: result["output_dir"],
                self.components['analysis_video_path_input']: result.get("video_path", "")
            })
            yield final_state
        else:
            yield self._set_ui_state(buttons, "ready")

    def run_analysis_wrapper(self, frames_folder, video_path,
                             disable_parallel, resume, enable_face_filter,
                             face_ref_img_path, face_ref_img_upload, face_model_name,
                             enable_subject_mask, dam4sam_model_name,
                             person_detector_model, seed_strategy, scene_detect, enable_dedup):
        buttons = (self.components['start_analysis_button'], self.components['stop_analysis_button'])
        yield self._set_ui_state(buttons, "loading", "Starting analysis...") | {
            self.components['analysis_output_dir_state']: gr.update(),
            self.components['analysis_metadata_path_state']: gr.update(),
            self.components['filtering_tab']: gr.update()
        }
        self.cancel_event.clear()

        try:
            if not frames_folder or not Path(frames_folder).is_dir(): raise ValueError("Valid frames folder required.")
            face_ref_full_path = face_ref_img_upload or face_ref_img_path
            if face_ref_full_path and face_ref_img_upload:
                dest = config.DIRS['downloads'] / Path(face_ref_img_upload).name
                shutil.copy2(face_ref_img_upload, dest)
                face_ref_full_path = str(dest)
        except Exception as e:
            yield self._set_ui_state(buttons, "error", f"Invalid input: {e}")
            return
        
        params = AnalysisParameters(
            output_folder=frames_folder, video_path=video_path, disable_parallel=disable_parallel, 
            resume=resume, enable_face_filter=enable_face_filter, face_ref_img_path=face_ref_full_path,
            face_model_name=face_model_name, enable_subject_mask=enable_subject_mask,
            dam4sam_model_name=dam4sam_model_name, person_detector_model=person_detector_model,
            seed_strategy=seed_strategy, scene_detect=scene_detect, enable_dedup=enable_dedup
        )
        yield from self._run_task(AnalysisPipeline(params, Queue(), self.cancel_event).run)

        result = self.last_task_result
        if result.get("done") and not self.cancel_event.is_set():
            final_state = self._set_ui_state(buttons, "success", f"Analysis complete. Output: {result['output_dir']}")
            final_state.update({
                self.components['analysis_output_dir_state']: result["output_dir"],
                self.components['analysis_metadata_path_state']: result["metadata_path"],
                self.components['filtering_tab']: gr.update(interactive=True)
            })
            yield final_state
        else:
            yield self._set_ui_state(buttons, "ready")
            
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
                        fps = processed / elapsed if elapsed > 0 else 0
                        eta = (total - processed) / fps if fps > 0 else 0
                        status = f"**{stage}:** {processed}/{total} ({processed/total:.1%}) | {fps:.1f} items/s | ETA: {int(eta//60):02d}:{int(eta%60):02d}"
                        yield {self.components['unified_log']: "\n".join(log_buffer), self.components['unified_status']: status}
                        last_yield = time.time()
                except Empty:
                    pass
        
        self.last_task_result = future.result() or {}
        if "log" in self.last_task_result: log_buffer.append(self.last_task_result["log"])
        if "error" in self.last_task_result: log_buffer.append(f"[ERROR] {self.last_task_result['error']}")
        
        status_text = "⏹️ Cancelled." if self.cancel_event.is_set() else f"❌ Error: {self.last_task_result.get('error')}" if 'error' in self.last_task_result else "✅ Complete."
        yield {self.components['unified_log']: "\n".join(log_buffer), self.components['unified_status']: status_text}

    def histogram_with_thresholds(self, hist_data, vmin, vmax, title=""):
        if hist_data is None: return None
        counts, bins = hist_data
        fig = go.Figure()
        fig.add_trace(go.Bar(x=bins[:-1], y=counts, marker_color="#7aa2ff", opacity=0.85, name="All"))
        fig.add_vrect(x0=vmin, x1=vmax, fillcolor="#00cc66", opacity=0.18, line_width=0)
        fig.update_layout(
            title_text=title, title_x=0.5, height=240,
            margin=dict(l=6, r=6, t=18, b=6),
            bargap=0.02, showlegend=False, uirevision="keep",
            modebar_remove=['zoom','select','lasso2d','autoScale2d','zoomIn2d','zoomOut2d','resetScale2d','toImage']
        )
        return fig

    @staticmethod
    def _apply_all_filters_vectorized(all_frames_data, filters):
        if not all_frames_data: return [], [], Counter(), {}

        num_frames = len(all_frames_data)
        filenames = [f['filename'] for f in all_frames_data]
        
        # Build metric arrays
        metric_arrays = {}
        for k in config.QUALITY_METRICS:
            metric_arrays[k] = np.array([f.get("metrics", {}).get(f"{k}_score", np.nan) for f in all_frames_data], dtype=np.float32)
        metric_arrays["face_sim"] = np.array([f.get("face_sim", np.nan) for f in all_frames_data], dtype=np.float32)
        metric_arrays["mask_area_pct"] = np.array([f.get("mask_area_pct", np.nan) for f in all_frames_data], dtype=np.float32)

        # Start with all frames kept, and a dict to track rejection reasons
        kept_mask = np.ones(num_frames, dtype=bool)
        reasons = defaultdict(list)

        # Apply filters
        for k in config.QUALITY_METRICS:
            min_val, max_val = filters.get(f"{k}_min", 0), filters.get(f"{k}_max", 100)
            low_mask = metric_arrays[k] < min_val
            high_mask = metric_arrays[k] > max_val
            for i in np.where(low_mask)[0]: reasons[filenames[i]].append(f"{k}_low")
            for i in np.where(high_mask)[0]: reasons[filenames[i]].append(f"{k}_high")
            kept_mask &= ~low_mask & ~high_mask

        if filters.get("face_sim_enabled"):
            min_val = filters.get("face_sim_min", 0.5)
            valid = ~np.isnan(metric_arrays["face_sim"])
            low_mask = valid & (metric_arrays["face_sim"] < min_val)
            for i in np.where(low_mask)[0]:
                reasons[filenames[i]].append("face_sim_low")
            kept_mask &= ~low_mask
            if filters.get("require_face_match"):
                missing_mask = ~valid
                for i in np.where(missing_mask)[0]:
                    reasons[filenames[i]].append("face_missing")
                kept_mask &= ~missing_mask
        
        if filters.get("mask_area_enabled"):
            min_val = filters.get("mask_area_pct_min", 1.0)
            low_mask = metric_arrays["mask_area_pct"] < min_val
            empty_mask = np.array([f.get("mask_empty", False) for f in all_frames_data])
            combined_mask = low_mask | empty_mask
            for i in np.where(combined_mask)[0]: reasons[filenames[i]].append("mask_too_small")
            kept_mask &= ~combined_mask

        # Deduplication on the remaining kept frames
        if filters.get("enable_dedup") and imagehash:
            kept_indices = np.where(kept_mask)[0]
            if len(kept_indices) > 1:
                sorted_kept = sorted([(filenames[i], i) for i in kept_indices])
                hashes = {i: imagehash.hex_to_hash(all_frames_data[i]['phash']) for _, i in sorted_kept if 'phash' in all_frames_data[i]}
                
                last_hash_idx = sorted_kept[0][1]
                for i in range(1, len(sorted_kept)):
                    current_idx = sorted_kept[i][1]
                    if last_hash_idx in hashes and current_idx in hashes:
                        dist = hashes[last_hash_idx] - hashes[current_idx]
                        if dist <= filters.get("dedup_thresh", 5):
                            kept_mask[current_idx] = False
                            reasons[filenames[current_idx]].append("duplicate")
                        else:
                            last_hash_idx = current_idx
                    else: # If hash is missing, keep the frame but don't update last_hash
                        last_hash_idx = current_idx

        # Final separation
        kept_indices = np.where(kept_mask)[0]
        rejected_indices = np.where(~kept_mask)[0]
        
        kept = [all_frames_data[i] for i in kept_indices]
        rejected = [all_frames_data[i] for i in rejected_indices]
        
        all_reasons_flat = [r for r_list in reasons.values() for r in r_list]
        counts = Counter(all_reasons_flat)

        return kept, rejected, counts, reasons

    def load_and_prep_filter_data(self, metadata_path):
        if not metadata_path or not Path(metadata_path).exists():
            return [], {}
        with Path(metadata_path).open('r') as f:
            next(f) # skip header
            all_frames = [json.loads(line) for line in f if line.strip()]

        metric_values = {}
        # Quality metrics
        for k in config.QUALITY_METRICS:
            values = np.asarray([f.get("metrics", {}).get(f"{k}_score") for f in all_frames if f.get("metrics", {}).get(f"{k}_score") is not None], dtype=float)
            if values.size > 0:
                counts, bins = np.histogram(values, bins=50, range=(0, 100))
                metric_values[k] = values.tolist()
                metric_values[f"{k}_hist"] = (counts.tolist(), bins.tolist())
        
        # Face similarity
        if any("face_sim" in f and f.get("face_sim") is not None for f in all_frames):
            values = np.asarray([f.get("face_sim") for f in all_frames if f.get("face_sim") is not None], dtype=float)
            if values.size > 0:
                counts, bins = np.histogram(values, bins=50, range=(0, 1))
                metric_values["face_sim"] = values.tolist()
                metric_values["face_sim_hist"] = (counts.tolist(), bins.tolist())

        # Mask area
        if any("mask_area_pct" in f for f in all_frames):
            values = np.asarray([f.get("mask_area_pct") for f in all_frames if f.get("mask_area_pct") is not None], dtype=float)
            if values.size > 0:
                counts, bins = np.histogram(values, bins=50, range=(0, 100))
                metric_values["mask_area_pct"] = values.tolist()
                metric_values["mask_area_pct_hist"] = (counts.tolist(), bins.tolist())
        
        return all_frames, metric_values

    def _update_gallery(self, all_frames_data, filters, output_dir, gallery_view, show_overlay, overlay_alpha):
        kept, rejected, counts, per_frame_reasons = self._apply_all_filters_vectorized(all_frames_data, filters or {})
        
        # Build active filters string safely
        active_filters = []
        for k in config.QUALITY_METRICS:
            if (filters or {}).get(f"{k}_min", 0) > 0:
                active_filters.append(f"{k}>={(filters.get(f'{k}_min', 0)):.1f}")
            if (filters or {}).get(f"{k}_max", 100) < 100:
                active_filters.append(f"{k}<={(filters.get(f'{k}_max', 100)):.1f}")

        if (filters or {}).get("face_sim_enabled", False):
            active_filters.append("face:on")
            if (filters or {}).get("face_sim_min", 0.5) > 0:
                active_filters.append(f"sim>={(filters.get('face_sim_min', 0.5)):.2f}")
            if (filters or {}).get("require_face_match", False):
                active_filters.append("req_face")

        if (filters or {}).get("mask_area_enabled", False):
            active_filters.append("mask:on")
            if (filters or {}).get("mask_area_pct_min", config.MIN_MASK_AREA_PCT) > 0:
                active_filters.append(f"mask>={(filters.get('mask_area_pct_min', config.MIN_MASK_AREA_PCT)):.1f}%")

        if (filters or {}).get("enable_dedup", False):
            active_filters.append(f"dedup<={(filters.get('dedup_thresh', config.UI_DEFAULTS['dedup_thresh']))}")

        total_frames = len(all_frames_data)
        status_parts = [f"**Kept:** {len(kept)}/{total_frames}"]
        if counts: status_parts.append(f"**Rejections:** {', '.join([f'{k}: {v}' for k,v in counts.most_common(3)])}")
        if active_filters: status_parts.append(f"(Active: {', '.join(active_filters)})")
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
                        mask_small = cv2.resize(mask_gray, (thumb.shape[1], thumb.shape[0]), interpolation=cv2.INTER_NEAREST)
                        thumb_overlay = render_mask_overlay(thumb, mask_small, float(overlay_alpha))
                        preview_images.append((cv2.cvtColor(thumb_overlay, cv2.COLOR_BGR2RGB), caption))
                    else:
                        preview_images.append((str(thumb_path), caption))
                else:
                    preview_images.append((str(thumb_path), caption))
        
        gallery_update = gr.update(
            value=preview_images, 
            rows=(1 if gallery_view == "Rejected Frames" else 2)
        )
        return status_text, gallery_update

    def on_filters_changed_fast(self, all_frames_data, per_metric_values, output_dir, gallery_view, show_overlay, overlay_alpha, require_face_match, dedup_thresh, *slider_values):
        if not all_frames_data: return "Run analysis to see results.", []
        slider_keys = sorted(self.components['metric_sliders'].keys())
        filters = {key: val for key, val in zip(slider_keys, slider_values)}
        filters.update({"require_face_match": require_face_match, "dedup_thresh": dedup_thresh,
                        "face_sim_enabled": bool(per_metric_values.get("face_sim")),
                        "mask_area_enabled": bool(per_metric_values.get("mask_area_pct")),
                        "enable_dedup": any('phash' in f for f in all_frames_data) if all_frames_data else False})
        status_text, gallery_update = self._update_gallery(all_frames_data, filters, output_dir, gallery_view, show_overlay, overlay_alpha)
        return status_text, gallery_update

    def on_filters_changed(self, all_frames_data, per_metric_values, output_dir, gallery_view, show_overlay, overlay_alpha, require_face_match, dedup_thresh, *slider_values):
        if not all_frames_data:
            return (gr.update(),) * len(self.components['metric_plots']) + ("Run analysis to see results.", [])

        slider_keys = sorted(self.components['metric_sliders'].keys())
        filters = {key: val for key, val in zip(slider_keys, slider_values)}
        filters.update({"require_face_match": require_face_match, "dedup_thresh": dedup_thresh,
                        "face_sim_enabled": bool(per_metric_values.get("face_sim")),
                        "mask_area_enabled": bool(per_metric_values.get("mask_area_pct")),
                        "enable_dedup": any('phash' in f for f in all_frames_data) if all_frames_data else False})
        
        status_text, gallery_update = self._update_gallery(all_frames_data, filters, output_dir, gallery_view, show_overlay, overlay_alpha)

        plot_updates = []
        for k in config.QUALITY_METRICS + ["face_sim", "mask_area_pct"]:
            vmin = filters.get(f"{k}_min", 0)
            vmax = filters.get(f"{k}_max", 100) if k in config.QUALITY_METRICS else (1.0 if k == "face_sim" else 100)
            plot_updates.append(self.histogram_with_thresholds(
                per_metric_values.get(f"{k}_hist"), vmin, vmax, title=""
            ))
        
        return tuple(plot_updates) + (status_text, gallery_update)

    def export_kept_frames(self, all_frames_data, output_dir, enable_crop, crop_ars, crop_padding, require_face_match, dedup_thresh, *slider_values):
        if not all_frames_data: return "No metadata to export."
        try:
            slider_keys = sorted(self.components['metric_sliders'].keys())
            filters = {key: val for key, val in zip(slider_keys, slider_values)}
            filters.update({"require_face_match": require_face_match, "dedup_thresh": dedup_thresh,
                            "face_sim_enabled": any("face_sim" in f for f in all_frames_data),
                            "mask_area_enabled": any("mask_area_pct" in f for f in all_frames_data),
                            "enable_dedup": "phash" in all_frames_data[0]})
            
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
                    if not src_path.exists():
                        logger.warning(f"Source file not found, skipping: {src_path}")
                        continue
                    
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
            logger.error(f"Error during export process: {e}", exc_info=True)
            return f"Error during export: {e}"

    def reset_filters(self, all_frames_data, per_metric_values, output_dir):
        updates = {}
        # 1) Reset control values to their initial defaults
        for k, s in self.components['metric_sliders'].items():
            # This relies on the component retaining its initial value. A safer way would be to get from config.
            # However, for this UI, the default is set on creation.
            initial_value = 0.0
            if k.endswith('_max'): initial_value = 100.0
            elif k == 'face_sim_min': initial_value = 0.5
            elif k == 'mask_area_pct_min': initial_value = config.MIN_MASK_AREA_PCT

            updates[s] = gr.update(value=initial_value)

        updates[self.components['require_face_match_input']] = gr.update(value=config.UI_DEFAULTS['require_face_match'])
        updates[self.components['dedup_thresh_input']] = gr.update(value=config.UI_DEFAULTS['dedup_thresh'])

        # 2) Recompute gallery/status using the same fast path with defaults
        if all_frames_data:
            slider_keys = sorted(self.components['metric_sliders'].keys())
            
            # Re-fetch default values for sliders
            slider_defaults = []
            for k in slider_keys:
                if k.endswith('_max'): slider_defaults.append(100.0)
                elif k == 'face_sim_min': slider_defaults.append(0.5)
                elif k == 'mask_area_pct_min': slider_defaults.append(config.MIN_MASK_AREA_PCT)
                else: slider_defaults.append(0.0)

            status_text, gallery_update = self.on_filters_changed_fast(
                all_frames_data,
                per_metric_values,
                output_dir,
                "Kept Frames", # gallery_view
                True, # show_overlay
                0.6, # overlay_alpha
                config.UI_DEFAULTS['require_face_match'],
                config.UI_DEFAULTS['dedup_thresh'],
                *slider_defaults
            )
            updates[self.components['filter_status_text']] = status_text
            updates[self.components['results_gallery']] = gallery_update
        else:
            updates[self.components['filter_status_text']] = "Load an analysis to begin."
            updates[self.components['results_gallery']] = []
        
        # Unpack the dict into a list of tuples for Gradio
        return list(updates.values())
    
    def auto_set_thresholds(self, per_metric_values, p=75):
        items = list(self.components['metric_sliders'].items())
        updates = []
        if not per_metric_values:
            return [gr.update() for _ in items]
        
        pmap = {}
        for k in config.QUALITY_METRICS:
            vals = per_metric_values.get(k) or []
            if len(vals) > 0:
                pmap[k] = float(np.percentile(np.asarray(vals, dtype=np.float32), p))
        
        for key, comp in items:
            if key.endswith('_min'):
                metric = key[:-4]
                if metric in pmap:
                    updates.append(gr.update(value=round(pmap[metric], 2)))
                else:
                    updates.append(gr.update())
            else:
                updates.append(gr.update())
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
        if mask is None: return img
        if mask.ndim == 3: mask = (mask > 0).any(axis=-1).astype(np.uint8)
        else: mask = (mask > 0).astype(np.uint8)
        
        ys, xs = np.where(mask > 0)
        if ys.size == 0: return img
        x1, x2, y1, y2 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
        
        bw, bh = x2 - x1, y2 - y1
        padp = max(padding, 0) / 100.0
        pad_x, pad_y = int(round(bw * padp)), int(round(bh * padp))
        x1, y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
        x2, y2 = min(w, x2 + pad_x), min(h, y2 + pad_y)
        
        bw, bh = x2 - x1, y2 - y1
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        
        ars = [self._parse_ar(s.strip()) for s in str(crop_ars).split(',') if s.strip()]
        ars = [(aw, ah) for aw, ah in ars if ah > 0]
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

        candidates = [res for r in [aw/ah for aw,ah in ars] if (res := expand_to_ar(r))]
        if candidates:
            x1n, y1n, x2n, y2n, _ = sorted(candidates, key=lambda t: t[4])[0]
            return img[y1n:y2n, x1n:x2n]
        return img[y1:y2, x1:x2]


if __name__ == "__main__":
    check_dependencies()
    AppUI().build_ui().launch()
