# keep app.py Monolithic!
"""
Frame Extractor & Analyzer v2.0
"""
import contextlib
import cv2
import dataclasses
from datetime import datetime
import functools
import gc
import gradio as gr
import hashlib
import imagehash
import io
import json
import logging
import math
import numpy as np
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import torch
import traceback
import urllib.request
import pandas as pd
from torchvision.ops import box_convert

from pathlib import Path

# --- Hugging Face Cache Setup ---
# Set cache directories for transformers and other Hugging Face assets
# to prevent re-downloading large models (like BERT for GroundingDINO) on every run.
# Use HF_HOME instead of deprecated TRANSFORMERS_CACHE
hf_home = Path(__file__).parent / 'models' / 'huggingface'
hf_home.mkdir(parents=True, exist_ok=True)
os.environ['HF_HOME'] = str(hf_home.resolve())

# Add submodules to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'Grounded-SAM-2'))
sys.path.insert(0, str(project_root / 'DAM4SAM'))

from collections import Counter, OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict, field, fields, is_dataclass
from enum import Enum
from functools import lru_cache
from queue import Empty, Queue
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar


# --- Standard Logging ---
def configure_logging(verbosity: int = 1, log_file: Optional[Path] = None) -> logging.Logger:
    logger = logging.getLogger("sfe")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.propagate = False

    fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    datefmt = "%H:%M:%S"
    stream_level = logging.WARNING if verbosity <= 0 else logging.INFO if verbosity == 1 else logging.DEBUG

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(stream_level)
    sh.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
    logger.addHandler(sh)

    if log_file:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
        logger.addHandler(fh)

    return logger

def get_logger() -> logging.Logger:
    return logging.getLogger("sfe")


# --- Step Timer ---
T = TypeVar('T')

def run_step(title: str, fn: Callable[[], T]) -> T:
    logger = get_logger()
    logger.info(f"Start: {title}")
    t0 = time.perf_counter()
    try:
        result = fn()
        return result
    except Exception:
        logger.exception(f"Error in step: {title}")
        raise
    finally:
        dt = time.perf_counter() - t0
        logger.info(f"End: {title} (took {dt:.2f}s)")

# --- Progress ---
Progress = Callable[[str, Optional[float]], None]

def noop_progress(msg: str, frac: Optional[float] = None) -> None:
    # Intentionally no-op
    return

# --- DEPENDENCY IMPORTS (with error handling) ---

# --- Global Model Cache ---
_yolo_model_cache = {}
_dino_model_cache = None
_dam4sam_model_cache = {}

try:
    from DAM4SAM.dam4sam_tracker import DAM4SAMTracker
    from DAM4SAM.utils import utils as dam_utils
except ImportError:
    DAM4SAMTracker = None
    dam_utils = None


try:
    from groundingdino.util.inference import (
        load_model as gdino_load_model,
        predict as gdino_predict,
    )
except ImportError:
    gdino_load_model = None
    gdino_predict = None


try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
except ImportError:
    mp = None
    python = None
    vision = None

try:
    from numba import njit
except ImportError:
    def njit(func):
        return func

try:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
except ImportError:
    plt = None
    mticker = None


try:
    from PIL import Image
except ImportError:
    Image = None

try:
    import pyiqa
except ImportError:
    pyiqa = None

try:
    from scenedetect import detect, ContentDetector
except ImportError:
    detect = None
    ContentDetector = None


try:
    import yt_dlp as ytdlp
except ImportError:
    ytdlp = None

# --- CONFIGURATION ---
@dataclass
class Config:
    @dataclass
    class Paths:
        logs: str = "logs"
        models: str = "models"
        downloads: str = "downloads"
        grounding_dino_config: str = "Grounded-SAM-2/grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        grounding_dino_checkpoint: str = "models/groundingdino_swint_ogc.pth"
        files: 'Config.Files' = field(default_factory=lambda: Config.Files())

    @dataclass
    class Files:
        frame_map_filename: str = "frame_map.json"
        sceneseeds_filename: str = "sceneseeds.json"
        masks_subdir: str = "masks"

    @dataclass
    class Models:
        user_agent: str = "Mozilla/5.0"
        grounding_dino: str = "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
        face_landmarker: str = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        dam4sam: Dict[str, str] = field(default_factory=lambda: {
            "sam21pp-T": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt",
            "sam21pp-S": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt",
            "sam21pp-B+": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt",
            "sam21pp-L": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
        })
        yolo: str = "https://huggingface.co/Ultralytics/YOLO11/resolve/main/"

    @dataclass
    class YouTubeDL:
        output_template: str = "%(id)s_%(title).40s_%(height)sp.%(ext)s"
        format_string: str = "bestvideo[height<={max_res}][ext=mp4]+bestaudio[ext=m4a]/best[height<={max_res}][ext=mp4]/best"

    @dataclass
    class Ffmpeg:
        log_level: str = "info"
        thumbnail_quality: int = 80
        scene_threshold: float = 0.4
        fast_scene_threshold: float = 0.5

    @dataclass
    class Cache:
        size: int = 200
        eviction_factor: float = 0.2
        cleanup_threshold: float = 0.8

    @dataclass
    class Retry:
        max_attempts: int = 3
        backoff_seconds: List[float] = field(default_factory=lambda: [1, 5, 15])

    @dataclass
    class QualityScaling:
        entropy_normalization: float = 8.0
        resolution_denominator: int = 500000
        contrast_clamp: float = 2.0
        niqe_offset: float = 10.0
        niqe_scale_factor: float = 10.0

    @dataclass
    class Masking:
        keep_largest_only: bool = True
        close_kernel_size: int = 5
        open_kernel_size: int = 5

    @dataclass
    class UIDefaults:
        thumbnails_only: bool = True
        thumb_megapixels: float = 0.5
        scene_detect: bool = True
        max_resolution: str = "maximum available"
        pre_analysis_enabled: bool = True
        pre_sample_nth: int = 5
        enable_face_filter: bool = True
        face_model_name: str = "buffalo_l"
        enable_subject_mask: bool = True
        dam4sam_model_name: str = "sam21pp-L"
        person_detector_model: str = "yolo11x.pt"
        primary_seed_strategy: str = "🤖 Automatic"
        seed_strategy: str = "Largest Person"
        text_prompt: str = "a person"
        resume: bool = False
        require_face_match: bool = False
        enable_dedup: bool = True
        dedup_thresh: int = 5
        method: str = "all"
        interval: float = 5.0
        fast_scene: bool = False
        use_png: bool = True
        nth_frame: int = 5
        disable_parallel: bool = False

    @dataclass
    class FilterDefaults:
        quality_score: Dict[str, float] = field(default_factory=lambda: {'min': 0.0, 'max': 100.0, 'step': 0.5, 'default_min': 0.0, 'default_max': 100.0})
        sharpness: Dict[str, float] = field(default_factory=lambda: {'min': 0.0, 'max': 100.0, 'step': 0.5, 'default_min': 0.0, 'default_max': 100.0})
        edge_strength: Dict[str, float] = field(default_factory=lambda: {'min': 0.0, 'max': 100.0, 'step': 0.5, 'default_min': 0.0, 'default_max': 100.0})
        contrast: Dict[str, float] = field(default_factory=lambda: {'min': 0.0, 'max': 100.0, 'step': 0.5, 'default_min': 0.0, 'default_max': 100.0})
        brightness: Dict[str, float] = field(default_factory=lambda: {'min': 0.0, 'max': 100.0, 'step': 0.5, 'default_min': 0.0, 'default_max': 100.0})
        entropy: Dict[str, float] = field(default_factory=lambda: {'min': 0.0, 'max': 100.0, 'step': 0.5, 'default_min': 0.0, 'default_max': 100.0})
        niqe: Dict[str, float] = field(default_factory=lambda: {'min': 0.0, 'max': 100.0, 'step': 0.5, 'default_min': 0.0, 'default_max': 100.0})
        face_sim: Dict[str, float] = field(default_factory=lambda: {'min': 0.0, 'max': 1.0, 'step': 0.01, 'default_min': 0.0})
        mask_area_pct: Dict[str, float] = field(default_factory=lambda: {'min': 0.0, 'max': 100.0, 'step': 0.1, 'default_min': 1.0})
        dedup_thresh: Dict[str, int] = field(default_factory=lambda: {'min': -1, 'max': 32, 'step': 1, 'default': -1})
        eyes_open: Dict[str, float] = field(default_factory=lambda: {'min': 0.0, 'max': 1.0, 'step': 0.01, 'default_min': 0.0})
        yaw: Dict[str, float] = field(default_factory=lambda: {'min': -180.0, 'max': 180.0, 'step': 1, 'default_min': -25, 'default_max': 25})
        pitch: Dict[str, float] = field(default_factory=lambda: {'min': -180.0, 'max': 180.0, 'step': 1, 'default_min': -25, 'default_max': 25})

    @dataclass
    class QualityWeights:
        sharpness: int = 25
        edge_strength: int = 15
        contrast: int = 15
        brightness: int = 10
        entropy: int = 15
        niqe: int = 20

    @dataclass
    class Choices:
        max_resolution: List[str] = field(default_factory=lambda: ["maximum available", "2160", "1080", "720"])
        extraction_method_toggle: List[str] = field(default_factory=lambda: ["Recommended Thumbnails", "Legacy Full-Frame"])
        method: List[str] = field(default_factory=lambda: ["keyframes", "interval", "every_nth_frame", "all", "scene"])
        primary_seed_strategy: List[str] = field(default_factory=lambda: ["👤 By Face", "📝 By Text", "🔄 Face + Text Fallback", "🧑‍🤝‍🧑 Find Prominent Person"])
        seed_strategy: List[str] = field(default_factory=lambda: ["Largest Person", "Center-most Person"])
        person_detector_model: List[str] = field(default_factory=lambda: ['yolo11x.pt', 'yolo11s.pt'])
        face_model_name: List[str] = field(default_factory=lambda: ["buffalo_l", "buffalo_s"])
        dam4sam_model_name: List[str] = field(default_factory=lambda: ["sam21pp-T", "sam21pp-S", "sam21pp-B+", "sam21pp-L"])
        gallery_view: List[str] = field(default_factory=lambda: ["Kept Frames", "Rejected Frames"])
        log_level: List[str] = field(default_factory=lambda: ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
        scene_gallery_view: List[str] = field(default_factory=lambda: ["Kept", "Rejected", "All"])

    @dataclass
    class GroundingDinoParams:
        box_threshold: float = 0.35
        text_threshold: float = 0.25

    @dataclass
    class PersonDetector:
        model: str = "yolo11x.pt"
        imgsz: int = 640
        conf: float = 0.3
    
    @dataclass
    class Logging:
        log_level: str = "INFO"
        log_format: str = '%(asctime)s | %(levelname)8s | %(name)s | %(message)s'
        colored_logs: bool = True
        structured_log_path: str = "structured_log.jsonl"

    paths: Paths = field(default_factory=Paths)
    models: Models = field(default_factory=Models)
    youtube_dl: YouTubeDL = field(default_factory=YouTubeDL)
    ffmpeg: Ffmpeg = field(default_factory=Ffmpeg)
    cache: Cache = field(default_factory=Cache)
    retry: Retry = field(default_factory=Retry)
    quality_scaling: QualityScaling = field(default_factory=QualityScaling)
    masking: Masking = field(default_factory=Masking)
    ui_defaults: UIDefaults = field(default_factory=UIDefaults)
    filter_defaults: FilterDefaults = field(default_factory=FilterDefaults)
    quality_weights: QualityWeights = field(default_factory=QualityWeights)
    choices: Choices = field(default_factory=Choices)
    grounding_dino_params: GroundingDinoParams = field(default_factory=GroundingDinoParams)
    person_detector: PersonDetector = field(default_factory=PersonDetector)
    logging: Logging = field(default_factory=Logging)
    
    sharpness_base_scale: int = 2500
    edge_strength_base_scale: int = 100
    min_mask_area_pct: float = 1.0
    
    @dataclass
    class Monitoring:
        memory_warning_threshold_mb: int = 8192
        memory_critical_threshold_mb: int = 16384
        cpu_warning_threshold_percent: int = 90
        gpu_memory_warning_threshold_percent: int = 90
        memory_limit_mb: int = 8192

    @dataclass
    class Processing:
        seed_frame_candidates: int = 10

    @dataclass
    class ExportOptions:
        enable_crop: bool = True
        crop_padding: int = 1
        crop_ars: str = "16:9,1:1,9:16"

    @dataclass
    class GradioDefaults:
        auto_pctl_input: int = 25
        show_mask_overlay: bool = True
        overlay_alpha: float = 0.6

    @dataclass
    class SeedingDefaults:
        face_similarity_threshold: float = 0.4
        yolo_iou_threshold: float = 0.3
        face_contain_score: int = 100
        confidence_score_multiplier: int = 20
        iou_bonus: int = 50
        face_to_body_expansion_factors: List[float] = field(default_factory=lambda: [4.0, 7.0, 0.75])
        final_fallback_box: List[float] = field(default_factory=lambda: [0.25, 0.25, 0.5, 0.5])

    @dataclass
    class UtilityDefaults:
        max_filename_length: int = 50
        video_extensions: List[str] = field(default_factory=lambda: ['.mp4','.mov','.mkv','.avi','.webm'])
        image_extensions: List[str] = field(default_factory=lambda: ['.png','.jpg','.jpeg','.webp','.bmp'])

    @dataclass
    class PostProcessing:
        mask_fill_kernel_size: int = 5

    @dataclass
    class Visualization:
        bbox_color: List[int] = field(default_factory=lambda: [255, 0, 0])
        bbox_thickness: int = 2

    @dataclass
    class Analysis:
        max_workers: int = 8
        default_batch_size: int = 32
        default_workers: int = 4
        
    @dataclass
    class ModelDefaults:
        face_analyzer_det_size: List[int] = field(default_factory=lambda: [640, 640])

    monitoring: Monitoring = field(default_factory=Monitoring)
    processing: Processing = field(default_factory=Processing)
    export_options: ExportOptions = field(default_factory=ExportOptions)
    gradio_defaults: GradioDefaults = field(default_factory=GradioDefaults)
    seeding_defaults: SeedingDefaults = field(default_factory=SeedingDefaults)
    utility_defaults: UtilityDefaults = field(default_factory=UtilityDefaults)
    post_processing: PostProcessing = field(default_factory=PostProcessing)
    visualization: Visualization = field(default_factory=Visualization)
    analysis: Analysis = field(default_factory=Analysis)
    model_defaults: ModelDefaults = field(default_factory=ModelDefaults)
    config_path: Optional[str] = "config.json"

    def __post_init__(self):
        config_dict = asdict(self)
        config_p = Path(self.config_path) if self.config_path else None
        if config_p and config_p.exists():
            try:
                with open(config_p, 'r', encoding='utf-8') as f:
                    file_config = json.load(f)
                if file_config:
                    self._merge_configs(config_dict, file_config)
            except Exception:
                print(f"Warning: Could not load or parse {self.config_path}")
        self._override_with_env_vars(config_dict)
        self._from_dict(config_dict)
        self._create_dirs()

    def _merge_configs(self, base, override):
        for key, value in override.items():
            if isinstance(value, dict) and isinstance(base.get(key), dict):
                base[key] = self._merge_configs(base.get(key, {}), value)
            else:
                base[key] = value
        return base

    def _override_with_env_vars(self, config_dict, prefix='APP'):
        for key, value in config_dict.items():
            new_prefix = f"{prefix}_{key.upper()}"
            if isinstance(value, dict):
                self._override_with_env_vars(value, new_prefix)
            else:
                env_var = os.getenv(new_prefix)
                if env_var is not None:
                    config_dict[key] = self._coerce_type(env_var, value)

    def _coerce_type(self, env_val, default_val):
        if isinstance(default_val, bool): return env_val.lower() in ['true', '1', 'yes']
        if isinstance(default_val, int): return int(env_val)
        if isinstance(default_val, float): return float(env_val)
        if isinstance(default_val, list):
            if not env_val: return []
            parts = [p.strip() for p in env_val.split(',') if p.strip()]
            element_default = default_val[0] if default_val else ""
            return [self._coerce_type(p, element_default) for p in parts]
        return env_val

    def _from_dict(self, data: Dict[str, Any]):
        for f in fields(self):
            if f.name in data:
                field_data = data[f.name]
                if is_dataclass(f.type) and isinstance(field_data, dict):
                    nested_instance = getattr(self, f.name)
                    nested_data = asdict(nested_instance)
                    self._merge_configs(nested_data, field_data)
                    setattr(self, f.name, f.type(**nested_data))
                else:
                    setattr(self, f.name, field_data)
        self._validate_config()

    def _validate_config(self):
        if sum(asdict(self.quality_weights).values()) == 0:
            raise ValueError("The sum of quality_weights cannot be zero.")

    def _create_dirs(self):
        for dir_path in [self.paths.logs, self.paths.models, self.paths.downloads]:
            if isinstance(dir_path, str):
                try:
                    Path(dir_path).mkdir(exist_ok=True, parents=True)
                except PermissionError as e:
                    raise RuntimeError(f"Cannot create directory at {dir_path}.") from e

    def save_config(self, path: str):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(_to_json_safe(asdict(self)), f, indent=2, ensure_ascii=False)

# Data classes and other definitions...
@dataclass
class UIEvent: pass

@dataclass
class ExtractionEvent(UIEvent):
    source_path: str
    upload_video: Optional[str]
    method: str
    interval: str
    nth_frame: str
    fast_scene: bool
    max_resolution: str
    use_png: bool
    thumbnails_only: bool
    thumb_megapixels: float
    scene_detect: bool

@dataclass
class PreAnalysisEvent(UIEvent):
    output_folder: str
    video_path: str
    resume: bool
    enable_face_filter: bool
    face_ref_img_path: str
    face_ref_img_upload: Optional[str]
    face_model_name: str
    enable_subject_mask: bool
    dam4sam_model_name: str
    person_detector_model: str
    best_frame_strategy: str
    scene_detect: bool
    text_prompt: str
    box_threshold: float
    text_threshold: float
    min_mask_area_pct: float
    sharpness_base_scale: float
    edge_strength_base_scale: float
    gdino_config_path: str
    gdino_checkpoint_path: str
    pre_analysis_enabled: bool
    pre_sample_nth: int
    primary_seed_strategy: str
    compute_quality_score: bool = True
    compute_sharpness: bool = True
    compute_edge_strength: bool = True
    compute_contrast: bool = True
    compute_brightness: bool = True
    compute_entropy: bool = True
    compute_eyes_open: bool = True
    compute_yaw: bool = True
    compute_pitch: bool = True
    compute_face_sim: bool = True
    compute_subject_mask_area: bool = True
    compute_niqe: bool = True
    compute_phash: bool = True
    enable_dedup: bool = True
    dedup_thresh: int = 5

@dataclass
class PropagationEvent(UIEvent):
    output_folder: str
    video_path: str
    scenes: list[dict[str, Any]]
    analysis_params: PreAnalysisEvent

@dataclass
class FilterEvent(UIEvent):
    all_frames_data: list[dict[str, Any]]
    per_metric_values: dict[str, Any]
    output_dir: str
    gallery_view: str
    show_overlay: bool
    overlay_alpha: float
    require_face_match: bool
    dedup_thresh: int
    slider_values: dict[str, float]

@dataclass
class ExportEvent(UIEvent):
    all_frames_data: list[dict[str, Any]]
    output_dir: str
    video_path: str
    enable_crop: bool
    crop_ars: str
    crop_padding: int
    filter_args: dict[str, Any]

@dataclass
class SessionLoadEvent(UIEvent):
    session_path: str

# --- UTILS ---

def _to_json_safe(obj):
    if isinstance(obj, (Path, datetime)):
        return str(obj)
    if hasattr(np, 'floating') and isinstance(obj, np.floating):
        return float(obj)
    if hasattr(np, 'integer') and isinstance(obj, np.integer):
        return int(obj)
    if dataclasses.is_dataclass(obj):
        return dataclasses.asdict(obj)
    if isinstance(obj, dict):
        return {k: _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(i) for i in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def estimate_totals(params: 'AnalysisParameters', video_info: dict, scenes: list['Scene'] | None) -> dict:
    fps = max(1, int(video_info.get("fps") or 30))
    total_frames = int(video_info.get("frame_count") or 0)

    # Extraction totals
    method = params.method
    if params.thumbnails_only:
        extraction_total = total_frames
    elif method == "interval":
        extraction_total = max(1, int(total_frames / max(0.1, params.interval) / fps))
    elif method == "every_nth_frame":
        extraction_total = max(1, int(total_frames / max(1, params.nth_frame)))
    elif method == "all":
        extraction_total = total_frames
    else:
        extraction_total = max(1, int(total_frames * 0.15))  # heuristic for keyframes/scene

    # Pre-analysis: one seed attempt per scene (adjust if you do retries)
    scenes_count = len(scenes or [])
    pre_analysis_total = max(0, scenes_count)

    # Propagation: frames per scene
    propagation_total = 0
    if scenes:
        for sc in scenes:
            propagation_total += max(0, sc.end_frame - sc.start_frame + 1)

    return {
        "extraction": extraction_total,
        "pre_analysis": pre_analysis_total,
        "propagation": propagation_total
    }

def sanitize_filename(name, config: 'Config', max_length=None):
    max_length = max_length or config.utility_defaults.max_filename_length
    return re.sub(r'[^\w\-_.]', '_', name)[:max_length]

def _coerce(val, to_type):
    if to_type is bool:
        if isinstance(val, bool): return val
        return str(val).strip().lower() in {"1", "true", "yes", "on"}
    if to_type in (int, float):
        try:
            return to_type(val)
        except (ValueError, TypeError):
            # Let the caller handle the exception if it's inappropriate
            # to fallback to a default. For AnalysisParameters, it has a try-except.
            raise
    return val

@contextlib.contextmanager
def safe_resource_cleanup():
    try: yield
    finally:
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

def is_image_folder(p: str | Path) -> bool:
    if not p:
        return False
    try:
        if not isinstance(p, (str, Path)):
            p = str(p)
        p = Path(p)
        return p.is_dir()
    except (TypeError, ValueError):
        return False

def list_images(p: str | Path, cfg: Config) -> list[Path]:
    p = Path(p)
    exts = {e.lower() for e in cfg.utility_defaults.image_extensions}
    return sorted([f for f in p.iterdir() if f.suffix.lower() in exts and f.is_file()])

def _sanitize_face_ref(runconfig: dict, config: Config, logger: logging.Logger) -> tuple[str, bool]:
    ref = (runconfig.get('face_ref_img_path') or '').strip()
    vid = (runconfig.get('video_path') or '').strip()
    if not ref:
        logger.info("No face reference in session; face similarity disabled on load.")
        return "", False
    bad_exts = set(config.utility_defaults.video_extensions)
    img_exts = set(config.utility_defaults.image_extensions)
    p = Path(ref)
    if ref == vid or p.suffix.lower() in bad_exts:
        logger.warning("Reference path appears to be a video or equals video_path; clearing safely.")
        return "", False
    if p.suffix.lower() not in img_exts or not p.is_file():
        logger.warning("Reference path is not a valid image on disk; clearing safely.", extra={'path': ref})
        return "", False
    return ref, True

# --- QUALITY ---

@njit
def compute_entropy(hist, entropy_norm):
    prob = hist / (np.sum(hist) + 1e-7)
    entropy = -np.sum(prob[prob > 0] * np.log2(prob[prob > 0]))
    return min(max(entropy / entropy_norm, 0), 1.0)

@dataclass
class QualityConfig:
    sharpness_base_scale: float
    edge_strength_base_scale: float
    enable_niqe: bool = True

# --- MODELS ---

@dataclass
class FrameMetrics:
    quality_score: float = 0.0
    sharpness_score: float = 0.0
    edge_strength_score: float = 0.0
    contrast_score: float = 0.0
    brightness_score: float = 0.0
    entropy_score: float = 0.0
    niqe_score: float = 0.0
    eyes_open: float = 0.0
    blink_prob: float = 0.0
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0

@dataclass
class Frame:
    image_data: np.ndarray
    frame_number: int
    metrics: FrameMetrics = field(default_factory=FrameMetrics)
    face_similarity_score: float | None = None
    max_face_confidence: float | None = None
    error: str | None = None
    bbox: Optional[list] = None
    mask_path: Optional[str] = None
    mask_area_pct: Optional[float] = None
    is_empty: Optional[bool] = None
    errors: List[str] = field(default_factory=list)

    def calculate_quality_metrics(self, thumb_image_rgb: np.ndarray, quality_config: QualityConfig, logger: logging.Logger,
                                  mask: np.ndarray | None = None, niqe_metric=None, main_config: 'Config' = None,
                                  face_landmarker=None, face_bbox: Optional[List[int]] = None,
                                  metrics_to_compute: Optional[Dict[str, bool]] = None):
        try:
            if metrics_to_compute is None:
                metrics_to_compute = {k: True for k in ['eyes_open', 'yaw', 'pitch', 'sharpness', 'edge_strength', 'contrast', 'brightness', 'entropy', 'quality']}

            if face_landmarker and any(metrics_to_compute.get(k) for k in ['eyes_open', 'yaw', 'pitch']):
                if face_bbox:
                    x1, y1, x2, y2 = face_bbox
                    face_img = thumb_image_rgb[y1:y2, x1:x2]
                else:
                    face_img = thumb_image_rgb

                if not face_img.flags['C_CONTIGUOUS']:
                    face_img = np.ascontiguousarray(face_img, dtype=np.uint8)
                if face_img.dtype != np.uint8:
                    face_img = face_img.astype(np.uint8)

                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=face_img)
                landmarker_result = face_landmarker.detect(mp_image)

                if landmarker_result.face_blendshapes:
                    blendshapes = {b.category_name: b.score for b in landmarker_result.face_blendshapes[0]}
                    if metrics_to_compute.get('eyes_open'):
                        self.metrics.eyes_open = 1.0 - max(blendshapes.get('eyeBlinkLeft', 0), blendshapes.get('eyeBlinkRight', 0))
                        self.metrics.blink_prob = max(blendshapes.get('eyeBlinkLeft', 0), blendshapes.get('eyeBlinkRight', 0))

                if landmarker_result.facial_transformation_matrixes and any(metrics_to_compute.get(k) for k in ['yaw', 'pitch']):
                    matrix = landmarker_result.facial_transformation_matrixes[0]
                    sy = math.sqrt(matrix[0, 0] * matrix[0, 0] + matrix[1, 0] * matrix[1, 0])
                    singular = sy < 1e-6
                    if not singular:
                        if metrics_to_compute.get('pitch'): self.metrics.pitch = math.degrees(math.atan2(-matrix[2, 0], sy))
                        if metrics_to_compute.get('yaw'): self.metrics.yaw = math.degrees(math.atan2(matrix[1, 0], matrix[0, 0]))
                        self.metrics.roll = math.degrees(math.atan2(matrix[2, 1], matrix[2, 2]))
                    else:
                        if metrics_to_compute.get('pitch'): self.metrics.pitch = math.degrees(math.atan2(-matrix[2, 0], sy))
                        if metrics_to_compute.get('yaw'): self.metrics.yaw = 0
                        self.metrics.roll = 0

            scores_norm = {}
            gray = cv2.cvtColor(thumb_image_rgb, cv2.COLOR_RGB2GRAY)
            active_mask = ((mask > 128) if mask is not None and mask.ndim == 2 else None)
            if active_mask is not None and np.sum(active_mask) < 100:
                active_mask = None

            def _calculate_and_store_score(name, value):
                normalized_value = min(max(value, 0.0), 1.0)
                scores_norm[name] = normalized_value
                setattr(self.metrics, f"{name}_score", float(normalized_value * 100))


            if metrics_to_compute.get('sharpness'):
                lap = cv2.Laplacian(gray, cv2.CV_64F)
                masked_lap = lap[active_mask] if active_mask is not None else lap
                sharpness = np.var(masked_lap) if masked_lap.size > 0 else 0
                sharpness_scaled = (sharpness / (quality_config.sharpness_base_scale * (gray.size / main_config.quality_scaling.resolution_denominator)))
                _calculate_and_store_score("sharpness", sharpness_scaled)

            if metrics_to_compute.get('edge_strength'):
                sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                edge_strength = np.mean(np.sqrt(sobelx**2 + sobely**2))
                edge_strength_scaled = (edge_strength / (quality_config.edge_strength_base_scale * (gray.size / main_config.quality_scaling.resolution_denominator)))
                _calculate_and_store_score("edge_strength", edge_strength_scaled)

            if metrics_to_compute.get('contrast') or metrics_to_compute.get('brightness'):
                pixels = gray[active_mask] if active_mask is not None else gray
                mean_br, std_br = (np.mean(pixels), np.std(pixels)) if pixels.size > 0 else (0, 0)
                if metrics_to_compute.get('brightness'):
                    brightness = mean_br / 255.0
                    _calculate_and_store_score("brightness", brightness)
                if metrics_to_compute.get('contrast'):
                    contrast = std_br / (mean_br + 1e-7)
                    contrast_scaled = min(contrast, main_config.quality_scaling.contrast_clamp) / main_config.quality_scaling.contrast_clamp
                    _calculate_and_store_score("contrast", contrast_scaled)

            if metrics_to_compute.get('entropy'):
                gray_full = cv2.cvtColor(self.image_data, cv2.COLOR_RGB2GRAY)
                active_mask_full = None
                if mask is not None:
                    mask_full = cv2.resize(mask, (gray_full.shape[1], gray_full.shape[0]), interpolation=cv2.INTER_NEAREST)
                    active_mask_full = (mask_full > 128).astype(np.uint8)
                hist = cv2.calcHist([gray_full], [0], active_mask_full, [256], [0, 256]).flatten()
                entropy = compute_entropy(hist, main_config.quality_scaling.entropy_normalization)
                scores_norm["entropy"] = entropy
                self.metrics.entropy_score = float(entropy * 100)

            if quality_config.enable_niqe and niqe_metric is not None:
                try:
                    rgb_image = self.image_data
                    if active_mask is not None:
                        active_mask_full = cv2.resize(mask, (rgb_image.shape[1], rgb_image.shape[0]), interpolation=cv2.INTER_NEAREST) > 128
                        mask_3ch = (np.stack([active_mask_full] * 3, axis=-1))
                        rgb_image = np.where(mask_3ch, rgb_image, 0)
                    img_tensor = (torch.from_numpy(rgb_image).float().permute(2, 0, 1).unsqueeze(0) / 255.0)
                    with (torch.no_grad(), torch.cuda.amp.autocast(enabled=torch.cuda.is_available())):
                        niqe_raw = float(niqe_metric(img_tensor.to(niqe_metric.device)))
                        niqe_score = max(0, min(100, (main_config.quality_scaling.niqe_offset - niqe_raw) * main_config.quality_scaling.niqe_scale_factor))
                        scores_norm["niqe"] = niqe_score / 100.0
                        self.metrics.niqe_score = float(niqe_score)
                except Exception as e:
                    logger.warning("NIQE calculation failed", extra={'frame': self.frame_number, 'error': e})
                    if torch.cuda.is_available(): torch.cuda.empty_cache()

            if main_config and metrics_to_compute.get('quality'):
                weights = asdict(main_config.quality_weights)
                quality_sum = sum(
                    scores_norm.get(k, 0) * (weights.get(k, 0) / 100.0)
                    for k in scores_norm.keys()
                )
                self.metrics.quality_score = float(quality_sum * 100)
        except Exception as e:
            self.error = f"Quality calc failed: {e}"
            logger.error(f"Frame quality calculation failed for frame {self.frame_number}", exc_info=True)

@dataclass
class Scene:
    scene_id: int
    start_frame: int
    end_frame: int
    status: str = "pending"
    best_frame: int = -1
    bbox: Optional[list] = None
    preview_path: Optional[str] = None
    is_overridden: bool = False
    is_ignored: bool = False
    frames: Dict[int, Frame] = field(default_factory=dict)

    def to_dict(self):
        return {
            "scene_id": self.scene_id,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "status": self.status,
            "best_frame": self.best_frame,
            "bbox": self.bbox,
            "is_overridden": self.is_overridden,
            "is_ignored": self.is_ignored,
        }

class AppUI:
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        # ... other initializations

    def _create_ui(self):
        pass

    def _setup_event_handlers(self):
        pass

    def download_model(self, url: str, dest_path: Path) -> None:
        if dest_path.exists():
            self.logger.info(f"Model already exists at {dest_path}")
            return
        self.logger.info(f"Downloading model from {url} to {dest_path}")
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        req = urllib.request.Request(url, headers={'User-Agent': self.config.models.user_agent})
        
        with urllib.request.urlopen(req) as response, open(dest_path, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)

    def _get_grounding_dino_model(self, gdino_config_path: str, gdino_checkpoint_path: str):
        global _dino_model_cache
        if _dino_model_cache is not None:
            return _dino_model_cache

        cfg_path = Path(gdino_config_path)
        if not cfg_path.is_absolute():
            cfg_path = project_root / cfg_path

        checkpoint_path = Path(gdino_checkpoint_path)
        if not checkpoint_path.exists():
            self.download_model(self.config.models.grounding_dino, checkpoint_path)

        model = gdino_load_model(model_config_path=str(cfg_path), model_checkpoint_path=str(checkpoint_path), device="cpu")
        _dino_model_cache = model
        return model

    def apply_all_filters_vectorized(self, all_frames_data, filters, config):
        if not all_frames_data:
            return [], [], {}, {}

        df = pd.DataFrame(all_frames_data)
        metrics_df = pd.json_normalize(df['metrics'])
        df = pd.concat([df.drop('metrics', axis=1), metrics_df], axis=1)

        initial_frames = set(df['filename'])
        rejection_reasons = {fname: [] for fname in initial_frames}
        rejected_filenames = set()
        per_metric_counts = {}

        if filters.get("face_sim_enabled"):
            min_val = filters.get("face_sim_min", 0.0)
            rejected_df = df[df['face_sim'] < min_val]
            for _, row in rejected_df.iterrows():
                rejected_filenames.add(row['filename'])
                rejection_reasons[row['filename']].append('face_sim')

        if filters.get("mask_area_enabled"):
            min_val = filters.get("mask_area_pct_min", 0.0)
            eligible_df = df[~df['filename'].isin(rejected_filenames)]
            rejected_df = eligible_df[eligible_df['mask_area_pct'] < min_val]
            for _, row in rejected_df.iterrows():
                rejected_filenames.add(row['filename'])
                rejection_reasons[row['filename']].append('mask_area_pct')

        kept_df = df[~df['filename'].isin(rejected_filenames)]
        
        kept_frames = kept_df.to_dict('records')
        rejected_frames = df[df['filename'].isin(rejected_filenames)].to_dict('records')

        # ... (rest of the logic)

        return kept_frames, rejected_frames, rejection_reasons, per_metric_counts


    def get_scene_status_text(self, scenes):
        if not scenes:
            return "No scenes loaded.", "No scenes have been loaded or defined yet."
        
        total_scenes = len(scenes)
        included_scenes = sum(1 for s in scenes if s.get('status') == 'included')
        excluded_scenes = sum(1 for s in scenes if s.get('status') == 'excluded')
        pending_scenes = total_scenes - included_scenes - excluded_scenes

        status_str = f"{included_scenes}/{total_scenes} scenes included"
        tooltip_str = f"Included: {included_scenes}, Excluded: {excluded_scenes}, Pending: {pending_scenes}"

        return status_str, tooltip_str
