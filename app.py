# keep app.py Monolithic!
"""
Frame Extractor & Analyzer v2.0
"""
# --- [1] IMPORTS ---

# Standard Library
import contextlib
import cv2
from datetime import datetime
import functools
import gc
import hashlib
import io
import json
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
import urllib.request
from collections import Counter, OrderedDict, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum
from functools import lru_cache
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Callable, Dict, Generator, List, Optional, Union

# Third-party Libraries
import gradio as gr
import imagehash
import lpips
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import mediapipe as mp
import numpy as np
import pyiqa
import torch
import yt_dlp as ytdlp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from numba import njit
from PIL import Image
from pydantic import BaseModel, Field, model_validator, field_validator, ConfigDict
from pydantic_settings import BaseSettings, SettingsConfigDict
from scenedetect import detect, ContentDetector
from skimage.metrics import structural_similarity as ssim
from sklearn.cluster import DBSCAN
from torchvision import transforms
from torchvision.ops import box_convert
from ultralytics import YOLO

# --- Hugging Face Cache Setup ---
hf_home = Path(__file__).parent / 'models' / 'huggingface'
hf_home.mkdir(parents=True, exist_ok=True)
os.environ['HF_HOME'] = str(hf_home.resolve())

# Local Submodules
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'DAM4SAM'))
from DAM4SAM.dam4sam_tracker import DAM4SAMTracker
from DAM4SAM.utils import utils as dam_utils
from groundingdino.util.inference import (
    load_model as gdino_load_model,
    predict as gdino_predict
)

# --- [2] CONFIGURATION ---

def json_config_settings_source() -> Dict[str, Any]:
    """A simple settings source that loads variables from a JSON file."""
    try:
        config_path = "config.json"
        if Path(config_path).is_file():
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    return {}

class Config(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix='APP_',
        env_nested_delimiter='_',
        case_sensitive=False
    )
    # Paths
    logs_dir: Path = Path("logs")
    models_dir: Path = Path("models")
    downloads_dir: Path = Path("downloads")
    grounding_dino_config_path: str = "GroundingDINO_SwinT_OGC.py"
    grounding_dino_checkpoint_path: str = "models/groundingdino_swint_ogc.pth"
    # Models
    user_agent: str = "Mozilla/5.0"
    grounding_dino_url: str = "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
    grounding_dino_sha256: str = "3b3ca2563c77c69f651d7bd133e97139c186df06231157a64c507099c52bc799"
    face_landmarker_url: str = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    face_landmarker_sha256: str = "9c899f78b8f2a0b1b117b3554b5f903e481b67f1390f7716e2a537f8842c0c7a"
    dam4sam_model_urls: Dict[str, str] = Field(default_factory=lambda: {
        "sam21pp-T": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt",
        "sam21pp-S": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt",
        "sam21pp-B+": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt",
        "sam21pp-L": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
    })
    dam4sam_sha256: Dict[str, str] = Field(default_factory=lambda: {
        "sam21pp-T": "7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69",
        "sam21pp-S": "6d1aa6f30de5c92224f8172114de081d104bbd23dd9dc5c58996f0cad5dc4d38",
        "sam21pp-B+": "a2345aede8715ab1d5d31b4a509fb160c5a4af1970f199d9054ccfb746c004c5",
        "sam21pp-L": "2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318",
    })
    yolo_url: str = "https://huggingface.co/Ultralytics/YOLOv5/resolve/main/"
    # YouTube-DL
    ytdl_output_template: str = "%(id)s_%(title).40s_%(height)sp.%(ext)s"
    ytdl_format_string: str = "bestvideo[height<={max_res}][ext=mp4]+bestaudio[ext=m4a]/best[height<={max_res}][ext=mp4]/best"
    # FFmpeg
    ffmpeg_log_level: str = "info"
    ffmpeg_thumbnail_quality: int = 80
    ffmpeg_scene_threshold: float = 0.4
    # Cache
    cache_size: int = 200
    cache_eviction_factor: float = 0.2
    cache_cleanup_threshold: float = 0.8
    # Retry
    retry_max_attempts: int = 3
    retry_backoff_seconds: List[float] = Field(default_factory=lambda: [1, 5, 15])
    # Quality Scaling
    quality_entropy_normalization: float = 8.0
    quality_resolution_denominator: int = 500000
    quality_contrast_clamp: float = 2.0
    quality_niqe_offset: float = 10.0
    quality_niqe_scale_factor: float = 10.0
    # Masking
    masking_keep_largest_only: bool = True
    masking_close_kernel_size: int = 5
    masking_open_kernel_size: int = 5
    # UI Defaults
    default_thumbnails_only: bool = True
    default_thumb_megapixels: float = 0.5
    default_scene_detect: bool = True
    default_max_resolution: str = "maximum available"
    default_pre_analysis_enabled: bool = True
    default_pre_sample_nth: int = 5
    default_enable_face_filter: bool = True
    default_face_model_name: str = "buffalo_l"
    default_enable_subject_mask: bool = True
    default_dam4sam_model_name: str = "sam21pp-L"
    default_person_detector_model: str = "yolo11x.pt"
    default_primary_seed_strategy: str = "🤖 Automatic"
    default_seed_strategy: str = "Largest Person"
    default_text_prompt: str = "a person"
    default_resume: bool = False
    default_require_face_match: bool = False
    default_enable_dedup: bool = True
    default_dedup_thresh: int = 5
    default_method: str = "all"
    default_interval: float = 5.0
    default_nth_frame: int = 5
    default_disable_parallel: bool = False
    # Filter Defaults (Grouped)
    filter_defaults: Dict[str, Dict[str, Any]] = Field(default_factory=lambda: {
        'quality_score': {'min': 0.0, 'max': 100.0, 'step': 0.5, 'default_min': 0.0, 'default_max': 100.0},
        'sharpness': {'min': 0.0, 'max': 100.0, 'step': 0.5, 'default_min': 0.0, 'default_max': 100.0},
        'edge_strength': {'min': 0.0, 'max': 100.0, 'step': 0.5, 'default_min': 0.0, 'default_max': 100.0},
        'contrast': {'min': 0.0, 'max': 100.0, 'step': 0.5, 'default_min': 0.0, 'default_max': 100.0},
        'brightness': {'min': 0.0, 'max': 100.0, 'step': 0.5, 'default_min': 0.0, 'default_max': 100.0},
        'entropy': {'min': 0.0, 'max': 100.0, 'step': 0.5, 'default_min': 0.0, 'default_max': 100.0},
        'niqe': {'min': 0.0, 'max': 100.0, 'step': 0.5, 'default_min': 0.0, 'default_max': 100.0},
        'face_sim': {'min': 0.0, 'max': 1.0, 'step': 0.01, 'default_min': 0.0},
        'mask_area_pct': {'min': 0.0, 'max': 100.0, 'step': 0.1, 'default_min': 1.0},
        'dedup_thresh': {'min': -1, 'max': 32, 'step': 1, 'default': -1},
        'eyes_open': {'min': 0.0, 'max': 1.0, 'step': 0.01, 'default_min': 0.0},
        'yaw': {'min': -180.0, 'max': 180.0, 'step': 1, 'default_min': -25, 'default_max': 25},
        'pitch': {'min': -180.0, 'max': 180.0, 'step': 1, 'default_min': -25, 'default_max': 25},
    })
    # Quality Weights (Grouped)
    quality_weights: Dict[str, int] = Field(default_factory=lambda: {
        'sharpness': 25, 'edge_strength': 15, 'contrast': 15,
        'brightness': 10, 'entropy': 15, 'niqe': 20,
    })
    # GroundingDINO Params
    gdino_box_threshold: float = 0.35
    gdino_text_threshold: float = 0.25
    # Person Detector Config
    person_detector_model: str = "yolo11x.pt"
    person_detector_imgsz: int = 640
    person_detector_conf: float = 0.3
    # Logging Config
    log_level: str = "INFO"
    log_format: str = '%(asctime)s | %(levelname)8s | %(name)s | %(message)s'
    log_colored: bool = True
    log_structured_path: str = "structured_log.jsonl"
    # Monitoring
    monitoring_memory_warning_threshold_mb: int = 8192
    monitoring_memory_critical_threshold_mb: int = 16384
    monitoring_cpu_warning_threshold_percent: float = 90.0
    monitoring_gpu_memory_warning_threshold_percent: int = 90
    monitoring_memory_limit_mb: int = 8192
    # Export Options
    export_enable_crop: bool = True
    export_crop_padding: int = 1
    export_crop_ars: str = "16:9,1:1,9:16"
    # Gradio Defaults
    gradio_auto_pctl_input: int = 25
    gradio_show_mask_overlay: bool = True
    gradio_overlay_alpha: float = 0.6
    # Seeding Defaults
    seeding_face_similarity_threshold: float = 0.4
    seeding_yolo_iou_threshold: float = 0.3
    seeding_face_contain_score: int = 100
    seeding_confidence_score_multiplier: int = 20
    seeding_iou_bonus: int = 50
    seeding_face_to_body_expansion_factors: List[float] = Field(default_factory=lambda: [4.0, 7.0, 0.75])
    seeding_final_fallback_box: List[float] = Field(default_factory=lambda: [0.25, 0.25, 0.5, 0.5])
    seeding_balanced_score_weights: Dict[str, float] = Field(default_factory=lambda: {'area': 0.4, 'confidence': 0.4, 'edge': 0.2})
    # Utility Defaults
    utility_max_filename_length: int = 50
    utility_video_extensions: List[str] = Field(default_factory=lambda: ['.mp4','.mov','.mkv','.avi','.webm'])
    utility_image_extensions: List[str] = Field(default_factory=lambda: ['.png','.jpg','.jpeg','.webp','.bmp'])
    # PostProcessing
    postprocessing_mask_fill_kernel_size: int = 5
    # Visualization
    visualization_bbox_color: List[int] = Field(default_factory=lambda: [255, 0, 0])
    visualization_bbox_thickness: int = 2
    # Analysis
    analysis_default_batch_size: int = 32
    analysis_default_workers: int = 4
    # Model Defaults
    model_face_analyzer_det_size: List[int] = Field(default_factory=lambda: [640, 640])
    sharpness_base_scale: int = 2500
    edge_strength_base_scale: int = 100
    min_mask_area_pct: float = 1.0

    def model_post_init(self, __context: Any) -> None:
        self._validate_paths()

    @model_validator(mode='after')
    def _validate_config(self):
        if sum(self.quality_weights.values()) == 0:
            raise ValueError("The sum of quality_weights cannot be zero.")
        return self

    def _validate_paths(self):
        required_dirs = [self.logs_dir, self.models_dir, self.downloads_dir]
        for dir_path in required_dirs:
            if not dir_path.exists():
                dir_path.mkdir(parents=True, exist_ok=True)
            if not os.access(dir_path, os.W_OK):
                raise PermissionError(f"No write permission for: {dir_path}")

    def save_config(self, path: str):
        with open(path, 'w', encoding='utf-8') as f:
            f.write(self.model_dump_json(indent=2))

    @classmethod
    def settings_customise_sources(
        cls, settings_cls, init_settings, env_settings,
        dotenv_settings, file_secret_settings,
    ):
        return (
            init_settings, env_settings, dotenv_settings,
            json_config_settings_source, file_secret_settings,
        )

# --- [3] LOGGING ---

SUCCESS_LEVEL_NUM = 25
logging.addLevelName(SUCCESS_LEVEL_NUM, "SUCCESS")

class LogEvent(BaseModel):
    timestamp: str
    level: str
    message: str
    component: str
    operation: Optional[str] = None
    duration_ms: Optional[float] = None
    error_type: Optional[str] = None
    stack_trace: Optional[str] = None
    user_context: Optional[Dict[str, Any]] = None
    performance_metrics: Optional[Dict[str, Any]] = None
    custom_fields: Optional[Dict[str, Any]] = None

class AppLogger:
    def __init__(self, config: 'Config', log_dir: Optional[Path] = None,
                 log_to_file: bool = True, log_to_console: bool = True):
        self.config = config
        self.log_dir = log_dir or Path(self.config.logs_dir)
        self.log_dir.mkdir(exist_ok=True, parents=True)
        self.progress_queue = None
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_log_file = self.log_dir / f"session_{self.session_id}.log"
        self.structured_log_file = self.log_dir / self.config.log_structured_path
        self.logger = logging.getLogger(f'enhanced_logger_{self.session_id}')
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False
        self.logger.handlers.clear()
        if log_to_console and self.config.log_colored:
            self._setup_console_handler()
        if log_to_file:
            self._setup_file_handlers()
        self._operation_stack: List[Dict[str, Any]] = []

    def _setup_console_handler(self):
        console_handler = logging.StreamHandler()
        console_formatter = ColoredFormatter(self.config.log_format)
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(self.config.log_level)
        self.logger.addHandler(console_handler)

    def _setup_file_handlers(self):
        file_handler = logging.FileHandler(self.session_log_file, encoding='utf-8')
        file_formatter = logging.Formatter(self.config.log_format)
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.DEBUG)
        self.logger.addHandler(file_handler)
        structured_handler = logging.FileHandler(self.structured_log_file, encoding='utf-8')
        structured_formatter = JsonFormatter()
        structured_handler.setFormatter(structured_formatter)
        structured_handler.setLevel(logging.DEBUG)
        self.logger.addHandler(structured_handler)

    def set_progress_queue(self, queue: Queue):
        self.progress_queue = queue

    @contextlib.contextmanager
    def operation(self, name: str, component: str = "system", tracker: Optional['AdvancedProgressTracker'] = None):
        t0 = time.time()
        if tracker:
            tracker.set_stage(name)
        self.info(f"Start {name}", component=component)
        try:
            yield
        except Exception as e:
            if tracker:
                tracker.set_stage(f"{name}: Failed", substage=str(e))
            self.error(f"Failed {name}", component=component, stack_trace=traceback.format_exc())
            raise
        finally:
            duration = (time.time() - t0) * 1000
            if tracker and not self.cancel_event.is_set():
                tracker.done_stage(f"{name} complete")
            self.success(f"Done {name} in {duration:.0f}ms", component=component)

    def _create_log_event(self, level: str, message: str, component: str, **kwargs) -> LogEvent:
        exc_info = kwargs.pop('exc_info', None)
        extra = kwargs.pop('extra', None)
        if 'stacktrace' in kwargs:
            kwargs['stack_trace'] = kwargs.pop('stacktrace')
        if exc_info: kwargs['stack_trace'] = traceback.format_exc()
        if extra:
            kwargs['custom_fields'] = kwargs.get('custom_fields', {})
            kwargs['custom_fields'].update(extra)
        return LogEvent(timestamp=datetime.now().isoformat(), level=level, message=message, component=component, **kwargs)

    def _log_event(self, event: LogEvent):
        log_level_name = event.level.upper()
        log_level = getattr(logging, log_level_name, logging.INFO)
        if log_level_name == "SUCCESS":
            log_level = SUCCESS_LEVEL_NUM
        extra_info = f" [{event.component}]"
        if event.operation: extra_info += f" [{event.operation}]"
        if event.duration_ms: extra_info += f" ({event.duration_ms:.1f}ms)"
        log_message = f"{event.message}{extra_info}"
        if event.stack_trace:
            log_message += f"\n{event.stack_trace}"
        self.logger.log(log_level, log_message, extra={'log_event': event})
        if self.progress_queue:
            ui_message = f"[{event.level}] {event.message}"
            if event.operation: ui_message = f"[{event.operation}] {ui_message}"
            self.progress_queue.put({"log": ui_message})

    def debug(self, message: str, component: str = "system", **kwargs): self._log_event(self._create_log_event("DEBUG", message, component, **kwargs))
    def info(self, message: str, component: str = "system", **kwargs): self._log_event(self._create_log_event("INFO", message, component, **kwargs))
    def warning(self, message: str, component: str = "system", **kwargs): self._log_event(self._create_log_event("WARNING", message, component, **kwargs))
    def error(self, message: str, component: str = "system", **kwargs): self._log_event(self._create_log_event("ERROR", message, component, **kwargs))
    def success(self, message: str, component: str = "system", **kwargs): self._log_event(self._create_log_event("SUCCESS", message, component, **kwargs))
    def critical(self, message: str, component: str = "system", **kwargs): self._log_event(self._create_log_event("CRITICAL", message, component, **kwargs))

class ColoredFormatter(logging.Formatter):
    COLORS = {'DEBUG': '\033[36m', 'INFO': '\033[37m', 'WARNING': '\033[33m',
              'ERROR': '\033[31m', 'CRITICAL': '\033[35m', 'SUCCESS': '\033[32m', 'RESET': '\033[0m'}

    def format(self, record: logging.LogRecord) -> str:
        original_levelname = record.levelname
        try:
            color = self.COLORS.get(original_levelname, self.COLORS['RESET'])
            record.levelname = f"{color}{original_levelname}{self.COLORS['RESET']}"
            return super().format(record)
        finally:
            record.levelname = original_levelname

class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_event_obj = getattr(record, 'log_event', None)
        if isinstance(log_event_obj, LogEvent):
            log_dict = log_event_obj.model_dump(exclude_none=True)
        else:
            log_dict = {
                'timestamp': self.formatTime(record, self.datefmt),
                'level': record.levelname, 'message': record.getMessage(),
                'component': record.name,
            }
            if record.exc_info:
                log_dict['stack_trace'] = self.formatException(record.exc_info)
        return json.dumps(log_dict, default=str, ensure_ascii=False)

# --- [4] ERROR HANDLING & EVENTS ---

class ErrorSeverity(Enum):
    LOW, MEDIUM, HIGH, CRITICAL = "low", "medium", "high", "critical"

class RecoveryStrategy(Enum):
    RETRY, FALLBACK, SKIP, ABORT = "retry", "fallback", "skip", "abort"

class ErrorHandler:
    def __init__(self, logger: 'AppLogger', max_attempts: int, backoff_seconds: list):
        self.logger, self.max_attempts, self.backoff_seconds = logger, max_attempts, backoff_seconds
        self.error_count, self.recovery_attempts = 0, {}

    def with_retry(self, max_attempts: Optional[int] = None, backoff_seconds: Optional[list] = None, recoverable_exceptions: tuple = (Exception,)):
        max_attempts, backoff_seconds = max_attempts or self.max_attempts, backoff_seconds or self.backoff_seconds
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs) -> Any:
                last_exception = None
                for attempt in range(max_attempts):
                    try: return func(*args, **kwargs)
                    except recoverable_exceptions as e:
                        last_exception = e
                        if attempt < max_attempts - 1:
                            sleep_time = backoff_seconds[min(attempt, len(backoff_seconds) - 1)]
                            self.logger.warning(f"Attempt {attempt + 1} failed, retrying in {sleep_time}s: {str(e)}", component="error_handler", custom_fields={'function': func.__name__, 'attempt': attempt + 1, 'max_attempts': max_attempts, 'retry_delay': sleep_time})
                            time.sleep(sleep_time)
                        else: self.logger.error(f"All retry attempts failed for {func.__name__}: {str(e)}", component="error_handler", error_type=type(e).__name__, stack_trace=traceback.format_exc(), custom_fields={'function': func.__name__, 'total_attempts': max_attempts})
                raise last_exception
            return wrapper
        return decorator

    def with_fallback(self, fallback_func: Callable):
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs) -> Any:
                try: return func(*args, **kwargs)
                except Exception as e:
                    self.logger.warning(f"Primary function {func.__name__} failed, using fallback: {str(e)}", component="error_handler", error_type=type(e).__name__, stack_trace=traceback.format_exc(), custom_fields={'primary_function': func.__name__, 'fallback_function': fallback_func.__name__})
                    try: return fallback_func(*args, **kwargs)
                    except Exception as fallback_error:
                        self.logger.error(f"Both primary and fallback functions failed for {func.__name__}", component="error_handler", error_type=type(fallback_error).__name__, stack_trace=traceback.format_exc(), custom_fields={'primary_function': func.__name__, 'fallback_function': fallback_func.__name__, 'primary_error': str(e), 'fallback_error': str(fallback_error)})
                        raise fallback_error
            return wrapper
        return decorator

class UIEvent(BaseModel):
    model_config = ConfigDict(validate_assignment=True, extra='ignore', str_strip_whitespace=True, arbitrary_types_allowed=True)

class ProgressEvent(BaseModel):
    stage: str; substage: Optional[str] = None; done: int = 0; total: int = 1; fraction: float = 0.0
    eta_seconds: Optional[float] = None; eta_formatted: str = "—"

class ExtractionEvent(UIEvent):
    source_path: str; upload_video: Optional[str] = None; method: str; interval: str
    nth_frame: str; max_resolution: str; thumbnails_only: bool; thumb_megapixels: float
    scene_detect: bool; output_folder: Optional[str] = None

class PreAnalysisEvent(UIEvent):
    output_folder: str; video_path: str
    @field_validator('face_ref_img_path')
    @classmethod
    def validate_face_ref(cls, v: str, info) -> str:
        if not v: return ""
        video_path = info.data.get('video_path', '');
        if v == video_path: return ""
        p = Path(v)
        valid_exts = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}
        if not p.is_file() or p.suffix.lower() not in valid_exts: return ""
        return v
    resume: bool; enable_face_filter: bool; face_ref_img_path: str; face_ref_img_upload: Optional[str]
    face_model_name: str; enable_subject_mask: bool; dam4sam_model_name: str; person_detector_model: str
    best_frame_strategy: str; scene_detect: bool; text_prompt: str; box_threshold: float; text_threshold: float
    min_mask_area_pct: float; sharpness_base_scale: float; edge_strength_base_scale: float
    gdino_config_path: str; gdino_checkpoint_path: str; pre_analysis_enabled: bool; pre_sample_nth: int
    primary_seed_strategy: str; compute_quality_score: bool = True; compute_sharpness: bool = True
    compute_edge_strength: bool = True; compute_contrast: bool = True; compute_brightness: bool = True
    compute_entropy: bool = True; compute_eyes_open: bool = True; compute_yaw: bool = True
    compute_pitch: bool = True; compute_face_sim: bool = True; compute_subject_mask_area: bool = True
    compute_niqe: bool = True; compute_phash: bool = True
    @model_validator(mode='after')
    def validate_strategy_consistency(self) -> 'PreAnalysisEvent':
        if not self.face_ref_img_path and self.enable_face_filter:
            self.enable_face_filter = False
        return self

class PropagationEvent(UIEvent):
    output_folder: str; video_path: str; scenes: list[dict[str, Any]]; analysis_params: PreAnalysisEvent

class FilterEvent(UIEvent):
    all_frames_data: list[dict[str, Any]]; per_metric_values: dict[str, Any]; output_dir: str
    gallery_view: str; show_overlay: bool; overlay_alpha: float; require_face_match: bool
    dedup_thresh: int; slider_values: dict[str, float]; dedup_method: str

class ExportEvent(UIEvent):
    all_frames_data: list[dict[str, Any]]; output_dir: str; video_path: str
    enable_crop: bool; crop_ars: str; crop_padding: int; filter_args: dict[str, Any]

class SessionLoadEvent(UIEvent):
    session_path: str

# --- [5] CORE DATA MODELS ---

class FrameMetrics(BaseModel):
    quality_score: float = 0.0; sharpness_score: float = 0.0; edge_strength_score: float = 0.0
    contrast_score: float = 0.0; brightness_score: float = 0.0; entropy_score: float = 0.0
    niqe_score: float = 0.0; eyes_open: float = 0.0; blink_prob: float = 0.0
    yaw: float = 0.0; pitch: float = 0.0; roll: float = 0.0

class Frame(BaseModel):
    image_data: np.ndarray; frame_number: int
    metrics: FrameMetrics = Field(default_factory=FrameMetrics)
    face_similarity_score: Optional[float] = None
    max_face_confidence: Optional[float] = None; error: Optional[str] = None
    model_config = ConfigDict(arbitrary_types_allowed=True)
    def calculate_quality_metrics(self, thumb_image_rgb: np.ndarray, quality_config: 'QualityConfig', logger: 'AppLogger',
                                  mask: Optional[np.ndarray] = None, niqe_metric: Optional[Callable] = None,
                                  main_config: Optional['Config'] = None, face_landmarker: Optional[Callable] = None,
                                  face_bbox: Optional[List[int]] = None,
                                  metrics_to_compute: Optional[Dict[str, bool]] = None):
        try:
            if metrics_to_compute is None: metrics_to_compute = {k: True for k in ['eyes_open', 'yaw', 'pitch', 'sharpness', 'edge_strength', 'contrast', 'brightness', 'entropy', 'quality']}
            if face_landmarker and any(metrics_to_compute.get(k) for k in ['eyes_open', 'yaw', 'pitch']):
                face_img = thumb_image_rgb[face_bbox[1]:face_bbox[3], face_bbox[0]:face_bbox[2]] if face_bbox else thumb_image_rgb
                if not face_img.flags['C_CONTIGUOUS']: face_img = np.ascontiguousarray(face_img, dtype=np.uint8)
                if face_img.dtype != np.uint8: face_img = face_img.astype(np.uint8)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=face_img)
                landmarker_result = face_landmarker.detect(mp_image)
                if landmarker_result.face_blendshapes:
                    blendshapes = {b.category_name: b.score for b in landmarker_result.face_blendshapes[0]}
                    if metrics_to_compute.get('eyes_open'):
                        self.metrics.eyes_open = 1.0 - max(blendshapes.get('eyeBlinkLeft', 0), blendshapes.get('eyeBlinkRight', 0))
                        self.metrics.blink_prob = max(blendshapes.get('eyeBlinkLeft', 0), blendshapes.get('eyeBlinkRight', 0))
                if landmarker_result.facial_transformation_matrixes and any(metrics_to_compute.get(k) for k in ['yaw', 'pitch']):
                    matrix, sy = landmarker_result.facial_transformation_matrixes[0], math.sqrt(matrix[0, 0]**2 + matrix[1, 0]**2)
                    if not sy < 1e-6:
                        if metrics_to_compute.get('pitch'): self.metrics.pitch = math.degrees(math.atan2(-matrix[2, 0], sy))
                        if metrics_to_compute.get('yaw'): self.metrics.yaw = math.degrees(math.atan2(matrix[1, 0], matrix[0, 0]))
                        self.metrics.roll = math.degrees(math.atan2(matrix[2, 1], matrix[2, 2]))
                    else:
                        if metrics_to_compute.get('pitch'): self.metrics.pitch = math.degrees(math.atan2(-matrix[2, 0], sy))
                        if metrics_to_compute.get('yaw'): self.metrics.yaw = 0
                        self.metrics.roll = 0
            scores_norm, gray = {}, cv2.cvtColor(thumb_image_rgb, cv2.COLOR_RGB2GRAY)
            active_mask = ((mask > 128) if mask is not None and mask.ndim == 2 else None)
            if active_mask is not None and np.sum(active_mask) < 100: active_mask = None
            def _calculate_and_store_score(name, value):
                normalized_value = min(max(value, 0.0), 1.0)
                scores_norm[name], setattr(self.metrics, f"{name}_score", float(normalized_value * 100))
            if metrics_to_compute.get('sharpness'):
                lap = cv2.Laplacian(gray, cv2.CV_64F)
                masked_lap = lap[active_mask] if active_mask is not None else lap
                sharpness = np.var(masked_lap) if masked_lap.size > 0 else 0
                sharpness_scaled = (sharpness / (quality_config.sharpness_base_scale * (gray.size / main_config.quality_resolution_denominator)))
                _calculate_and_store_score("sharpness", sharpness_scaled)
            if metrics_to_compute.get('edge_strength'):
                sobelx, sobely = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3), cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                edge_strength = np.mean(np.sqrt(sobelx**2 + sobely**2))
                edge_strength_scaled = (edge_strength / (quality_config.edge_strength_base_scale * (gray.size / main_config.quality_resolution_denominator)))
                _calculate_and_store_score("edge_strength", edge_strength_scaled)
            if metrics_to_compute.get('contrast') or metrics_to_compute.get('brightness'):
                pixels = gray[active_mask] if active_mask is not None else gray
                mean_br, std_br = (np.mean(pixels), np.std(pixels)) if pixels.size > 0 else (0, 0)
                if metrics_to_compute.get('brightness'): _calculate_and_store_score("brightness", mean_br / 255.0)
                if metrics_to_compute.get('contrast'):
                    contrast = std_br / (mean_br + 1e-7)
                    _calculate_and_store_score("contrast", min(contrast, main_config.quality_contrast_clamp) / main_config.quality_contrast_clamp)
            if metrics_to_compute.get('entropy'):
                gray_full = cv2.cvtColor(self.image_data, cv2.COLOR_RGB2GRAY)
                active_mask_full = cv2.resize(mask, (gray_full.shape[1], gray_full.shape[0]), interpolation=cv2.INTER_NEAREST) > 128 if mask is not None else None
                hist = cv2.calcHist([gray_full], [0], active_mask_full, [256], [0, 256]).flatten()
                entropy = compute_entropy(hist, main_config.quality_entropy_normalization)
                scores_norm["entropy"], self.metrics.entropy_score = entropy, float(entropy * 100)
            if quality_config.enable_niqe and niqe_metric is not None:
                try:
                    rgb_image = self.image_data
                    if active_mask is not None:
                        mask_3ch = np.stack([cv2.resize(mask, (rgb_image.shape[1], rgb_image.shape[0])) > 128] * 3, axis=-1)
                        rgb_image = np.where(mask_3ch, rgb_image, 0)
                    img_tensor = (torch.from_numpy(rgb_image).float().permute(2, 0, 1).unsqueeze(0) / 255.0)
                    with (torch.no_grad(), torch.amp.autocast('cuda', enabled=niqe_metric.device.type == 'cuda')):
                        niqe_raw = float(niqe_metric(img_tensor.to(niqe_metric.device)))
                        niqe_score = max(0, min(100, (main_config.quality_niqe_offset - niqe_raw) * main_config.quality_niqe_scale_factor))
                        scores_norm["niqe"], self.metrics.niqe_score = niqe_score / 100.0, float(niqe_score)
                except Exception as e:
                    logger.warning("NIQE calculation failed", extra={'frame': self.frame_number, 'error': e})
                    if niqe_metric.device.type == 'cuda': torch.cuda.empty_cache()
            if main_config and metrics_to_compute.get('quality'):
                weights = main_config.quality_weights
                quality_sum = sum(scores_norm.get(k, 0) * (weights.get(k, 0) / 100.0) for k in scores_norm.keys())
                self.metrics.quality_score = float(quality_sum * 100)
        except Exception as e:
            self.error = f"Quality calc failed: {e}"
            logger.error("Frame quality calculation failed", exc_info=True, extra={'frame': self.frame_number})

class Scene(BaseModel):
    shot_id: int; start_frame: int; end_frame: int; status: str = "pending"; best_frame: Optional[int] = None
    seed_metrics: dict = Field(default_factory=dict); seed_frame_idx: Optional[int] = None
    seed_config: dict = Field(default_factory=dict); seed_type: Optional[str] = None
    seed_result: dict = Field(default_factory=dict); preview_path: Optional[str] = None
    manual_status_change: bool = False; is_overridden: bool = False; initial_bbox: Optional[list] = None
    selected_bbox: Optional[list] = None; yolo_detections: List[dict] = Field(default_factory=list)

class AnalysisParameters(BaseModel):
    source_path: str = ""; method: str = ""; interval: float = 0.0; max_resolution: str = ""
    output_folder: str = ""; video_path: str = ""; disable_parallel: bool = False; resume: bool = False
    enable_face_filter: bool = False; face_ref_img_path: str = ""; face_model_name: str = ""
    enable_subject_mask: bool = False; dam4sam_model_name: str = ""; person_detector_model: str = ""
    seed_strategy: str = ""; scene_detect: bool = False; nth_frame: int = 0; require_face_match: bool = False
    text_prompt: str = ""; thumbnails_only: bool = True; thumb_megapixels: float = 0.5
    pre_analysis_enabled: bool = False; pre_sample_nth: int = 1; primary_seed_strategy: str = "🤖 Automatic"
    gdino_config_path: str = ""; gdino_checkpoint_path: str = ""; box_threshold: float = 0.35
    text_threshold: float = 0.25; min_mask_area_pct: float = 1.0; sharpness_base_scale: float = 2500.0
    edge_strength_base_scale: float = 100.0; compute_quality_score: bool = True; compute_sharpness: bool = True
    compute_edge_strength: bool = True; compute_contrast: bool = True; compute_brightness: bool = True
    compute_entropy: bool = True; compute_eyes_open: bool = True; compute_yaw: bool = True; compute_pitch: bool = True
    compute_face_sim: bool = True; compute_subject_mask_area: bool = True; compute_niqe: bool = True
    compute_phash: bool = True; need_masks_now: bool = False

    @model_validator(mode='before')
    @classmethod
    def _coerce_ui_inputs(cls, data: Any) -> Any:
        if isinstance(data, dict):
            for key, value in data.items():
                if key in cls.model_fields:
                    target_type = cls.model_fields[key].annotation
                    if target_type is bool:
                        data[key] = str(value).strip().lower() in {"1", "true", "yes", "on"}
                    elif target_type in (int, float):
                        try: data[key] = target_type(value)
                        except (ValueError, TypeError): pass
        return data

    @model_validator(mode='after')
    def _validate_strategy_consistency(self) -> 'AnalysisParameters':
        if not self.face_ref_img_path and self.enable_face_filter:
            self.enable_face_filter = False
        return self

class MaskingResult(BaseModel):
    mask_path: Optional[str] = None; shot_id: Optional[int] = None; seed_type: Optional[str] = None
    seed_face_sim: Optional[float] = None; mask_area_pct: Optional[float] = None
    mask_empty: bool = True; error: Optional[str] = None

class QualityConfig(BaseModel):
    sharpness_base_scale: float; edge_strength_base_scale: float; enable_niqe: bool = True

# --- [6] PROGRESS TRACKING ---

class AdvancedProgressTracker:
    def __init__(self, progress: Callable, queue: Queue, logger: AppLogger, ui_stage_name: str = ""):
        self.progress, self.queue, self.logger = progress, queue, logger
        self.stage, self.substage = ui_stage_name or "Working", None
        self.total, self.done = 1, 0; self._t0 = time.time(); self._last_ts = self._t0
        self._ema_dt, self._alpha, self._last_update_ts = None, 0.2, 0.0
        self.throttle_interval = 0.1; self.pause_event = threading.Event(); self.pause_event.set()

    def start(self, total_items: int, desc: Optional[str] = None):
        self.total = max(1, int(total_items)); self.done = 0
        if desc: self.stage = desc
        self.substage = None; self._t0, self._last_ts, self._ema_dt = time.time(), time.time(), None; self._overlay(force=True)

    def step(self, n: int = 1, desc: Optional[str] = None, substage: Optional[str] = None):
        self.pause_event.wait(); now = time.time(); dt = now - self._last_ts; self._last_ts = now
        if dt > 0:
            self._ema_dt = self._alpha * (dt / max(1, n)) + (1 - self._alpha) * self._ema_dt if self._ema_dt else dt / max(1, n)
        self.done = min(self.total, self.done + n)
        if desc: self.stage = desc
        if substage is not None: self.substage = substage
        self._overlay()

    def set(self, done: int, desc: Optional[str] = None, substage: Optional[str] = None):
        delta = max(0, done - self.done)
        if delta > 0: self.step(delta, desc=desc, substage=substage)

    def set_stage(self, stage: str, substage: Optional[str] = None):
        self.stage, self.substage = stage, substage; self._overlay(force=True)

    def done_stage(self, final_text: Optional[str] = None):
        self.done = self.total; self._overlay(force=True)
        if final_text: self.logger.info(final_text, component="progress")

    def _overlay(self, force: bool = False):
        now, fraction = time.time(), self.done / max(1, self.total)
        if not force and (now - self._last_update_ts < self.throttle_interval): return
        self._last_update_ts = now; eta_s = self._eta_seconds(); eta_str = self._fmt_eta(eta_s)
        desc_parts = [f"{self.stage} ({self.done}/{self.total})"]
        if self.substage: desc_parts.append(self.substage)
        desc_parts.append(f"ETA {eta_str}")
        if self.progress: self.progress(fraction, desc=" • ".join(desc_parts))
        progress_event = ProgressEvent(stage=self.stage, substage=self.substage, done=self.done, total=self.total, fraction=fraction, eta_seconds=eta_s, eta_formatted=eta_str)
        self.queue.put({"progress": progress_event.model_dump()})

    def _eta_seconds(self) -> Optional[float]:
        return self._ema_dt * max(0, self.total - self.done) if self._ema_dt else None

    @staticmethod
    def _fmt_eta(eta_s: Optional[float]) -> str:
        if eta_s is None: return "—"
        if eta_s < 60: return f"{int(eta_s)}s"
        m, s = divmod(int(eta_s), 60)
        if m < 60: return f"{m}m {s}s"
        h, m = divmod(m, 60); return f"{h}h {m}m"

# --- [7] UTILITIES ---

def handle_common_errors(func: Callable) -> Callable:
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try: return func(*args, **kwargs)
        except FileNotFoundError as e: return {"error": f"File not found: {e}. Please check the path."}
        except (ValueError, TypeError) as e: return {"error": f"Invalid input: {e}. Please check your parameters."}
        except RuntimeError as e: return {"error": "GPU memory full. Try reducing batch size or using CPU mode."} if "CUDA out of memory" in str(e) else {"error": f"Processing error: {e}"}
        except Exception as e: return {"error": f"An unexpected error occurred: {e}"}
    return wrapper

def monitor_memory_usage(logger: 'AppLogger', device: str, threshold_mb: int = 8000):
    if device == 'cuda':
        allocated = torch.cuda.memory_allocated() / 1024**2
        if allocated > threshold_mb:
            logger.warning(f"High GPU memory usage: {allocated:.1f}MB"); torch.cuda.empty_cache()

def validate_video_file(video_path: str):
    path = Path(video_path)
    if not path.exists(): raise FileNotFoundError(f"Video file not found: {video_path}")
    if path.stat().st_size == 0: raise ValueError(f"Video file is empty: {video_path}")
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened(): raise ValueError(f"Could not open video file: {video_path}")
        cap.release()
    except Exception as e: raise ValueError(f"Invalid video file: {e}")

def estimate_totals(params: 'AnalysisParameters', video_info: dict, scenes: Optional[list['Scene']]) -> dict:
    fps, total_frames = max(1, int(video_info.get("fps") or 30)), int(video_info.get("frame_count") or 0)
    method = params.method
    if method == "interval": extraction_total = max(1, int(total_frames / max(0.1, params.interval) / fps))
    elif method == "every_nth_frame": extraction_total = max(1, int(total_frames / max(1, params.nth_frame)))
    elif method == "all": extraction_total = total_frames
    elif method in ("keyframes", "nth_plus_keyframes"): extraction_total = max(1, int(total_frames * 0.15))
    else: extraction_total = total_frames
    pre_analysis_total = max(0, len(scenes or []))
    propagation_total = sum(max(0, sc.end_frame - sc.start_frame + 1) for sc in (scenes or []))
    return {"extraction": extraction_total, "pre_analysis": pre_analysis_total, "propagation": propagation_total}

def sanitize_filename(name: str, config: 'Config', max_length: Optional[int] = None) -> str:
    max_length = max_length or config.utility_max_filename_length
    return re.sub(r'[^\w\-_.]', '_', name)[:max_length]

def _to_json_safe(obj: Any) -> Any:
    if isinstance(obj, dict): return {k: _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [_to_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, (np.bool_,)): return bool(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, Path): return str(obj)
    if isinstance(obj, BaseModel): return obj.model_dump()
    return obj

def _coerce(val: Any, to_type: type) -> Any:
    if to_type is bool: return str(val).strip().lower() in {"1", "true", "yes", "on"} if isinstance(val, bool) is False else val
    if to_type in (int, float):
        try: return to_type(val)
        except (ValueError, TypeError): raise
    return val

@contextlib.contextmanager
def safe_resource_cleanup(device: str):
    try: yield
    finally:
        gc.collect()
        if device == 'cuda': torch.cuda.empty_cache()

def is_image_folder(p: Union[str, Path]) -> bool:
    if not p: return False
    try: return Path(str(p) if not isinstance(p, (str, Path)) else p).is_dir()
    except (TypeError, ValueError): return False

def list_images(p: Union[str, Path], cfg: Config) -> list[Path]:
    p, exts = Path(p), {e.lower() for e in cfg.utility_image_extensions}
    return sorted([f for f in p.iterdir() if f.suffix.lower() in exts and f.is_file()])

@njit
def compute_entropy(hist: np.ndarray, entropy_norm: float) -> float:
    prob = hist / (np.sum(hist) + 1e-7)
    entropy = -np.sum(prob[prob > 0] * np.log2(prob[prob > 0]))
    return min(max(entropy / entropy_norm, 0), 1.0)

# --- [8] CACHING & OPTIMIZATION ---

class ThumbnailManager:
    def __init__(self, logger: 'AppLogger', config: 'Config'):
        self.logger, self.config, self.cache = logger, config, OrderedDict()
        self.max_size = self.config.cache_size
        self.logger.info(f"ThumbnailManager initialized with cache size {self.max_size}")

    def get(self, thumb_path: Path) -> Optional[np.ndarray]:
        thumb_path = Path(thumb_path) if not isinstance(thumb_path, Path) else thumb_path
        if thumb_path in self.cache:
            self.cache.move_to_end(thumb_path); return self.cache[thumb_path]
        if not thumb_path.exists(): return None
        if len(self.cache) > self.max_size * self.config.cache_cleanup_threshold: self._cleanup_old_entries()
        try:
            with Image.open(thumb_path) as pil_thumb: thumb_img = np.array(pil_thumb.convert("RGB"))
            self.cache[thumb_path] = thumb_img
            while len(self.cache) > self.max_size: self.cache.popitem(last=False)
            return thumb_img
        except Exception as e:
            self.logger.warning("Failed to load thumbnail with Pillow", extra={'path': str(thumb_path), 'error': e}); return None

    def clear_cache(self): self.cache.clear(); gc.collect()

    def _cleanup_old_entries(self):
        for _ in range(int(self.max_size * self.config.cache_eviction_factor)):
            if not self.cache: break
            self.cache.popitem(last=False)

# --- [9] MODEL LOADING & MANAGEMENT ---

thread_local = threading.local()

def _compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(8192): h.update(chunk)
    return h.hexdigest()

def download_model(url: str, dest_path: Union[str, Path], description: str, logger: 'AppLogger',
                   error_handler: 'ErrorHandler', user_agent: str, min_size: int = 1_000_000,
                   expected_sha256: Optional[str] = None):
    dest_path = Path(dest_path); dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.is_file():
        if expected_sha256 and _compute_sha256(dest_path) == expected_sha256:
            logger.info(f"Using cached and verified {description}: {dest_path}"); return
        elif not expected_sha256 and (min_size is None or dest_path.stat().st_size >= min_size):
            logger.info(f"Using cached {description} (SHA not verified): {dest_path}"); return
        else: logger.warning(f"Cached {description} has incorrect SHA256. Re-downloading."); dest_path.unlink()

    @error_handler.with_retry(recoverable_exceptions=(urllib.error.URLError, TimeoutError, RuntimeError))
    def download_func():
        logger.info(f"Downloading {description}", extra={'url': url, 'dest': dest_path})
        req = urllib.request.Request(url, headers={"User-Agent": user_agent})
        with urllib.request.urlopen(req, timeout=180) as resp, open(dest_path, "wb") as out: shutil.copyfileobj(resp, out)
        if not dest_path.exists(): raise RuntimeError(f"Download of {description} failed.")
        if expected_sha256 and _compute_sha256(dest_path) != expected_sha256: raise RuntimeError(f"SHA256 mismatch for {description}.")
        if not expected_sha256 and dest_path.stat().st_size < min_size: raise RuntimeError(f"Downloaded {description} seems incomplete.")
        logger.success(f"{description} downloaded and verified successfully.")
    try: download_func()
    except Exception as e:
        logger.error(f"Failed to download {description}", exc_info=True, extra={'url': url})
        if dest_path.exists(): dest_path.unlink()
        raise RuntimeError(f"Failed to download required model: {description}") from e

def get_face_landmarker(model_path: str, logger: 'AppLogger') -> vision.FaceLandmarker:
    if hasattr(thread_local, 'face_landmarker_instance'): return thread_local.face_landmarker_instance
    logger.info("Initializing MediaPipe FaceLandmarker for new thread.", component="face_landmarker")
    try:
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(base_options=base_options, output_face_blendshapes=True, output_facial_transformation_matrixes=True, num_faces=1, min_face_detection_confidence=0.3, min_face_presence_confidence=0.3)
        detector = vision.FaceLandmarker.create_from_options(options)
        thread_local.face_landmarker_instance = detector
        logger.success("Face landmarker model initialized successfully for this thread.")
        return detector
    except Exception as e:
        logger.error(f"Could not initialize MediaPipe face landmarker model. Error: {e}", component="face_landmarker")
        raise RuntimeError("Could not initialize MediaPipe face landmarker model.") from e

@lru_cache(maxsize=2)
def get_face_analyzer(model_name: str, models_path: str, det_size_tuple: tuple, logger: 'AppLogger', device: str = 'cpu') -> 'FaceAnalysis':
    from insightface.app import FaceAnalysis
    logger.info(f"Loading or getting cached face model: {model_name} on device: {device}")
    try:
        is_cuda = device == 'cuda'
        providers = (['CUDAExecutionProvider', 'CPUExecutionProvider'] if is_cuda else ['CPUExecutionProvider'])
        analyzer = FaceAnalysis(name=model_name, root=models_path, providers=providers)
        analyzer.prepare(ctx_id=0 if is_cuda else -1, det_size=det_size_tuple)
        logger.success(f"Face model loaded with {'CUDA' if is_cuda else 'CPU'}.")
        return analyzer
    except Exception as e:
        if "out of memory" in str(e) and device == 'cuda':
            torch.cuda.empty_cache(); logger.warning("CUDA OOM, retrying with CPU...")
            try:
                analyzer = FaceAnalysis(name=model_name, root=models_path, providers=['CPUExecutionProvider'])
                analyzer.prepare(ctx_id=-1, det_size=det_size_tuple); return analyzer
            except Exception as cpu_e: logger.error(f"CPU fallback also failed: {cpu_e}")
        raise RuntimeError(f"Could not initialize face analysis model. Error: {e}") from e

class PersonDetector:
    def __init__(self, logger: 'AppLogger', model_path: Union[Path, str], imgsz: int, conf: float, device: str = 'cuda'):
        self.logger, self.device = logger, device; model_p = Path(model_path)
        model_str_for_yolo = model_p.name if not model_p.exists() else str(model_p)
        if not model_p.exists(): logger.info(f"Local YOLO model not found at '{model_p}'. Attempting to load by name for auto-download.", component="person_detector", extra={'model_name': model_str_for_yolo})
        try:
            self.model = YOLO(model_str_for_yolo); self.model.to(self.device); self.imgsz, self.conf = imgsz, conf
            logger.info("YOLO person detector loaded", component="person_detector", custom_fields={'device': self.device, 'model': model_str_for_yolo})
        except Exception as e: logger.error("Failed to load YOLO model", component="person_detector", exc_info=True); raise e

    def detect_boxes(self, img_rgb: np.ndarray) -> List[dict]:
        res, out = self.model.predict(img_rgb, imgsz=self.imgsz, conf=self.conf, classes=[0], verbose=False), []
        for r in res:
            if getattr(r, "boxes", None) is None: continue
            for b in r.boxes.cpu():
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist()); conf = float(b.conf[0])
                out.append({"bbox": [x1, y1, x2, y2], "conf": conf, "type": "yolo"})
        return out

@lru_cache(maxsize=4)
def get_person_detector(model_path_str: str, device: str, imgsz: int, conf: float, logger: 'AppLogger') -> 'PersonDetector':
    try:
        logger.info(f"Loading YOLO model: {Path(model_path_str).name} (first use)", component="person_detector")
        return PersonDetector(logger=logger, model_path=Path(model_path_str), imgsz=imgsz, conf=conf, device=device)
    except Exception as e:
        logger.warning(f"Primary YOLO detector '{model_path_str}' failed to load, attempting fallback to yolo11s.pt. Error: {e}", component="person_detector")
        return get_person_detector(model_path_str="yolo11s.pt", device=device, imgsz=imgsz, conf=conf, logger=logger)

def resolve_grounding_dino_config(config_path: str) -> str:
    try:
        import importlib.resources as pkg_resources; from groundingdino import config as gdino_config_module
        with pkg_resources.path(gdino_config_module, config_path) as config_file: return str(config_file)
    except (ImportError, ModuleNotFoundError, FileNotFoundError):
        raise RuntimeError(f"Could not resolve GroundingDINO config '{config_path}'.")

@lru_cache(maxsize=2)
def get_grounding_dino_model(gdino_config_path: str, gdino_checkpoint_path: str, models_path: str, grounding_dino_url: str, user_agent: str, retry_params: tuple, device: str, logger: Optional['AppLogger'] = None) -> Optional[torch.nn.Module]:
    _logger = logger or AppLogger(config=Config()); _logger.info("Loading GroundingDINO model (first use)...", component="grounding")
    error_handler = ErrorHandler(_logger, *retry_params)
    try:
        models_dir = Path(models_path); models_dir.mkdir(parents=True, exist_ok=True)
        config_path = resolve_grounding_dino_config(gdino_config_path or Config().paths.grounding_dino_config)
        ckpt_path = Path(gdino_checkpoint_path); ckpt_path = models_dir / ckpt_path.name if not ckpt_path.is_absolute() else ckpt_path
        download_model(grounding_dino_url, ckpt_path, "GroundingDINO Swin-T model", _logger, error_handler, user_agent, expected_sha256=Config().grounding_dino_sha256)
        model = gdino_load_model(model_config_path=config_path, model_checkpoint_path=str(ckpt_path), device=device)
        _logger.success("GroundingDINO model loaded successfully", component="grounding", custom_fields={'model_path': str(ckpt_path)}); return model
    except Exception as e: _logger.error("Grounding DINO model loading failed.", component="grounding", exc_info=True); return None

def predict_grounding_dino(model: torch.nn.Module, image_tensor: torch.Tensor, caption: str, box_threshold: float, text_threshold: float, device: str) -> tuple:
    with torch.no_grad(), torch.amp.autocast('cuda', enabled=(device == 'cuda')):
        return gdino_predict(model=model, image=image_tensor.to(device), caption=caption, box_threshold=float(box_threshold), text_threshold=float(text_threshold))

@lru_cache(maxsize=4)
def get_dam4sam_tracker(model_name: str, models_path: str, model_urls_tuple: tuple, user_agent: str, retry_params: tuple, logger: 'AppLogger', device: str) -> Optional['DAM4SAMTracker']:
    model_urls = dict(model_urls_tuple); selected_name = model_name or Config().default_dam4sam_model_name or next(iter(model_urls.keys()))
    logger.info(f"Loading DAM4SAM model: {selected_name} (first use)", component="dam4sam"); error_handler = ErrorHandler(logger, *retry_params)
    if device != 'cuda': logger.error("DAM4SAM requires CUDA but it's not available."); return None
    try:
        models_dir = Path(models_path); models_dir.mkdir(parents=True, exist_ok=True)
        if selected_name not in model_urls: raise ValueError(f"Unknown DAM4SAM model: {selected_name}")
        url, expected_sha256, checkpoint_path = model_urls[selected_name], Config().dam4sam_sha256.get(selected_name), models_dir / Path(url).name
        download_model(url, checkpoint_path, f"DAM4SAM {selected_name}", logger, error_handler, user_agent, expected_sha256=expected_sha256)
        actual_path, _ = dam_utils.determine_tracker(selected_name)
        if not Path(actual_path).exists(): Path(actual_path).parent.mkdir(exist_ok=True, parents=True); shutil.copy(checkpoint_path, actual_path)
        tracker = DAM4SAMTracker(selected_name); logger.success(f"DAM4SAM tracker {selected_name} initialized.", component="dam4sam"); return tracker
    except Exception as e:
        logger.error(f"Failed to initialize DAM4SAM tracker {selected_name}: {str(e)}", exc_info=True)
        if torch.cuda.is_available(): logger.error(f"CUDA memory: {torch.cuda.memory_allocated()/1024**3:.1f}GB allocated, {torch.cuda.memory_reserved()/1024**3:.1f}GB reserved"); torch.cuda.empty_cache()
        return None

@lru_cache(maxsize=2)
def get_lpips_metric(model_name: str = 'alex', device: str = 'cpu') -> torch.nn.Module:
    return lpips.LPIPS(net=model_name).to(device)

def initialize_analysis_models(params: 'AnalysisParameters', config: 'Config', logger: 'AppLogger') -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"; face_analyzer, ref_emb, person_detector, face_landmarker = None, None, None, None
    if params.primary_seed_strategy == "🧑‍🤝‍🧑 Find Prominent Person":
        person_detector = get_person_detector(model_path_str=str(Path(config.models_dir) / params.person_detector_model), device=device, imgsz=config.person_detector_imgsz, conf=config.person_detector_conf, logger=logger)
        return {"face_analyzer": None, "ref_emb": None, "person_detector": person_detector, "face_landmarker": None, "device": device}
    if params.enable_face_filter:
        face_analyzer = get_face_analyzer(model_name=params.face_model_name, models_path=str(config.models_dir), det_size_tuple=tuple(config.model_face_analyzer_det_size), logger=logger, device=device)
        if face_analyzer and params.face_ref_img_path and Path(params.face_ref_img_path).is_file():
            try:
                ref_img = cv2.imread(str(params.face_ref_img_path))
                if ref_img is not None:
                    faces = face_analyzer.get(ref_img)
                    if faces: ref_emb = max(faces, key=lambda x: x.det_score).normed_embedding; logger.info("Reference face embedding created successfully.")
                    else: logger.warning("No face found in reference image.", extra={'path': params.face_ref_img_path})
                else: logger.warning("Could not read reference face image.", extra={'path': params.face_ref_img_path})
            except Exception as e: logger.error("Failed to process reference face image.", exc_info=True)
        else: logger.warning("Reference face image path does not exist.", extra={'path': params.face_ref_img_path})
    person_detector = get_person_detector(model_path_str=str(Path(config.models_dir) / params.person_detector_model), device=device, imgsz=config.person_detector_imgsz, conf=config.person_detector_conf, logger=logger)
    landmarker_path = Path(config.models_dir) / Path(config.face_landmarker_url).name
    error_handler = ErrorHandler(logger, config.retry_max_attempts, config.retry_backoff_seconds)
    download_model(config.face_landmarker_url, landmarker_path, "MediaPipe Face Landmarker", logger, error_handler, config.user_agent, expected_sha256=config.face_landmarker_sha256)
    if landmarker_path.exists(): face_landmarker = get_face_landmarker(str(landmarker_path), logger)
    return {"face_analyzer": face_analyzer, "ref_emb": ref_emb, "person_detector": person_detector, "face_landmarker": face_landmarker, "device": device}

# --- [10] VIDEO & FRAME PROCESSING ---

class VideoManager:
    def __init__(self, source_path: str, config: 'Config', max_resolution: Optional[str] = None):
        self.source_path, self.config, self.max_resolution = source_path, config, max_resolution or config.default_max_resolution
        self.is_youtube = ("youtube.com/" in source_path or "youtu.be/" in source_path)

    def prepare_video(self, logger: 'AppLogger') -> str:
        if self.is_youtube:
            logger.info("Downloading video", component="video", user_context={'source': self.source_path})
            max_h = None if self.max_resolution == "maximum available" else int(self.max_resolution)
            ydl_opts = {'outtmpl': str(Path(self.config.downloads_dir) / self.config.ytdl_output_template), 'format': self.config.ytdl_format_string.format(max_res=max_h) if max_h else "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best", 'merge_output_format': 'mp4', 'noprogress': True, 'quiet': True}
            try:
                with ytdlp.YoutubeDL(ydl_opts) as ydl: info = ydl.extract_info(self.source_path, download=True); return str(Path(ydl.prepare_filename(info)))
            except ytdlp.utils.DownloadError as e: raise RuntimeError(f"Download failed. Resolution may not be available. Details: {e}") from e
        validate_video_file(self.source_path); return str(self.source_path)

    @staticmethod
    def get_video_info(video_path: str) -> dict:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened(): raise IOError(f"Could not open video: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        if not np.isfinite(fps) or fps <= 0: fps = 30.0
        info = {"width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), "fps": fps, "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}
        cap.release(); return info

def run_scene_detection(video_path: str, output_dir: Path, logger: Optional['AppLogger'] = None) -> list:
    logger = logger or AppLogger(config=Config()); logger.info("Detecting scenes...", component="video")
    try:
        scene_list = detect(str(video_path), ContentDetector())
        shots = ([(s.get_frames(), e.get_frames()) for s, e in scene_list] if scene_list else [])
        with (output_dir / "scenes.json").open('w', encoding='utf-8') as f: json.dump(shots, f)
        logger.success(f"Found {len(shots)} scenes.", component="video"); return shots
    except Exception as e:
        logger.error("Scene detection failed.", component="video", exc_info=True); return []

def make_photo_thumbs(image_paths: list[Path], out_dir: Path, params: 'AnalysisParameters', cfg: 'Config', logger: 'AppLogger', tracker: Optional['AdvancedProgressTracker'] = None) -> dict:
    thumbs_dir = out_dir / "thumbs"; thumbs_dir.mkdir(parents=True, exist_ok=True)
    target_area = params.thumb_megapixels * 1_000_000; frame_map, image_manifest = {}, {}
    if tracker: tracker.start(len(image_paths), desc="Generating thumbnails")
    for i, img_path in enumerate(image_paths, start=1):
        if tracker and tracker.pause_event.is_set(): tracker.step()
        try:
            bgr = cv2.imread(str(img_path))
            if bgr is None: logger.warning(f"Could not read image file: {img_path}"); continue
            h, w = bgr.shape[:2]; scale = math.sqrt(target_area / float(max(1, w * h)))
            if scale < 1.0: bgr = cv2.resize(bgr, (max(2, int((w*scale)//2*2)), max(2, int((h*scale)//2*2))), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB); out_name = f"frame_{i:06d}.webp"; out_path = thumbs_dir / out_name
            Image.fromarray(rgb).save(out_path, format="WEBP", quality=cfg.ffmpeg_thumbnail_quality)
            frame_map[i], image_manifest[i] = out_name, str(img_path.resolve())
        except Exception as e: logger.error(f"Failed to process image {img_path}", exc_info=True)
        finally:
            if tracker: tracker.step()
    (out_dir / "frame_map.json").write_text(json.dumps(frame_map, indent=2), encoding="utf-8")
    (out_dir / "image_manifest.json").write_text(json.dumps(image_manifest, indent=2), encoding="utf-8")
    if tracker: tracker.done_stage("Thumbnails generated")
    return frame_map

def run_ffmpeg_extraction(video_path: str, output_dir: Path, video_info: dict, params: 'AnalysisParameters', progress_queue: Queue, cancel_event: threading.Event, logger: 'AppLogger', config: 'Config', tracker: Optional['AdvancedProgressTracker'] = None):
    cmd_base = ['ffmpeg', '-y', '-i', str(video_path), '-hide_banner', '-progress', 'pipe:1', '-nostats', '-loglevel', 'info']
    thumb_dir = output_dir / "thumbs"; thumb_dir.mkdir(exist_ok=True)
    w, h = video_info.get('width', 1920), video_info.get('height', 1080)
    scale_factor = math.sqrt(params.thumb_megapixels * 1_000_000 / (w * h)) if w * h > 0 else 1.0
    vf_scale = f"scale=w=trunc(iw*{scale_factor}/2)*2:h=trunc(ih*{scale_factor}/2)*2"
    fps, N, interval = max(1, int(video_info.get('fps', 30))), max(1, int(params.nth_frame or 0)), max(0.1, float(params.interval or 0.0))
    select_map = {"keyframes": "select='eq(pict_type,I)'", "every_nth_frame": f"select='not(mod(n,{N}))'", "nth_plus_keyframes": f"select='or(eq(pict_type,I),not(mod(n,{N})))'", "interval": f"fps=1/{interval}", "all": f"fps={fps}"}
    vf_select = select_map.get(params.method, f"fps={fps}")
    vf = f"{vf_select},{vf_scale},showinfo" if params.thumbnails_only else f"{vf_select},showinfo"
    cmd = cmd_base + (["-vf", vf, "-c:v", "libwebp", "-lossless", "0", "-quality", str(config.ffmpeg_thumbnail_quality), "-vsync", "vfr", str(thumb_dir / "frame_%06d.webp")] if params.thumbnails_only else ["-vf", vf, "-c:v", "png", "-vsync", "vfr", str(thumb_dir / "frame_%06d.png")])
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8', bufsize=1)
    frame_map_list = []
    with process.stdout, process.stderr:
        total_duration_s = video_info.get("frame_count", 0) / max(0.01, video_info.get("fps", 30))
        stdout_thread = threading.Thread(target=lambda: _process_ffmpeg_stream(process.stdout, tracker, "Extracting frames", total_duration_s))
        stderr_thread = threading.Thread(target=lambda: frame_map_list.extend(_process_ffmpeg_showinfo(process.stderr)))
        stdout_thread.start(); stderr_thread.start()
        while process.poll() is None:
            if cancel_event.is_set(): process.terminate(); break
            try: process.wait(timeout=0.1)
            except subprocess.TimeoutExpired: continue
        stdout_thread.join(); stderr_thread.join()
    process.wait()
    if frame_map_list: (output_dir / "frame_map.json").write_text(json.dumps(sorted(frame_map_list)), encoding='utf-8')
    if process.returncode not in [0, -9] and not cancel_event.is_set(): raise RuntimeError(f"FFmpeg failed with code {process.returncode}.")

def _process_ffmpeg_stream(stream, tracker: Optional['AdvancedProgressTracker'], desc: str, total_duration_s: float):
    progress_data = {}
    for line in iter(stream.readline, ''):
        try:
            key, value = line.strip().split('=', 1); progress_data[key] = value
            if key == 'progress' and value == 'end':
                if tracker: tracker.set(tracker.total, desc=desc)
                break
            if key == 'out_time_us' and total_duration_s > 0:
                if tracker: tracker.set(int((int(value) / (total_duration_s * 1_000_000)) * tracker.total), desc=desc)
            elif key == 'frame' and tracker and total_duration_s <= 0: tracker.set(int(value), desc=desc)
        except ValueError: pass
    stream.close()

def _process_ffmpeg_showinfo(stream) -> list:
    frame_numbers = []
    for line in iter(stream.readline, ''):
        if match := re.search(r' n:\s*(\d+)', line): frame_numbers.append(int(match.group(1)))
    stream.close(); return frame_numbers

def postprocess_mask(mask: np.ndarray, config: 'Config', fill_holes: bool = True, keep_largest_only: bool = True) -> np.ndarray:
    if mask is None or mask.size == 0: return mask
    binary_mask = (mask > 128).astype(np.uint8)
    if fill_holes: binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (config.masking_close_kernel_size, config.masking_close_kernel_size)))
    if keep_largest_only and config.masking_keep_largest_only:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
        if num_labels > 1: binary_mask = (labels == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
    return (binary_mask * 255).astype(np.uint8)

def render_mask_overlay(frame_rgb: np.ndarray, mask_gray: np.ndarray, alpha: float, logger: 'AppLogger') -> np.ndarray:
    if mask_gray is None or frame_rgb is None: return frame_rgb if frame_rgb is not None else np.array([])
    h, w = frame_rgb.shape[:2]
    if mask_gray.shape[:2] != (h, w): mask_gray = cv2.resize(mask_gray, (w, h), interpolation=cv2.INTER_NEAREST)
    m = (mask_gray > 128)
    red_layer = np.zeros_like(frame_rgb, dtype=np.uint8); red_layer[..., 0] = 255
    blended = cv2.addWeighted(frame_rgb, 1.0 - alpha, red_layer, alpha, 0.0)
    if m.ndim == 2: m = m[..., np.newaxis]
    elif m.ndim == 3 and m.shape[2] != 1: logger.warning("Unexpected mask shape. Skipping overlay.", extra={'shape': m.shape}); return frame_rgb
    return np.where(m, blended, frame_rgb)

def rgb_to_pil(image_rgb: np.ndarray) -> Image.Image: return Image.fromarray(image_rgb)

def create_frame_map(output_dir: Path, logger: 'AppLogger', ext: str = ".webp") -> dict:
    logger.info("Loading frame map...", component="frames"); frame_map_path = output_dir / "frame_map.json"
    try:
        with open(frame_map_path, 'r', encoding='utf-8') as f: frame_map_list = json.load(f)
        return {orig_num: f"frame_{i+1:06d}{ext}" for i, orig_num in enumerate(sorted(map(int, frame_map_list)))}
    except (FileNotFoundError, json.JSONDecodeError, TypeError) as e:
        logger.error(f"Could not load or parse frame_map.json: {e}. Frame mapping will be empty.", exc_info=False); return {}

# --- [11] MASKING & PROPAGATION LOGIC ---

class MaskPropagator:
    # ... (Implementation from original file)
    pass
class SeedSelector:
    # ... (Implementation from original file)
    pass
class SubjectMasker:
    # ... (Implementation from original file)
    pass

# --- [12] FILTERING & SCENE LOGIC ---

def load_and_prep_filter_data(metadata_path: str, get_all_filter_keys: Callable, config: 'Config') -> tuple[list, dict]:
    # ... (Implementation from original file)
    pass
def build_all_metric_svgs(per_metric_values: dict, get_all_filter_keys: Callable, logger: 'AppLogger') -> dict:
    # ... (Implementation from original file)
    pass
def histogram_svg(hist_data: tuple, title: str = "", logger: Optional['AppLogger'] = None) -> str:
    # ... (Implementation from original file)
    pass
def apply_all_filters_vectorized(all_frames_data: list[dict], filters: dict, config: 'Config', thumbnail_manager: Optional['ThumbnailManager'] = None, output_dir: Optional[str] = None) -> tuple[list, list, Counter, dict]:
    # ... (Implementation from original file)
    pass
def on_filters_changed(event: 'FilterEvent', thumbnail_manager: 'ThumbnailManager', config: 'Config', logger: Optional['AppLogger'] = None) -> dict:
    # ... (Implementation from original file)
    pass
def auto_set_thresholds(per_metric_values: dict, p: int, slider_keys: list[str], selected_metrics: list[str]) -> dict:
    # ... (Implementation from original file)
    pass
def save_scene_seeds(scenes_list: list[dict], output_dir_str: str, logger: 'AppLogger'):
    # ... (Implementation from original file)
    pass
def get_scene_status_text(scenes_list: list[dict]) -> tuple[str, gr.update]:
    # ... (Implementation from original file)
    pass
def build_scene_gallery_items(scenes: list[dict], view: str, output_dir: str) -> tuple[list[tuple], list[int]]:
    # ... (Implementation from original file)
    pass
def toggle_scene_status(scenes_list: list[dict], selected_shot_id: int, new_status: str, output_folder: str, logger: 'AppLogger') -> tuple[list, str, str, gr.update]:
    # ... (Implementation from original file)
    pass

# --- [13] PIPELINE CLASSES ---

class Pipeline:
    def __init__(self, config: 'Config', logger: 'AppLogger', params: 'AnalysisParameters', progress_queue: Queue, cancel_event: threading.Event):
        self.config, self.logger, self.params, self.progress_queue, self.cancel_event = config, logger, params, progress_queue, cancel_event

class ExtractionPipeline(Pipeline):
    def run(self, tracker: Optional['AdvancedProgressTracker'] = None) -> dict:
        source_p = Path(self.params.source_path)
        return self._extract_from_image_folder(source_p, tracker) if is_image_folder(source_p) else self._extract_from_video(source_p, tracker)
    def _extract_from_image_folder(self, source_p: Path, tracker: Optional['AdvancedProgressTracker']) -> dict:
        output_dir = Path(self.params.output_folder) if self.params.output_folder else self.config.downloads_dir / source_p.name
        output_dir.mkdir(exist_ok=True, parents=True); self._save_run_config(output_dir, video_path="")
        self.logger.info(f"Processing image folder: {source_p.name}"); images = list_images(source_p, self.config)
        if not images: self.logger.warning("No images found."); return {"done": False, "log": "No images found."}
        make_photo_thumbs(images, output_dir, self.params, self.config, self.logger, tracker=tracker)
        with (output_dir / "scenes.json").open('w', encoding='utf-8') as f: json.dump([[i, i] for i in range(1, len(images) + 1)], f)
        return {"done": True, "output_dir": str(output_dir), "video_path": ""}
    def _extract_from_video(self, source_p: Path, tracker: Optional['AdvancedProgressTracker']) -> dict:
        self.logger.info("Preparing video source...")
        vid_manager = VideoManager(self.params.source_path, self.config, self.params.max_resolution)
        video_path = Path(vid_manager.prepare_video(self.logger))
        output_dir = Path(self.params.output_folder) if self.params.output_folder else self.config.downloads_dir / video_path.stem
        output_dir.mkdir(exist_ok=True, parents=True); self._save_run_config(output_dir, video_path=str(video_path))
        self.logger.info("Video ready", user_context={'path': sanitize_filename(video_path.name, self.config)})
        video_info = VideoManager.get_video_info(video_path)
        if tracker: tracker.start(estimate_totals(self.params, video_info, None)["extraction"], desc="Extracting frames")
        if self.params.scene_detect: self._run_scene_detection(video_path, output_dir)
        self._run_ffmpeg(video_path, output_dir, video_info, tracker=tracker)
        if self.cancel_event.is_set():
            self.logger.info("Extraction cancelled.");
            if tracker: tracker.done_stage("Extraction cancelled");
            return {"done": False, "log": "Extraction cancelled"}
        if tracker: tracker.done_stage("Extraction complete")
        self.logger.success("Extraction complete.");
        return {"done": True, "output_dir": str(output_dir), "video_path": str(video_path)}
    def _save_run_config(self, output_dir: Path, video_path: str):
        params_dict = self.params.model_dump(); params_dict.update({'output_folder': str(output_dir), 'video_path': video_path})
        try: (output_dir / "run_config.json").write_text(json.dumps(_to_json_safe(params_dict), indent=2), encoding='utf-8')
        except OSError as e: self.logger.warning(f"Could not write run config: {e}")
    def _run_scene_detection(self, video_path: str, output_dir: Path) -> list:
        return run_scene_detection(video_path, output_dir, self.logger)
    def _run_ffmpeg(self, video_path: str, output_dir: Path, video_info: dict, tracker: Optional['AdvancedProgressTracker'] = None):
        return run_ffmpeg_extraction(video_path, output_dir, video_info, self.params, self.progress_queue, self.cancel_event, self.logger, self.config, tracker=tracker)

class EnhancedExtractionPipeline(ExtractionPipeline):
    def __init__(self, config: 'Config', logger: 'AppLogger', params: 'AnalysisParameters', progress_queue: Queue, cancel_event: threading.Event):
        super().__init__(config, logger, params, progress_queue, cancel_event)
        self.error_handler = ErrorHandler(self.logger, self.config.retry_max_attempts, self.config.retry_backoff_seconds)
        self.run = self.error_handler.with_retry()(self.run)

class AnalysisPipeline(Pipeline):
    # ... (Implementation from original file)
    pass

# --- [14] PIPELINE EXECUTION FUNCTIONS ---

@handle_common_errors
def execute_extraction(event: 'ExtractionEvent', progress_queue: Queue, cancel_event: threading.Event, logger: 'AppLogger', config: 'Config', **kwargs) -> Generator[dict, None, None]:
    try:
        params_dict = event.model_dump()
        if event.upload_video:
            dest = str(Path(config.downloads_dir) / Path(event.upload_video).name)
            shutil.copy2(params_dict.pop('upload_video'), dest); params_dict['source_path'] = dest
        params = AnalysisParameters.model_validate(params_dict)
        tracker = AdvancedProgressTracker(kwargs.get('progress'), progress_queue, logger, ui_stage_name="Extracting")
        result = EnhancedExtractionPipeline(config, logger, params, progress_queue, cancel_event).run(tracker=tracker)
        if result and result.get("done"):
            yield {"log": "Extraction complete.", "status": f"Output: {result['output_dir']}", "extracted_video_path_state": result.get("video_path", ""), "extracted_frames_dir_state": result["output_dir"], "done": True}
    except Exception as e:
        logger.error("Extraction execution failed", exc_info=True)
        yield {"log": f"[ERROR] Extraction failed unexpectedly: {e}", "done": False}

@handle_common_errors
def execute_pre_analysis(event: 'PreAnalysisEvent', progress_queue: Queue, cancel_event: threading.Event, logger: 'AppLogger', config: 'Config', thumbnail_manager: 'ThumbnailManager', cuda_available: bool, progress: Optional[Callable] = None) -> Generator[dict, None, None]:
    # ... (Implementation from original file)
    pass
@handle_common_errors
def execute_propagation(event: PropagationEvent, progress_queue: Queue, cancel_event: threading.Event, logger: AppLogger, config: Config, thumbnail_manager, cuda_available, progress=None):
    # ... (Implementation from original file)
    pass
@handle_common_errors
def execute_analysis(event: PropagationEvent, progress_queue: Queue, cancel_event: threading.Event, logger: AppLogger, config: Config, thumbnail_manager, cuda_available, progress=None):
    # ... (Implementation from original file)
    pass
def execute_session_load(app_ui: 'AppUI', event: 'SessionLoadEvent', logger: 'AppLogger', config: 'Config', thumbnail_manager: 'ThumbnailManager') -> Generator[dict, None, None]:
    # ... (Implementation from original file)
    pass

# --- [15] UI CLASS ---

class AppUI:
    # ... (Implementation from original file)
    pass
class EnhancedAppUI(AppUI):
    # ... (Implementation from original file)
    pass

# --- [16] COMPOSITION & MAIN ---

def cleanup_models():
    get_face_analyzer.cache_clear(); get_person_detector.cache_clear(); get_grounding_dino_model.cache_clear()
    get_dam4sam_tracker.cache_clear(); get_lpips_metric.cache_clear(); torch.cuda.empty_cache(); gc.collect()

class CompositionRoot:
    def __init__(self):
        self.config = Config(); self.logger = AppLogger(config=self.config)
        self.thumbnail_manager = ThumbnailManager(self.logger, self.config)
        self.progress_queue, self.cancel_event = Queue(), threading.Event()
        self.logger.set_progress_queue(self.progress_queue); self._app_ui = None
    def get_app_ui(self) -> 'EnhancedAppUI':
        if self._app_ui is None:
            self._app_ui = EnhancedAppUI(config=self.config, logger=self.logger, progress_queue=self.progress_queue, cancel_event=self.cancel_event, thumbnail_manager=self.get_thumbnail_manager())
        return self._app_ui
    def get_config(self) -> 'Config': return self.config
    def get_logger(self) -> 'AppLogger': return self.logger
    def get_thumbnail_manager(self) -> 'ThumbnailManager': return self.thumbnail_manager
    def cleanup(self):
        cleanup_models(); self.thumbnail_manager.clear_cache()
        if hasattr(self, '_app_ui'): self._app_ui = None
        self.cancel_event.set()

def main():
    try:
        composition = CompositionRoot(); logger = composition.get_logger()
        demo = composition.get_app_ui().build_ui()
        logger.info("Frame Extractor & Analyzer v2.0\nStarting application...")
        demo.launch()
    except KeyboardInterrupt: logger.info("\nApplication stopped by user")
    except Exception as e:
        logger.error(f"Error starting application: {e}", exc_info=True); sys.exit(1)
    finally:
        if 'composition' in locals(): composition.cleanup()

if __name__ == "__main__":
    main()
