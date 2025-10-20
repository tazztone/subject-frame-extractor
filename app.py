# keep app.py Monolithic!
"""
Frame Extractor & Analyzer v2.0
"""
import atexit
import contextlib
import cv2
import dataclasses
from datetime import datetime
import functools
import gc
import gradio as gr
import hashlib
import imagehash
import importlib
import io
import json
import logging
import math
import numpy as np
import os
import psutil
import re
import shutil
import subprocess
import sys
import textwrap
import threading
import time
import torch
import traceback
import urllib.request
import yaml

from pathlib import Path

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
from queue import Queue, Empty
from typing import Any, Callable, Dict, List, Optional, Tuple

# --- DEPENDENCY IMPORTS (with error handling) ---
try:
    from DAM4SAM.dam4sam_tracker import DAM4SAMTracker
    from DAM4SAM.utils import utils as dam_utils
except ImportError:
    DAM4SAMTracker = None
    dam_utils = None

try:
    import GPUtil
except ImportError:
    GPUtil = None

try:
    from groundingdino.util.inference import (
        load_model as gdino_load_model,
        predict as gdino_predict,
    )
    from torchvision.ops import box_convert
except ImportError:
    gdino_load_model = None
    gdino_predict = None
    box_convert = None


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
        face_sim: Dict[str, float] = field(default_factory=lambda: {'min': 0.0, 'max': 1.0, 'step': 0.01, 'default_min': 0.5})
        mask_area_pct: Dict[str, float] = field(default_factory=lambda: {'min': 0.0, 'max': 100.0, 'step': 0.1, 'default_min': 1.0})
        dedup_thresh: Dict[str, int] = field(default_factory=lambda: {'min': -1, 'max': 32, 'step': 1, 'default': -1})
        eyes_open: Dict[str, float] = field(default_factory=lambda: {'min': 0.0, 'max': 1.0, 'step': 0.01, 'default_min': 0.3})
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
        log_level: List[str] = field(default_factory=lambda: ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'SUCCESS', 'CRITICAL'])
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
    class ExportOptions:
        enable_crop: bool = True
        crop_padding: int = 1
        crop_ars: str = "16:9,1:1,9:16"

    @dataclass
    class GradioDefaults:
        auto_pctl_input: int = 75
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
    export_options: ExportOptions = field(default_factory=ExportOptions)
    gradio_defaults: GradioDefaults = field(default_factory=GradioDefaults)
    seeding_defaults: SeedingDefaults = field(default_factory=SeedingDefaults)
    utility_defaults: UtilityDefaults = field(default_factory=UtilityDefaults)
    post_processing: PostProcessing = field(default_factory=PostProcessing)
    visualization: Visualization = field(default_factory=Visualization)
    analysis: Analysis = field(default_factory=Analysis)
    model_defaults: ModelDefaults = field(default_factory=ModelDefaults)
    config_path: Optional[str] = "config.yml"

    def __post_init__(self):
        # Start from a snapshot of the initialized defaults
        config_dict = asdict(self)

        # 2. Override with file config
        config_p = Path(self.config_path) if self.config_path else None
        if config_p and config_p.exists():
            try:
                with open(config_p, 'r', encoding='utf-8') as f:
                    file_config = yaml.safe_load(f)
                if file_config:
                    self._merge_configs(config_dict, file_config)
            except Exception as e:
                # Log this instead of raising, so we can fall back to defaults
                print(f"Warning: Could not load or parse {self.config_path}: {e}") # Replace with logger later

        # 3. Override with environment variables
        self._override_with_env_vars(config_dict)

        # 4. Populate the dataclass from the final merged dictionary
        self._from_dict(config_dict)

        # 5. Create necessary directories
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
        if isinstance(default_val, bool):
            return env_val.lower() in ['true', '1', 'yes']
        if isinstance(default_val, int):
            return int(env_val)
        if isinstance(default_val, float):
            return float(env_val)
        if isinstance(default_val, list):
            return [self._coerce_type(v.strip(), default_val[0] if default_val else "") for v in env_val.split(',')]
        return env_val

    def _from_dict(self, data: Dict[str, Any]):
        for f in fields(self):
            if f.name in data:
                field_data = data[f.name]
                # Check if the field is a dataclass and the data is a dict
                if is_dataclass(f.type) and isinstance(field_data, dict):
                    # Get the existing nested dataclass instance
                    nested_instance = getattr(self, f.name)
                    # Create a new dictionary from the nested instance
                    nested_data = asdict(nested_instance)
                    # Merge the new data into the existing data
                    self._merge_configs(nested_data, field_data)
                    # Create a new instance from the merged data
                    setattr(self, f.name, f.type(**nested_data))
                else:
                    setattr(self, f.name, field_data)
        self._validate_config()

    def _validate_config(self):
        if sum(asdict(self.quality_weights).values()) == 0:
            raise ValueError("The sum of quality_weights cannot be zero.")

    def _create_dirs(self):
        dir_paths = [
            self.paths.logs,
            self.paths.models,
            self.paths.downloads,
        ]
        for dir_path in dir_paths:
            if isinstance(dir_path, str):
                try:
                    Path(dir_path).mkdir(exist_ok=True, parents=True)
                except PermissionError as e:
                    raise RuntimeError(f"Cannot create directory at {dir_path}. Check permissions.") from e

    def save_config(self, path: str):
        """Saves the current (resolved) configuration to a YAML file."""
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)


# --- LOGGING ---

SUCCESS_LEVEL_NUM = 25
logging.addLevelName(SUCCESS_LEVEL_NUM, "SUCCESS")

@dataclass
class LogEvent:
    timestamp: str
    level: str
    message: str
    component: str
    operation: Optional[str] = None
    duration_ms: Optional[float] = None
    memory_mb: Optional[float] = None
    gpu_memory_mb: Optional[float] = None
    error_type: Optional[str] = None
    stack_trace: Optional[str] = None
    user_context: Optional[Dict[str, Any]] = None
    performance_metrics: Optional[Dict[str, Any]] = None
    custom_fields: Optional[Dict[str, Any]] = None

class PerformanceMonitor:
    def __init__(self):
        self.process = psutil.Process()
        self.gpu_available = self._check_gpu_availability()

    def _check_gpu_availability(self) -> bool:
        if not GPUtil: return False
        try:
            GPUtil.getGPUs()
            return True
        except:
            return False

    def get_system_metrics(self) -> Dict[str, Any]:
        metrics = {
            'cpu_percent': psutil.cpu_percent(interval=0.1),
            'memory_percent': psutil.virtual_memory().percent,
            'memory_available_mb': psutil.virtual_memory().available / (1024**2),
            'disk_usage_percent': psutil.disk_usage('/').percent,
            'process_memory_mb': self.process.memory_info().rss / (1024**2),
            'process_cpu_percent': self.process.cpu_percent(interval=0.1),
        }
        if self.gpu_available:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    metrics.update({
                        'gpu_memory_used_mb': gpu.memoryUsed,
                        'gpu_memory_total_mb': gpu.memoryTotal,
                        'gpu_memory_percent': gpu.memoryUtil * 100,
                        'gpu_load_percent': gpu.load * 100,
                        'gpu_temperature': gpu.temperature
                    })
            except Exception:
                pass
        return metrics

class EnhancedLogger:
    def __init__(self, config: 'Config', log_dir: Optional[Path] = None,
                 enable_performance_monitoring: bool = True, log_to_file: bool = True,
                 log_to_console: bool = True):
        self.config = config
        self.log_dir = log_dir or Path(self.config.paths.logs)
        self.log_dir.mkdir(exist_ok=True, parents=True)
        self.progress_queue = None
        self.performance_monitor = PerformanceMonitor() if enable_performance_monitoring else None
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_log_file = self.log_dir / f"session_{self.session_id}.log"
        self.structured_log_file = self.log_dir / self.config.logging.structured_log_path
        self.logger = logging.getLogger(f'enhanced_logger_{self.session_id}')
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False
        self.logger.handlers.clear()
        if log_to_console and self.config.logging.colored_logs:
            self._setup_console_handler()
        if log_to_file:
            self._setup_file_handlers()
        self._operation_stack: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    def _setup_console_handler(self):
        console_handler = logging.StreamHandler()
        console_formatter = ColoredFormatter(self.config.logging.log_format)
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(self.config.logging.log_level)
        self.logger.addHandler(console_handler)

    def _setup_file_handlers(self):
        file_handler = logging.FileHandler(self.session_log_file, encoding='utf-8')
        file_formatter = logging.Formatter(self.config.logging.log_format)
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.DEBUG)
        self.logger.addHandler(file_handler)
        # The structured handler is removed to prevent duplicate/mixed logging,
        # as structured logs are written manually in _log_event.
        # self.structured_handler = logging.FileHandler(self.structured_log_file, encoding='utf-8')
        # self.structured_handler.setLevel(logging.DEBUG)
        # self.logger.addHandler(self.structured_handler)

    def set_progress_queue(self, queue):
        self.progress_queue = queue

    @contextlib.contextmanager
    def operation(self, name, component="system"):
        t0 = time.time()
        self.info(f"Start {name}", component=component)
        try:
            yield
            self.success(f"Done {name} in {(time.time()-t0)*1000:.0f}ms", component=component)
        except Exception:
            self.error(f"Failed {name}", component=component, stack_trace=traceback.format_exc())
            raise

    def _create_log_event(self, level: str, message: str, component: str, **kwargs) -> LogEvent:
        current_metrics = self.performance_monitor.get_system_metrics() if self.performance_monitor else {}
        exc_info = kwargs.pop('exc_info', None)
        extra = kwargs.pop('extra', None)
        # Map legacy "stacktrace" to "stack_trace" for backward compatibility
        if 'stacktrace' in kwargs:
            kwargs['stack_trace'] = kwargs.pop('stacktrace')
        if exc_info: kwargs['stack_trace'] = traceback.format_exc()
        if extra:
            kwargs['custom_fields'] = kwargs.get('custom_fields', {})
            kwargs['custom_fields'].update(extra)
        return LogEvent(timestamp=datetime.now().isoformat(), level=level, message=message, component=component,
                        memory_mb=current_metrics.get('process_memory_mb'), gpu_memory_mb=current_metrics.get('gpu_memory_used_mb'), **kwargs)

    def _log_event(self, event: LogEvent):
        log_level_name = event.level.upper()
        log_level = getattr(logging, log_level_name, logging.INFO)
        # Use custom level number for "SUCCESS"
        if log_level_name == "SUCCESS":
            log_level = SUCCESS_LEVEL_NUM

        extra_info = f" [{event.component}]"
        if event.operation: extra_info += f" [{event.operation}]"
        if event.duration_ms: extra_info += f" ({event.duration_ms:.1f}ms)"

        log_message = f"{event.message}{extra_info}"
        if event.stack_trace:
            log_message += f"\n{event.stack_trace}"
        self.logger.log(log_level, log_message)

        # Manual write to JSONL file
        json_line = json.dumps(asdict(event), default=str, ensure_ascii=False)
        with open(self.structured_log_file, 'a', encoding='utf-8') as f:
            f.write(json_line + '\n')

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
    def format(self, record):
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        record.levelname = f"{color}{record.levelname}{self.COLORS['RESET']}"
        return super().format(record)



# --- ERROR HANDLING ---

class ErrorSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class RecoveryStrategy(Enum):
    RETRY = "retry"
    FALLBACK = "fallback"
    SKIP = "skip"
    ABORT = "abort"

class ErrorHandler:
    def __init__(self, logger, config):
        self.logger = logger
        self.config = config
        self.error_count = 0
        self.recovery_attempts = {}

    def with_retry(self, max_attempts=None, backoff_seconds=None, recoverable_exceptions: tuple = (Exception,)):
        max_attempts = max_attempts or self.config.retry.max_attempts
        backoff_seconds = backoff_seconds or self.config.retry.backoff_seconds
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs) -> Any:
                last_exception = None
                for attempt in range(max_attempts):
                    try:
                        return func(*args, **kwargs)
                    except recoverable_exceptions as e:
                        last_exception = e
                        if attempt < max_attempts - 1:
                            sleep_time = backoff_seconds[min(attempt, len(backoff_seconds) - 1)]
                            self.logger.warning(
                                f"Attempt {attempt + 1} failed, retrying in {sleep_time}s: {str(e)}", component="error_handler",
                                custom_fields={'function': func.__name__, 'attempt': attempt + 1, 'max_attempts': max_attempts, 'retry_delay': sleep_time})
                            time.sleep(sleep_time)
                        else:
                            self.logger.error(
                                f"All retry attempts failed for {func.__name__}: {str(e)}", component="error_handler",
                                error_type=type(e).__name__, stack_trace=traceback.format_exc(),
                                custom_fields={'function': func.__name__, 'total_attempts': max_attempts})
                raise last_exception
            return wrapper
        return decorator

    def with_fallback(self, fallback_func: Callable):
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs) -> Any:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    self.logger.warning(
                        f"Primary function {func.__name__} failed, using fallback: {str(e)}", component="error_handler",
                        custom_fields={'primary_function': func.__name__, 'fallback_function': fallback_func.__name__, 'error': str(e)})
                    try:
                        return fallback_func(*args, **kwargs)
                    except Exception as fallback_error:
                        self.logger.error(
                            f"Both primary and fallback functions failed", component="error_handler", error_type=type(fallback_error).__name__,
                            stack_trace=traceback.format_exc(),
                            custom_fields={'primary_function': func.__name__, 'fallback_function': fallback_func.__name__,
                                           'primary_error': str(e), 'fallback_error': str(fallback_error)})
                        raise fallback_error
            return wrapper
        return decorator

# --- EVENTS ---

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
    seed_strategy: str
    scene_detect: bool
    enable_dedup: bool
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

def _to_json_safe(obj):
    if isinstance(obj, dict): return {k: _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [_to_json_safe(v) for v in obj]
    if isinstance(obj, Path): return str(obj)
    if isinstance(obj, np.generic): return _to_json_safe(obj.item())
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, float): return round(obj, 4)
    if hasattr(obj, '__dataclass_fields__'): return _to_json_safe(asdict(obj))
    return obj

def _sanitize_face_ref(runconfig: dict, logger) -> tuple[str, bool]:
    ref = (runconfig.get('face_ref_img_path') or '').strip()
    vid = (runconfig.get('video_path') or '').strip()
    if not ref:
        logger.info("No face reference in session; face similarity disabled on load.", component="session_loader")
        return "", False
    bad_exts = set(logger.config.utility_defaults.video_extensions)
    img_exts = set(logger.config.utility_defaults.image_extensions)
    p = Path(ref)
    if ref == vid or p.suffix.lower() in bad_exts:
        logger.warning("Reference path appears to be a video or equals video_path; clearing safely.", component="session_loader")
        return "", False
    if p.suffix.lower() not in img_exts or not p.is_file():
        logger.warning("Reference path is not a valid image on disk; clearing safely.", component="session_loader", extra={'path': ref})
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

    def calculate_quality_metrics(self, thumb_image_rgb: np.ndarray, quality_config: QualityConfig, logger: 'EnhancedLogger',
                                  mask: np.ndarray | None = None, niqe_metric=None, main_config: 'Config' = None, face_landmarker=None):
        try:
            if face_landmarker:
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=thumb_image_rgb)
                landmarker_result = face_landmarker.detect(mp_image)

                if landmarker_result.face_blendshapes:
                    blendshapes = {b.category_name: b.score for b in landmarker_result.face_blendshapes[0]}
                    self.metrics.eyes_open = 1.0 - max(blendshapes.get('eyeBlinkLeft', 0), blendshapes.get('eyeBlinkRight', 0))
                    self.metrics.blink_prob = max(blendshapes.get('eyeBlinkLeft', 0), blendshapes.get('eyeBlinkRight', 0))

                if landmarker_result.facial_transformation_matrixes:
                    matrix = landmarker_result.facial_transformation_matrixes[0]
                    sy = math.sqrt(matrix[0, 0] * matrix[0, 0] + matrix[1, 0] * matrix[1, 0])
                    singular = sy < 1e-6
                    if not singular:
                        self.metrics.pitch = math.degrees(math.atan2(-matrix[2, 0], sy))
                        self.metrics.yaw = math.degrees(math.atan2(matrix[1, 0], matrix[0, 0]))
                        self.metrics.roll = math.degrees(math.atan2(matrix[2, 1], matrix[2, 2]))
                    else:
                        self.metrics.pitch = math.degrees(math.atan2(-matrix[2, 0], sy))
                        self.metrics.yaw = 0
                        self.metrics.roll = 0

            gray = cv2.cvtColor(thumb_image_rgb, cv2.COLOR_RGB2GRAY)
            active_mask = ((mask > 128) if mask is not None and mask.ndim == 2 else None)
            if active_mask is not None and np.sum(active_mask) < 100:
                active_mask = None # fallback to full-frame stats instead of raising
            lap = cv2.Laplacian(gray, cv2.CV_64F)
            masked_lap = lap[active_mask] if active_mask is not None else lap
            sharpness = np.var(masked_lap) if masked_lap.size > 0 else 0
            sharpness_scaled = (sharpness / (quality_config.sharpness_base_scale * (gray.size / main_config.quality_scaling.resolution_denominator)))
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            edge_strength = np.mean(np.sqrt(sobelx**2 + sobely**2))
            edge_strength_scaled = (edge_strength / (quality_config.edge_strength_base_scale * (gray.size / main_config.quality_scaling.resolution_denominator)))
            pixels = gray[active_mask] if active_mask is not None else gray
            mean_br, std_br = (np.mean(pixels), np.std(pixels)) if pixels.size > 0 else (0, 0)
            brightness = mean_br / 255.0
            contrast = std_br / (mean_br + 1e-7)
            gray_full = cv2.cvtColor(self.image_data, cv2.COLOR_RGB2GRAY)
            active_mask_full = None
            if mask is not None:
                mask_full = cv2.resize(mask, (gray_full.shape[1], gray_full.shape[0]), interpolation=cv2.INTER_NEAREST)
                active_mask_full = (mask_full > 128).astype(np.uint8)
            hist = cv2.calcHist([gray_full], [0], active_mask_full, [256], [0, 256]).flatten()
            entropy = compute_entropy(hist, main_config.quality_scaling.entropy_normalization)
            niqe_score = 0.0
            if quality_config.enable_niqe and niqe_metric is not None:
                try:
                    rgb_image = self.image_data
                    if active_mask_full is not None:
                        mask_3ch = (np.stack([active_mask_full] * 3, axis=-1) > 0)
                        rgb_image = np.where(mask_3ch, rgb_image, 0)
                    img_tensor = (torch.from_numpy(rgb_image).float().permute(2, 0, 1).unsqueeze(0) / 255.0)
                    with (torch.no_grad(), torch.cuda.amp.autocast(enabled=torch.cuda.is_available())):
                        niqe_raw = float(niqe_metric(img_tensor.to(niqe_metric.device)))
                        niqe_score = max(0, min(100, (main_config.quality_scaling.niqe_offset - niqe_raw) * main_config.quality_scaling.niqe_scale_factor))
                except Exception as e:
                    logger.warning("NIQE calculation failed", extra={'frame': self.frame_number, 'error': e})
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            scores_norm = {"sharpness": min(sharpness_scaled, 1.0), "edge_strength": min(edge_strength_scaled, 1.0),
                           "contrast": min(contrast, main_config.quality_scaling.contrast_clamp) / main_config.quality_scaling.contrast_clamp, "brightness": brightness, "entropy": entropy, "niqe": niqe_score / 100.0}
            self.metrics = FrameMetrics(**{f"{k}_score": float(v * 100) for k, v in scores_norm.items()})
            # The quality_weights are part of the main config, not QualityConfig, so we'll need to pass them separately or access them differently.
            if main_config:
                weights = asdict(main_config.quality_weights)
                quality_sum = sum(
                    scores_norm[k] * (weights[k] / 100.0)
                    for k in scores_norm.keys() if k in weights
                )
                self.metrics.quality_score = float(quality_sum * 100)
        except Exception as e:
            self.error = f"Quality calc failed: {e}"
            logger.error("Frame quality calculation failed", exc_info=True, extra={'frame': self.frame_number})

@dataclass
class Scene:
    shot_id: int
    start_frame: int
    end_frame: int
    status: str = "pending"
    best_seed_frame: int | None = None
    seed_metrics: dict = field(default_factory=dict)
    seed_frame_idx: int | None = None
    seed_config: dict = field(default_factory=dict)
    seed_type: str | None = None
    seed_result: dict = field(default_factory=dict)
    preview_path: str | None = None
    manual_status_change: bool = False

@dataclass
class AnalysisParameters:
    source_path: str = ""
    method: str = ""
    interval: float = 0.0
    max_resolution: str = ""
    fast_scene: bool = False
    use_png: bool = True
    output_folder: str = ""
    video_path: str = ""
    disable_parallel: bool = False
    resume: bool = False
    enable_face_filter: bool = False
    face_ref_img_path: str = ""
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
    thumbnails_only: bool = True
    thumb_megapixels: float = 0.5
    pre_analysis_enabled: bool = False
    pre_sample_nth: int = 1
    primary_seed_strategy: str = "🤖 Automatic"
    gdino_config_path: str = ""
    gdino_checkpoint_path: str = ""
    box_threshold: float = 0.35
    text_threshold: float = 0.25
    min_mask_area_pct: float = 1.0
    sharpness_base_scale: float = 2500.0
    edge_strength_base_scale: float = 100.0

    def __post_init__(self):
        # This post_init is now less critical as config is passed on creation,
        # but can be used for validation or complex defaults.
        pass

    @classmethod
    def from_ui(cls, logger: 'EnhancedLogger', config: 'Config', **kwargs):
        if 'face_ref_img_path' in kwargs or 'video_path' in kwargs:
            sanitized_face_ref, face_filter_enabled = _sanitize_face_ref(kwargs, logger)
            kwargs['face_ref_img_path'] = sanitized_face_ref
            kwargs['enable_face_filter'] = face_filter_enabled
        
        if 'thumb_megapixels' in kwargs:
            thumb_mp = kwargs['thumb_megapixels']
            if not isinstance(thumb_mp, (int, float)) or thumb_mp <= 0:
                logger.warning(f"Invalid thumb_megapixels: {thumb_mp}, using default")
                kwargs['thumb_megapixels'] = config.ui_defaults.thumb_megapixels

        if 'pre_sample_nth' in kwargs:
            sample_nth = kwargs['pre_sample_nth']
            if not isinstance(sample_nth, int) or sample_nth < 1:
                logger.warning(f"Invalid pre_sample_nth: {sample_nth}, using 1")
                kwargs['pre_sample_nth'] = 1

        valid_keys = {f.name for f in fields(cls)}
        filtered_defaults = {k: v for k, v in asdict(config.ui_defaults).items() if k in valid_keys}
        instance = cls(**filtered_defaults)
        for key, value in kwargs.items():
            if hasattr(instance, key) and value is not None and value != '':
                default = getattr(instance, key)
                try:
                    setattr(instance, key, _coerce(value, type(default)))
                except (ValueError, TypeError):
                    logger.warning(f"Could not coerce UI value for '{key}' to {type(default)}. Using default.", extra={'key': key, 'value': value})
        return instance

    def _get_config_hash(self, output_dir: Path) -> str:
        data_to_hash = json.dumps(_to_json_safe(asdict(self)), sort_keys=True)
        scene_seeds_path = output_dir / "scene_seeds.json"
        if scene_seeds_path.exists(): data_to_hash += scene_seeds_path.read_text(encoding='utf-8')
        return hashlib.sha256(data_to_hash.encode()).hexdigest()

@dataclass
class MaskingResult:
    mask_path: str | None = None
    shot_id: int | None = None
    seed_type: str | None = None
    seed_face_sim: float | None = None
    mask_area_pct: float | None = None
    mask_empty: bool = True
    error: str | None = None

# --- BASE PIPELINE ---

class Pipeline:
    def __init__(self, config: 'Config', logger: 'EnhancedLogger', params: 'AnalysisParameters',
                 progress_queue: Queue, cancel_event: threading.Event):
        self.config = config
        self.logger = logger
        self.params = params
        self.progress_queue = progress_queue
        self.cancel_event = cancel_event

# --- CACHING & OPTIMIZATION ---

class ThumbnailManager:
    def __init__(self, logger, config: 'Config'):
        self.logger = logger
        self.config = config
        self.cache = OrderedDict()
        self.max_size = self.config.cache.size
        self.logger.info(f"ThumbnailManager initialized with cache size {self.max_size}")

    def get(self, thumb_path: Path):
        if not isinstance(thumb_path, Path): thumb_path = Path(thumb_path)
        if thumb_path in self.cache:
            self.cache.move_to_end(thumb_path)
            return self.cache[thumb_path]
        if not thumb_path.exists(): return None

        if len(self.cache) > self.max_size * self.config.cache.cleanup_threshold:
            self._cleanup_old_entries()

        if Image is None:
            self.logger.warning("Pillow not available; attempting to load with OpenCV", extra={'path': str(thumb_path)})
            if cv2 and thumb_path.suffix.lower() in {'.jpg','.jpeg','.png','.webp'}:
                try:
                    thumb_img_bgr = cv2.imread(str(thumb_path))
                    if thumb_img_bgr is not None:
                        thumb_img = cv2.cvtColor(thumb_img_bgr, cv2.COLOR_BGR2RGB)
                        self.cache[thumb_path] = thumb_img
                        while len(self.cache) > self.max_size:
                            self.cache.popitem(last=False)
                        return thumb_img
                except Exception as e:
                    self.logger.error("OpenCV failed to load thumbnail", extra={'path': str(thumb_path), 'error': e})
            return None

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
        """Force clear the thumbnail cache to free memory"""
        self.cache.clear()
        gc.collect()

    def _cleanup_old_entries(self):
        """Clean up a percentage of the cache to make room for new entries."""
        num_to_remove = int(self.max_size * self.config.cache.eviction_factor)
        for _ in range(num_to_remove):
            if not self.cache:
                break
            self.cache.popitem(last=False)


# --- MODEL LOADING & MANAGEMENT ---

def download_model(url, dest_path, description, logger, error_handler: ErrorHandler, config: 'Config', min_size=1_000_000):
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.is_file() and (min_size is None or dest_path.stat().st_size >= min_size):
        logger.info(f"Using cached {description}: {dest_path}")
        return
    @error_handler.with_retry(recoverable_exceptions=(urllib.error.URLError, TimeoutError, RuntimeError))
    def download_func():
        logger.info(f"Downloading {description}", extra={'url': url, 'dest': dest_path})
        req = urllib.request.Request(url, headers={"User-Agent": config.models.user_agent})
        with urllib.request.urlopen(req, timeout=60) as resp, open(dest_path, "wb") as out:
            shutil.copyfileobj(resp, out)
        if not dest_path.exists() or dest_path.stat().st_size < min_size:
            raise RuntimeError(f"Downloaded {description} seems incomplete")
        logger.success(f"{description} downloaded successfully.")
    try:
        download_func()
    except Exception as e:
        logger.error(f"Failed to download {description}", exc_info=True, extra={'url': url})
        raise RuntimeError(f"Failed to download required model: {description}") from e

@lru_cache(maxsize=None)
def get_face_landmarker(model_path: str):
    if not vision:
        raise ImportError("MediaPipe vision components are not installed.")
    logger = EnhancedLogger(config=Config())
    logger.info("Loading or getting cached MediaPipe face landmarker model.", component="face_landmarker")
    try:
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1,
        )
        detector = vision.FaceLandmarker.create_from_options(options)
        logger.success("Face landmarker model loaded successfully.")
        return detector
    except Exception as e:
        logger.error(f"Could not initialize MediaPipe face landmarker model. Error: {e}", component="face_landmarker")
        raise RuntimeError("Could not initialize MediaPipe face landmarker model.") from e

@lru_cache(maxsize=None)
def get_face_analyzer(model_name: str, models_path: str, det_size_tuple: tuple):
    from insightface.app import FaceAnalysis
    logger = EnhancedLogger(config=Config())
    logger.info(f"Loading or getting cached face model: {model_name}")
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        providers = (['CUDAExecutionProvider', 'CPUExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider'])
        analyzer = FaceAnalysis(name=model_name, root=models_path, providers=providers)
        analyzer.prepare(ctx_id=0 if device == 'cuda' else -1, det_size=det_size_tuple)
        logger.success(f"Face model loaded with {'CUDA' if device == 'cuda' else 'CPU'}.")
        return analyzer
    except Exception as e:
        if "out of memory" in str(e) and torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.warning("CUDA OOM, retrying with CPU...")
            try:
                # Retry with CPU
                analyzer = FaceAnalysis(name=model_name, root=models_path,
                                      providers=['CPUExecutionProvider'])
                analyzer.prepare(ctx_id=-1, det_size=det_size_tuple)
                return analyzer
            except Exception as cpu_e:
                logger.error(f"CPU fallback also failed: {cpu_e}")
        raise RuntimeError(f"Could not initialize face analysis model. Error: {e}") from e

class PersonDetector:
    def __init__(self, logger: 'EnhancedLogger', model_path: Path, imgsz: int, conf: float, device: str = 'cuda'):
        from ultralytics import YOLO
        self.logger = logger
        self.device = device if torch.cuda.is_available() else 'cpu'
        if not model_path.exists():
            raise FileNotFoundError(f"Person detector model not found at {model_path}")
        self.model = YOLO(str(model_path))
        self.model.to(self.device)
        self.imgsz = imgsz
        self.conf = conf
        self.logger.info("YOLO person detector loaded", component="person_detector",
                         custom_fields={'device': self.device, 'model': model_path.name})

    def detect_boxes(self, img_rgb):
        res = self.model.predict(img_rgb, imgsz=self.imgsz, conf=self.conf, classes=[0], verbose=False, device=self.device)
        return [
            (*map(int, b.xyxy[0].tolist()), float(b.conf[0]))
            for r in res if getattr(r, "boxes", None) is not None
            for b in r.boxes.cpu()
        ]

@lru_cache(maxsize=None)
def get_person_detector(model_path_str: str, device: str, imgsz: int, conf: float):
    logger = EnhancedLogger(config=Config())
    logger.info(f"Loading or getting cached person detector from: {model_path_str}", component="person_detector")
    return PersonDetector(logger=logger, model_path=Path(model_path_str), imgsz=imgsz, conf=conf, device=device)

@lru_cache(maxsize=None)
def get_grounding_dino_model(gdino_config_path: str, gdino_checkpoint_path: str, models_path: str, grounding_dino_url: str, device="cuda"):
    if not gdino_load_model: raise ImportError("GroundingDINO is not installed.")
    logger = EnhancedLogger(config=Config()) # Fallback for standalone usage
    error_handler = ErrorHandler(logger, Config()) # Fallback for standalone usage
    try:
        models_dir = Path(models_path)
        models_dir.mkdir(parents=True, exist_ok=True)

        # Robust path handling for the config file
        config_file_path = gdino_config_path or Config().paths.grounding_dino_config
        config_path = Path(config_file_path)
        if not config_path.is_absolute():
            config_path = project_root / config_path

        ckpt_path = Path(gdino_checkpoint_path)
        if not ckpt_path.is_absolute():
            ckpt_path = models_dir / Path(grounding_dino_url).name
        
        download_model(grounding_dino_url,
                       ckpt_path, "GroundingDINO Swin-T model", logger, error_handler, config=Config(), min_size=500_000_000)
        
        gdino_model = gdino_load_model(model_config_path=str(config_path), model_checkpoint_path=str(ckpt_path), device=device)
        logger.info("Grounding DINO model loaded.", component="grounding", custom_fields={'model_path': str(ckpt_path)})
        return gdino_model
    except Exception as e:
        logger.warning("Grounding DINO unavailable.", component="grounding", exc_info=True)
        return None

def predict_grounding_dino(model, image_tensor, caption, box_threshold, text_threshold, device="cuda"):
    if not gdino_predict: raise ImportError("GroundingDINO is not installed.")
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=(device == 'cuda')):
        return gdino_predict(model=model, image=image_tensor.to(device), caption=caption,
                             box_threshold=float(box_threshold), text_threshold=float(text_threshold))

@lru_cache(maxsize=None)
def get_dam4sam_tracker(model_name: str, models_path: str, model_urls_tuple: tuple):
    if not DAM4SAMTracker or not dam_utils: raise ImportError("DAM4SAM is not installed.")
    logger = EnhancedLogger(config=Config()) # Fallback
    error_handler = ErrorHandler(logger, Config()) # Fallback
    model_urls = dict(model_urls_tuple)

    if not (DAM4SAMTracker and torch and torch.cuda.is_available()):
        logger.error("DAM4SAM dependencies or CUDA not available.")
        return None
    try:
        models_dir = Path(models_path)
        models_dir.mkdir(parents=True, exist_ok=True)
        # Choose a safe default if UI passes empty string
        selected_name = (model_name or Config().ui_defaults.dam4sam_model_name or next(iter(model_urls.keys())))
        if selected_name not in model_urls:
            raise ValueError(f"Unknown DAM4SAM model: {selected_name}")
        url = model_urls[selected_name]
        checkpoint_path = models_dir / Path(url).name
        logger.info(f"Initializing DAM4SAM tracker", custom_fields={'selected_model': selected_name, 'checkpoint_path': str(checkpoint_path)})
        download_model(url, checkpoint_path, f"DAM4SAM {selected_name}", logger, error_handler, config=Config(), min_size=100_000_000)
        actual_path, _ = dam_utils.determine_tracker(selected_name)
        if not Path(actual_path).exists():
            Path(actual_path).parent.mkdir(exist_ok=True, parents=True)
            shutil.copy(checkpoint_path, actual_path)
        tracker = DAM4SAMTracker(selected_name)
        logger.success("DAM4SAM tracker initialized.")
        return tracker
    except Exception as e:
        logger.error("Failed to initialize DAM4SAM tracker", exc_info=True)
        if "out of memory" in str(e) and torch.cuda.is_available():
            torch.cuda.empty_cache()
        return None

def initialize_analysis_models(params: AnalysisParameters, config: Config, logger: EnhancedLogger, cuda_available: bool):
    device = "cuda" if cuda_available else "cpu"
    face_analyzer, ref_emb, person_detector, face_landmarker = None, None, None, None
    if params.enable_face_filter:
        face_analyzer = get_face_analyzer(
            model_name=params.face_model_name,
            models_path=str(config.paths.models),
            det_size_tuple=tuple(config.model_defaults.face_analyzer_det_size)
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
                        else: logger.warning("No face found in reference image.", extra={'path': ref_path})
                    else: logger.warning("Could not read reference face image.", extra={'path': ref_path})
                except Exception as e: logger.error("Failed to process reference face image.", exc_info=True)
            else: logger.warning("Reference face image path does not exist.", extra={'path': ref_path})

    model_path = Path(config.paths.models) / params.person_detector_model
    person_detector = get_person_detector(
        model_path_str=str(model_path),
        device=device,
        imgsz=config.person_detector.imgsz,
        conf=config.person_detector.conf
    )

    # Initialize MediaPipe Face Landmarker
    landmarker_path = Path(config.paths.models) / Path(config.models.face_landmarker).name
    download_model(config.models.face_landmarker, landmarker_path, "MediaPipe Face Landmarker", logger, ErrorHandler(logger, config), config)
    if landmarker_path.exists():
        face_landmarker = get_face_landmarker(str(landmarker_path))

    return {"face_analyzer": face_analyzer, "ref_emb": ref_emb, "person_detector": person_detector, "face_landmarker": face_landmarker, "device": device}

# --- VIDEO & FRAME PROCESSING ---

class VideoManager:
    def __init__(self, source_path: str, config: 'Config', max_resolution: Optional[str] = None):
        self.source_path = source_path
        self.config = config
        self.max_resolution = max_resolution or self.config.ui_defaults.max_resolution
        self.is_youtube = ("youtube.com/" in source_path or "youtu.be/" in source_path)

    def prepare_video(self, logger: 'EnhancedLogger') -> str:
        if self.is_youtube:
            if not ytdlp:
                raise ImportError("yt-dlp not installed.")
            logger.info("Downloading video", component="video", user_context={'source': self.source_path})
            
            tmpl = self.config.youtube_dl.output_template
            fmt = self.config.youtube_dl.format_string.replace("{max_res}", str(self.max_resolution)) if self.max_resolution != "maximum available" else self.config.youtube_dl.format_string.replace("[height<={max_res}]", "")

            ydl_opts = {
                'outtmpl': str(Path(self.config.paths.downloads) / tmpl),
                'format': fmt,
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

def run_scene_detection(video_path, output_dir, logger=None):
    if not detect: raise ImportError("scenedetect is not installed.")
    logger = logger or EnhancedLogger(config=Config())
    logger.info("Detecting scenes...", component="video")
    try:
        scene_list = detect(str(video_path), ContentDetector())
        shots = ([(s.frame_num, e.frame_num) for s, e in scene_list] if scene_list else [])
        with (output_dir / "scenes.json").open('w', encoding='utf-8') as f: json.dump(shots, f)
        logger.success(f"Found {len(shots)} scenes.", component="video")
        return shots
    except Exception as e:
        logger.error("Scene detection failed.", component="video", exc_info=True)
        return []

def run_ffmpeg_extraction(video_path, output_dir, video_info, params, progress_queue, cancel_event, logger: 'EnhancedLogger', config: 'Config'):
    log_file_path = output_dir / "ffmpeg_log.txt"
    cmd_base = ['ffmpeg', '-y', '-i', str(video_path), '-hide_banner', '-loglevel', config.ffmpeg.log_level]
    if params.thumbnails_only:
        thumb_dir = output_dir / "thumbs"
        thumb_dir.mkdir(exist_ok=True)
        target_area = params.thumb_megapixels * 1_000_000
        w, h = video_info.get('width', 1920), video_info.get('height', 1080)
        scale_factor = math.sqrt(target_area / (w * h))
        vf_scale = f"scale=w=trunc(iw*{scale_factor}/2)*2:h=trunc(ih*{scale_factor}/2)*2"
        fps = video_info.get('fps', 30)
        vf_filter = f"fps={fps},{vf_scale},showinfo"
        cmd = cmd_base + ['-vf', vf_filter, '-c:v', 'libwebp', '-lossless', '0', '-quality', str(config.ffmpeg.thumbnail_quality), '-vsync', 'vfr', str(thumb_dir / "frame_%06d.webp")]
    else:
        select_filter_map = {'interval': f"fps=1/{max(0.1, float(params.interval))}", 'keyframes': "select='eq(pict_type,I)'",
                             'scene': f"select='gt(scene,{config.ffmpeg.fast_scene_threshold if params.fast_scene else config.ffmpeg.scene_threshold})'", 'all': f"fps={video_info.get('fps', 30)}",
                             'every_nth_frame': f"select='not(mod(n,{max(1, int(params.nth_frame))}))'"}
        select_filter = select_filter_map.get(params.method)
        vf_filter = (select_filter + ",showinfo") if select_filter else "showinfo"
        ext = 'png' if params.use_png else 'jpg'
        cmd = cmd_base + ['-vf', vf_filter, '-vsync', 'vfr', '-f', 'image2', str(output_dir / f"frame_%06d.{ext}")]
    with open(log_file_path, 'w', encoding='utf-8') as stderr_handle:
        process = subprocess.Popen(cmd, stderr=stderr_handle, text=True, encoding='utf-8', bufsize=1)
        while process.poll() is None:
            if cancel_event.is_set(): process.terminate(); break
            time.sleep(0.1)
        process.wait()
    try:
        with open(log_file_path, 'r', encoding='utf-8') as f: log_content = f.read()
        frame_map_list = [int(m.group(1)) for m in re.finditer(r' n:\s*(\d+)', log_content)]
        with open(output_dir / "frame_map.json", 'w', encoding='utf-8') as f: json.dump(frame_map_list, f)
    finally:
        log_file_path.unlink(missing_ok=True)
    if process.returncode != 0 and not cancel_event.is_set():
        raise RuntimeError(f"FFmpeg failed with code {process.returncode}.")

def postprocess_mask(mask: np.ndarray, config: 'Config', fill_holes: bool = True, keep_largest_only: bool = True) -> np.ndarray:
    if mask is None or mask.size == 0: return mask
    binary_mask = (mask > 128).astype(np.uint8)
    if fill_holes:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (config.masking.close_kernel_size, config.masking.close_kernel_size))
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    if keep_largest_only and config.masking.keep_largest_only:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
        if num_labels > 1:
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            binary_mask = (labels == largest_label).astype(np.uint8)
    return (binary_mask * 255).astype(np.uint8)

def render_mask_overlay(frame_rgb: np.ndarray, mask_gray: np.ndarray, alpha: float, logger: 'EnhancedLogger') -> np.ndarray:
    if mask_gray is None or frame_rgb is None:
        return frame_rgb if frame_rgb is not None else np.array([])
    h, w = frame_rgb.shape[:2]
    if mask_gray.shape[:2] != (h, w): mask_gray = cv2.resize(mask_gray, (w, h), interpolation=cv2.INTER_NEAREST)
    m = (mask_gray > 128)
    red_layer = np.zeros_like(frame_rgb, dtype=np.uint8)
    red_layer[..., 0] = 255
    blended = cv2.addWeighted(frame_rgb, 1.0 - alpha, red_layer, alpha, 0.0)
    if m.ndim == 2: m = m[..., np.newaxis]
    elif m.ndim == 3 and m.shape[2] != 1:
        logger.warning("Unexpected mask shape. Skipping overlay.", extra={'shape': m.shape})
        return frame_rgb
    return np.where(m, blended, frame_rgb)

def rgb_to_pil(image_rgb: np.ndarray) -> Image.Image:
    if not Image: raise ImportError("Pillow is not installed.")
    return Image.fromarray(image_rgb)

def create_frame_map(output_dir: Path, logger: 'EnhancedLogger'):
    logger.info("Loading frame map...", component="frames")
    frame_map_path = output_dir / "frame_map.json"
    try:
        with open(frame_map_path, 'r', encoding='utf-8') as f: frame_map_list = json.load(f)
        # Ensure all frame numbers are integers, as JSON can load them as strings.
        sorted_frames = sorted(map(int, frame_map_list))
        return {orig_num: f"frame_{i+1:06d}.webp" for i, orig_num in enumerate(sorted_frames)}
    except (FileNotFoundError, json.JSONDecodeError, TypeError) as e:
        logger.error(f"Could not load or parse frame_map.json: {e}. Frame mapping will be empty. This can happen if extraction was incomplete.", exc_info=False)
        return {}

# --- MASKING & PROPAGATION ---

class MaskPropagator:
    def __init__(self, params, dam_tracker, cancel_event, progress_queue, config: 'Config', logger=None):
        self.params = params
        self.dam_tracker = dam_tracker
        self.cancel_event = cancel_event
        self.progress_queue = progress_queue
        self.config = config
        self.logger = logger or EnhancedLogger(config=Config())
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    def propagate(self, shot_frames_rgb, seed_idx, bbox_xywh):
        if not self.dam_tracker or not shot_frames_rgb:
            err_msg = "Tracker not initialized" if not self.dam_tracker else "No frames"
            shape = shot_frames_rgb[0].shape[:2] if shot_frames_rgb else (100, 100)
            num_frames = len(shot_frames_rgb)
            return ([np.zeros(shape, np.uint8)] * num_frames, [0.0] * num_frames, [True] * num_frames, [err_msg] * num_frames)
        self.logger.info("Propagating masks", component="propagator", user_context={'num_frames': len(shot_frames_rgb), 'seed_index': seed_idx})
        masks = [None] * len(shot_frames_rgb)
        def _propagate_direction(start_idx, end_idx, step):
            for i in range(start_idx, end_idx, step):
                if self.cancel_event.is_set(): break
                outputs = self.dam_tracker.track(rgb_to_pil(shot_frames_rgb[i]))
                mask = outputs.get('pred_mask')
                if mask is not None: mask = postprocess_mask((mask * 255).astype(np.uint8), config=self.config, fill_holes=True, keep_largest_only=True)
                masks[i] = mask if mask is not None else np.zeros_like(shot_frames_rgb[i], dtype=np.uint8)[:, :, 0]
        try:
            with torch.cuda.amp.autocast(enabled=self._device == 'cuda'):
                outputs = self.dam_tracker.initialize(rgb_to_pil(shot_frames_rgb[seed_idx]), None, bbox=bbox_xywh)
                mask = outputs.get('pred_mask')
                if mask is not None: mask = postprocess_mask((mask * 255).astype(np.uint8), config=self.config, fill_holes=True, keep_largest_only=True)
                masks[seed_idx] = mask if mask is not None else np.zeros_like(shot_frames_rgb[seed_idx], dtype=np.uint8)[:, :, 0]
                _propagate_direction(seed_idx + 1, len(shot_frames_rgb), 1)
                self.dam_tracker.initialize(rgb_to_pil(shot_frames_rgb[seed_idx]), None, bbox=bbox_xywh)
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
            self.logger.critical("DAM4SAM propagation failed", component="propagator", exc_info=True)
            h, w = shot_frames_rgb[0].shape[:2]
            error_msg = f"Propagation failed: {e}"
            num_frames = len(shot_frames_rgb)
            return ([np.zeros((h, w), np.uint8)] * num_frames, [0.0] * num_frames, [True] * num_frames, [error_msg] * num_frames)

class SeedSelector:
    def __init__(self, params, config: 'Config', face_analyzer, reference_embedding, person_detector, tracker, gdino_model, logger=None):
        self.params = params
        self.config = config
        self.face_analyzer = face_analyzer
        self.reference_embedding = reference_embedding
        self.person_detector = person_detector
        self.tracker = tracker
        self._gdino = gdino_model
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger = logger or EnhancedLogger(config=Config())

    def _get_param(self, source, key, default=None):
        """Safely gets a parameter from a source that can be a dict or an object."""
        if isinstance(source, dict):
            return source.get(key, default)
        return getattr(source, key, default)

    def select_seed(self, frame_rgb, current_params=None):
        params_source = current_params if current_params is not None else self.params
        p = params_source  # Keep for passing to other methods

        primary_strategy = self._get_param(params_source, 'primary_seed_strategy', "🤖 Automatic")
        use_face_filter = self._get_param(params_source, 'enable_face_filter', False)

        if primary_strategy == "👤 By Face":
            if self.face_analyzer and self.reference_embedding is not None and use_face_filter:
                self.logger.info("Starting 'Identity-First' seeding.")
                return self._identity_first_seed(frame_rgb, p)
            else:
                self.logger.warning("Face strategy selected but no reference face provided.")
                return self._object_first_seed(frame_rgb, p)
        elif primary_strategy == "📝 By Text":
            self.logger.info("Starting 'Object-First' seeding.")
            return self._object_first_seed(frame_rgb, p)
        elif primary_strategy == "🔄 Face + Text Fallback":
            self.logger.info("Starting 'Face-First with Text Fallback' seeding.")
            return self._face_with_text_fallback_seed(frame_rgb, p)
        else:
            self.logger.info("Starting 'Automatic' seeding.")
            return self._choose_person_by_strategy(frame_rgb, p)

    def _face_with_text_fallback_seed(self, frame_rgb, params):
        # If no reference embedding is available, go straight to text fallback.
        if self.reference_embedding is None:
            self.logger.warning("No reference face for face-first strategy, falling back to text prompt.", extra={'reason': 'no_ref_emb'})
            return self._object_first_seed(frame_rgb, params)

        # First, attempt the identity-first strategy.
        box, details = self._identity_first_seed(frame_rgb, params)

        # If it succeeds (i.e., finds a matching face and subject), return the result.
        if box is not None:
            self.logger.info("Face-first strategy successful.")
            return box, details

        # If it fails, log the failure and fall back to the object-first (text) strategy.
        self.logger.warning("Face detection failed or no match found, falling back to text prompt strategy.", extra=details)
        return self._object_first_seed(frame_rgb, params)

    def _identity_first_seed(self, frame_rgb, params):
        target_face, details = self._find_target_face(frame_rgb)
        if not target_face:
            self.logger.warning("Target face not found in scene.", extra=details)
            return None, {"type": "no_subject_found"}
        yolo_boxes, dino_boxes = self._get_yolo_boxes(frame_rgb), self._get_dino_boxes(frame_rgb, params)[0]
        best_box, best_details = self._score_and_select_candidate(target_face, yolo_boxes, dino_boxes)
        if best_box:
            self.logger.success("Evidence-based seed selected.", extra=best_details)
            return best_box, best_details
        self.logger.warning("No high-confidence body box found, expanding face box as fallback.")
        expanded_box = self._expand_face_to_body(target_face['bbox'], frame_rgb.shape)
        return expanded_box, {"type": "expanded_box_from_face", "seed_face_sim": details.get('seed_face_sim', 0)}

    def _object_first_seed(self, frame_rgb, params):
        dino_boxes, dino_details = self._get_dino_boxes(frame_rgb, params)
        if dino_boxes:
            yolo_boxes = self._get_yolo_boxes(frame_rgb)
            if yolo_boxes:
                best_iou, best_match = -1, None
                for d_box in dino_boxes:
                    for y_box in yolo_boxes:
                        iou = self._calculate_iou(d_box['bbox'], y_box['bbox'])
                        if iou > best_iou:
                            best_iou, best_match = iou, {'bbox': d_box['bbox'], 'type': 'dino_yolo_intersect', 'iou': iou,
                                                         'dino_conf': d_box['conf'], 'yolo_conf': y_box['conf']}
                if best_match and best_match['iou'] > self.config.seeding_defaults.yolo_iou_threshold:
                    self.logger.info("Found high-confidence DINO+YOLO intersection.", extra=best_match)
                    return self._xyxy_to_xywh(best_match['bbox']), best_match
            self.logger.info("Using best DINO box without YOLO validation.", extra=dino_details)
            return self._xyxy_to_xywh(dino_boxes[0]['bbox']), dino_details
        self.logger.info("No DINO results, falling back to YOLO-only strategy.")
        return self._choose_person_by_strategy(frame_rgb, params)

    def _find_target_face(self, frame_rgb):
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        try: faces = self.face_analyzer.get(frame_bgr)
        except Exception as e:
            self.logger.error("Face analysis failed.", exc_info=True)
            return None, {"error": str(e)}
        if not faces: return None, {"error": "no_faces_detected"}
        best_face, best_sim = None, 0.0
        for face in faces:
            sim = np.dot(face.normed_embedding, self.reference_embedding)
            if sim > best_sim: best_sim, best_face = sim, face
        if best_face and best_sim > self.config.seeding_defaults.face_similarity_threshold:
            return {'bbox': best_face.bbox.astype(int), 'embedding': best_face.normed_embedding}, {'type': 'face_match', 'seed_face_sim': best_sim}
        return None, {'error': 'no_matching_face', 'best_sim': best_sim}

    def _get_yolo_boxes(self, frame_rgb):
        if not self.person_detector: return []
        try:
            boxes = self.person_detector.detect_boxes(frame_rgb)
            return [{'bbox': b[:4], 'conf': b[4], 'type': 'yolo'} for b in boxes]
        except Exception as e:
            self.logger.warning("YOLO person detector failed.", exc_info=True)
            return []

    def _get_dino_boxes(self, frame_rgb, params):
        prompt = self._get_param(params, "text_prompt", "")
        if not self._gdino or not prompt: return [], {}
        box_th = self._get_param(params, "box_threshold", self.params.box_threshold)
        text_th = self._get_param(params, "text_threshold", self.params.text_threshold)
        image_source, image_tensor = self._load_image_from_array(frame_rgb)
        h, w = image_source.shape[:2]
        try:
            boxes_norm, confs, labels = predict_grounding_dino(model=self._gdino, image_tensor=image_tensor, caption=prompt,
                                                               box_threshold=float(box_th), text_threshold=float(text_th), device=self._device)
        except Exception as e:
            self.logger.error("Grounding DINO prediction failed.", exc_info=True)
            if "out of memory" in str(e) and torch.cuda.is_available():
                torch.cuda.empty_cache()
            return [], {"error": str(e)}
        if boxes_norm is None or len(boxes_norm) == 0: return [], {"type": "text_prompt", "error": "no_boxes"}
        scale = torch.tensor([w, h, w, h], device=boxes_norm.device, dtype=boxes_norm.dtype)
        xyxy_boxes = box_convert(boxes=(boxes_norm * scale).cpu(), in_fmt="cxcywh", out_fmt="xyxy").numpy()
        results = [{'bbox': box.astype(int), 'conf': confs[i].item(), 'label': labels[i], 'type': 'dino'} for i, box in enumerate(xyxy_boxes)]
        results.sort(key=lambda x: x['conf'], reverse=True)
        return results, {**results[0], "all_boxes_count": len(results)}

    def _score_and_select_candidate(self, target_face, yolo_boxes, dino_boxes):
        candidates = yolo_boxes + dino_boxes
        if not candidates: return None, {}
        scored_candidates = []
        for cand in candidates:
            score, details = 0, {'orig_conf': cand['conf'], 'orig_type': cand['type']}
            if self._box_contains(cand['bbox'], target_face['bbox']):
                score += self.config.seeding_defaults.face_contain_score
                details['face_contained'] = True
            score += cand['conf'] * self.config.seeding_defaults.confidence_score_multiplier
            scored_candidates.append({'score': score, 'box': cand['bbox'], 'details': details})
        best_iou, best_pair = -1, None
        for y_box in yolo_boxes:
            for d_box in dino_boxes:
                iou = self._calculate_iou(y_box['bbox'], d_box['bbox'])
                if iou > best_iou: best_iou, best_pair = iou, (y_box, d_box)
        if best_iou > self.config.seeding_defaults.yolo_iou_threshold:
            for cand in scored_candidates:
                if np.array_equal(cand['box'], best_pair[0]['bbox']) or np.array_equal(cand['box'], best_pair[1]['bbox']):
                    cand['score'] += self.config.seeding_defaults.iou_bonus
                    cand['details']['high_iou_pair'] = True
        if not scored_candidates: return None, {}
        winner = max(scored_candidates, key=lambda x: x['score'])
        return self._xyxy_to_xywh(winner['box']), {'type': 'evidence_based_selection', 'final_score': winner['score'], **winner['details']}

    def _choose_person_by_strategy(self, frame_rgb, params):
        boxes = self._get_yolo_boxes(frame_rgb)
        if not boxes:
            self.logger.warning("No persons found for fallback strategy.")
            return self._final_fallback_box(frame_rgb.shape), {'type': 'fallback_rect'}
        strategy = getattr(params, "seed_strategy", "Largest Person")
        if isinstance(params, dict): strategy = params.get('seed_strategy', strategy)
        h, w = frame_rgb.shape[:2]
        cx, cy = w / 2, h / 2
        score_func = {"Largest Person": lambda b: (b['bbox'][2] - b['bbox'][0]) * (b['bbox'][3] - b['bbox'][1]),
                      "Center-most Person": lambda b: -math.hypot((b['bbox'][0] + b['bbox'][2]) / 2 - cx, (b['bbox'][1] + b['bbox'][3]) / 2 - cy)}.get(strategy)
        best_person = sorted(boxes, key=score_func, reverse=True)[0]
        return self._xyxy_to_xywh(best_person['bbox']), {'type': f'person_{strategy.lower().replace(" ", "_")}', 'conf': best_person['conf']}

    def _load_image_from_array(self, image_rgb: np.ndarray):
        from torchvision import transforms
        transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        return image_rgb, transform(image_rgb)

    def _calculate_iou(self, box1, box2):
        x1, y1, x2, y2 = box1
        x1_p, y1_p, x2_p, y2_p = box2
        inter_x1, inter_y1, inter_x2, inter_y2 = max(x1, x1_p), max(y1, y1_p), min(x2, x2_p), min(y2, y2_p)
        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        union_area = (x2 - x1) * (y2 - y1) + (x2_p - x1_p) * (y2_p - y1_p) - inter_area
        return inter_area / (union_area + 1e-6)

    def _box_contains(self, cb, ib): return cb[0] <= ib[0] and cb[1] <= ib[1] and cb[2] >= ib[2] and cb[3] >= ib[3]
    def _expand_face_to_body(self, face_bbox, img_shape):
        H, W, (x1, y1, x2, y2) = *img_shape[:2], *face_bbox
        w, h, cx = x2 - x1, y2 - y1, x1 + w / 2
        expansion_factors = self.config.seeding_defaults.face_to_body_expansion_factors
        new_w, new_h = min(W, w * expansion_factors[0]), min(H, h * expansion_factors[1])
        new_x1, new_y1 = max(0, cx - new_w / 2), max(0, y1 - h * expansion_factors[2])
        return [int(v) for v in [new_x1, new_y1, min(W, new_x1 + new_w) - new_x1, min(H, new_y1 + new_h) - new_y1]]

    def _final_fallback_box(self, img_shape):
        h, w, _ = img_shape
        fallback_box = self.config.seeding_defaults.final_fallback_box
        return [int(w * fallback_box[0]), int(h * fallback_box[1]), int(w * fallback_box[2]), int(h * fallback_box[3])]
    def _xyxy_to_xywh(self, box): x1, y1, x2, y2 = box; return [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]

    def _sam2_mask_for_bbox(self, frame_rgb_small, bbox_xywh):
        if not self.tracker or bbox_xywh is None:
            return None
        try:
            outputs = self.tracker.initialize(rgb_to_pil(frame_rgb_small), None, bbox=bbox_xywh)
            mask = outputs.get('pred_mask')
            if mask is not None: mask = postprocess_mask((mask * 255).astype(np.uint8), config=self.config, fill_holes=True, keep_largest_only=True)
            return mask
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            self.logger.warning(f"GPU error in mask generation: {e}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error in mask generation: {e}")
            return None

class SubjectMasker:
    def __init__(self, params, progress_queue, cancel_event, config: 'Config', frame_map=None, face_analyzer=None,
                 reference_embedding=None, person_detector=None, thumbnail_manager=None, niqe_metric=None, logger=None, face_landmarker=None):
        self.params, self.config, self.progress_queue, self.cancel_event = params, config, progress_queue, cancel_event
        self.logger = logger or EnhancedLogger(config=Config())
        self.frame_map = frame_map
        self.face_analyzer, self.reference_embedding, self.person_detector, self.face_landmarker = face_analyzer, reference_embedding, person_detector, face_landmarker
        self.dam_tracker, self.mask_dir, self.shots = None, None, []
        self._gdino, self._sam2_img = None, None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self.thumbnail_manager = thumbnail_manager if thumbnail_manager is not None else ThumbnailManager(self.logger, self.config)
        self.niqe_metric = niqe_metric
        self._initialize_models()
        self.seed_selector = SeedSelector(
            params=params,
            config=self.config,
            face_analyzer=face_analyzer,
            reference_embedding=reference_embedding,
            person_detector=person_detector,
            tracker=self.dam_tracker,
            gdino_model=self._gdino,
            logger=self.logger,
        )
        self.mask_propagator = MaskPropagator(params, self.dam_tracker, cancel_event, progress_queue, config=self.config, logger=self.logger)

    def _initialize_models(self): self._init_grounder(); self._initialize_tracker()
    def _init_grounder(self):
        if self._gdino is not None: return True
        self._gdino = get_grounding_dino_model(
            gdino_config_path=self.params.gdino_config_path,
            gdino_checkpoint_path=self.params.gdino_checkpoint_path,
            models_path=str(self.config.paths.models),
            grounding_dino_url=self.config.models.grounding_dino,
            device=self._device
        )
        return self._gdino is not None
    def _initialize_tracker(self):
        if self.dam_tracker: return True
        model_urls_tuple = tuple(self.config.models.dam4sam.items())
        self.dam_tracker = get_dam4sam_tracker(
            model_name=self.params.dam4sam_model_name,
            models_path=str(self.config.paths.models),
            model_urls_tuple=model_urls_tuple
        )
        return self.dam_tracker is not None

    def run_propagation(self, frames_dir: str, scenes_to_process) -> dict:
        self.mask_dir = Path(frames_dir) / "masks"
        self.mask_dir.mkdir(exist_ok=True)
        self.logger.info("Starting subject mask propagation...")
        if not self.dam_tracker:
            self.logger.error("Tracker not initialized; skipping masking.")
            return {}
        self.frame_map = self.frame_map or self._create_frame_map(frames_dir)
        mask_metadata, total_scenes = {}, len(scenes_to_process)
        for i, scene in enumerate(scenes_to_process):
            with safe_resource_cleanup():
                if self.cancel_event.is_set(): break
                self.logger.info(f"Masking scene {i+1}/{total_scenes}", user_context={'shot_id': scene.shot_id, 'start_frame': scene.start_frame, 'end_frame': scene.end_frame})
                seed_frame_num = scene.best_seed_frame
                shot_frames_data = self._load_shot_frames(frames_dir, scene.start_frame, scene.end_frame)
                if not shot_frames_data: continue
                frame_numbers, small_images, dims = zip(*shot_frames_data)
                try: seed_idx_in_shot = frame_numbers.index(seed_frame_num)
                except ValueError:
                    self.logger.warning(f"Seed frame {seed_frame_num} not found in loaded shot frames for {scene.shot_id}, skipping.")
                    continue
                bbox, seed_details = scene.seed_result.get('bbox'), scene.seed_result.get('details', {})
                if bbox is None:
                    for fn in frame_numbers:
                        if (fname := self.frame_map.get(fn)): mask_metadata[fname] = asdict(MaskingResult(error="Subject not found", shot_id=scene.shot_id))
                    continue
                masks, areas, empties, errors = self.mask_propagator.propagate(small_images, seed_idx_in_shot, bbox)
                for j, (original_fn, _, (h, w)) in enumerate(shot_frames_data):
                    frame_fname_webp = self.frame_map.get(original_fn)
                    if not frame_fname_webp: continue
                    frame_fname_png, mask_path = f"{Path(frame_fname_webp).stem}.png", self.mask_dir / f"{Path(frame_fname_webp).stem}.png"
                    result_args = {"shot_id": scene.shot_id, "seed_type": seed_details.get('type'), "seed_face_sim": seed_details.get('seed_face_sim'),
                                   "mask_area_pct": areas[j], "mask_empty": empties[j], "error": errors[j]}
                    if masks[j] is not None and np.any(masks[j]):
                        mask_full_res = cv2.resize(masks[j], (w, h), interpolation=cv2.INTER_NEAREST)
                        if mask_full_res.ndim == 3: mask_full_res = mask_full_res[:, :, 0]
                        cv2.imwrite(str(mask_path), mask_full_res)
                        mask_metadata[frame_fname_png] = asdict(MaskingResult(mask_path=str(mask_path), **result_args))
                    else:
                        mask_metadata[frame_fname_png] = asdict(MaskingResult(mask_path=None, **result_args))
        self.logger.success("Subject masking complete.")
        return mask_metadata

    def _create_frame_map(self, frames_dir): return create_frame_map(Path(frames_dir), self.logger)
    def _load_shot_frames(self, frames_dir, start, end):
        frames = []
        if not self.frame_map: self.frame_map = self._create_frame_map(frames_dir)
        thumb_dir = Path(frames_dir) / "thumbs"
        for fn in sorted(fn for fn in self.frame_map if start <= fn < end):
            thumb_p, thumb_img = thumb_dir / f"{Path(self.frame_map[fn]).stem}.webp", self.thumbnail_manager.get(thumb_dir / f"{Path(self.frame_map[fn]).stem}.webp")
            if thumb_img is None: continue
            frames.append((fn, thumb_img, thumb_img.shape[:2]))
        return frames

    def _select_best_seed_frame_in_scene(self, scene, frames_dir: str):
        if not self.params.pre_analysis_enabled:
            scene.best_seed_frame, scene.seed_metrics = scene.start_frame, {'reason': 'pre-analysis disabled'}
            return
        shot_frames = self._load_shot_frames(frames_dir, scene.start_frame, scene.end_frame)
        if not shot_frames:
            scene.best_seed_frame, scene.seed_metrics = scene.start_frame, {'reason': 'no frames loaded'}
            return
        candidates, scores = shot_frames[::max(1, self.params.pre_sample_nth)], []
        for frame_num, thumb_rgb, _ in candidates:
            niqe_score = 10.0
            if self.niqe_metric:
                img_tensor = (torch.from_numpy(thumb_rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0)
                with (torch.no_grad(), torch.cuda.amp.autocast(enabled=self._device == 'cuda')):
                    niqe_score = float(self.niqe_metric(img_tensor.to(self.niqe_metric.device)))
            face_sim = 0.0
            if self.face_analyzer and self.reference_embedding is not None:
                faces = self.face_analyzer.get(cv2.cvtColor(thumb_rgb, cv2.COLOR_RGB2BGR))
                if faces: face_sim = np.dot(max(faces, key=lambda x: x.det_score).normed_embedding, self.reference_embedding)
            scores.append((10 - niqe_score) + (face_sim * 10))
        if not scores:
            best_local_idx = 0
            scene.best_seed_frame = scene.start_frame # Fallback
            scene.seed_metrics = {'reason': 'pre-analysis failed, no scores', 'score': 0}
            return
        best_local_idx = int(np.argmax(scores))
        scene.best_seed_frame = candidates[best_local_idx][0]
        scene.seed_metrics = {'reason': 'pre-analysis complete', 'score': max(scores), 'best_niqe': niqe_score, 'best_face_sim': face_sim}

    def get_seed_for_frame(self, frame_rgb: np.ndarray, seed_config: dict): return self.seed_selector.select_seed(frame_rgb, current_params=seed_config)
    def get_mask_for_bbox(self, frame_rgb_small, bbox_xywh): return self.seed_selector._sam2_mask_for_bbox(frame_rgb_small, bbox_xywh)
    def draw_bbox(self, img_rgb, xywh, color=None, thickness=None):
        color = color or tuple(self.config.visualization.bbox_color)
        thickness = thickness or self.config.visualization.bbox_thickness
        x, y, w, h = map(int, xywh or [0, 0, 0, 0])
        img_out = img_rgb.copy()
        cv2.rectangle(img_out, (x, y), (x + w, y + h), color, thickness)
        return img_out

# --- PIPELINES ---

class ExtractionPipeline(Pipeline):
    def run(self):
        self.logger.info("Preparing video source...")
        vid_manager = VideoManager(self.params.source_path, self.config, self.params.max_resolution)
        video_path = Path(vid_manager.prepare_video(self.logger))
        output_dir = Path(self.config.paths.downloads) / video_path.stem
        output_dir.mkdir(exist_ok=True, parents=True)

        params_dict = asdict(self.params)
        params_dict['output_folder'] = str(output_dir)
        params_dict['video_path'] = str(video_path)
        # Ensure output_dir is canonical and exists before saving
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "run_config.json").open('w', encoding='utf-8') as f:
            json.dump(_to_json_safe(params_dict), f, indent=2)

        self.logger.info("Video ready", user_context={'path': sanitize_filename(video_path.name, self.config)})
        video_info = VideoManager.get_video_info(video_path)
        if self.params.scene_detect:
            self._run_scene_detection(video_path, output_dir)
        self._run_ffmpeg(video_path, output_dir, video_info)
        if self.cancel_event.is_set():
            self.logger.info("Extraction cancelled by user.")
            return
        self.logger.success("Extraction complete.")
        return {"done": True, "output_dir": str(output_dir), "video_path": str(video_path)}

    def _run_scene_detection(self, video_path, output_dir): return run_scene_detection(video_path, output_dir, self.logger)
    def _run_ffmpeg(self, video_path, output_dir, video_info): return run_ffmpeg_extraction(video_path, output_dir, video_info, self.params, self.progress_queue, self.cancel_event, self.logger, self.config)

class EnhancedExtractionPipeline(ExtractionPipeline):
    def __init__(self, config: 'Config', logger: 'EnhancedLogger', params: 'AnalysisParameters',
                 progress_queue: Queue, cancel_event: threading.Event):
        super().__init__(config, logger, params, progress_queue, cancel_event)
        self.error_handler = ErrorHandler(self.logger, self.config)
        self.run = self.error_handler.with_retry()(self.run)

class AnalysisPipeline(Pipeline):
    def __init__(self, config: 'Config', logger: 'EnhancedLogger', params: 'AnalysisParameters',
                 progress_queue: Queue, cancel_event: threading.Event,
                 thumbnail_manager: 'ThumbnailManager'):
        super().__init__(config, logger, params, progress_queue, cancel_event)
        self.output_dir = Path(self.params.output_folder)
        self.thumb_dir = self.output_dir / "thumbs"
        self.masks_dir = self.output_dir / "masks"
        self.metadata_path = self.output_dir / "metadata.jsonl"
        self.processing_lock = threading.Lock()
        self.face_analyzer, self.reference_embedding, self.mask_metadata, self.face_landmarker = None, None, {}, None
        self.scene_map, self.niqe_metric = {}, None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.thumbnail_manager = thumbnail_manager

    def _initialize_niqe_metric(self):
        if self.niqe_metric is None:
            try:
                import pyiqa
                self.niqe_metric = pyiqa.create_metric('niqe', device=self.device)
                self.logger.info("NIQE metric initialized successfully")
            except ImportError:
                self.logger.warning("pyiqa is not installed, NIQE metric is unavailable.")
            except Exception as e:
                self.logger.warning("Failed to initialize NIQE metric", extra={'error': e})

    def run_full_analysis(self, scenes_to_process):
        try:
            # Ensure metadata file is properly handled
            if self.metadata_path.exists():
                self.metadata_path.unlink()

            with open(self.metadata_path, 'w', encoding='utf-8') as f:
                f.write(json.dumps(_to_json_safe({"params": asdict(self.params)})) + '\n')

            self.scene_map = {s.shot_id: s for s in scenes_to_process}
            self.logger.info("Initializing Models")
            models = initialize_analysis_models(self.params, self.config, self.logger, self.device == 'cuda')
            self.face_analyzer = models['face_analyzer']
            self.reference_embedding = models['ref_emb']
            self.face_landmarker = models['face_landmarker']
            person_detector = models['person_detector']

            if self.face_analyzer and self.params.face_ref_img_path:
                self._process_reference_face()

            masker = SubjectMasker(self.params, self.progress_queue, self.cancel_event, self.config, self._create_frame_map(),
                                   self.face_analyzer, self.reference_embedding, person_detector, thumbnail_manager=self.thumbnail_manager,
                                   niqe_metric=self.niqe_metric, logger=self.logger, face_landmarker=self.face_landmarker)
            self.mask_metadata = masker.run_propagation(str(self.output_dir), scenes_to_process)
            self._initialize_niqe_metric()
            self._run_analysis_loop(scenes_to_process)
            if self.cancel_event.is_set():
                self.logger.info("Analysis cancelled by user.")
                return {"log": "Analysis cancelled.", "done": False}
            self.logger.success("Analysis complete.", extra={'output_dir': self.output_dir})
            return {"done": True, "metadata_path": str(self.metadata_path), "output_dir": str(self.output_dir)}
        except Exception as e:
            self.logger.error("Analysis pipeline failed", component="analysis", exc_info=True, extra={'error': str(e)})
            return {"error": str(e), "done": False}

    def _create_frame_map(self): return create_frame_map(self.output_dir, self.logger)
    def _process_reference_face(self):
        if not self.face_analyzer: return
        ref_path = Path(self.params.face_ref_img_path)
        if not ref_path.is_file(): raise FileNotFoundError(f"Reference face image not found: {ref_path}")
        self.logger.info("Processing reference face...")
        ref_img = cv2.imread(str(ref_path))
        if ref_img is None: raise ValueError("Could not read reference image.")
        ref_faces = self.face_analyzer.get(ref_img)
        if not ref_faces: raise ValueError("No face found in reference image.")
        self.reference_embedding = max(ref_faces, key=lambda x: x.det_score).normed_embedding
        self.logger.success("Reference face processed.")

    def _run_analysis_loop(self, scenes_to_process):
        frame_map = self._create_frame_map()
        all_frame_nums_to_process = {fn for scene in scenes_to_process for fn in range(scene.start_frame, scene.end_frame) if fn in frame_map}
        image_files_to_process = [self.thumb_dir / f"{Path(frame_map[fn]).stem}.webp" for fn in sorted(list(all_frame_nums_to_process))]
        self.logger.info(f"Analyzing {len(image_files_to_process)} frames")
        num_workers = 1 if self.params.disable_parallel else min(os.cpu_count() or 4, self.config.analysis.max_workers)
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(self._process_single_frame, path) for path in image_files_to_process]

            completed_count = 0
            for future in as_completed(futures):
                if self.cancel_event.is_set():
                    for f in futures:
                        f.cancel()
                    break

                try:
                    future.result()  # To raise exceptions if any
                except Exception as e:
                    self.logger.error(f"Error processing future: {e}")
                completed_count += 1

    def _process_single_frame(self, thumb_path):
        if self.cancel_event.is_set(): return
        if not (frame_num_match := re.search(r'frame_(\d+)', thumb_path.name)): return
        log_context = {'file': thumb_path.name}
        try:
            thumb_image_rgb = self.thumbnail_manager.get(thumb_path)
            if thumb_image_rgb is None: raise ValueError("Could not read thumbnail.")
            frame, base_filename = Frame(thumb_image_rgb, -1), thumb_path.name.replace('.webp', '.png')
            mask_meta = self.mask_metadata.get(base_filename, {})
            mask_thumb = None
            if mask_meta.get("mask_path"):
                mask_full_path = Path(mask_meta["mask_path"])
                if not mask_full_path.is_absolute(): mask_full_path = self.masks_dir / mask_full_path.name
                if mask_full_path.exists():
                    mask_full = cv2.imread(str(mask_full_path), cv2.IMREAD_GRAYSCALE)
                    if mask_full is not None: mask_thumb = cv2.resize(mask_full, (thumb_image_rgb.shape[1], thumb_image_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
            quality_conf = QualityConfig(
                sharpness_base_scale=self.config.sharpness_base_scale,
                edge_strength_base_scale=self.config.edge_strength_base_scale,
                # enable NIQE if the metric object was successfully initialized
                enable_niqe=(self.niqe_metric is not None)
            )
            frame.calculate_quality_metrics(thumb_image_rgb, quality_conf, self.logger, mask=mask_thumb, niqe_metric=self.niqe_metric, main_config=self.config, face_landmarker=self.face_landmarker)
            if self.params.enable_face_filter and self.reference_embedding is not None and self.face_analyzer: self._analyze_face_similarity(frame, thumb_image_rgb)
            meta = {"filename": base_filename, "metrics": asdict(frame.metrics)}
            if frame.face_similarity_score is not None: meta["face_sim"] = frame.face_similarity_score
            if frame.max_face_confidence is not None: meta["face_conf"] = frame.max_face_confidence
            meta.update(mask_meta)
            if meta.get("shot_id") is not None and (scene := self.scene_map.get(meta["shot_id"])) and scene.seed_metrics:
                meta['seed_face_sim'] = scene.seed_metrics.get('best_face_sim')
            if self.params.enable_dedup and imagehash: meta['phash'] = str(imagehash.phash(rgb_to_pil(thumb_image_rgb)))
            if frame.error: meta["error"] = frame.error
            if meta.get("mask_path"): meta["mask_path"] = Path(meta["mask_path"]).name
            with self.processing_lock:
                with self.metadata_path.open('a', encoding='utf-8') as f:
                    json.dump(_to_json_safe(meta), f)
                    f.write('\n')
        except Exception as e:
            self.logger.critical("Error processing frame", exc_info=True, extra={**log_context, 'error': e})
            with self.processing_lock:
                with self.metadata_path.open('a', encoding='utf-8') as f:
                    json.dump({"filename": thumb_path.name, "error": f"processing_failed: {e}"}, f)
                    f.write('\n')

    def _analyze_face_similarity(self, frame, image_rgb):
        try:
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            with self.processing_lock:
                faces = self.face_analyzer.get(image_bgr)
            if faces:
                best_face = max(faces, key=lambda x: x.det_score)
                distance = 1 - np.dot(best_face.normed_embedding, self.reference_embedding)
                frame.face_similarity_score, frame.max_face_confidence = 1.0 - float(distance), float(best_face.det_score)
        except Exception as e:
            frame.error = f"Face similarity failed: {e}"
            if "out of memory" in str(e) and torch.cuda.is_available():
                torch.cuda.empty_cache()

# --- FILTERING & SCENE LOGIC ---

def load_and_prep_filter_data(metadata_path, get_all_filter_keys):
    if not metadata_path or not Path(metadata_path).exists(): return [], {}
    with Path(metadata_path).open('r', encoding='utf-8') as f:
        try: next(f)
        except StopIteration: return [], {}
        all_frames = [json.loads(line) for line in f if line.strip()]
    metric_values = {}
    for k in get_all_filter_keys():
        values = np.asarray([f.get(k, f.get("metrics", {}).get(f"{k}_score")) for f in all_frames
                             if (f.get(k) is not None or f.get("metrics", {}).get(f"{k}_score") is not None)], dtype=float)
        if values.size > 0:
            if k == 'face_sim' or k == 'eyes_open':
                hist_range = (0, 1)
            elif k == 'yaw' or k == 'pitch':
                hist_range = (-45, 45)
            else:
                hist_range = (0, 100)
            counts, bins = np.histogram(values, bins=50, range=hist_range)
            metric_values[k], metric_values[f"{k}_hist"] = values.tolist(), (counts.tolist(), bins.tolist())
    return all_frames, metric_values

def build_all_metric_svgs(per_metric_values, get_all_filter_keys, logger):
    svgs = {}
    for k in get_all_filter_keys():
        if (h := per_metric_values.get(f"{k}_hist")): svgs[k] = histogram_svg(h, title="", logger=logger)
    return svgs

def histogram_svg(hist_data, title="", logger=None):
    if not hist_data or not plt: return ""
    try:
        counts, bins = hist_data
        if not isinstance(counts, list) or not isinstance(bins, list) or len(bins) != len(counts) + 1: return ""
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
    except Exception as e:
        if logger: logger.error("Failed to generate histogram SVG.", exc_info=True)
        return ""

def apply_all_filters_vectorized(all_frames_data, filters, config: 'Config'):
    if not all_frames_data: return [], [], Counter(), {}
    num_frames, filenames = len(all_frames_data), [f['filename'] for f in all_frames_data]
    quality_metric_keys = asdict(config.quality_weights).keys()
    metric_arrays = {k: np.array([f.get("metrics", {}).get(f"{k}_score", np.nan) for f in all_frames_data], dtype=np.float32) for k in quality_metric_keys}
    metric_arrays.update({"face_sim": np.array([f.get("face_sim", np.nan) for f in all_frames_data], dtype=np.float32),
                          "mask_area_pct": np.array([f.get("mask_area_pct", np.nan) for f in all_frames_data], dtype=np.float32),
                          "eyes_open": np.array([f.get("metrics", {}).get("eyes_open", np.nan) for f in all_frames_data], dtype=np.float32),
                          "yaw": np.array([f.get("metrics", {}).get("yaw", np.nan) for f in all_frames_data], dtype=np.float32),
                          "pitch": np.array([f.get("metrics", {}).get("pitch", np.nan) for f in all_frames_data], dtype=np.float32)})
    kept_mask, reasons, dedup_mask = np.ones(num_frames, dtype=bool), defaultdict(list), np.ones(num_frames, dtype=bool)
    if filters.get("enable_dedup") and imagehash and filters.get("dedup_thresh", 5) != -1:
        sorted_indices, hashes = sorted(range(num_frames), key=lambda i: filenames[i]), {i: imagehash.hex_to_hash(all_frames_data[i]['phash']) for i in range(num_frames) if 'phash' in all_frames_data[i]}
        for i in range(1, len(sorted_indices)):
            c_idx, p_idx = sorted_indices[i], sorted_indices[i - 1]
            if p_idx in hashes and c_idx in hashes and (hashes[p_idx] - hashes[c_idx]) <= filters.get("dedup_thresh", 5):
                if dedup_mask[c_idx]: reasons[filenames[c_idx]].append('duplicate')
                dedup_mask[c_idx] = False
    metric_filter_mask = np.ones(num_frames, dtype=bool)
    for k in quality_metric_keys:
        min_v, max_v = filters.get(f"{k}_min", 0), filters.get(f"{k}_max", 100)
        metric_filter_mask &= (np.nan_to_num(metric_arrays[k], nan=min_v) >= min_v) & (np.nan_to_num(metric_arrays[k], nan=max_v) <= max_v)
    if filters.get("face_sim_enabled"):
        face_sim_min, face_sim_values = filters.get("face_sim_min", 0.5), metric_arrays["face_sim"]
        has_face_sim = ~np.isnan(face_sim_values)
        metric_filter_mask[has_face_sim] &= (face_sim_values[has_face_sim] >= face_sim_min)
        if filters.get("require_face_match"): metric_filter_mask &= has_face_sim
    if filters.get("mask_area_enabled"):
        mask_area_min = filters.get("mask_area_pct_min", 1.0)
        metric_filter_mask &= (np.nan_to_num(metric_arrays["mask_area_pct"], nan=0.0) >= mask_area_min)

    eyes_open_min = filters.get("eyes_open_min", 0.3)
    metric_filter_mask &= (np.nan_to_num(metric_arrays["eyes_open"], nan=eyes_open_min) >= eyes_open_min)

    yaw_min, yaw_max = filters.get("yaw_min", -25), filters.get("yaw_max", 25)
    metric_filter_mask &= (np.nan_to_num(metric_arrays["yaw"], nan=yaw_min) >= yaw_min) & (np.nan_to_num(metric_arrays["yaw"], nan=yaw_max) <= yaw_max)

    pitch_min, pitch_max = filters.get("pitch_min", -25), filters.get("pitch_max", 25)
    metric_filter_mask &= (np.nan_to_num(metric_arrays["pitch"], nan=pitch_min) >= pitch_min) & (np.nan_to_num(metric_arrays["pitch"], nan=pitch_max) <= pitch_max)

    kept_mask = dedup_mask & metric_filter_mask
    metric_rejection_mask = ~metric_filter_mask & dedup_mask
    for i in np.where(metric_rejection_mask)[0]:
        for k in quality_metric_keys:
            min_v, max_v = filters.get(f"{k}_min", 0), filters.get(f"{k}_max", 100)
            if not (min_v <= metric_arrays[k][i] <= max_v): reasons[filenames[i]].append(f"{k}_{'low' if metric_arrays[k][i] < min_v else 'high'}")
        if filters.get("face_sim_enabled"):
            if metric_arrays["face_sim"][i] < filters.get("face_sim_min", 0.5): reasons[filenames[i]].append("face_sim_low")
            if filters.get("require_face_match") and np.isnan(metric_arrays["face_sim"][i]): reasons[filenames[i]].append("face_missing")
        if filters.get("mask_area_enabled") and metric_arrays["mask_area_pct"][i] < filters.get("mask_area_pct_min", 1.0): reasons[filenames[i]].append("mask_too_small")
    kept, rejected = [all_frames_data[i] for i in np.where(kept_mask)[0]], [all_frames_data[i] for i in np.where(~kept_mask)[0]]
    return kept, rejected, Counter(r for r_list in reasons.values() for r in r_list), reasons

def on_filters_changed(event: FilterEvent, thumbnail_manager, config: 'Config', logger=None):
    logger = logger or EnhancedLogger(config=Config())
    if not event.all_frames_data: return {"filter_status_text": "Run analysis to see results.", "results_gallery": []}
    filters = event.slider_values.copy()
    filters.update({"require_face_match": event.require_face_match, "dedup_thresh": event.dedup_thresh,
                    "face_sim_enabled": bool(event.per_metric_values.get("face_sim")),
                    "mask_area_enabled": bool(event.per_metric_values.get("mask_area_pct")),
                    "enable_dedup": any('phash' in f for f in event.all_frames_data) if event.all_frames_data else False})
    status_text, gallery_update = _update_gallery(event.all_frames_data, filters, event.output_dir, event.gallery_view,
                                                  event.show_overlay, event.overlay_alpha, thumbnail_manager, config, logger)
    return {"filter_status_text": status_text, "results_gallery": gallery_update}

def _update_gallery(all_frames_data, filters, output_dir, gallery_view, show_overlay, overlay_alpha, thumbnail_manager, config: 'Config', logger):
    kept, rejected, counts, per_frame_reasons = apply_all_filters_vectorized(all_frames_data, filters or {}, config)
    status_parts = [f"**Kept:** {len(kept)}/{len(all_frames_data)}"]
    if counts: status_parts.append(f"**Rejections:** {', '.join([f'{k}: {v}' for k, v in counts.most_common(3)])}")
    status_text, frames_to_show, preview_images = " | ".join(status_parts), rejected if gallery_view == "Rejected Frames" else kept, []
    if output_dir:
        output_path, thumb_dir, masks_dir = Path(output_dir), Path(output_dir) / "thumbs", Path(output_dir) / "masks"
        for f_meta in frames_to_show[:100]:
            thumb_path = thumb_dir / f"{Path(f_meta['filename']).stem}.webp"
            caption = f"Reasons: {', '.join(per_frame_reasons.get(f_meta['filename'], []))}" if gallery_view == "Rejected Frames" else ""
            thumb_rgb_np = thumbnail_manager.get(thumb_path)
            if thumb_rgb_np is None: continue
            if show_overlay and not f_meta.get("mask_empty", True) and (mask_name := f_meta.get("mask_path")):
                mask_path = masks_dir / mask_name
                if mask_path.exists():
                    mask_gray = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                    preview_images.append((render_mask_overlay(thumb_rgb_np, mask_gray, float(overlay_alpha), logger=logger), caption))
                else: preview_images.append((thumb_rgb_np, caption))
            else: preview_images.append((thumb_rgb_np, caption))
    return status_text, gr.update(value=preview_images, rows=1 if gallery_view == "Rejected Frames" else 2)

def reset_filters(all_frames_data, per_metric_values, output_dir, config, slider_keys, thumbnail_manager):
    output_values, slider_default_values = {}, []
    for key in slider_keys:
        metric_key, default_key = re.sub(r'_(min|max)$', '', key), 'default_max' if key.endswith('_max') else 'default_min'
        default_val = getattr(config.filter_defaults, metric_key)[default_key]
        output_values[f'slider_{key}'] = gr.update(value=default_val)
        slider_default_values.append(default_val)
    face_match_default, dedup_default = config.ui_defaults.require_face_match, config.filter_defaults.dedup_thresh['default']
    output_values.update({'require_face_match_input': gr.update(value=face_match_default), 'dedup_thresh_input': gr.update(value=dedup_default)})
    if all_frames_data:
        slider_defaults_dict = {key: val for key, val in zip(slider_keys, slider_default_values)}
        filter_event = FilterEvent(all_frames_data, per_metric_values, output_dir, "Kept Frames", config.gradio_defaults.show_mask_overlay, config.gradio_defaults.overlay_alpha, face_match_default, dedup_default, slider_defaults_dict)
        updates = on_filters_changed(filter_event, thumbnail_manager, config)
        output_values.update({'filter_status_text': updates['filter_status_text'], 'results_gallery': updates['results_gallery']})
    else:
        output_values.update({'filter_status_text': "Load an analysis to begin.", 'results_gallery': []})
    return output_values

def auto_set_thresholds(per_metric_values, p, slider_keys):
    updates = {}
    if not per_metric_values: return {f'slider_{key}': gr.update() for key in slider_keys}
    pmap = {k: float(np.percentile(np.asarray(vals, dtype=np.float32), p)) for k, vals in per_metric_values.items() if not k.endswith('_hist') and vals}
    for key in slider_keys:
        updates[f'slider_{key}'] = gr.update()
        if key.endswith('_min') and (metric := key[:-4]) in pmap: updates[f'slider_{key}'] = gr.update(value=round(pmap[metric], 2))
    return updates

def save_scene_seeds(scenes_list, output_dir_str, logger):
    if not scenes_list or not output_dir_str: return
    scene_seeds = {str(s['shot_id']): {'best_seed_frame': s.get('best_seed_frame'), 'seed_frame_idx': s.get('seed_frame_idx'), 'seed_type': s.get('seed_result', {}).get('details', {}).get('type'),
                                       'seed_config': s.get('seed_config', {}), 'status': s.get('status', 'pending'), 'seed_metrics': s.get('seed_metrics', {})} for s in scenes_list}
    try:
        (Path(output_dir_str) / "scene_seeds.json").write_text(json.dumps(_to_json_safe(scene_seeds), indent=2), encoding='utf-8')
        logger.info("Saved scene_seeds.json")
    except Exception as e: logger.error("Failed to save scene_seeds.json", exc_info=True)

def get_scene_status_text(scenes_list):
    if not scenes_list: return "No scenes loaded."
    return f"{sum(1 for s in scenes_list if s.get('status', 'pending') == 'included')}/{len(scenes_list)} scenes included for propagation."

def toggle_scene_status(scenes_list, selected_shot_id, new_status, output_folder, logger):
    if selected_shot_id is None or not scenes_list: return (scenes_list, get_scene_status_text(scenes_list), "No scene selected.")
    scene_found = False
    for s in scenes_list:
        if s['shot_id'] == selected_shot_id:
            s['status'], s['manual_status_change'], scene_found = new_status, True, True
            break
    if scene_found:
        save_scene_seeds(scenes_list, output_folder, logger)
        return (scenes_list, get_scene_status_text(scenes_list), f"Scene {selected_shot_id} status set to {new_status}.")
    return (scenes_list, get_scene_status_text(scenes_list), f"Could not find scene {selected_shot_id}.")

def apply_bulk_scene_filters(scenes, min_mask_area, min_face_sim, min_confidence, enable_face_filter, output_folder, logger):
    if not scenes: return [], "No scenes to filter."
    logger.info("Applying bulk scene filters", extra={"min_mask_area": min_mask_area, "min_face_sim": min_face_sim, "min_confidence": min_confidence, "enable_face_filter": enable_face_filter})
    for scene in scenes:
        scene['manual_status_change'] = False
        is_excluded = False
        details, seed_metrics = scene.get('seed_result', {}).get('details', {}), scene.get('seed_metrics', {})
        if details.get('mask_area_pct', 101) < min_mask_area: is_excluded = True
        if enable_face_filter and not is_excluded and seed_metrics.get('best_face_sim', 1.01) < min_face_sim: is_excluded = True
        if seed_metrics.get('score', 101.0) < min_confidence: is_excluded = True
        scene['status'] = 'excluded' if is_excluded else 'included'
    save_scene_seeds(scenes, output_folder, logger)
    return scenes, get_scene_status_text(scenes)

def apply_scene_overrides(scenes_list, selected_shot_id, prompt, box_th, text_th, output_folder, ana_ui_map_keys,
                          ana_input_components, cuda_available, thumbnail_manager, config: 'Config', logger: 'EnhancedLogger'):
    if selected_shot_id is None or not scenes_list: return (None, scenes_list, "No scene selected to apply overrides.")
    scene_idx, scene_dict = next(((i, s) for i, s in enumerate(scenes_list) if s['shot_id'] == selected_shot_id), (None, None))
    if scene_dict is None: return (None, scenes_list, "Error: Selected scene not found in state.")
    try:
        scene_dict['seed_config'] = {'text_prompt': prompt, 'box_threshold': box_th, 'text_threshold': text_th}
        ui_args = dict(zip(ana_ui_map_keys, ana_input_components))
        ui_args['output_folder'] = output_folder
        params, models = AnalysisParameters.from_ui(logger, config, **ui_args), initialize_analysis_models(AnalysisParameters.from_ui(logger, config, **ui_args), config, logger, cuda_available)
        masker = SubjectMasker(params, Queue(), threading.Event(), config, face_analyzer=models["face_analyzer"],
                               reference_embedding=models["ref_emb"], person_detector=models["person_detector"],
                               thumbnail_manager=thumbnail_manager, logger=logger)
        masker.frame_map = masker._create_frame_map(output_folder)
        seed_frame_num = scene_dict.get('best_seed_frame') or scene_dict.get('seed_frame_idx') or scene_dict.get('start_frame')
        if seed_frame_num is None:
            raise ValueError(f"Scene {scene_dict.get('shot_id')} has no seed or start frame.")
        fname = masker.frame_map.get(seed_frame_num)
        if not fname:
            raise ValueError(f"Framemap lookup failed for re-seeding shot {scene_dict.get('shot_id')} frame {seed_frame_num}.")
        thumb_rgb = thumbnail_manager.get(Path(output_folder) / "thumbs" / f"{Path(fname).stem}.webp")
        bbox, details = masker.get_seed_for_frame(thumb_rgb, scene_dict['seed_config'])
        scene_dict['seed_result'] = {'bbox': bbox, 'details': details}
        save_scene_seeds(scenes_list, output_folder, logger)
        # The call to a non-existent function is removed. The UI handler for this
        # function is responsible for regenerating previews if necessary.
        return (None, scenes_list, f"Scene {selected_shot_id} updated and saved.")
    except Exception as e:
        logger.error("Failed to apply scene overrides", exc_info=True)
        return None, scenes_list, f"[ERROR] {e}"

# --- Cleaned Scene Recomputation Workflow ---

def _create_analysis_context(config, logger, thumbnail_manager, cuda_available, ana_ui_map_keys, ana_input_components) -> SubjectMasker:
    """
    Factory function to create a fully initialized SubjectMasker from UI state.
    This centralizes context creation and ensures robustness.
    """
    ui_args = dict(zip(ana_ui_map_keys, ana_input_components))
    
    # --- Robust Path Handling ---
    output_folder_str = ui_args.get('output_folder')
    if not output_folder_str or not Path(output_folder_str).exists():
        raise FileNotFoundError(f"Output folder is not valid or does not exist: {output_folder_str}")
    resolved_outdir = Path(output_folder_str).resolve()
    ui_args['output_folder'] = str(resolved_outdir)

    # Create parameters and initialize all necessary models
    params = AnalysisParameters.from_ui(logger, config, **ui_args)
    models = initialize_analysis_models(params, config, logger, cuda_available)
    frame_map = create_frame_map(resolved_outdir, logger)
    
    if not frame_map:
        raise RuntimeError("Failed to create frame map. Check if frame_map.json exists and is valid.")

    # Return a fully configured SubjectMasker instance
    return SubjectMasker(
        params=params, progress_queue=Queue(), cancel_event=threading.Event(), config=config,
        frame_map=frame_map, face_analyzer=models["face_analyzer"],
        reference_embedding=models["ref_emb"], person_detector=models["person_detector"],
        niqe_metric=None, thumbnail_manager=thumbnail_manager, logger=logger
    )

def _recompute_single_preview(scene: dict, masker: SubjectMasker, overrides: dict, thumbnail_manager, logger):
    """
    Recomputes the seed for a single scene using a pre-built masker and specific overrides.
    Updates the scene dictionary in-place.
    """
    out_dir = Path(masker.params.output_folder)
    seed_frame_num = scene.get('best_seed_frame') or scene.get('start_frame')
    if seed_frame_num is None:
        raise ValueError(f"Scene {scene.get('shot_id')} has no seed frame number.")

    fname = masker.frame_map.get(int(seed_frame_num))
    if not fname:
        raise FileNotFoundError(f"Seed frame {seed_frame_num} not found in project's frame map.")

    thumb_rgb = thumbnail_manager.get(out_dir / "thumbs" / f"{Path(fname).stem}.webp")
    if thumb_rgb is None:
        raise FileNotFoundError(f"Thumbnail for frame {seed_frame_num} not found on disk.")

    # Create a temporary config for this specific seed selection run
    seed_config = {**asdict(masker.params), **overrides}
    
    # If the user provides a text prompt in the editor, it's a strong signal
    # to use the text-first seeding strategy for this specific re-computation.
    if overrides.get("text_prompt", "").strip():
        seed_config['primary_seed_strategy'] = "📝 By Text"
        logger.info(f"Recomputing scene {scene.get('shot_id')} with text-first strategy due to override.", extra={'prompt': overrides.get("text_prompt")})

    # Recompute seed and update scene dictionary
    bbox, details = masker.get_seed_for_frame(thumb_rgb, seed_config=seed_config)
    scene['seed_config'].update(overrides)
    scene['seed_result'] = {'bbox': bbox, 'details': details}
    
    # Update metrics that are displayed in the caption
    new_score = details.get('final_score') or details.get('conf') or details.get('dino_conf')
    if new_score is not None:
        scene.setdefault('seed_metrics', {})['score'] = new_score

    # Generate and save a new preview image
    mask = masker.get_mask_for_bbox(thumb_rgb, bbox) if bbox else None
    if mask is not None:
        h, w = mask.shape[:2]; area = (h * w)
        scene['seed_result']['details']['mask_area_pct'] = (np.sum(mask > 0) / area * 100.0) if area > 0 else 0.0

    overlay_rgb = render_mask_overlay(thumb_rgb, mask, 0.6, logger=logger) if mask is not None else masker.draw_bbox(thumb_rgb, bbox)
    previews_dir = out_dir / "previews"
    previews_dir.mkdir(parents=True, exist_ok=True)
    preview_path = previews_dir / f"scene_{int(scene['shot_id']):05d}.jpg"
    try:
        Image.fromarray(overlay_rgb).save(preview_path)
        scene['preview_path'] = str(preview_path)
    except Exception:
        logger.error(f"Failed to save preview for scene {scene['shot_id']}", exc_info=True)

def _wire_recompute_handler(config, logger, thumbnail_manager, scenes, shot_id, outdir, text_prompt, box_thresh, text_thresh,
                            view, ana_ui_map_keys, ana_input_components, cuda_available):
    """
    Orchestrator for the 'Recompute Preview' button. Cleanly builds context and executes logic.
    """
    try:
        # 1. Create the full analysis context from the current UI state
        # We need to combine the specific inputs from the recompute button with the general analysis inputs
        ui_args = dict(zip(ana_ui_map_keys, ana_input_components))
        ui_args['output_folder'] = outdir # This is critical
        
        # The ana_input_components passed to this handler already contain the full UI state,
        # so we can create the masker directly from them.
        masker = _create_analysis_context(config, logger, thumbnail_manager, cuda_available,
                                          ana_ui_map_keys, ana_input_components)

        # 2. Find the target scene and apply overrides
        scene_idx = next((i for i, s in enumerate(scenes) if s.get('shot_id') == shot_id), None)
        if scene_idx is None:
            return scenes, gr.update(), gr.update(), f"Error: Scene {shot_id} not found."

        overrides = {"text_prompt": text_prompt, "box_threshold": float(box_thresh), "text_threshold": float(text_thresh)}
        _recompute_single_preview(scenes[scene_idx], masker, overrides, thumbnail_manager, logger)

        # 3. Persist the changes and update the UI
        save_scene_seeds(scenes, outdir, logger)
        gallery_items, index_map = build_scene_gallery_items(scenes, view, outdir)
        msg = f"Scene {shot_id} preview recomputed successfully."
        return scenes, gr.update(value=gallery_items), gr.update(value=index_map), msg

    except Exception as e:
        logger.error("Failed to recompute scene preview", exc_info=True)
        return scenes, gr.update(), gr.update(), f"[ERROR] Recompute failed: {str(e)}"

# --- PIPELINE LOGIC ---

def execute_extraction(event: ExtractionEvent, progress_queue: Queue, cancel_event: threading.Event, logger: EnhancedLogger, config: Config):
    params_dict = asdict(event)
    if event.upload_video:
        source, dest = params_dict.pop('upload_video'), str(Path(config.paths.downloads) / Path(event.upload_video).name)
        shutil.copy2(source, dest)
        params_dict['source_path'] = dest
    params = AnalysisParameters.from_ui(logger, config, **params_dict)
    pipeline = EnhancedExtractionPipeline(config, logger, params, progress_queue, cancel_event)
    result = pipeline.run()
    if result and result.get("done"):
        yield {"log": "Extraction complete.", "status": f"Output: {result['output_dir']}",
               "extracted_video_path_state": result.get("video_path", ""), "extracted_frames_dir_state": result["output_dir"], "done": True}

def execute_pre_analysis(event: PreAnalysisEvent, progress_queue: Queue, cancel_event: threading.Event, logger: EnhancedLogger,
                         config: Config, thumbnail_manager, cuda_available):
    yield {"unified_log": "", "unified_status": "Starting Pre-Analysis..."}
    params_dict = asdict(event)
    final_face_ref_path = params_dict.get('face_ref_img_path')
    if event.face_ref_img_upload:
        ref_upload, dest = params_dict.pop('face_ref_img_upload'), Path(config.paths.downloads) / Path(event.face_ref_img_upload).name
        shutil.copy2(ref_upload, dest)
        params_dict['face_ref_img_path'] = str(dest)
        final_face_ref_path = str(dest)
    params, output_dir = AnalysisParameters.from_ui(logger, config, **params_dict), Path(params_dict['output_folder'])
    try:
        with (output_dir / "run_config.json").open('w', encoding='utf-8') as f:
            json.dump({k: v for k, v in params_dict.items() if k != 'face_ref_img_upload'}, f, indent=4)
    except Exception as e: logger.error(f"Failed to save run configuration: {e}", exc_info=True)
    scenes_path = output_dir / "scenes.json"
    if not scenes_path.exists():
        yield {"log": "[ERROR] scenes.json not found. Run extraction with scene detection."}
        return
    with scenes_path.open('r', encoding='utf-8') as f: scenes = [Scene(shot_id=i, start_frame=s, end_frame=e) for i, (s, e) in enumerate(json.load(f))]
    logger.info("Loading Models")
    models, niqe_metric = initialize_analysis_models(params, config, logger, cuda_available), None
    if params.pre_analysis_enabled and pyiqa: niqe_metric = pyiqa.create_metric('niqe', device=models['device'])
    masker = SubjectMasker(params, progress_queue, cancel_event, config, face_analyzer=models["face_analyzer"],
                           reference_embedding=models["ref_emb"], person_detector=models["person_detector"],
                           niqe_metric=niqe_metric, thumbnail_manager=thumbnail_manager, logger=logger)
    masker.frame_map = masker._create_frame_map(str(output_dir))
    def pre_analysis_task():
        logger.info(f"Pre-analyzing {len(scenes)} scenes")
        previews_dir = output_dir / "previews"
        previews_dir.mkdir(exist_ok=True)
        for i, scene in enumerate(scenes):
            if cancel_event.is_set(): break
            if not scene.best_seed_frame: masker._select_best_seed_frame_in_scene(scene, str(output_dir))
            fname = masker.frame_map.get(scene.best_seed_frame)
            if not fname: continue
            thumb_rgb = thumbnail_manager.get(output_dir / "thumbs" / f"{Path(fname).stem}.webp")
            if thumb_rgb is None: continue
            bbox, details = masker.get_seed_for_frame(thumb_rgb, seed_config=scene.seed_config or params)
            scene.seed_result = {'bbox': bbox, 'details': details}
            mask = masker.get_mask_for_bbox(thumb_rgb, bbox) if bbox else None
            if mask is not None:
                h, w = mask.shape[:2]
                scene.seed_result['details']['mask_area_pct'] = (np.sum(mask > 0) / (h * w)) * 100 if h * w > 0 else 0.0
            overlay_rgb = render_mask_overlay(thumb_rgb, mask, 0.6, logger=logger) if mask is not None else masker.draw_bbox(thumb_rgb, bbox)
            
            preview_path = previews_dir / f"scene_{scene.shot_id:05d}.jpg"
            try:
                Image.fromarray(overlay_rgb).save(preview_path)
                scene.preview_path = str(preview_path)
            except Exception as e:
                logger.error(f"Failed to save preview for scene {scene.shot_id}", exc_info=True)

            if scene.status == 'pending': scene.status = 'included'
        return {"done": True, "scenes": [asdict(s) for s in scenes]}
    result = pre_analysis_task()
    if result.get("done"):
        final_yield = {"log": "Pre-analysis complete.", "status": f"{len(result['scenes'])} scenes found.",
               "scenes": result['scenes'], "output_dir": str(output_dir), "done": True}
        if final_face_ref_path:
            final_yield['final_face_ref_path'] = final_face_ref_path
        yield final_yield

def validate_session_dir(path: str | Path) -> tuple[Path | None, str | None]:
    try:
        p = Path(path).expanduser().resolve()
        return (p if p.exists() and p.is_dir() else None,
                None if p.exists() and p.is_dir() else f"Session directory does not exist: {p}")
    except Exception as e:
        return None, f"Invalid session path: {e}"


def execute_session_load(
    app_ui,
    event: SessionLoadEvent,
    logger: EnhancedLogger,
    config: Config,
    thumbnail_manager,
):
    if not event.session_path or not event.session_path.strip():
        logger.error("No session path provided.", component="session_loader")
        yield {"log": "[ERROR] Please enter a path to a session directory.", "status": "Session load failed."}
        return

    session_path, error = validate_session_dir(event.session_path)
    if error:
        logger.error(f"Invalid session path provided: {event.session_path}", component="session_loader")
        yield {"log": f"[ERROR] {error}", "status": "Session load failed."}
        return

    config_path = session_path / "run_config.json"
    scene_seeds_path = session_path / "scene_seeds.json"
    metadata_path = session_path / "metadata.jsonl"

    def _resolve_output_dir(base: Path, output_folder: str | None) -> Path | None:
        if not output_folder:
            return None
        try:
            p = Path(output_folder)
            # If the given path already exists (absolute or relative to CWD), use it as-is.
            if p.exists():
                return p.resolve()
            # Otherwise, resolve relative to the session directory.
            if not p.is_absolute():
                resolved = (base / p).resolve()
                return resolved
            return p
        except Exception:
            return None

    with logger.operation("Load Session", component="session_loader"):
        # Stage 1: Load config
        logger.info("Loading config")
        if not config_path.exists():
            msg = f"Session load failed: run_config.json not found in {session_path}"
            logger.error(msg, component="session_loader")
            yield {
                "log": f"[ERROR] Could not find 'run_config.json' in the specified directory: {session_path}",
                "status": "Session load failed."
            }
            return

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                run_config = json.load(f)
        except json.JSONDecodeError as e:
            msg = f"run_config.json is invalid JSON: {e}"
            logger.error(msg, component="session_loader", error_type=type(e).__name__)
            yield {"log": f"[ERROR] {msg}", "status": "Session load failed."}
            return

        output_dir = _resolve_output_dir(session_path, run_config.get("output_folder"))
        # If resolution failed or points to a non-existent place, trust the session directory the user entered.
        if output_dir is None or not output_dir.exists():
            output_dir = session_path
            logger.warning("Output folder missing or invalid; defaulting to session directory.", component="session_loader")


        # Prepare initial UI updates from config
        updates = {
            "source_input": gr.update(value=run_config.get("source_path", "")),
            "max_resolution": gr.update(value=run_config.get("max_resolution", "1080")),
            "extraction_method_toggle_input": gr.update(value=("Recommended Thumbnails" if run_config.get('thumbnails_only', True) else "Legacy Full-Frame")),
            "thumb_megapixels_input": gr.update(value=run_config.get("thumb_megapixels", 0.5)),
            "ext_scene_detect_input": gr.update(value=run_config.get("scene_detect", True)),
            "method_input": gr.update(value=run_config.get("method", "scene")),
            "use_png_input": gr.update(value=run_config.get("use_png", False)),
            "pre_analysis_enabled_input": gr.update(value=run_config.get("pre_analysis_enabled", True)),
            "pre_sample_nth_input": gr.update(value=run_config.get("pre_sample_nth", 1)),
            "enable_face_filter_input": gr.update(value=run_config.get("enable_face_filter", False)),
            "face_model_name_input": gr.update(value=run_config.get("face_model_name", "buffalo_l")),
            "face_ref_img_path_input": gr.update(value=run_config.get("face_ref_img_path", "")),
            "text_prompt_input": gr.update(value=run_config.get("text_prompt", "")),
            "seed_strategy_input": gr.update(value=run_config.get("seed_strategy", "Largest Person")),
            "person_detector_model_input": gr.update(value=run_config.get("person_detector_model", "yolo11x.pt")),
            "dam4sam_model_name_input": gr.update(value=run_config.get("dam4sam_model_name", "sam21pp-T")),
            "enable_dedup_input": gr.update(value=run_config.get("enable_dedup", False)),
            "extracted_video_path_state": run_config.get("video_path", ""),
            "extracted_frames_dir_state": str(output_dir),
            # Ensure the state carries an absolute, normalized path for downstream steps
            "analysis_output_dir_state": str(Path(str(output_dir)).resolve()),
        }

        # Stage 2: Load scenes
        logger.info("Load scenes")
        scenes_as_dict: list[dict[str, Any]] = []
        scenes_json_path = session_path / "scenes.json"

        if scenes_json_path.exists():
            try:
                with open(scenes_json_path, "r", encoding="utf-8") as f:
                    scene_ranges = json.load(f)
                scenes_as_dict = [
                    {"shot_id": i, "start_frame": s, "end_frame": e}
                    for i, (s, e) in enumerate(scene_ranges)
                ]
                logger.info(f"Loaded {len(scenes_as_dict)} scene ranges from scenes.json")
            except Exception as e:
                logger.error(f"Failed to parse scenes.json: {e}", component="session_loader")
                # Yield an error if base scene structure can't be loaded
                yield {"log": f"[ERROR] Failed to read scenes.json: {e}", "status": "Session load failed."}
                return

        if scene_seeds_path.exists():
            try:
                with open(scene_seeds_path, "r", encoding="utf-8") as f:
                    scenes_from_file = json.load(f)
                
                # Create a lookup for faster merging
                seeds_lookup = {int(k): v for k, v in scenes_from_file.items()}
                
                # Merge data into the scenes_as_dict
                for scene in scenes_as_dict:
                    shot_id = scene.get("shot_id")
                    if shot_id in seeds_lookup:
                        scene.update(seeds_lookup[shot_id])

                # Auto-include missing scene statuses on load (set to "included" as approved)
                for s in scenes_as_dict:
                    s.setdefault("status", "included")

                logger.info(f"Merged data for {len(seeds_lookup)} scenes from {scene_seeds_path}", component="session_loader")
            except Exception as e:
                logger.warning(f"Failed to parse or merge scene_seeds.json: {e}", component="session_loader", error_type=type(e).__name__)

        # Stage 3: Load previews (with cap and fallback)
        logger.info("Load previews")

        if scenes_as_dict and output_dir:
            updates.update({
                "scenes_state": scenes_as_dict,
                "propagate_masks_button": gr.update(interactive=True),
                "seeding_results_column": gr.update(visible=True),
                "propagation_group": gr.update(visible=True),
                "scene_filter_status": get_scene_status_text(scenes_as_dict),
                "scene_face_sim_min_input": gr.update(
                    visible=any((s.get("seed_metrics") or {}).get("best_face_sim") is not None for s in scenes_as_dict)
                ),
            })

            # Initialize scene gallery for loaded session
            gallery_items, index_map = build_scene_gallery_items(scenes_as_dict, "Kept", str(output_dir))
            updates.update({
                "scene_gallery": gr.update(value=gallery_items),
                "scene_gallery_index_map_state": index_map
            })

        # Enable filtering if metadata exists
        if metadata_path.exists():
            updates.update({
                "analysis_metadata_path_state": str(metadata_path),
                "filtering_tab": gr.update(interactive=True),
            })
            logger.info(
                f"Found analysis metadata at {metadata_path}. Filtering tab enabled.",
                component="session_loader",
            )

        # Always finalize successfully if we got here
        updates.update({
            "log": f"Successfully loaded session from: {session_path}",
            "status": "... Session loaded. You can now proceed from where you left off.",
        })
        yield updates

def execute_propagation(event: PropagationEvent, progress_queue: Queue, cancel_event: threading.Event, logger: EnhancedLogger,
                        config: Config, thumbnail_manager, cuda_available):
    scene_fields = {f.name for f in fields(Scene)}
    scenes_to_process = [
        Scene(**{k: v for k, v in s.items() if k in scene_fields})
        for s in event.scenes if s.get('status') == 'included'
    ]
    if not scenes_to_process:
        yield {"log": "No scenes were included for propagation.", "status": "Propagation skipped."}
        return

    params = AnalysisParameters.from_ui(logger, config, **asdict(event.analysis_params))
    pipeline = AnalysisPipeline(
        config=config,
        logger=logger,
        params=params,
        progress_queue=progress_queue,
        cancel_event=cancel_event,
        thumbnail_manager=thumbnail_manager
    )
    result = pipeline.run_full_analysis(scenes_to_process)
    if result and result.get("done"):
        yield {"log": "Propagation and analysis complete.", "status": f"Metadata saved to {result['metadata_path']}",
               "output_dir": result['output_dir'], "metadata_path": result['metadata_path'], "done": True}

# --- Scene gallery helpers (module-level) ---
def scene_matches_view(scene: dict, view: str) -> bool:
    status = scene.get('status', 'pending')
    if view == "All":
        return status in ("included", "excluded", "pending")
    if view == "Kept":
        return status == "included"
    if view == "Rejected":
        return status == "excluded"
    return False

def scene_caption(s: dict) -> str:
    shot = s.get('shot_id', '?')
    start_f = s.get('start_frame', '?')
    end_f = s.get('end_frame', '?')
    metrics = s.get('seed_metrics', {}) or {}
    face = metrics.get('best_face_sim')
    conf = metrics.get('score')
    mask = (s.get('seed_result', {}).get('details', {}) or {}).get('mask_area_pct')
    bits = [f"Scene {shot} [{start_f}-{end_f}]"]
    if conf is not None: bits.append(f"conf {conf:.2f}")
    if face is not None: bits.append(f"face {face:.2f}")
    if mask is not None: bits.append(f"mask {mask:.1f}%")
    return " | ".join(bits)

def scene_thumb(s: dict, output_dir: str) -> str | None:
    p = s.get('preview_path')
    if p and os.path.isfile(p):
        return p
    shot_id = s.get('shot_id')
    if shot_id is not None:
        candidate = os.path.join(output_dir, "previews", f"scene_{int(shot_id):05d}.jpg")
        if os.path.isfile(candidate):
            return candidate
    return None

def build_scene_gallery_items(scenes: list[dict], view: str, output_dir: str):
    items: list[tuple[str | None, str]] = []
    index_map: list[int] = []
    # Sort scenes by shot_id but keep track of original index in scenes list
    if not scenes:
        return [], []
    indexed_scenes = sorted(list(enumerate(scenes)), key=lambda x: x[1].get('shot_id', 0))

    for original_index, s in indexed_scenes:
        if not scene_matches_view(s, view):
            continue
        img = scene_thumb(s, output_dir)
        if img is None:
            continue
        cap = scene_caption(s)
        items.append((img, cap))
        index_map.append(original_index)
    return items, index_map

# --- UI ---

class AppUI:
    def __init__(self, config: 'Config', logger: 'EnhancedLogger', progress_queue: Queue,
                 cancel_event: threading.Event, thumbnail_manager: 'ThumbnailManager'):
        self.config = config
        self.logger = logger
        self.progress_queue = progress_queue
        self.cancel_event = cancel_event
        self.thumbnail_manager = thumbnail_manager
        self.components, self.cuda_available = {}, torch.cuda.is_available()
        self.ext_ui_map_keys = ['source_path', 'upload_video', 'method', 'interval', 'nth_frame', 'fast_scene',
                                'max_resolution', 'use_png', 'extraction_method_toggle', 'thumb_megapixels', 'scene_detect']
        self.ana_ui_map_keys = ['output_folder', 'video_path', 'resume', 'enable_face_filter', 'face_ref_img_path', 'face_ref_img_upload',
                                'face_model_name', 'enable_subject_mask', 'dam4sam_model_name', 'person_detector_model', 'seed_strategy',
                                'scene_detect', 'enable_dedup', 'text_prompt', 'box_threshold', 'text_threshold', 'min_mask_area_pct',
                                'sharpness_base_scale', 'edge_strength_base_scale', 'gdino_config_path', 'gdino_checkpoint_path',
                                'pre_analysis_enabled', 'pre_sample_nth', 'primary_seed_strategy']
        self.session_load_keys = ['unified_log', 'unified_status', 'progress_bar', 'progress_details', 'cancel_button', 'pause_button',
                                  'source_input', 'max_resolution', 'extraction_method_toggle_input', 'thumb_megapixels_input', 'ext_scene_detect_input',
                                  'method_input', 'use_png_input', 'pre_analysis_enabled_input', 'pre_sample_nth_input', 'enable_face_filter_input',
                                  'face_model_name_input', 'face_ref_img_path_input', 'text_prompt_input', 'seed_strategy_input',
                                  'person_detector_model_input', 'dam4sam_model_name_input', 'enable_dedup_input', 'extracted_video_path_state',
                                  'extracted_frames_dir_state', 'analysis_output_dir_state', 'analysis_metadata_path_state', 'scenes_state',
                                  'propagate_masks_button', 'seeding_results_column', 'propagation_group',
                                  'scene_filter_status', 'scene_face_sim_min_input', 'filtering_tab',
                                  'scene_gallery', 'scene_gallery_index_map_state']

    def build_ui(self):
        css = """.plot-and-slider-column { max-width: 560px !important; margin: auto; } .scene-editor { border: 1px solid #444; padding: 10px; border-radius: 5px; } .log-container > .gr-utils-error { display: none !important; } .progress-details { font-size: 0.8em; color: #888; text-align: center; }"""
        with gr.Blocks(theme=gr.themes.Default(), css=css) as demo:
            self._build_header()
            with gr.Accordion("🔄 resume previous Session", open=False):
                with gr.Row():
                    self._create_component('session_path_input', 'textbox', {'label': "Load previous run", 'placeholder': "Path to a previous run's output folder..."})
                    self._create_component('load_session_button', 'button', {'value': "📂 Load Session"})
                    self._create_component('save_config_button', 'button', {'value': "💾 Save Current Config"})
            self._build_main_tabs()
            self._build_footer()
            self._create_event_handlers()
        return demo

    def _create_component(self, name, comp_type, kwargs):
        comp_map = {'button': gr.Button, 'textbox': gr.Textbox, 'dropdown': gr.Dropdown, 'slider': gr.Slider, 'checkbox': gr.Checkbox,
                    'file': gr.File, 'radio': gr.Radio, 'gallery': gr.Gallery, 'plot': gr.Plot, 'markdown': gr.Markdown, 'html': gr.HTML, 'number': gr.Number}
        self.components[name] = comp_map[comp_type](**kwargs)
        return self.components[name]

    def _build_header(self):
        gr.Markdown("# 🎬 Frame Extractor & Analyzer v2.0")
        if not self.cuda_available: gr.Markdown("⚠️ **CPU Mode** — GPU-dependent features are disabled or will be slow.")

    def _build_main_tabs(self):
        with gr.Tabs() as main_tabs:
            self.components['main_tabs'] = main_tabs
            with gr.Tab("📹 1. Frame Extraction"): self._create_extraction_tab()
            with gr.Tab("🎯 2. Seeding & Scene Selection", id=1) as analysis_tab: self.components['analysis_tab'] = analysis_tab; self._create_analysis_tab()
            with gr.Tab("📊 3. Filtering & Export", id=2) as filtering_tab: self.components['filtering_tab'] = filtering_tab; self._create_filtering_tab()

    def _build_footer(self):
        with gr.Row():
            with gr.Column(scale=2): self._create_component('unified_log', 'textbox', {'label': "📋 Processing Log", 'lines': 10, 'interactive': False, 'autoscroll': True})
            with gr.Column(scale=1):
                self._create_component('unified_status', 'textbox', {'label': "📊 Status Summary", 'lines': 2, 'interactive': False})
                with gr.Row(): self.components['progress_bar'] = gr.Progress()

    def _create_extraction_tab(self):
        gr.Markdown("### Step 1: Provide a Video Source")
        with gr.Row():
            with gr.Column(scale=2): self._create_component('source_input', 'textbox', {'label': "Video URL or Local Path", 'placeholder': "Enter YouTube URL or local video file path", 'info': "The application can download videos directly from YouTube or use a video file you have on your computer."})
            with gr.Column(scale=1): self._create_component('max_resolution', 'dropdown', {'choices': self.config.choices.max_resolution, 'value': self.config.ui_defaults.max_resolution, 'label': "Max Download Resolution", 'info': "For YouTube videos, select the maximum resolution to download. 'Maximum available' will get the best quality possible."})
        self._create_component('upload_video_input', 'file', {'label': "Or Upload a Video File", 'file_types': ["video"], 'type': "filepath"})
        gr.Markdown("---"); gr.Markdown("### Step 2: Configure Extraction Method")
        # Toggle between Recommended Thumbnails and Legacy Full-Frame
        self._create_component('extraction_method_toggle_input', 'radio', {
            'label': "Extraction Method",
            'choices': self.config.choices.extraction_method_toggle,
            'value': "Recommended Thumbnails" if self.config.ui_defaults.thumbnails_only else "Legacy Full-Frame",
            'info': "Choose between the modern, efficient thumbnail-based workflow (Recommended) or the classic direct full-frame extraction (Legacy)."
        })

        # Recommended (thumbnails) group
        with gr.Group(visible=self.config.ui_defaults.thumbnails_only) as thumbnail_group:
            self.components['thumbnail_group'] = thumbnail_group
            gr.Markdown("**Recommended Method (Thumbnail Extraction):** This is the fastest and most efficient way to process your video. It quickly extracts low-resolution, lightweight thumbnails for every frame. This allows you to perform scene analysis, find the best shots, and select your desired frames *before* extracting the final, full-resolution images. This workflow saves significant time and disk space.")
            self._create_component('thumb_megapixels_input', 'slider', {
                'label': "Thumbnail Size (MP)", 'minimum': 0.1, 'maximum': 2.0, 'step': 0.1,
                'value': self.config.ui_defaults.thumb_megapixels,
                'info': "Controls the resolution of the extracted thumbnails. Higher values create larger, more detailed thumbnails but increase extraction time and disk usage. 0.5 MP is a good balance for most videos."
            })
            self._create_component('ext_scene_detect_input', 'checkbox', {
                'label': "Use Scene Detection",
                'value': self.config.ui_defaults.scene_detect,
                'info': "Automatically detects scene changes in the video. This is highly recommended as it groups frames into logical shots, making it much easier to find the best content in the next step."
            })

        # Legacy (full‑frame) group
        with gr.Group(visible=not self.config.ui_defaults.thumbnails_only) as legacy_group:
            self.components['legacy_group'] = legacy_group
            gr.Markdown("**Legacy Method (Direct Full-Frame Extraction):** This method extracts full-resolution frames directly from the video based on the selected criteria. Be aware that this can be very slow and consume a large amount of disk space, especially for long, high-resolution videos. It is recommended for advanced users or specific use cases where the thumbnail workflow is not suitable.")
            self._create_component('method_input', 'dropdown', {
                'choices': self.config.choices.method,
                'value': self.config.ui_defaults.method,
                'label': "Extraction Method",
                'info': textwrap.dedent("""\
                    - **Keyframes:** Extracts only the keyframes (I-frames). Good for a quick summary.
                    - **Interval:** Extracts one frame every X seconds.
                    - **Every Nth Frame:** Extracts one frame every N video frames.
                    - **All:** Extracts every single frame. (Warning: massive disk usage and time).
                    - **Scene:** Extracts frames where a scene change is detected.""")
            })
            self._create_component('interval_input', 'textbox', {
                'label': "Interval (seconds)",
                'value': self.config.ui_defaults.interval,
                'visible': False
            })
            self._create_component('nth_frame_input', 'textbox', {
                'label': "N-th Frame Value",
                'value': self.config.ui_defaults.nth_frame,
                'visible': False
            })
            self._create_component('fast_scene_input', 'checkbox', {
                'label': "Fast Scene Detect (Lower Quality)",
                'info': "Uses a faster but less precise algorithm for scene detection.",
                'visible': False
            })
            self._create_component('use_png_input', 'checkbox', {
                'label': "Save as PNG (slower, larger files)",
                'value': self.config.ui_defaults.use_png,
                'info': "PNG is lossless but results in much larger files than JPG. Recommended only if you need perfect image fidelity."
            })
        gr.Markdown("---"); gr.Markdown("### Step 3: Start Extraction")
        self.components.update({'start_extraction_button': gr.Button("🚀 Start Extraction", variant="primary")})

    def _create_analysis_tab(self):
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 🎯 Step 1: Choose Your Seeding Strategy")
                gr.Markdown(
                    """
                    The goal of this step is to find the best single frame (the "seed") in each scene to represent the subject you're interested in.
                    This seed frame is then used in the next step to track the subject across the entire scene.
                    - **By Face**: Best for tracking a specific person. Requires a clear reference photo.
                    - **By Text**: Good for tracking objects or people by description (e.g., "person in a red shirt").
                    - **Face + Text Fallback**: The most robust option. It tries to find the person first, but if it can't, it will use the text prompt as a backup.
                    - **Find Prominent Person**: A quick, automatic option that finds the largest or most central person. Less precise but very fast.
                    """
                )
                self._create_component('primary_seed_strategy_input', 'radio', {'choices': self.config.choices.primary_seed_strategy, 'value': self.config.choices.primary_seed_strategy[2], 'label': "Primary Seeding Strategy", 'info': "Select the main method for identifying the subject in each scene. This initial identification is called the 'seed'."})
                with gr.Group(visible=("By Face" in self.config.choices.primary_seed_strategy[2] or "Fallback" in self.config.choices.primary_seed_strategy[2])) as face_seeding_group:
                    self.components['face_seeding_group'] = face_seeding_group
                    gr.Markdown("#### 👤 Configure Face Seeding"); gr.Markdown("This strategy prioritizes finding a specific person. Upload a clear, frontal photo of the person you want to track. The system will analyze each scene to find the frame where this person is most clearly visible and use it as the starting point (the 'seed').")
                    with gr.Row():
                        self._create_component('face_ref_img_upload_input', 'file', {'label': "Upload Face Reference Image", 'type': "filepath"})
                        with gr.Column():
                            self._create_component('face_ref_img_path_input', 'textbox', {'label': "Or provide a local file path"})
                            self._create_component('enable_face_filter_input', 'checkbox', {'label': "Enable Face Similarity (must be checked for face seeding)", 'value': ("By Face" in self.config.choices.primary_seed_strategy[2] or "Fallback" in self.config.choices.primary_seed_strategy[2]), 'interactive': False, 'visible': False})
                with gr.Group(visible=("By Text" in self.config.choices.primary_seed_strategy[2] or "Fallback" in self.config.choices.primary_seed_strategy[2])) as text_seeding_group:
                    self.components['text_seeding_group'] = text_seeding_group
                    gr.Markdown("#### 📝 Configure Text Seeding"); gr.Markdown("This strategy uses a text description to find the subject. It's useful for identifying objects, or people described by their clothing or appearance when a reference photo isn't available.")
                    self._create_component('text_prompt_input', 'textbox', {'label': "Text Prompt", 'placeholder': "e.g., 'a woman in a red dress'", 'value': self.config.ui_defaults.text_prompt, 'info': "Describe the main subject (e.g., 'player wearing number 10', 'person in the green shirt')."})
                with gr.Group(visible=("Prominent Person" in self.config.choices.primary_seed_strategy[2])) as auto_seeding_group:
                    self.components['auto_seeding_group'] = auto_seeding_group
                    gr.Markdown("#### 🧑‍🤝‍🧑 Configure Prominent Person Seeding"); gr.Markdown("This is a simple, fully automatic mode. It uses an object detector (YOLO) to find all people in the scene and then selects one based on a simple rule, like who is largest or most central. It's fast but less precise, as it doesn't use face identity or text descriptions.")
                    self._create_component('seed_strategy_input', 'dropdown', {'choices': self.config.choices.seed_strategy, 'value': "Largest Person", 'label': "Selection Method", 'info': "'Largest' picks the person taking up the most screen area. 'Center-most' picks the person closest to the frame's center."})
                with gr.Accordion("Advanced Settings", open=False):
                    gr.Markdown("These settings control the underlying models and analysis parameters. Adjust them only if you understand their effect.")
                    self._create_component('pre_analysis_enabled_input', 'checkbox', {'label': 'Enable Pre-Analysis to find best seed frame', 'value': self.config.ui_defaults.pre_analysis_enabled, 'info': "Analyzes a subset of frames in each scene to automatically find the highest quality frame to use as the 'seed' for masking. Highly recommended."})
                    self._create_component('pre_sample_nth_input', 'number', {'label': 'Sample every Nth thumbnail for pre-analysis', 'value': self.config.ui_defaults.pre_sample_nth, 'interactive': True, 'info': "For faster pre-analysis, check every Nth frame in a scene instead of all of them. A value of 5 is a good starting point."})
                    self._create_component('person_detector_model_input', 'dropdown', {'choices': self.config.choices.person_detector_model, 'value': self.config.ui_defaults.person_detector_model, 'label': "Person Detector Model", 'info': "YOLO Model for finding people. 'x' (large) is more accurate but slower; 's' (small) is much faster but may miss people."})
                    self._create_component('face_model_name_input', 'dropdown', {'choices': self.config.choices.face_model_name, 'value': self.config.ui_defaults.face_model_name, 'label': "Face Recognition Model", 'info': "InsightFace model for face matching. 'l' (large) is more accurate; 's' (small) is faster and uses less memory."})
                    self._create_component('dam4sam_model_name_input', 'dropdown', {'choices': self.config.choices.dam4sam_model_name, 'value': self.config.ui_defaults.dam4sam_model_name, 'label': "Mask Tracking Model", 'info': "The Segment Anything 2 model used for tracking the subject mask across frames. Larger models (L) are more robust but use more VRAM; smaller models (T) are faster."})
                    self._create_component('enable_dedup_input', 'checkbox', {'label': "Enable Deduplication (pHash)", 'value': self.config.ui_defaults.enable_dedup, 'info': "Computes a 'perceptual hash' for each frame. This is used in the Filtering tab to help remove visually similar or identical frames."})
                self._create_component('start_pre_analysis_button', 'button', {'value': '🌱 Find & Preview Scene Seeds', 'variant': 'primary'})
                with gr.Group(visible=False) as propagation_group:
                    self.components['propagation_group'] = propagation_group
            with gr.Column(scale=2, visible=False) as seeding_results_column:
                self.components['seeding_results_column'] = seeding_results_column
                gr.Markdown("### 🎭 Step 2: Review & Refine Scenes")
                with gr.Accordion("Bulk Scene Actions & Filters", open=False):
                    self._create_component('scene_filter_status', 'markdown', {'value': 'No scenes loaded.'})
                    with gr.Row():
                        self._create_component('scene_mask_area_min_input', 'slider', {'label': "Min Seed Mask Area %", 'minimum': 0.0, 'maximum': 100.0, 'value': self.config.min_mask_area_pct, 'step': 0.1})
                        self._create_component('scene_face_sim_min_input', 'slider', {'label': "Min Seed Face Sim", 'minimum': 0.0, 'maximum': 1.0, 'value': 0.0, 'step': 0.05, 'visible': False})
                        self._create_component('scene_confidence_min_input', 'slider', {'label': "Min Seed Confidence", 'minimum': 0.0, 'maximum': 10.0, 'value': 0.0, 'step': 0.05})


                with gr.Accordion("Scene Gallery", open=True):
                    self._create_component(
                        'scene_gallery_view_toggle',
                        'radio',
                        {
                            'label': "Show",
                            'choices': ["Kept", "Rejected", "All"],
                            'value': "Kept"
                        }
                    )
                    # A gallery to visualize scenes by status
                    self.components['scene_gallery'] = gr.Gallery(
                        label="Scenes",
                        columns=6,
                        height=320,
                        show_label=True,
                        allow_preview=True
                    )
                with gr.Accordion("Scene Editor", open=False, elem_classes="scene-editor") as sceneeditoraccordion:
                    self.components["sceneeditoraccordion"] = sceneeditoraccordion
                    self._create_component("sceneeditorstatusmd", "markdown", {"value": "Select a scene to edit."})
                    with gr.Row():
                        self._create_component("sceneeditorpromptinput", "textbox", {"label": "Per-Scene Text Prompt", "info": "Override the main text prompt for this scene only. This will force a text-based search."})
                    with gr.Row():
                        info_box = "Confidence for detecting an object's bounding box. Higher = fewer, more confident detections."
                        self._create_component("sceneeditorboxthreshinput", "slider", {
                            "label": "Box Thresh", "minimum": 0.0, "maximum": 1.0, "step": 0.05, "info": info_box,
                            "value": self.config.grounding_dino_params.box_threshold,})
                        info_text = "Confidence for matching the prompt to an object. Higher = stricter text match."
                        self._create_component("sceneeditortextthreshinput", "slider", {
                            "label": "Text Thresh", "minimum": 0.0, "maximum": 1.0, "step": 0.05, "info": info_text,
                            "value": self.config.grounding_dino_params.text_threshold,})
                    with gr.Row():
                        self._create_component("scenerecomputebutton", "button", {"value": "▶️Recompute Preview"})
                        self._create_component("sceneincludebutton", "button", {"value": "✅keep scene"})
                        self._create_component("sceneexcludebutton", "button", {"value": "❌reject scene"})
                gr.Markdown("---"); gr.Markdown("### 🔬 Step 3: Propagate Masks"); gr.Markdown("Once you are satisfied with the seeds, propagate the masks to the rest of the frames in the selected scenes.")
                self._create_component('propagate_masks_button', 'button', {'value': '🔬 Propagate Masks on Kept Scenes', 'variant': 'primary', 'interactive': False})

    def _create_filtering_tab(self):
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 🎛️ Filter Controls")
                gr.Markdown("Use these controls to refine your selection of frames. You can set minimum and maximum thresholds for various quality metrics.")
                self._create_component('auto_pctl_input', 'slider', {'label': 'Auto-Threshold Percentile', 'minimum': 1, 'maximum': 99, 'value': self.config.gradio_defaults.auto_pctl_input, 'step': 1, 'info': "Quickly set all 'Min' sliders to a certain percentile of the data. For example, setting this to 75 and clicking 'Apply' will automatically reject the bottom 75% of frames for each metric."})
                with gr.Row():
                    self._create_component('apply_auto_button', 'button', {'value': 'Apply Percentile to Mins'})
                    self._create_component('reset_filters_button', 'button', {'value': "Reset Filters"})
                self._create_component('filter_status_text', 'markdown', {'value': "Load an analysis to begin."})
                self.components['metric_plots'], self.components['metric_sliders'] = {}, {}
                with gr.Accordion("Deduplication", open=True, visible=True):
                    f_def = self.config.filter_defaults.dedup_thresh
                    self._create_component('dedup_thresh_input', 'slider', {'label': "Similarity Threshold", 'minimum': f_def['min'], 'maximum': f_def['max'], 'value': f_def['default'], 'step': f_def['step'], 'info': "Filters out visually similar frames. A lower value is stricter (more filtering). A value of 0 means only identical images will be removed. Set to -1 to disable."})
                for metric_name, open_default in [('quality_score', True), ('niqe', False), ('sharpness', True), ('edge_strength', True), ('contrast', True),
                                                  ('brightness', False), ('entropy', False), ('face_sim', False), ('mask_area_pct', False),
                                                  ('eyes_open', True), ('yaw', True), ('pitch', True)]:
                    if not hasattr(self.config.filter_defaults, metric_name): continue
                    f_def = getattr(self.config.filter_defaults, metric_name)
                    with gr.Accordion(metric_name.replace('_', ' ').title(), open=open_default):
                        gr.Markdown(self.get_metric_description(metric_name), elem_classes="metric-description")
                        with gr.Column(elem_classes="plot-and-slider-column"):
                            self.components['metric_plots'][metric_name] = self._create_component(f'plot_{metric_name}', 'html', {'visible': False})
                            self.components['metric_sliders'][f"{metric_name}_min"] = self._create_component(f'slider_{metric_name}_min', 'slider', {'label': "Min", 'minimum': f_def['min'], 'maximum': f_def['max'], 'value': f_def['default_min'], 'step': f_def['step'], 'interactive': True, 'visible': False})
                            if 'default_max' in f_def: self.components['metric_sliders'][f"{metric_name}_max"] = self._create_component(f'slider_{metric_name}_max', 'slider', {'label': "Max", 'minimum': f_def['min'], 'maximum': f_def['max'], 'value': f_def['default_max'], 'step': f_def['step'], 'interactive': True, 'visible': False})
                            if metric_name == "face_sim": self._create_component('require_face_match_input', 'checkbox', {'label': "Reject if no face", 'value': self.config.ui_defaults.require_face_match, 'visible': False, 'info': "If checked, any frame without a detected face that meets the similarity threshold will be rejected."})
            with gr.Column(scale=2):
                with gr.Group(visible=False) as results_group:
                    self.components['results_group'] = results_group
                    gr.Markdown("### 🖼️ Step 2: Review Results")
                    with gr.Row():
                        self._create_component('gallery_view_toggle', 'radio', {'choices': self.config.choices.gallery_view, 'value': "Kept Frames", 'label': "Show in Gallery"})
                        self._create_component('show_mask_overlay_input', 'checkbox', {'label': "Show Mask Overlay", 'value': self.config.gradio_defaults.show_mask_overlay})
                        self._create_component('overlay_alpha_slider', 'slider', {'label': "Overlay Alpha", 'minimum': 0.0, 'maximum': 1.0, 'value': self.config.gradio_defaults.overlay_alpha, 'step': 0.1})
                    self._create_component('results_gallery', 'gallery', {'columns': [4, 6, 8], 'rows': 2, 'height': 'auto', 'preview': True, 'allow_preview': True, 'object_fit': 'contain'})
                with gr.Group(visible=False) as export_group:
                    self.components['export_group'] = export_group
                    gr.Markdown("### 📤 Step 3: Export")
                    self._create_component('export_button', 'button', {'value': "Export Kept Frames", 'variant': "primary"})
                    with gr.Accordion("Export Options", open=True):
                        with gr.Row():
                            self._create_component('enable_crop_input', 'checkbox', {'label': "✂️ Crop to Subject", 'value': self.config.export_options.enable_crop})
                            self._create_component('crop_padding_input', 'slider', {'label': "Padding %", 'value': self.config.export_options.crop_padding})
                        self._create_component('crop_ar_input', 'textbox', {'label': "Crop ARs", 'value': self.config.export_options.crop_ars, 'info': "Comma-separated list (e.g., 16:9, 1:1). The best-fitting AR for each subject's mask will be chosen automatically."})

    def get_all_filter_keys(self):
        return list(asdict(self.config.quality_weights).keys()) + ["quality_score", "face_sim", "mask_area_pct", "eyes_open", "yaw", "pitch"]

    def get_metric_description(self, metric_name):
        descriptions = {
            "quality_score": "A weighted average of all other quality metrics, providing an overall 'goodness' score for the frame.",
            "niqe": "Natural Image Quality Evaluator. A no-reference, opinion-unaware quality score. Lower is generally better, but it's scaled here so higher is better (like other metrics). Tends to favor clean, natural-looking images.",
            "sharpness": "Measures the amount of fine detail and edge clarity. Higher values indicate a sharper, more in-focus image.",
            "edge_strength": "Specifically measures the prominence of edges in the image. It's related to sharpness but focuses more on strong outlines.",
            "contrast": "The difference between the brightest and darkest parts of the image. Very high or very low contrast can be undesirable.",
            "brightness": "The overall lightness or darkness of the image.",
            "entropy": "Measures the amount of 'information' or complexity in the image. A very blurry or plain image will have low entropy.",
            "face_sim": "Face Similarity. How closely the best-detected face in the frame matches the reference face image. Only appears if a reference face is used.",
            "mask_area_pct": "Mask Area Percentage. The percentage of the screen taken up by the subject's mask. Useful for filtering out frames where the subject is too small or distant.",
            "eyes_open": "A score from 0.0 to 1.0 indicating how open the eyes are. A value of 1.0 means the eyes are fully open, and 0.0 means they are fully closed.",
            "yaw": "The rotation of the head around the vertical axis (turning left or right).",
            "pitch": "The rotation of the head around the side-to-side axis (looking up or down)."
        }
        return descriptions.get(metric_name, "No description available.")

    def _create_event_handlers(self):
        self.components.update({'extracted_video_path_state': gr.State(""), 'extracted_frames_dir_state': gr.State(""),
                                'analysis_output_dir_state': gr.State(""), 'analysis_metadata_path_state': gr.State(""),
                                'all_frames_data_state': gr.State([]), 'per_metric_values_state': gr.State({}),
                                'scenes_state': gr.State([]), 'selected_scene_id_state': gr.State(None),
                                'scene_gallery_index_map_state': gr.State([])})
        self._setup_visibility_toggles(); self._setup_pipeline_handlers(); self._setup_filtering_handlers(); self._setup_bulk_scene_handlers()
        self.components['save_config_button'].click(
            lambda: self.config.save_config('config_dump.yml'), [], []
        ).then(lambda: "Configuration saved to config_dump.yml", [], self.components['unified_log'])

class EnhancedAppUI(AppUI):
    def __init__(self, config: 'Config', logger: 'EnhancedLogger', progress_queue: Queue,
                 cancel_event: threading.Event, thumbnail_manager: 'ThumbnailManager'):
        super().__init__(config, logger, progress_queue, cancel_event, thumbnail_manager)
        self.enhanced_logger = logger
        self.performance_metrics, self.log_filter_level, self.all_logs = {}, "INFO", []

    def _build_footer(self):
        with gr.Row():
            with gr.Column(scale=3):
                self._create_component('unified_log', 'textbox', {'label': '📋 Enhanced Processing Log', 'lines': 15, 'interactive': False, 'autoscroll': True, 'elem_classes': ['log-container']})
                with gr.Row():
                    self._create_component('log_level_filter', 'dropdown', {'choices': self.config.choices.log_level, 'value': 'DEBUG', 'label': 'Log Level Filter', 'scale': 1})
                    self._create_component('clear_logs_button', 'button', {'value': '🗑️ Clear Logs', 'scale': 1})
                    self._create_component('export_logs_button', 'button', {'value': '📥 Export Logs', 'scale': 1})
            with gr.Column(scale=1):
                self._create_component('unified_status', 'html', {'value': self._format_status_display('Idle', 0, 'Ready'),})
                self.components['progress_bar'] = gr.Progress()
                self._create_component('progress_details', 'html', {'value': '', 'elem_classes': ['progress-details']})
                with gr.Row():
                    self._create_component('pause_button', 'button', {'value': '⏸️ Pause', 'interactive': False})
                    self._create_component('cancel_button', 'button', {'value': '⏹️ Cancel', 'interactive': False})

    def _format_metric_card(self, label: str, value: str) -> str: return f"""<div class="metric-card"><div class="metric-value">{value}</div><div class="metric-label">{label}</div></div>"""
    def _format_status_display(self, op: str, prog: float, stage: str) -> str: return f"""<div style="margin: 10px 0;"><h4>{op}</h4><div style="background: #e0e0e0; border-radius: 4px; height: 20px; margin: 5px 0;"><div style="background: #007bff; height: 100%; width: {int(prog*100)}%; border-radius: 4px; transition: width 0.3s ease;"></div></div><div style="font-size: 12px; color: #666;">{stage} - {prog:.1%}</div></div>"""

    def _run_task_with_progress(self, task_func, output_components, progress, *args):
        self.cancel_event.clear()
        yield {self.components['cancel_button']: gr.update(interactive=True), self.components['pause_button']: gr.update(interactive=False)}
        op_name = getattr(task_func, '__name__', 'Unknown Task').replace('_wrapper', '')
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(task_func, *args)
            while not future.done():
                if self.cancel_event.is_set(): future.cancel(); break
                try:
                    msg, update_dict = self.progress_queue.get(timeout=0.1), {}
                    if "log" in msg:
                        self.all_logs.append(msg['log'])
                        if self.log_filter_level.upper() == "DEBUG" or f"[{self.log_filter_level.upper()}]" in msg['log']:
                            update_dict[self.components['unified_log']] = gr.update(value="\n".join([l for l in self.all_logs if self.log_filter_level.upper() == "DEBUG" or f"[{self.log_filter_level.upper()}]" in l][-1000:]))
                    if update_dict: yield update_dict
                except Empty: pass
                time.sleep(0.05)
        final_updates, final_msg, final_label = {}, "✅ Task completed successfully.", "Complete"
        try:
            result_dict = future.result() or {}
            if self.cancel_event.is_set(): final_msg, final_label = "⏹️ Task cancelled by user.", "Cancelled"
            else: final_msg, final_updates = result_dict.get("unified_log", final_msg), result_dict
        except Exception as e:
            self.enhanced_logger.error("Task execution failed in UI runner", exc_info=True)
            final_msg, final_label = f"❌ Task failed: {e}", "Failed"
        self.all_logs.append(f"[{final_label.upper()}] {final_msg}")
        filtered_logs = [l for l in self.all_logs if self.log_filter_level.upper() == "DEBUG" or f"[{self.log_filter_level.upper()}]" in l]
        final_updates_with_comps = {self.components.get(k): v for k, v in final_updates.items() if self.components.get(k)}
        final_updates_with_comps[self.components['unified_log']] = "\n".join(filtered_logs[-1000:])
        progress(1.0, final_label)
        final_updates_with_comps[self.components['unified_status']] = self._format_status_display(op_name, 1.0, final_label)
        final_updates_with_comps[self.components['progress_details']] = ""
        final_updates_with_comps[self.components['cancel_button']], final_updates_with_comps[self.components['pause_button']] = gr.update(interactive=False), gr.update(interactive=False)
        yield final_updates_with_comps

    def on_select_for_edit(self, evt: gr.SelectData, scenes, view, indexmap, outputdir):
        # validate selection
        if not scenes or not indexmap or evt is None or evt.index is None:
            return (scenes, get_scene_status_text(scenes), gr.update(), indexmap,
                    None, gr.update(), gr.update(), gr.update(), gr.update(), gr.update())
        if not (0 <= evt.index < len(indexmap)):
            return (scenes, get_scene_status_text(scenes), gr.update(), indexmap,
                    None, gr.update(), gr.update(), gr.update(), gr.update(), gr.update())

        scene_idx_in_state = indexmap[evt.index]
        if not (0 <= scene_idx_in_state < len(scenes)):
            return (scenes, get_scene_status_text(scenes), gr.update(), indexmap,
                    None, gr.update(), gr.update(), gr.update(), gr.update(), gr.update())

        scene = scenes[scene_idx_in_state]
        cfg = scene.get("seed_config") or {}
        shotid = scene.get("shot_id")
        # Editor status text
        if scene.get("start_frame") is not None and scene.get("end_frame") is not None:
            status_md = f"Editing Scene {shotid}  •  Frames {scene['start_frame']}-{scene['end_frame']}"
        else:
            status_md = f"Editing Scene {shotid}"
        # Prefill values with scene seedconfig or defaults
        prompt = cfg.get("text_prompt", "")
        boxth = cfg.get("box_threshold", self.config.grounding_dino_params.box_threshold)
        textth = cfg.get("text_threshold", self.config.grounding_dino_params.text_threshold)
        return (
            scenes,
            get_scene_status_text(scenes),
            gr.update(),                           # no immediate gallery change here
            indexmap,
            shotid,
            gr.update(value=status_md),            # sceneeditorstatusmd
            gr.update(value=prompt),               # sceneeditorpromptinput
            gr.update(value=boxth),                # sceneeditorboxthreshinput
            gr.update(value=textth),               # sceneeditortextthreshinput
            gr.update(open=True),                  # sceneeditoraccordion
        )

    def on_recompute(self, scenes, selected_shotid, prompt, boxth, textth, outputfolder, *anaargs):
        # Update seed for the selected scene using per-scene prompt/thresholds
        _, updated_scenes, msg = apply_scene_overrides(
            scenes, selected_shotid, prompt, boxth, textth,
            outputfolder, self.ana_ui_map_keys, anaargs,
            self.cuda_available, self.thumbnail_manager, self.config, self.logger
        )
        # After reseeding, refresh gallery items so the saved preview is reread
        gallery_items, index_map = build_scene_gallery_items(updated_scenes, self.components["scene_gallery_view_toggle"].value, outputfolder)
        return (
            updated_scenes,
            gr.update(value=gallery_items),       # scenegallery
            gr.update(value=index_map),           # scenegalleryindexmapstate
            gr.update(value=msg),                 # sceneeditorstatusmd (feedback)
        )

    def on_editor_toggle(self, scenes, selected_shotid, outputfolder, view, new_status):
        scenes, status_text, _ = self.on_toggle_scene_status(scenes, selected_shotid, outputfolder, new_status)
        items, index_map = build_scene_gallery_items(scenes, view, outputfolder)
        return scenes, status_text, gr.update(value=items), gr.update(value=index_map)

    def _create_event_handlers(self):
        super()._create_event_handlers()
        c = self.components
        c['cancel_button'].click(lambda: self.cancel_event.set(), [], [])
        c['clear_logs_button'].click(lambda: (self.all_logs.clear(), "")[1], [], c['unified_log'])
        c['log_level_filter'].change(lambda level: (setattr(self, 'log_filter_level', level), "\n".join([l for l in self.all_logs if level.upper() == "DEBUG" or f"[{level.upper()}]" in l][-1000:]))[1], c['log_level_filter'], c['unified_log'])

    def _create_pre_analysis_event(self, *args):
        ui_args = dict(zip(self.ana_ui_map_keys, args))
        strategy = ui_args.get('primary_seed_strategy', '🧑‍🤝‍🧑 Find Prominent Person')
        if strategy == "👤 By Face": ui_args.update({'enable_face_filter': True, 'text_prompt': ""})
        elif strategy == "📝 By Text": ui_args.update({'enable_face_filter': False, 'face_ref_img_path': "", 'face_ref_img_upload': None})
        elif strategy == "🔄 Face + Text Fallback": ui_args['enable_face_filter'] = True
        elif strategy == "🧑‍🤝‍🧑 Find Prominent Person": ui_args.update({'enable_face_filter': False, 'text_prompt': "", 'face_ref_img_path': "", 'face_ref_img_upload': None})
        for k, v_type, default in [('pre_sample_nth', int, 5), ('min_mask_area_pct', float, 0.0), ('sharpness_base_scale', float, 1.0), ('edge_strength_base_scale', float, 1.0)]:
            try: ui_args[k] = v_type(ui_args.get(k)) if v_type != int or int(ui_args.get(k)) > 0 else default
            except (TypeError, ValueError): ui_args[k] = default
        return PreAnalysisEvent(**ui_args)

    def run_extraction_wrapper(self, *args):
        ui_args = dict(zip(self.ext_ui_map_keys, args))
        # Map radio toggle -> boolean expected by pipeline
        if 'extraction_method_toggle' in ui_args:
            ui_args['thumbnails_only'] = (ui_args.pop('extraction_method_toggle') == "Recommended Thumbnails")
        
        # Construct ExtractionEvent with only its defined fields
        event_fields = [f.name for f in dataclasses.fields(ExtractionEvent)]
        event_args = {k: v for k, v in ui_args.items() if k in event_fields}
        event = ExtractionEvent(**event_args)
        try:
            for result in execute_extraction(event, self.progress_queue, self.cancel_event, self.enhanced_logger, self.config):
                if isinstance(result, dict):
                    if self.cancel_event.is_set():
                        return {"unified_log": "Extraction cancelled."}
                    if result.get("done"):
                        return {"unified_log": result.get("log", "✅ Extraction completed successfully."),
                                "extracted_video_path_state": result.get("video_path", "") or result.get("extracted_video_path_state", ""),
                                "extracted_frames_dir_state": result.get("output_dir", "") or result.get("extracted_frames_dir_state", "")}
            return {"unified_log": "❌ Extraction failed."}
        except Exception as e: raise

    def run_pre_analysis_wrapper(self, *args):
        event = self._create_pre_analysis_event(*args)
        try:
            for result in execute_pre_analysis(event, self.progress_queue, self.cancel_event, self.enhanced_logger, self.config, self.thumbnail_manager, self.cuda_available):
                if isinstance(result, dict):
                    if self.cancel_event.is_set():
                        return {"unified_log": "Pre-analysis cancelled."}
                    if result.get("done"):
                        scenes = result.get('scenes', [])
                        if scenes: save_scene_seeds(scenes, result['output_dir'], self.enhanced_logger)
                        updates = {"unified_log": result.get("log", "✅ Pre-analysis completed successfully."),
                                   "scenes_state": scenes, "propagate_masks_button": gr.update(interactive=True), "scene_filter_status": get_scene_status_text(scenes),
                                   "scene_face_sim_min_input": gr.update(visible=any(s.get('seed_metrics', {}).get('best_face_sim') is not None for s in scenes)),
                                   "seeding_results_column": gr.update(visible=True), "propagation_group": gr.update(visible=True)}
                        # Initialize scene gallery
                        gallery_items, index_map = build_scene_gallery_items(scenes, "Kept", result.get('output_dir', ''))
                        updates.update({
                            "scene_gallery": gr.update(value=gallery_items),
                            "scene_gallery_index_map_state": index_map
                        })
                        if result.get("final_face_ref_path"):
                            updates["face_ref_img_path_input"] = result["final_face_ref_path"]
                        return updates
            return {"unified_log": "❌ Pre-analysis failed."}
        except Exception as e: raise

    def run_propagation_wrapper(self, scenes, *args):
        event = PropagationEvent(output_folder=self._create_pre_analysis_event(*args).output_folder, video_path=self._create_pre_analysis_event(*args).video_path,
                                 scenes=scenes, analysis_params=self._create_pre_analysis_event(*args))
        try:
            for result in execute_propagation(event, self.progress_queue, self.cancel_event, self.enhanced_logger, self.config, self.thumbnail_manager, self.cuda_available):
                if isinstance(result, dict):
                    if self.cancel_event.is_set():
                        return {"unified_log": "Propagation cancelled."}
                    if result.get("done"):
                        return {"unified_log": result.get("log", "✅ Propagation completed successfully."), "analysis_output_dir_state": result.get('output_dir', ""),
                                "analysis_metadata_path_state": result.get('metadata_path', ""), "filtering_tab": gr.update(interactive=True)}
            return {"unified_log": "❌ Propagation failed."}
        except Exception as e: raise

    def run_session_load_wrapper(self, session_path):
        try:
            final_result = {}
            for result in execute_session_load(self, SessionLoadEvent(session_path=session_path), self.enhanced_logger, self.config, self.thumbnail_manager):
                if isinstance(result, dict):
                    if 'log' in result: result['unified_log'] = result.pop('log')
                    final_result.update(result)
            return final_result
        except Exception as e: raise

    def _setup_visibility_toggles(self):
        c = self.components
        c['method_input'].change(lambda m: (gr.update(visible=m == 'interval'), gr.update(visible=m == 'scene'), gr.update(visible=m == 'every_nth_frame')),
                                 c['method_input'], [c['interval_input'], c['fast_scene_input'], c['nth_frame_input']])

        # Show/hide groups when switching extraction method
        c['extraction_method_toggle_input'].change(
            lambda method: (
                gr.update(visible=(method == "Recommended Thumbnails")),
                gr.update(visible=(method == "Legacy Full-Frame")),
            ),
            inputs=[c['extraction_method_toggle_input']],
            outputs=[c['thumbnail_group'], c['legacy_group']],
        )

        c['primary_seed_strategy_input'].change(lambda s: {c['face_seeding_group']: gr.update(visible=s == "👤 By Face" or s == "🔄 Face + Text Fallback"),
                                                           c['text_seeding_group']: gr.update(visible=s == "📝 By Text" or s == "🔄 Face + Text Fallback"),
                                                           c['auto_seeding_group']: gr.update(visible=s == "🧑‍🤝‍🧑 Find Prominent Person"),
                                                           c['enable_face_filter_input']: gr.update(value=s == "👤 By Face" or s == "🔄 Face + Text Fallback")},
                                                 [c['primary_seed_strategy_input']], [c['face_seeding_group'], c['text_seeding_group'],
                                                                                     c['auto_seeding_group'], c['enable_face_filter_input']])

    def _setup_pipeline_handlers(self):
        c = self.components
        all_outputs = [v for v in c.values() if hasattr(v, "_id")]
        
        def session_load_handler(session_path, progress=gr.Progress()):
            session_load_keys_filtered = [k for k in self.session_load_keys if k != 'progress_bar']
            session_load_outputs = [c[key] for key in session_load_keys_filtered if key in c and hasattr(c[key], "_id")]
            yield from self._run_task_with_progress(
                self.run_session_load_wrapper, session_load_outputs, progress, session_path
            )

        def extraction_handler(*args, progress=gr.Progress()):
            yield from self._run_task_with_progress(
                self.run_extraction_wrapper, all_outputs, progress, *args
            )

        def pre_analysis_handler(*args, progress=gr.Progress()):
            yield from self._run_task_with_progress(
                self.run_pre_analysis_wrapper, all_outputs, progress, *args
            )

        def propagation_handler(scenes, *args, progress=gr.Progress()):
            yield from self._run_task_with_progress(
                self.run_propagation_wrapper, all_outputs, progress, scenes, *args
            )

        c['load_session_button'].click(
            fn=session_load_handler,
            inputs=[c['session_path_input']],
            outputs=all_outputs,
            show_progress="hidden"
        )
        ext_inputs = [c[{'source_path': 'source_input', 'upload_video': 'upload_video_input', 'max_resolution': 'max_resolution',
                         'scene_detect': 'ext_scene_detect_input', **{k: f"{k}_input" for k in self.ext_ui_map_keys if k not in
                         ['source_path', 'upload_video', 'max_resolution', 'scene_detect']}}[k]] for k in self.ext_ui_map_keys]
        self.ana_input_components = [c.get(k, k) for k in [{'output_folder': 'extracted_frames_dir_state', 'video_path': 'extracted_video_path_state',
                                                           'resume': gr.State(self.config.ui_defaults.resume), 'enable_face_filter': 'enable_face_filter_input',
                                                           'face_ref_img_path': 'face_ref_img_path_input', 'face_ref_img_upload': 'face_ref_img_upload_input',
                                                           'face_model_name': 'face_model_name_input', 'enable_subject_mask': gr.State(self.config.ui_defaults.enable_subject_mask),
                                                           'dam4sam_model_name': 'dam4sam_model_name_input', 'person_detector_model': 'person_detector_model_input',
                                                           'seed_strategy': 'seed_strategy_input', 'scene_detect': 'ext_scene_detect_input',
                                                           'enable_dedup': 'enable_dedup_input', 'text_prompt': 'text_prompt_input',
                                                           'box_threshold': gr.State(self.config.grounding_dino_params.box_threshold),
                                                           'text_threshold': gr.State(self.config.grounding_dino_params.text_threshold),
                                                           'min_mask_area_pct': gr.State(self.config.min_mask_area_pct),
                                                           'sharpness_base_scale': gr.State(self.config.sharpness_base_scale),
                                                           'edge_strength_base_scale': gr.State(self.config.edge_strength_base_scale),
                                                           'gdino_config_path': gr.State(str(self.config.paths.grounding_dino_config)),
                                                           'gdino_checkpoint_path': gr.State(str(self.config.paths.grounding_dino_checkpoint)),
                                                           'pre_analysis_enabled': 'pre_analysis_enabled_input', 'pre_sample_nth': 'pre_sample_nth_input',
                                                           'primary_seed_strategy': 'primary_seed_strategy_input'}[k] for k in self.ana_ui_map_keys]]
        prop_inputs = [c['scenes_state']] + self.ana_input_components
        c['start_extraction_button'].click(fn=extraction_handler,
                                         inputs=ext_inputs, outputs=all_outputs, show_progress="hidden").then(lambda d: gr.update(selected=1) if d else gr.update(), c['extracted_frames_dir_state'], c['main_tabs'])
        c['start_pre_analysis_button'].click(fn=pre_analysis_handler,
                                           inputs=self.ana_input_components, outputs=all_outputs, show_progress="hidden")
        c['propagate_masks_button'].click(fn=propagation_handler,
                                        inputs=prop_inputs, outputs=all_outputs, show_progress="hidden").then(lambda p: gr.update(selected=2) if p else gr.update(), c['analysis_metadata_path_state'], c['main_tabs'])

    def _setup_bulk_scene_handlers(self):
        c = self.components

        def _refresh_scene_gallery(scenes, view, output_dir):
            items, index_map = build_scene_gallery_items(scenes, view, output_dir)
            return gr.update(value=items), index_map

        # On view toggle change
        c['scene_gallery_view_toggle'].change(
            _refresh_scene_gallery,
            [c['scenes_state'], c['scene_gallery_view_toggle'], c['extracted_frames_dir_state']],
            [c['scene_gallery'], c['scene_gallery_index_map_state']]
        )

        c['scene_gallery'].select(
            self.on_select_for_edit,
            inputs=[c['scenes_state'], c['scene_gallery_view_toggle'], c['scene_gallery_index_map_state'], c['extracted_frames_dir_state']],
            outputs=[
                c['scenes_state'], c['scene_filter_status'], c['scene_gallery'], c['scene_gallery_index_map_state'],
                c['selected_scene_id_state'],
                c['sceneeditorstatusmd'], c['sceneeditorpromptinput'], c['sceneeditorboxthreshinput'], c['sceneeditortextthreshinput'],
                c['sceneeditoraccordion'],
            ]
        )

        # Wire recompute to use current editor controls and state
        c['scenerecomputebutton'].click(
            fn=lambda scenes, shot_id, outdir, view, txt, bth, tth, *ana_args: _wire_recompute_handler(
                self.config, self.enhanced_logger, self.thumbnail_manager, scenes, shot_id, outdir, txt, bth, tth, view,
                self.ana_ui_map_keys, ana_args, self.cuda_available
            ),
            inputs=[
                c['scenes_state'],
                c['selected_scene_id_state'],
                c['analysis_output_dir_state'],
                c['scene_gallery_view_toggle'],
                c['sceneeditorpromptinput'], c['sceneeditorboxthreshinput'], c['sceneeditortextthreshinput'],
                *self.ana_input_components
            ],
            outputs=[
                c['scenes_state'],
                c['scene_gallery'],
                c['scene_gallery_index_map_state'],
                c['sceneeditorstatusmd'],
            ],
        )

        c['sceneincludebutton'].click(
            lambda s, sid, out, v: self.on_editor_toggle(s, sid, out, v, "included"),
            inputs=[c['scenes_state'], c['selected_scene_id_state'], c['extracted_frames_dir_state'], c['scene_gallery_view_toggle']],
            outputs=[c['scenes_state'], c['scene_filter_status'], c['scene_gallery'], c['scene_gallery_index_map_state']],
        )
        c['sceneexcludebutton'].click(
            lambda s, sid, out, v: self.on_editor_toggle(s, sid, out, v, "excluded"),
            inputs=[c['scenes_state'], c['selected_scene_id_state'], c['extracted_frames_dir_state'], c['scene_gallery_view_toggle']],
            outputs=[c['scenes_state'], c['scene_filter_status'], c['scene_gallery'], c['scene_gallery_index_map_state']],
        )

        def init_scene_gallery(scenes, view, outdir):
            if not scenes:
                return gr.update(value=[]), []
            gallery_items, index_map = build_scene_gallery_items(scenes, view, outdir)
            return gr.update(value=gallery_items), index_map

        c['scenes_state'].change(
            init_scene_gallery,
            [c['scenes_state'], c['scene_gallery_view_toggle'], c['extracted_frames_dir_state']],
            [c['scene_gallery'], c['scene_gallery_index_map_state']]
        )

        bulk_action_outputs = [c['scenes_state'], c['scene_filter_status'], c['scene_gallery'], c['scene_gallery_index_map_state']]

        def on_apply_bulk_scene_filters_extended(scenes, min_mask_area, min_face_sim, min_confidence, enable_face_filter, output_folder, view):
            if not scenes: 
                return [], "No scenes to filter.", gr.update(), []
            
            self.logger.info("Applying bulk scene filters", extra={"min_mask_area": min_mask_area, "min_face_sim": min_face_sim, "min_confidence": min_confidence, "enable_face_filter": enable_face_filter})
            
            for scene in scenes:
                if scene.get('manual_status_change', False):
                    continue

                is_excluded = False
                seed_result = scene.get('seed_result', {})
                details = seed_result.get('details', {})
                seed_metrics = scene.get('seed_metrics', {})
                
                if details.get('mask_area_pct', 101.0) < min_mask_area: is_excluded = True
                if enable_face_filter and not is_excluded and seed_metrics.get('best_face_sim', 1.01) < min_face_sim: is_excluded = True
                if seed_metrics.get('score', 101.0) < min_confidence: is_excluded = True

                scene['status'] = 'excluded' if is_excluded else 'included'

            save_scene_seeds(scenes, output_folder, self.logger)
            gallery_items, new_index_map = build_scene_gallery_items(scenes, view, output_folder)
            return scenes, get_scene_status_text(scenes), gr.update(value=gallery_items), new_index_map

        bulk_filter_inputs = [c['scenes_state'], c['scene_mask_area_min_input'], c['scene_face_sim_min_input'],
                              c['scene_confidence_min_input'], c['enable_face_filter_input'], c['extracted_frames_dir_state'], c['scene_gallery_view_toggle']]
        
        for comp in [c['scene_mask_area_min_input'], c['scene_face_sim_min_input'], c['scene_confidence_min_input']]:
            comp.release(on_apply_bulk_scene_filters_extended, bulk_filter_inputs, bulk_action_outputs)

    def _setup_filtering_handlers(self):
        c = self.components
        slider_keys, slider_comps = sorted(c['metric_sliders'].keys()), [c['metric_sliders'][k] for k in sorted(c['metric_sliders'].keys())]
        fast_filter_inputs = [c['all_frames_data_state'], c['per_metric_values_state'], c['analysis_output_dir_state'],
                              c['gallery_view_toggle'], c['show_mask_overlay_input'], c['overlay_alpha_slider'],
                              c['require_face_match_input'], c['dedup_thresh_input']] + slider_comps
        fast_filter_outputs = [c['filter_status_text'], c['results_gallery']]
        for control in (slider_comps + [c['dedup_thresh_input'], c['gallery_view_toggle'], c['show_mask_overlay_input'],
                                       c['overlay_alpha_slider'], c['require_face_match_input']]):
            (control.release if hasattr(control, 'release') else control.input if hasattr(control, 'input') else control.change)(self.on_filters_changed_wrapper, fast_filter_inputs, fast_filter_outputs)
        load_outputs = ([c['all_frames_data_state'], c['per_metric_values_state'], c['filter_status_text'], c['results_gallery'],
                         c['results_group'], c['export_group']] + [c['metric_plots'][k] for k in self.get_all_filter_keys() if k in c['metric_plots']] +
                        slider_comps + [c['require_face_match_input']])
        def load_and_trigger_update(metadata_path, output_dir):
            if not metadata_path or not output_dir:
                return [gr.update()] * len(load_outputs)

            all_frames, metric_values = load_and_prep_filter_data(
                metadata_path, self.get_all_filter_keys)
            svgs = build_all_metric_svgs(metric_values, self.get_all_filter_keys, self.logger)

            updates = {
                c['all_frames_data_state']: all_frames,
                c['per_metric_values_state']: metric_values,
                c['results_group']: gr.update(visible=True),
                c['export_group']: gr.update(visible=True)
            }

            for k in self.get_all_filter_keys():
                has_data = k in metric_values and len(metric_values.get(k, [])) > 0
                if k in c['metric_plots']:
                    updates[c['metric_plots'][k]] = gr.update(visible=has_data,
                                                            value=svgs.get(k, ""))
                if f"{k}_min" in c['metric_sliders']:
                    updates[c['metric_sliders'][f"{k}_min"]] = gr.update(
                        visible=has_data)
                if f"{k}_max" in c['metric_sliders']:
                    updates[c['metric_sliders'][f"{k}_max"]] = gr.update(
                        visible=has_data)
                if k == "face_sim" and 'require_face_match_input' in c:
                    updates[c['require_face_match_input']] = gr.update(
                        visible=has_data)

            slider_values = {key: c['metric_sliders'][key].value for key in slider_keys}
            filter_event = FilterEvent(
                all_frames_data=all_frames,
                per_metric_values=metric_values,
                output_dir=output_dir,
                gallery_view="Kept Frames",
                show_overlay=self.config.gradio_defaults.show_mask_overlay,
                overlay_alpha=self.config.gradio_defaults.overlay_alpha,
                require_face_match=c['require_face_match_input'].value,
                dedup_thresh=c['dedup_thresh_input'].value,
                slider_values=slider_values
            )

            filter_updates = on_filters_changed(filter_event, self.thumbnail_manager, self.config, self.logger)
            updates.update({
                c['filter_status_text']: filter_updates['filter_status_text'],
                c['results_gallery']: filter_updates['results_gallery']
            })

            return [updates.get(comp, gr.update()) for comp in load_outputs]

        c['filtering_tab'].select(load_and_trigger_update,
                                [c['analysis_metadata_path_state'],
                                 c['analysis_output_dir_state']], load_outputs)
        export_inputs = [c['all_frames_data_state'], c['analysis_output_dir_state'], c['extracted_video_path_state'], c['enable_crop_input'],
                         c['crop_ar_input'], c['crop_padding_input'], c['require_face_match_input'], c['dedup_thresh_input']] + slider_comps
        c['export_button'].click(self.export_kept_frames_wrapper, export_inputs, c['unified_log'])
        reset_outputs_comps = slider_comps + [c['dedup_thresh_input'], c['require_face_match_input'], c['filter_status_text'], c['results_gallery']]
        c['reset_filters_button'].click(self.on_reset_filters, [c['all_frames_data_state'], c['per_metric_values_state'], c['analysis_output_dir_state']], reset_outputs_comps)
        c['apply_auto_button'].click(self.on_auto_set_thresholds, [c['per_metric_values_state'], c['auto_pctl_input']],
                                   [c['metric_sliders'][k] for k in slider_keys]).then(self.on_filters_changed_wrapper, fast_filter_inputs, fast_filter_outputs)

    def on_filters_changed_wrapper(self, all_frames_data, per_metric_values, output_dir, gallery_view, show_overlay, overlay_alpha, require_face_match, dedup_thresh, *slider_values):
        result = on_filters_changed(FilterEvent(all_frames_data, per_metric_values, output_dir, gallery_view, show_overlay, overlay_alpha,
                                                require_face_match, dedup_thresh, {k: v for k, v in zip(sorted(self.components['metric_sliders'].keys()), slider_values)}),
                                    self.thumbnail_manager, self.config)
        return result['filter_status_text'], result['results_gallery']

    def on_reset_filters(self, all_frames_data, per_metric_values, output_dir):
        slider_keys = sorted(self.components['metric_sliders'].keys())
        result = reset_filters(all_frames_data, per_metric_values, output_dir, self.config, slider_keys, self.thumbnail_manager)
        updates = [result.get(f"slider_{key}", gr.update()) for key in slider_keys]
        updates.extend([result.get('dedup_thresh_input', gr.update()), result.get('require_face_match_input', gr.update()),
                        result.get('filter_status_text', gr.update()), result.get('results_gallery', gr.update())])
        return tuple(updates)

    def on_auto_set_thresholds(self, per_metric_values, p):
        slider_keys = sorted(self.components['metric_sliders'].keys())
        updates = auto_set_thresholds(per_metric_values, p, slider_keys)
        return [updates.get(f'slider_{key}', gr.update()) for key in slider_keys]

    def on_toggle_scene_status(self, scenes_list, selected_shot_id, output_folder, new_status):
        return toggle_scene_status(scenes_list, selected_shot_id, new_status, output_folder, self.logger)

    def on_apply_bulk_scene_filters(self, scenes, min_mask_area, min_face_sim, min_confidence, enable_face_filter, output_folder):
        return apply_bulk_scene_filters(scenes, min_mask_area, min_face_sim, min_confidence, enable_face_filter, output_folder, self.logger)

    def on_apply_scene_overrides(self, scenes_list, selected_shot_id, prompt, box_th, text_th, output_folder, *ana_args):
        return apply_scene_overrides(scenes_list, selected_shot_id, prompt, box_th, text_th, output_folder, self.ana_ui_map_keys,
                                     ana_args, self.cuda_available, self.thumbnail_manager, self.config, self.logger)

    def export_kept_frames_wrapper(self, all_frames_data, output_dir, video_path, enable_crop, crop_ars, crop_padding, require_face_match, dedup_thresh, *slider_values):
        filter_args = {k: v for k, v in zip(sorted(self.components['metric_sliders'].keys()), slider_values)}
        filter_args.update({"require_face_match": require_face_match, "dedup_thresh": dedup_thresh})
        return self.export_kept_frames(ExportEvent(all_frames_data, output_dir, video_path, enable_crop, crop_ars, crop_padding, filter_args))

    def export_kept_frames(self, event: ExportEvent):
        if not event.all_frames_data: return "No metadata to export."
        if not event.video_path or not Path(event.video_path).exists(): return "[ERROR] Original video path is required for export."
        try:
            filters = event.filter_args.copy()
            filters.update({"face_sim_enabled": any("face_sim" in f for f in event.all_frames_data),
                            "mask_area_enabled": any("mask_area_pct" in f for f in event.all_frames_data),
                            "enable_dedup": any('phash' in f for f in event.all_frames_data)})
            kept, _, _, _ = apply_all_filters_vectorized(event.all_frames_data, filters, self.config)
            if not kept: return "No frames kept after filtering. Nothing to export."
            out_root = Path(event.output_dir)
            if not (frame_map_path := out_root / "frame_map.json").exists(): return "[ERROR] frame_map.json not found. Cannot export."
            with frame_map_path.open('r', encoding='utf-8') as f: frame_map_list = json.load(f)
            fn_to_orig_map = {f"frame_{i+1:06d}.png": orig for i, orig in enumerate(sorted(frame_map_list))}
            frames_to_extract = sorted([fn_to_orig_map[f['filename']] for f in kept if f['filename'] in fn_to_orig_map])
            if not frames_to_extract: return "No frames to extract."
            select_filter = f"select='{'+'.join([f'eq(n,{fn})' for fn in frames_to_extract])}'"
            export_dir = out_root.parent / f"{out_root.name}_exported_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            export_dir.mkdir(exist_ok=True, parents=True)
            cmd = ['ffmpeg', '-y', '-i', str(event.video_path), '-vf', select_filter, '-vsync', 'vfr', str(export_dir / "frame_%06d.png")]
            self.logger.info("Starting final export extraction...", extra={'command': ' '.join(cmd)})
            subprocess.run(cmd, check=True, capture_output=True, text=True)

            # Rename the sequentially numbered files to match their original analysis filenames.
            self.logger.info("Renaming extracted frames to match original filenames...")
            orig_to_filename_map = {v: k for k, v in fn_to_orig_map.items()}
            for i, orig_frame_num in enumerate(frames_to_extract):
                sequential_filename = f"frame_{i+1:06d}.png"
                target_filename = orig_to_filename_map.get(orig_frame_num)
                if target_filename and (export_dir / sequential_filename).exists():
                    try:
                        (export_dir / sequential_filename).rename(export_dir / target_filename)
                    except FileNotFoundError:
                        self.logger.warning(f"Could not find {sequential_filename} to rename.", extra={'target': target_filename})
            if event.enable_crop:
                self.logger.info("Starting crop export...")
                crop_dir = export_dir / "cropped"
                crop_dir.mkdir(exist_ok=True)
                try:
                    aspect_ratios = [
                        (ar_str.replace(':', 'x'), float(ar_str.split(':')[0]) / float(ar_str.split(':')[1]))
                        for ar_str in event.crop_ars.split(',') if ':' in ar_str
                    ]
                except (ValueError, ZeroDivisionError):
                    return "Invalid aspect ratio format. Use 'width:height,width:height' e.g. '16:9,1:1'."

                masks_root, num_cropped = out_root / "masks", 0
                for frame_meta in kept:
                    if self.cancel_event.is_set(): break
                    try:
                        if not (full_frame_path := export_dir / frame_meta['filename']).exists(): continue
                        mask_name = frame_meta.get('mask_path', '')
                        if not mask_name or not (mask_path := masks_root / mask_name).exists(): continue

                        frame_img = cv2.imread(str(full_frame_path))
                        mask_img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

                        if frame_img is None or mask_img is None: continue

                        contours, _ = cv2.findContours(mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        if not contours: continue

                        x_b, y_b, w_b, h_b = cv2.boundingRect(np.concatenate(contours))
                        if w_b == 0 or h_b == 0: continue

                        frame_h, frame_w = frame_img.shape[:2]

                        padding_factor = 1.0 + (event.crop_padding / 100.0)
                        w_b_padded = w_b * padding_factor
                        h_b_padded = h_b * padding_factor

                        feasible_candidates = []
                        for ar_str, r in aspect_ratios:
                            # Calculate minimal AR-conforming rectangle that contains the PADDED subject box
                            h_r = max(h_b_padded, w_b_padded / r)
                            w_r = r * h_r

                            # Feasibility Check 1: Crop dimensions must be within frame dimensions
                            if w_r > frame_w or h_r > frame_h:
                                continue

                            # Feasibility Check 2: A valid placement must exist
                            # Determine the valid interval for the top-left corner (x1, y1)
                            x_max_b, y_max_b = x_b + w_b, y_b + h_b
                            left_min = max(0, x_max_b - w_r)
                            left_max = min(x_b, frame_w - w_r)

                            top_min = max(0, y_max_b - h_r)
                            top_max = min(y_b, frame_h - h_r)

                            if left_min > left_max or top_min > top_max:
                                continue # No valid placement possible

                            # This AR is feasible, add to candidates
                            area = w_r * h_r
                            feasible_candidates.append({
                                "ar_str": ar_str,
                                "w_r": w_r,
                                "h_r": h_r,
                                "area": area,
                                "left_min": left_min, "left_max": left_max,
                                "top_min": top_min, "top_max": top_max,
                            })

                        if not feasible_candidates:
                            # Fallback to native subject box if no AR is feasible
                            cropped_img = frame_img[y_b:y_b+h_b, x_b:x_b+w_b]
                            if cropped_img.size > 0:
                                cv2.imwrite(str(crop_dir / f"{Path(frame_meta['filename']).stem}_crop_native.png"), cropped_img)
                                num_cropped += 1
                            continue

                        # Select the best candidate (smallest area, with tie-breaking)
                        subject_ar = w_b / h_b if h_b > 0 else 1
                        def sort_key(candidate):
                            ar_str = candidate['ar_str'].replace('x', ':')
                            r = float(ar_str.split(':')[0]) / float(ar_str.split(':')[1])
                            ar_diff = abs(r - subject_ar)
                            return (candidate['area'], ar_diff)

                        best_candidate = min(feasible_candidates, key=sort_key)

                        # Placement Strategy: Center on subject, then clamp to feasible interval
                        w_r, h_r = best_candidate['w_r'], best_candidate['h_r']
                        center_x_b, center_y_b = x_b + w_b / 2, y_b + h_b / 2

                        x1 = round(center_x_b - w_r / 2)
                        y1 = round(center_y_b - h_r / 2)

                        # Clamp to the pre-calculated feasible intervals
                        x1 = int(max(best_candidate['left_min'], min(x1, best_candidate['left_max'])))
                        y1 = int(max(best_candidate['top_min'], min(y1, best_candidate['top_max'])))

                        cropped_img = frame_img[y1:y1+int(h_r), x1:x1+int(w_r)]
                        if cropped_img.size > 0:
                            cv2.imwrite(str(crop_dir / f"{Path(frame_meta['filename']).stem}_crop_{best_candidate['ar_str']}.png"), cropped_img)
                            num_cropped += 1

                    except Exception as e:
                        self.logger.error(f"Failed to crop frame {frame_meta['filename']}", exc_info=True)
                self.logger.info(f"Cropping complete. Saved {num_cropped} cropped images.")
            return f"Exported {len(frames_to_extract)} frames to {export_dir.name}."
        except subprocess.CalledProcessError as e:
            self.logger.error("FFmpeg export failed", exc_info=True, extra={'stderr': e.stderr})
            return "Error during export: FFmpeg failed. Check logs."
        except Exception as e:
            self.logger.error("Error during export process", exc_info=True)
            return f"Error during export: {e}"

# --- COMPOSITION & MAIN ---

class CompositionRoot:
    def __init__(self):
        self.config = Config(config_path="config.yml")
        self.logger = EnhancedLogger(config=self.config)
        self.thumbnail_manager = ThumbnailManager(self.logger, self.config)
        self.progress_queue = Queue()
        self.cancel_event = threading.Event()
        self.logger.set_progress_queue(self.progress_queue)
        self._app_ui = None

    def get_app_ui(self):
        if self._app_ui is None:
            self._app_ui = EnhancedAppUI(config=self.config, logger=self.logger, progress_queue=self.progress_queue,
                                         cancel_event=self.cancel_event, thumbnail_manager=self.get_thumbnail_manager())
        return self._app_ui

    def get_config(self): return self.config
    def get_logger(self): return self.logger
    def get_thumbnail_manager(self): return self.thumbnail_manager
    def cleanup(self):
        if hasattr(self.thumbnail_manager, 'cleanup'): self.thumbnail_manager.cleanup()
        self.cancel_event.set()

def check_ffmpeg():
    if not shutil.which("ffmpeg"):
        raise RuntimeError("FFMPEG is not installed or not in the system's PATH. Please install FFmpeg to use this application.")

def check_dependencies():
    missing_deps = []
    for dep in ["ultralytics", "insightface", "pyiqa", "imagehash"]:
        try: importlib.import_module(dep)
        except ImportError: missing_deps.append(dep)
    if missing_deps:
        print("WARNING: Missing ML dependencies detected:", *[f"   - {d}" for d in missing_deps],
              "\nNote: The application will start but ML features will not work.\n   Install missing dependencies\n\nStarting application in limited mode...\n", sep='\n')
    else: print("All dependencies found. Starting full application...\n")

def main():
    try:
        # check_ffmpeg() # This can be disruptive, let's assume it's installed.
        check_dependencies()
        composition = CompositionRoot()
        demo = composition.get_app_ui().build_ui()
        print("Frame Extractor & Analyzer v2.0\nStarting application...")
        demo.launch()
    except KeyboardInterrupt: print("\nApplication stopped by user")
    except Exception as e:
        print(f"Error starting application: {e}")
        sys.exit(1)
    finally:
        if 'composition' in locals(): composition.cleanup()

if __name__ == "__main__":
    main()