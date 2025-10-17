#!/usr/bin/env python3
"""
Frame Extractor & Analyzer v2.0
Monolithic application file.
"""

# --- IMPORTS ---
import argparse
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
import inspect
import io
import json
import logging
import math
import numpy as np
import os
import psutil
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import torch
import traceback
import types
import urllib.request
import yaml

from pathlib import Path

# Add submodules to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'Grounded-SAM-2'))
sys.path.insert(0, str(project_root / 'DAM4SAM'))

from collections import Counter, OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict, field, fields
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

class Config:
    BASE_DIR = Path(__file__).parent
    DIRS = {
        'logs': BASE_DIR / "logs",
        'models': BASE_DIR / "models",
        'downloads': BASE_DIR / "downloads"
    }
    DEFAULT_CONFIG = """
# External configuration for UI defaults, model paths, and quality metric parameters.
# This file allows for easy tuning without modifying the application code.

# --- UI & Filter Defaults ---
# Default values for the Gradio user interface components.
ui_defaults:
  # --- Extraction Settings (New Workflow) ---
  thumbnails_only: True              # Recommended: Extract only thumbnails first. Full-res frames are extracted on-demand during export.
  thumb_megapixels: 0.5              # Target resolution for thumbnails (e.g., 0.5 = ~960x540 for a 16:9 video).
  scene_detect: True                 # Enable scene detection during extraction. Crucial for the new scene-based workflow.
  max_resolution: "maximum available" # Max resolution for video downloads (e.g., from YouTube).

  # --- Pre-Analysis & Seeding Settings ---
  pre_analysis_enabled: True         # Automatically analyze scene thumbnails to find the best frame for seeding.
  pre_sample_nth: 5                  # When pre-analyzing, check every Nth frame in a scene. 1 = all frames. Higher values are faster for long scenes.
  enable_face_filter: True           # Enable face similarity scoring (requires reference image).
  face_model_name: "buffalo_l"       # InsightFace model for face analysis.
  enable_subject_mask: True          # Enable subject masking and propagation.
  dam4sam_model_name: "sam21pp-L"    # DAM4SAM model for mask propagation.
  person_detector_model: "yolo11x.pt" # YOLO model for detecting persons.
  primary_seed_strategy: "🤖 Automatic" # Default strategy on UI load
  seed_strategy: "Largest Person"      # Default for 'Automatic' strategy
  text_prompt: ""                      # Global text prompt for seeding (can be overridden per-scene).

  # --- Final Analysis & Filtering ---
  resume: False                      # Attempt to resume a previous analysis run (less relevant in new workflow).
  require_face_match: False          # In final filtering, reject frames that have no face detected.
  enable_dedup: True                 # Enable perceptual hash-based near-duplicate removal during analysis.
  dedup_thresh: 5                    # pHash distance threshold for deduplication (lower is stricter).

  # --- Legacy Full-Frame Extraction Settings (used if 'thumbnails_only' is false) ---
  method: "all"
  interval: 5.0
  fast_scene: False
  use_png: True
  nth_frame: 5
  disable_parallel: False


# Central registry for all filter sliders and checkboxes.
# Used for component creation and reset behavior in the Filtering & Export tab.
filter_defaults:
  quality_score: { min: 0.0, max: 100.0, step: 0.5, default_min: 0.0, default_max: 100.0 }
  sharpness: { min: 0.0, max: 100.0, step: 0.5, default_min: 0.0, default_max: 100.0 }
  edge_strength: { min: 0.0, max: 100.0, step: 0.5, default_min: 0.0, default_max: 100.0 }
  contrast: { min: 0.0, max: 100.0, step: 0.5, default_min: 0.0, default_max: 100.0 }
  brightness: { min: 0.0, max: 100.0, step: 0.5, default_min: 0.0, default_max: 100.0 }
  entropy: { min: 0.0, max: 100.0, step: 0.5, default_min: 0.0, default_max: 100.0 }
  niqe: { min: 0.0, max: 100.0, step: 0.5, default_min: 0.0, default_max: 100.0 }
  face_sim: { min: 0.0, max: 1.0, step: 0.01, default_min: 0.5 }
  mask_area_pct: { min: 0.0, max: 100.0, step: 0.1, default_min: 1.0 }
  dedup_thresh: { min: -1, max: 32, step: 1, default: -1 } # -1 disables dedup in the filter tab UI by default

# --- Quality Metrics Configuration ---
# Weights for combining individual quality scores into a single quality score.
quality_weights:
  sharpness: 25
  edge_strength: 15
  contrast: 15
  brightness: 10
  entropy: 15
  niqe: 20

# Base scaling constant for sharpness, adjusted by image resolution.
# A higher value makes the score less sensitive.
sharpness_base_scale: 2500
edge_strength_base_scale: 100

# Minimum percentage of the frame area that a subject mask must occupy to be considered valid.
min_mask_area_pct: 1.0

# --- Model & Path Configuration ---
# Paths for models and other external dependencies.
model_paths:
  grounding_dino_config: "Grounded-SAM-2/grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
  grounding_dino_checkpoint: "models/groundingdino_swint_ogc.pth"

# --- Grounded DINO/SAM2 Inference Parameters ---
grounding_dino_params:
  box_threshold: 0.35
  text_threshold: 0.25

monitoring:
  # Performance monitoring thresholds
  memory_warning_threshold_mb: 8192  # Warn when process uses > 8GB
  memory_critical_threshold_mb: 16384  # Critical when > 16GB
  cpu_warning_threshold_percent: 90
  gpu_memory_warning_threshold_percent: 90
"""

    def __init__(self):
        self.settings = yaml.safe_load(self.DEFAULT_CONFIG)
        self._validate_config()

        for dir_name, dir_path in self.DIRS.items():
            try:
                dir_path.mkdir(exist_ok=True, parents=True)
            except PermissionError:
                raise RuntimeError(f"Cannot create {dir_name} directory at {dir_path}. Check permissions.")

        for key, value in self.settings.items():
            setattr(self, key, value)

        self.thumbnail_cache_size = self.settings.get('thumbnail_cache_size', 200)

        self.GROUNDING_DINO_CONFIG = self.BASE_DIR / self.model_paths['grounding_dino_config']
        ckpt_cfg = Path(self.model_paths['grounding_dino_checkpoint'])
        self.GROUNDING_DINO_CKPT = ckpt_cfg if ckpt_cfg.is_absolute() else (self.DIRS['models'] / ckpt_cfg.name)
        self.GROUNDING_BOX_THRESHOLD = self.grounding_dino_params['box_threshold']
        self.GROUNDING_TEXT_THRESHOLD = self.grounding_dino_params['text_threshold']
        self.QUALITY_METRICS = list(self.quality_weights.keys())

    def _validate_config(self):
        """Basic validation to ensure essential keys exist."""
        required_keys = ['ui_defaults', 'filter_defaults', 'quality_weights', 'model_paths']
        for key in required_keys:
            if key not in self.settings:
                raise ValueError(f"Missing required configuration section: '{key}'")

        if not all(k in self.settings['quality_weights'] for k in ['sharpness', 'contrast']):
             raise ValueError("quality_weights config is missing essential metrics.")

        # Validate that the sum of quality_weights is not zero
        if sum(self.settings['quality_weights'].values()) == 0:
            raise ValueError("The sum of quality_weights cannot be zero.")


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
    def __init__(self, log_dir: Optional[Path] = None, enable_performance_monitoring: bool = True,
                 log_to_file: bool = True, log_to_console: bool = True):
        self.log_dir = log_dir or Path("logs")
        self.log_dir.mkdir(exist_ok=True, parents=True)
        self.progress_queue = None
        self.performance_monitor = PerformanceMonitor() if enable_performance_monitoring else None
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_log_file = self.log_dir / f"session_{self.session_id}.log"
        self.structured_log_file = self.log_dir / f"structured_{self.session_id}.jsonl"
        self.logger = logging.getLogger(f'enhanced_logger_{self.session_id}')
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False
        self.logger.handlers.clear()
        if log_to_console: self._setup_console_handler()
        if log_to_file: self._setup_file_handlers()
        self._operation_stack: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    def _setup_console_handler(self):
        console_handler = logging.StreamHandler()
        console_formatter = ColoredFormatter('%(asctime)s | %(levelname)8s | %(name)s | %(message)s')
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(logging.INFO)
        self.logger.addHandler(console_handler)

    def _setup_file_handlers(self):
        file_handler = logging.FileHandler(self.session_log_file, encoding='utf-8')
        file_formatter = logging.Formatter('%(asctime)s | %(levelname)8s | %(name)s | %(message)s')
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
    def operation_context(self, operation_name: str, component: str, user_context: Optional[Dict[str, Any]] = None):
        start_time = time.time()
        start_metrics = self.performance_monitor.get_system_metrics() if self.performance_monitor else {}
        operation_data = {'operation': operation_name, 'component': component, 'start_time': start_time,
                          'start_metrics': start_metrics, 'user_context': user_context or {}}
        with self._lock:
            self._operation_stack.append(operation_data)
        try:
            self.info(f"Starting {operation_name}", component=component, operation=operation_name, user_context=user_context)
            yield operation_data
            duration = (time.time() - start_time) * 1000
            self.success(f"Completed {operation_name}", component=component, operation=operation_name, duration_ms=duration, user_context=user_context)
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            self.error(f"Failed {operation_name}: {str(e)}", component=component, operation=operation_name, duration_ms=duration,
                       error_type=type(e).__name__, stack_trace=traceback.format_exc(), user_context=user_context)
            raise
        finally:
            with self._lock:
                if self._operation_stack:
                    self._operation_stack.pop()

    def _create_log_event(self, level: str, message: str, component: str, **kwargs) -> LogEvent:
        current_metrics = self.performance_monitor.get_system_metrics() if self.performance_monitor else {}
        exc_info = kwargs.pop('exc_info', None)
        extra = kwargs.pop('extra', None)
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

# --- PROGRESS TRACKING ---

@dataclass
class ProgressState:
    operation: str
    stage: str
    current: int
    total: int
    stage_current: int = 0
    stage_total: int = 0
    start_time: float = 0.0
    stage_start_time: float = 0.0
    substages: List[str] = None
    current_substage: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class AdvancedProgressTracker:
    def __init__(self, progress_queue: Queue, logger=None):
        self.progress_queue = progress_queue
        self.logger = logger
        self.current_state: Optional[ProgressState] = None
        self.history: List[Dict[str, Any]] = []
        self.stage_history: Dict[str, List[float]] = {}
        self.lock = threading.Lock()

    def start_operation(self, operation: str, total_items: int, stages: List[str] = None, metadata: Dict[str, Any] = None):
        with self.lock:
            self.current_state = ProgressState(operation=operation, stage="initializing", current=0, total=total_items,
                                               start_time=time.time(), substages=stages or [], metadata=metadata or {})
        self._update_ui()
        if self.logger:
            self.logger.info(f"Started operation: {operation}", component="progress", operation=operation, custom_fields={'total_items': total_items})

    def start_stage(self, stage_name: str, stage_items: int = None, substage: str = None):
        if not self.current_state: return
        with self.lock:
            if self.current_state.stage != "initializing":
                stage_duration = time.time() - self.current_state.stage_start_time
                if self.current_state.stage not in self.stage_history: self.stage_history[self.current_state.stage] = []
                self.stage_history[self.current_state.stage].append(stage_duration)
            self.current_state.stage = stage_name
            self.current_state.stage_current = 0
            self.current_state.stage_total = stage_items or 0
            self.current_state.stage_start_time = time.time()
            self.current_state.current_substage = substage
        self._update_ui()
        if self.logger:
            self.logger.info(f"Started stage: {stage_name}", component="progress", operation=self.current_state.operation, custom_fields={'stage_items': stage_items})

    def update_progress(self, items_processed: int = 1, stage_items_processed: int = None, substage: str = None, metadata: Dict[str, Any] = None):
        if not self.current_state: return
        with self.lock:
            self.current_state.current = min(self.current_state.current + items_processed, self.current_state.total)
            if stage_items_processed is not None: self.current_state.stage_current = min(stage_items_processed, self.current_state.stage_total)
            if substage: self.current_state.current_substage = substage
            if metadata: self.current_state.metadata.update(metadata)
        self._update_ui()

    def complete_operation(self, success: bool = True, message: str = None):
        if not self.current_state: return
        total_duration = time.time() - self.current_state.start_time
        with self.lock:
            if self.current_state.stage != "initializing":
                stage_duration = time.time() - self.current_state.stage_start_time
                if self.current_state.stage not in self.stage_history: self.stage_history[self.current_state.stage] = []
                self.stage_history[self.current_state.stage].append(stage_duration)
            self.history.append({'operation': self.current_state.operation, 'total_items': self.current_state.total, 'duration': total_duration,
                                 'success': success, 'timestamp': time.time(), 'stages': self.current_state.substages, 'final_metadata': self.current_state.metadata})
            if success:
                self.current_state.current = self.current_state.total
                self.current_state.stage = "completed"
            else:
                self.current_state.stage = "failed"
        self._update_ui(force_complete=True)
        if self.logger:
            status = "SUCCESS" if success else "ERROR"
            self.logger._log_event(self.logger._create_log_event(
                status, message or f"Operation {self.current_state.operation} {'completed' if success else 'failed'}", "progress",
                operation=self.current_state.operation, duration_ms=total_duration * 1000,
                custom_fields={'total_items_processed': self.current_state.current, 'success': success}))

    def _calculate_eta(self) -> tuple[float, str]:
        if not self.current_state or self.current_state.current == 0:
            return float('inf'), "calculating..."

        elapsed = time.time() - self.current_state.start_time
        items_per_second = self.current_state.current / elapsed
        remaining_items = self.current_state.total - self.current_state.current

        eta_seconds = remaining_items / items_per_second if items_per_second > 0 else float('inf')

        if eta_seconds == float('inf'):
            return eta_seconds, "calculating..."
        elif eta_seconds < 60:
            return eta_seconds, f"{int(eta_seconds)}s"
        elif eta_seconds < 3600:
            return eta_seconds, f"{int(eta_seconds / 60)}m {int(eta_seconds % 60)}s"
        else:
            return eta_seconds, f"{int(eta_seconds / 3600)}h {int((eta_seconds % 3600) / 60)}m"

    def _calculate_rate(self) -> float:
        if not self.current_state or self.current_state.current == 0: return 0.0
        elapsed = time.time() - self.current_state.start_time
        return self.current_state.current / elapsed if elapsed > 0 else 0.0

    def _update_ui(self, force_complete: bool = False):
        if not self.current_state: return
        progress_ratio = self.current_state.current / self.current_state.total if self.current_state.total > 0 else 0
        eta_seconds, eta_str = self._calculate_eta()
        rate = self._calculate_rate()
        stage_progress = ""
        if self.current_state.stage_total > 0:
            stage_ratio = self.current_state.stage_current / self.current_state.stage_total
            stage_progress = f" | Stage: {self.current_state.stage_current}/{self.current_state.stage_total} ({stage_ratio:.1%})"
        substage_info = f" | {self.current_state.current_substage}" if self.current_state.current_substage else ""
        progress_message = {"stage": self.current_state.stage, "progress": progress_ratio, "current": self.current_state.current,
                            "total": self.current_state.total, "eta": eta_str, "rate": f"{rate:.1f}/s" if rate > 0 else "0.0/s",
                            "detailed_status": f"{self.current_state.operation} | {self.current_state.stage} | {self.current_state.current}/{self.current_state.total} ({progress_ratio:.1%}) | ETA: {eta_str} | Rate: {rate:.1f}/s{stage_progress}{substage_info}"}
        if self.current_state.metadata: progress_message["metadata"] = self.current_state.metadata
        self.progress_queue.put(progress_message)

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

    def with_retry(self, max_attempts: int = 3, backoff_seconds: List[float] = [1, 5, 15], recoverable_exceptions: tuple = (Exception,)):
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

def sanitize_filename(name, max_length=50):
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

# --- QUALITY ---

@njit
def compute_entropy(hist):
    prob = hist / (np.sum(hist) + 1e-7)
    entropy = -np.sum(prob[prob > 0] * np.log2(prob[prob > 0]))
    return min(max(entropy / 8.0, 0), 1.0)

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

@dataclass
class Frame:
    image_data: np.ndarray
    frame_number: int
    metrics: FrameMetrics = field(default_factory=FrameMetrics)
    face_similarity_score: float | None = None
    max_face_confidence: float | None = None
    error: str | None = None

    def calculate_quality_metrics(self, thumb_image_rgb: np.ndarray, quality_config: QualityConfig, logger: 'EnhancedLogger',
                                  mask: np.ndarray | None = None, niqe_metric=None):
        try:
            gray = cv2.cvtColor(thumb_image_rgb, cv2.COLOR_RGB2GRAY)
            active_mask = ((mask > 128) if mask is not None and mask.ndim == 2 else None)
            if active_mask is not None and np.sum(active_mask) < 100:
                active_mask = None # fallback to full-frame stats instead of raising
            lap = cv2.Laplacian(gray, cv2.CV_64F)
            masked_lap = lap[active_mask] if active_mask is not None else lap
            sharpness = np.var(masked_lap) if masked_lap.size > 0 else 0
            sharpness_scaled = (sharpness / (quality_config.sharpness_base_scale * (gray.size / 500_000)))
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            edge_strength = np.mean(np.sqrt(sobelx**2 + sobely**2))
            edge_strength_scaled = (edge_strength / (quality_config.edge_strength_base_scale * (gray.size / 500_000)))
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
            entropy = compute_entropy(hist)
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
                        niqe_score = max(0, min(100, (10 - niqe_raw) * 10))
                except Exception as e:
                    logger.warning("NIQE calculation failed", extra={'frame': self.frame_number, 'error': e})
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            scores_norm = {"sharpness": min(sharpness_scaled, 1.0), "edge_strength": min(edge_strength_scaled, 1.0),
                           "contrast": min(contrast, 2.0) / 2.0, "brightness": brightness, "entropy": entropy, "niqe": niqe_score / 100.0}
            self.metrics = FrameMetrics(**{f"{k}_score": float(v * 100) for k, v in scores_norm.items()})
            # The quality_weights are part of the main config, not QualityConfig, so we'll need to pass them separately or access them differently.
            # For now, let's assume the main config is accessible.
            main_config = Config()
            quality_sum = sum(scores_norm[k] * (main_config.quality_weights[k] / 100.0) for k in main_config.QUALITY_METRICS)
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
    seed_config: dict = field(default_factory=dict)
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
        config = Config()
        if not self.gdino_config_path: self.gdino_config_path = str(config.GROUNDING_DINO_CONFIG)
        if not self.gdino_checkpoint_path: self.gdino_checkpoint_path = str(config.GROUNDING_DINO_CKPT)
        if self.box_threshold == 0.35: self.box_threshold = config.GROUNDING_BOX_THRESHOLD
        if self.text_threshold == 0.25: self.text_threshold = config.GROUNDING_TEXT_THRESHOLD
        if self.min_mask_area_pct == 1.0: self.min_mask_area_pct = config.min_mask_area_pct
        if self.sharpness_base_scale == 2500.0: self.sharpness_base_scale = config.sharpness_base_scale
        if self.edge_strength_base_scale == 100.0: self.edge_strength_base_scale = config.edge_strength_base_scale

    @classmethod
    def from_ui(cls, logger: 'EnhancedLogger', config: 'Config', **kwargs):
        if 'thumb_megapixels' in kwargs:
            thumb_mp = kwargs['thumb_megapixels']
            if not isinstance(thumb_mp, (int, float)) or thumb_mp <= 0:
                logger.warning(f"Invalid thumb_megapixels: {thumb_mp}, using default")
                kwargs['thumb_megapixels'] = 0.5

        if 'pre_sample_nth' in kwargs:
            sample_nth = kwargs['pre_sample_nth']
            if not isinstance(sample_nth, int) or sample_nth < 1:
                logger.warning(f"Invalid pre_sample_nth: {sample_nth}, using 1")
                kwargs['pre_sample_nth'] = 1

        valid_keys = {f.name for f in fields(cls)}
        filtered_defaults = {k: v for k, v in config.ui_defaults.items() if k in valid_keys}
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
    def __init__(self, params, progress_queue: Queue, cancel_event: threading.Event, logger: EnhancedLogger = None, tracker=None):
        self.params = params
        self.progress_queue = progress_queue
        self.cancel_event = cancel_event
        self.logger = logger or EnhancedLogger()
        self.tracker = tracker

# --- CACHING & OPTIMIZATION ---

class ThumbnailManager:
    def __init__(self, logger, max_size=200):
        self.logger = logger
        self.cache = OrderedDict()
        self.max_size = max_size
        self.logger.info(f"ThumbnailManager initialized with cache size {max_size}")

    def get(self, thumb_path: Path):
        if not isinstance(thumb_path, Path): thumb_path = Path(thumb_path)
        if thumb_path in self.cache:
            self.cache.move_to_end(thumb_path)
            return self.cache[thumb_path]
        if not thumb_path.exists(): return None

        if len(self.cache) > self.max_size * 0.8:
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
        num_to_remove = int(self.max_size * 0.2)  # Remove 20% of the cache
        for _ in range(num_to_remove):
            if not self.cache:
                break
            self.cache.popitem(last=False)

class AdaptiveResourceManager:
    def __init__(self, logger, config):
        self.logger = logger
        self.config = config
        self.monitoring_active = False
        self.current_limits = {'batch_size': getattr(config, 'default_batch_size', 32),
                               'num_workers': getattr(config, 'default_workers', 4),
                               'memory_limit_mb': getattr(config, 'memory_limit_mb', 8192)}
        self.performance_history = []
        self._monitor_thread = None

    def start_monitoring(self):
        self.monitoring_active = True
        self._monitor_thread = threading.Thread(target=self._monitor_resources, daemon=True)
        self._monitor_thread.start()
        self.logger.info("Started adaptive resource monitoring", component="resource_manager")

    def stop_monitoring(self):
        self.monitoring_active = False
        if self._monitor_thread: self._monitor_thread.join(timeout=1)
        self.logger.info("Stopped adaptive resource monitoring", component="resource_manager")

    def _monitor_resources(self):
        while self.monitoring_active:
            try:
                metrics = self._get_resource_metrics()
                self._adjust_parameters(metrics)
                self.performance_history.append({'timestamp': time.time(), 'metrics': metrics, 'limits': self.current_limits.copy()})
                cutoff_time = time.time() - 3600
                self.performance_history = [h for h in self.performance_history if h['timestamp'] > cutoff_time]
                time.sleep(5)
            except Exception as e:
                self.logger.error(f"Resource monitoring error: {e}", component="resource_manager")
                time.sleep(10)

    def _get_resource_metrics(self) -> Dict[str, Any]:
        process = psutil.Process()
        metrics = {'cpu_percent': psutil.cpu_percent(interval=1), 'memory_percent': psutil.virtual_memory().percent,
                   'memory_available_mb': psutil.virtual_memory().available / (1024**2),
                   'process_memory_mb': process.memory_info().rss / (1024**2),
                   'disk_io': psutil.disk_io_counters()._asdict() if psutil.disk_io_counters() else {},
                   'load_average': psutil.getloadavg() if hasattr(psutil, 'getloadavg') else [0, 0, 0]}
        if GPUtil:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    metrics.update({'gpu_memory_percent': gpu.memoryUtil * 100, 'gpu_load_percent': gpu.load * 100, 'gpu_temperature': gpu.temperature})
            except Exception: pass
        return metrics

    def _adjust_parameters(self, metrics: Dict[str, Any]):
        monitoring_config = self.config.settings.get('monitoring', {})
        mem_used = metrics['process_memory_mb']
        mem_crit = monitoring_config.get('memory_critical_threshold_mb', 16384)
        mem_warn = monitoring_config.get('memory_warning_threshold_mb', 8192)

        if mem_used > mem_crit:
            if self.current_limits['batch_size'] > 1:
                old_batch = self.current_limits['batch_size']
                self.current_limits['batch_size'] = 1 # Aggressive reduction
                self.logger.critical(f"Critical memory usage ({mem_used}MB), forcing batch size to 1 from {old_batch}",
                                     component="resource_manager", custom_fields={'memory_mb': mem_used})
                gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()
        elif mem_used > mem_warn:
            if self.current_limits['batch_size'] > 1:
                old_batch_size = self.current_limits['batch_size']
                self.current_limits['batch_size'] = max(1, int(old_batch_size * 0.7))
                self.logger.warning(f"High memory usage detected, reducing batch size from {old_batch_size} to {self.current_limits['batch_size']}",
                                    component="resource_manager", custom_fields={'memory_mb': metrics['process_memory_mb']})
                gc.collect()

        if metrics['cpu_percent'] > monitoring_config.get('cpu_warning_threshold_percent', 90):
            if self.current_limits['num_workers'] > 1:
                old_workers = self.current_limits['num_workers']
                self.current_limits['num_workers'] = max(1, int(old_workers * 0.8))
                self.logger.warning(f"High CPU usage detected, reducing workers from {old_workers} to {self.current_limits['num_workers']}",
                                    component="resource_manager", custom_fields={'cpu_percent': metrics['cpu_percent']})
        if 'gpu_memory_percent' in metrics and metrics['gpu_memory_percent'] > monitoring_config.get('gpu_memory_warning_threshold_percent', 90):
            self.logger.warning("High GPU memory usage detected, consider reducing batch size or switching to CPU",
                                component="resource_manager", custom_fields={'gpu_memory_percent': metrics['gpu_memory_percent']})

    def get_current_limits(self) -> Dict[str, Any]: return self.current_limits.copy()
    def get_performance_summary(self) -> Dict[str, Any]:
        if not self.performance_history: return {}
        recent_metrics = self.performance_history[-10:]
        avg_cpu = sum(m['metrics']['cpu_percent'] for m in recent_metrics) / len(recent_metrics)
        avg_memory = sum(m['metrics']['process_memory_mb'] for m in recent_metrics) / len(recent_metrics)
        return {'avg_cpu_percent': avg_cpu, 'avg_memory_mb': avg_memory, 'current_batch_size': self.current_limits['batch_size'],
                'current_workers': self.current_limits['num_workers'], 'monitoring_duration_minutes': (time.time() - self.performance_history[0]['timestamp']) / 60}

# --- MODEL LOADING & MANAGEMENT ---

def download_model(url, dest_path, description, logger, error_handler: ErrorHandler, min_size=1_000_000):
    if dest_path.is_file() and dest_path.stat().st_size >= min_size: return
    @error_handler.with_retry(recoverable_exceptions=(urllib.error.URLError, TimeoutError, RuntimeError))
    def download_func():
        logger.info(f"Downloading {description}", extra={'url': url, 'dest': dest_path})
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with (urllib.request.urlopen(req, timeout=60) as resp, open(dest_path, "wb") as out):
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
def get_face_analyzer(model_name, config: 'Config', logger: 'EnhancedLogger'):
    from insightface.app import FaceAnalysis
    logger.info(f"Loading or getting cached face model: {model_name}")
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        providers = (['CUDAExecutionProvider', 'CPUExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider'])
        analyzer = FaceAnalysis(name=model_name, root=str(config.DIRS['models']), providers=providers)
        analyzer.prepare(ctx_id=0 if device == 'cuda' else -1, det_size=(640, 640))
        logger.success(f"Face model loaded with {'CUDA' if device == 'cuda' else 'CPU'}.")
        return analyzer
    except Exception as e:
        if "out of memory" in str(e) and torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.warning("CUDA OOM, retrying with CPU...")
            try:
                # Retry with CPU
                analyzer = FaceAnalysis(name=model_name, root=str(config.DIRS['models']),
                                      providers=['CPUExecutionProvider'])
                analyzer.prepare(ctx_id=-1, det_size=(640, 640))
                return analyzer
            except Exception as cpu_e:
                logger.error(f"CPU fallback also failed: {cpu_e}")
        raise RuntimeError(f"Could not initialize face analysis model. Error: {e}") from e

class PersonDetector:
    def __init__(self, model="yolo11x.pt", imgsz=640, conf=0.3, device='cuda', config=None, logger=None):
        from ultralytics import YOLO
        self.config = config or Config()
        self.logger = logger or EnhancedLogger()
        error_handler = ErrorHandler(self.logger, self.config)
        model_path = self.config.DIRS['models'] / model
        model_path.parent.mkdir(exist_ok=True)
        model_url = f"https://huggingface.co/Ultralytics/YOLO11/resolve/main/{model}"
        download_model(model_url, model_path, "YOLO person detector", self.logger, error_handler)
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model = YOLO(str(model_path))
        self.model.to(self.device)
        self.imgsz = imgsz
        self.conf = conf
        self.logger.info("YOLO person detector loaded", component="person_detector", user_context={'device': self.device, 'model': model})

    def detect_boxes(self, img_rgb):
        res = self.model.predict(img_rgb, imgsz=self.imgsz, conf=self.conf, classes=[0], verbose=False, device=self.device)
        boxes = []
        for r in res:
            if getattr(r, "boxes", None) is None: continue
            for b in r.boxes.cpu():
                boxes.append((*map(int, b.xyxy[0].tolist()), float(b.conf[0])))
        return boxes

@lru_cache(maxsize=None)
def get_person_detector(model_name, device, config: 'Config', logger=None):
    logger = logger or EnhancedLogger()
    logger.info(f"Loading or getting cached person detector: {model_name}", component="person_detector")
    return PersonDetector(model=model_name, device=device, config=config, logger=logger)

@lru_cache(maxsize=None)
def get_grounding_dino_model(gdino_config_path: str, gdino_checkpoint_path: str, config: 'Config', device="cuda", logger=None):
    if not gdino_load_model: raise ImportError("GroundingDINO is not installed.")
    logger = logger or EnhancedLogger()
    error_handler = ErrorHandler(logger, config)
    try:
        ckpt_path = Path(gdino_checkpoint_path)
        if not ckpt_path.is_absolute(): ckpt_path = config.DIRS['models'] / ckpt_path.name
        download_model("https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth",
                       ckpt_path, "GroundingDINO Swin-T model", logger, error_handler, min_size=500_000_000)
        gdino_model = gdino_load_model(model_config_path=gdino_config_path, model_checkpoint_path=str(ckpt_path), device=device)
        logger.info("Grounding DINO model loaded.", component="grounding", user_context={'model_path': str(ckpt_path)})
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
def get_dam4sam_tracker(model_name: str, config: 'Config', logger=None):
    if not DAM4SAMTracker or not dam_utils: raise ImportError("DAM4SAM is not installed.")
    logger = logger or EnhancedLogger()
    error_handler = ErrorHandler(logger, config)
    if not (DAM4SAMTracker and torch and torch.cuda.is_available()):
        logger.error("DAM4SAM dependencies or CUDA not available.")
        return None
    try:
        logger.info("Initializing DAM4SAM tracker", extra={'model': model_name})
        model_urls = {"sam21pp-T": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt",
                      "sam21pp-S": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt",
                      "sam21pp-B+": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt",
                      "sam21pp-L": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"}
        checkpoint_path = config.DIRS['models'] / Path(model_urls[model_name]).name
        download_model(model_urls[model_name], checkpoint_path, f"{model_name} model", logger, error_handler, 100_000_000)
        actual_path, _ = dam_utils.determine_tracker(model_name)
        if not Path(actual_path).exists():
            Path(actual_path).parent.mkdir(exist_ok=True, parents=True)
            shutil.copy(checkpoint_path, actual_path)
        tracker = DAM4SAMTracker(model_name)
        logger.success("DAM4SAM tracker initialized.")
        return tracker
    except Exception as e:
        logger.error("Failed to initialize DAM4SAM tracker", exc_info=True)
        if "out of memory" in str(e) and torch.cuda.is_available():
            torch.cuda.empty_cache()
        return None

def initialize_analysis_models(params: AnalysisParameters, config: Config, logger: EnhancedLogger, cuda_available: bool):
    device = "cuda" if cuda_available else "cpu"
    face_analyzer, ref_emb, person_detector = None, None, None
    if params.enable_face_filter:
        face_analyzer = get_face_analyzer(params.face_model_name, config, logger)
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
    person_detector = get_person_detector(params.person_detector_model, device, config, logger)
    return {"face_analyzer": face_analyzer, "ref_emb": ref_emb, "person_detector": person_detector, "device": device}

# --- VIDEO & FRAME PROCESSING ---

class VideoManager:
    def __init__(self, source_path, max_resolution="maximum available"):
        self.source_path = source_path
        self.max_resolution = max_resolution
        self.is_youtube = ("youtube.com/" in source_path or "youtu.be/" in source_path)

    def prepare_video(self, config: 'Config', logger=None):
        logger = logger or EnhancedLogger()
        if self.is_youtube:
            if not ytdlp: raise ImportError("yt-dlp not installed.")
            logger.info("Downloading video", component="video", user_context={'source': self.source_path})
            res_filter = f"[height<={self.max_resolution}]" if self.max_resolution != "maximum available" else ""
            ydl_opts = {'outtmpl': str(config.DIRS['downloads'] / '%(id)s_%(title).40s_%(height)sp.%(ext)s'),
                        'format': f'bestvideo{res_filter}[ext=mp4]+bestaudio[ext=m4a]/best{res_filter}[ext=mp4]/best',
                        'merge_output_format': 'mp4', 'noprogress': True, 'quiet': True}
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
    logger = logger or EnhancedLogger()
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

def run_ffmpeg_extraction(video_path, output_dir, video_info, params, progress_queue, cancel_event, logger=None, tracker=None):
    logger = logger or EnhancedLogger()
    log_file_path = output_dir / "ffmpeg_log.txt"
    cmd_base = ['ffmpeg', '-y', '-i', str(video_path), '-hide_banner', '-loglevel', 'info']
    if params.thumbnails_only:
        thumb_dir = output_dir / "thumbs"
        thumb_dir.mkdir(exist_ok=True)
        target_area = params.thumb_megapixels * 1_000_000
        w, h = video_info.get('width', 1920), video_info.get('height', 1080)
        scale_factor = math.sqrt(target_area / (w * h))
        vf_scale = f"scale=w=trunc(iw*{scale_factor}/2)*2:h=trunc(ih*{scale_factor}/2)*2"
        fps = video_info.get('fps', 30)
        vf_filter = f"fps={fps},{vf_scale},showinfo"
        cmd = cmd_base + ['-vf', vf_filter, '-c:v', 'libwebp', '-lossless', '0', '-quality', '80', '-vsync', 'vfr', str(thumb_dir / "frame_%06d.webp")]
    else:
        select_filter_map = {'interval': f"fps=1/{max(0.1, float(params.interval))}", 'keyframes': "select='eq(pict_type,I)'",
                             'scene': f"select='gt(scene,{0.5 if params.fast_scene else 0.4})'", 'all': f"fps={video_info.get('fps', 30)}",
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

def postprocess_mask(mask: np.ndarray, fill_holes: bool = True, keep_largest_only: bool = True) -> np.ndarray:
    if mask is None or mask.size == 0: return mask
    binary_mask = (mask > 128).astype(np.uint8)
    if fill_holes:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    if keep_largest_only:
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
    if not frame_map_path.exists():
        thumb_files = sorted(list((output_dir / "thumbs").glob("frame_*.webp")), key=lambda p: int(re.search(r'frame_(\d+)', p.name).group(1)))
        return {int(re.search(r'frame_(\d+)', f.name).group(1)): f.name for f in thumb_files}
    try:
        with open(frame_map_path, 'r', encoding='utf-8') as f: frame_map_list = json.load(f)
        return {orig_num: f"frame_{i+1:06d}.webp" for i, orig_num in enumerate(sorted(frame_map_list))}
    except Exception:
        logger.error("Failed to parse frame_map.json. Using fallback.", exc_info=True)
        return {}

# --- MASKING & PROPAGATION ---

class MaskPropagator:
    def __init__(self, params, dam_tracker, progress_tracker, cancel_event, progress_queue, logger=None):
        self.params = params
        self.dam_tracker = dam_tracker
        self.progress_tracker = progress_tracker
        self.cancel_event = cancel_event
        self.progress_queue = progress_queue
        self.logger = logger or EnhancedLogger()
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
                if mask is not None: mask = postprocess_mask((mask * 255).astype(np.uint8), fill_holes=True, keep_largest_only=True)
                masks[i] = mask if mask is not None else np.zeros_like(shot_frames_rgb[i], dtype=np.uint8)[:, :, 0]
                if self.progress_tracker: self.progress_tracker.update_progress()
        try:
            with torch.cuda.amp.autocast(enabled=self._device == 'cuda'):
                outputs = self.dam_tracker.initialize(rgb_to_pil(shot_frames_rgb[seed_idx]), None, bbox=bbox_xywh)
                mask = outputs.get('pred_mask')
                if mask is not None: mask = postprocess_mask((mask * 255).astype(np.uint8), fill_holes=True, keep_largest_only=True)
                masks[seed_idx] = mask if mask is not None else np.zeros_like(shot_frames_rgb[seed_idx], dtype=np.uint8)[:, :, 0]
                if self.progress_tracker: self.progress_tracker.update_progress()
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
    def __init__(self, params, face_analyzer, reference_embedding, person_detector, tracker, gdino_model, logger=None):
        self.params = params
        self.face_analyzer = face_analyzer
        self.reference_embedding = reference_embedding
        self.person_detector = person_detector
        self.tracker = tracker
        self._gdino = gdino_model
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger = logger or EnhancedLogger()

    def select_seed(self, frame_rgb, current_params=None):
        p = self.params if current_params is None else current_params
        primary_strategy = getattr(p, "primary_seed_strategy", "🤖 Automatic")
        if isinstance(current_params, dict): primary_strategy = current_params.get('primary_seed_strategy', primary_strategy)
        use_face_filter = getattr(p, "enable_face_filter", False)
        if isinstance(current_params, dict): use_face_filter = current_params.get('enable_face_filter', use_face_filter)
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
        target_face, face_details = self._find_target_face(frame_rgb)
        if target_face and face_details.get('type') == 'face_match':
            self.logger.info("Face match found, proceeding with identity-first strategy.")
            yolo_boxes, dino_boxes = self._get_yolo_boxes(frame_rgb), self._get_dino_boxes(frame_rgb, params)[0]
            best_box, best_details = self._score_and_select_candidate(target_face, yolo_boxes, dino_boxes)
            if best_box:
                self.logger.success("Face-based seed selected.", extra=best_details)
                return best_box, best_details
            else:
                expanded_box = self._expand_face_to_body(target_face['bbox'], frame_rgb.shape)
                return expanded_box, {"type": "expanded_box_from_face", "seed_face_sim": face_details.get('seed_face_sim', 0)}
        self.logger.warning("Face detection failed, falling back to text prompt strategy.", extra=face_details)
        text_prompt = getattr(params, "text_prompt", "")
        if isinstance(params, dict): text_prompt = params.get('text_prompt', text_prompt)
        if not text_prompt:
            self.logger.warning("No text prompt provided for fallback. Using automatic strategy.")
            return self._choose_person_by_strategy(frame_rgb, params)
        dino_boxes, dino_details = self._get_dino_boxes(frame_rgb, params)
        if dino_boxes:
            yolo_boxes = self._get_yolo_boxes(frame_rgb)
            if yolo_boxes:
                best_iou, best_match = -1, None
                for d_box in dino_boxes:
                    for y_box in yolo_boxes:
                        iou = self._calculate_iou(d_box['bbox'], y_box['bbox'])
                        if iou > best_iou:
                            best_iou, best_match = iou, {'bbox': d_box['bbox'], 'type': 'text_fallback_dino_yolo', 'iou': iou,
                                                         'dino_conf': d_box['conf'], 'yolo_conf': y_box['conf'],
                                                         'fallback_reason': face_details.get('error', 'face_detection_failed')}
                if best_match and best_match['iou'] > 0.3:
                    self.logger.info("Text fallback successful with DINO+YOLO intersection.", extra=best_match)
                    return self._xyxy_to_xywh(best_match['bbox']), best_match
            fallback_details = {**dino_details, 'type': 'text_fallback_dino_only', 'fallback_reason': face_details.get('error', 'face_detection_failed')}
            self.logger.info("Text fallback using best DINO box.", extra=fallback_details)
            return self._xyxy_to_xywh(dino_boxes[0]['bbox']), fallback_details
        self.logger.warning("Both face and text strategies failed, using automatic fallback.")
        return self._choose_person_by_strategy(frame_rgb, params)

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
                if best_match and best_match['iou'] > 0.3:
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
        if best_face and best_sim > 0.4:
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
        prompt = getattr(params, "text_prompt", "")
        if isinstance(params, dict): prompt = params.get('text_prompt', prompt)
        if not self._gdino or not prompt: return [], {}
        box_th, text_th = getattr(params, "box_threshold", self.params.box_threshold), getattr(params, "text_threshold", self.params.text_threshold)
        if isinstance(params, dict):
            box_th, text_th = params.get('box_threshold', box_th), params.get('text_threshold', text_th)
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
                score += 100
                details['face_contained'] = True
            score += cand['conf'] * 20
            scored_candidates.append({'score': score, 'box': cand['bbox'], 'details': details})
        best_iou, best_pair = -1, None
        for y_box in yolo_boxes:
            for d_box in dino_boxes:
                iou = self._calculate_iou(y_box['bbox'], d_box['bbox'])
                if iou > best_iou: best_iou, best_pair = iou, (y_box, d_box)
        if best_iou > 0.5:
            for cand in scored_candidates:
                if np.array_equal(cand['box'], best_pair[0]['bbox']) or np.array_equal(cand['box'], best_pair[1]['bbox']):
                    cand['score'] += 50
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
        new_w, new_h = min(W, w * 4.0), min(H, h * 7.0)
        new_x1, new_y1 = max(0, cx - new_w / 2), max(0, y1 - h * 0.75)
        return [int(v) for v in [new_x1, new_y1, min(W, new_x1 + new_w) - new_x1, min(H, new_y1 + new_h) - new_y1]]

    def _final_fallback_box(self, img_shape): h, w, _ = img_shape; return [w // 4, h // 4, w // 2, h // 2]
    def _xyxy_to_xywh(self, box): x1, y1, x2, y2 = box; return [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]

    def _sam2_mask_for_bbox(self, frame_rgb_small, bbox_xywh):
        if not self.tracker or bbox_xywh is None:
            return None
        try:
            outputs = self.tracker.initialize(rgb_to_pil(frame_rgb_small), None, bbox=bbox_xywh)
            mask = outputs.get('pred_mask')
            if mask is not None: mask = postprocess_mask((mask * 255).astype(np.uint8), fill_holes=True, keep_largest_only=True)
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
                 reference_embedding=None, person_detector=None, thumbnail_manager=None, niqe_metric=None, logger=None, tracker=None):
        self.params, self.config, self.progress_queue, self.cancel_event = params, config, progress_queue, cancel_event
        self.logger = logger or EnhancedLogger()
        self.tracker, self.frame_map = tracker, frame_map
        self.face_analyzer, self.reference_embedding, self.person_detector = face_analyzer, reference_embedding, person_detector
        self.dam_tracker, self.mask_dir, self.shots = None, None, []
        self._gdino, self._sam2_img = None, None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self.thumbnail_manager = thumbnail_manager if thumbnail_manager is not None else ThumbnailManager(self.logger)
        self.niqe_metric = niqe_metric
        self._initialize_models()
        self.seed_selector = SeedSelector(params, face_analyzer, reference_embedding, person_detector, self.dam_tracker, self._gdino, logger=self.logger)
        self.mask_propagator = MaskPropagator(params, self.dam_tracker, self.tracker, cancel_event, progress_queue, logger=self.logger)

    def _initialize_models(self): self._init_grounder(); self._initialize_tracker()
    def _init_grounder(self):
        if self._gdino is not None: return True
        self._gdino = get_grounding_dino_model(self.params.gdino_config_path, self.params.gdino_checkpoint_path, self.config, self._device, self.logger)
        return self._gdino is not None
    def _initialize_tracker(self):
        if self.dam_tracker: return True
        self.dam_tracker = get_dam4sam_tracker(self.params.dam4sam_model_name, self.config, self.logger)
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
        self.tracker.start_stage("Mask Propagation", stage_items=total_scenes)
        for i, scene in enumerate(scenes_to_process):
            with safe_resource_cleanup():
                if self.cancel_event.is_set(): break
                self.logger.info(f"Masking scene {i+1}/{total_scenes}", user_context={'shot_id': scene.shot_id, 'start_frame': scene.start_frame, 'end_frame': scene.end_frame})
                self.tracker.update_progress(stage_items_processed=i, substage=f"Scene {scene.shot_id}")
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
        self.tracker.update_progress(stage_items_processed=total_scenes)
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
    def draw_bbox(self, img_rgb, xywh, color=(255, 0, 0), thickness=2):
        x, y, w, h = map(int, xywh or [0, 0, 0, 0])
        img_out = img_rgb.copy()
        cv2.rectangle(img_out, (x, y), (x + w, y + h), color, thickness)
        return img_out

# --- PIPELINES ---

class ExtractionPipeline(Pipeline):
    def __init__(self, params, progress_queue, cancel_event, logger=None, tracker=None):
        super().__init__(params, progress_queue, cancel_event, logger, tracker)
        self.config = None

    def run(self):
        self.tracker.start_stage("Preparing Video", 1)
        self.logger.info("Preparing video source...")
        vid_manager = VideoManager(self.params.source_path, self.params.max_resolution)
        video_path = Path(vid_manager.prepare_video(self.config, self.logger))
        output_dir = self.config.DIRS['downloads'] / video_path.stem
        output_dir.mkdir(exist_ok=True, parents=True)
        self.tracker.update_progress(stage_items_processed=1)
        self.logger.info("Video ready", user_context={'path': sanitize_filename(video_path.name)})
        video_info = VideoManager.get_video_info(video_path)
        if self.params.scene_detect:
            self.tracker.start_stage("Scene Detection")
            self._run_scene_detection(video_path, output_dir)
        self.tracker.start_stage("FFmpeg Extraction")
        self._run_ffmpeg(video_path, output_dir, video_info)
        if self.cancel_event.is_set():
            self.logger.info("Extraction cancelled by user.")
            return
        self.logger.success("Extraction complete.")
        return {"done": True, "output_dir": str(output_dir), "video_path": str(video_path)}

    def _run_scene_detection(self, video_path, output_dir): return run_scene_detection(video_path, output_dir, self.logger)
    def _run_ffmpeg(self, video_path, output_dir, video_info): return run_ffmpeg_extraction(video_path, output_dir, video_info, self.params, self.progress_queue, self.cancel_event, self.logger, self.tracker)

class EnhancedExtractionPipeline(ExtractionPipeline):
    def __init__(self, params, progress_queue, cancel_event, config: 'Config', logger=None, tracker=None):
        super().__init__(params, progress_queue, cancel_event, logger, tracker)
        self.config = config
        self.error_handler = ErrorHandler(self.logger, self.config)
        self.tracker = tracker
        self.run = self.error_handler.with_retry(max_attempts=3, backoff_seconds=[1, 5, 15])(self.run)

class AnalysisPipeline(Pipeline):
    def __init__(self, params, progress_queue, cancel_event, config: 'Config', thumbnail_manager=None, logger=None, tracker=None):
        super().__init__(params, progress_queue, cancel_event, logger=logger, tracker=tracker)
        self.config, self.output_dir = config, Path(self.params.output_folder)
        self.thumb_dir, self.masks_dir = self.output_dir / "thumbs", self.output_dir / "masks"
        self.frame_map_path, self.metadata_path = self.output_dir / "frame_map.json", self.output_dir / "metadata.jsonl"
        self.processing_lock = threading.Lock()
        self.face_analyzer, self.reference_embedding, self.mask_metadata = None, None, {}
        self.scene_map, self.niqe_metric = {}, None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.thumbnail_manager = thumbnail_manager if thumbnail_manager is not None else ThumbnailManager(self.logger)

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
            self.tracker.start_stage("Initializing Models")
            if self.params.enable_face_filter:
                self.face_analyzer = get_face_analyzer(self.params.face_model_name, self.config, logger=self.logger)
                if self.params.face_ref_img_path: self._process_reference_face()
            person_detector = get_person_detector(self.params.person_detector_model, self.device, self.config, logger=self.logger)
            masker = SubjectMasker(self.params, self.progress_queue, self.cancel_event, self.config, self._create_frame_map(),
                                   self.face_analyzer, self.reference_embedding, person_detector, thumbnail_manager=self.thumbnail_manager,
                                   niqe_metric=self.niqe_metric, logger=self.logger, tracker=self.tracker)
            self.mask_metadata = masker.run_propagation(str(self.output_dir), scenes_to_process)
            self._run_analysis_loop(scenes_to_process)
            if self.cancel_event.is_set(): return {"log": "Analysis cancelled."}
            self.logger.success("Analysis complete.", extra={'output_dir': self.output_dir})
            return {"done": True, "metadata_path": str(self.metadata_path), "output_dir": str(self.output_dir)}
        except Exception as e:
            self.logger.error("Analysis pipeline failed", component="analysis", exc_info=True, extra={'error': str(e)})
            return {"error": str(e)}

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
        self.tracker.start_stage("Analyzing Frames", stage_items=len(image_files_to_process))
        num_workers = 1 if self.params.disable_parallel else min(os.cpu_count() or 4, 8)
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(self._process_single_frame, path) for path in image_files_to_process]

            completed_count = 0
            while completed_count < len(futures):
                if self.cancel_event.is_set():
                    for f in futures:
                        f.cancel()
                    break

                # Non-blocking check for completed futures
                for future in futures[:]:  # Iterate over a copy
                    if future.done():
                        try:
                            future.result()  # To raise exceptions if any
                        except Exception as e:
                            self.logger.error(f"Error processing future: {e}")
                        completed_count +=1
                        self.tracker.update_progress(stage_items_processed=completed_count)
                time.sleep(0.1)  # Avoid busy-waiting

    def _process_single_frame(self, thumb_path):
        if self.cancel_event.is_set(): return
        if not (frame_num_match := re.search(r'frame_(\d+)', thumb_path.name)): return
        log_context = {'file': thumb_path.name}
        try:
            self._initialize_niqe_metric()
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
                enable_niqe='niqe' in self.config.QUALITY_METRICS
            )
            frame.calculate_quality_metrics(thumb_image_rgb, quality_conf, self.logger, mask=mask_thumb, niqe_metric=self.niqe_metric)
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
            hist_range = (0, 1) if k == 'face_sim' else (0, 100)
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
    metric_arrays = {k: np.array([f.get("metrics", {}).get(f"{k}_score", np.nan) for f in all_frames_data], dtype=np.float32) for k in config.QUALITY_METRICS}
    metric_arrays.update({"face_sim": np.array([f.get("face_sim", np.nan) for f in all_frames_data], dtype=np.float32),
                          "mask_area_pct": np.array([f.get("mask_area_pct", np.nan) for f in all_frames_data], dtype=np.float32)})
    kept_mask, reasons, dedup_mask = np.ones(num_frames, dtype=bool), defaultdict(list), np.ones(num_frames, dtype=bool)
    if filters.get("enable_dedup") and imagehash and filters.get("dedup_thresh", 5) != -1:
        sorted_indices, hashes = sorted(range(num_frames), key=lambda i: filenames[i]), {i: imagehash.hex_to_hash(all_frames_data[i]['phash']) for i in range(num_frames) if 'phash' in all_frames_data[i]}
        for i in range(1, len(sorted_indices)):
            c_idx, p_idx = sorted_indices[i], sorted_indices[i - 1]
            if p_idx in hashes and c_idx in hashes and (hashes[p_idx] - hashes[c_idx]) <= filters.get("dedup_thresh", 5):
                if dedup_mask[c_idx]: reasons[filenames[c_idx]].append('duplicate')
                dedup_mask[c_idx] = False
    metric_filter_mask = np.ones(num_frames, dtype=bool)
    for k in config.QUALITY_METRICS:
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
    kept_mask = dedup_mask & metric_filter_mask
    metric_rejection_mask = ~metric_filter_mask & dedup_mask
    for i in np.where(metric_rejection_mask)[0]:
        for k in config.QUALITY_METRICS:
            min_v, max_v = filters.get(f"{k}_min", 0), filters.get(f"{k}_max", 100)
            if not (min_v <= metric_arrays[k][i] <= max_v): reasons[filenames[i]].append(f"{k}_{'low' if metric_arrays[k][i] < min_v else 'high'}")
        if filters.get("face_sim_enabled"):
            if metric_arrays["face_sim"][i] < filters.get("face_sim_min", 0.5): reasons[filenames[i]].append("face_sim_low")
            if filters.get("require_face_match") and np.isnan(metric_arrays["face_sim"][i]): reasons[filenames[i]].append("face_missing")
        if filters.get("mask_area_enabled") and metric_arrays["mask_area_pct"][i] < filters.get("mask_area_pct_min", 1.0): reasons[filenames[i]].append("mask_too_small")
    kept, rejected = [all_frames_data[i] for i in np.where(kept_mask)[0]], [all_frames_data[i] for i in np.where(~kept_mask)[0]]
    return kept, rejected, Counter(r for r_list in reasons.values() for r in r_list), reasons

def on_filters_changed(event: FilterEvent, thumbnail_manager, config: 'Config', logger=None):
    logger = logger or EnhancedLogger()
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
        default_val = config.filter_defaults[metric_key][default_key]
        output_values[f'slider_{key}'] = gr.update(value=default_val)
        slider_default_values.append(default_val)
    face_match_default, dedup_default = config.ui_defaults['require_face_match'], config.filter_defaults['dedup_thresh']['default']
    output_values.update({'require_face_match_input': gr.update(value=face_match_default), 'dedup_thresh_input': gr.update(value=dedup_default)})
    if all_frames_data:
        slider_defaults_dict = {key: val for key, val in zip(slider_keys, slider_default_values)}
        filter_event = FilterEvent(all_frames_data, per_metric_values, output_dir, "Kept Frames", True, 0.6, face_match_default, dedup_default, slider_defaults_dict)
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
    scene_seeds = {str(s['shot_id']): {'seed_frame_idx': s.get('best_seed_frame'), 'seed_type': s.get('seed_result', {}).get('details', {}).get('type'),
                                       'seed_config': s.get('seed_config', {}), 'status': s.get('status', 'pending'), 'seed_metrics': s.get('seed_metrics', {})} for s in scenes_list}
    try:
        (Path(output_dir_str) / "scene_seeds.json").write_text(json.dumps(_to_json_safe(scene_seeds), indent=2), encoding='utf-8')
        logger.info("Saved scene_seeds.json")
    except Exception as e: logger.error("Failed to save scene_seeds.json", exc_info=True)

def get_scene_status_text(scenes_list):
    if not scenes_list: return "No scenes loaded."
    return f"{sum(1 for s in scenes_list if s['status'] == 'included')}/{len(scenes_list)} scenes included for propagation."

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
        fname = masker.frame_map.get(scene_dict['best_seed_frame'])
        if not fname:
            raise ValueError(f"Framemap lookup failed for re-seeding shot {scene_dict.get('shot_id')} frame {scene_dict.get('best_seed_frame')}.")
        thumb_rgb = thumbnail_manager.get(Path(output_folder) / "thumbs" / f"{Path(fname).stem}.webp")
        bbox, details = masker.get_seed_for_frame(thumb_rgb, scene_dict['seed_config'])
        scene_dict['seed_result'] = {'bbox': bbox, 'details': details}
        save_scene_seeds(scenes_list, output_folder, logger)
        updated_gallery_previews = _regenerate_all_previews(scenes_list, output_folder, masker, thumbnail_manager, logger)
        return (updated_gallery_previews, scenes_list, f"Scene {selected_shot_id} updated and saved.")
    except Exception as e:
        logger.error("Failed to apply scene overrides", exc_info=True)
        return None, scenes_list, f"[ERROR] {e}"

def _regenerate_all_previews(scenes_list, output_folder, masker, thumbnail_manager, logger):
    previews, output_dir = [], Path(output_folder)
    for scene_dict in scenes_list:
        fname = masker.frame_map.get(scene_dict['best_seed_frame'])
        if not fname: continue
        thumb_rgb = thumbnail_manager.get(output_dir / "thumbs" / f"{Path(fname).stem}.webp")
        if thumb_rgb is None: continue
        bbox, details = scene_dict.get('seed_result', {}).get('bbox'), scene_dict.get('seed_result', {}).get('details', {})
        mask = masker._sam2_mask_for_bbox(thumb_rgb, bbox) if bbox else None
        overlay_rgb = render_mask_overlay(thumb_rgb, mask, 0.6, logger=logger) if mask is not None else masker.draw_bbox(thumb_rgb, bbox)
        previews.append((overlay_rgb, f"Scene {scene_dict['shot_id']} (Seed: {scene_dict['best_seed_frame']}) | {details.get('type', 'N/A')}"))
    return previews

# --- PIPELINE LOGIC ---

def execute_extraction(event: ExtractionEvent, progress_queue: Queue, cancel_event: threading.Event, logger: EnhancedLogger, config: Config, tracker: AdvancedProgressTracker):
    params_dict = asdict(event)
    if event.upload_video:
        source, dest = params_dict.pop('upload_video'), str(config.DIRS['downloads'] / Path(event.upload_video).name)
        shutil.copy2(source, dest)
        params_dict['source_path'] = dest
    params = AnalysisParameters.from_ui(logger, config, **params_dict)
    pipeline = EnhancedExtractionPipeline(params, progress_queue, cancel_event, config, logger, tracker)
    result = pipeline.run()
    if result and result.get("done"):
        yield {"log": "Extraction complete.", "status": f"Output: {result['output_dir']}",
               "extracted_video_path_state": result.get("video_path", ""), "extracted_frames_dir_state": result["output_dir"], "done": True}

def execute_pre_analysis(event: PreAnalysisEvent, progress_queue: Queue, cancel_event: threading.Event, logger: EnhancedLogger,
                         config: Config, thumbnail_manager, cuda_available, tracker: AdvancedProgressTracker):
    yield {"unified_log": "", "unified_status": "Starting Pre-Analysis..."}
    params_dict = asdict(event)
    final_face_ref_path = params_dict.get('face_ref_img_path')
    if event.face_ref_img_upload:
        ref_upload, dest = params_dict.pop('face_ref_img_upload'), config.DIRS['downloads'] / Path(event.face_ref_img_upload).name
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
    tracker.start_stage("Loading Models", 2)
    models, niqe_metric = initialize_analysis_models(params, config, logger, cuda_available), None
    tracker.update_progress(stage_items_processed=1)
    if params.pre_analysis_enabled and pyiqa: niqe_metric = pyiqa.create_metric('niqe', device=models['device'])
    masker = SubjectMasker(params, progress_queue, cancel_event, config, face_analyzer=models["face_analyzer"],
                           reference_embedding=models["ref_emb"], person_detector=models["person_detector"],
                           niqe_metric=niqe_metric, thumbnail_manager=thumbnail_manager, logger=logger, tracker=tracker)
    masker.frame_map = masker._create_frame_map(str(output_dir))
    tracker.update_progress(stage_items_processed=2)
    def pre_analysis_task():
        tracker.start_stage("Pre-analyzing scenes", stage_items=len(scenes))
        previews = []
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
            previews.append((overlay_rgb, f"Scene {scene.shot_id} (Seed: {scene.best_seed_frame}) | {details.get('type', 'N/A')}"))
            if scene.status == 'pending': scene.status = 'included'
            tracker.update_progress(stage_items_processed=i + 1)
        return {"done": True, "previews": previews, "scenes": [asdict(s) for s in scenes]}
    result = pre_analysis_task()
    if result.get("done"):
        final_yield = {"log": "Pre-analysis complete.", "status": f"{len(result['scenes'])} scenes found.", "previews": result['previews'],
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
    event: SessionLoadEvent,
    logger: EnhancedLogger,
    config: Config,
    thumbnail_manager,
    tracker: AdvancedProgressTracker,
):
    session_path = Path(event.session_path).expanduser().resolve()
    config_path = session_path / "run_config.json"
    scene_seeds_path = session_path / "scene_seeds.json"
    metadata_path = session_path / "metadata.jsonl"

    def _resolve_output_dir(base: Path, output_folder: str | None) -> Path | None:
        if not output_folder:
            return None
        try:
            p = Path(output_folder)
            if not p.is_absolute():
                p = (base / p).resolve()
            return p
        except Exception:
            return None

    with logger.operation_context("Load Session", component="session_loader"):
        tracker.start_operation("Load Session", total_items=3, stages=["Load config", "Load scenes", "Load previews"])

        # Stage 1: Load config
        tracker.start_stage("Load config", stage_items=1)
        if not config_path.exists():
            msg = f"Session load failed: run_config.json not found in {session_path}"
            logger.error(msg, component="session_loader")
            tracker.complete_operation(False, msg)
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
            tracker.complete_operation(False, msg)
            yield {"log": f"[ERROR] {msg}", "status": "Session load failed."}
            return

        output_dir = _resolve_output_dir(session_path, run_config.get("output_folder"))
        if output_dir is None:
            logger.warning("Output folder missing or invalid; some previews may be unavailable.", component="session_loader")

        tracker.update_progress(items_processed=1, stage_items_processed=1)

        # Prepare initial UI updates from config
        updates = {
            "source_input": gr.update(value=run_config.get("source_path", "")),
            "max_resolution": gr.update(value=run_config.get("max_resolution", "1080")),
            "thumbnails_only_input": gr.update(value=run_config.get("thumbnails_only", True)),
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
            "extracted_frames_dir_state": str(output_dir) if output_dir else "",
            "analysis_output_dir_state": str(output_dir) if output_dir else "",
        }

        # Stage 2: Load scenes
        tracker.start_stage("Load scenes", stage_items=1)
        scenes_as_dict: list[dict[str, Any]] = []
        if scene_seeds_path.exists():
            try:
                with open(scene_seeds_path, "r", encoding="utf-8") as f:
                    scenes_from_file = json.load(f)
                # Normalize and keep shot_id as int
                scenes_as_dict = [
                    {"shot_id": int(shot_id), **(scene_data or {})}
                    for shot_id, scene_data in (scenes_from_file or {}).items()
                ]
                logger.info(f"Loaded {len(scenes_as_dict)} scenes from {scene_seeds_path}", component="session_loader")
            except Exception as e:
                logger.warning(f"Failed to parse scene_seeds.json: {e}", component="session_loader", error_type=type(e).__name__)
        tracker.update_progress(stage_items_processed=1)

        # Stage 3: Load previews (with cap and fallback)
        preview_cap = 100
        tracker.start_stage("Load previews", stage_items=min(len(scenes_as_dict), preview_cap))
        previews: list[tuple[np.ndarray, str]] = []

        frame_map: dict[int, str] = {}
        if output_dir and output_dir.exists():
            try:
                frame_map = create_frame_map(output_dir, logger)
            except Exception as e:
                logger.warning(
                    f"create_frame_map failed: {e}",
                    component="session_loader",
                    error_type=type(e).__name__,
                )

        def _load_preview(scene: dict[str, Any]) -> tuple[np.ndarray, str] | None:
            seed_idx = scene.get("best_seed_frame") or scene.get("seed_frame_idx")
            if seed_idx is None:
                return None
            fname = frame_map.get(int(seed_idx)) if frame_map else None
            if not fname:
                return None
            stem = Path(fname).stem
            thumb_path = (output_dir / "thumbs" / f"{stem}.webp") if output_dir else None
            if thumb_path is None:
                return None
            # Prefer cache manager; fall back to direct disk read
            thumb_rgb = thumbnail_manager.get(thumb_path)
            if thumb_rgb is None and thumb_path.exists():
                try:
                    # Avoid cv2.imread unicode issues on Windows
                    data = np.fromfile(str(thumb_path), dtype=np.uint8)
                    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
                    if bgr is not None:
                        thumb_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                except Exception:
                    thumb_rgb = None
            if thumb_rgb is not None:
                return (thumb_rgb, f"Scene {scene.get('shot_id', '?')} (Seed: {seed_idx})")
            return None

        if scenes_as_dict:
            try:
                from concurrent.futures import ThreadPoolExecutor
                max_workers = min(8, (os.cpu_count() or 4))
                with ThreadPoolExecutor(max_workers=max_workers) as ex:
                    for i, res in enumerate(ex.map(_load_preview, scenes_as_dict[:preview_cap])):
                        if res:
                            previews.append(res)
                        tracker.update_progress(stage_items_processed=i + 1)
                if len(scenes_as_dict) > preview_cap:
                    logger.info(
                        f"Preview cap reached ({preview_cap}). Remaining previews will load on demand.",
                        component="session_loader"
                    )
            except Exception as e:
                logger.warning(f"Concurrent preview loading failed: {e}", component="session_loader", error_type=type(e).__name__)

            updates.update({
                "scenes_state": scenes_as_dict,
                "propagate_masks_button": gr.update(interactive=True),
                "seeding_preview_gallery": gr.update(value=previews, visible=True),
                "seeding_results_column": gr.update(visible=True),
                "propagation_group": gr.update(visible=True),
                "scene_filter_status": get_scene_status_text(scenes_as_dict),
                "scene_face_sim_min_input": gr.update(
                    visible=any((s.get("seed_metrics") or {}).get("best_face_sim") is not None for s in scenes_as_dict)
                ),
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
        tracker.complete_operation(True, "Session loaded")
        yield updates

def execute_propagation(event: PropagationEvent, progress_queue: Queue, cancel_event: threading.Event, logger: EnhancedLogger,
                        config: Config, thumbnail_manager, cuda_available, tracker: AdvancedProgressTracker):
    scenes_to_process = [Scene(**s) for s in event.scenes if s['status'] == 'included']
    if not scenes_to_process:
        yield {"log": "No scenes were included for propagation.", "status": "Propagation skipped."}
        return
    params, pipeline = AnalysisParameters.from_ui(logger, config, **asdict(event.analysis_params)), AnalysisPipeline(AnalysisParameters.from_ui(logger, config, **asdict(event.analysis_params)), progress_queue, cancel_event, config, thumbnail_manager=thumbnail_manager, logger=logger, tracker=tracker)
    result = pipeline.run_full_analysis(scenes_to_process)
    if result and result.get("done"):
        yield {"log": "Propagation and analysis complete.", "status": f"Metadata saved to {result['metadata_path']}",
               "output_dir": result['output_dir'], "metadata_path": result['metadata_path'], "done": True}

# --- UI ---

class AppUI:
    def __init__(self, config=None, logger=None, progress_queue=None, cancel_event=None, thumbnail_manager=None):
        self.config = config or Config()
        self.logger = logger or EnhancedLogger()
        self.progress_queue = progress_queue or Queue()
        self.cancel_event = cancel_event or threading.Event()
        self.components, self.cuda_available = {}, torch.cuda.is_available()
        self.thumbnail_manager = thumbnail_manager or ThumbnailManager(logger=self.logger, max_size=self.config.thumbnail_cache_size)
        self.ext_ui_map_keys = ['source_path', 'upload_video', 'method', 'interval', 'nth_frame', 'fast_scene',
                                'max_resolution', 'use_png', 'thumbnails_only', 'thumb_megapixels', 'scene_detect']
        self.ana_ui_map_keys = ['output_folder', 'video_path', 'resume', 'enable_face_filter', 'face_ref_img_path', 'face_ref_img_upload',
                                'face_model_name', 'enable_subject_mask', 'dam4sam_model_name', 'person_detector_model', 'seed_strategy',
                                'scene_detect', 'enable_dedup', 'text_prompt', 'box_threshold', 'text_threshold', 'min_mask_area_pct',
                                'sharpness_base_scale', 'edge_strength_base_scale', 'gdino_config_path', 'gdino_checkpoint_path',
                                'pre_analysis_enabled', 'pre_sample_nth', 'primary_seed_strategy']
        self.session_load_keys = ['unified_log', 'unified_status', 'progress_bar', 'progress_details', 'cancel_button', 'pause_button',
                                  'source_input', 'max_resolution', 'thumbnails_only_input', 'thumb_megapixels_input', 'ext_scene_detect_input',
                                  'method_input', 'use_png_input', 'pre_analysis_enabled_input', 'pre_sample_nth_input', 'enable_face_filter_input',
                                  'face_model_name_input', 'face_ref_img_path_input', 'text_prompt_input', 'seed_strategy_input',
                                  'person_detector_model_input', 'dam4sam_model_name_input', 'enable_dedup_input', 'extracted_video_path_state',
                                  'extracted_frames_dir_state', 'analysis_output_dir_state', 'analysis_metadata_path_state', 'scenes_state',
                                  'propagate_masks_button', 'seeding_preview_gallery', 'seeding_results_column', 'propagation_group',
                                  'scene_filter_status', 'scene_face_sim_min_input', 'filtering_tab']

    def build_ui(self):
        css = """.plot-and-slider-column { max-width: 560px !important; margin: auto; } .scene-editor { border: 1px solid #444; padding: 10px; border-radius: 5px; } .log-container > .gr-utils-error { display: none !important; } .progress-details { font-size: 0.8em; color: #888; text-align: center; }"""
        with gr.Blocks(theme=gr.themes.Default(), css=css) as demo:
            self._build_header()
            with gr.Accordion("🔄 resume previous Session", open=False):
                with gr.Row():
                    self._create_component('session_path_input', 'textbox', {'label': "Load previous run", 'placeholder': "Path to a previous run's output folder..."})
                    self._create_component('load_session_button', 'button', {'value': "📂 Load Session"})
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
            with gr.Column(scale=2): self._create_component('source_input', 'textbox', {'label': "Video URL or Local Path", 'placeholder': "Enter YouTube URL or local video file path"})
            with gr.Column(scale=1): self._create_component('max_resolution', 'dropdown', {'choices': ["maximum available", "2160", "1080", "720"], 'value': self.config.ui_defaults['max_resolution'], 'label': "Download Resolution"})
        self._create_component('upload_video_input', 'file', {'label': "Or Upload a Video File", 'file_types': ["video"], 'type': "filepath"})
        gr.Markdown("---"); gr.Markdown("### Step 2: Configure Extraction Method")
        self._create_component('thumbnails_only_input', 'checkbox', {'label': "Use Recommended Thumbnail Extraction (Faster, For Pre-Analysis)", 'value': self.config.ui_defaults['thumbnails_only']})
        with gr.Accordion("Thumbnail Settings", open=True) as recommended_accordion:
            self.components['recommended_accordion'] = recommended_accordion
            gr.Markdown("This is the fastest and most efficient method. It extracts lightweight thumbnails for scene analysis, allowing you to quickly find the best frames *before* extracting full-resolution images.")
            self._create_component('thumb_megapixels_input', 'slider', {'label': "Thumbnail Size (MP)", 'minimum': 0.1, 'maximum': 2.0, 'step': 0.1, 'value': self.config.ui_defaults['thumb_megapixels']})
            self._create_component('ext_scene_detect_input', 'checkbox', {'label': "Use Scene Detection (Recommended)", 'value': self.config.ui_defaults['scene_detect']})
        with gr.Accordion("Advanced: Legacy Full-Frame Extraction", open=False) as legacy_accordion:
            self.components['legacy_accordion'] = legacy_accordion
            gr.Markdown("This method extracts full-resolution frames directly, which can be slow and generate a large number of files. Use this only if you have specific needs and understand the performance implications.")
            self._create_component('method_input', 'dropdown', {'choices': ["keyframes", "interval", "every_nth_frame", "all", "scene"], 'value': self.config.ui_defaults['method'], 'label': "Extraction Method"})
            self._create_component('interval_input', 'textbox', {'label': "Interval (seconds)", 'value': self.config.ui_defaults['interval'], 'visible': False})
            self._create_component('nth_frame_input', 'textbox', {'label': "N-th Frame Value", 'value': self.config.ui_defaults["nth_frame"], 'visible': False})
            self._create_component('fast_scene_input', 'checkbox', {'label': "Fast Scene Detect (for 'scene' method)", 'visible': False})
            self._create_component('use_png_input', 'checkbox', {'label': "Save as PNG (slower, larger files)", 'value': self.config.ui_defaults['use_png']})
        gr.Markdown("---"); gr.Markdown("### Step 3: Start Extraction")
        self.components.update({'start_extraction_button': gr.Button("🚀 Start Extraction", variant="primary")})

    def _create_analysis_tab(self):
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 🎯 Step 1: Choose Your Seeding Strategy")
                default_strategy = "🔄 Face + Text Fallback"
                self._create_component('primary_seed_strategy_input', 'radio', {'choices': ["👤 By Face", "📝 By Text", "🔄 Face + Text Fallback", "🤖 Automatic"], 'value': default_strategy, 'label': "Primary Seeding Strategy"})
                with gr.Group(visible=(default_strategy == "👤 By Face" or default_strategy == "🔄 Face + Text Fallback")) as face_seeding_group:
                    self.components['face_seeding_group'] = face_seeding_group
                    gr.Markdown("#### 👤 Configure Face Seeding"); gr.Markdown("Upload a clear image of the person you want to find. The system will search for this person in the video.")
                    self._create_component('face_ref_img_upload_input', 'file', {'label': "Upload Face Reference Image", 'type': "filepath"})
                    self._create_component('face_ref_img_path_input', 'textbox', {'label': "Or provide a local file path"})
                    self._create_component('enable_face_filter_input', 'checkbox', {'label': "Enable Face Similarity (must be checked for face seeding)", 'value': (default_strategy == "👤 By Face" or default_strategy == "🔄 Face + Text Fallback"), 'interactive': False})
                with gr.Group(visible=(default_strategy == "📝 By Text" or default_strategy == "🔄 Face + Text Fallback")) as text_seeding_group:
                    self.components['text_seeding_group'] = text_seeding_group
                    gr.Markdown("#### 📝 Configure Text Seeding"); gr.Markdown("Describe the subject or object you want to find. Be as specific as possible for better results.")
                    self._create_component('text_prompt_input', 'textbox', {'label': "Text Prompt", 'placeholder': "e.g., 'a woman in a red dress'", 'value': self.config.ui_defaults['text_prompt']})
                with gr.Group(visible=(default_strategy == "🤖 Automatic")) as auto_seeding_group:
                    self.components['auto_seeding_group'] = auto_seeding_group
                    gr.Markdown("#### 🤖 Configure Automatic Seeding"); gr.Markdown("The system will automatically identify the most prominent person in each scene. This is a good general-purpose starting point.")
                    self._create_component('seed_strategy_input', 'dropdown', {'choices': ["Largest Person", "Center-most Person"], 'value': "Largest Person", 'label': "Automatic Seeding Method"})
                with gr.Accordion("Advanced Settings", open=False):
                    gr.Markdown("These settings control the underlying models and analysis parameters. Adjust them only if you understand their effect.")
                    self._create_component('pre_analysis_enabled_input', 'checkbox', {'label': 'Enable Pre-Analysis to find best seed frame', 'value': self.config.ui_defaults['pre_analysis_enabled']})
                    self._create_component('pre_sample_nth_input', 'number', {'label': 'Sample every Nth thumbnail for pre-analysis', 'value': self.config.ui_defaults['pre_sample_nth'], 'interactive': True})
                    self._create_component('person_detector_model_input', 'dropdown', {'choices': ['yolo11x.pt', 'yolo11s.pt'], 'value': self.config.ui_defaults['person_detector_model'], 'label': "Person Detector (for Automatic)"})
                    self._create_component('face_model_name_input', 'dropdown', {'choices': ["buffalo_l", "buffalo_s"], 'value': self.config.ui_defaults['face_model_name'], 'label': "Face Model (for Face Seeding)"})
                    self._create_component('dam4sam_model_name_input', 'dropdown', {'choices': ["sam21pp-T", "sam21pp-S", "sam21pp-B+", "sam21pp-L"], 'value': self.config.ui_defaults['dam4sam_model_name'], 'label': "SAM Tracker Model"})
                    self._create_component('enable_dedup_input', 'checkbox', {'label': "Enable Deduplication (pHash)", 'value': self.config.ui_defaults.get('enable_dedup', False)})
                self._create_component('start_pre_analysis_button', 'button', {'value': '🌱 Find & Preview Scene Seeds', 'variant': 'primary'})
                with gr.Group(visible=False) as propagation_group:
                    self.components['propagation_group'] = propagation_group
                    gr.Markdown("---"); gr.Markdown("### 🔬 Step 3: Propagate Masks"); gr.Markdown("Once you are satisfied with the seeds, propagate the masks to the rest of the frames in the selected scenes.")
                    self._create_component('propagate_masks_button', 'button', {'value': '🔬 Propagate Masks on Kept Scenes', 'variant': 'primary', 'interactive': False})
            with gr.Column(scale=2, visible=False) as seeding_results_column:
                self.components['seeding_results_column'] = seeding_results_column
                gr.Markdown("### 🎭 Step 2: Review & Refine Seeds")
                self._create_component('seeding_preview_gallery', 'gallery', {'label': 'Scene Seed Previews', 'columns': [4, 6, 8], 'rows': 2, 'height': 'auto', 'preview': True, 'allow_preview': True, 'object_fit': 'contain'})
                with gr.Accordion("Scene Editor", open=False, elem_classes="scene-editor") as scene_editor_accordion:
                    self.components['scene_editor_accordion'] = scene_editor_accordion
                    self._create_component('scene_editor_status_md', 'markdown', {'value': "Select a scene to edit."})
                    with gr.Row(): self._create_component('scene_editor_prompt_input', 'textbox', {'label': 'Per-Scene Text Prompt'})
                    with gr.Row():
                        self._create_component('scene_editor_box_thresh_input', 'slider', {'label': "Box Thresh", 'minimum': 0.0, 'maximum': 1.0, 'step': 0.05, 'value': self.config.grounding_dino_params['box_threshold']})
                        self._create_component('scene_editor_text_thresh_input', 'slider', {'label': "Text Thresh", 'minimum': 0.0, 'maximum': 1.0, 'step': 0.05, 'value': self.config.grounding_dino_params['text_threshold']})
                    with gr.Row():
                        self._create_component('scene_recompute_button', 'button', {'value': '🔄 Recompute Preview'})
                        self._create_component('scene_include_button', 'button', {'value': '👍 Include'})
                        self._create_component('scene_exclude_button', 'button', {'value': '👎 Exclude'})
                with gr.Accordion("Bulk Scene Actions & Filters", open=True):
                    self._create_component('scene_filter_status', 'markdown', {'value': 'No scenes loaded.'})
                    self._create_component('scene_mask_area_min_input', 'slider', {'label': "Min Seed Mask Area %", 'minimum': 0.0, 'maximum': 100.0, 'value': self.config.min_mask_area_pct, 'step': 0.1})
                    self._create_component('scene_face_sim_min_input', 'slider', {'label': "Min Seed Face Sim", 'minimum': 0.0, 'maximum': 1.0, 'value': 0.5, 'step': 0.05, 'visible': False})
                    self._create_component('scene_confidence_min_input', 'slider', {'label': "Min Seed Confidence", 'minimum': 0.0, 'maximum': 1.0, 'value': 0.0, 'step': 0.05})
                    with gr.Row():
                        self._create_component('bulk_include_all_button', 'button', {'value': 'Include All'})
                        self._create_component('bulk_exclude_all_button', 'button', {'value': 'Exclude All'})

    def _create_filtering_tab(self):
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 🎛️ Filter Controls")
                self._create_component('auto_pctl_input', 'slider', {'label': 'Auto-Threshold Percentile', 'minimum': 1, 'maximum': 99, 'value': 75, 'step': 1})
                with gr.Row():
                    self._create_component('apply_auto_button', 'button', {'value': 'Apply Percentile to Mins'})
                    self._create_component('reset_filters_button', 'button', {'value': "Reset Filters"})
                self._create_component('filter_status_text', 'markdown', {'value': "Load an analysis to begin."})
                self.components['metric_plots'], self.components['metric_sliders'] = {}, {}
                with gr.Accordion("Deduplication", open=True, visible=True):
                    f_def = self.config.filter_defaults['dedup_thresh']
                    self._create_component('dedup_thresh_input', 'slider', {'label': "Similarity Threshold", 'minimum': f_def['min'], 'maximum': f_def['max'], 'value': f_def['default'], 'step': f_def['step']})
                for metric_name, open_default in [('niqe', True), ('sharpness', True), ('edge_strength', True), ('contrast', True),
                                                 ('brightness', False), ('entropy', False), ('face_sim', False), ('mask_area_pct', False)]:
                    if metric_name not in self.config.filter_defaults: continue
                    f_def = self.config.filter_defaults[metric_name]
                    with gr.Accordion(metric_name.replace('_', ' ').title(), open=open_default):
                        with gr.Column(elem_classes="plot-and-slider-column"):
                            self.components['metric_plots'][metric_name] = self._create_component(f'plot_{metric_name}', 'html', {'visible': False})
                            self.components['metric_sliders'][f"{metric_name}_min"] = self._create_component(f'slider_{metric_name}_min', 'slider', {'label': "Min", 'minimum': f_def['min'], 'maximum': f_def['max'], 'value': f_def['default_min'], 'step': f_def['step'], 'interactive': True, 'visible': False})
                            if 'default_max' in f_def: self.components['metric_sliders'][f"{metric_name}_max"] = self._create_component(f'slider_{metric_name}_max', 'slider', {'label': "Max", 'minimum': f_def['min'], 'maximum': f_def['max'], 'value': f_def['default_max'], 'step': f_def['step'], 'interactive': True, 'visible': False})
                            if metric_name == "face_sim": self._create_component('require_face_match_input', 'checkbox', {'label': "Reject if no face", 'value': self.config.ui_defaults['require_face_match'], 'visible': False})
            with gr.Column(scale=2):
                with gr.Group(visible=False) as results_group:
                    self.components['results_group'] = results_group
                    gr.Markdown("### 🖼️ Step 2: Review Results")
                    with gr.Row():
                        self._create_component('gallery_view_toggle', 'radio', {'choices': ["Kept Frames", "Rejected Frames"], 'value': "Kept Frames", 'label': "Show in Gallery"})
                        self._create_component('show_mask_overlay_input', 'checkbox', {'label': "Show Mask Overlay", 'value': True})
                        self._create_component('overlay_alpha_slider', 'slider', {'label': "Overlay Alpha", 'minimum': 0.0, 'maximum': 1.0, 'value': 0.6, 'step': 0.1})
                    self._create_component('results_gallery', 'gallery', {'columns': [4, 6, 8], 'rows': 2, 'height': 'auto', 'preview': True, 'allow_preview': True, 'object_fit': 'contain'})
                with gr.Group(visible=False) as export_group:
                    self.components['export_group'] = export_group
                    gr.Markdown("### 📤 Step 3: Export")
                    self._create_component('export_button', 'button', {'value': "Export Kept Frames", 'variant': "primary"})
                    with gr.Accordion("Export Options", open=True):
                        with gr.Row():
                            self._create_component('enable_crop_input', 'checkbox', {'label': "✂️ Crop to Subject", 'value': True})
                            self._create_component('crop_padding_input', 'slider', {'label': "Padding %", 'value': 1})
                        self._create_component('crop_ar_input', 'textbox', {'label': "Crop ARs", 'value': "16:9,1:1,9:16", 'info': "Comma-separated list (e.g., 16:9, 1:1). The best-fitting AR for each subject's mask will be chosen automatically."})

    def get_all_filter_keys(self): return self.config.QUALITY_METRICS + ["face_sim", "mask_area_pct"]
    def _create_event_handlers(self):
        self.components.update({'extracted_video_path_state': gr.State(""), 'extracted_frames_dir_state': gr.State(""),
                                'analysis_output_dir_state': gr.State(""), 'analysis_metadata_path_state': gr.State(""),
                                'all_frames_data_state': gr.State([]), 'per_metric_values_state': gr.State({}),
                                'scenes_state': gr.State([]), 'selected_scene_id_state': gr.State(None)})
        self._setup_visibility_toggles(); self._setup_pipeline_handlers(); self._setup_filtering_handlers(); self._setup_scene_editor_handlers(); self._setup_bulk_scene_handlers()

class EnhancedAppUI(AppUI):
    def __init__(self, config=None, logger=None, progress_queue=None, cancel_event=None, thumbnail_manager=None, resource_manager=None, progress_tracker=None):
        super().__init__(config, logger, progress_queue, cancel_event, thumbnail_manager)
        self.enhanced_logger, self.progress_tracker, self.resource_manager = logger, progress_tracker, resource_manager
        self.performance_metrics, self.log_filter_level, self.all_logs = {}, "INFO", []

    def _build_footer(self):
        with gr.Row():
            with gr.Column(scale=3):
                with gr.Row():
                    self._create_component('log_level_filter', 'dropdown', {'choices': ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'SUCCESS', 'CRITICAL'], 'value': 'INFO', 'label': 'Log Level Filter', 'scale': 1})
                    self._create_component('clear_logs_button', 'button', {'value': '🗑️ Clear Logs', 'scale': 1})
                    self._create_component('export_logs_button', 'button', {'value': '📥 Export Logs', 'scale': 1})
                self._create_component('unified_log', 'textbox', {'label': '📋 Enhanced Processing Log', 'lines': 15, 'interactive': False, 'autoscroll': True, 'elem_classes': ['log-container']})
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
            future = executor.submit(task_func, *args, tracker=self.progress_tracker)
            while not future.done():
                if self.cancel_event.is_set(): future.cancel(); break
                try:
                    msg, update_dict = self.progress_queue.get(timeout=0.1), {}
                    if "log" in msg:
                        self.all_logs.append(msg['log'])
                        if self.log_filter_level.upper() == "DEBUG" or f"[{self.log_filter_level.upper()}]" in msg['log']:
                            update_dict[self.components['unified_log']] = gr.update(value="\n".join([l for l in self.all_logs if self.log_filter_level.upper() == "DEBUG" or f"[{self.log_filter_level.upper()}]" in l][-1000:]))
                    elif "detailed_status" in msg:
                        progress(msg.get('progress', 0.0), msg.get('detailed_status', '...'))
                        update_dict[self.components['unified_status']] = self._format_status_display(msg.get('operation', op_name), msg.get('progress', 0), msg.get('stage', '...'))
                        update_dict[self.components['progress_details']] = f"ETA: {msg.get('eta', 'N/A')} | Rate: {msg.get('rate', 'N/A')}"
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

    def _create_event_handlers(self):
        super()._create_event_handlers()
        c = self.components
        c['cancel_button'].click(lambda: self.cancel_event.set(), [], [])
        c['clear_logs_button'].click(lambda: (self.all_logs.clear(), "")[1], [], c['unified_log'])
        c['log_level_filter'].change(lambda level: (setattr(self, 'log_filter_level', level), "\n".join([l for l in self.all_logs if level.upper() == "DEBUG" or f"[{level.upper()}]" in l][-1000:]))[1], c['log_level_filter'], c['unified_log'])

    def _create_pre_analysis_event(self, *args):
        ui_args = dict(zip(self.ana_ui_map_keys, args))
        strategy = ui_args.get('primary_seed_strategy', '🤖 Automatic')
        if strategy == "👤 By Face": ui_args.update({'enable_face_filter': True, 'text_prompt': ""})
        elif strategy == "📝 By Text": ui_args.update({'enable_face_filter': False, 'face_ref_img_path': "", 'face_ref_img_upload': None})
        elif strategy == "🔄 Face + Text Fallback": ui_args['enable_face_filter'] = True
        elif strategy == "🤖 Automatic": ui_args.update({'enable_face_filter': False, 'text_prompt': "", 'face_ref_img_path': "", 'face_ref_img_upload': None})
        for k, v_type, default in [('pre_sample_nth', int, 5), ('min_mask_area_pct', float, 0.0), ('sharpness_base_scale', float, 1.0), ('edge_strength_base_scale', float, 1.0)]:
            try: ui_args[k] = v_type(ui_args.get(k)) if v_type != int or int(ui_args.get(k)) > 0 else default
            except (TypeError, ValueError): ui_args[k] = default
        return PreAnalysisEvent(**ui_args)

    def run_extraction_wrapper(self, *args, tracker):
        tracker.start_operation("Extraction", 1)
        event = ExtractionEvent(**dict(zip(self.ext_ui_map_keys, args)))
        try:
            for result in execute_extraction(event, self.progress_queue, self.cancel_event, self.enhanced_logger, self.config, tracker):
                if isinstance(result, dict):
                    if self.cancel_event.is_set():
                        tracker.complete_operation(success=False, message="User cancelled")
                        return {"unified_log": "Extraction cancelled."}
                    if result.get("done"):
                        tracker.complete_operation(success=True)
                        return {"unified_log": result.get("log", "✅ Extraction completed successfully."),
                                "extracted_video_path_state": result.get("video_path", "") or result.get("extracted_video_path_state", ""),
                                "extracted_frames_dir_state": result.get("output_dir", "") or result.get("extracted_frames_dir_state", "")}
            tracker.complete_operation(success=False, message="Extraction failed.")
            return {"unified_log": "❌ Extraction failed."}
        except Exception as e: tracker.complete_operation(success=False, message=str(e)); raise

    def run_pre_analysis_wrapper(self, *args, tracker):
        tracker.start_operation("Pre-Analysis", 1)
        event = self._create_pre_analysis_event(*args)
        try:
            for result in execute_pre_analysis(event, self.progress_queue, self.cancel_event, self.enhanced_logger, self.config, self.thumbnail_manager, self.cuda_available, tracker):
                if isinstance(result, dict):
                    if self.cancel_event.is_set():
                        tracker.complete_operation(success=False, message="User cancelled")
                        return {"unified_log": "Pre-analysis cancelled."}
                    if result.get("done"):
                        scenes = result.get('scenes', [])
                        if scenes: save_scene_seeds(scenes, result['output_dir'], self.enhanced_logger)
                        tracker.complete_operation(success=True)
                        updates = {"unified_log": result.get("log", "✅ Pre-analysis completed successfully."), "seeding_preview_gallery": gr.update(value=result.get('previews')),
                                   "scenes_state": scenes, "propagate_masks_button": gr.update(interactive=True), "scene_filter_status": get_scene_status_text(scenes),
                                   "scene_face_sim_min_input": gr.update(visible=any(s.get('seed_metrics', {}).get('best_face_sim') is not None for s in scenes)),
                                   "seeding_results_column": gr.update(visible=True), "propagation_group": gr.update(visible=True)}
                        if result.get("final_face_ref_path"):
                            updates["face_ref_img_path_input"] = result["final_face_ref_path"]
                        return updates
            tracker.complete_operation(success=False, message="Pre-analysis failed.")
            return {"unified_log": "❌ Pre-analysis failed."}
        except Exception as e: tracker.complete_operation(success=False, message=str(e)); raise

    def run_propagation_wrapper(self, scenes, *args, tracker):
        tracker.start_operation("Propagation", 1)
        event = PropagationEvent(output_folder=self._create_pre_analysis_event(*args).output_folder, video_path=self._create_pre_analysis_event(*args).video_path,
                                 scenes=scenes, analysis_params=self._create_pre_analysis_event(*args))
        try:
            for result in execute_propagation(event, self.progress_queue, self.cancel_event, self.enhanced_logger, self.config, self.thumbnail_manager, self.cuda_available, tracker):
                if isinstance(result, dict):
                    if self.cancel_event.is_set():
                        tracker.complete_operation(success=False, message="User cancelled")
                        return {"unified_log": "Propagation cancelled."}
                    if result.get("done"):
                        tracker.complete_operation(success=True)
                        return {"unified_log": result.get("log", "✅ Propagation completed successfully."), "analysis_output_dir_state": result.get('output_dir', ""),
                                "analysis_metadata_path_state": result.get('metadata_path', ""), "filtering_tab": gr.update(interactive=True)}
            tracker.complete_operation(success=False, message="Propagation failed.")
            return {"unified_log": "❌ Propagation failed."}
        except Exception as e: tracker.complete_operation(success=False, message=str(e)); raise

    def run_session_load_wrapper(self, session_path, tracker):
        tracker.start_operation("Session Load", 1)
        try:
            final_result = {}
            for result in execute_session_load(SessionLoadEvent(session_path=session_path), self.enhanced_logger, self.config, self.thumbnail_manager, tracker):
                if isinstance(result, dict):
                    if 'log' in result: result['unified_log'] = result.pop('log')
                    final_result.update(result)
            tracker.complete_operation(success=True)
            return final_result
        except Exception as e: tracker.complete_operation(success=False, message=str(e)); raise

    def _setup_visibility_toggles(self):
        c = self.components
        c['method_input'].change(lambda m: (gr.update(visible=m == 'interval'), gr.update(visible=m == 'scene'), gr.update(visible=m == 'every_nth_frame')),
                                 c['method_input'], [c['interval_input'], c['fast_scene_input'], c['nth_frame_input']])
        c['thumbnails_only_input'].change(lambda r: {c['recommended_accordion']: gr.update(visible=r), c['legacy_accordion']: gr.update(visible=not r)},
                                          [c['thumbnails_only_input']], [c['recommended_accordion'], c['legacy_accordion']])
        c['primary_seed_strategy_input'].change(lambda s: {c['face_seeding_group']: gr.update(visible=s == "👤 By Face" or s == "🔄 Face + Text Fallback"),
                                                           c['text_seeding_group']: gr.update(visible=s == "📝 By Text" or s == "🔄 Face + Text Fallback"),
                                                           c['auto_seeding_group']: gr.update(visible=s == "🤖 Automatic"),
                                                           c['enable_face_filter_input']: gr.update(value=s == "👤 By Face" or s == "🔄 Face + Text Fallback")},
                                                 [c['primary_seed_strategy_input']], [c['face_seeding_group'], c['text_seeding_group'],
                                                                                     c['auto_seeding_group'], c['enable_face_filter_input']])

    def _setup_pipeline_handlers(self):
        c = self.components
        all_outputs = [v for v in c.values() if hasattr(v, "_id")]
        session_outputs = [c[k] for k in self.session_load_keys if k != 'progress_bar' and k in c and hasattr(c[k], "_id")]
        def session_load_handler(session_path, progress=gr.Progress()):
            # Remove 'progress_bar' from keys if it exists, as it causes errors with new Gradio versions
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
                                                           'resume': gr.State(False), 'enable_face_filter': 'enable_face_filter_input',
                                                           'face_ref_img_path': 'face_ref_img_path_input', 'face_ref_img_upload': 'face_ref_img_upload_input',
                                                           'face_model_name': 'face_model_name_input', 'enable_subject_mask': gr.State(True),
                                                           'dam4sam_model_name': 'dam4sam_model_name_input', 'person_detector_model': 'person_detector_model_input',
                                                           'seed_strategy': 'seed_strategy_input', 'scene_detect': 'ext_scene_detect_input',
                                                           'enable_dedup': 'enable_dedup_input', 'text_prompt': 'text_prompt_input',
                                                           'box_threshold': 'scene_editor_box_thresh_input', 'text_threshold': 'scene_editor_text_thresh_input',
                                                           'min_mask_area_pct': gr.State(self.config.min_mask_area_pct),
                                                           'sharpness_base_scale': gr.State(self.config.sharpness_base_scale),
                                                           'edge_strength_base_scale': gr.State(self.config.edge_strength_base_scale),
                                                           'gdino_config_path': gr.State(str(self.config.GROUNDING_DINO_CONFIG)),
                                                           'gdino_checkpoint_path': gr.State(str(self.config.GROUNDING_DINO_CKPT)),
                                                           'pre_analysis_enabled': 'pre_analysis_enabled_input', 'pre_sample_nth': 'pre_sample_nth_input',
                                                           'primary_seed_strategy': 'primary_seed_strategy_input'}[k] for k in self.ana_ui_map_keys]]
        prop_inputs = [c['scenes_state']] + self.ana_input_components
        c['start_extraction_button'].click(fn=extraction_handler,
                                         inputs=ext_inputs, outputs=all_outputs, show_progress="hidden").then(lambda d: gr.update(selected=1) if d else gr.update(), c['extracted_frames_dir_state'], c['main_tabs'])
        c['start_pre_analysis_button'].click(fn=pre_analysis_handler,
                                           inputs=self.ana_input_components, outputs=all_outputs, show_progress="hidden")
        c['propagate_masks_button'].click(fn=propagation_handler,
                                        inputs=prop_inputs, outputs=all_outputs, show_progress="hidden").then(lambda p: gr.update(selected=2) if p else gr.update(), c['analysis_metadata_path_state'], c['main_tabs'])

    def _setup_scene_editor_handlers(self):
        c = self.components
        def on_select_scene(scenes, evt: gr.SelectData):
            if not scenes or evt.index is None:
                return (gr.update(open=False), None, "",
                       self.config.grounding_dino_params['box_threshold'],
                       self.config.grounding_dino_params['text_threshold'])
            scene = scenes[evt.index]
            cfg = scene.get('seed_config', {})
            status_md = (f"**Editing Scene {scene['shot_id']}** "
                        f"(Frames {scene['start_frame']}-{scene['end_frame']})")
            prompt = cfg.get('text_prompt', '') if cfg else ''

            return (gr.update(open=True, value=status_md), scene['shot_id'],
                   prompt,
                   cfg.get('box_threshold', self.config.grounding_dino_params['box_threshold']),
                   cfg.get('text_threshold', self.config.grounding_dino_params['text_threshold']))

        c['seeding_preview_gallery'].select(
            on_select_scene, [c['scenes_state']],
            [c['scene_editor_accordion'], c['selected_scene_id_state'],
             c['scene_editor_prompt_input'],
             c['scene_editor_box_thresh_input'],
             c['scene_editor_text_thresh_input']]
        )
        recompute_inputs = [c['scenes_state'], c['selected_scene_id_state'], c['scene_editor_prompt_input'],
                            c['scene_editor_box_thresh_input'], c['scene_editor_text_thresh_input'], c['extracted_frames_dir_state']] + self.ana_input_components
        c['scene_recompute_button'].click(self.on_apply_scene_overrides, recompute_inputs, [c['seeding_preview_gallery'], c['scenes_state'], c['unified_status']])
        include_exclude_inputs, include_exclude_outputs = [c['scenes_state'], c['selected_scene_id_state'], c['extracted_frames_dir_state']], [c['scenes_state'], c['scene_filter_status'], c['unified_status']]
        c['scene_include_button'].click(self.on_toggle_scene_status, include_exclude_inputs + [gr.State('included')], include_exclude_outputs)
        c['scene_exclude_button'].click(self.on_toggle_scene_status, include_exclude_inputs + [gr.State('excluded')], include_exclude_outputs)

    def _setup_bulk_scene_handlers(self):
        c = self.components
        def bulk_toggle(s, status, folder):
            if not s: return [], "No scenes to update."
            for scene in s: scene['status'], scene['manual_status_change'] = status, True
            save_scene_seeds(s, folder, self.logger)
            return s, get_scene_status_text(s)
        c['bulk_include_all_button'].click(lambda s, f: bulk_toggle(s, 'included', f), [c['scenes_state'], c['extracted_frames_dir_state']], [c['scenes_state'], c['scene_filter_status']])
        c['bulk_exclude_all_button'].click(lambda s, f: bulk_toggle(s, 'excluded', f), [c['scenes_state'], c['extracted_frames_dir_state']], [c['scenes_state'], c['scene_filter_status']])
        bulk_filter_inputs = [c['scenes_state'], c['scene_mask_area_min_input'], c['scene_face_sim_min_input'],
                              c['scene_confidence_min_input'], c['enable_face_filter_input'], c['extracted_frames_dir_state']]
        bulk_filter_outputs = [c['scenes_state'], c['scene_filter_status']]
        c['scene_mask_area_min_input'].release(self.on_apply_bulk_scene_filters, bulk_filter_inputs, bulk_filter_outputs)
        for comp in [c['scene_face_sim_min_input'], c['scene_confidence_min_input']]:
            comp.release(self.on_apply_bulk_scene_filters, bulk_filter_inputs, bulk_filter_outputs)

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
                show_overlay=True,
                overlay_alpha=0.6,
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
            fn_to_orig_map.update({f"frame_{i+1:06d}.webp": orig for i, orig in enumerate(sorted(frame_map_list))})
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
                crop_dir = export_dir / "cropped"; crop_dir.mkdir(exist_ok=True)
                try: aspect_ratios = [tuple(map(int, ar.strip().split(':'))) for ar in event.crop_ars.split(',') if ar.strip()]
                except Exception: return "Invalid aspect ratio format. Use 'width:height,width:height' e.g. '16:9,1:1'."
                masks_root, num_cropped = out_root / "masks", 0
                for frame_meta in kept:
                    if self.cancel_event.is_set(): break
                    try:
                        if not (full_frame_path := export_dir / frame_meta['filename']).exists(): continue
                        if not (mask_path := masks_root / frame_meta.get('mask_path', '')).exists(): continue
                        frame_img, mask_img = cv2.imread(str(full_frame_path)), cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                        if frame_img is None or mask_img is None: continue
                        contours, _ = cv2.findContours(mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        if not contours: continue
                        x, y, w, h = cv2.boundingRect(np.concatenate(contours))
                        frame_h, frame_w, mask_h, mask_w = *frame_img.shape[:2], *mask_img.shape[:2]
                        padding_px_w, padding_px_h = int(w * (event.crop_padding / 100.0)), int(h * (event.crop_padding / 100.0))
                        x1, y1, x2, y2 = max(0, x - padding_px_w), max(0, y - padding_px_h), min(frame_w, x + w + padding_px_w), min(frame_h, y + h + padding_px_h)
                        w_pad, h_pad = x2 - x1, y2 - y1
                        if w_pad <= 0 or h_pad <= 0: continue
                        center_x, center_y = x1 + w_pad // 2, y1 + h_pad // 2

                        def expand_to_ar(target_ar_val):
                            if w_pad / (h_pad + 1e-9) < target_ar_val: new_w, new_h = int(np.ceil(h_pad * target_ar_val)), h_pad
                            else: new_w, new_h = w_pad, int(np.ceil(w_pad / target_ar_val))
                            if new_w > frame_w or new_h > frame_h: return None # Crop would be larger than frame
                            nx1, ny1 = int(round(center_x - new_w / 2)), int(round(center_y - new_h / 2))
                            if nx1 < 0: nx1 = 0
                            if ny1 < 0: ny1 = 0
                            if nx1 + new_w > frame_w: nx1 = frame_w - new_w
                            if ny1 + new_h > frame_h: ny1 = frame_h - new_h
                            nx2, ny2 = nx1 + new_w, ny1 + new_h
                            # Ensure the original subject is still contained
                            if nx1 > x1 or ny1 > y1 or nx2 < x2 or ny2 < y2: return None
                            return (nx1, ny1, nx2, ny2, (new_w * new_h) / max(1, w_pad * h_pad))

                        candidates = []
                        for ar_w, ar_h in aspect_ratios:
                            if ar_h > 0:
                                res = expand_to_ar(ar_w / ar_h)
                                if res: candidates.append(res + (f"{ar_w}x{ar_h}",))

                        if candidates:
                            nx1_final, ny1_final, nx2_final, ny2_final, _, ar_str = sorted(candidates, key=lambda t: t[4])[0] # Get the one with the tightest fit
                            cropped_img = frame_img[ny1_final:ny2_final, nx1_final:nx2_final]
                            cv2.imwrite(str(crop_dir / f"{Path(frame_meta['filename']).stem}_crop_{ar_str}.png"), cropped_img)
                            num_cropped += 1
                        else: # Fallback to just cropping the padded bounding box if no AR fits
                            cropped_img = frame_img[y1:y2, x1:x2]
                        if cropped_img.size > 0:
                            cv2.imwrite(str(crop_dir / f"{Path(frame_meta['filename']).stem}_crop_native.png"), cropped_img)
                            num_cropped += 1
                    except Exception as e: self.logger.error(f"Failed to crop frame {frame_meta['filename']}", exc_info=True)
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
        self.config = Config()
        self.logger = EnhancedLogger(log_dir=self.config.DIRS['logs'], enable_performance_monitoring=True)
        self.thumbnail_manager = ThumbnailManager(logger=self.logger, max_size=self.config.thumbnail_cache_size)
        self.resource_manager = AdaptiveResourceManager(logger=self.logger, config=self.config)
        self.resource_manager.start_monitoring()
        self.progress_queue = Queue()
        self.cancel_event = threading.Event()
        self.logger.set_progress_queue(self.progress_queue)
        self.progress_tracker = AdvancedProgressTracker(self.progress_queue, self.logger)
        self._app_ui = None

    def get_app_ui(self):
        if self._app_ui is None:
            self._app_ui = EnhancedAppUI(config=self.config, logger=self.logger, progress_queue=self.progress_queue,
                                         cancel_event=self.cancel_event, thumbnail_manager=self.get_thumbnail_manager(),
                                         resource_manager=self.get_resource_manager(), progress_tracker=self.progress_tracker)
        return self._app_ui

    def get_config(self): return self.config
    def get_logger(self): return self.logger
    def get_thumbnail_manager(self): return self.thumbnail_manager
    def get_resource_manager(self): return self.resource_manager
    def cleanup(self):
        if hasattr(self.thumbnail_manager, 'cleanup'): self.thumbnail_manager.cleanup()
        self.resource_manager.stop_monitoring()
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