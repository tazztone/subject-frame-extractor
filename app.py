# keep app.py Monolithic!
"""
Frame Extractor & Analyzer v2.0
"""
import contextlib
import cv2
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
from torchvision.ops import box_convert
from skimage.metrics import structural_similarity as ssim
import lpips
from torchvision import transforms
import urllib.request

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
sys.path.insert(0, str(project_root / 'DAM4SAM'))

from collections import Counter, OrderedDict, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from database import Database
from enum import Enum
from functools import lru_cache
from pydantic import BaseModel, Field, model_validator, field_validator, ConfigDict
from pydantic_settings import BaseSettings, SettingsConfigDict
from queue import Empty, Queue
from typing import Any, Callable, Dict, Generator, List, Optional, Union

# --- DEPENDENCY IMPORTS (with error handling) ---

from DAM4SAM.dam4sam_tracker import DAM4SAMTracker
from DAM4SAM.utils import utils as dam_utils


from groundingdino.util.inference import (
        load_model as gdino_load_model,
        predict as gdino_predict
    )


import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from numba import njit

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


from PIL import Image

import pyiqa

from scenedetect import detect, ContentDetector

from sklearn.cluster import DBSCAN


import yt_dlp as ytdlp

from ultralytics import YOLO

# --- CONFIGURATION ---
def json_config_settings_source() -> Dict[str, Any]:
    """Loads settings from a JSON file for Pydantic settings.

    This function is a pydantic-settings custom source. It attempts to load
    configuration variables from a `config.json` file located in the
    application's root directory. If the file is not found or contains invalid
    JSON, it returns an empty dictionary, allowing other settings sources
    to take precedence.

    Returns:
        A dictionary of configuration settings loaded from the JSON file,
        or an empty dictionary if the file cannot be loaded.
    """
    try:
        config_path = "config.json" # Hardcoded as in the original logic
        if Path(config_path).is_file():
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        pass # Ignore if file not found or is invalid
    return {}
class Config(BaseSettings):
    """Manages the application's configuration settings.

    This class uses pydantic-settings to load configuration from environment
    variables, a .env file, and a JSON configuration file (config.json).
    It defines all tunable parameters for the application, from file paths
    and model URLs to UI defaults and quality metric weights.

    Attributes:
        model_config: Pydantic-settings configuration dictionary.
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix='APP_',
        env_nested_delimiter='_',
        case_sensitive=False
    )

    # Paths
    logs_dir: str = "logs"
    models_dir: str = "models"
    downloads_dir: str = "downloads"
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

    # Filter Defaults
    filter_default_quality_score: Dict[str, float] = Field(default_factory=lambda: {'min': 0.0, 'max': 100.0, 'step': 0.5, 'default_min': 0.0, 'default_max': 100.0})
    filter_default_sharpness: Dict[str, float] = Field(default_factory=lambda: {'min': 0.0, 'max': 100.0, 'step': 0.5, 'default_min': 0.0, 'default_max': 100.0})
    filter_default_edge_strength: Dict[str, float] = Field(default_factory=lambda: {'min': 0.0, 'max': 100.0, 'step': 0.5, 'default_min': 0.0, 'default_max': 100.0})
    filter_default_contrast: Dict[str, float] = Field(default_factory=lambda: {'min': 0.0, 'max': 100.0, 'step': 0.5, 'default_min': 0.0, 'default_max': 100.0})
    filter_default_brightness: Dict[str, float] = Field(default_factory=lambda: {'min': 0.0, 'max': 100.0, 'step': 0.5, 'default_min': 0.0, 'default_max': 100.0})
    filter_default_entropy: Dict[str, float] = Field(default_factory=lambda: {'min': 0.0, 'max': 100.0, 'step': 0.5, 'default_min': 0.0, 'default_max': 100.0})
    filter_default_niqe: Dict[str, float] = Field(default_factory=lambda: {'min': 0.0, 'max': 100.0, 'step': 0.5, 'default_min': 0.0, 'default_max': 100.0})
    filter_default_face_sim: Dict[str, float] = Field(default_factory=lambda: {'min': 0.0, 'max': 1.0, 'step': 0.01, 'default_min': 0.0})
    filter_default_mask_area_pct: Dict[str, float] = Field(default_factory=lambda: {'min': 0.0, 'max': 100.0, 'step': 0.1, 'default_min': 1.0})
    filter_default_dedup_thresh: Dict[str, int] = Field(default_factory=lambda: {'min': -1, 'max': 32, 'step': 1, 'default': -1})
    filter_default_eyes_open: Dict[str, float] = Field(default_factory=lambda: {'min': 0.0, 'max': 1.0, 'step': 0.01, 'default_min': 0.0})
    filter_default_yaw: Dict[str, float] = Field(default_factory=lambda: {'min': -180.0, 'max': 180.0, 'step': 1, 'default_min': -25, 'default_max': 25})
    filter_default_pitch: Dict[str, float] = Field(default_factory=lambda: {'min': -180.0, 'max': 180.0, 'step': 1, 'default_min': -25, 'default_max': 25})

    # Quality Weights
    quality_weights_sharpness: int = 25
    quality_weights_edge_strength: int = 15
    quality_weights_contrast: int = 15
    quality_weights_brightness: int = 10
    quality_weights_entropy: int = 15
    quality_weights_niqe: int = 20

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
        """Post-initialization hook for Pydantic models.

        This method is called by Pydantic after the model has been initialized.
        It's used here to trigger path validation.

        Args:
            __context: The Pydantic context, not used here.
        """
        self._validate_paths()

    @model_validator(mode='after')
    def _validate_config(self) -> 'Config':
        """Validates the entire model after initialization.

        This validator checks that the sum of quality weights is non-zero to
        prevent division-by-zero errors in quality score calculations.

        Returns:
            The validated Config instance.

        Raises:
            ValueError: If the sum of all quality_weights is zero.
        """
        if sum(self.quality_weights.values()) == 0:
            raise ValueError("The sum of quality_weights cannot be zero.")
        return self

    @property
    def quality_weights(self) -> Dict[str, int]:
        """Returns a dictionary of all quality weight parameters.

        This property gathers all individual quality weight settings into a
        single dictionary for easier access during quality score computation.

        Returns:
            A dictionary mapping quality metric names to their integer weights.
        """
        return {
            'sharpness': self.quality_weights_sharpness,
            'edge_strength': self.quality_weights_edge_strength,
            'contrast': self.quality_weights_contrast,
            'brightness': self.quality_weights_brightness,
            'entropy': self.quality_weights_entropy,
            'niqe': self.quality_weights_niqe,
        }

    def _validate_paths(self):
        """Ensures required directories exist and are writable.

        This method checks for the existence of essential directories like
        logs, models, and downloads. If they don't exist, it creates them.
        It also verifies that the application has write permissions for these
        directories.

        Raises:
            PermissionError: If any of the required directories are not writable.
        """
        required_dirs = [self.logs_dir, self.models_dir, self.downloads_dir]
        for dir_path in required_dirs:
            path = Path(dir_path)
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)
            if not os.access(path, os.W_OK):
                raise PermissionError(f"No write permission for: {path}")

    def save_config(self, path: str):
        """Saves the current configuration to a JSON file.

        Args:
            path: The file path where the configuration will be saved.
        """
        with open(path, 'w', encoding='utf-8') as f:
            f.write(self.model_dump_json(indent=2))

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ):
        """Customizes the settings sources for pydantic-settings.

        This class method injects the `json_config_settings_source` into the
        list of sources that pydantic-settings uses to load configuration,
        allowing the application to be configured via a `config.json` file.

        Args:
            settings_cls: The settings class.
            init_settings: The initial settings.
            env_settings: The environment variable settings.
            dotenv_settings: The .env file settings.
            file_secret_settings: The file-based secret settings.

        Returns:
            A tuple of settings sources in the desired order of precedence.
        """
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            json_config_settings_source,
            file_secret_settings,
        )


# --- LOGGING ---

SUCCESS_LEVEL_NUM = 25
logging.addLevelName(SUCCESS_LEVEL_NUM, "SUCCESS")

class LogEvent(BaseModel):
    """Represents a structured log entry.

    Attributes:
        timestamp: The ISO 8601 timestamp of the event.
        level: The log level (e.g., "INFO", "ERROR").
        message: The log message.
        component: The application component that generated the log.
        operation: The high-level operation during which the event occurred.
        duration_ms: The duration of the operation in milliseconds.
        error_type: The type of error, if any.
        stack_trace: The stack trace, if an error occurred.
        user_context: A dictionary of user-provided context.
        performance_metrics: A dictionary of performance metrics.
        custom_fields: A dictionary for any other custom data.
    """
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
    """A comprehensive logger for the application.

    This logger handles both colored console logging and structured JSONL file
    logging. It also integrates with a progress queue to send log messages
    to the Gradio UI.
    """
    def __init__(self, config: 'Config', log_dir: Optional[Path] = None,
                 log_to_file: bool = True,
                 log_to_console: bool = True):
        """Initializes the AppLogger.

        Args:
            config (Config): The application's configuration object.
            log_dir (Optional[Path]): The directory to store log files in.
                Defaults to the path specified in the config.
            log_to_file (bool): Whether to enable file logging.
            log_to_console (bool): Whether to enable console logging.
        """
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
        """Sets up the console handler with colored output."""
        console_handler = logging.StreamHandler()
        console_formatter = ColoredFormatter(self.config.log_format)
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(self.config.log_level)
        self.logger.addHandler(console_handler)

    def _setup_file_handlers(self):
        """Sets up file handlers for both plain text and structured logs."""
        # Handler for plain text session log
        file_handler = logging.FileHandler(self.session_log_file, encoding='utf-8')
        file_formatter = logging.Formatter(self.config.log_format)
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.DEBUG)
        self.logger.addHandler(file_handler)

        # Handler for structured JSONL log
        structured_handler = logging.FileHandler(self.structured_log_file, encoding='utf-8')
        structured_formatter = JsonFormatter()
        structured_handler.setFormatter(structured_formatter)
        structured_handler.setLevel(logging.DEBUG)
        self.logger.addHandler(structured_handler)

    def set_progress_queue(self, queue: Queue):
        """Sets the queue for sending progress and log updates to the UI.

        Args:
            queue (Queue): The queue instance to use.
        """
        self.progress_queue = queue

    @contextlib.contextmanager
    def operation(self, name: str, component: str = "system", tracker: Optional['AdvancedProgressTracker'] = None):
        """A context manager for logging the start, end, and duration of an operation.

        This provides a convenient way to wrap a block of code, automatically
        logging its start and end times, duration, and any exceptions that
        occur. It can also integrate with an `AdvancedProgressTracker`.

        Args:
            name (str): The name of the operation (e.g., "Frame Extraction").
            component (str): The application component performing the operation.
            tracker (Optional[AdvancedProgressTracker]): An optional tracker
                to update with the operation's status.

        Yields:
            None
        """
        t0 = time.time()
        if tracker:
            tracker.set_stage(name)
        self.info(f"Start {name}", component=component)
        try:
            yield
        except Exception as e:
            if tracker:
                # Freeze progress and show failure state
                tracker.set_stage(f"{name}: Failed", substage=str(e))
            self.error(f"Failed {name}", component=component, stack_trace=traceback.format_exc())
            raise
        finally:
            duration = (time.time() - t0) * 1000
            if tracker and not getattr(self, "cancel_event", threading.Event()).is_set():
                tracker.done_stage(f"{name} complete")
            self.success(f"Done {name} in {duration:.0f}ms", component=component)

    def _create_log_event(self, level: str, message: str, component: str, **kwargs) -> LogEvent:
        """Creates a LogEvent object from log data.

        This internal helper function constructs a structured `LogEvent` from
        the provided arguments, preparing it for serialization.

        Args:
            level (str): The log level (e.g., "INFO").
            message (str): The main log message.
            component (str): The component generating the log.
            **kwargs: Additional fields for the `LogEvent`, such as
                `stack_trace` or `custom_fields`.

        Returns:
            A populated `LogEvent` object.
        """
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
                        **kwargs)

    def _log_event(self, event: LogEvent):
        """Handles the actual logging of a LogEvent.

        This method is the final step in the logging process. It writes the
        event to the plain text log and the structured JSONL log, and sends a
        formatted message to the UI progress queue if it's available.

        Args:
            event (LogEvent): The LogEvent to process.
        """
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

        # Pass the LogEvent object to the logger using the 'extra' kwarg.
        # The custom JsonFormatter will know how to handle it.
        self.logger.log(log_level, log_message, extra={'log_event': event})

        if self.progress_queue:
            ui_message = f"[{event.level}] {event.message}"
            if event.operation: ui_message = f"[{event.operation}] {ui_message}"
            self.progress_queue.put({"log": ui_message})

    def debug(self, message: str, component: str = "system", **kwargs):
        """Logs a message with level DEBUG."""
        self._log_event(self._create_log_event("DEBUG", message, component, **kwargs))
    def info(self, message: str, component: str = "system", **kwargs):
        """Logs a message with level INFO."""
        self._log_event(self._create_log_event("INFO", message, component, **kwargs))
    def warning(self, message: str, component: str = "system", **kwargs):
        """Logs a message with level WARNING."""
        self._log_event(self._create_log_event("WARNING", message, component, **kwargs))
    def error(self, message: str, component: str = "system", **kwargs):
        """Logs a message with level ERROR."""
        self._log_event(self._create_log_event("ERROR", message, component, **kwargs))
    def success(self, message: str, component: str = "system", **kwargs):
        """Logs a message with custom level SUCCESS."""
        self._log_event(self._create_log_event("SUCCESS", message, component, **kwargs))
    def critical(self, message: str, component: str = "system", **kwargs):
        """Logs a message with level CRITICAL."""
        self._log_event(self._create_log_event("CRITICAL", message, component, **kwargs))

class ColoredFormatter(logging.Formatter):
    """A logging formatter that adds ANSI color codes to log levels."""
    COLORS = {'DEBUG': '\033[36m', 'INFO': '\033[37m', 'WARNING': '\033[33m',
              'ERROR': '\033[31m', 'CRITICAL': '\033[35m', 'SUCCESS': '\033[32m', 'RESET': '\033[0m'}

    def format(self, record: logging.LogRecord) -> str:
        """Formats the log record with color.

        Args:
            record (logging.LogRecord): The log record to format.

        Returns:
            str: The formatted log message string with ANSI color codes.
        """
        original_levelname = record.levelname
        try:
            color = self.COLORS.get(original_levelname, self.COLORS['RESET'])
            record.levelname = f"{color}{original_levelname}{self.COLORS['RESET']}"
            return super().format(record)
        finally:
            record.levelname = original_levelname

class JsonFormatter(logging.Formatter):
    """Formats log records as a JSON string for structured logging."""
    def format(self, record: logging.LogRecord) -> str:
        """Formats a log record into a JSON string.

        If the record has a `log_event` attribute that is a `LogEvent` instance,
        it will be serialized directly. Otherwise, a standard JSON log entry
        is created from the basic record attributes.

        Args:
            record (logging.LogRecord): The log record to format.

        Returns:
            str: A JSON string representation of the log record.
        """
        log_event_obj = getattr(record, 'log_event', None)
        if isinstance(log_event_obj, LogEvent):
            # If a LogEvent is passed directly, use its model_dump
            log_dict = log_event_obj.model_dump(exclude_none=True)
        else:
            # Fallback for standard log records
            log_dict = {
                'timestamp': self.formatTime(record, self.datefmt),
                'level': record.levelname,
                'message': record.getMessage(),
                'component': record.name,
            }
            if record.exc_info:
                log_dict['stack_trace'] = self.formatException(record.exc_info)

        return json.dumps(log_dict, default=str, ensure_ascii=False)

class AdvancedProgressTracker:
    """Manages and displays progress for long-running tasks in the UI."""
    def __init__(self, progress: Callable, queue: Queue, logger: AppLogger, ui_stage_name: str = ""):
        """Initializes the AdvancedProgressTracker.

        This class provides a sophisticated progress tracking mechanism that
        calculates ETA, throttles UI updates, and can be paused/resumed.

        Args:
            progress (Callable): The Gradio `gr.Progress` object.
            queue (Queue): The queue for sending updates to the UI thread.
            logger (AppLogger): The application logger.
            ui_stage_name (str): The initial stage name to display in the UI.
        """
        self.progress = progress
        self.queue = queue
        self.logger = logger
        self.stage = ui_stage_name or "Working"
        self.substage: Optional[str] = None
        self.total = 1
        self.done = 0
        self._t0 = time.time()
        self._last_ts = self._t0
        self._ema_dt = None
        self._alpha = 0.2
        self._last_update_ts: float = 0.0
        self.throttle_interval: float = 0.1  # 10 Hz
        self.pause_event = threading.Event()
        self.pause_event.set()

    def start(self, total_items: int, desc: Optional[str] = None):
        """Starts or resets the progress tracker.

        Args:
            total_items (int): The total number of items to process.
            desc (Optional[str]): The description of the current stage.
        """
        self.total = max(1, int(total_items))
        self.done = 0
        if desc:
            self.stage = desc
        self.substage = None
        self._t0 = time.time()
        self._last_ts = self._t0
        self._ema_dt = None
        self._overlay(force=True)

    def step(self, n: int = 1, desc: Optional[str] = None, substage: Optional[str] = None):
        """Increments the progress by a given amount.

        This method updates the progress count and recalculates the ETA based
        on an exponential moving average of the time per item.

        Args:
            n (int): The number of items completed in this step. Defaults to 1.
            desc (Optional[str]): An optional new description for the stage.
            substage (Optional[str]): An optional description for the current
                sub-task.
        """
        self.pause_event.wait()
        now = time.time()
        dt = now - self._last_ts
        self._last_ts = now
        if dt > 0:
            if self._ema_dt is None:
                self._ema_dt = dt / max(1, n)
            else:
                self._ema_dt = self._alpha * (dt / max(1, n)) + (1 - self._alpha) * self._ema_dt
        self.done = min(self.total, self.done + n)
        if desc:
            self.stage = desc
        if substage is not None:
            self.substage = substage
        self._overlay()

    def set(self, done: int, desc: Optional[str] = None, substage: Optional[str] = None):
        """Sets the progress to a specific value.

        Args:
            done (int): The new number of completed items.
            desc (Optional[str]): An optional new description for the stage.
            substage (Optional[str]): An optional description for the current
                sub-task.
        """
        delta = max(0, done - self.done)
        if delta > 0:
            self.step(delta, desc=desc, substage=substage)

    def set_stage(self, stage: str, substage: Optional[str] = None):
        """Updates the current stage and substage descriptions.

        This is useful for providing high-level context to the user about
        what part of a multi-step process is currently running.

        Args:
            stage (str): The new description for the main stage.
            substage (Optional[str]): The new description for the sub-task.
        """
        self.stage = stage
        self.substage = substage
        self._overlay(force=True)

    def done_stage(self, final_text: Optional[str] = None):
        """Marks the current stage as complete.

        This sets the progress to 100% and optionally logs a completion message.

        Args:
            final_text (Optional[str]): An optional message to log upon
                completion.
        """
        self.done = self.total
        self._overlay(force=True)
        if final_text:
            self.logger.info(final_text, component="progress")

    def _overlay(self, force: bool = False):
        """Updates the Gradio progress UI and sends a progress event to the queue.

        This method is throttled to prevent overwhelming the UI with too many
        updates in a short period.

        Args:
            force (bool): If True, bypasses the throttling mechanism.
        """
        now = time.time()
        fraction = self.done / max(1, self.total)
        if not force and (now - self._last_update_ts < self.throttle_interval):
            return
        self._last_update_ts = now

        eta_s = self._eta_seconds()
        eta_str = self._fmt_eta(eta_s)

        desc_parts = [f"{self.stage} ({self.done}/{self.total})"]
        if self.substage:
            desc_parts.append(self.substage)
        desc_parts.append(f"ETA {eta_str}")
        gradio_desc = " • ".join(desc_parts)

        if self.progress:
            self.progress(fraction, desc=gradio_desc)

        progress_event = ProgressEvent(
            stage=self.stage,
            substage=self.substage,
            done=self.done,
            total=self.total,
            fraction=fraction,
            eta_seconds=eta_s,
            eta_formatted=eta_str
        )
        self.queue.put({"progress": progress_event.model_dump()})

    def _eta_seconds(self) -> Optional[float]:
        """Calculates the estimated time remaining in seconds.

        Returns:
            Optional[float]: The estimated seconds remaining, or None if not
            enough data is available to make an estimate.
        """
        if self._ema_dt is None:
            return None
        remaining = max(0, self.total - self.done)
        return self._ema_dt * remaining

    @staticmethod
    def _fmt_eta(eta_s: Optional[float]) -> str:
        """Formats ETA in seconds into a human-readable string (e.g., "1m 30s").

        Args:
            eta_s (Optional[float]): The ETA in seconds.

        Returns:
            str: A formatted string representation of the ETA.
        """
        if eta_s is None:
            return "—"
        if eta_s < 60:
            return f"{int(eta_s)}s"
        m, s = divmod(int(eta_s), 60)
        if m < 60:
            return f"{m}m {s}s"
        h, m = divmod(m, 60)
        return f"{h}h {m}m"

# --- ERROR HANDLING ---

class ErrorSeverity(Enum):
    """Enumeration for the severity of an error."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class RecoveryStrategy(Enum):
    """Enumeration for possible error recovery strategies."""
    RETRY = "retry"
    FALLBACK = "fallback"
    SKIP = "skip"
    ABORT = "abort"

class ErrorHandler:
    """Provides decorators for implementing error handling strategies."""
    def __init__(self, logger: 'AppLogger', max_attempts: int, backoff_seconds: list):
        """Initializes the ErrorHandler.

        Args:
            logger (AppLogger): The application logger.
            max_attempts (int): The default maximum number of retry attempts.
            backoff_seconds (list): The default list of backoff delays for
                retries (e.g., [1, 5, 15]).
        """
        self.logger = logger
        self.max_attempts = max_attempts
        self.backoff_seconds = backoff_seconds
        self.error_count = 0
        self.recovery_attempts = {}

    def with_retry(self, max_attempts: Optional[int] = None, backoff_seconds: Optional[list] = None, recoverable_exceptions: tuple = (Exception,)):
        """A decorator that retries a function upon failure.

        This decorator wraps a function and automatically re-executes it if it
        raises one of the specified `recoverable_exceptions`. It implements an
        exponential backoff strategy using the provided delays.

        Args:
            max_attempts (Optional[int]): The maximum number of attempts.
                Defaults to the instance's default.
            backoff_seconds (Optional[list]): A list of delays between retries.
                Defaults to the instance's default.
            recoverable_exceptions (tuple): A tuple of exception types that
                should trigger a retry.

        Returns:
            Callable: The decorated function.
        """
        max_attempts = max_attempts or self.max_attempts
        backoff_seconds = backoff_seconds or self.backoff_seconds
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
        """A decorator that executes a fallback function if the primary function fails.

        If the decorated function raises any exception, this decorator will
        catch it, log it, and then execute the `fallback_func` with the same
        arguments. If the fallback also fails, the exception is re-raised.

        Args:
            fallback_func (Callable): The function to execute if the decorated
                function fails.

        Returns:
            Callable: The decorated function.
        """
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs) -> Any:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    self.logger.warning(
                        f"Primary function {func.__name__} failed, using fallback: {str(e)}",
                        component="error_handler",
                        error_type=type(e).__name__,
                        stack_trace=traceback.format_exc(),
                        custom_fields={'primary_function': func.__name__, 'fallback_function': fallback_func.__name__})
                    try:
                        return fallback_func(*args, **kwargs)
                    except Exception as fallback_error:
                        self.logger.error(
                            f"Both primary and fallback functions failed for {func.__name__}",
                            component="error_handler",
                            error_type=type(fallback_error).__name__,
                            stack_trace=traceback.format_exc(),
                            custom_fields={'primary_function': func.__name__, 'fallback_function': fallback_func.__name__,
                                           'primary_error': str(e), 'fallback_error': str(fallback_error)})
                        raise fallback_error
            return wrapper
        return decorator

# --- EVENTS ---

class ProgressEvent(BaseModel):
    """Represents a progress update for the UI."""
    stage: str
    substage: Optional[str] = None
    done: int = 0
    total: int = 1
    fraction: float = 0.0
    eta_seconds: Optional[float] = None
    eta_formatted: str = "—"

class UIEvent(BaseModel):
    """Base class for events originating from the UI."""
    model_config = ConfigDict(
        validate_assignment=True,
        extra='ignore',
        str_strip_whitespace=True,
        arbitrary_types_allowed=True
    )

class ExtractionEvent(UIEvent):
    """Data for an extraction pipeline run."""
    source_path: str
    upload_video: Optional[str] = None
    method: str
    interval: str
    nth_frame: str
    max_resolution: str
    thumbnails_only: bool
    thumb_megapixels: float
    scene_detect: bool
    output_folder: Optional[str] = None

class PreAnalysisEvent(UIEvent):
    """Data for a pre-analysis pipeline run."""
    output_folder: str
    video_path: str

    @field_validator('face_ref_img_path')
    @classmethod
    def validate_face_ref(cls, v: str, info) -> str:
        if not v:
            return ""

        # Get video_path from the validation context
        video_path = info.data.get('video_path', '')

        # Check if it's the video path itself
        if v == video_path:
            return ""

        # Check file existence and extension
        p = Path(v)
        valid_exts = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}
        if not p.is_file() or p.suffix.lower() not in valid_exts:
            return ""

        return v
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

    @model_validator(mode='after')
    def validate_strategy_consistency(self) -> 'PreAnalysisEvent':
        # Auto-disable face filter if no reference provided
        if not self.face_ref_img_path and self.enable_face_filter:
            self.enable_face_filter = False
        return self

class PropagationEvent(UIEvent):
    """Data for a mask propagation pipeline run."""
    output_folder: str
    video_path: str
    scenes: list[dict[str, Any]]
    analysis_params: PreAnalysisEvent

class FilterEvent(UIEvent):
    """Data for a filter change event."""
    all_frames_data: list[dict[str, Any]]
    per_metric_values: dict[str, Any]
    output_dir: str
    gallery_view: str
    show_overlay: bool
    overlay_alpha: float
    require_face_match: bool
    dedup_thresh: int
    slider_values: dict[str, float]
    dedup_method: str

class ExportEvent(UIEvent):
    """Data for an export event."""
    all_frames_data: list[dict[str, Any]]
    output_dir: str
    video_path: str
    enable_crop: bool
    crop_ars: str
    crop_padding: int
    filter_args: dict[str, Any]

class SessionLoadEvent(UIEvent):
    """Data for a session load event."""
    session_path: str

# --- UTILS ---

def handle_common_errors(func: Callable) -> Callable:
    """A decorator to catch and standardize common exceptions for the UI.

    This decorator wraps a pipeline function and catches common exceptions like
    `FileNotFoundError`, `ValueError`, `TypeError`, and `RuntimeError`. It
    formats them into a standardized dictionary that the UI can use to display
    a user-friendly error message, including a remediation hint. This prevents
    the application from crashing and provides helpful feedback to the user.

    Args:
        func (Callable): The function to wrap.

    Returns:
        Callable: A wrapped function that returns a standardized error
        dictionary on failure.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            # Attempt to execute the wrapped function
            return func(*args, **kwargs)
        except FileNotFoundError as e:
            return {
                "log": f"[ERROR] File not found: {e}. Please verify file paths and permissions.",
                "status_message": "A required file was not found.",
                "error_message": str(e),
                "remediation_hint": "Please check that the file path is correct and the file exists."
            }
        except (ValueError, TypeError) as e:
            return {
                "log": f"[ERROR] Invalid input provided: {e}. Check function arguments.",
                "status_message": "Invalid input detected.",
                "error_message": str(e),
                "remediation_hint": "Review the parameters you've provided. The type or value may be incorrect."
            }
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                return {
                    "log": "[ERROR] CUDA out of memory. The GPU does not have enough memory to complete the task.",
                    "status_message": "GPU memory error.",
                    "error_message": "CUDA out of memory.",
                    "remediation_hint": "Try reducing the model size, processing resolution, or batch size. Closing other GPU-intensive applications may also help."
                }
            return {
                "log": f"[ERROR] A runtime processing error occurred: {e}",
                "status_message": "A processing error occurred.",
                "error_message": str(e),
                "remediation_hint": "This may be a temporary issue. If it persists, please check the logs for more details."
            }
        except Exception as e:
            # Catch any other unexpected exceptions to prevent crashes
            return {
                "log": f"[CRITICAL] An unexpected error occurred: {e}\n{traceback.format_exc()}",
                "status_message": "An unexpected critical error occurred.",
                "error_message": str(e),
                "remediation_hint": "Please report this issue. Check the detailed logs for a stack trace."
            }
    return wrapper

def monitor_memory_usage(logger: 'AppLogger', device: str, threshold_mb: int = 8000):
    """Checks GPU memory usage and logs a warning if it exceeds a threshold.

    If the specified device is 'cuda', this function checks the allocated GPU
    memory. If it exceeds the `threshold_mb`, it logs a warning and attempts
    to free up cached memory.

    Args:
        logger (AppLogger): The application logger.
        device (str): The device to check memory usage on (e.g., 'cuda', 'cpu').
        threshold_mb (int): The memory usage threshold in megabytes.
    """
    if device == 'cuda':
        allocated = torch.cuda.memory_allocated() / 1024**2
        if allocated > threshold_mb:
            logger.warning(f"High GPU memory usage: {allocated:.1f}MB")
            torch.cuda.empty_cache()

def validate_video_file(video_path: str):
    """Performs basic validation on a video file.

    This function checks for the existence and non-zero size of the file. It
    also attempts to open the file with OpenCV to verify that it's a valid
    and readable video format.

    Args:
        video_path (str): The path to the video file.

    Raises:
        FileNotFoundError: If the video file does not exist.
        ValueError: If the file is empty or cannot be opened by OpenCV.
    """
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if path.stat().st_size == 0:
        raise ValueError(f"Video file is empty: {video_path}")

    # Quick format check
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        cap.release()
    except Exception as e:
        raise ValueError(f"Invalid video file: {e}")

def estimate_totals(params: 'AnalysisParameters', video_info: dict, scenes: Optional[list['Scene']]) -> dict:
    """Estimates the total number of items for each pipeline stage.

    This function provides estimations for the total number of frames or items
    that will be processed in each stage of the pipeline (extraction,
    pre-analysis, and propagation). These estimates are used to configure the
    total steps for the progress bar in the UI.

    Args:
        params (AnalysisParameters): The analysis parameters, which include
            the extraction method.
        video_info (dict): A dictionary with video metadata, including 'fps'
            and 'frame_count'.
        scenes (Optional[list[Scene]]): A list of detected scenes.

    Returns:
        A dictionary with estimated total counts for the 'extraction',
        'pre_analysis', and 'propagation' stages.
    """
    fps = max(1, int(video_info.get("fps") or 30))
    total_frames = int(video_info.get("frame_count") or 0)

    # Extraction totals
    method = params.method
    if method == "interval":
        extraction_total = max(1, int(total_frames / max(0.1, params.interval) / fps))
    elif method == "every_nth_frame":
        extraction_total = max(1, int(total_frames / max(1, params.nth_frame)))
    elif method == "all":
        extraction_total = total_frames
    elif method in ("keyframes", "nth_plus_keyframes"):
        extraction_total = max(1, int(total_frames * 0.15))  # keep heuristic
    else:
        extraction_total = total_frames

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

def sanitize_filename(name: str, config: 'Config', max_length: Optional[int] = None) -> str:
    """
    Sanitizes a string to be used as a valid filename.

    Replaces invalid characters with underscores and truncates to a maximum length.

    Args:
        name: The input string.
        config: The application configuration.
        max_length: The maximum allowed length of the filename.

    Returns:
        The sanitized filename string.
    """
    max_length = max_length or config.utility_max_filename_length
    return re.sub(r'[^\w\-_.]', '_', name)[:max_length]

def _to_json_safe(obj: Any) -> Any:
    """
    Recursively converts an object to a JSON-serializable format.

    Handles common non-serializable types like NumPy arrays and Path objects.

    Args:
        obj: The object to convert.

    Returns:
        A JSON-serializable representation of the object.
    """
    import numpy as np
    from pathlib import Path
    if isinstance(obj, dict):
        return {k: _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, BaseModel):
        return obj.model_dump()
    return obj

def _coerce(val: Any, to_type: type) -> Any:
    """Coerces a value to a specified type, with special handling for UI inputs.

    This function attempts to convert a given value to a target type. It has
    special logic for converting string representations of booleans (e.g.,
    "true", "1") and handles numeric types. It's used to safely convert raw
    input from the Gradio UI into the expected Python types for the backend.

    Args:
        val (Any): The value to coerce.
        to_type (type): The target type (e.g., `bool`, `int`, `float`).

    Returns:
        Any: The coerced value.

    Raises:
        ValueError: If a numeric string cannot be converted.
        TypeError: If the value cannot be coerced to the target type.
    """
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
def safe_resource_cleanup(device: str):
    """
    A context manager to perform garbage collection and clear CUDA cache.
    """
    try: yield
    finally:
        gc.collect()
        if device == 'cuda':
            torch.cuda.empty_cache()

def is_image_folder(p: Union[str, Path]) -> bool:
    """Checks if a given path is an existing directory.

    Args:
        p (Union[str, Path]): The path to check.

    Returns:
        bool: True if the path is a directory, False otherwise.
    """
    if not p:
        return False
    try:
        if not isinstance(p, (str, Path)):
            p = str(p)
        p = Path(p)
        return p.is_dir()
    except (TypeError, ValueError):
        return False

def list_images(p: Union[str, Path], cfg: Config) -> list[Path]:
    """Lists all image files in a given directory.

    Args:
        p (Union[str, Path]): The path to the directory.
        cfg (Config): The application configuration, which contains the list
            of valid image extensions.

    Returns:
        list[Path]: A sorted list of `Path` objects for the image files found
        in the directory.
    """
    p = Path(p)
    exts = {e.lower() for e in cfg.utility_image_extensions}
    return sorted([f for f in p.iterdir() if f.suffix.lower() in exts and f.is_file()])

# --- QUALITY ---

@njit
def compute_entropy(hist: np.ndarray, entropy_norm: float) -> float:
    """Computes the normalized Shannon entropy of an image from its histogram.

    This function is accelerated with Numba's Just-In-Time (JIT) compiler.

    Args:
        hist (np.ndarray): The histogram of the image's pixel intensities.
        entropy_norm (float): The normalization factor for the entropy value
            (e.g., 8.0 for an 8-bit grayscale image).

    Returns:
        float: The normalized entropy value, clamped between 0.0 and 1.0.
    """
    prob = hist / (np.sum(hist) + 1e-7)
    entropy = -np.sum(prob[prob > 0] * np.log2(prob[prob > 0]))
    return min(max(entropy / entropy_norm, 0), 1.0)

class QualityConfig(BaseModel):
    """Configuration for quality metric calculations.

    Attributes:
        sharpness_base_scale: The base scaling factor for sharpness calculation.
        edge_strength_base_scale: The base scaling factor for edge strength.
        enable_niqe: A flag to enable or disable NIQE metric calculation.
    """
    sharpness_base_scale: float
    edge_strength_base_scale: float
    enable_niqe: bool = True

# --- MODELS ---

class FrameMetrics(BaseModel):
    """A Pydantic model to store all calculated metrics for a single frame.

    Attributes:
        quality_score: The overall weighted quality score.
        sharpness_score: Score for image sharpness.
        edge_strength_score: Score for the strength of edges.
        contrast_score: Score for image contrast.
        brightness_score: Score for image brightness.
        entropy_score: Score for image information content (entropy).
        niqe_score: Score from the NIQE perceptual quality model.
        eyes_open: A score indicating how open the subject's eyes are.
        blink_prob: The probability of a blink being detected.
        yaw: The yaw (head rotation side-to-side) in degrees.
        pitch: The pitch (head rotation up-down) in degrees.
        roll: The roll (head tilt) in degrees.
    """
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

class Frame(BaseModel):
    """Represents a single video frame and its associated data.

    This class holds the image data for a frame, its frame number, and all
    associated metrics and scores. It provides a method to calculate these
    metrics.

    Attributes:
        image_data (np.ndarray): The raw image data as a NumPy array.
        frame_number (int): The original frame number from the video.
        metrics (FrameMetrics): A nested model containing all quality metrics.
        face_similarity_score (Optional[float]): The similarity score to the
            reference face, if calculated.
        max_face_confidence (Optional[float]): The confidence score of the
            best-detected face.
        error (Optional[str]): A string to hold any error message that occurred
            during processing of this frame.
        model_config (ConfigDict): Pydantic model configuration.
    """
    image_data: np.ndarray
    frame_number: int
    metrics: FrameMetrics = Field(default_factory=FrameMetrics)
    face_similarity_score: Optional[float] = None
    max_face_confidence: Optional[float] = None
    error: Optional[str] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def calculate_quality_metrics(self, thumb_image_rgb: np.ndarray, quality_config: 'QualityConfig', logger: 'AppLogger',
                                  mask: Optional[np.ndarray] = None, niqe_metric: Optional[Callable] = None,
                                  main_config: Optional['Config'] = None, face_landmarker: Optional[Callable] = None,
                                  face_bbox: Optional[List[int]] = None,
                                  metrics_to_compute: Optional[Dict[str, bool]] = None):
        """Calculates a suite of quality metrics for the frame.

        This method computes various metrics like sharpness, contrast,
        brightness, entropy, and NIQE score. It can also calculate face-specific
        metrics like eye-openness and head pose if a face landmarker is provided.
        Calculations can be restricted to a masked area of the image.

        Args:
            thumb_image_rgb (np.ndarray): The thumbnail image data in RGB format.
            quality_config (QualityConfig): Configuration for quality calculations.
            logger (AppLogger): The application logger.
            mask (Optional[np.ndarray]): An optional mask to apply to the image,
                restricting metric calculation to the masked region.
            niqe_metric (Optional[Callable]): The pre-initialized NIQE metric model.
            main_config (Optional[Config]): The main application configuration.
            face_landmarker (Optional[Callable]): The MediaPipe Face Landmarker
                instance for face metric calculations.
            face_bbox (Optional[List[int]]): The bounding box of the face to
                analyze for pose and eye metrics.
            metrics_to_compute (Optional[Dict[str, bool]]): A dictionary
                specifying which metrics to compute (e.g., `{'sharpness': True}`).
        """
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
                sharpness_scaled = (sharpness / (quality_config.sharpness_base_scale * (gray.size / main_config.quality_resolution_denominator)))
                _calculate_and_store_score("sharpness", sharpness_scaled)

            if metrics_to_compute.get('edge_strength'):
                sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                edge_strength = np.mean(np.sqrt(sobelx**2 + sobely**2))
                edge_strength_scaled = (edge_strength / (quality_config.edge_strength_base_scale * (gray.size / main_config.quality_resolution_denominator)))
                _calculate_and_store_score("edge_strength", edge_strength_scaled)

            if metrics_to_compute.get('contrast') or metrics_to_compute.get('brightness'):
                pixels = gray[active_mask] if active_mask is not None else gray
                mean_br, std_br = (np.mean(pixels), np.std(pixels)) if pixels.size > 0 else (0, 0)
                if metrics_to_compute.get('brightness'):
                    brightness = mean_br / 255.0
                    _calculate_and_store_score("brightness", brightness)
                if metrics_to_compute.get('contrast'):
                    contrast = std_br / (mean_br + 1e-7)
                    contrast_scaled = min(contrast, main_config.quality_contrast_clamp) / main_config.quality_contrast_clamp
                    _calculate_and_store_score("contrast", contrast_scaled)

            if metrics_to_compute.get('entropy'):
                gray_full = cv2.cvtColor(self.image_data, cv2.COLOR_RGB2GRAY)
                active_mask_full = None
                if mask is not None:
                    mask_full = cv2.resize(mask, (gray_full.shape[1], gray_full.shape[0]), interpolation=cv2.INTER_NEAREST)
                    active_mask_full = (mask_full > 128).astype(np.uint8)
                hist = cv2.calcHist([gray_full], [0], active_mask_full, [256], [0, 256]).flatten()
                entropy = compute_entropy(hist, main_config.quality_entropy_normalization)
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
                    with (torch.no_grad(), torch.amp.autocast('cuda', enabled=niqe_metric.device.type == 'cuda')):
                        niqe_raw = float(niqe_metric(img_tensor.to(niqe_metric.device)))
                        niqe_score = max(0, min(100, (main_config.quality_niqe_offset - niqe_raw) * main_config.quality_niqe_scale_factor))
                        scores_norm["niqe"] = niqe_score / 100.0
                        self.metrics.niqe_score = float(niqe_score)
                except Exception as e:
                    logger.warning("NIQE calculation failed", extra={'frame': self.frame_number, 'error': e})
                    if niqe_metric.device.type == 'cuda':
                        torch.cuda.empty_cache()

            if main_config and metrics_to_compute.get('quality'):
                weights = {
                    'sharpness': main_config.quality_weights_sharpness,
                    'edge_strength': main_config.quality_weights_edge_strength,
                    'contrast': main_config.quality_weights_contrast,
                    'brightness': main_config.quality_weights_brightness,
                    'entropy': main_config.quality_weights_entropy,
                    'niqe': main_config.quality_weights_niqe,
                }
                quality_sum = sum(
                    scores_norm.get(k, 0) * (weights.get(k, 0) / 100.0)
                    for k in scores_norm.keys()
                )
                self.metrics.quality_score = float(quality_sum * 100)
        except Exception as e:
            self.error = f"Quality calc failed: {e}"
            logger.error("Frame quality calculation failed", exc_info=True, extra={'frame': self.frame_number})

class Scene(BaseModel):
    """Represents a single continuous shot or scene in the video.

    This Pydantic model holds all data related to a single scene, including
    its frame boundaries, processing status, and the results of the seed
    selection process.

    Attributes:
        shot_id: A unique identifier for the scene.
        start_frame: The starting frame number of the scene.
        end_frame: The ending frame number of the scene.
        status: The current processing status (e.g., 'pending', 'included',
            'excluded').
        best_frame: The frame number identified as the best seed frame.
        seed_metrics: A dictionary of metrics related to the seed frame selection.
        seed_frame_idx: The index of the seed frame within the scene's frames.
        seed_config: A dictionary of configuration used for seed selection.
        seed_type: The type of seed selection strategy used.
        seed_result: A dictionary containing the results of the seed selection
            (e.g., the bounding box).
        preview_path: The file path to the preview image for this scene.
        manual_status_change: A flag indicating if the status was manually changed
            by the user.
        is_overridden: A flag indicating if the seed has been manually
            overridden by the user.
        initial_bbox: The initially detected bounding box for the subject.
        selected_bbox: The currently selected bounding box, which may have been
            manually changed.
        yolo_detections: A list of all YOLO detections for the best frame.
        rejection_reasons: A list of reasons why the scene was rejected, if applicable.
    """
    shot_id: int
    start_frame: int
    end_frame: int
    status: str = "pending"
    best_frame: Optional[int] = None
    seed_metrics: dict = Field(default_factory=dict)
    seed_frame_idx: Optional[int] = None
    seed_config: dict = Field(default_factory=dict)
    seed_type: Optional[str] = None
    seed_result: dict = Field(default_factory=dict)
    preview_path: Optional[str] = None
    manual_status_change: bool = False
    is_overridden: bool = False
    initial_bbox: Optional[list] = None
    selected_bbox: Optional[list] = None
    yolo_detections: List[dict] = Field(default_factory=list)
    rejection_reasons: Optional[list] = None

class SceneState:
    """Manages the state of a single scene and provides methods for mutation.

    This class wraps a scene dictionary and provides a state machine-like
    interface for modifying the scene's status, bounding box, and other
    properties in a controlled way.
    """
    def __init__(self, scene_dict: dict):
        """Initializes the SceneState.

        Args:
            scene_dict (dict): The dictionary representing the scene's data.
        """
        self._scene = scene_dict
        # Set initial_bbox if it's not already set from the first seed result
        if self._scene.get('initial_bbox') is None and self._scene.get('seed_result', {}).get('bbox'):
            self._scene['initial_bbox'] = self._scene['seed_result']['bbox']
            self._scene['selected_bbox'] = self._scene['seed_result']['bbox']


    @property
    def data(self) -> dict:
        """Provides access to the underlying scene data dictionary.

        Returns:
            dict: The scene's data.
        """
        return self._scene

    def set_manual_bbox(self, bbox: list[int], source: str):
        """Sets a new bounding box, marking it as a manual override.

        Args:
            bbox (list[int]): The new bounding box [x, y, w, h].
            source (str): A string indicating the source of the manual override
                (e.g., 'user_editor').
        """
        self._scene['selected_bbox'] = bbox
        # An override is only true if the new box is different from the initial one.
        if self._scene.get('initial_bbox') and self._scene['initial_bbox'] != bbox:
             self._scene['is_overridden'] = True
        else:
             self._scene['is_overridden'] = False # Not an override if it matches the original

        self._scene.setdefault('seed_config', {})['override_source'] = source
        self._scene['status'] = 'included' # A manual change implies inclusion
        self._scene['manual_status_change'] = True


    def reset(self):
        """Resets the scene's bounding box and overrides to their initial state.

        This method reverts any manual changes to the bounding box (`selected_bbox`)
        back to the automatically detected `initial_bbox`. It also clears any
        override flags and seeding configurations. The scene's inclusion `status`
        is intentionally not changed, preserving any explicit inclusion or
        exclusion choices.
        """
        self._scene['selected_bbox'] = self._scene.get('initial_bbox')
        self._scene['is_overridden'] = False
        self._scene['seed_config'] = {}
        # The 'status' is NOT reset, to preserve any explicit include/exclude action.
        self._scene['manual_status_change'] = False


    def include(self):
        """Marks the scene as included for propagation."""
        self._scene['status'] = 'included'
        self._scene['manual_status_change'] = True

    def exclude(self):
        """Marks the scene as excluded from propagation."""
        self._scene['status'] = 'excluded'
        self._scene['manual_status_change'] = True

    def update_seed_result(self, bbox: Optional[list[int]], details: dict):
        """Updates the scene with a new seed result after re-computation.

        This is used when a scene's seed is re-calculated, for example, after
        a user changes the text prompt in the scene editor.

        Args:
            bbox (Optional[list[int]]): The new bounding box for the seed.
            details (dict): A dictionary of details about the new seed result.
        """
        self._scene['seed_result'] = {'bbox': bbox, 'details': details}
        # If this is the *first* seed result, it becomes the initial state.
        if self._scene.get('initial_bbox') is None:
            self._scene['initial_bbox'] = bbox
        # If not overridden, the selected bbox should follow the latest computation.
        if not self._scene.get('is_overridden', False):
            self._scene['selected_bbox'] = bbox

class AnalysisParameters(BaseModel):
    """A container for all parameters related to an analysis run.

    This Pydantic model consolidates all user-configurable parameters from the
    UI that are required for the various processing pipelines. It includes
    settings for frame extraction, model selection, and quality metric
    computation.

    Attributes:
        source_path: The path to the source video or image folder.
        method: The frame extraction method (e.g., 'interval', 'keyframes').
        ... and many others.
    """
    source_path: str = ""
    method: str = ""
    interval: float = 0.0
    max_resolution: str = ""
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
    need_masks_now: bool = False


    @classmethod
    def from_ui(cls, logger: 'AppLogger', config: 'Config', **kwargs) -> 'AnalysisParameters':
        """Creates an AnalysisParameters instance from UI component values.

        This class method is a factory that takes the raw, untyped values from
        the Gradio UI, coerces them into the correct Python types, applies
        default values from the main configuration, and returns a validated
        `AnalysisParameters` instance.

        Args:
            logger (AppLogger): The application logger.
            config (Config): The main application configuration.
            **kwargs: A dictionary of UI values, where keys are the parameter
                names.

        Returns:
            A populated and validated `AnalysisParameters` instance.
        """
        if 'face_ref_img_path' in kwargs or 'video_path' in kwargs:
            sanitized_face_ref, face_filter_enabled = _sanitize_face_ref(kwargs, logger)
            kwargs['face_ref_img_path'] = sanitized_face_ref
            kwargs['enable_face_filter'] = face_filter_enabled
        
        if 'thumb_megapixels' in kwargs:
            thumb_mp = kwargs['thumb_megapixels']
            if not isinstance(thumb_mp, (int, float)) or thumb_mp <= 0:
                logger.warning(f"Invalid thumb_megapixels: {thumb_mp}, using default")
                kwargs['thumb_megapixels'] = config.default_thumb_megapixels

        if 'pre_sample_nth' in kwargs:
            sample_nth = kwargs['pre_sample_nth']
            if not isinstance(sample_nth, int) or sample_nth < 1:
                logger.warning(f"Invalid pre_sample_nth: {sample_nth}, using 1")
                kwargs['pre_sample_nth'] = 1

        valid_keys = set(cls.model_fields.keys())

        # Start with all-False for compute flags, then selectively enable
        defaults = {f: False for f in valid_keys if f.startswith('compute_')}

        # Get all default values from the config
        config_defaults = config.model_dump()

        # Merge UI defaults
        for key in valid_keys:
            if f"default_{key}" in config_defaults:
                defaults[key] = config_defaults[f"default_{key}"]

        # Explicitly set compute defaults from filter_defaults as a baseline
        # This reflects the current behavior where if a filter exists, we compute the metric
        for metric in [k.replace('filter_default_', '') for k in config_defaults if k.startswith('filter_default_')]:
            compute_key = f"compute_{metric}"
            if compute_key in valid_keys:
                defaults[compute_key] = True

        # Also handle phash/dedup specially
        defaults['compute_phash'] = True

        instance = cls(**defaults)

        for key, value in kwargs.items():
            if hasattr(instance, key) and value is not None:
                # Allow empty strings for certain text fields if needed, but otherwise skip
                if isinstance(value, str) and not value.strip() and key not in ['text_prompt', 'face_ref_img_path']:
                    continue

                default = getattr(instance, key)
                try:
                    setattr(instance, key, _coerce(value, type(default)))
                except (ValueError, TypeError):
                    logger.warning(f"Could not coerce UI value for '{key}' to {type(default)}. Using default.", extra={'key': key, 'value': value})

        return instance

class MaskingResult(BaseModel):
    """Contains the results of a masking operation for a single frame.

    Attributes:
        mask_path: The file path to the saved mask image.
        shot_id: The ID of the scene (shot) this frame belongs to.
        seed_type: The type of seed strategy used for this scene.
        seed_face_sim: The face similarity score of the seed frame for this scene.
        mask_area_pct: The percentage of the frame area covered by the mask.
        mask_empty: A boolean flag indicating if the mask is considered empty
            (i.e., below a minimum area threshold).
        error: An error message if the masking operation failed for this frame.
    """
    mask_path: Optional[str] = None
    shot_id: Optional[int] = None
    seed_type: Optional[str] = None
    seed_face_sim: Optional[float] = None
    mask_area_pct: Optional[float] = None
    mask_empty: bool = True
    error: Optional[str] = None

# --- BASE PIPELINE ---

class Pipeline:
    """Base class for all processing pipelines."""
    def __init__(self, config: 'Config', logger: 'AppLogger', params: 'AnalysisParameters',
                 progress_queue: Queue, cancel_event: threading.Event):
        """Initializes the Pipeline.

        Args:
            config (Config): The application configuration.
            logger (AppLogger): The application logger.
            params (AnalysisParameters): The parameters for the analysis run.
            progress_queue (Queue): A queue to send progress updates to the UI.
            cancel_event (threading.Event): An event to signal cancellation of the task.
        """
        self.config = config
        self.logger = logger
        self.params = params
        self.progress_queue = progress_queue
        self.cancel_event = cancel_event

# --- CACHING & OPTIMIZATION ---

class ThumbnailManager:
    """Manages an in-memory cache for thumbnail images.

    This class uses an OrderedDict to implement a simple LRU (Least Recently Used)
    cache, which helps to avoid re-reading thumbnail files from disk repeatedly.
    """
    def __init__(self, logger: 'AppLogger', config: 'Config'):
        """Initializes the ThumbnailManager.

        Args:
            logger (AppLogger): The application logger.
            config (Config): The application configuration, used to determine
                cache size and eviction policy.
        """
        self.logger = logger
        self.config = config
        self.cache = OrderedDict()
        self.max_size = self.config.cache_size
        self.logger.info(f"ThumbnailManager initialized with cache size {self.max_size}")

    def get(self, thumb_path: Path) -> Optional[np.ndarray]:
        """Retrieves a thumbnail from the cache or loads it from disk.

        If the thumbnail is in the cache, it's returned directly and moved to
        the end of the LRU list. Otherwise, it's loaded from the given path,
        added to the cache, and then returned. The cache size is managed
        according to the configured limits.

        Args:
            thumb_path (Path): The file path of the thumbnail to retrieve.

        Returns:
            Optional[np.ndarray]: The thumbnail image as a NumPy array in RGB
            format, or None if the file does not exist or cannot be loaded.
        """
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
        """Forces the cache to be cleared to free up memory."""
        self.cache.clear()
        gc.collect()

    def _cleanup_old_entries(self):
        """Removes a percentage of the oldest entries from the cache.

        This method implements the cache eviction policy. It removes a
        configurable percentage of the least recently used items from the
        cache to make space for new entries.
        """
        num_to_remove = int(self.max_size * self.config.cache_eviction_factor)
        for _ in range(num_to_remove):
            if not self.cache:
                break
            self.cache.popitem(last=False)


# --- MODEL LOADING & MANAGEMENT ---

class ModelRegistry:
    """A thread-safe, lazy-loading registry for ML models."""
    def __init__(self, logger: Optional['AppLogger'] = None):
        """Initializes the ModelRegistry.

        Args:
            logger (Optional[AppLogger]): An optional logger instance. If not
                provided, a standard logger will be used.
        """
        self._models: Dict[str, Any] = {}
        self._locks: Dict[str, threading.Lock] = defaultdict(threading.Lock)
        self.logger = logger or logging.getLogger(__name__) # Fallback to standard logger

    def get_or_load(self, key: str, loader_fn: Callable[[], Any]) -> Any:
        """Retrieves a model from the registry, loading it if it's the first time.

        This method implements a thread-safe, lazy-loading pattern. It uses
        double-checked locking to ensure that the `loader_fn` is called only
        once per key, even if multiple threads request the same model
        concurrently.

        Args:
            key (str): A unique identifier for the model (e.g., "yolo_detector").
            loader_fn (Callable[[], Any]): A zero-argument function that loads
                and returns the model instance.

        Returns:
            Any: The loaded model instance.
        """
        # Double-checked locking for thread-safe lazy initialization
        if key not in self._models:
            with self._locks[key]:
                if key not in self._models:
                    if self.logger:
                        self.logger.info(f"Loading model '{key}' for the first time...")
                    self._models[key] = loader_fn()
                    if self.logger:
                        self.logger.success(f"Model '{key}' loaded successfully.")
        return self._models[key]

    def clear(self):
        """Removes all cached models from the registry and releases resources."""
        if self.logger:
            self.logger.info("Clearing all models from the registry.")
        self._models.clear()
        # The locks in defaultdict don't need explicit clearing.

def _compute_sha256(path: Path) -> str:
    """Computes the SHA256 hash of a file.

    Args:
        path (Path): The path to the file.

    Returns:
        str: The hexadecimal SHA256 hash string.
    """
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def download_model(url: str, dest_path: Union[str, Path], description: str, logger: 'AppLogger',
                   error_handler: 'ErrorHandler', user_agent: str, min_size: int = 1_000_000,
                   expected_sha256: Optional[str] = None):
    """Downloads a model file with caching, retry, and checksum verification.

    This function handles the robust downloading of model files. It first checks
    if a valid file already exists at the destination path by verifying its SHA256
    checksum. If not, it downloads the file, retrying on failure. After
    downloading, it verifies the file's integrity against the expected checksum
    or a minimum file size.

    Args:
        url (str): The URL to download the model from.
        dest_path (Union[str, Path]): The local destination path to save the file.
        description (str): A user-friendly description of the model for logging.
        logger (AppLogger): The application logger.
        error_handler (ErrorHandler): The error handler for retry logic.
        user_agent (str): The user agent string for the download request.
        min_size (int): The minimum acceptable file size in bytes, used as a
            fallback verification if no SHA256 is provided.
        expected_sha256 (Optional[str]): The expected SHA256 checksum for the
            downloaded file.

    Raises:
        RuntimeError: If the download fails after all retries or if the
            downloaded file fails verification.
    """
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.is_file():
        if expected_sha256:
            actual_sha256 = _compute_sha256(dest_path)
            if actual_sha256 == expected_sha256:
                logger.info(f"Using cached and verified {description}: {dest_path}")
                return
            else:
                logger.warning(f"Cached {description} has incorrect SHA256. Re-downloading.",
                               extra={'expected': expected_sha256, 'actual': actual_sha256})
                dest_path.unlink()
        elif min_size is None or dest_path.stat().st_size >= min_size:
            logger.info(f"Using cached {description} (SHA not verified): {dest_path}")
            return

    @error_handler.with_retry(recoverable_exceptions=(urllib.error.URLError, TimeoutError, RuntimeError))
    def download_func():
        logger.info(f"Downloading {description}", extra={'url': url, 'dest': dest_path})
        req = urllib.request.Request(url, headers={"User-Agent": user_agent})
        with urllib.request.urlopen(req, timeout=180) as resp, open(dest_path, "wb") as out:
            shutil.copyfileobj(resp, out)

        if not dest_path.exists():
            raise RuntimeError(f"Download of {description} failed (file not found after download).")

        if expected_sha256:
            actual_sha256 = _compute_sha256(dest_path)
            if actual_sha256 != expected_sha256:
                raise RuntimeError(f"SHA256 mismatch for {description}. Expected {expected_sha256}, got {actual_sha256}.")
        elif dest_path.stat().st_size < min_size:
            raise RuntimeError(f"Downloaded {description} seems incomplete (file size too small).")

        logger.success(f"{description} downloaded and verified successfully.")

    try:
        download_func()
    except Exception as e:
        logger.error(f"Failed to download {description}", exc_info=True, extra={'url': url})
        # Clean up partial download if it exists
        if dest_path.exists():
            dest_path.unlink()
        raise RuntimeError(f"Failed to download required model: {description}") from e

# Thread-local storage for non-thread-safe models
thread_local = threading.local()

def get_face_landmarker(model_path: str, logger: 'AppLogger') -> vision.FaceLandmarker:
    """Initializes and returns a thread-local instance of the MediaPipe FaceLandmarker.

    This function uses a `threading.local()` object to ensure that each thread
    gets its own instance of the FaceLandmarker, as the object is not
    thread-safe and would cause issues in a multi-threaded environment like
    the analysis pipeline.

    Args:
        model_path (str): The path to the Face Landmarker model file (.task).
        logger (AppLogger): The application logger.

    Returns:
        vision.FaceLandmarker: An initialized FaceLandmarker instance that is
        local to the current thread.

    Raises:
        RuntimeError: If the model cannot be initialized.
    """

    # Check if a landmarker instance already exists for this thread
    if hasattr(thread_local, 'face_landmarker_instance'):
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

        # Cache the instance on the current thread
        thread_local.face_landmarker_instance = detector

        logger.success("Face landmarker model initialized successfully for this thread.")
        return detector
    except Exception as e:
        logger.error(f"Could not initialize MediaPipe face landmarker model. Error: {e}", component="face_landmarker")
        raise RuntimeError("Could not initialize MediaPipe face landmarker model.") from e

def get_face_analyzer(model_name: str, models_path: str, det_size_tuple: tuple, logger: 'AppLogger', device: str = 'cpu') -> 'FaceAnalysis':
    """Loads and caches an InsightFace FaceAnalysis model.

    This function uses the global `ModelRegistry` to lazily load and cache the
    InsightFace `FaceAnalysis` model. It constructs a unique key based on the
    model parameters to ensure that the same model is not loaded more than once.
    It also includes a fallback to CPU if GPU loading fails due to OOM errors.

    Args:
        model_name (str): The name of the InsightFace model (e.g., "buffalo_l").
        models_path (str): The root directory for storing models.
        det_size_tuple (tuple): A tuple specifying the detection size (width, height).
        logger (AppLogger): The application logger.
        device (str): The device to run the model on ('cuda' or 'cpu').

    Returns:
        FaceAnalysis: An initialized `FaceAnalysis` instance.

    Raises:
        RuntimeError: If the model fails to initialize, even with a CPU fallback.
    """
    from insightface.app import FaceAnalysis
    # The key includes all parameters that affect the loaded model.
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
                    # Retry with CPU
                    analyzer = FaceAnalysis(name=model_name, root=models_path,
                                          providers=['CPUExecutionProvider'])
                    analyzer.prepare(ctx_id=-1, det_size=det_size_tuple)
                    return analyzer
                except Exception as cpu_e:
                    logger.error(f"CPU fallback also failed: {cpu_e}")
            raise RuntimeError(f"Could not initialize face analysis model. Error: {e}") from e

    return model_registry.get_or_load(model_key, _loader)

class PersonDetector:
    """A wrapper for the YOLO person detection model."""
    def __init__(self, logger: 'AppLogger', model_path: Union[Path, str], imgsz: int, conf: float, device: str = 'cuda'):
        """Initializes the PersonDetector.

        This class loads a YOLO model and configures it for person detection.
        It includes logic to handle auto-downloading of models from Ultralytics
        if the specified model file is not found locally.

        Args:
            logger (AppLogger): The application logger.
            model_path (Union[Path, str]): The path to the YOLO model file.
            imgsz (int): The image size for inference (e.g., 640).
            conf (float): The confidence threshold for detection.
            device (str): The device to run the model on ('cuda' or 'cpu').
        """
        from ultralytics import YOLO
        self.logger = logger
        self.device = device
        model_p = Path(model_path)
        model_str = str(model_p)

        if not model_p.exists():
            # If the path doesn't exist, pass just the filename to YOLO
            # so it can trigger the auto-download from Ultralytics Hub.
            model_str_for_yolo = model_p.name
            self.logger.info(f"Local YOLO model not found at '{model_str}'. Attempting to load by name for auto-download.",
                             component="person_detector", extra={'model_name': model_str_for_yolo})
        else:
            model_str_for_yolo = model_str

        try:
            self.model = YOLO(model_str_for_yolo)
            self.model.to(self.device)
            self.imgsz = imgsz
            self.conf = conf
            self.logger.info("YOLO person detector loaded", component="person_detector",
                             custom_fields={'device': self.device, 'model': model_str_for_yolo})
        except Exception as e:
            self.logger.error("Failed to load YOLO model", component="person_detector", exc_info=True)
            raise e

    def detect_boxes(self, img_rgb: np.ndarray) -> List[dict]:
        """Detects people in an image.

        Args:
            img_rgb (np.ndarray): The input image as a NumPy array in RGB format.

        Returns:
            List[dict]: A list of dictionaries, where each dictionary
            represents a detected person and contains 'bbox' (in xyxy format),
            'conf' (confidence score), and 'type' ('yolo') keys.
        """
        res = self.model.predict(img_rgb, imgsz=self.imgsz, conf=self.conf, classes=[0], verbose=False)
        out = []
        for r in res:
            if getattr(r, "boxes", None) is None:
                continue
            for b in r.boxes.cpu():
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                conf = float(b.conf[0])
                out.append({"bbox": [x1, y1, x2, y2], "conf": conf, "type": "yolo"})
        return out

def get_person_detector(model_path_str: str, device: str, imgsz: int, conf: float, logger: 'AppLogger') -> 'PersonDetector':
    """Factory and registry function for the YOLO person detector.

    Uses the global model_registry to ensure lazy loading and caching of the
    `PersonDetector` instance. It constructs a unique key based on the model
    parameters to avoid reloading the same model. Includes a fallback to a
    smaller default model if the primary one fails to load.

    Args:
        model_path_str (str): The path to the YOLO model file (e.g., 'yolo11x.pt').
        device (str): The device to run the model on ('cuda' or 'cpu').
        imgsz (int): The image size for inference (e.g., 640).
        conf (float): The confidence threshold for detection.
        logger (AppLogger): The application logger.

    Returns:
        An initialized `PersonDetector` instance.
    """
    model_key = f"person_detector_{Path(model_path_str).name}_{device}_{imgsz}_{conf}"

    def _loader():
        try:
            return PersonDetector(logger=logger, model_path=Path(model_path_str), imgsz=imgsz, conf=conf, device=device)
        except Exception as e:
            logger.warning(f"Primary YOLO detector '{model_path_str}' failed to load, attempting fallback to yolo11s.pt. Error: {e}", component="person_detector")
            # Fallback to a smaller, more reliable model if the primary one fails
            return get_person_detector(model_path_str="yolo11s.pt", device=device, imgsz=imgsz, conf=conf, logger=logger)

    return model_registry.get_or_load(model_key, _loader)


def resolve_grounding_dino_config(config_path: str) -> str:
    """Resolves the path to the GroundingDINO config file within its package.

    This function uses `importlib.resources` to safely locate the configuration
    file within the installed `groundingdino` package, making the application
    robust to changes in the environment or installation location.

    Args:
        config_path (str): The filename of the config file (e.g.,
            "GroundingDINO_SwinT_OGC.py").

    Returns:
        str: The absolute path to the config file.

    Raises:
        RuntimeError: If the config file cannot be found within the package,
            indicating a potential installation issue.
    """
    try:
        import importlib.resources as pkg_resources
        from groundingdino import config as gdino_config_module
        with pkg_resources.path(gdino_config_module, config_path) as config_file:
            return str(config_file)
    except (ImportError, ModuleNotFoundError, FileNotFoundError):
        raise RuntimeError(
            f"Could not resolve GroundingDINO config '{config_path}'. "
            "Ensure the 'groundingdino-py' package is installed correctly and the config file exists within it."
        )

def get_grounding_dino_model(gdino_config_path: str, gdino_checkpoint_path: str, models_path: str,
                             grounding_dino_url: str, user_agent: str, retry_params: tuple,
                             device: str, logger: Optional['AppLogger'] = None) -> Optional[torch.nn.Module]:
    """Factory and registry function for the GroundingDINO model.

    Uses the global `model_registry` to ensure lazy loading and caching. It handles
    downloading the model checkpoint if it doesn't exist, resolving the
    package-internal config file path, and initializing the model on the
    specified device.

    Args:
        gdino_config_path (str): The filename of the model's configuration file.
        gdino_checkpoint_path (str): The path to the model's checkpoint file.
        models_path (str): The root directory for storing downloaded models.
        grounding_dino_url (str): The URL to download the model from.
        user_agent (str): The user agent for the download request.
        retry_params (tuple): A tuple of (max_attempts, backoff_seconds) for
            download retries.
        device (str): The device to run the model on (e.g., 'cuda', 'cpu').
        logger (Optional[AppLogger]): The application logger.

    Returns:
        Optional[torch.nn.Module]: The initialized GroundingDINO model, or None
        if loading fails.
    """
    model_key = f"grounding_dino_{Path(gdino_checkpoint_path).name}_{device}"

    def _loader():
        _logger = logger or AppLogger(config=Config())
        error_handler = ErrorHandler(_logger, *retry_params)
        try:
            models_dir = Path(models_path)
            models_dir.mkdir(parents=True, exist_ok=True)

            config_file_path = gdino_config_path or Config().paths.grounding_dino_config
            config_path = resolve_grounding_dino_config(config_file_path)
            ckpt_path = Path(gdino_checkpoint_path)
            if not ckpt_path.is_absolute():
                ckpt_path = models_dir / ckpt_path.name

            download_model(grounding_dino_url, ckpt_path, "GroundingDINO Swin-T model",
                           _logger, error_handler, user_agent,
                           expected_sha256=Config().grounding_dino_sha256)

            model = gdino_load_model(
                model_config_path=config_path,
                model_checkpoint_path=str(ckpt_path),
                device=device
            )
            return model
        except Exception as e:
            _logger.error("Grounding DINO model loading failed.", component="grounding", exc_info=True)
            return None # Return None on failure

    return model_registry.get_or_load(model_key, _loader)

def predict_grounding_dino(model: torch.nn.Module, image_tensor: torch.Tensor, caption: str,
                           box_threshold: float, text_threshold: float, device: str) -> tuple:
    """Runs inference with the GroundingDINO model.

    This function wraps the `gdino_predict` call, enabling automatic mixed
    precision (`torch.amp.autocast`) for better performance on CUDA devices.

    Args:
        model (torch.nn.Module): The initialized GroundingDINO model.
        image_tensor (torch.Tensor): The preprocessed input image tensor.
        caption (str): The text prompt for detection.
        box_threshold (float): The object detection confidence threshold.
        text_threshold (float): The text matching confidence threshold.
        device (str): The device to run inference on (e.g., 'cuda', 'cpu').

    Returns:
        tuple: A tuple containing the detected boxes, confidences, and labels.
    """

    with torch.no_grad(), torch.amp.autocast('cuda', enabled=(device == 'cuda')):
        return gdino_predict(model=model, image=image_tensor.to(device), caption=caption,
                             box_threshold=float(box_threshold), text_threshold=float(text_threshold))

def get_dam4sam_tracker(model_name: str, models_path: str, model_urls_tuple: tuple, user_agent: str,
                        retry_params: tuple, logger: 'AppLogger', device: str) -> Optional['DAM4SAMTracker']:
    """Factory and registry function for the DAM4SAM tracker model.

    Uses the global `model_registry` to ensure lazy loading and caching. It
    handles downloading the model checkpoint if it doesn't exist and copies it
    to the location expected by the DAM4SAM submodule. This function requires
    CUDA to be available.

    Args:
        model_name (str): The name of the DAM4SAM model to load (e.g., "sam21pp-L").
        models_path (str): The root directory for storing downloaded models.
        model_urls_tuple (tuple): A tuple of (name, url) pairs for all
            available DAM4SAM models.
        user_agent (str): The user agent for download requests.
        retry_params (tuple): A tuple of (max_attempts, backoff_seconds) for
            download retries.
        logger (AppLogger): The application logger.
        device (str): The device to run the model on. Must be 'cuda'.

    Returns:
        Optional[DAM4SAMTracker]: An initialized `DAM4SAMTracker` instance, or
        None if loading fails or CUDA is not available.
    """
    model_urls = dict(model_urls_tuple)
    selected_name = model_name or Config().default_dam4sam_model_name or next(iter(model_urls.keys()))
    model_key = f"dam4sam_tracker_{selected_name}_{device}"

    def _loader():
        error_handler = ErrorHandler(logger, *retry_params)

        if device != 'cuda':
            logger.error("DAM4SAM requires CUDA but it's not available.")
            return None

        try:
            models_dir = Path(models_path)
            models_dir.mkdir(parents=True, exist_ok=True)
            if selected_name not in model_urls:
                raise ValueError(f"Unknown DAM4SAM model: {selected_name}")

            url = model_urls[selected_name]
            expected_sha256 = Config().dam4sam_sha256.get(selected_name)
            checkpoint_path = models_dir / Path(url).name

            download_model(url, checkpoint_path, f"DAM4SAM {selected_name}", logger, error_handler, user_agent,
                           expected_sha256=expected_sha256)

            actual_path, _ = dam_utils.determine_tracker(selected_name)
            if not Path(actual_path).exists():
                Path(actual_path).parent.mkdir(exist_ok=True, parents=True)
                shutil.copy(checkpoint_path, actual_path)

            tracker = DAM4SAMTracker(selected_name)
            return tracker
        except Exception as e:
            logger.error(f"Failed to initialize DAM4SAM tracker {selected_name}: {str(e)}", exc_info=True)
            if torch.cuda.is_available():
                logger.error(f"CUDA memory: {torch.cuda.memory_allocated()/1024**3:.1f}GB allocated, {torch.cuda.memory_reserved()/1024**3:.1f}GB reserved")
                torch.cuda.empty_cache()
            return None

    return model_registry.get_or_load(model_key, _loader)


def get_lpips_metric(model_name: str = 'alex', device: str = 'cpu') -> torch.nn.Module:
    """Factory and registry function for the LPIPS model.

    Uses the global `model_registry` to ensure lazy loading and caching of the
    LPIPS (Learned Perceptual Image Patch Similarity) metric model.

    Args:
        model_name (str): The name of the LPIPS model backbone (e.g., 'alex'
            or 'vgg'). Defaults to 'alex'.
        device (str): The device to run the model on. Defaults to 'cpu'.

    Returns:
        torch.nn.Module: An initialized LPIPS model instance.
    """
    return lpips.LPIPS(net=model_name).to(device)

def initialize_analysis_models(params: 'AnalysisParameters', config: 'Config', logger: 'AppLogger') -> dict:
    """
    Initializes and returns all models required for a specific analysis run.

    This function selectively loads models based on the provided analysis parameters
    to avoid unnecessary memory usage.

    Args:
        params: The parameters for the analysis run.
        config: The main application configuration.
        logger: The application logger.

    Returns:
        A dictionary containing the initialized models.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    face_analyzer, ref_emb, person_detector, face_landmarker = None, None, None, None

    # For YOLO-only mode, only the person detector is needed.
    if params.primary_seed_strategy == "🧑‍🤝‍🧑 Find Prominent Person":
        model_path = Path(config.models_dir) / params.person_detector_model
        person_detector = get_person_detector(
            model_path_str=str(model_path),
            device=device,
            imgsz=config.person_detector_imgsz,
            conf=config.person_detector_conf,
            logger=logger
        )
        return {"face_analyzer": None, "ref_emb": None, "person_detector": person_detector, "face_landmarker": None, "device": device}

    if params.enable_face_filter:
        face_analyzer = get_face_analyzer(
            model_name=params.face_model_name,
            models_path=str(config.models_dir),
            det_size_tuple=tuple(config.model_face_analyzer_det_size),
            logger=logger,
            device=device
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

    model_path = Path(config.models_dir) / params.person_detector_model
    person_detector = get_person_detector(
        model_path_str=str(model_path),
        device=device,
        imgsz=config.person_detector_imgsz,
        conf=config.person_detector_conf,
        logger=logger
    )

    # Initialize MediaPipe Face Landmarker
    landmarker_path = Path(config.models_dir) / Path(config.face_landmarker_url).name
    error_handler = ErrorHandler(logger, config.retry_max_attempts, config.retry_backoff_seconds)
    download_model(config.face_landmarker_url, landmarker_path, "MediaPipe Face Landmarker", logger, error_handler,
                   config.user_agent, expected_sha256=config.face_landmarker_sha256)
    if landmarker_path.exists():
        face_landmarker = get_face_landmarker(str(landmarker_path), logger)

    return {"face_analyzer": face_analyzer, "ref_emb": ref_emb, "person_detector": person_detector, "face_landmarker": face_landmarker, "device": device}

# --- VIDEO & FRAME PROCESSING ---

class VideoManager:
    """Handles video downloading from YouTube and local file validation.

    This class abstracts the source of the video, whether it's a remote URL
    that needs to be downloaded or a local file that needs to be validated. It
    provides a unified interface to prepare the video for processing and to
    extract its metadata.

    Attributes:
        source_path (str): The URL or local path of the video.
        config (Config): The application configuration.
        max_resolution (str): The maximum resolution for YouTube downloads.
        is_youtube (bool): A flag indicating if the source is a YouTube URL.
    """
    def __init__(self, source_path: str, config: 'Config', max_resolution: Optional[str] = None):
        """Initializes the VideoManager.

        Args:
            source_path (str): The URL or local path of the video.
            config (Config): The application configuration.
            max_resolution (Optional[str]): The maximum resolution to download
                for YouTube videos. Defaults to the value in the config.
        """
        self.source_path = source_path
        self.config = config
        self.max_resolution = max_resolution or self.config.default_max_resolution
        self.is_youtube = ("youtube.com/" in source_path or "youtu.be/" in source_path)

    def prepare_video(self, logger: 'AppLogger') -> str:
        """Ensures the video is available locally.

        If the source is a YouTube URL, it downloads the video using `yt-dlp`.
        If it's a local path, it validates the file's existence and format.

        Args:
            logger (AppLogger): The application logger.

        Returns:
            str: The local path to the prepared video file.

        Raises:
            RuntimeError: If a YouTube download fails or the local file is
                invalid.
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
        """Retrieves metadata from a video file using OpenCV.

        Args:
            video_path (str): The path to the video file.

        Returns:
            dict: A dictionary containing the video's "width", "height", "fps",
            and "frame_count". Includes a fallback for FPS if the value is
            invalid.
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened(): raise IOError(f"Could not open video: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        if not np.isfinite(fps) or fps <= 0:
            fps = 30.0
        info = {"width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "fps": fps, "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}
        cap.release()
        return info

def run_scene_detection(video_path: str, output_dir: Path, logger: Optional['AppLogger'] = None) -> list:
    """Detects scene changes in a video file using PySceneDetect.

    This function uses a content-aware detector to find shot boundaries in the
    video. The resulting list of scenes (start and end frame numbers) is saved
    to a `scenes.json` file in the specified output directory.

    Args:
        video_path (str): The path to the video file.
        output_dir (Path): The directory to save the `scenes.json` file in.
        logger (Optional[AppLogger]): The application logger.

    Returns:
        list: A list of tuples, where each tuple contains the start and end
        frame numbers of a detected scene. Returns an empty list on failure.
    """

    logger = logger or AppLogger(config=Config())
    logger.info("Detecting scenes...", component="video")
    try:
        scene_list = detect(str(video_path), ContentDetector())
        shots = ([(s.get_frames(), e.get_frames()) for s, e in scene_list] if scene_list else [])
        with (output_dir / "scenes.json").open('w', encoding='utf-8') as f: json.dump(shots, f)
        logger.success(f"Found {len(shots)} scenes.", component="video")
        return shots
    except Exception as e:
        logger.error("Scene detection failed.", component="video", exc_info=True)
        return []

def make_photo_thumbs(image_paths: list[Path], out_dir: Path, params: 'AnalysisParameters', cfg: 'Config',
                      logger: 'AppLogger', tracker: Optional['AdvancedProgressTracker'] = None) -> dict:
    """Generates thumbnails for a list of image files.

    This function is used when processing a folder of images. It resizes each
    image to a standard thumbnail resolution, saves it as a WebP file, and
    creates `frame_map.json` and `image_manifest.json` files to maintain
    compatibility with the video processing pipeline.

    Args:
        image_paths (list[Path]): A list of paths to the source images.
        out_dir (Path): The output directory to save thumbnails and manifests.
        params (AnalysisParameters): The analysis parameters, used for thumbnail
            resolution.
        cfg (Config): The application configuration.
        logger (AppLogger): The application logger.
        tracker (Optional[AdvancedProgressTracker]): An optional progress tracker.

    Returns:
        dict: A dictionary mapping frame numbers (image sequence index) to
        thumbnail filenames.
    """
    thumbs_dir = out_dir / "thumbs"
    thumbs_dir.mkdir(parents=True, exist_ok=True)
    target_area = params.thumb_megapixels * 1_000_000
    frame_map, image_manifest = {}, {}

    if tracker:
        tracker.start(len(image_paths), desc="Generating thumbnails")

    for i, img_path in enumerate(image_paths, start=1):
        if tracker and tracker.pause_event.is_set(): tracker.step()
        try:
            bgr = cv2.imread(str(img_path))
            if bgr is None:
                logger.warning(f"Could not read image file: {img_path}")
                continue

            h, w = bgr.shape[:2]
            scale = math.sqrt(target_area / float(max(1, w * h)))
            if scale < 1.0:
                new_w, new_h = int((w * scale) // 2 * 2), int((h * scale) // 2 * 2)
                bgr = cv2.resize(bgr, (max(2, new_w), max(2, new_h)), interpolation=cv2.INTER_AREA)

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            out_name = f"frame_{i:06d}.webp"
            out_path = thumbs_dir / out_name

            Image.fromarray(rgb).save(out_path, format="WEBP", quality=cfg.ffmpeg_thumbnail_quality)

            frame_map[i] = out_name
            image_manifest[i] = str(img_path.resolve())
        except Exception as e:
            logger.error(f"Failed to process image {img_path}", exc_info=True)
        finally:
            if tracker: tracker.step()

    (out_dir / "frame_map.json").write_text(json.dumps(frame_map, indent=2), encoding="utf-8")
    (out_dir / "image_manifest.json").write_text(json.dumps(image_manifest, indent=2), encoding="utf-8")

    if tracker: tracker.done_stage("Thumbnails generated")
    return frame_map


def run_ffmpeg_extraction(video_path: str, output_dir: Path, video_info: dict, params: 'AnalysisParameters',
                          progress_queue: Queue, cancel_event: threading.Event, logger: 'AppLogger',
                          config: 'Config', tracker: Optional['AdvancedProgressTracker'] = None):
    """Runs FFmpeg to extract frames from a video.

    This function constructs and executes a complex FFmpeg command to extract
    frames based on the selected method (e.g., keyframes, interval). It uses
    FFmpeg's native progress reporting for efficient UI updates and captures
    frame metadata from the `showinfo` filter to create a `frame_map.json`.

    Args:
        video_path (str): The path to the source video file.
        output_dir (Path): The directory to save the extracted frames.
        video_info (dict): A dictionary with video metadata.
        params (AnalysisParameters): The analysis parameters for the extraction.
        progress_queue (Queue): The queue for UI progress updates.
        cancel_event (threading.Event): The event to signal cancellation.
        logger (AppLogger): The application logger.
        config (Config): The main application configuration.
        tracker (Optional[AdvancedProgressTracker]): An optional progress tracker.

    Raises:
        RuntimeError: If the FFmpeg process fails.
    """
    cmd_base = ['ffmpeg', '-y', '-i', str(video_path), '-hide_banner']

    # Use native progress reporting for better performance and reliability
    progress_args = ['-progress', 'pipe:1', '-nostats', '-loglevel', 'info']
    cmd_base.extend(progress_args)

    thumb_dir = output_dir / "thumbs"
    thumb_dir.mkdir(exist_ok=True)

    target_area = params.thumb_megapixels * 1_000_000
    w, h = video_info.get('width', 1920), video_info.get('height', 1080)
    scale_factor = math.sqrt(target_area / (w * h)) if w * h > 0 else 1.0
    vf_scale = f"scale=w=trunc(iw*{scale_factor}/2)*2:h=trunc(ih*{scale_factor}/2)*2"

    fps = max(1, int(video_info.get('fps', 30)))
    N = max(1, int(params.nth_frame or 0))
    interval = max(0.1, float(params.interval or 0.0))

    select_map = {
        "keyframes": "select='eq(pict_type,I)'",
        "every_nth_frame": f"select='not(mod(n,{N}))'",
        "nth_plus_keyframes": f"select='or(eq(pict_type,I),not(mod(n,{N})))'",
        "interval": f"fps=1/{interval}",
        "all": f"fps={fps}",
    }
    vf_select = select_map.get(params.method, f"fps={fps}")

    if params.thumbnails_only:
        vf = f"{vf_select},{vf_scale},showinfo"
        cmd = cmd_base + ["-vf", vf, "-c:v", "libwebp", "-lossless", "0",
                          "-quality", str(config.ffmpeg_thumbnail_quality),
                          "-vsync", "vfr", str(thumb_dir / "frame_%06d.webp")]
    else:
        vf = f"{vf_select},showinfo"
        cmd = cmd_base + ["-vf", vf, "-c:v", "png", "-vsync", "vfr", str(thumb_dir / "frame_%06d.png")]

    # We need both stdout (for progress) and stderr (for showinfo)
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8', bufsize=1)

    frame_map_list = []

    # Live processing of both streams
    stderr_output = ""
    with process.stdout, process.stderr:
        total_duration_s = video_info.get("frame_count", 0) / max(0.01, video_info.get("fps", 30))
        stdout_thread = threading.Thread(target=lambda: _process_ffmpeg_stream(process.stdout, tracker, "Extracting frames", total_duration_s))

        # We need a way to get the stderr back from the thread
        stderr_results = {}
        def process_stderr_and_store():
            nonlocal stderr_results
            frame_map, full_stderr = _process_ffmpeg_showinfo(process.stderr)
            stderr_results['frame_map'] = frame_map
            stderr_results['full_stderr'] = full_stderr

        stderr_thread = threading.Thread(target=process_stderr_and_store)
        stdout_thread.start()
        stderr_thread.start()

        while True:
            if cancel_event.is_set():
                process.terminate()
                break

            # Non-blocking check for process completion
            if process.poll() is not None:
                break

            # Wait for a short period before checking again
            try:
                process.wait(timeout=0.1)
            except subprocess.TimeoutExpired:
                continue

        stdout_thread.join()
        stderr_thread.join()

    process.wait()

    frame_map_list = stderr_results.get('frame_map', [])
    stderr_output = stderr_results.get('full_stderr', '')

    if frame_map_list:
        with open(output_dir / "frame_map.json", 'w', encoding='utf-8') as f:
            json.dump(sorted(frame_map_list), f)

    if process.returncode not in [0, -9] and not cancel_event.is_set(): # -9 is SIGKILL, can happen on cancel
        logger.error("FFmpeg extraction failed", extra={'returncode': process.returncode, 'stderr': stderr_output})
        raise RuntimeError(f"FFmpeg failed with code {process.returncode}. Check logs for details.")

def _process_ffmpeg_stream(stream, tracker: Optional['AdvancedProgressTracker'], desc: str, total_duration_s: float):
    """Parses FFmpeg's native progress output from a stream.

    This is an internal helper function that reads the `pipe:1` progress output
    from an FFmpeg process line by line. It parses key-value pairs to update
    the `AdvancedProgressTracker` with the current status of the extraction.

    Args:
        stream: The stdout stream of the FFmpeg process.
        tracker (Optional[AdvancedProgressTracker]): The progress tracker to
            update.
        desc (str): The description for the progress stage.
        total_duration_s (float): The total duration of the video in seconds,
            used for calculating the progress fraction.
    """
    progress_data = {}
    for line in iter(stream.readline, ''):
        try:
            key, value = line.strip().split('=', 1)
            progress_data[key] = value

            if key == 'progress' and value == 'end':
                if tracker:
                    # Set to 100% and finalize
                    tracker.set(tracker.total, desc=desc)
                break

            # Use out_time_us for a more accurate fraction than frame number
            if key == 'out_time_us' and total_duration_s > 0:
                us = int(value)
                fraction = us / (total_duration_s * 1_000_000)
                if tracker:
                    done = int(fraction * tracker.total)
                    tracker.set(done, desc=desc)
            # Fallback to frame if out_time_us is not available or duration is unknown
            elif key == 'frame' and tracker and total_duration_s <= 0:
                 current_frame = int(value)
                 tracker.set(current_frame, desc=desc)

        except ValueError:
            pass  # Ignore malformed lines
    stream.close()

def _process_ffmpeg_showinfo(stream) -> tuple[list, str]:
    """Extracts frame numbers from FFmpeg's `showinfo` filter output.

    This internal helper function reads the `stderr` stream of an FFmpeg
    process. It uses a regular expression to find lines generated by the
    `showinfo` filter and extracts the original frame number (`n:`) from them.
    It also captures the full stderr output for logging purposes.

    Args:
        stream: The stderr stream of the FFmpeg process.

    Returns:
        tuple[list, str]: A tuple containing:
        - A list of integer frame numbers that were extracted.
        - The full stderr output as a single string.
    """
    frame_numbers = []
    stderr_lines = []
    for line in iter(stream.readline, ''):
        stderr_lines.append(line)
        match = re.search(r' n:\s*(\d+)', line)
        if match:
            frame_numbers.append(int(match.group(1)))
    stream.close()
    return frame_numbers, "".join(stderr_lines)

def postprocess_mask(mask: np.ndarray, config: 'Config', fill_holes: bool = True, keep_largest_only: bool = True) -> np.ndarray:
    """Cleans up a binary mask using morphological operations.

    This function takes a raw binary mask (e.g., from a segmentation model)
    and applies morphological closing to fill small holes and optionally
    removes all but the largest connected component to eliminate noise.

    Args:
        mask (np.ndarray): The input mask as a NumPy array.
        config (Config): The application configuration, used to get kernel sizes.
        fill_holes (bool): Whether to perform a closing operation to fill
            small holes. Defaults to True.
        keep_largest_only (bool): Whether to keep only the largest connected
            component. Defaults to True.

    Returns:
        np.ndarray: The post-processed binary mask.
    """
    if mask is None or mask.size == 0: return mask
    binary_mask = (mask > 128).astype(np.uint8)
    if fill_holes:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (config.masking_close_kernel_size, config.masking_close_kernel_size))
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    if keep_largest_only and config.masking_keep_largest_only:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
        if num_labels > 1:
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            binary_mask = (labels == largest_label).astype(np.uint8)
    return (binary_mask * 255).astype(np.uint8)

def render_mask_overlay(frame_rgb: np.ndarray, mask_gray: np.ndarray, alpha: float, logger: 'AppLogger') -> np.ndarray:
    """Renders a semi-transparent red overlay onto an image based on a mask.

    Args:
        frame_rgb (np.ndarray): The source image in RGB format.
        mask_gray (np.ndarray): The grayscale mask to use for the overlay.
        alpha (float): The transparency of the overlay (0.0 to 1.0).
        logger (AppLogger): The application logger.

    Returns:
        np.ndarray: The image with the mask rendered as a red overlay.
    """
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
    """Converts a NumPy array in RGB format to a Pillow Image.

    Args:
        image_rgb (np.ndarray): The input image as an RGB NumPy array.

    Returns:
        Image.Image: The converted Pillow Image object.
    """
    return Image.fromarray(image_rgb)

def create_frame_map(output_dir: Path, logger: 'AppLogger', ext: str = ".webp") -> dict:
    """Loads and reconstructs the frame map from `frame_map.json`.

    The `frame_map.json` file contains a list of the original frame numbers
    extracted by FFmpeg. This function reads that list, sorts it, and creates
    a dictionary that maps each original frame number to its corresponding
    sequentially-named thumbnail file (e.g., `{1234: "frame_000001.webp"}`).

    Args:
        output_dir (Path): The output directory where `frame_map.json` is
            located.
        logger (AppLogger): The application logger.
        ext (str): The file extension of the thumbnail images. Defaults to
            ".webp".

    Returns:
        dict: A dictionary mapping original integer frame numbers to thumbnail
        filenames. Returns an empty dictionary if the file cannot be read.
    """
    logger.info("Loading frame map...", component="frames")
    frame_map_path = output_dir / "frame_map.json"
    try:
        with open(frame_map_path, 'r', encoding='utf-8') as f: frame_map_list = json.load(f)
        # Ensure all frame numbers are integers, as JSON can load them as strings.
        sorted_frames = sorted(map(int, frame_map_list))
        return {orig_num: f"frame_{i+1:06d}{ext}" for i, orig_num in enumerate(sorted_frames)}
    except (FileNotFoundError, json.JSONDecodeError, TypeError) as e:
        logger.error(f"Could not load or parse frame_map.json: {e}. Frame mapping will be empty. This can happen if extraction was incomplete.", exc_info=False)
        return {}

# --- MASKING & PROPAGATION ---

class MaskPropagator:
    """Handles the propagation of a subject mask across a sequence of frames.

    This class wraps the DAM4SAM tracker. It takes a seed frame with an initial
    bounding box and propagates the resulting mask both forwards and backwards
    to cover all frames in a given shot (scene).

    Attributes:
        params (AnalysisParameters): The analysis parameters.
        dam_tracker (DAM4SAMTracker): An initialized DAM4SAMTracker instance.
        cancel_event (threading.Event): An event to signal cancellation.
        progress_queue (Queue): A queue for sending progress updates.
        config (Config): The main application configuration.
        logger (AppLogger): The application logger.
    """
    def __init__(self, params: 'AnalysisParameters', dam_tracker: 'DAM4SAMTracker', cancel_event: threading.Event,
                 progress_queue: Queue, config: 'Config', logger: Optional['AppLogger'] = None, device: str = "cpu"):
        """Initializes the MaskPropagator.

        Args:
            params (AnalysisParameters): The analysis parameters.
            dam_tracker (DAM4SAMTracker): An initialized DAM4SAMTracker instance.
            cancel_event (threading.Event): An event to signal cancellation.
            progress_queue (Queue): A queue for sending progress updates.
            config (Config): The main application configuration.
            logger (Optional[AppLogger]): The application logger.
            device (str): The compute device ('cpu' or 'cuda').
        """
        self.params = params
        self.dam_tracker = dam_tracker
        self.cancel_event = cancel_event
        self.progress_queue = progress_queue
        self.config = config
        self.logger = logger or AppLogger(config=Config())
        self._device = device

    def propagate(self, shot_frames_rgb: list[np.ndarray], seed_idx: int, bbox_xywh: list[int],
                  tracker: Optional['AdvancedProgressTracker'] = None) -> tuple[list, list, list, list]:
        """Propagates a mask from a seed frame to all other frames in a shot.

        The method first initializes the tracker on the seed frame using the
        provided bounding box. It then propagates the mask forward to the end
        of the shot, re-initializes, and propagates backward to the start.

        Args:
            shot_frames_rgb (list[np.ndarray]): A list of frames (as RGB NumPy
                arrays) in the shot.
            seed_idx (int): The index of the seed frame within the
                `shot_frames_rgb` list.
            bbox_xywh (list[int]): The initial bounding box [x, y, width, height]
                on the seed frame.
            tracker (Optional['AdvancedProgressTracker']): An optional progress
                tracker.

        Returns:
            tuple[list, list, list, list]: A tuple containing four lists, each
            with one entry per frame in the shot:
            - The propagated masks (as NumPy arrays).
            - The mask area percentages.
            - Booleans indicating if a mask is considered empty.
            - Error messages for each frame, if any.
        """
        if not self.dam_tracker or not shot_frames_rgb:
            err_msg = "Tracker not initialized" if not self.dam_tracker else "No frames"
            shape = shot_frames_rgb[0].shape[:2] if shot_frames_rgb else (100, 100)
            num_frames = len(shot_frames_rgb)
            return ([np.zeros(shape, np.uint8)] * num_frames, [0.0] * num_frames, [True] * num_frames, [err_msg] * num_frames)
        self.logger.info("Propagating masks", component="propagator", user_context={'num_frames': len(shot_frames_rgb), 'seed_index': seed_idx})
        masks = [None] * len(shot_frames_rgb)

        if tracker:
            tracker.set_stage(f"Propagating masks for {len(shot_frames_rgb)} frames")

        def _propagate_direction(start_idx, end_idx, step, desc):
            for i in range(start_idx, end_idx, step):
                if self.cancel_event.is_set(): break
                outputs = self.dam_tracker.track(rgb_to_pil(shot_frames_rgb[i]))
                mask = outputs.get('pred_mask')
                if mask is not None: mask = postprocess_mask((mask * 255).astype(np.uint8), config=self.config, fill_holes=True, keep_largest_only=True)
                masks[i] = mask if mask is not None else np.zeros_like(shot_frames_rgb[i], dtype=np.uint8)[:, :, 0]
                if tracker: tracker.step(1, desc=desc)
        try:
            with torch.amp.autocast('cuda', enabled=self._device == 'cuda'):
                outputs = self.dam_tracker.initialize(rgb_to_pil(shot_frames_rgb[seed_idx]), None, bbox=bbox_xywh)
                mask = outputs.get('pred_mask')
                if mask is not None: mask = postprocess_mask((mask * 255).astype(np.uint8), config=self.config, fill_holes=True, keep_largest_only=True)
                masks[seed_idx] = mask if mask is not None else np.zeros_like(shot_frames_rgb[seed_idx], dtype=np.uint8)[:, :, 0]
                if tracker:
                    tracker.step(1, desc="Propagation (seed)")

                # Propagate forward
                _propagate_direction(seed_idx + 1, len(shot_frames_rgb), 1, "Propagation (→)")

                # Re-initialize for backward propagation
                self.dam_tracker.initialize(rgb_to_pil(shot_frames_rgb[seed_idx]), None, bbox=bbox_xywh)
                _propagate_direction(seed_idx - 1, -1, -1, "Propagation (←)")

            h, w = shot_frames_rgb[0].shape[:2]
            final_results = []
            for i, mask in enumerate(masks):
                if self.cancel_event.is_set() or mask is None: mask = np.zeros((h, w), dtype=np.uint8)
                img_area = h * w
                area_pct = (np.sum(mask > 0) / img_area) * 100 if img_area > 0 else 0.0
                is_empty = area_pct < self.params.min_mask_area_pct
                error = "Empty mask" if is_empty else None
                final_results.append((mask, float(area_pct), bool(is_empty), error))
            if not final_results:
                return ([], [], [], [])
            masks, areas, empties, errors = map(list, zip(*final_results))
            return masks, areas, empties, errors
        except Exception as e:
            self.logger.critical("DAM4SAM propagation failed", component="propagator", exc_info=True)
            h, w = shot_frames_rgb[0].shape[:2]
            error_msg = f"Propagation failed: {e}"
            num_frames = len(shot_frames_rgb)
            return ([np.zeros((h, w), np.uint8)] * num_frames, [0.0] * num_frames, [True] * num_frames, [error_msg] * num_frames)

class SeedSelector:
    """Orchestrates the selection of the initial seed bounding box for tracking.

    This class encapsulates the complex logic for identifying the subject of
    interest in a given frame. It supports multiple strategies, including
    face matching, text-based detection with GroundingDINO, and various
    YOLO-based heuristics (e.g., largest person, center-most person).

    Attributes:
        params (AnalysisParameters): The analysis parameters.
        config (Config): The main application configuration.
        face_analyzer (FaceAnalysis): The InsightFace face analysis model.
        reference_embedding (np.ndarray): The embedding of the reference face.
        person_detector (PersonDetector): The YOLO person detector model.
        tracker (DAM4SAMTracker): The DAM4SAM tracker instance.
        logger (AppLogger): The application logger.
    """
    def __init__(self, params: 'AnalysisParameters', config: 'Config', face_analyzer: 'FaceAnalysis',
                 reference_embedding: np.ndarray, person_detector: 'PersonDetector', tracker: 'DAM4SAMTracker',
                 gdino_model: torch.nn.Module, logger: Optional['AppLogger'] = None, device: str = "cpu"):
        """Initializes the SeedSelector.

        Args:
            params (AnalysisParameters): The analysis parameters.
            config (Config): The main application configuration.
            face_analyzer (FaceAnalysis): The InsightFace face analysis model.
            reference_embedding (np.ndarray): The embedding of the reference
                face image.
            person_detector (PersonDetector): The YOLO person detector model.
            tracker (DAM4SAMTracker): The DAM4SAM tracker instance.
            gdino_model (torch.nn.Module): The GroundingDINO model.
            logger (Optional[AppLogger]): The application logger.
            device (str): The compute device ('cpu' or 'cuda').
        """
        self.params = params
        self.config = config
        self.face_analyzer = face_analyzer
        self.reference_embedding = reference_embedding
        self.person_detector = person_detector
        self.tracker = tracker
        self._gdino = gdino_model
        self._device = device
        self.logger = logger or AppLogger(config=Config())

    def _get_param(self, source: Union[dict, object], key: str, default: Any = None) -> Any:
        """Safely gets a parameter from a source that can be a dict or an object.

        This internal helper allows methods to accept either a dictionary or an
        `AnalysisParameters` object as their source of parameters, providing
        flexibility for different calling contexts (e.g., UI events vs.
        internal re-computation).

        Args:
            source (Union[dict, object]): The dictionary or object to get the
                parameter from.
            key (str): The key or attribute name.
            default (Any): The default value to return if the key is not found.
                Defaults to None.

        Returns:
            Any: The value of the parameter.
        """
        if isinstance(source, dict):
            return source.get(key, default)
        return getattr(source, key, default)

    def select_seed(self, frame_rgb: np.ndarray, current_params: Optional[dict] = None,
                    scene: Optional['Scene'] = None) -> tuple[Optional[list], dict]:
        """Selects the best seed bounding box based on the configured strategy.

        This is the main entry point for the class. It acts as a router,
        calling the appropriate private method based on the `primary_seed_strategy`
        parameter.

        Args:
            frame_rgb (np.ndarray): The RGB image of the frame to analyze.
            current_params (Optional[dict]): An optional dictionary of
                parameters to override instance defaults for this specific call.
            scene (Optional['Scene']): The scene object, which may contain
                pre-computed YOLO detections to improve performance.

        Returns:
            tuple[Optional[list], dict]: A tuple containing the selected
            bounding box [x, y, w, h] and a dictionary with details about the
            selection process.
        """
        params_source = current_params if current_params is not None else self.params
        p = params_source  # Keep for passing to other methods

        primary_strategy = self._get_param(params_source, 'primary_seed_strategy', "🤖 Automatic")
        use_face_filter = self._get_param(params_source, 'enable_face_filter', False)

        if primary_strategy == "👤 By Face":
            if self.face_analyzer and self.reference_embedding is not None and use_face_filter:
                self.logger.info("Starting 'Identity-First' seeding.")
                return self._identity_first_seed(frame_rgb, p, scene)
            else:
                self.logger.warning("Face strategy selected but no reference face provided.")
                return self._object_first_seed(frame_rgb, p, scene)
        elif primary_strategy == "📝 By Text":
            self.logger.info("Starting 'Object-First' seeding.")
            return self._object_first_seed(frame_rgb, p, scene)
        elif primary_strategy == "🔄 Face + Text Fallback":
            self.logger.info("Starting 'Face-First with Text Fallback' seeding.")
            return self._face_with_text_fallback_seed(frame_rgb, p, scene)
        else:
            self.logger.info("Starting 'Automatic' seeding.")
            return self._choose_person_by_strategy(frame_rgb, p, scene)

    def _face_with_text_fallback_seed(self, frame_rgb: np.ndarray, params: Union[dict, 'AnalysisParameters'],
                                      scene: Optional['Scene'] = None) -> tuple[Optional[list], dict]:
        """Attempts face-based seeding, falling back to text-based seeding.

        This strategy first tries the `_identity_first_seed` method. If that
        method fails to find a suitable bounding box (e.g., no matching face
        is found), it then automatically calls the `_object_first_seed` method
        as a fallback.

        Args:
            frame_rgb (np.ndarray): The RGB image of the frame.
            params (Union[dict, 'AnalysisParameters']): The analysis parameters.
            scene (Optional['Scene']): The current scene object.

        Returns:
            tuple[Optional[list], dict]: The result of either the identity-first
            or object-first seeding strategy.
        """
        # If no reference embedding is available, go straight to text fallback.
        if self.reference_embedding is None:
            self.logger.warning("No reference face for face-first strategy, falling back to text prompt.", extra={'reason': 'no_ref_emb'})
            return self._object_first_seed(frame_rgb, params, scene)

        # First, attempt the identity-first strategy.
        box, details = self._identity_first_seed(frame_rgb, params, scene)

        # If it succeeds (i.e., finds a matching face and subject), return the result.
        if box is not None:
            self.logger.info("Face-first strategy successful.")
            return box, details

        # If it fails, log the failure and fall back to the object-first (text) strategy.
        self.logger.warning("Face detection failed or no match found, falling back to text prompt strategy.", extra=details)
        return self._object_first_seed(frame_rgb, params, scene)

    def _identity_first_seed(self, frame_rgb: np.ndarray, params: Union[dict, 'AnalysisParameters'],
                             scene: Optional['Scene'] = None) -> tuple[Optional[list], dict]:
        """Finds a seed box by first identifying the target face.

        This strategy uses the reference face embedding to find the matching
        face in the frame. Once the face is located, it gathers candidate body
        boxes from both YOLO and GroundingDINO and selects the one that best
        encloses the target face. If no suitable body box is found, it falls
        back to expanding the face box to an estimated body size.

        Args:
            frame_rgb (np.ndarray): The RGB image of the frame.
            params (Union[dict, 'AnalysisParameters']): The analysis parameters.
            scene (Optional['Scene']): The current scene object.

        Returns:
            tuple[Optional[list], dict]: A tuple containing the selected
            bounding box and details about the selection.
        """
        target_face, details = self._find_target_face(frame_rgb)
        if not target_face:
            self.logger.warning("Target face not found in scene.", extra=details)
            return None, {"type": "no_subject_found"}
        yolo_boxes, dino_boxes = self._get_yolo_boxes(frame_rgb, scene), self._get_dino_boxes(frame_rgb, params)[0]
        best_box, best_details = self._score_and_select_candidate(target_face, yolo_boxes, dino_boxes)
        if best_box:
            self.logger.success("Evidence-based seed selected.", extra=best_details)
            return best_box, best_details
        self.logger.warning("No high-confidence body box found, expanding face box as fallback.")
        expanded_box = self._expand_face_to_body(target_face['bbox'], frame_rgb.shape)
        return expanded_box, {"type": "expanded_box_from_face", "seed_face_sim": details.get('seed_face_sim', 0)}

    def _object_first_seed(self, frame_rgb: np.ndarray, params: Union[dict, 'AnalysisParameters'],
                             scene: Optional['Scene'] = None) -> tuple[Optional[list], dict]:
        """Finds a seed box by first running GroundingDINO with a text prompt.

        This strategy uses the text prompt to get initial candidate boxes from
        GroundingDINO. It then attempts to find a high-IoU match with a YOLO
        person detection to increase confidence. If no YOLO match is found, it
        uses the best GroundingDINO result. If GroundingDINO fails, it falls
        back to a YOLO-only strategy.

        Args:
            frame_rgb (np.ndarray): The RGB image of the frame.
            params (Union[dict, 'AnalysisParameters']): The analysis parameters.
            scene (Optional['Scene']): The current scene object.

        Returns:
            tuple[Optional[list], dict]: A tuple containing the selected
            bounding box and details about the selection.
        """
        dino_boxes, dino_details = self._get_dino_boxes(frame_rgb, params)
        if dino_boxes:
            yolo_boxes = self._get_yolo_boxes(frame_rgb, scene)
            if yolo_boxes:
                best_iou, best_match = -1, None
                for d_box in dino_boxes:
                    for y_box in yolo_boxes:
                        iou = self._calculate_iou(d_box['bbox'], y_box['bbox'])
                        if iou > best_iou:
                            best_iou, best_match = iou, {'bbox': d_box['bbox'], 'type': 'dino_yolo_intersect', 'iou': iou,
                                                         'dino_conf': d_box['conf'], 'yolo_conf': y_box['conf']}
                if best_match and best_match['iou'] > self.config.seeding_yolo_iou_threshold:
                    self.logger.info("Found high-confidence DINO+YOLO intersection.", extra=best_match)
                    return self._xyxy_to_xywh(best_match['bbox']), best_match
            self.logger.info("Using best DINO box without YOLO validation.", extra=dino_details)
            return self._xyxy_to_xywh(dino_boxes[0]['bbox']), dino_details
        self.logger.info("No DINO results, falling back to YOLO-only strategy.")
        return self._choose_person_by_strategy(frame_rgb, params, scene)

    def _find_target_face(self, frame_rgb: np.ndarray) -> tuple[Optional[dict], dict]:
        """Finds the face in an image that best matches the reference embedding.

        This method runs face detection on the frame, then calculates the dot
        product (cosine similarity) between each detected face's embedding and
        the stored reference embedding.

        Args:
            frame_rgb (np.ndarray): The RGB image of the frame.

        Returns:
            tuple[Optional[dict], dict]: A tuple containing:
            - A dictionary with the best matching face's 'bbox' and
              'embedding', or None if no match exceeds the similarity
              threshold.
            - A dictionary with details of the match, including the similarity
              score or an error message.
        """
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
        if best_face and best_sim > self.config.seeding_face_similarity_threshold:
            return {'bbox': best_face.bbox.astype(int), 'embedding': best_face.normed_embedding}, {'type': 'face_match', 'seed_face_sim': best_sim}
        return None, {'error': 'no_matching_face', 'best_sim': best_sim}

    def _get_yolo_boxes(self, frame_rgb: np.ndarray, scene: Optional['Scene'] = None) -> list[dict]:
        """Gets person detections from YOLO.

        This method optimizes performance by prioritizing pre-computed YOLO
        detections if they are already stored on the `Scene` object. If not, it
        runs the YOLO model as a fallback. It can also synthesize a detection
        from a manually selected bounding box.

        Args:
            frame_rgb (np.ndarray): The RGB image of the frame.
            scene (Optional['Scene']): The current scene object, which may
                contain pre-computed detections.

        Returns:
            list[dict]: A list of detected person bounding boxes.
        """
        if scene and getattr(scene, 'yolo_detections', None):
            return scene.yolo_detections
        # use any saved bbox to synthesize a detection
        if scene and (scene.selected_bbox or scene.initial_bbox):
            xywh = scene.selected_bbox or scene.initial_bbox
            x, y, w, h = xywh
            xyxy = [x, y, x + w, y + h]
            return [{'bbox': xyxy, 'conf': 1.0, 'type': 'selected'}]
        if not self.person_detector:
            return []
        try:
            self.logger.warning("No pre-computed YOLO boxes found, running detection as a fallback.", component="system")
            return self.person_detector.detect_boxes(frame_rgb)
        except Exception as e:
            self.logger.warning("YOLO person detector failed.", exc_info=True)
            return []

    def _get_dino_boxes(self, frame_rgb: np.ndarray, params: Union[dict, 'AnalysisParameters']) -> tuple[list[dict], dict]:
        """Gets object detections from GroundingDINO based on a text prompt.

        This method preprocesses the image, runs inference with the GroundingDINO
        model using the provided text prompt, and converts the normalized output
        boxes to pixel coordinates.

        Args:
            frame_rgb (np.ndarray): The RGB image of the frame.
            params (Union[dict, 'AnalysisParameters']): The analysis parameters
                containing the text prompt and confidence thresholds.

        Returns:
            tuple[list[dict], dict]: A tuple containing:
            - A list of all detected object boxes, sorted by confidence.
            - A dictionary with details about the highest-confidence detection.
        """
        prompt = self._get_param(params, "text_prompt", "").strip()
        if not self._gdino or not prompt:
            return [], {}
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

    def _score_and_select_candidate(self, target_face: dict, yolo_boxes: list[dict], dino_boxes: list[dict]) -> tuple[Optional[list], dict]:
        """Scores candidate body boxes based on heuristics and selects the best one.

        This method implements an evidence-based scoring system. A candidate
        box receives points for containing the target face and for having a
        high IoU with a detection from the other model (e.g., a YOLO box that
        overlaps significantly with a DINO box). The original detection
        confidence is also factored in.

        Args:
            target_face (dict): The dictionary for the target face.
            yolo_boxes (list[dict]): A list of candidate boxes from YOLO.
            dino_boxes (list[dict]): A list of candidate boxes from GroundingDINO.

        Returns:
            tuple[Optional[list], dict]: A tuple containing the best bounding
            box and details about the scoring.
        """
        candidates = yolo_boxes + dino_boxes
        if not candidates: return None, {}
        scored_candidates = []
        for cand in candidates:
            score, details = 0, {'orig_conf': cand['conf'], 'orig_type': cand['type']}
            if self._box_contains(cand['bbox'], target_face['bbox']):
                score += self.config.seeding_face_contain_score
                details['face_contained'] = True
            score += cand['conf'] * self.config.seeding_confidence_score_multiplier
            scored_candidates.append({'score': score, 'box': cand['bbox'], 'details': details})
        best_iou, best_pair = -1, None
        for y_box in yolo_boxes:
            for d_box in dino_boxes:
                iou = self._calculate_iou(y_box['bbox'], d_box['bbox'])
                if iou > best_iou: best_iou, best_pair = iou, (y_box, d_box)
        if best_iou > self.config.seeding_yolo_iou_threshold:
            for cand in scored_candidates:
                if np.array_equal(cand['box'], best_pair[0]['bbox']) or np.array_equal(cand['box'], best_pair[1]['bbox']):
                    cand['score'] += self.config.seeding_iou_bonus
                    cand['details']['high_iou_pair'] = True
        if not scored_candidates: return None, {}
        winner = max(scored_candidates, key=lambda x: x['score'])
        return self._xyxy_to_xywh(winner['box']), {'type': 'evidence_based_selection', 'final_score': winner['score'], **winner['details']}

    def _choose_person_by_strategy(self, frame_rgb: np.ndarray, params: Union[dict, 'AnalysisParameters'],
                                     scene: Optional['Scene'] = None) -> tuple[list, dict]:
        """Selects a person from YOLO detections based on a scoring strategy.

        This is the fallback or "Automatic" method used when specific identity
        (face/text) is not required. It scores all detected people based on a
        chosen heuristic (e.g., "Largest Person", "Center-most Person") and
        returns the best one. It includes a final fallback to a default region
        if no people are detected at all.

        Args:
            frame_rgb (np.ndarray): The RGB image of the frame.
            params (Union[dict, 'AnalysisParameters']): The analysis parameters
                containing the scoring strategy.
            scene (Optional['Scene']): The current scene object.

        Returns:
            tuple[list, dict]: A tuple containing the selected person's bounding
            box and details about the selection strategy.
        """
        boxes = self._get_yolo_boxes(frame_rgb, scene)  # This now uses pre-computed results
        if not boxes:
            self.logger.warning(f"No people detected in scene - using fallback region")
            fallback_box = self._final_fallback_box(frame_rgb.shape)
            return fallback_box, {
                'type': 'no_people_fallback',
                'reason': 'No people detected in best frame',
                'detection_attempted': True
            }
        strategy = getattr(params, "seed_strategy", "Largest Person")
        if isinstance(params, dict): strategy = params.get('seed_strategy', strategy)
        h, w = frame_rgb.shape[:2]
        cx, cy = w / 2, h / 2

        def area(b):
            x1, y1, x2, y2 = b['bbox']
            return (x2 - x1) * (y2 - y1)

        def height(b):
            x1, y1, x2, y2 = b['bbox']
            return y2 - y1

        def center_dist(b):
            x1, y1, x2, y2 = b['bbox']
            bx, by = (x1 + x2) / 2, (y1 + y2) / 2
            return math.hypot(bx - cx, by - cy)

        def thirds_dist(b):
            thirds = [(w / 3, h / 3), (2 * w / 3, h / 3), (w / 3, 2 * h / 3), (2 * w / 3, 2 * h / 3)]
            x1, y1, x2, y2 = b['bbox']
            bx, by = (x1 + x2) / 2, (y1 + y2) / 2
            return min(math.hypot(bx - tx, by - ty) for tx, ty in thirds)

        def min_dist_to_edge(b):
            x1, y1, x2, y2 = b['bbox']
            return min(x1, y1, w - x2, h - y2)

        def balanced_score(b):
            weights = self.config.seeding_balanced_score_weights
            norm_area = area(b) / (w * h)
            norm_edge = min_dist_to_edge(b) / (min(w, h) / 2)
            return weights['area'] * norm_area + weights['confidence'] * b['conf'] + weights['edge'] * norm_edge

        # Memoize face detection results for the current frame
        all_faces = None
        if strategy == "Best Face" and self.face_analyzer:
            all_faces = self.face_analyzer.get(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))

        def best_face_score(b):
            if not all_faces:
                return 0.0

            yolo_bbox = b['bbox']
            faces_in_box = []
            for face in all_faces:
                face_cx = face.bbox[0] + face.bbox[2] / 2
                face_cy = face.bbox[1] + face.bbox[3] / 2
                if yolo_bbox[0] <= face_cx < yolo_bbox[2] and yolo_bbox[1] <= face_cy < yolo_bbox[3]:
                    faces_in_box.append(face)

            if not faces_in_box:
                return 0.0

            return max(f.det_score for f in faces_in_box)

        score_funcs = {
            "Largest Person": lambda b: area(b),
            "Center-most Person": lambda b: -center_dist(b),
            "Highest Confidence": lambda b: b['conf'],
            "Tallest Person": lambda b: height(b),
            "Area x Confidence": lambda b: area(b) * b['conf'],
            "Rule-of-Thirds": lambda b: -thirds_dist(b),
            "Edge-avoiding": lambda b: min_dist_to_edge(b),
            "Balanced": balanced_score,
            "Best Face": best_face_score,
        }

        score = score_funcs.get(strategy, score_funcs["Largest Person"])

        # Tie-breaking with confidence, then area
        best_person = sorted(boxes, key=lambda b: (score(b), b['conf'], area(b)), reverse=True)[0]

        return self._xyxy_to_xywh(best_person['bbox']), {'type': f'person_{strategy.lower().replace(" ", "_")}', 'conf': best_person['conf']}

    def _load_image_from_array(self, image_rgb: np.ndarray) -> tuple[np.ndarray, torch.Tensor]:
        """Prepares a NumPy image array for model inference.

        This helper function applies the standard torchvision transformations
        (ToTensor, Normalize) required by models like GroundingDINO.

        Args:
            image_rgb (np.ndarray): The input image as an RGB NumPy array.

        Returns:
            tuple[np.ndarray, torch.Tensor]: A tuple containing the original
            image and the transformed image tensor.
        """
        from torchvision import transforms
        transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        return image_rgb, transform(image_rgb)

    def _calculate_iou(self, box1: list, box2: list) -> float:
        """Calculates the Intersection over Union (IoU) of two bounding boxes.

        Args:
            box1 (list): The first bounding box in [x1, y1, x2, y2] format.
            box2 (list): The second bounding box in [x1, y1, x2, y2] format.

        Returns:
            float: The IoU value, between 0.0 and 1.0.
        """
        x1, y1, x2, y2 = box1
        x1_p, y1_p, x2_p, y2_p = box2
        inter_x1, inter_y1, inter_x2, inter_y2 = max(x1, x1_p), max(y1, y1_p), min(x2, x2_p), min(y2, y2_p)
        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        union_area = (x2 - x1) * (y2 - y1) + (x2_p - x1_p) * (y2_p - y1_p) - inter_area
        return inter_area / (union_area + 1e-6)

    def _box_contains(self, cb: list, ib: list) -> bool:
        """Checks if the containing box `cb` fully contains the inner box `ib`.

        Args:
            cb (list): The container box in [x1, y1, x2, y2] format.
            ib (list): The inner box in [x1, y1, x2, y2] format.

        Returns:
            bool: True if `cb` fully contains `ib`, False otherwise.
        """
        return cb[0] <= ib[0] and cb[1] <= ib[1] and cb[2] >= ib[2] and cb[3] >= ib[3]
    def _expand_face_to_body(self, face_bbox: list, img_shape: tuple) -> list[int]:
        """
        Expands a face bounding box to an estimated body bounding box.

        Args:
            face_bbox: The input face bounding box [x1, y1, x2, y2].
            img_shape: The shape of the image (h, w).

        Returns:
            The expanded body bounding box [x, y, w, h].
        """
        H, W, (x1, y1, x2, y2) = *img_shape[:2], *face_bbox
        w, h, cx = x2 - x1, y2 - y1, x1 + w / 2
        expansion_factors = self.config.seeding_face_to_body_expansion_factors
        new_w, new_h = min(W, w * expansion_factors[0]), min(H, h * expansion_factors[1])
        new_x1, new_y1 = max(0, cx - new_w / 2), max(0, y1 - h * expansion_factors[2])
        return [int(v) for v in [new_x1, new_y1, min(W, new_x1 + new_w) - new_x1, min(H, new_y1 + new_h) - new_y1]]

    def _final_fallback_box(self, img_shape: tuple) -> list[int]:
        """Returns a default fallback bounding box if no detections are found.

        Args:
            img_shape (tuple): The shape of the image (h, w, c).

        Returns:
            list[int]: The fallback bounding box in [x, y, w, h] format, based
            on a percentage of the image dimensions defined in the config.
        """
        h, w, _ = img_shape
        fallback_box = self.config.seeding_final_fallback_box
        return [int(w * fallback_box[0]), int(h * fallback_box[1]), int(w * fallback_box[2]), int(h * fallback_box[3])]
    def _xyxy_to_xywh(self, box: list) -> list[int]:
        """Converts a bounding box from [x1, y1, x2, y2] to [x, y, w, h] format.

        Args:
            box (list): The bounding box in [x1, y1, x2, y2] format.

        Returns:
            list[int]: The bounding box in [x, y, w, h] format.
        """
        x1, y1, x2, y2 = box; return [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]

    def _sam2_mask_for_bbox(self, frame_rgb_small: np.ndarray, bbox_xywh: list) -> Optional[np.ndarray]:
        """Generates a mask for a given bounding box using the SAM2 model.

        This method initializes the DAM4SAM tracker with the provided bounding
        box to get an initial segmentation mask. This is used for generating
        previews in the scene selection tab.

        Args:
            frame_rgb_small (np.ndarray): The (thumbnail) RGB image.
            bbox_xywh (list): The bounding box in [x, y, w, h] format.

        Returns:
            Optional[np.ndarray]: The generated mask as a NumPy array, or None
            on failure.
        """
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
    """Orchestrates the process of finding a subject and propagating their mask.

    This class combines model management, seed selection, and mask propagation.
    It's responsible for the end-to-end workflow of identifying a subject in
    one or more scenes and then tracking that subject's mask across all frames
    in those scenes.

    Attributes:
        params (AnalysisParameters): The analysis parameters for the current run.
        config (Config): The main application configuration.
        logger (AppLogger): The application logger.
        seed_selector (SeedSelector): The instance used for finding the initial
            subject bounding box.
        mask_propagator (MaskPropagator): The instance used for propagating the
            mask across frames.
    """
    def __init__(self, params: 'AnalysisParameters', progress_queue: Queue, cancel_event: threading.Event, config: 'Config',
                 frame_map: Optional[dict] = None, face_analyzer: Optional['FaceAnalysis'] = None,
                 reference_embedding: Optional[np.ndarray] = None, person_detector: Optional['PersonDetector'] = None,
                 thumbnail_manager: Optional['ThumbnailManager'] = None, niqe_metric: Optional[Callable] = None,
                 logger: Optional['AppLogger'] = None, face_landmarker: Optional['FaceLandmarker'] = None,
                 device: str = "cpu"):
        """Initializes the SubjectMasker.

        Args:
            params (AnalysisParameters): The analysis parameters.
            progress_queue (Queue): The queue for UI updates.
            cancel_event (threading.Event): The event to signal cancellation.
            config (Config): The main application configuration.
            frame_map (Optional[dict]): A map of original frame numbers to
                thumbnail filenames.
            face_analyzer (Optional[FaceAnalysis]): An initialized InsightFace
                FaceAnalysis model.
            reference_embedding (Optional[np.ndarray]): The embedding of the
                reference face.
            person_detector (Optional[PersonDetector]): An initialized YOLO
                PersonDetector.
            thumbnail_manager (Optional[ThumbnailManager]): The manager for the
                thumbnail cache.
            niqe_metric (Optional[Callable]): An initialized NIQE metric model.
            logger (Optional[AppLogger]): The application logger.
            face_landmarker (Optional[FaceLandmarker]): An initialized
                MediaPipe FaceLandmarker.
            device (str): The compute device ('cpu' or 'cuda').
        """
        self.params, self.config, self.progress_queue, self.cancel_event = params, config, progress_queue, cancel_event
        self.logger = logger or AppLogger(config=Config())
        self.frame_map = frame_map
        self.face_analyzer, self.reference_embedding, self.person_detector, self.face_landmarker = face_analyzer, reference_embedding, person_detector, face_landmarker
        self.dam_tracker, self.mask_dir, self.shots = None, None, []
        self._gdino, self._sam2_img = None, None
        self._device = device
        self.thumbnail_manager = thumbnail_manager if thumbnail_manager is not None else ThumbnailManager(self.logger, self.config)
        self.niqe_metric = niqe_metric
        self.initialize_models()
        self.seed_selector = SeedSelector(
            params=params,
            config=self.config,
            face_analyzer=face_analyzer,
            reference_embedding=reference_embedding,
            person_detector=person_detector,
            tracker=self.dam_tracker,
            gdino_model=self._gdino,
            logger=self.logger,
            device=self._device
        )
        self.mask_propagator = MaskPropagator(params, self.dam_tracker, cancel_event, progress_queue, config=self.config, logger=self.logger, device=self._device)

    def initialize_models(self):
        """Initializes the necessary AI models based on the current parameters.

        This method selectively loads models like GroundingDINO and DAM4SAM
        based on the chosen `primary_seed_strategy` to avoid loading
        unnecessary models and save memory. For example, in the "YOLO-only"
        mode, all other models are skipped.
        """
        primary_strategy = self.params.primary_seed_strategy
        if primary_strategy == "🧑‍🤝‍🧑 Find Prominent Person":
            self.logger.info("YOLO-only mode enabled for 'Find Prominent Person' strategy. Skipping other model loads.")
            self._gdino = None
            # Note: face_analyzer and face_landmarker are handled externally
            return

        # Continue with normal model loading for other strategies
        if self.params.enable_face_filter and self.face_analyzer is None:
            self.logger.warning("Face analyzer is not available but face filter is enabled.")

        # Defer tracker unless explicitly needed now
        if not getattr(self.params, "need_masks_now", False):
            return  # do not load DAM4SAM during pre-analysis
        if self.params.enable_subject_mask:
            self._initialize_tracker()

        text_based_seeding = ("📝 By Text" in primary_strategy or "Fallback" in primary_strategy)
        if text_based_seeding:
            self._init_grounder()
    def _init_grounder(self) -> bool:
        """Initializes the GroundingDINO model on demand.

        This internal helper ensures the GroundingDINO model is only loaded if
        it's actually needed by the current seeding strategy.

        Returns:
            bool: True if the model is initialized successfully, False otherwise.
        """
        if self._gdino is not None: return True
        # Skip grounder for strategies that don't need it
        if self.params.primary_seed_strategy == "🧑‍🤝‍🧑 Find Prominent Person":
            return False
        retry_params = (self.config.retry_max_attempts, tuple(self.config.retry_backoff_seconds))
        self._gdino = get_grounding_dino_model(
            gdino_config_path=self.params.gdino_config_path,
            gdino_checkpoint_path=self.params.gdino_checkpoint_path,
            models_path=str(self.config.models_dir),
            grounding_dino_url=self.config.grounding_dino_url,
            user_agent=self.config.user_agent,
            retry_params=retry_params,
            device=self._device,
            logger=self.logger
        )
        return self._gdino is not None

    def _initialize_tracker(self) -> bool:
        """Initializes the DAM4SAM tracker model on demand.

        This internal helper ensures the DAM4SAM tracker is only loaded if it's
        actually needed (i.e., when mask propagation is about to begin).

        Returns:
            bool: True if the model is initialized successfully, False otherwise.
        """
        if self.dam_tracker: return True

        try:
            model_urls_tuple = tuple(self.config.dam4sam_model_urls.items())
            retry_params = (self.config.retry_max_attempts, tuple(self.config.retry_backoff_seconds))

            self.logger.info(f"Initializing DAM4SAM tracker: {self.params.dam4sam_model_name}")
            self.dam_tracker = get_dam4sam_tracker(
                model_name=self.params.dam4sam_model_name,
                models_path=str(self.config.models_dir),
                model_urls_tuple=model_urls_tuple,
                user_agent=self.config.user_agent,
                retry_params=retry_params,
                logger=self.logger,
                device=self._device
            )

            if self.dam_tracker is None or self.dam_tracker == "failed":
                self.logger.error("DAM4SAM tracker initialization returned None/failed")
                return False

            self.logger.success("DAM4SAM tracker initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Exception during DAM4SAM tracker initialization: {e}", exc_info=True)
            return False

    def run_propagation(self, frames_dir: str, scenes_to_process: list['Scene'],
                        tracker: Optional['AdvancedProgressTracker'] = None) -> dict:
        """Runs the full mask propagation pipeline for a list of scenes.

        This method iterates through the provided scenes. For each scene, it
        loads the relevant frames, uses the `MaskPropagator` to generate masks,
        saves the masks to disk, and compiles metadata about the results.

        Args:
            frames_dir (str): The directory containing the extracted frames.
            scenes_to_process (list['Scene']): A list of `Scene` objects to
                process.
            tracker (Optional['AdvancedProgressTracker']): An optional progress
                tracker.

        Returns:
            dict: A dictionary containing metadata about the generated masks,
            keyed by filename.
        """
        self.mask_dir = Path(frames_dir) / "masks"
        self.mask_dir.mkdir(exist_ok=True)
        self.logger.info("Starting subject mask propagation...")
        # Ensure the tracker is initialized before proceeding. This is crucial for strategies
        # like "YOLO-only" that defer tracker loading.
        if not self._initialize_tracker():
            self.logger.error("DAM4SAM tracker could not be initialized; mask propagation failed.")
            # Return error structure instead of empty dict
            return {"error": "DAM4SAM tracker initialization failed", "completed": False}

        thumb_dir = Path(frames_dir) / "thumbs"
        mask_metadata, total_scenes = {}, len(scenes_to_process)
        progress_file = self.mask_dir.parent / "progress.json"
        for i, scene in enumerate(scenes_to_process):
            monitor_memory_usage(self.logger, self.config.monitoring_memory_warning_threshold_mb)
            with safe_resource_cleanup():
                if self.cancel_event.is_set(): break
                self.logger.info(f"Masking scene {i+1}/{total_scenes}", user_context={'shot_id': scene.shot_id, 'start_frame': scene.start_frame, 'end_frame': scene.end_frame})

                shot_frames_data = self._load_shot_frames(frames_dir, thumb_dir, scene.start_frame, scene.end_frame)
                if not shot_frames_data: continue

                if tracker:
                    tracker.set_stage(f"Scene {i+1}/{len(scenes_to_process)}", substage=f"{len(shot_frames_data)} frames")

                frame_numbers, small_images, dims = zip(*shot_frames_data)

                try:
                    best_frame_num = scene.best_frame
                    seed_idx_in_shot = frame_numbers.index(best_frame_num)
                except (ValueError, AttributeError):
                    self.logger.warning(f"Best frame {scene.best_frame} not found in loaded shot frames for {scene.shot_id}, skipping.")
                    continue

                bbox, seed_details = scene.seed_result.get('bbox'), scene.seed_result.get('details', {})
                if bbox is None:
                    for fn in frame_numbers:
                        if (fname := self.frame_map.get(fn)): mask_metadata[fname] = MaskingResult(error="Subject not found", shot_id=scene.shot_id).model_dump()
                    continue

                masks, areas, empties, errors = self.mask_propagator.propagate(small_images, seed_idx_in_shot, bbox, tracker=tracker)

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
                        mask_metadata[frame_fname_png] = MaskingResult(mask_path=str(mask_path), **result_args).model_dump()
                    else:
                        mask_metadata[frame_fname_png] = MaskingResult(mask_path=None, **result_args).model_dump()
        self.logger.success("Subject masking complete.")
        try:
            with (self.mask_dir.parent / "mask_metadata.json").open('w', encoding='utf-8') as f:
                json.dump(mask_metadata, f, indent=2)
            self.logger.info("Saved mask metadata.")
        except Exception as e:
            self.logger.error("Failed to save mask metadata", exc_info=True)
        return mask_metadata

    def _load_shot_frames(self, frames_dir: str, thumb_dir: Path, start: int, end: int) -> list[tuple[int, np.ndarray, tuple[int, int]]]:
        """Loads all thumbnail images for a given frame range (a shot).

        Args:
            frames_dir (str): The root directory of the extracted frames.
            thumb_dir (Path): The pre-calculated path to the thumbnails directory.
            start (int): The starting frame number of the shot.
            end (int): The ending frame number of the shot.

        Returns:
            list[tuple[int, np.ndarray, tuple[int, int]]]: A list of tuples,
            where each tuple contains the original frame number, the loaded
            thumbnail image as an RGB NumPy array, and the dimensions of the
            thumbnail.
        """
        frames = []
        if not self.frame_map:
            ext = ".webp" if self.params.thumbnails_only else ".png"
            self.frame_map = create_frame_map(Path(frames_dir), self.logger, ext=ext)

        for fn in sorted(fn for fn in self.frame_map if start <= fn < end):
            thumb_path = thumb_dir / f"{Path(self.frame_map[fn]).stem}.webp"
            thumb_img = self.thumbnail_manager.get(thumb_path)
            if thumb_img is None: continue
            frames.append((fn, thumb_img, thumb_img.shape[:2]))
        return frames

    def _select_best_frame_in_scene(self, scene: 'Scene', frames_dir: str):
        """Analyzes frames in a scene to find the best one for seeding.

        This pre-analysis step samples frames from a scene and scores them
        based on image quality (NIQE) and, if enabled, face similarity to the
        reference. The frame with the highest score is designated as the 'best
        frame' for that scene.

        Args:
            scene (Scene): The scene object to be updated.
            frames_dir (str): The directory containing the extracted frames.
        """
        if not self.params.pre_analysis_enabled:
            scene.best_frame, scene.seed_metrics = scene.start_frame, {'reason': 'pre-analysis disabled'}
            return

        thumb_dir = Path(frames_dir) / "thumbs"
        shot_frames = self._load_shot_frames(frames_dir, thumb_dir, scene.start_frame, scene.end_frame)
        if not shot_frames:
            scene.best_frame, scene.seed_metrics = scene.start_frame, {'reason': 'no frames loaded'}
            return
        candidates, scores = shot_frames[::max(1, self.params.pre_sample_nth)], []
        for frame_num, thumb_rgb, _ in candidates:
            niqe_score = 10.0
            if self.niqe_metric:
                img_tensor = (torch.from_numpy(thumb_rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0)
                with (torch.no_grad(), torch.amp.autocast('cuda', enabled=self._device == 'cuda')):
                    niqe_score = float(self.niqe_metric(img_tensor.to(self.niqe_metric.device)))
            face_sim = 0.0
            if self.face_analyzer and self.reference_embedding is not None:
                faces = self.face_analyzer.get(cv2.cvtColor(thumb_rgb, cv2.COLOR_RGB2BGR))
                if faces: face_sim = np.dot(max(faces, key=lambda x: x.det_score).normed_embedding, self.reference_embedding)
            scores.append((10 - niqe_score) + (face_sim * 10))
        if not scores:
            best_local_idx = 0
            scene.best_frame = scene.start_frame # Fallback
            scene.seed_metrics = {'reason': 'pre-analysis failed, no scores', 'score': 0}
            return
        best_local_idx = int(np.argmax(scores))
        scene.best_frame = candidates[best_local_idx][0]
        scene.seed_metrics = {'reason': 'pre-analysis complete', 'score': max(scores), 'best_niqe': niqe_score, 'best_face_sim': face_sim}

    def get_seed_for_frame(self, frame_rgb: np.ndarray, seed_config: dict = None, scene: Optional['Scene'] = None) -> tuple[Optional[list], dict]:
        """
        Gets a seed bounding box for a single frame.

        Args:
            frame_rgb: The RGB image of the frame.
            seed_config: Optional dictionary of seed configuration overrides.
            scene: The scene object.

        Returns:
            A tuple containing the seed bounding box and details about the selection.
        """
        if isinstance(seed_config, dict) and seed_config.get('manual_bbox_xywh'):
            return seed_config['manual_bbox_xywh'], {'type': seed_config.get('seed_type', 'manual')}

        # Only initialize grounder for strategies that need it
        primary_strategy = getattr(self.params, 'primary_seed_strategy', 'Automatic')
        if primary_strategy != "🧑‍🤝‍🧑 Find Prominent Person":
            self._init_grounder()

        # Pre-run YOLO detection and store all results on the scene
        if scene is not None and self.seed_selector.person_detector:
            scene.yolo_detections = self.seed_selector._get_yolo_boxes(frame_rgb, scene=None)

        self._initialize_tracker()
        return self.seed_selector.select_seed(frame_rgb, current_params=seed_config, scene=scene)
    def get_mask_for_bbox(self, frame_rgb_small: np.ndarray, bbox_xywh: list) -> Optional[np.ndarray]:
        """Gets a mask for a specific bounding box on a frame.

        This is a convenience method that wraps the `_sam2_mask_for_bbox`
        method of the `SeedSelector`.

        Args:
            frame_rgb_small (np.ndarray): The (thumbnail) RGB image.
            bbox_xywh (list): The bounding box in [x, y, w, h] format.

        Returns:
            Optional[np.ndarray]: The generated mask as a NumPy array.
        """
        return self.seed_selector._sam2_mask_for_bbox(frame_rgb_small, bbox_xywh)
    def draw_bbox(self, img_rgb: np.ndarray, xywh: list, color: Optional[tuple] = None,
                  thickness: Optional[int] = None, label: Optional[str] = None) -> np.ndarray:
        """Draws a bounding box on an image.

        Args:
            img_rgb (np.ndarray): The input RGB image.
            xywh (list): The bounding box in [x, y, w, h] format.
            color (Optional[tuple]): The BGR color of the box. Defaults to
                the value in the config.
            thickness (Optional[int]): The thickness of the box lines. Defaults
                to the value in the config.
            label (Optional[str]): An optional text label to draw.

        Returns:
            np.ndarray: A copy of the image with the bounding box drawn on it.
        """
        color = color or tuple(self.config.visualization_bbox_color)
        thickness = thickness or self.config.visualization_bbox_thickness
        x, y, w, h = map(int, xywh or [0, 0, 0, 0])
        img_out = img_rgb.copy()
        cv2.rectangle(img_out, (x, y), (x + w, y + h), color, thickness)
        if label:
            # Put the label inside the box at the top-left
            font_scale = 1
            font = cv2.FONT_HERSHEY_SIMPLEX
            (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            text_x = x + 5
            text_y = y + text_height + 5
            # Add a filled rectangle as a background for the text
            cv2.rectangle(img_out, (x, y), (x + text_width + 10, y + text_height + 10), color, -1)
            cv2.putText(img_out, label, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)
        return img_out

# --- PIPELINES ---

class ExtractionPipeline(Pipeline):
    """Pipeline for extracting frames or thumbnails from a video or image folder.

    This pipeline handles the initial step of the workflow: converting a video
    or a folder of images into a standardized set of frames (or thumbnails)
    and associated metadata.

    It orchestrates video downloading, scene detection (if enabled), and frame
    extraction using FFmpeg.
    """
    def run(self, tracker: Optional['AdvancedProgressTracker'] = None) -> dict:
        """Executes the extraction process.

        This method is the main entry point for the pipeline. It handles both
        video files and folders of images. For videos, it downloads from
        YouTube if necessary, runs scene detection, and then uses FFmpeg to
        extract frames. For image folders, it generates thumbnails directly.

        Args:
            tracker (Optional['AdvancedProgressTracker']): An optional progress
                tracker to provide UI updates.

        Returns:
            dict: A dictionary containing the "output_dir" path, the "video_path"
            (if applicable), and a "done" status boolean.
        """
        source_p = Path(self.params.source_path)
        is_folder = is_image_folder(source_p)

        if is_folder:
            output_dir = Path(self.params.output_folder) if self.params.output_folder else Path(self.config.downloads_dir) / source_p.name
            output_dir.mkdir(exist_ok=True, parents=True)

            params_dict = self.params.model_dump()
            params_dict['output_folder'] = str(output_dir)
            params_dict['video_path'] = "" # No video path for image folders

            run_cfg_path = output_dir / "run_config.json"
            try:
                with run_cfg_path.open('w', encoding='utf-8') as f:
                    json.dump(_to_json_safe(params_dict), f, indent=2)
            except OSError as e:
                self.logger.warning(f"Could not write run config to {run_cfg_path}: {e}")

            self.logger.info(f"Processing image folder: {source_p.name}")
            images = list_images(source_p, self.config)
            if not images:
                self.logger.warning("No images found in the specified folder.")
                return {"done": False, "log": "No images found."}

            make_photo_thumbs(images, output_dir, self.params, self.config, self.logger, tracker=tracker)

            # For image folders, each image is a "scene" of one frame.
            # We create a synthetic scenes.json to maintain compatibility.
            num_images = len(images)
            scenes = [[i, i] for i in range(1, num_images + 1)]
            with (output_dir / "scenes.json").open('w', encoding='utf-8') as f:
                json.dump(scenes, f)

            return {"done": True, "output_dir": str(output_dir), "video_path": ""}
        else:
            self.logger.info("Preparing video source...")
            vid_manager = VideoManager(self.params.source_path, self.config, self.params.max_resolution)
            video_path = Path(vid_manager.prepare_video(self.logger))
            output_dir = Path(self.params.output_folder) if self.params.output_folder else Path(self.config.downloads_dir) / video_path.stem
            output_dir.mkdir(exist_ok=True, parents=True)

            params_dict = self.params.model_dump()
            params_dict['output_folder'] = str(output_dir)
            params_dict['video_path'] = str(video_path)

            run_cfg_path = output_dir / "run_config.json"
            try:
                with run_cfg_path.open('w', encoding='utf-8') as f:
                    json.dump(_to_json_safe(params_dict), f, indent=2)
            except OSError as e:
                self.logger.warning(f"Could not write run config to {run_cfg_path}: {e}")

            self.logger.info("Video ready", user_context={'path': sanitize_filename(video_path.name, self.config)})
            video_info = VideoManager.get_video_info(video_path)

            if tracker:
                totals = estimate_totals(self.params, video_info, None)
                tracker.start(totals["extraction"], desc="Extracting frames")

            if self.params.scene_detect:
                self._run_scene_detection(video_path, output_dir)

            self._run_ffmpeg(video_path, output_dir, video_info, tracker=tracker)

            if self.cancel_event.is_set():
                self.logger.info("Extraction cancelled by user.")
                if tracker: tracker.done_stage("Extraction cancelled")
                return {"done": False, "log": "Extraction cancelled"}

            if tracker: tracker.done_stage("Extraction complete")
            self.logger.success("Extraction complete.")
            return {"done": True, "output_dir": str(output_dir), "video_path": str(video_path)}

    def _run_scene_detection(self, video_path: str, output_dir: Path) -> list:
        """Wrapper for the `run_scene_detection` utility function.

        Args:
            video_path (str): The path to the video file.
            output_dir (Path): The directory to save the `scenes.json` file in.

        Returns:
            list: The list of detected scenes.
        """
        return run_scene_detection(video_path, output_dir, self.logger)
    def _run_ffmpeg(self, video_path: str, output_dir: Path, video_info: dict, tracker: Optional['AdvancedProgressTracker'] = None):
        """Wrapper for the `run_ffmpeg_extraction` utility function.

        Args:
            video_path (str): The path to the source video file.
            output_dir (Path): The directory to save the extracted frames.
            video_info (dict): A dictionary with video metadata.
            tracker (Optional['AdvancedProgressTracker']): An optional progress
                tracker.
        """
        return run_ffmpeg_extraction(video_path, output_dir, video_info, self.params, self.progress_queue, self.cancel_event, self.logger, self.config, tracker=tracker)

class EnhancedExtractionPipeline(ExtractionPipeline):
    """An `ExtractionPipeline` subclass that includes retry logic.

    This class decorates the `run` method of the base `ExtractionPipeline`
    with a retry mechanism provided by the `ErrorHandler`. This makes the
    extraction process more resilient to transient errors, such as network
    issues during video downloads.
    """
    def __init__(self, config: 'Config', logger: 'AppLogger', params: 'AnalysisParameters',
                 progress_queue: Queue, cancel_event: threading.Event):
        """
        Initializes the EnhancedExtractionPipeline.

        Args:
            config: The application configuration.
            logger: The application logger.
            params: The parameters for the analysis run.
            progress_queue: The queue for UI progress updates.
            cancel_event: The event to signal cancellation.
        """
        super().__init__(config, logger, params, progress_queue, cancel_event)
        self.error_handler = ErrorHandler(self.logger, self.config.retry_max_attempts, self.config.retry_backoff_seconds)
        self.run = self.error_handler.with_retry()(self.run)

class AnalysisPipeline(Pipeline):
    """Pipeline for running the main analysis, from propagation to metric calculation.

    This class is responsible for the core analysis phase of the workflow. It
    is divided into two main stages:
    1. `run_full_analysis`: Handles mask propagation for selected scenes.
    2. `run_analysis_only`: Handles frame-by-frame metric calculation.

    It manages the initialization of analysis-specific models and orchestrates
    the processing of frames in parallel.
    """
    def __init__(self, config: 'Config', logger: 'AppLogger', params: 'AnalysisParameters',
                 progress_queue: Queue, cancel_event: threading.Event,
                 thumbnail_manager: 'ThumbnailManager'):
        """
        Initializes the AnalysisPipeline.

        Args:
            config: The application configuration.
            logger: The application logger.
            params: The parameters for the analysis run.
            progress_queue: The queue for UI progress updates.
            cancel_event: The event to signal cancellation.
            thumbnail_manager: The manager for the thumbnail cache.
        """
        super().__init__(config, logger, params, progress_queue, cancel_event)
        self.output_dir = Path(self.params.output_folder)
        self.db = Database(self.output_dir / "metadata.db")
        self.thumb_dir = self.output_dir / "thumbs"
        self.masks_dir = self.output_dir / "masks"
        self.processing_lock = threading.Lock()
        self.face_analyzer, self.reference_embedding, self.mask_metadata, self.face_landmarker = None, None, {}, None
        self.scene_map, self.niqe_metric = {}, None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.thumbnail_manager = thumbnail_manager

    def _initialize_niqe_metric(self):
        """Initializes the NIQE metric model if the `pyiqa` package is available."""
        if self.niqe_metric is None:
            try:
                import pyiqa
                self.niqe_metric = pyiqa.create_metric('niqe', device=self.device)
                self.logger.info("NIQE metric initialized successfully")
            except ImportError:
                self.logger.warning("pyiqa is not installed, NIQE metric is unavailable.")
            except Exception as e:
                self.logger.warning("Failed to initialize NIQE metric", extra={'error': e})

    def run_full_analysis(self, scenes_to_process: list['Scene'], tracker: Optional['AdvancedProgressTracker'] = None) -> dict:
        """
        Runs the mask propagation stage of the analysis.

        Args:
            scenes_to_process: A list of Scene objects to process.
            tracker: An optional progress tracker.

        Returns:
            A dictionary indicating the result and output directory.
        """
        is_folder_mode = not self.params.video_path

        if is_folder_mode:
            return self._run_image_folder_analysis(tracker=tracker)
        else:
            # This is the existing video analysis pipeline.
            try:
                # Check for existing progress
                progress_file = self.output_dir / "progress.json"
                if progress_file.exists() and self.params.resume:
                    with open(progress_file) as f:
                        progress_data = json.load(f)
                    # Resume from last completed scene
                    scenes_to_process = self._filter_completed_scenes(scenes_to_process, progress_data)
                # Connect to the database and create tables
                self.db.connect()
                self.db.create_tables()
                if not self.params.resume:
                    self.db.clear_metadata()

                self.scene_map = {s.shot_id: s for s in scenes_to_process}
                self.logger.info("Initializing Models")
                models = initialize_analysis_models(self.params, self.config, self.logger)
                self.face_analyzer = models['face_analyzer']
                self.reference_embedding = models['ref_emb']
                self.face_landmarker = models['face_landmarker']
                person_detector = models['person_detector']

                if self.face_analyzer and self.params.face_ref_img_path:
                    self._process_reference_face()

                # When starting propagation, flip the flag on the params you pass into SubjectMasker
                self.params.need_masks_now = True
                self.params.enable_subject_mask = True
                ext = ".webp" if self.params.thumbnails_only else ".png"
                masker = SubjectMasker(self.params, self.progress_queue, self.cancel_event, self.config, create_frame_map(self.output_dir, self.logger, ext=ext),
                                       self.face_analyzer, self.reference_embedding, person_detector, thumbnail_manager=self.thumbnail_manager,
                                       niqe_metric=self.niqe_metric, logger=self.logger, face_landmarker=self.face_landmarker,
                                       device=models['device'])
                for scene in scenes_to_process:
                    if self.cancel_event.is_set():
                        self.logger.info("Propagation cancelled by user.")
                        break
                    self.mask_metadata.update(masker.run_propagation(str(self.output_dir), [scene], tracker=tracker))
                    self._save_progress(scene, progress_file)

                if self.cancel_event.is_set():
                    self.logger.info("Propagation cancelled by user.")
                    return {"log": "Propagation cancelled.", "done": False}

                self.logger.success("Propagation complete.", extra={'output_dir': self.output_dir})
                return {"done": True, "output_dir": str(self.output_dir)}
            except Exception as e:
                self.logger.error("Propagation pipeline failed", component="analysis", exc_info=True, extra={'error': str(e)})
                return {"error": str(e), "done": False}

    def run_analysis_only(self, scenes_to_process: list['Scene'], tracker: Optional['AdvancedProgressTracker'] = None) -> dict:
        """
        Runs the frame-by-frame metric calculation stage of the analysis.

        This assumes that mask propagation (if enabled) has already completed.

        Args:
            scenes_to_process: A list of Scene objects to process.
            tracker: An optional progress tracker.

        Returns:
            A dictionary indicating the result and paths to output files.
        """
        try:
            self.db.connect()
            self.db.create_tables()
            if not self.params.resume:
                self.db.clear_metadata()
            self.scene_map = {s.shot_id: s for s in scenes_to_process}
            self.logger.info("Initializing Models for Analysis")
            models = initialize_analysis_models(self.params, self.config, self.logger)
            self.face_analyzer = models['face_analyzer']
            self.reference_embedding = models['ref_emb']
            self.face_landmarker = models['face_landmarker']
            if self.face_analyzer and self.params.face_ref_img_path: self._process_reference_face()
            mask_metadata_path = self.output_dir / "mask_metadata.json"
            if mask_metadata_path.exists():
                with open(mask_metadata_path, 'r', encoding='utf-8') as f: self.mask_metadata = json.load(f)
            else: self.mask_metadata = {}
            if tracker: tracker.set_stage("Analyzing frames")
            self._initialize_niqe_metric()
            metrics_to_compute = {
                'quality': self.params.compute_quality_score,
                'sharpness': self.params.compute_sharpness,
                'edge_strength': self.params.compute_edge_strength,
                'contrast': self.params.compute_contrast,
                'brightness': self.params.compute_brightness,
                'entropy': self.params.compute_entropy,
                'eyes_open': self.params.compute_eyes_open,
                'yaw': self.params.compute_yaw,
                'pitch': self.params.compute_pitch,
            }
            self._run_analysis_loop(scenes_to_process, metrics_to_compute, tracker=tracker)
            if self.cancel_event.is_set(): return {"log": "Analysis cancelled.", "done": False}
            self.logger.success("Analysis complete.", extra={'output_dir': self.output_dir})
            return {"done": True, "output_dir": str(self.output_dir)}
        except Exception as e:
            self.logger.error("Analysis pipeline failed", exc_info=True, extra={'error': str(e)})
            return {"error": str(e), "done": False}

    def _filter_completed_scenes(self, scenes: list['Scene'], progress_data: dict) -> list['Scene']:
        """Filters out scenes that have already been processed in a previous run.

        This is used by the "Resume" functionality. It checks a progress file
        to see which scenes were completed in a previous run and removes them
        from the list of scenes to be processed.

        Args:
            scenes (list['Scene']): The full list of scenes for the current run.
            progress_data (dict): The data loaded from a `progress.json` file.

        Returns:
            list['Scene']: A filtered list of scenes that still need to be
            processed.
        """
        completed_scenes = progress_data.get("completed_scenes", [])
        return [s for s in scenes if s.shot_id not in completed_scenes]

    def _save_progress(self, current_scene: 'Scene', progress_file: Path):
        """Saves the ID of the most recently completed scene to a progress file.

        This supports the "Resume" functionality by keeping track of which
        scenes have been fully processed.

        Args:
            current_scene (Scene): The scene that was just completed.
            progress_file (Path): The path to the `progress.json` file.
        """
        progress_data = {"completed_scenes": []}
        if progress_file.exists():
            with open(progress_file) as f:
                progress_data = json.load(f)
        progress_data["completed_scenes"].append(current_scene.shot_id)
        with open(progress_file, 'w') as f:
            json.dump(progress_data, f)
    def _process_reference_face(self):
        """Loads and processes the reference face image to get its embedding.

        Raises:
            FileNotFoundError: If the reference face image is not found.
            ValueError: If the image cannot be read or no face is detected.
        """
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

    def _run_image_folder_analysis(self, tracker: Optional['AdvancedProgressTracker'] = None) -> dict:
        """Runs the analysis pipeline specifically for a folder of images.

        This method is a placeholder for the logic that will handle the
        analysis of pre-existing images, as opposed to frames extracted from a
        video.

        Args:
            tracker (Optional['AdvancedProgressTracker']): An optional progress
                tracker.

        Returns:
            dict: A dictionary indicating the result of the analysis.
        """
        self.logger.info("Starting image folder analysis...")

        # Stage 1: Pre-filter on thumbnails
        self.logger.info("Running pre-filter on thumbnails...")
        # TODO: Implement pre-filtering logic here.

        # Stage 2: Full analysis on kept images
        self.logger.info("Running full analysis on kept images...")
        # TODO: Implement full analysis logic here.

        self.logger.success("Image folder analysis complete.")
        metadata_path = self.output_dir / "metadata.db"
        return {"done": True, "metadata_path": str(metadata_path), "output_dir": str(self.output_dir)}

    def _run_analysis_loop(self, scenes_to_process: list['Scene'], metrics_to_compute: dict, tracker: Optional['AdvancedProgressTracker'] = None):
        """Orchestrates the parallel processing of frames for metric calculation.

        This method sets up a `ThreadPoolExecutor` to process frames in
        parallel batches. It gathers all frames from the scenes that need to be
        processed, divides them into batches, and submits them to the thread
        pool.

        Args:
            scenes_to_process (list['Scene']): A list of scenes whose frames
                need to be analyzed.
            metrics_to_compute (dict): A dictionary of booleans indicating
                which metrics to compute.
            tracker (Optional['AdvancedProgressTracker']): An optional progress
                tracker.
        """
        frame_map = self._create_frame_map()
        all_frame_nums_to_process = {fn for scene in scenes_to_process for fn in range(scene.start_frame, scene.end_frame) if fn in frame_map}
        image_files_to_process = [self.thumb_dir / frame_map[fn] for fn in sorted(list(all_frame_nums_to_process)) if frame_map.get(fn)]
        self.logger.info(f"Analyzing {len(image_files_to_process)} frames")
        num_workers = 1 if self.params.disable_parallel else min(os.cpu_count() or 4, self.config.analysis_default_workers)
        batch_size = self.config.analysis_default_batch_size
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            batches = [image_files_to_process[i:i + batch_size] for i in range(0, len(image_files_to_process), batch_size)]
            futures = [executor.submit(self._process_batch, batch, metrics_to_compute) for batch in batches]
            for future in as_completed(futures):
                monitor_memory_usage(self.logger, self.config.monitoring.memory_warning_threshold_mb)
                if self.cancel_event.is_set():
                    for f in futures: f.cancel()
                    break
                try:
                    num_processed = future.result()
                    if tracker and num_processed:
                        tracker.step(num_processed)
                except Exception as e:
                    self.logger.error(f"Error processing batch future: {e}")

    def _process_batch(self, batch_paths: list[Path], metrics_to_compute: dict) -> int:
        """Processes a batch of frames in a single thread.

        This is the target function for the `ThreadPoolExecutor`. It iterates
        through a list of thumbnail paths and calls `_process_single_frame`
        for each one.

        Args:
            batch_paths (list[Path]): A list of `Path` objects for the thumbnail
                images to process.
            metrics_to_compute (dict): A dictionary of booleans indicating
                which metrics to compute.

        Returns:
            int: The number of frames processed in the batch.
        """
        for path in batch_paths:
            self._process_single_frame(path, metrics_to_compute)
        return len(batch_paths)

    def _process_single_frame(self, thumb_path: Path, metrics_to_compute: dict):
        """Calculates and records all metrics for a single frame.

        This is the core of the analysis loop. For a single frame, it:
        1. Loads the thumbnail image.
        2. Retrieves associated mask metadata.
        3. Performs face similarity analysis (if enabled).
        4. Calculates all selected quality metrics.
        5. Computes the perceptual hash (pHash).
        6. Appends the resulting metadata dictionary to the `metadata.jsonl` file.

        Args:
            thumb_path (Path): The path to the thumbnail image to process.
            metrics_to_compute (dict): A dictionary of booleans indicating
                which metrics to compute.
        """
        if self.cancel_event.is_set(): return
        if not (frame_num_match := re.search(r'frame_(\d+)', thumb_path.name)): return
        log_context = {'file': thumb_path.name}
        try:
            thumb_image_rgb = self.thumbnail_manager.get(thumb_path)
            if thumb_image_rgb is None: raise ValueError("Could not read thumbnail.")
            frame, base_filename = Frame(thumb_image_rgb, -1), thumb_path.name
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
                enable_niqe=(self.niqe_metric is not None and self.params.compute_niqe)
            )
            face_bbox = None
            if self.params.compute_face_sim and self.face_analyzer:
                face_bbox = self._analyze_face_similarity(frame, thumb_image_rgb)

            # Only call calculate_quality_metrics if at least one metric is requested
            if any(metrics_to_compute.values()) or self.params.compute_niqe:
                 frame.calculate_quality_metrics(
                    thumb_image_rgb, quality_conf, self.logger,
                    mask=mask_thumb, niqe_metric=self.niqe_metric, main_config=self.config,
                    face_landmarker=self.face_landmarker, face_bbox=face_bbox,
                    metrics_to_compute=metrics_to_compute
                )

            meta = {"filename": base_filename, "metrics": frame.metrics.model_dump()}
            if self.params.compute_face_sim:
                if frame.face_similarity_score is not None:
                    meta["face_sim"] = frame.face_similarity_score
                if frame.max_face_confidence is not None:
                    meta["face_conf"] = frame.max_face_confidence

            if self.params.compute_subject_mask_area:
                meta.update(mask_meta)

            if meta.get("shot_id") is not None and (scene := self.scene_map.get(meta["shot_id"])) and scene.seed_metrics:
                meta['seed_face_sim'] = scene.seed_metrics.get('best_face_sim')

            if self.params.compute_phash and imagehash:
                meta['phash'] = str(imagehash.phash(rgb_to_pil(thumb_image_rgb)))

            # This is to ensure the UI slider is respected
            if 'dedup_thresh' in self.params.__dict__:
                meta['dedup_thresh'] = self.params.dedup_thresh

            if frame.error: meta["error"] = frame.error
            if meta.get("mask_path"): meta["mask_path"] = Path(meta["mask_path"]).name

            # Sanitize the dictionary to ensure all values are JSON-serializable
            meta = _to_json_safe(meta)

            self.db.insert_metadata(meta)
        except Exception as e:
            self.logger.critical("Error processing frame", exc_info=True, extra={**log_context, 'error': e})
            self.db.insert_metadata({"filename": thumb_path.name, "error": f"processing_failed: {e}"})

    def _analyze_face_similarity(self, frame: 'Frame', image_rgb: np.ndarray) -> Optional[list[int]]:
        """Analyzes face similarity for a frame and updates the frame object.

        Args:
            frame (Frame): The `Frame` object to update with the similarity
                score.
            image_rgb (np.ndarray): The RGB image data for the frame.

        Returns:
            Optional[list[int]]: The bounding box of the best-detected face, or
            None if no face was found.
        """
        face_bbox = None
        try:
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            with self.processing_lock:
                faces = self.face_analyzer.get(image_bgr)
            if faces:
                best_face = max(faces, key=lambda x: x.det_score)
                face_bbox = best_face.bbox.astype(int)
                if self.params.enable_face_filter and self.reference_embedding is not None:
                    distance = 1 - np.dot(best_face.normed_embedding, self.reference_embedding)
                    frame.face_similarity_score, frame.max_face_confidence = 1.0 - float(distance), float(best_face.det_score)
        except Exception as e:
            frame.error = f"Face similarity failed: {e}"
            if "out of memory" in str(e) and torch.cuda.is_available():
                torch.cuda.empty_cache()
        return face_bbox

# --- FILTERING & SCENE LOGIC ---

def load_and_prep_filter_data(output_dir: str, get_all_filter_keys: Callable, config: 'Config') -> tuple[list, dict]:
    """
    Loads frame metadata from a SQLite database and prepares it for the filtering UI.

    This function connects to the `metadata.db` SQLite database, loads all
    frame metadata, and then processes it. It extracts the values for each
    filterable metric, calculates histogram data for visualization, and
    returns all the data in a structured format.

    Args:
        output_dir: The path to the output directory containing the database.
        get_all_filter_keys: A function that returns a list of all metric keys to process.
        config: The main application configuration.

    Returns:
        tuple[list, dict]: A tuple containing:
        - A list of all frame data dictionaries.
        - A dictionary where keys are metric names and values are either a list
          of all values for that metric or a tuple of (counts, bins) for a
          histogram.
    """
    db_path = Path(output_dir) / "metadata.db"
    if not db_path.exists():
        return [], {}

    db = Database(db_path)
    all_frames = db.load_all_metadata()
    db.close()

    metric_values = {}
    metric_configs = {
        'quality_score': {'path': ("metrics", "quality_score"), 'range': (0, 100)},
        'yaw': {'path': ("metrics", "yaw"), 'range': (config.filter_default_yaw['min'], config.filter_default_yaw['max'])},
        'pitch': {'path': ("metrics", "pitch"), 'range': (config.filter_default_pitch['min'], config.filter_default_pitch['max'])},
        'eyes_open': {'path': ("metrics", "eyes_open"), 'range': (0, 1)},
        'face_sim': {'path': ("face_sim",), 'range': (0, 1)},
    }
    # Add other metrics with a default range
    for k in get_all_filter_keys():
        if k not in metric_configs:
            metric_configs[k] = {'path': (k,), 'alt_path': ("metrics", f"{k}_score"), 'range': (0, 100)}

    for k in get_all_filter_keys():
        config = metric_configs.get(k)
        if not config: continue

        path = config.get('path')
        alt_path = config.get('alt_path')

        values = []
        for f in all_frames:
            val = None
            if path:
                if len(path) == 1: val = f.get(path[0])
                else: val = f.get(path[0], {}).get(path[1])

            if val is None and alt_path:
                if len(alt_path) == 1: val = f.get(alt_path[0])
                else: val = f.get(alt_path[0], {}).get(alt_path[1])

            if val is not None:
                values.append(val)

        values = np.asarray(values, dtype=float)

        if values.size > 0:
            hist_range = config.get('range', (0, 100))
            counts, bins = np.histogram(values, bins=50, range=hist_range)
            metric_values[k] = values.tolist()
            metric_values[f"{k}_hist"] = (counts.tolist(), bins.tolist())

    return all_frames, metric_values

def build_all_metric_svgs(per_metric_values: dict, get_all_filter_keys: Callable, logger: 'AppLogger') -> dict:
    """Generates SVG histograms for all available metrics.

    This function iterates through all filterable metrics, finds their
    pre-calculated histogram data, and calls `histogram_svg` to generate an
    SVG plot for each one.

    Args:
        per_metric_values (dict): The dictionary of metric data from
            `load_and_prep_filter_data`.
        get_all_filter_keys (Callable): A function that returns a list of all
            metric keys.
        logger (AppLogger): The application logger.

    Returns:
        dict: A dictionary mapping metric names to their SVG histogram strings.
    """
    svgs = {}
    for k in get_all_filter_keys:
        if (h := per_metric_values.get(f"{k}_hist")): svgs[k] = histogram_svg(h, title="", logger=logger)
    return svgs

def histogram_svg(hist_data: tuple, title: str = "", logger: Optional['AppLogger'] = None) -> str:
    """Creates an SVG histogram from histogram data using Matplotlib.

    Args:
        hist_data (tuple): A tuple of (counts, bins) from `np.histogram`.
        title (str): An optional title for the plot. Defaults to "".
        logger (Optional[AppLogger]): The application logger. Defaults to None.

    Returns:
        str: A string containing the SVG data for the histogram plot. Returns
        an error message SVG if Matplotlib is not installed or plotting fails.
    """
    if not plt:
        return """<svg width="100" height="20" xmlns="http://www.w3.org/2000/svg"><text x="5" y="15" font-family="sans-serif" font-size="10" fill="orange">Matplotlib missing</text></svg>"""
    if not hist_data: return ""
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
        return """<svg width="100" height="20" xmlns="http://www.w3.org/2000/svg"><text x="5" y="15" font-family="sans-serif" font-size="10" fill="red">Plotting failed</text></svg>"""

def _extract_metric_arrays(all_frames_data: list[dict], config: 'Config') -> dict:
    """Extracts metric data from frame dictionaries into NumPy arrays.

    This function is a performance optimization for the filtering process. It
    converts lists of metric values stored in dictionaries into contiguous
    NumPy arrays, which enables fast, vectorized filtering operations instead
    of slow Python loops.

    Args:
        all_frames_data (list[dict]): A list of frame metadata dictionaries.
        config (Config): The main application configuration.

    Returns:
        dict: A dictionary mapping metric names (e.g., "quality_score") to
        their corresponding NumPy arrays.
    """
    quality_weights_keys = [k.replace('quality_weights_', '') for k in config.model_dump().keys() if k.startswith('quality_weights_')]
    metric_sources = {
        **{k: ("metrics", f"{k}_score") for k in quality_weights_keys},
        "quality_score": ("metrics", "quality_score"),
        "face_sim": ("face_sim",),
        "mask_area_pct": ("mask_area_pct",),
        "eyes_open": ("metrics", "eyes_open"),
        "yaw": ("metrics", "yaw"),
        "pitch": ("metrics", "pitch"),
    }
    metric_arrays = {}
    for key, path in metric_sources.items():
        if len(path) == 1:
            metric_arrays[key] = np.array([f.get(path[0], np.nan) for f in all_frames_data], dtype=np.float32)
        else:
            metric_arrays[key] = np.array([f.get(path[0], {}).get(path[1], np.nan) for f in all_frames_data], dtype=np.float32)
    return metric_arrays


def _apply_deduplication_filter(all_frames_data: list[dict], filters: dict, thumbnail_manager: 'ThumbnailManager',
                                config: 'Config', output_dir: str) -> tuple[np.ndarray, defaultdict]:
    """Applies deduplication filters to the set of frames.

    This function supports multiple deduplication methods, including perceptual
    hash (pHash), SSIM, LPIPS, and a two-stage pHash+LPIPS approach. It
    identifies visually similar frames and keeps only the one with the highest
    quality score, marking the others for rejection.

    Args:
        all_frames_data (list[dict]): A list of frame metadata dictionaries.
        filters (dict): A dictionary of active filter settings.
        thumbnail_manager (ThumbnailManager): The thumbnail cache manager.
        config (Config): The main application configuration.
        output_dir (str): The session's output directory.

    Returns:
        tuple[np.ndarray, defaultdict]: A tuple containing:
        - A boolean NumPy array (the mask of frames to keep).
        - A dictionary mapping filenames to their rejection reasons.
    """
    num_frames = len(all_frames_data)
    filenames = [f['filename'] for f in all_frames_data]
    dedup_mask = np.ones(num_frames, dtype=bool)
    reasons = defaultdict(list)
    dedup_method = filters.get("dedup_method", "pHash")

    if filters.get("enable_dedup"):
        if dedup_method == "pHash" and imagehash and filters.get("dedup_thresh", -1) != -1:
            sorted_indices = sorted(range(num_frames), key=lambda i: filenames[i])
            hashes = {i: imagehash.hex_to_hash(all_frames_data[i]['phash']) for i in range(num_frames) if 'phash' in all_frames_data[i]}
            kept_hashes = {}  # Store hashes of frames that are kept
            for i in sorted_indices:
                if i not in hashes:
                    continue
                is_duplicate = False
                for kept_idx, kept_hash in kept_hashes.items():
                    if (hashes[i] - kept_hash) <= filters.get("dedup_thresh", 5):
                        is_duplicate = True
                        # Compare quality scores to decide which to keep
                        if all_frames_data[i].get('metrics', {}).get('quality_score', 0) > all_frames_data[kept_idx].get('metrics', {}).get('quality_score', 0):
                            # Mark the previously kept frame as a duplicate
                            if dedup_mask[kept_idx]:
                                reasons[filenames[kept_idx]].append('duplicate')
                            dedup_mask[kept_idx] = False
                            # Replace with the current, higher-quality frame
                            del kept_hashes[kept_idx]
                            kept_hashes[i] = hashes[i]
                        else:
                            # Mark the current frame as a duplicate
                            if dedup_mask[i]:
                                reasons[filenames[i]].append('duplicate')
                            dedup_mask[i] = False
                        break  # Move to the next frame
                if not is_duplicate:
                    kept_hashes[i] = hashes[i]
        elif dedup_method == "SSIM" and thumbnail_manager:
            dedup_mask, reasons = apply_ssim_dedup(all_frames_data, filters, dedup_mask, reasons, thumbnail_manager, config, output_dir)
        elif dedup_method == "LPIPS" and thumbnail_manager:
            dedup_mask, reasons = apply_lpips_dedup(all_frames_data, filters, dedup_mask, reasons, thumbnail_manager, config, output_dir)
        elif dedup_method == "pHash then LPIPS" and thumbnail_manager and imagehash:
            # First pass: pHash
            sorted_indices = sorted(range(num_frames), key=lambda i: filenames[i])
            hashes = {i: imagehash.hex_to_hash(all_frames_data[i]['phash']) for i in range(num_frames) if 'phash' in all_frames_data[i]}
            p_hash_duplicates = []
            kept_hashes = {}  # Store hashes of frames that are kept
            for i in sorted_indices:
                if i not in hashes:
                    continue
                is_duplicate = False
                for kept_idx, kept_hash in kept_hashes.items():
                    if (hashes[i] - kept_hash) <= filters.get("dedup_thresh", 5):
                        p_hash_duplicates.append((kept_idx, i))
                        is_duplicate = True
                        break # Move to the next frame
                if not is_duplicate:
                    kept_hashes[i] = hashes[i]

            # Second pass: LPIPS on pHash duplicates
            if p_hash_duplicates:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                loss_fn = get_lpips_metric(device=device)
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])

                batch_size = 32
                for i in range(0, len(p_hash_duplicates), batch_size):
                    batch = p_hash_duplicates[i:i+batch_size]
                    img1_batch = []
                    img2_batch = []
                    valid_indices = []

                    for p_idx, c_idx in batch:
                        img1 = thumbnail_manager.get(Path(output_dir) / "thumbs" / all_frames_data[p_idx]['filename'])
                        img2 = thumbnail_manager.get(Path(output_dir) / "thumbs" / all_frames_data[c_idx]['filename'])
                        if img1 is not None and img2 is not None:
                            img1_batch.append(transform(img1))
                            img2_batch.append(transform(img2))
                            valid_indices.append((p_idx, c_idx))

                    if not valid_indices:
                        continue

                    img1_t = torch.stack(img1_batch).to(device)
                    img2_t = torch.stack(img2_batch).to(device)

                    with torch.no_grad():
                        distances = loss_fn.forward(img1_t, img2_t).squeeze().cpu().numpy()

                    for j, (p_idx, c_idx) in enumerate(valid_indices):
                        if distances[j] <= filters.get("lpips_threshold", 0.1):
                            if all_frames_data[c_idx].get('metrics', {}).get('quality_score', 0) > all_frames_data[p_idx].get('metrics', {}).get('quality_score', 0):
                                if dedup_mask[p_idx]:
                                    reasons[filenames[p_idx]].append('duplicate')
                                dedup_mask[p_idx] = False
                            else:
                                if dedup_mask[c_idx]:
                                    reasons[filenames[c_idx]].append('duplicate')
                                dedup_mask[c_idx] = False
    return dedup_mask, reasons

def _apply_metric_filters(all_frames_data: list[dict], metric_arrays: dict, filters: dict,
                          config: 'Config') -> tuple[np.ndarray, defaultdict]:
    """Applies filters based on numeric metric thresholds from the UI sliders.

    This function uses the pre-extracted NumPy arrays of metric data to
    perform fast, vectorized filtering based on the min/max values set by the
    user in the UI.

    Args:
        all_frames_data (list[dict]): A list of frame metadata dictionaries.
        metric_arrays (dict): A dictionary of metric data as NumPy arrays.
        filters (dict): A dictionary of active filter settings from the UI.
        config (Config): The main application configuration.

    Returns:
        tuple[np.ndarray, defaultdict]: A tuple containing:
        - A boolean NumPy array (the mask of frames to keep).
        - A dictionary mapping filenames to their rejection reasons.
    """
    num_frames = len(all_frames_data)
    filenames = [f['filename'] for f in all_frames_data]
    reasons = defaultdict(list)

    quality_weights_keys = [k.replace('quality_weights_', '') for k in config.model_dump().keys() if k.startswith('quality_weights_')]
    filter_definitions = [
        *[{'key': k, 'type': 'range'} for k in quality_weights_keys],
        {'key': 'quality_score', 'type': 'range'},
        {'key': 'face_sim', 'type': 'min', 'enabled_key': 'face_sim_enabled', 'reason_low': 'face_sim_low', 'reason_missing': 'face_missing'},
        {'key': 'mask_area_pct', 'type': 'min', 'enabled_key': 'mask_area_enabled', 'reason_low': 'mask_too_small'},
        {'key': 'eyes_open', 'type': 'min', 'reason_low': 'eyes_closed'},
        {'key': 'yaw', 'type': 'range', 'reason_range': 'yaw_out_of_range'},
        {'key': 'pitch', 'type': 'range', 'reason_range': 'pitch_out_of_range'},
    ]

    metric_filter_mask = np.ones(num_frames, dtype=bool)

    for f_def in filter_definitions:
        key, f_type = f_def['key'], f_def['type']
        if f_def.get('enabled_key') and not filters.get(f_def['enabled_key']):
            continue

        arr = metric_arrays.get(key)
        if arr is None: continue

        f_defaults = getattr(config, f"filter_default_{key}", {})

        if f_type == 'range':
            min_v = filters.get(f"{key}_min", f_defaults.get('default_min', -np.inf))
            max_v = filters.get(f"{key}_max", f_defaults.get('default_max', np.inf))
            nan_fill = f_defaults.get('default_min', min_v)
            mask = (np.nan_to_num(arr, nan=nan_fill) >= min_v) & (np.nan_to_num(arr, nan=nan_fill) <= max_v)
            metric_filter_mask &= mask

        elif f_type == 'min':
            min_v = filters.get(f"{key}_min", f_defaults.get('default_min', -np.inf))
            nan_fill = f_defaults.get('default_min', min_v)
            if key == 'face_sim':
                has_face_sim = ~np.isnan(arr)
                mask = np.ones(num_frames, dtype=bool)
                mask[has_face_sim] = (arr[has_face_sim] >= min_v)
                if filters.get("require_face_match"):
                    mask &= has_face_sim
            else:
                mask = np.nan_to_num(arr, nan=nan_fill) >= min_v
            metric_filter_mask &= mask

    # Collect rejection reasons for frames that failed the metric filters
    metric_rejection_mask = ~metric_filter_mask
    for i in np.where(metric_rejection_mask)[0]:
        for f_def in filter_definitions:
            key, f_type = f_def['key'], f_def['type']
            if f_def.get('enabled_key') and not filters.get(f_def['enabled_key']):
                continue
            arr = metric_arrays.get(key)
            if arr is None: continue

            f_defaults = getattr(config, f"filter_default_{key}", {})
            v = arr[i]

            if f_type == 'range':
                min_v = filters.get(f"{key}_min", f_defaults.get('default_min', -np.inf))
                max_v = filters.get(f"{key}_max", f_defaults.get('default_max', np.inf))
                if not np.isnan(v):
                    reason = f_def.get('reason_range')
                    if v < min_v: reasons[filenames[i]].append(reason or f_def.get('reason_low', f"{key}_low"))
                    if v > max_v: reasons[filenames[i]].append(reason or f_def.get('reason_high', f"{key}_high"))
            elif f_type == 'min':
                min_v = filters.get(f"{key}_min", f_defaults.get('default_min', -np.inf))
                if not np.isnan(v) and v < min_v:
                    reasons[filenames[i]].append(f_def.get('reason_low', f"{key}_low"))
                if key == 'face_sim' and filters.get('require_face_match') and np.isnan(v):
                    reasons[filenames[i]].append(f_def.get('reason_missing', 'face_missing'))

    return metric_filter_mask, reasons

def apply_all_filters_vectorized(all_frames_data: list[dict], filters: dict, config: 'Config',
                                 thumbnail_manager: Optional['ThumbnailManager'] = None, output_dir: Optional[str] = None) -> tuple[list, list, Counter, dict]:
    """Applies all filtering logic to a list of frames using vectorized operations.

    This function orchestrates the entire filtering process. It calls helper
    functions to extract metrics into NumPy arrays, apply deduplication, and
    apply numeric threshold filters. It then combines the results to produce
    the final lists of kept and rejected frames.

    Args:
        all_frames_data (list[dict]): A list of frame metadata dictionaries.
        filters (dict): A dictionary of active filter settings from the UI.
        config (Config): The main application configuration.
        thumbnail_manager (Optional['ThumbnailManager']): The thumbnail cache
            manager, required for SSIM/LPIPS dedup. Defaults to None.
        output_dir (Optional[str]): The session's output directory, required for
            SSIM/LPIPS dedup. Defaults to None.

    Returns:
        tuple[list, list, Counter, dict]: A tuple containing:
        - A list of 'kept' frame dictionaries.
        - A list of 'rejected' frame dictionaries.
        - A `Counter` of all rejection reasons.
        - A dictionary mapping filenames to their specific rejection reasons.
    """
    if not all_frames_data:
        return [], [], Counter(), {}

    num_frames = len(all_frames_data)
    filenames = [f['filename'] for f in all_frames_data]

    # 1. Consolidate metric data extraction
    metric_arrays = _extract_metric_arrays(all_frames_data, config)

    # 2. Handle deduplication
    dedup_mask, reasons = _apply_deduplication_filter(
        all_frames_data, filters, thumbnail_manager, config, output_dir
    )

    # 3. Apply metric-based filters
    metric_filter_mask, metric_reasons = _apply_metric_filters(
        all_frames_data, metric_arrays, filters, config
    )
    # 4. Combine masks and reasons
    kept_mask = dedup_mask & metric_filter_mask
    # Merge the reasons from both filtering steps
    for fname, reason_list in metric_reasons.items():
        reasons[fname].extend(reason_list)

    # 5. Finalize results
    kept = [all_frames_data[i] for i in np.where(kept_mask)[0]]
    rejected = [all_frames_data[i] for i in np.where(~kept_mask)[0]]
    total_reasons = Counter(r for r_list in reasons.values() for r in r_list)

    return kept, rejected, total_reasons, reasons

def apply_ssim_dedup(all_frames_data: list[dict], filters: dict, dedup_mask: np.ndarray, reasons: defaultdict,
                     thumbnail_manager: 'ThumbnailManager', config: 'Config', output_dir: str) -> tuple[np.ndarray, defaultdict]:
    """Applies deduplication using Structural Similarity Index (SSIM).

    This function compares adjacent frames (sorted by filename) and calculates
    their SSIM. If the similarity is above a threshold, the frame with the
    lower quality score is marked for rejection.

    Args:
        all_frames_data (list[dict]): A list of frame metadata dictionaries.
        filters (dict): A dictionary of active filter settings.
        dedup_mask (np.ndarray): The current deduplication mask to be updated.
        reasons (defaultdict): A dictionary to store rejection reasons.
        thumbnail_manager (ThumbnailManager): The thumbnail cache manager.
        config (Config): The main application configuration.
        output_dir (str): The session's output directory.

    Returns:
        tuple[np.ndarray, defaultdict]: A tuple of the updated deduplication
        mask and reasons dictionary.
    """
    threshold = filters.get("ssim_threshold", 0.95)
    num_frames = len(all_frames_data)
    sorted_indices = sorted(range(num_frames), key=lambda i: all_frames_data[i]['filename'])

    for i in range(1, len(sorted_indices)):
        c_idx, p_idx = sorted_indices[i], sorted_indices[i - 1]

        c_frame_data = all_frames_data[c_idx]
        p_frame_data = all_frames_data[p_idx]

        c_thumb_path = Path(output_dir) / "thumbs" / c_frame_data['filename']
        p_thumb_path = Path(output_dir) / "thumbs" / p_frame_data['filename']

        img1 = thumbnail_manager.get(p_thumb_path)
        img2 = thumbnail_manager.get(c_thumb_path)

        if img1 is not None and img2 is not None:
            img1_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
            img2_gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

            similarity = ssim(img1_gray, img2_gray)

            if similarity >= threshold:
                # Keep the one with the higher quality score
                if all_frames_data[c_idx].get('metrics', {}).get('quality_score', 0) > all_frames_data[p_idx].get('metrics', {}).get('quality_score', 0):
                    if dedup_mask[p_idx]:
                        reasons[all_frames_data[p_idx]['filename']].append('duplicate')
                    dedup_mask[p_idx] = False
                else:
                    if dedup_mask[c_idx]:
                        reasons[all_frames_data[c_idx]['filename']].append('duplicate')
                    dedup_mask[c_idx] = False
    return dedup_mask, reasons

def apply_lpips_dedup(all_frames_data: list[dict], filters: dict, dedup_mask: np.ndarray, reasons: defaultdict,
                      thumbnail_manager: 'ThumbnailManager', config: 'Config', output_dir: str) -> tuple[np.ndarray, defaultdict]:
    """Applies deduplication using Learned Perceptual Image Patch Similarity (LPIPS).

    This function compares adjacent frames (sorted by filename) and calculates
    their LPIPS distance. If the distance is below a threshold, the frame with
    the lower quality score is marked for rejection.

    Args:
        all_frames_data (list[dict]): A list of frame metadata dictionaries.
        filters (dict): A dictionary of active filter settings.
        dedup_mask (np.ndarray): The current deduplication mask to be updated.
        reasons (defaultdict): A dictionary to store rejection reasons.
        thumbnail_manager (ThumbnailManager): The thumbnail cache manager.
        config (Config): The main application configuration.
        output_dir (str): The session's output directory.

    Returns:
        tuple[np.ndarray, defaultdict]: A tuple of the updated deduplication
        mask and reasons dictionary.
    """
    threshold = filters.get("lpips_threshold", 0.1)
    loss_fn = get_lpips_metric()
    num_frames = len(all_frames_data)
    sorted_indices = sorted(range(num_frames), key=lambda i: all_frames_data[i]['filename'])

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    for i in range(1, len(sorted_indices)):
        c_idx, p_idx = sorted_indices[i], sorted_indices[i - 1]

        c_frame_data = all_frames_data[c_idx]
        p_frame_data = all_frames_data[p_idx]

        c_thumb_path = Path(output_dir) / "thumbs" / c_frame_data['filename']
        p_thumb_path = Path(output_dir) / "thumbs" / p_frame_data['filename']

        img1 = thumbnail_manager.get(p_thumb_path)
        img2 = thumbnail_manager.get(c_thumb_path)

        if img1 is not None and img2 is not None:
            img1_t = transform(img1).unsqueeze(0)
            img2_t = transform(img2).unsqueeze(0)

            distance = loss_fn.forward(img1_t, img2_t).item()

            if distance <= threshold:
                if all_frames_data[c_idx].get('metrics', {}).get('quality_score', 0) > all_frames_data[p_idx].get('metrics', {}).get('quality_score', 0):
                    if dedup_mask[p_idx]:
                        reasons[all_frames_data[p_idx]['filename']].append('duplicate')
                    dedup_mask[p_idx] = False
                else:
                    if dedup_mask[c_idx]:
                        reasons[all_frames_data[c_idx]['filename']].append('duplicate')
                    dedup_mask[c_idx] = False
    return dedup_mask, reasons


def on_filters_changed(event: 'FilterEvent', thumbnail_manager: 'ThumbnailManager',
                       config: 'Config', logger: Optional['AppLogger'] = None) -> dict:
    """Gradio event handler for when any filter control is changed.

    This function is triggered whenever a slider, checkbox, or dropdown in the
    filtering tab is changed. It gathers all current filter values, calls the
    `_update_gallery` helper to re-apply the filters, and returns the UI updates.

    Args:
        event (FilterEvent): A `FilterEvent` object containing all relevant data
            from the UI.
        thumbnail_manager (ThumbnailManager): The thumbnail cache manager.
        config (Config): The main application configuration.
        logger (Optional[AppLogger]): The application logger.

    Returns:
        dict: A dictionary of Gradio updates for the "filter_status_text" and
        "results_gallery" components.
    """
    logger = logger or AppLogger(config=Config())
    if not event.all_frames_data: return {"filter_status_text": "Run analysis to see results.", "results_gallery": []}
    filters = event.slider_values.copy()
    filters.update({"require_face_match": event.require_face_match, "dedup_thresh": event.dedup_thresh,
                    "face_sim_enabled": bool(event.per_metric_values.get("face_sim")),
                    "mask_area_enabled": bool(event.per_metric_values.get("mask_area_pct")),
                    "enable_dedup": any('phash' in f for f in event.all_frames_data) if event.all_frames_data else False,
                    "dedup_method": event.dedup_method})
    status_text, gallery_update = _update_gallery(event.all_frames_data, filters, event.output_dir, event.gallery_view,
                                                  event.show_overlay, event.overlay_alpha, thumbnail_manager, config, logger)
    return {"filter_status_text": status_text, "results_gallery": gallery_update}

def _update_gallery(all_frames_data: list[dict], filters: dict, output_dir: str, gallery_view: str,
                    show_overlay: bool, overlay_alpha: float, thumbnail_manager: 'ThumbnailManager',
                    config: 'Config', logger: 'AppLogger') -> tuple[str, gr.update]:
    """Helper function to re-apply filters and update the results gallery.

    This function calls `apply_all_filters_vectorized` to get the latest lists
    of kept and rejected frames. It then prepares the images (with optional
    mask overlays) and captions for display in the Gradio gallery component.

    Args:
        all_frames_data (list[dict]): A list of all frame metadata.
        filters (dict): A dictionary of the current filter settings.
        output_dir (str): The main output directory for the run.
        gallery_view (str): The current view ("Kept" or "Rejected").
        show_overlay (bool): Whether to display the mask overlay.
        overlay_alpha (float): The transparency of the mask overlay.
        thumbnail_manager (ThumbnailManager): The thumbnail cache manager.
        config (Config): The main application configuration.
        logger (AppLogger): The application logger.

    Returns:
        tuple[str, gr.update]: A tuple containing the updated status text and a
        Gradio update object for the gallery.
    """
    kept, rejected, counts, per_frame_reasons = apply_all_filters_vectorized(all_frames_data, filters or {}, config, thumbnail_manager, output_dir)
    status_parts = [f"**Kept:** {len(kept)}/{len(all_frames_data)}"]
    if counts:
        rejection_reasons = ', '.join([f'{k}: {v}' for k, v in counts.most_common()])
        status_parts.append(f"**Rejections:** {rejection_reasons}")

    status_text, frames_to_show, preview_images = " | ".join(status_parts), rejected if gallery_view == "Rejected" else kept, []
    if output_dir:
        output_path, thumb_dir, masks_dir = Path(output_dir), Path(output_dir) / "thumbs", Path(output_dir) / "masks"
        for f_meta in frames_to_show[:100]:
            thumb_path = thumb_dir / f"{Path(f_meta['filename']).stem}.webp"
            caption = f"Reasons: {', '.join(per_frame_reasons.get(f_meta['filename'], []))}" if gallery_view == "Rejected" else ""
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


def auto_set_thresholds(per_metric_values: dict, p: int, slider_keys: list[str], selected_metrics: list[str]) -> dict:
    """Calculates percentile-based thresholds for sliders.

    This function provides a "quick set" feature for the filter sliders. It
    calculates the value at a given percentile (`p`) for each of the selected
    metrics and updates the corresponding `_min` slider to that value.

    Args:
        per_metric_values (dict): A dictionary of metric values.
        p (int): The percentile to calculate (1-99).
        slider_keys (list[str]): A list of all slider keys in the UI.
        selected_metrics (list[str]): A list of metric names to apply the
            auto-threshold to.

    Returns:
        dict: A dictionary of Gradio update objects for the sliders.
    """
    updates = {}
    if not per_metric_values: return {f'slider_{key}': gr.update() for key in slider_keys}
    # Calculate percentile values only for the selected metrics
    pmap = {
        k: float(np.percentile(np.asarray(vals, dtype=np.float32), p))
        for k, vals in per_metric_values.items()
        if not k.endswith('_hist') and vals and k in selected_metrics
    }
    for key in slider_keys:
        metric_name = key.replace('_min', '').replace('_max', '')
        # Only update the slider if its corresponding metric was selected
        if key.endswith('_min') and metric_name in pmap:
            updates[f'slider_{key}'] = gr.update(value=round(pmap[metric_name], 2))
        else:
            # Otherwise, send a no-op update to leave it unchanged
            updates[f'slider_{key}'] = gr.update()
    return updates

def save_scene_seeds(scenes_list: list['Scene'], output_dir_str: str, logger: 'AppLogger'):
    """Saves the current state of scenes to `scene_seeds.json`.

    This function persists the results of the pre-analysis and any manual
    changes made in the scene selection tab. This allows the state to be
    reloaded later or used by subsequent pipeline stages.

    Args:
        scenes_list (list['Scene']): A list of scene objects.
        output_dir_str (str): The path to the output directory.
        logger (AppLogger): The application logger.
    """
    if not scenes_list or not output_dir_str: return
    scene_seeds = {}
    for s in scenes_list:
        data = {
            'best_frame': s.best_frame,
            'seed_frame_idx': s.seed_frame_idx,
            'seed_type': s.seed_type,
            'seed_config': s.seed_config,
            'status': s.status,
            'seed_metrics': s.seed_metrics
        }
        scene_seeds[str(s.shot_id)] = data
    try:
        (Path(output_dir_str) / "scene_seeds.json").write_text(json.dumps(_to_json_safe(scene_seeds), indent=2), encoding='utf-8')
        logger.info("Saved scene_seeds.json")
    except Exception as e: logger.error("Failed to save scene_seeds.json", exc_info=True)

def get_scene_status_text(scenes_list: list['Scene']) -> tuple[str, gr.update]:
    """Generates a status summary text and updates the propagation button.

    This function calculates how many scenes are included, how many are ready
    for propagation (i.e., have a valid seed), and provides a summary of
    rejection reasons. It also controls the interactivity of the "Propagate"
    button based on whether any scenes are ready.

    Args:
        scenes_list (list['Scene']): A list of scene objects.

    Returns:
        tuple[str, gr.update]: A tuple containing the status string and a Gradio
        update object for the propagation button.
    """
    if not scenes_list:
        return "No scenes loaded.", gr.update(interactive=False)


    included_scenes = [s for s in scenes_list if s.status == 'included']
    # A scene is ready if it's included AND has a valid seed result.
    ready_for_propagation_count = sum(
        1 for s in included_scenes if s.seed_result and s.seed_result.get('bbox')
    )

    total_count = len(scenes_list)
    included_count = len(included_scenes)


    rejection_counts = Counter()
    for scene in scenes_list:
        if scene.status == 'excluded' and scene.rejection_reasons:
            reasons = scene.rejection_reasons
            if reasons:
                rejection_counts.update(reasons)

    status_text = f"{included_count}/{total_count} scenes included for propagation."

    if rejection_counts:
        reasons_summary = ", ".join([f"{reason}: {count}" for reason, count in rejection_counts.items()])
        status_text += f" (Rejected: {reasons_summary})"

    button_text = f"🔬 Propagate Masks on {ready_for_propagation_count} Ready Scenes"
    return status_text, gr.update(value=button_text, interactive=ready_for_propagation_count > 0)

def draw_boxes_preview(img: np.ndarray, boxes_xyxy: list[list[int]], cfg: 'Config') -> np.ndarray:
    """Draws multiple bounding boxes on an image for preview.

    Args:
        img (np.ndarray): The input image as a NumPy array.
        boxes_xyxy (list[list[int]]): A list of bounding boxes in
            [x1, y1, x2, y2] format.
        cfg (Config): The application configuration for visualization settings.

    Returns:
        np.ndarray: The image with bounding boxes drawn on it.
    """
    img = img.copy()
    for x1,y1,x2,y2 in boxes_xyxy:
        cv2.rectangle(img, (int(x1),int(y1)), (int(x2),int(y2)),
                      cfg.visualization.bbox_color, cfg.visualization.bbox_thickness)
    return img

def toggle_scene_status(scenes_list: list['Scene'], selected_shot_id: int, new_status: str,
                        output_folder: str, logger: 'AppLogger') -> tuple[list, str, str, gr.update]:
    """Updates the status of a selected scene.

    This function finds a scene by its ID, updates its status ('included' or
    'excluded'), and saves the change to `scene_seeds.json`.

    Args:
        scenes_list (list['Scene']): The list of all scene objects.
        selected_shot_id (int): The ID of the scene to update.
        new_status (str): The new status to set ("included" or "excluded").
        output_folder (str): The output directory for saving the updated scene data.
        logger (AppLogger): The application logger.

    Returns:
        tuple[list, str, str, gr.update]: A tuple containing the updated scenes
        list, the new status text, a confirmation message, and a Gradio update
        object for the propagation button.
    """
    if selected_shot_id is None or not scenes_list:
        status_text, button_update = get_scene_status_text(scenes_list)
        return scenes_list, status_text, "No scene selected.", button_update

    scene_to_update = next((s for s in scenes_list if s.shot_id == selected_shot_id), None)

    if scene_to_update:
        scene_to_update.status = new_status
        scene_to_update.manual_status_change = True
        save_scene_seeds(scenes_list, output_folder, logger)
        status_text, button_update = get_scene_status_text(scenes_list)
        return (scenes_list, status_text, f"Scene {selected_shot_id} status set to {new_status}.", button_update)

    status_text, button_update = get_scene_status_text(scenes_list)
    return (scenes_list, status_text, f"Could not find scene {selected_shot_id}.", button_update)


# --- Cleaned Scene Recomputation Workflow ---

def _create_analysis_context(config: 'Config', logger: 'AppLogger', thumbnail_manager: 'ThumbnailManager',
                             cuda_available: bool, ana_ui_map_keys: list[str], ana_input_components: list) -> 'SubjectMasker':
    """Factory function to create a fully initialized SubjectMasker from UI state.

    This function centralizes the creation of the analysis context. It gathers
    all relevant values from the Gradio UI, creates a validated
    `AnalysisParameters` object, initializes all necessary models for that
    configuration, and returns a fully configured `SubjectMasker` instance
    ready for use.

    Args:
        config (Config): The main application configuration.
        logger (AppLogger): The application logger.
        thumbnail_manager (ThumbnailManager): The thumbnail cache manager.
        cuda_available (bool): Boolean indicating if CUDA is available.
        ana_ui_map_keys (list[str]): A list of keys for the analysis UI
            components.
        ana_input_components (list): A list of the analysis UI component values.

    Returns:
        'SubjectMasker': A fully configured `SubjectMasker` instance.

    Raises:
        FileNotFoundError: If the specified output folder is invalid or does
            not exist.
        RuntimeError: If the frame map cannot be created.
    """
    ui_args = dict(zip(ana_ui_map_keys, ana_input_components))

    # --- Defensive Patch for Mismatched UI Args ---
    if 'outputfolder' in ui_args and 'output_folder' not in ui_args:
        ui_args['output_folder'] = ui_args.pop('outputfolder')

    output_folder_str = ui_args.get('output_folder')

    # Defensive fix for mis-mapped booleans or missing values
    if not output_folder_str or isinstance(output_folder_str, bool):
        logger.error(f"Output folder is not valid (was '{output_folder_str}', type: {type(output_folder_str)}). This is likely due to a UI argument mapping error.", component="analysis")
        raise FileNotFoundError(f"Output folder is not valid or does not exist: {output_folder_str}")

    # --- Robust Path Handling ---
    if not Path(output_folder_str).exists():
        raise FileNotFoundError(f"Output folder is not valid or does not exist: {output_folder_str}")
    resolved_outdir = Path(output_folder_str).resolve()
    ui_args['output_folder'] = str(resolved_outdir)

    # Create parameters and initialize all necessary models
    params = AnalysisParameters.from_ui(logger, config, **ui_args)
    models = initialize_analysis_models(params, config, logger)
    frame_map = create_frame_map(resolved_outdir, logger)
    
    if not frame_map:
        raise RuntimeError("Failed to create frame map. Check if frame_map.json exists and is valid.")

    # Return a fully configured SubjectMasker instance
    return SubjectMasker(
        params=params, progress_queue=Queue(), cancel_event=threading.Event(), config=config,
        frame_map=frame_map, face_analyzer=models["face_analyzer"],
        reference_embedding=models["ref_emb"], person_detector=models["person_detector"],
        niqe_metric=None, thumbnail_manager=thumbnail_manager, logger=logger,
        face_landmarker=models["face_landmarker"], device=models["device"]
    )

def _recompute_single_preview(scene_state: 'SceneState', masker: 'SubjectMasker', overrides: dict,
                              thumbnail_manager: 'ThumbnailManager', logger: 'AppLogger'):
    """Recomputes the seed and preview for a single scene.

    This function is called by the scene editor. It takes a `SceneState`
    object, applies user-defined overrides (like a new text prompt), runs the
    `SeedSelector` to get a new bounding box, generates a new mask and preview
    image, and updates the scene state in place.

    Args:
        scene_state (SceneState): The `SceneState` object for the scene to
            recompute.
        masker (SubjectMasker): The initialized `SubjectMasker` context.
        overrides (dict): A dictionary of parameters to override for this
            computation (e.g., `{'text_prompt': 'new prompt'}`).
        thumbnail_manager (ThumbnailManager): The thumbnail cache manager.
        logger (AppLogger): The application logger.
    """
    scene = scene_state.data
    out_dir = Path(masker.params.output_folder)
    best_frame_num = scene.get('best_frame') or scene.get('start_frame')
    if best_frame_num is None:
        raise ValueError(f"Scene {scene.get('shot_id')} has no best frame number.")

    fname = masker.frame_map.get(int(best_frame_num))
    if not fname:
        raise FileNotFoundError(f"Best frame {best_frame_num} not found in project's frame map.")

    thumb_rgb = thumbnail_manager.get(out_dir / "thumbs" / f"{Path(fname).stem}.webp")
    if thumb_rgb is None:
        raise FileNotFoundError(f"Thumbnail for frame {best_frame_num} not found on disk.")

    # Create a temporary config for this specific seed selection run
    seed_config = {**masker.params.model_dump(), **overrides}
    
    # If the user provides a text prompt in the editor, it's a strong signal
    # to use the text-first seeding strategy for this specific re-computation.
    if overrides.get("text_prompt", "").strip():
        seed_config['primary_seed_strategy'] = "📝 By Text"
        logger.info(f"Recomputing scene {scene.get('shot_id')} with text-first strategy due to override.", extra={'prompt': overrides.get("text_prompt")})

    # Recompute seed and update scene dictionary
    bbox, details = masker.get_seed_for_frame(thumb_rgb, seed_config=seed_config, scene=scene)
    scene_state.update_seed_result(bbox, details)
    scene_state.data['seed_config'].update(overrides)
    
    # Update metrics that are displayed in the caption
    new_score = details.get('final_score') or details.get('conf') or details.get('dino_conf')
    if new_score is not None:
        scene.setdefault('seed_metrics', {})['score'] = new_score

    # Generate and save a new preview image
    mask = masker.get_mask_for_bbox(thumb_rgb, bbox) if bbox else None
    if mask is not None:
        h, w = mask.shape[:2]; area = (h * w)
        scene.get('seed_result', {}).get('details', {})['mask_area_pct'] = (np.sum(mask > 0) / area * 100.0) if area > 0 else 0.0

    overlay_rgb = render_mask_overlay(thumb_rgb, mask, 0.6, logger=logger) if mask is not None else masker.draw_bbox(thumb_rgb, bbox)
    previews_dir = out_dir / "previews"
    previews_dir.mkdir(parents=True, exist_ok=True)
    preview_path = previews_dir / f"scene_{int(scene['shot_id']):05d}.jpg"
    try:
        Image.fromarray(overlay_rgb).save(preview_path)
        scene['preview_path'] = str(preview_path)
    except Exception:
        logger.error(f"Failed to save preview for scene {scene['shot_id']}", exc_info=True)


def _wire_recompute_handler(config: 'Config', logger: 'AppLogger', thumbnail_manager: 'ThumbnailManager',
                            scenes: list['Scene'], shot_id: int, outdir: str, text_prompt: str,
                            box_thresh: float, text_thresh: float, view: str, ana_ui_map_keys: list[str],
                            ana_input_components: list, cuda_available: bool) -> tuple:
    """Orchestrator for the 'Recompute Preview' button event in the UI.

    This function is the event handler for the "Recompute Preview" button in
    the scene editor. It gathers all necessary data from the UI, creates the
    analysis context, calls the recomputation logic, and returns a tuple of
    Gradio updates.

    Args:
        config (Config): The main application configuration.
        logger (AppLogger): The application logger.
        thumbnail_manager (ThumbnailManager): The thumbnail cache manager.
        scenes (list['Scene']): The list of all scene objects.
        shot_id (int): The ID of the currently selected scene.
        outdir (str): The path to the output directory.
        text_prompt (str): The text prompt from the UI.
        box_thresh (float): The box threshold from the UI.
        text_thresh (float): The text threshold from the UI.
        view (str): The current gallery view ("Kept", "Rejected", "All").
        ana_ui_map_keys (list[str]): Keys for the analysis UI components.
        ana_input_components (list): Values from the analysis UI components.
        cuda_available (bool): Boolean indicating if CUDA is available.

    Returns:
        tuple: A tuple of Gradio update objects for the UI.
    """
    try:
        # If the DINO text prompt is empty, do nothing.
        if not text_prompt or not text_prompt.strip():
            return scenes, gr.update(), gr.update(), "Enter a text prompt to use advanced seeding."

        # 1. Create the full analysis context from the current UI state
        # We need to combine the specific inputs from the recompute button with the general analysis inputs
        ui_args = dict(zip(ana_ui_map_keys, ana_input_components))
        ui_args['output_folder'] = outdir # This is critical
        
        # The ana_input_components passed to this handler already contain the full UI state,
        # so we can create the masker directly from them.
        masker = _create_analysis_context(config, logger, thumbnail_manager,
                                          ana_ui_map_keys, ana_input_components)

        # 2. Find the target scene and apply overrides
        scene_idx = next((i for i, s in enumerate(scenes) if s.get('shot_id') == shot_id), None)
        if scene_idx is None:
            return scenes, gr.update(), gr.update(), f"Error: Scene {shot_id} not found."

        overrides = {"text_prompt": text_prompt, "box_threshold": float(box_thresh), "text_threshold": float(text_thresh)}
        scene_state = SceneState(scenes[scene_idx])
        _recompute_single_preview(scene_state, masker, overrides, thumbnail_manager, logger)

        # 3. Persist the changes and update the UI
        save_scene_seeds(scenes, outdir, logger)
        gallery_items, index_map = build_scene_gallery_items(scenes, view, outdir)
        msg = f"Scene {shot_id} preview recomputed successfully with DINO."
        return scenes, gr.update(value=gallery_items), gr.update(value=index_map), msg

    except Exception as e:
        logger.error("Failed to recompute scene preview", exc_info=True)
        gallery_items, index_map = build_scene_gallery_items(scenes, view, outdir)
        return scenes, gr.update(value=gallery_items), gr.update(value=index_map), f"[ERROR] Recompute failed: {str(e)}"

# --- PIPELINE LOGIC ---

@handle_common_errors
def execute_extraction(event: 'ExtractionEvent', progress_queue: Queue, cancel_event: threading.Event,
                       logger: 'AppLogger', config: 'Config', thumbnail_manager: Optional['ThumbnailManager'] = None,
                       cuda_available: Optional[bool] = None, progress: Optional[Callable] = None) -> Generator[dict, None, None]:
    """Executes the frame extraction pipeline.

    This function is a generator that serves as the backend entry point for the
    extraction process. It is wrapped by UI logic that handles threading and
    progress updates. It instantiates and runs the `EnhancedExtractionPipeline`.

    Args:
        event (ExtractionEvent): The `ExtractionEvent` from the UI.
        progress_queue (Queue): The queue for UI updates.
        cancel_event (threading.Event): The event to signal cancellation.
        logger (AppLogger): The application logger.
        config (Config): The main application configuration.
        thumbnail_manager (Optional['ThumbnailManager']): The thumbnail manager
            (unused in this pipeline).
        cuda_available (Optional[bool]): Flag for CUDA availability (unused in
            this pipeline).
        progress (Optional[Callable]): The Gradio progress object.

    Yields:
        dict: A dictionary containing UI updates. The final yielded dictionary
        indicates the completion status and final results.
    """
    try:
        params_dict = asdict(event)
        if event.upload_video:
            source, dest = params_dict.pop('upload_video'), str(Path(config.paths.downloads) / Path(event.upload_video).name)
            shutil.copy2(source, dest)
            params_dict['source_path'] = dest

        params = AnalysisParameters.from_ui(logger, config, **params_dict)
        tracker = AdvancedProgressTracker(progress, progress_queue, logger, ui_stage_name="Extracting")
        pipeline = EnhancedExtractionPipeline(config, logger, params, progress_queue, cancel_event)

        result = pipeline.run(tracker=tracker)

        if result and result.get("done"):
            yield {
                "unified_log": "Extraction complete. You can now proceed to the next step.",
                "extracted_video_path_state": result.get("video_path", ""),
                "extracted_frames_dir_state": result["output_dir"],
                "done": True
            }
        else:
            yield {
                "unified_log": f"Extraction failed. Reason: {result.get('log', 'Unknown error')}",
                "done": False
            }
    except Exception as e:
        logger.error("Extraction execution failed", exc_info=True)
        yield {"unified_log": f"[ERROR] Extraction failed unexpectedly: {e}", "done": False}


@handle_common_errors
def execute_pre_analysis(event: 'PreAnalysisEvent', progress_queue: Queue, cancel_event: threading.Event,
                         logger: 'AppLogger', config: 'Config', thumbnail_manager: 'ThumbnailManager',
                         cuda_available: bool, progress: Optional[Callable] = None) -> Generator[dict, None, None]:
    """Executes the pre-analysis pipeline (seed selection and preview generation).

    This function is a generator that serves as the backend entry point for the
    pre-analysis process. It instantiates the `SubjectMasker` and iterates
    through each scene to find the best seed frame, select the subject, and
    generate a preview image.

    Args:
        event (PreAnalysisEvent): The `PreAnalysisEvent` from the UI.
        progress_queue (Queue): The queue for UI updates.
        cancel_event (threading.Event): The event to signal cancellation.
        logger (AppLogger): The application logger.
        config (Config): The main application configuration.
        thumbnail_manager (ThumbnailManager): The thumbnail manager.
        cuda_available (bool): Flag for CUDA availability.
        progress (Optional[Callable]): The Gradio progress object.

    Yields:
        dict: A dictionary containing UI updates. The final yielded dictionary
        contains the full list of processed scene data.
    """
    try:
        tracker = AdvancedProgressTracker(progress, progress_queue, logger, ui_stage_name="Pre-Analysis")
        params_dict = event.model_dump()
        is_folder_mode = not params_dict.get("video_path")
        if event.face_ref_img_upload:
            ref_upload, dest = params_dict.pop('face_ref_img_upload'), Path(config.downloads_dir) / Path(event.face_ref_img_upload).name
            shutil.copy2(ref_upload, dest)
            params_dict['face_ref_img_path'] = str(dest)
        params = AnalysisParameters.from_ui(logger, config, **params_dict)
        output_dir = Path(params.output_folder)
        (output_dir / "run_config.json").write_text(json.dumps({k: v for k, v in params_dict.items() if k != 'face_ref_img_upload'}, indent=4))
        models = initialize_analysis_models(params, config, logger)
        niqe_metric = pyiqa.create_metric('niqe', device=models['device']) if not is_folder_mode and params.pre_analysis_enabled and pyiqa and params.primary_seed_strategy != "🧑‍🤝‍🧑 Find Prominent Person" else None
        masker = SubjectMasker(params, progress_queue, cancel_event, config, face_analyzer=models["face_analyzer"],
                               reference_embedding=models["ref_emb"], person_detector=models["person_detector"],
                               niqe_metric=niqe_metric, thumbnail_manager=thumbnail_manager, logger=logger,
                               face_landmarker=models["face_landmarker"], device=models["device"])
        masker.frame_map = masker._create_frame_map(str(output_dir))
        scenes_path = output_dir / "scenes.json"
        if not scenes_path.exists():
            yield {"unified_log": "[ERROR] scenes.json not found. Run extraction first.", "done": False}
            return
        scenes = [Scene(shot_id=i, start_frame=s, end_frame=e) for i, (s, e) in enumerate(json.load(scenes_path.open('r', encoding='utf-8')))]
        tracker.start(len(scenes), desc="Analyzing Scenes" if is_folder_mode else "Pre-analyzing Scenes")
        previews_dir = output_dir / "previews"; previews_dir.mkdir(exist_ok=True)
        for scene in scenes:
            if cancel_event.is_set(): break
            tracker.step(1, desc=f"Scene {scene.shot_id}")
            if is_folder_mode: scene.best_frame = scene.start_frame
            elif not scene.best_frame: masker._select_best_frame_in_scene(scene, str(output_dir))
            fname = masker.frame_map.get(scene.best_frame)
            if not fname: continue
            thumb_rgb = thumbnail_manager.get(output_dir / "thumbs" / f"{Path(fname).stem}.webp")
            if thumb_rgb is None: continue
            bbox, details = masker.get_seed_for_frame(thumb_rgb, seed_config=scene.seed_config or params, scene=scene)
            scene.seed_result = {'bbox': bbox, 'details': details}
            mask = masker.get_mask_for_bbox(thumb_rgb, bbox) if bbox and params.enable_subject_mask else None
            if mask is not None and mask.size > 0:
                h, w = mask.shape[:2]; area = h * w
                scene.seed_result['details']['mask_area_pct'] = (np.sum(mask > 0) / area * 100) if area > 0 else 0.0
            overlay_rgb = render_mask_overlay(thumb_rgb, mask, 0.6, logger=logger) if mask is not None else masker.draw_bbox(thumb_rgb, bbox)
            preview_path = previews_dir / f"scene_{scene.shot_id:05d}.jpg"
            Image.fromarray(overlay_rgb).save(preview_path)
            scene.preview_path, scene.status = str(preview_path), 'included'
        save_scene_seeds([s.model_dump() for s in scenes], str(output_dir), logger)
        tracker.done_stage("Pre-analysis complete")
        final_yield = {
            "unified_log": "Pre-analysis complete. Review scenes in the next tab.",
            "scenes": [s.model_dump() for s in scenes], "output_dir": str(output_dir), "done": True,
            "seeding_results_column": gr.update(visible=True), "propagation_group": gr.update(visible=True)
        }
        if params.face_ref_img_path:
            final_yield['final_face_ref_path'] = params.face_ref_img_path
        yield final_yield
    except Exception as e:
        logger.error("Pre-analysis execution failed", exc_info=True)
        yield {"unified_log": f"[ERROR] Pre-analysis failed unexpectedly: {e}", "done": False}

def validate_session_dir(path: Union[str, Path]) -> tuple[Optional[Path], Optional[str]]:
    """Validates that a path is an existing directory.

    Args:
        path (Union[str, Path]): The path to validate.

    Returns:
        tuple[Optional[Path], Optional[str]]: A tuple of (Path, None) if the
        path is a valid, existing directory, or (None, error_message) if it is
        not.
    """
    try:
        p = Path(path).expanduser().resolve()
        return (p if p.exists() and p.is_dir() else None,
                None if p.exists() and p.is_dir() else f"Session directory does not exist: {p}")
    except Exception as e:
        return None, f"Invalid session path: {e}"


def execute_session_load(
    app_ui: 'AppUI',
    event: 'SessionLoadEvent',
    logger: 'AppLogger',
    config: 'Config',
    thumbnail_manager: 'ThumbnailManager',
) -> Generator[dict, None, None]:
    """Executes the session loading process.

    This function is a generator that loads a previous session from a directory.
    It reads `run_config.json`, `scenes.json`, `scene_seeds.json`, and
    `metadata.jsonl` to restore the state of the UI and all processing steps.

    Args:
        app_ui (AppUI): The main AppUI instance.
        event (SessionLoadEvent): The `SessionLoadEvent` from the UI.
        logger (AppLogger): The application logger.
        config (Config): The main application configuration.
        thumbnail_manager (ThumbnailManager): The thumbnail manager.

    Yields:
        dict: A dictionary of Gradio UI updates to restore the application
        state.
    """
    if not event.session_path or not event.session_path.strip():
        logger.error("No session path provided.", component="session_loader")
        yield {"unified_log": "[ERROR] Please enter a path to a session directory."}
        return

    session_path, error = validate_session_dir(event.session_path)
    if error:
        logger.error(f"Invalid session path provided: {event.session_path}", component="session_loader")
        yield {"unified_log": f"[ERROR] {error}"}
        return

    config_path = session_path / "run_config.json"
    scene_seeds_path = session_path / "scene_seeds.json"
    metadata_path = session_path / "metadata.db"

    def _resolve_output_dir(base: Path, output_folder: str | None) -> Path | None:
        if not output_folder: return None
        p = Path(output_folder)
        if p.exists(): return p.resolve()
        if not p.is_absolute(): return (base / p).resolve()
        return p

    with logger.operation("Load Session", component="session_loader"):
        if not config_path.exists():
            yield {"unified_log": f"[ERROR] Could not find 'run_config.json' in {session_path}."}
            return
        try:
            run_config = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            yield {"unified_log": f"[ERROR] run_config.json is invalid: {e}"}
            return

        output_dir = _resolve_output_dir(session_path, run_config.get("output_folder")) or session_path
        updates = {
            "source_input": gr.update(value=run_config.get("source_path", "")),
            "max_resolution": gr.update(value=run_config.get("max_resolution", "1080")),
            "thumb_megapixels_input": gr.update(value=run_config.get("thumb_megapixels", 0.5)),
            "ext_scene_detect_input": gr.update(value=run_config.get("scene_detect", True)),
            "method_input": gr.update(value=run_config.get("method", "scene")),
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
            "analysis_output_dir_state": str(output_dir.resolve()),
        }
        scenes_as_dict = []
        scenes_json_path = session_path / "scenes.json"
        if scenes_json_path.exists():
            try:
                scenes_as_dict = [{"shot_id": i, "start_frame": s, "end_frame": e} for i, (s, e) in enumerate(json.loads(scenes_json_path.read_text(encoding="utf-8")))]
            except Exception as e:
                yield {"unified_log": f"[ERROR] Failed to read scenes.json: {e}", "done": False}; return
        if scene_seeds_path.exists():
            try:
                seeds_lookup = {int(k): v for k, v in json.loads(scene_seeds_path.read_text(encoding="utf-8")).items()}
                for scene in scenes_as_dict:
                    if (shot_id := scene.get("shot_id")) in seeds_lookup:
                        rec = seeds_lookup[shot_id]
                        rec['best_frame'] = rec.get('best_frame', rec.get('best_seed_frame'))
                        scene.update(rec)
                    scene.setdefault("status", "included")
            except Exception as e: logger.warning(f"Failed to parse scene_seeds.json: {e}")
        if scenes_as_dict and output_dir:
            status_text, button_update = get_scene_status_text(scenes_as_dict)
            gallery_items, index_map = build_scene_gallery_items(scenes_as_dict, "Kept", str(output_dir))
            updates.update({
                "scenes_state": scenes_as_dict, "propagate_masks_button": button_update,
                "seeding_results_column": gr.update(visible=True), "propagation_group": gr.update(visible=True),
                "scene_filter_status": status_text,
                "scene_face_sim_min_input": gr.update(visible=any((s.get("seed_metrics") or {}).get("best_face_sim") is not None for s in (scenes_as_dict or []))),
                "scene_gallery": gr.update(value=gallery_items), "scene_gallery_index_map_state": index_map
            })
        if metadata_path.exists(): updates.update({"analysis_output_dir_state": str(session_path), "filtering_tab": gr.update(interactive=True)})
        for metric in app_ui.ana_ui_map_keys:
            if metric.startswith('compute_'): updates[metric] = gr.update(value=run_config.get(metric, True))
        updates.update({"unified_log": f"Successfully loaded session from: {session_path}", "main_tabs": gr.update(selected=3)})
        yield updates
        if scenes_as_dict and output_dir:
            gallery_items, index_map = build_scene_gallery_items(scenes_as_dict, "Kept", str(output_dir))
            yield {"scene_gallery": gr.update(value=gallery_items), "scene_gallery_index_map_state": index_map}

def execute_propagation(event: PropagationEvent, progress_queue: Queue, cancel_event: threading.Event, logger: AppLogger,
                        config: Config, thumbnail_manager, cuda_available, progress=None) -> Generator[dict, None, None]:
    """Executes the mask propagation pipeline.

    This function is a generator that serves as the backend entry point for the
    mask propagation process. It filters for scenes marked as 'included',
    instantiates the `AnalysisPipeline`, and calls its `run_full_analysis`
    method to perform the propagation.

    Args:
        event (PropagationEvent): The `PropagationEvent` from the UI.
        progress_queue (Queue): The queue for UI updates.
        cancel_event (threading.Event): The event to signal cancellation.
        logger (AppLogger): The application logger.
        config (Config): The main application configuration.
        thumbnail_manager: The thumbnail manager.
        cuda_available: Flag for CUDA availability.
        progress: The Gradio progress object.

    Yields:
        dict: A dictionary containing UI updates.
    """
    try:
        params = AnalysisParameters.from_ui(logger, config, **event.analysis_params.model_dump())
        is_folder_mode = not params.video_path
        scene_fields = {f.name for f in fields(Scene)}
        scenes_to_process = [Scene(**{k: v for k, v in s.items() if k in scene_fields}) for s in event.scenes if is_folder_mode or s.get('status') == 'included']
        if not scenes_to_process:
            yield {"unified_log": "No scenes were included for processing. Nothing to do."}; return
        tracker = AdvancedProgressTracker(progress, progress_queue, logger, ui_stage_name="Analysis")
        if is_folder_mode:
            tracker.start(len(scenes_to_process), desc="Analyzing Images")
        else:
            video_info = VideoManager.get_video_info(params.video_path)
            totals = estimate_totals(params, video_info, scenes_to_process)
            tracker.start(totals.get("propagation", 0) + len(scenes_to_process), desc="Propagating Masks & Analyzing")
        pipeline = AnalysisPipeline(config, logger, params, progress_queue, cancel_event, thumbnail_manager)
        result = pipeline.run_full_analysis(scenes_to_process, tracker=tracker)
        if result and result.get("done"):
            masks_dir = Path(result['output_dir']) / "masks"
            mask_files = list(masks_dir.glob("*.png")) if masks_dir.exists() else []
            if not mask_files:
                yield {"unified_log": "❌ Propagation failed - no masks were generated. Check DAM4SAM model logs.", "done": False}; return
            yield {"unified_log": f"✅ Propagation complete. Generated {len(mask_files)} masks.", "output_dir": result['output_dir'], "done": True}
        else:
            yield {"unified_log": f"❌ Propagation failed. Reason: {result.get('error', 'Unknown error')}", "done": False}
    except Exception as e:
        logger.error("Propagation execution failed", exc_info=True)
        yield {"unified_log": f"[ERROR] Propagation failed unexpectedly: {e}", "done": False}


@handle_common_errors
def execute_analysis(event: PropagationEvent, progress_queue: Queue, cancel_event: threading.Event, logger: AppLogger,
                     config: Config, thumbnail_manager, cuda_available, progress=None) -> Generator[dict, None, None]:
    """Executes the frame analysis pipeline.

    This function is a generator that serves as the backend entry point for the
    frame analysis (metric calculation) process. It calls the `run_analysis_only`
    method of the `AnalysisPipeline`.

    Args:
        event (PropagationEvent): The `PropagationEvent` from the UI.
        progress_queue (Queue): The queue for UI updates.
        cancel_event (threading.Event): The event to signal cancellation.
        logger (AppLogger): The application logger.
        config (Config): The main application configuration.
        thumbnail_manager: The thumbnail manager.
        cuda_available: Flag for CUDA availability.
        progress: The Gradio progress object.

    Yields:
        dict: A dictionary containing UI updates.
    """
    try:
        params = AnalysisParameters.from_ui(logger, config, **event.analysis_params.model_dump())
        scenes_to_process = [Scene(**{k: v for k, v in s.items() if k in {f.name for f in fields(Scene)}}) for s in event.scenes if s.get('status') == 'included']
        if not scenes_to_process:
            yield {"unified_log": "No scenes to analyze. Nothing to do."}; return
        video_info = VideoManager.get_video_info(params.video_path)
        totals = estimate_totals(params, video_info, scenes_to_process)
        tracker = AdvancedProgressTracker(progress, progress_queue, logger, ui_stage_name="Analyzing")
        tracker.start(sum(s.end_frame - s.start_frame for s in scenes_to_process), desc="Analyzing Frames")
        pipeline = AnalysisPipeline(config, logger, params, progress_queue, cancel_event, thumbnail_manager)
        result = pipeline.run_analysis_only(scenes_to_process, tracker=tracker)
        if result and result.get("done"):
            yield {"unified_log": "Analysis complete. You can now proceed to the Filtering & Export tab.",
                   "output_dir": result['output_dir'], "done": True}
        else:
            yield {"unified_log": f"❌ Analysis failed. Reason: {result.get('error', 'Unknown error')}", "done": False}
    except Exception as e:
        logger.error("Analysis execution failed", exc_info=True)
        yield {"unified_log": f"[ERROR] Analysis failed unexpectedly: {e}", "done": False}

# --- Scene gallery helpers (module-level) ---
def scene_matches_view(scene: 'Scene', view: str) -> bool:
    """Checks if a scene's status matches the current gallery view.

    Args:
        scene (Scene): The scene object.
        view (str): The current view ("All", "Kept", "Rejected").

    Returns:
        bool: True if the scene should be visible, False otherwise.
    """
    status = scene.status
    if view == "All":
        return status in ("included", "excluded", "pending")
    if view == "Kept":
        return status == "included"
    if view == "Rejected":
        return status == "excluded"
    return False

def scene_caption(s: 'Scene') -> str:
    """Generates a descriptive caption for a scene thumbnail in the gallery.

    Args:
        s (Scene): The scene object.

    Returns:
        str: A formatted caption string containing scene ID, frame range, and
        key metrics.
    """
    shot = s.shot_id
    start_f = s.start_frame
    end_f = s.end_frame
    metrics = s.seed_metrics or {}
    face = metrics.get('best_face_sim')
    conf = metrics.get('score')
    mask = (s.seed_result.get('details', {}) or {}).get('mask_area_pct')
    bits = [f"Scene {shot} [{start_f}-{end_f}]"]
    if s.is_overridden:
        bits[0] += " ✏️"
    else:
        bits[0] += " 🤖"
    if conf is not None: bits.append(f"conf {conf:.2f}")
    if face is not None: bits.append(f"face {face:.2f}")
    if mask is not None: bits.append(f"mask {mask:.1f}%")
    return " | ".join(bits)

def scene_thumb(s: 'Scene', output_dir: str) -> Optional[str]:
    """Finds the path to the preview thumbnail for a scene.

    Args:
        s (Scene): The scene object.
        output_dir (str): The main output directory for the run.

    Returns:
        Optional[str]: The path to the thumbnail image, or None if not found.
    """
    p = s.preview_path
    if p and os.path.isfile(p):
        return p
    shot_id = s.shot_id
    if shot_id is not None:
        candidate = os.path.join(output_dir, "previews", f"scene_{int(shot_id):05d}.jpg")
        if os.path.isfile(candidate):
            return candidate
    return None

def build_scene_gallery_items(scenes: list[dict], view: str, output_dir: str, page_num: int = 1, page_size: int = 20) -> tuple[list[tuple], list[int], int]:
    """Builds the list of items to display in the Gradio scene gallery.

    This function filters the scenes based on the current view, applies
    pagination, and generates the necessary data structure for the Gradio
    gallery, including image paths, captions, and an index map to link gallery
    items back to the original `scenes_state` list.

    Args:
        scenes (list[dict]): The list of all scene dictionaries.
        view (str): The current gallery view ("Kept", "Rejected", "All").
        output_dir (str): The main output directory for the run.
        page_num (int): The current page number (1-indexed). Defaults to 1.
        page_size (int): The number of items per page. Defaults to 20.

    Returns:
        tuple[list[tuple], list[int], int]: A tuple containing:
        - A list of (image_path, caption) tuples for the gallery.
        - An index map linking the gallery item index to the original scene
          list index.
        - The total number of pages.
    """
    items: list[tuple[str | None, str]] = []
    index_map: list[int] = []
    if not scenes:
        return [], [], 1

    previews_dir = Path(output_dir) / "previews"
    previews_dir.mkdir(parents=True, exist_ok=True)

    filtered_scenes = [(i, s) for i, s in enumerate(scenes) if scene_matches_view(s, view)]
    total_pages = max(1, (len(filtered_scenes) + page_size - 1) // page_size)
    page_num = max(1, min(page_num, total_pages))
    start_index = (page_num - 1) * page_size
    end_index = start_index + page_size
    paginated_scenes = filtered_scenes[start_index:end_index]


    for i, s in paginated_scenes:
        img_path = scene_thumb(s, output_dir)
        if img_path is None:
            continue

        if s.is_overridden:
            thumb_img_np = cv2.imread(img_path)
            if thumb_img_np is not None:
                badged_thumb = create_scene_thumbnail_with_badge(thumb_img_np, i, True)

                # Save the modified thumbnail to a new file
                shot_id = s.shot_id
                override_preview_path = previews_dir / f"scene_{shot_id:05d}_override.jpg"
                cv2.imwrite(str(override_preview_path), badged_thumb)
                img_path = str(override_preview_path)

        cap = scene_caption(s)
        items.append((img_path, cap))
        index_map.append(i)

    return items, index_map, total_pages
def create_scene_thumbnail_with_badge(scene_img: np.ndarray, scene_index: int, is_overridden: bool) -> np.ndarray:
    """Adds a visual indicator badge to a scene thumbnail image.

    This function draws a teal border and a pencil icon on a thumbnail if the
    scene has been manually overridden by the user, providing a clear visual
    cue in the gallery.

    Args:
        scene_img (np.ndarray): The scene thumbnail image as a NumPy array.
        scene_index (int): The index of the scene (unused in current
            implementation).
        is_overridden (bool): Boolean indicating if the scene has been
            manually changed.

    Returns:
        np.ndarray: The thumbnail image with the badge drawn on it.
    """
    thumb = scene_img.copy()
    h, w = thumb.shape[:2]

    if is_overridden:
        # Add teal border (4px thick)
        border_color = (33, 128, 141)  # var(--color-teal-500)
        cv2.rectangle(thumb, (0, 0), (w-1, h-1), border_color, 4)

        # Add "✏️" emoji badge in top-right corner
        badge_size = int(min(w, h) * 0.15)  # 15% of smallest dimension
        badge_pos = (w - badge_size - 5, 5)

        # Draw white circle background for badge
        cv2.circle(thumb,
                    (badge_pos[0] + badge_size//2, badge_pos[1] + badge_size//2),
                   badge_size//2, (255, 255, 255), -1)

        # Add text "✏" (pencil as ASCII alternative)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(thumb, "E", badge_pos, font, 0.5, border_color, 2)

    return thumb


# --- UI ---

class AppUI:
    """Handles the construction and event handling of the Gradio user interface.

    This class is responsible for defining the layout of the Gradio application,
    creating all the UI components (e.g., buttons, sliders, galleries), and
    wiring up the event handlers that connect the UI to the backend processing
    pipelines.

    Attributes:
        config (Config): The main application configuration.
        logger (AppLogger): The application logger.
        progress_queue (Queue): A queue for sending progress updates to the UI.
        cancel_event (threading.Event): An event to signal cancellation of
            long-running tasks.
        thumbnail_manager (ThumbnailManager): The manager for the thumbnail cache.
        components (dict): A dictionary to store created Gradio components.
        cuda_available (bool): A flag indicating if a CUDA-enabled GPU is
            available.
        ext_ui_map_keys (list[str]): A list of keys for the extraction UI
            components, used for mapping UI values to event objects.
        ana_ui_map_keys (list[str]): A list of keys for the analysis UI
            components.
        session_load_keys (list[str]): A list of keys for components that are
            updated when a session is loaded.
    """
    MAX_RESOLUTION_CHOICES: List[str] = ["maximum available", "2160", "1080", "720"]
    EXTRACTION_METHOD_TOGGLE_CHOICES: List[str] = ["Recommended Thumbnails", "Legacy Full-Frame"]
    METHOD_CHOICES: List[str] = ["keyframes", "interval", "every_nth_frame", "nth_plus_keyframes", "all"]
    PRIMARY_SEED_STRATEGY_CHOICES: List[str] = ["🤖 Automatic", "👤 By Face", "📝 By Text", "🔄 Face + Text Fallback", "🧑‍🤝‍🧑 Find Prominent Person"]
    SEED_STRATEGY_CHOICES: List[str] = ["Largest Person", "Center-most Person", "Highest Confidence", "Tallest Person", "Area x Confidence", "Rule-of-Thirds", "Edge-avoiding", "Balanced", "Best Face"]
    PERSON_DETECTOR_MODEL_CHOICES: List[str] = ['yolo11x.pt', 'yolo11s.pt']
    FACE_MODEL_NAME_CHOICES: List[str] = ["buffalo_l", "buffalo_s"]
    DAM4SAM_MODEL_NAME_CHOICES: List[str] = ["sam21pp-T", "sam21pp-S", "sam21pp-B+", "sam21pp-L"]
    GALLERY_VIEW_CHOICES: List[str] = ["Kept", "Rejected"]
    LOG_LEVEL_CHOICES: List[str] = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'SUCCESS', 'CRITICAL']
    SCENE_GALLERY_VIEW_CHOICES: List[str] = ["Kept", "Rejected", "All"]
    FILTER_PRESETS: Dict[str, Dict[str, float]] = {
        "Sharp Portraits": {
            "sharpness_min": 60.0,
            "sharpness_max": 100.0,
            "edge_strength_min": 50.0,
            "edge_strength_max": 100.0,
            "face_sim_min": 0.5,
            "mask_area_pct_min": 10.0,
            "eyes_open_min": 0.8,
            "yaw_min": -15.0,
            "yaw_max": 15.0,
            "pitch_min": -15.0,
            "pitch_max": 15.0,
        },
        "Close-up Subject": {
            "mask_area_pct_min": 25.0,
            "mask_area_pct_max": 100.0,
            "quality_score_min": 50.0,
        },
        "High Naturalness": {
            "niqe_min": 0.0,
            "niqe_max": 40.0,
            "contrast_min": 20.0,
            "contrast_max": 80.0,
            "brightness_min": 30.0,
            "brightness_max": 70.0,
        }
    }

    def __init__(self, config: 'Config', logger: 'AppLogger', progress_queue: Queue,
                 cancel_event: threading.Event, thumbnail_manager: 'ThumbnailManager'):
        """Initializes the AppUI class.

        Args:
            config (Config): The main application configuration.
            logger (AppLogger): The application logger.
            progress_queue (Queue): A queue for sending progress updates to the
                UI.
            cancel_event (threading.Event): An event to signal cancellation of
                long-running tasks.
            thumbnail_manager (ThumbnailManager): The manager for the thumbnail
                cache.
        """
        self.config = config
        self.logger = logger
        self.progress_queue = progress_queue
        self.cancel_event = cancel_event
        self.thumbnail_manager = thumbnail_manager
        self.components, self.cuda_available = {}, torch.cuda.is_available()
        self.ext_ui_map_keys = ['source_path', 'upload_video', 'method', 'interval', 'nth_frame',
                                'max_resolution', 'thumb_megapixels', 'scene_detect']
        self.ana_ui_map_keys = [
            'output_folder', 'video_path', 'resume', 'enable_face_filter', 'face_ref_img_path', 'face_ref_img_upload',
            'face_model_name', 'enable_subject_mask', 'dam4sam_model_name', 'person_detector_model', 'best_frame_strategy',
            'scene_detect', 'text_prompt', 'box_threshold', 'text_threshold', 'min_mask_area_pct',
            'sharpness_base_scale', 'edge_strength_base_scale', 'gdino_config_path', 'gdino_checkpoint_path',
            'pre_analysis_enabled', 'pre_sample_nth', 'primary_seed_strategy',
            'compute_quality_score', 'compute_sharpness', 'compute_edge_strength', 'compute_contrast',
            'compute_brightness', 'compute_entropy', 'compute_eyes_open', 'compute_yaw', 'compute_pitch',
            'compute_face_sim', 'compute_subject_mask_area', 'compute_niqe', 'compute_phash'
        ]
        self.session_load_keys = ['unified_log', 'unified_status', 'progress_details', 'cancel_button', 'pause_button',
                                  'source_input', 'max_resolution', 'thumb_megapixels_input', 'ext_scene_detect_input',
                                  'method_input', 'pre_analysis_enabled_input', 'pre_sample_nth_input', 'enable_face_filter_input',
                                  'face_ref_img_path_input', 'text_prompt_input', 'best_frame_strategy_input',
                                  'person_detector_model_input', 'dam4sam_model_name_input', 'extracted_video_path_state',
                                  'extracted_frames_dir_state', 'analysis_output_dir_state', 'analysis_metadata_path_state', 'scenes_state',
                                  'propagate_masks_button', 'seeding_results_column', 'propagation_group',
                                  'scene_filter_status', 'scene_face_sim_min_input', 'filtering_tab',
                                  'scene_gallery', 'scene_gallery_index_map_state']

    def build_ui(self) -> gr.Blocks:
        """
        Constructs the entire Gradio UI layout.

        Returns:
            The main `gr.Blocks` instance for the application.
        """
        css = """.gradio-gallery { overflow-y: hidden !important; }
                 .gradio-gallery img {
                    width: 100%;
                    height: 100%;
                    object-fit: scale-down;
                    object-position: top left;
                 }
.plot-and-slider-column { max-width: 560px !important; margin: auto; } .scene-editor { border: 1px solid #444; padding: 10px; border-radius: 5px; } .log-container > .gr-utils-error { display: none !important; } .progress-details { font-size: 1rem !important; color: #333 !import ant; font-weight: 500; padding: 8px 0; } .gr-progress .progress { height: 28px !important; }"""
        with gr.Blocks(theme=gr.themes.Default(), css=css) as demo:
            self._build_header()
            with gr.Accordion("🔄 resume previous Session", open=False):
                with gr.Row():
                    self._create_component('session_path_input', 'textbox', {'label': "Load previous run", 'placeholder': "Path to a previous run's output folder..."})
                    self._create_component('load_session_button', 'button', {'value': "📂 Load Session"})
                    self._create_component('save_config_button', 'button', {'value': "💾 Save Current Config"})
            with gr.Accordion("⚙️ System Diagnostics", open=False):
                self._create_component('run_diagnostics_button', 'button', {'value': "Run System Diagnostics"})
            self._build_main_tabs()
            self._build_footer()
            self._create_event_handlers()
        return demo

    def _get_comp(self, name: str) -> Optional[gr.components.Component]:
        """Safely retrieves a component from the internal components dictionary.

        Args:
            name (str): The name of the component to retrieve.

        Returns:
            Optional[gr.components.Component]: The Gradio component instance, or
            None if not found.
        """
        return self.components.get(name)

    def _create_component(self, name: str, comp_type: str, kwargs: dict) -> gr.components.Component:
        """Creates a Gradio component, stores it, and returns it.

        This helper method simplifies the creation and tracking of UI
        components by mapping a string `comp_type` to the corresponding Gradio
        class and storing the created instance in the `self.components`
        dictionary.

        Args:
            name (str): The key to use for storing the component.
            comp_type (str): The type of component to create (e.g., 'button').
            kwargs (dict): The keyword arguments to pass to the component's
                constructor.

        Returns:
            gr.components.Component: The created Gradio component instance.
        """
        comp_map = {'button': gr.Button, 'textbox': gr.Textbox, 'dropdown': gr.Dropdown, 'slider': gr.Slider, 'checkbox': gr.Checkbox,
                    'file': gr.File, 'radio': gr.Radio, 'gallery': gr.Gallery, 'plot': gr.Plot, 'markdown': gr.Markdown, 'html': gr.HTML, 'number': gr.Number, 'cbg': gr.CheckboxGroup, 'image': gr.Image}
        self.components[name] = comp_map[comp_type](**kwargs)
        return self.components[name]

    def _build_header(self):
        """Builds the header section of the UI."""
        gr.Markdown("# 🎬 Frame Extractor & Analyzer v2.0")
        status_color = "🟢" if self.cuda_available else "🟡"
        status_text = "GPU Accelerated" if self.cuda_available else "CPU Mode (Slower)"
        gr.Markdown(f"{status_color} **{status_text}**")
        if not self.cuda_available:
            gr.Markdown("⚠️ **CPU Mode** — GPU-dependent features are disabled or will be slow.")

    def _build_main_tabs(self):
        """Builds the main tabbed interface for the workflow."""
        with gr.Tabs() as main_tabs:
            self.components['main_tabs'] = main_tabs
            with gr.Tab("📹 1. Frame Extraction", id=0):
                self._create_extraction_tab()
            with gr.Tab("👩🏼‍🦰 2. Define Subject", id=1) as define_subject_tab:
                self.components['define_subject_tab'] = define_subject_tab
                self._create_define_subject_tab()
            with gr.Tab("🎞️ 3. Scene Selection", id=2) as scene_selection_tab:
                self.components['scene_selection_tab'] = scene_selection_tab
                self._create_scene_selection_tab()
            with gr.Tab("📝 4. Metrics", id=3) as metrics_tab:
                self.components['metrics_tab'] = metrics_tab
                self._create_metrics_tab()
            with gr.Tab("📊 5. Filtering & Export", id=4) as filtering_tab:
                self.components['filtering_tab'] = filtering_tab
                self._create_filtering_tab()

    def _build_footer(self):
        """Builds the footer section containing logs and status."""
        with gr.Row():
            with gr.Column(scale=2): self._create_component('unified_log', 'textbox', {'label': "📋 Processing Log", 'lines': 10, 'interactive': False, 'autoscroll': True})
            with gr.Column(scale=1):
                self._create_component('unified_status', 'textbox', {'label': "📊 Status Summary", 'lines': 2, 'interactive': False})

    def _create_extraction_tab(self):
        """Builds the UI components for the 'Frame Extraction' tab."""
        gr.Markdown("### Step 1: Provide a Video Source")
        with gr.Row():
            with gr.Column(scale=2): self._create_component('source_input', 'textbox', {'label': "Video URL or Local Path", 'placeholder': "Enter YouTube URL or local video file path", 'info': "The application can download videos directly from YouTube or use a video file you have on your computer."})
            with gr.Column(scale=1): self._create_component('max_resolution', 'dropdown', {'choices': self.MAX_RESOLUTION_CHOICES, 'value': self.config.default_max_resolution, 'label': "Max Download Resolution", 'info': "For YouTube videos, select the maximum resolution to download. 'Maximum available' will get the best quality possible."})
        self._create_component('upload_video_input', 'file', {'label': "Or Upload a Video File", 'file_types': ["video"], 'type': "filepath"})
        gr.Markdown("---"); gr.Markdown("### Step 2: Configure Extraction Method")

        with gr.Group(visible=True) as thumbnail_group:
            self.components['thumbnail_group'] = thumbnail_group
            gr.Markdown("**Thumbnail Extraction:** This is the fastest and most efficient way to process your video. It quickly extracts low-resolution, lightweight thumbnails for every frame. This allows you to perform scene analysis, find the best shots, and select your desired frames *before* extracting the final, full-resolution images. This workflow saves significant time and disk space.")
            with gr.Accordion("Advanced Settings", open=False):
                self._create_component('thumb_megapixels_input', 'slider', {
                    'label': "Thumbnail Size (MP)", 'minimum': 0.1, 'maximum': 2.0, 'step': 0.1,
                    'value': self.config.default_thumb_megapixels,
                    'info': "Controls the resolution of the extracted thumbnails. Higher values create larger, more detailed thumbnails but increase extraction time and disk usage. 0.5 MP is a good balance for most videos."
                })
                self._create_component('ext_scene_detect_input', 'checkbox', {
                    'label': "Use Scene Detection",
                    'value': self.config.default_scene_detect,
                    'info': "Automatically detects scene changes in the video. This is highly recommended as it groups frames into logical shots, making it much easier to find the best content in the next step."
                })
                self._create_component('method_input', 'dropdown', {
                    'choices': self.METHOD_CHOICES,
                    'value': self.config.default_method,
                    'label': "Frame Selection Method",
                    'info': """
                        - **Keyframes:** Extracts only the keyframes (I-frames). Good for a quick summary.
                        - **Interval:** Extracts one frame every X seconds.
                        - **Every Nth Frame:** Extracts one frame every N decoded frames.
                        - **Nth + Keyframes:** Keeps keyframes plus frames at a regular cadence.
                        - **All:** Extracts every single frame. (Warning: massive disk usage and time)."""
                })
                self._create_component('interval_input', 'number', {
                    'label': "Interval (seconds)",
                    'value': self.config.default_interval,
                    'minimum': 0.1,
                    'step': 0.1,
                    'visible': self.config.default_method == 'interval'
                })
                self._create_component('nth_frame_input', 'number', {
                    'label': "N-th Frame Value",
                    'value': self.config.default_nth_frame,
                    'minimum': 1,
                    'step': 1,
                    'visible': self.config.default_method in ['every_nth_frame', 'nth_plus_keyframes']
                })
        gr.Markdown("---"); gr.Markdown("### Step 3: Start Extraction")
        self.components.update({'start_extraction_button': gr.Button("🚀 Start Extraction", variant="primary")})

    def _create_define_subject_tab(self):
        """Builds the UI components for the 'Define Subject' tab."""
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 🎯 Step 1: Choose Your Seeding Strategy")
                gr.Markdown("""This step analyzes each scene to find the best frame and automatically detects people using YOLO. The system will: 1. Find the highest quality frame in each scene 2. Detect all people in that frame 3. Select the best subject based on your chosen strategy 4. Generate a preview with the subject highlighted""")
                self._create_component('primary_seed_strategy_input', 'radio', {'choices': self.PRIMARY_SEED_STRATEGY_CHOICES, 'value': self.config.default_primary_seed_strategy, 'label': "Primary Best-Frame Selection Strategy", 'info': "Select the main method for identifying the subject in each scene. This initial identification is called the 'best-frame selection'."})

                with gr.Group(visible="By Face" in self.config.default_primary_seed_strategy or "Fallback" in self.config.default_primary_seed_strategy) as face_seeding_group:
                    self.components['face_seeding_group'] = face_seeding_group
                    gr.Markdown("#### 👤 Configure Face Selection")
                    gr.Markdown("This strategy prioritizes finding a specific person. Upload a clear, frontal photo of the person you want to track. The system will analyze each scene to find the frame where this person is most clearly visible and use it as the starting point (the 'best frame').")
                    with gr.Row():
                        self._create_component('face_ref_img_upload_input', 'file', {'label': "Upload Face Reference Image", 'type': "filepath"})
                        self._create_component('face_ref_image', 'image', {'label': "Reference Image", 'interactive': False})
                        with gr.Column():
                            self._create_component('face_ref_img_path_input', 'textbox', {'label': "Or provide a local file path"})
                            self._create_component('enable_face_filter_input', 'checkbox', {'label': "Enable Face Similarity (must be checked for face selection)", 'value': self.config.default_enable_face_filter, 'interactive': True, 'visible': "By Face" in self.config.default_primary_seed_strategy or "Fallback" in self.config.default_primary_seed_strategy})
                    self._create_component('find_people_button', 'button', {'value': "Find People From Video"})
                    with gr.Group(visible=False) as discovered_people_group:
                        self.components['discovered_people_group'] = discovered_people_group
                        self._create_component('discovered_faces_gallery', 'gallery', {'label': "Discovered People", 'columns': 8, 'height': 'auto'})
                        self._create_component('identity_confidence_slider', 'slider', {'label': "Identity Confidence", 'minimum': 0.0, 'maximum': 1.0, 'step': 0.05, 'value': 0.5})

                with gr.Group(visible="By Text" in self.config.default_primary_seed_strategy or "Fallback" in self.config.default_primary_seed_strategy) as text_seeding_group:
                    self.components['text_seeding_group'] = text_seeding_group
                    gr.Markdown("#### 📝 Configure Text Selection")
                    gr.Markdown("This strategy uses a text description to find the subject. It's useful for identifying objects, or people described by their clothing or appearance when a reference photo isn't available.")
                    with gr.Accordion("🔬 Advanced Detection (GroundingDINO)", open=True):
                        gr.Markdown("Use GroundingDINO for text-based object detection with custom prompts.")
                        self._create_component('text_prompt_input', 'textbox', {'label': "Text Prompt", 'placeholder': "e.g., 'a woman in a red dress'", 'value': self.config.default_text_prompt, 'info': "Describe the main subject to find the best frame (e.g., 'player wearing number 10', 'person in the green shirt')."})
                        with gr.Row():
                            self._create_component('box_threshold', 'slider', {'minimum': 0.0, 'maximum': 1.0, 'value': self.config.gdino_box_threshold, 'label': "Box Threshold", 'interactive': True})
                            self._create_component('text_threshold', 'slider', {'minimum': 0.0, 'maximum': 1.0, 'value': self.config.gdino_text_threshold, 'label': "Text Threshold", 'interactive': True})

                with gr.Group(visible="Prominent Person" in self.config.default_primary_seed_strategy) as auto_seeding_group:
                    self.components['auto_seeding_group'] = auto_seeding_group
                    gr.Markdown("#### 🧑‍🤝‍🧑 Configure Prominent Person Selection")
                    gr.Markdown("This is a simple, fully automatic mode. It uses an object detector (YOLO) to find all people in the scene and then selects one based on a simple rule, like who is largest or most central. It's fast but less precise, as it doesn't use face identity or text descriptions.")
                    self._create_component('best_frame_strategy_input', 'dropdown', {'choices': self.SEED_STRATEGY_CHOICES, 'value': "Largest Person", 'label': "Selection Method", 'info': "'Largest' picks the person with the biggest bounding box. 'Center-most' picks the person closest to the center. 'Highest Confidence' selects the person with the highest detection confidence. 'Tallest Person' prefers subjects that are standing. 'Area x Confidence' balances size and confidence. 'Rule-of-Thirds' prefers subjects near the thirds lines. 'Edge-avoiding' avoids subjects near the frame's edge. 'Balanced' provides a good mix of area, confidence, and edge-avoidance. 'Best Face' selects the person with the highest quality face detection."})

                self._create_component('person_radio', 'radio', {'label': "Select Person", 'choices': [], 'visible': False})

                with gr.Accordion("Advanced Settings", open=False):
                    gr.Markdown("These settings control the underlying models and analysis parameters. Adjust them only if you understand their effect.")
                    self._create_component('pre_analysis_enabled_input', 'checkbox', {'label': 'Enable Pre-Analysis to find best frame', 'value': self.config.default_pre_analysis_enabled, 'info': "Analyzes a subset of frames in each scene to automatically find the highest quality frame to use as the 'best frame' for masking. Highly recommended."})
                    self._create_component('pre_sample_nth_input', 'number', {'label': 'Sample every Nth thumbnail for pre-analysis', 'value': self.config.default_pre_sample_nth, 'interactive': True, 'info': "For faster pre-analysis, check every Nth frame in a scene instead of all of them. A value of 5 is a good starting point."})
                    self._create_component('person_detector_model_input', 'dropdown', {'choices': self.PERSON_DETECTOR_MODEL_CHOICES, 'value': self.config.default_person_detector_model, 'label': "Person Detector Model", 'info': "YOLO Model for finding people. 'x' (large) is more accurate but slower; 's' (small) is much faster but may miss people."})
                    self._create_component('face_model_name_input', 'dropdown', {'choices': self.FACE_MODEL_NAME_CHOICES, 'value': self.config.default_face_model_name, 'label': "Face Recognition Model", 'info': "InsightFace model for face matching. 'l' (large) is more accurate; 's' (small) is faster and uses less memory."})
                    self._create_component('dam4sam_model_name_input', 'dropdown', {'choices': self.DAM4SAM_MODEL_NAME_CHOICES, 'value': self.config.default_dam4sam_model_name, 'label': "Mask Tracking Model", 'info': "The Segment Anything 2 model used for tracking the subject mask across frames. Larger models (L) are more robust but use more VRAM; smaller models (T) are faster."})
                    self._create_component('resume_input', 'checkbox', {'label': 'Resume', 'value': self.config.default_resume, 'interactive': True, 'visible': False})
                    self._create_component('enable_subject_mask_input', 'checkbox', {'label': 'Enable Subject Mask', 'value': self.config.default_enable_subject_mask, 'interactive': True, 'visible': False})
                    self._create_component('min_mask_area_pct_input', 'slider', {'label': 'Min Mask Area Pct', 'value': self.config.min_mask_area_pct, 'interactive': True, 'visible': False})
                    self._create_component('sharpness_base_scale_input', 'slider', {'label': 'Sharpness Base Scale', 'value': self.config.sharpness_base_scale, 'interactive': True, 'visible': False})
                    self._create_component('edge_strength_base_scale_input', 'slider', {'label': 'Edge Strength Base Scale', 'value': self.config.edge_strength_base_scale, 'interactive': True, 'visible': False})
                    self._create_component('gdino_config_path_input', 'textbox', {'label': 'GroundingDINO Config Path', 'value': self.config.grounding_dino_config_path, 'interactive': True, 'visible': False})
                    self._create_component('gdino_checkpoint_path_input', 'textbox', {'label': 'GroundingDINO Checkpoint Path', 'value': self.config.grounding_dino_checkpoint_path, 'interactive': True, 'visible': False})

                self._create_component('start_pre_analysis_button', 'button', {'value': '🌱 Find & Preview Best Frames', 'variant': 'primary'})

                with gr.Group(visible=False) as propagation_group:
                    self.components['propagation_group'] = propagation_group

    def _create_scene_selection_tab(self):
        """Builds the UI components for the 'Scene Selection' tab."""
        with gr.Column(scale=2, visible=False) as seeding_results_column:
            self.components['seeding_results_column'] = seeding_results_column
            gr.Markdown("""### 🎭 Step 2: Review & Refine Scene Selection
Review the automatically detected subjects and refine the selection if needed. Each scene shows the best frame with the selected subject highlighted.""")

            with gr.Accordion("Scene Filtering", open=True):
                self._create_component('scene_filter_status', 'markdown', {'value': 'No scenes loaded.'})
                with gr.Row():
                    self._create_component('scene_mask_area_min_input', 'slider', {'label': "Min Best Frame Mask Area %", 'minimum': 0.0, 'maximum': 100.0, 'value': self.config.min_mask_area_pct, 'step': 0.1})
                    self._create_component('scene_face_sim_min_input', 'slider', {'label': "Min Best Frame Face Sim", 'minimum': 0.0, 'maximum': 1.0, 'value': 0.0, 'step': 0.05, 'visible': False})
                    self._create_component('scene_confidence_min_input', 'slider', {'label': "Min Best Frame Confidence", 'minimum': 0.0, 'maximum': 1.0, 'value': 0.0, 'step': 0.05})

            with gr.Accordion("Scene Gallery", open=True):
                self._create_component('scene_gallery_view_toggle', 'radio', {'label': "Show", 'choices': ["Kept", "Rejected", "All"], 'value': "Kept"})
                with gr.Row(elem_id="pagination_row"):
                    self._create_component('prev_page_button', 'button', {'value': '⬅️ Previous'})
                    self._create_component('page_number_input', 'number', {'label': 'Page', 'value': 1, 'precision': 0})
                    self._create_component('total_pages_label', 'markdown', {'value': '/ 1 pages'})
                    self._create_component('next_page_button', 'button', {'value': 'Next ➡️'})
                self.components['scene_gallery'] = gr.Gallery(label="Scenes", columns=10, rows=2, height=560, show_label=True, allow_preview=True, container=True)

            with gr.Accordion("Scene Editor", open=False, elem_classes="scene-editor") as sceneeditoraccordion:
                self.components["sceneeditoraccordion"] = sceneeditoraccordion
                self._create_component("sceneeditorstatusmd", "markdown", {"value": "Select a scene to edit."})

                with gr.Group() as yolo_seed_group:
                    self.components['yolo_seed_group'] = yolo_seed_group
                    self._create_component('scene_editor_yolo_subject_id', 'radio', {'label': "Detected Subjects", 'info': "Select the auto-detected subject to use for seeding.", 'interactive': True, 'choices': [], 'visible': False})

                with gr.Accordion("Advanced Seeding (optional)", open=False):
                    gr.Markdown("Use a text prompt for seeding. This will override the YOLO detection above.")
                    self._create_component("sceneeditorpromptinput", "textbox", {"label": "DINO Text Prompt", "info": "e.g., 'person in a red shirt'"})
                    info_box = "Confidence for detecting an object's bounding box. Higher = fewer, more confident detections."
                    self._create_component("sceneeditorboxthreshinput", "slider", {"label": "Box Thresh", "minimum": 0.0, "maximum": 1.0, "step": 0.05, "info": info_box, "value": self.config.gdino_box_threshold})
                    info_text = "Confidence for matching the prompt to an object. Higher = stricter text match."
                    self._create_component("sceneeditortextthreshinput", "slider", {"label": "Text Thresh", "minimum": 0.0, "maximum": 1.0, "step": 0.05, "info": info_text, "value": self.config.gdino_text_threshold})

                with gr.Row():
                    self._create_component("scenerecomputebutton", "button", {"value": "▶️ Recompute Preview"})
                    self._create_component("sceneincludebutton", "button", {"value": "✅ Keep Scene"})
                    self._create_component("sceneexcludebutton", "button", {"value": "❌ Reject Scene"})
                    self._create_component("sceneresetbutton", "button", {"value": "🔄 Reset Scene"})

            gr.Markdown("---"); gr.Markdown("### 🔬 Step 3: Propagate Masks"); gr.Markdown("Once you are satisfied with the seeds, propagate the masks to the rest of the frames in the selected scenes.")
            self._create_component('propagate_masks_button', 'button', {'value': '🔬 Propagate Masks on Kept Scenes', 'variant': 'primary', 'interactive': False})

    def _create_metrics_tab(self):
        """Builds the UI components for the 'Metrics' tab."""
        gr.Markdown("### Step 4: Select Metrics to Compute")
        gr.Markdown("Choose which metrics to calculate during the analysis phase. More metrics provide more filtering options but may increase processing time.")

        with gr.Row():
            with gr.Column():
                self._create_component('compute_quality_score', 'checkbox', {'label': "Quality Score", 'value': True})
                self._create_component('compute_sharpness', 'checkbox', {'label': "Sharpness", 'value': True})
                self._create_component('compute_edge_strength', 'checkbox', {'label': "Edge Strength", 'value': True})
                self._create_component('compute_contrast', 'checkbox', {'label': "Contrast", 'value': True})
                self._create_component('compute_brightness', 'checkbox', {'label': "Brightness", 'value': True})
                self._create_component('compute_entropy', 'checkbox', {'label': "Entropy", 'value': True})
            with gr.Column():
                self._create_component('compute_eyes_open', 'checkbox', {'label': "Eyes Open", 'value': True})
                self._create_component('compute_yaw', 'checkbox', {'label': "Yaw", 'value': True})
                self._create_component('compute_pitch', 'checkbox', {'label': "Pitch", 'value': True})
                self._create_component('compute_face_sim', 'checkbox', {'label': "Face Similarity", 'value': True})
                self._create_component('compute_subject_mask_area', 'checkbox', {'label': "Subject Mask Area", 'value': True})
                self._create_component('compute_niqe', 'checkbox', {'label': "NIQE", 'value': pyiqa is not None, 'interactive': pyiqa is not None, 'info': "Requires 'pyiqa' to be installed."})

        with gr.Accordion("Deduplication Settings", open=True):
            self._create_component('compute_phash', 'checkbox', {'label': "Compute p-hash for Deduplication", 'value': True})
        self.components['start_analysis_button'] = gr.Button("Analyze Selected Frames", variant="primary")


    def _create_analysis_tab(self):
        """Placeholder for a potential, more detailed analysis tab."""
        pass

    def _create_filtering_tab(self):
        """Builds the UI components for the 'Filtering & Export' tab."""
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 🎛️ Filter Controls")
                gr.Markdown("Use these controls to refine your selection of frames. You can set minimum and maximum thresholds for various quality metrics.")
                self._create_component('filter_preset_dropdown', 'dropdown', {'label': "Filter Presets", 'choices': ["None"] + list(self.FILTER_PRESETS.keys())})
                self._create_component('auto_pctl_input', 'slider', {'label': 'Auto-Threshold Percentile', 'minimum': 1, 'maximum': 99, 'value': self.config.gradio_auto_pctl_input, 'step': 1, 'info': "Quickly set all 'Min' sliders to a certain percentile of the data. For example, setting this to 75 and clicking 'Apply' will automatically reject the bottom 75% of frames for each metric."})
                with gr.Row():
                    self._create_component('apply_auto_button', 'button', {'value': 'Apply Percentile to Mins'})
                    self._create_component('reset_filters_button', 'button', {'value': "Reset Filters"})
                with gr.Row():
                    self._create_component('expand_all_metrics_button', 'button', {'value': 'Expand All'})
                    self._create_component('collapse_all_metrics_button', 'button', {'value': 'Collapse All'})
                self._create_component('filter_status_text', 'markdown', {'value': "Load an analysis to begin."})
                self.components['metric_plots'], self.components['metric_sliders'], self.components['metric_accs'], self.components['metric_auto_threshold_cbs'] = {}, {}, {}, {}

                with gr.Accordion("Deduplication", open=True, visible=True) as dedup_acc:
                    self.components['metric_accs']['dedup'] = dedup_acc
                    f_def = self.config.filter_default_dedup_thresh
                    self._create_component('dedup_method_input', 'dropdown', {'label': "Deduplication Method", 'choices': ["None", "pHash", "SSIM", "LPIPS", "pHash then LPIPS"], 'value': "pHash"})
                    self._create_component('dedup_thresh_input', 'slider', {'label': "pHash Threshold", 'minimum': f_def['min'], 'maximum': f_def['max'], 'value': f_def['default'], 'step': f_def['step'], 'info': "Filters out visually similar frames. A lower value is stricter (more filtering). A value of 0 means only identical images will be removed. Set to -1 to disable."})
                    self._create_component('ssim_threshold_input', 'slider', {'label': "SSIM Threshold", 'minimum': 0.0, 'maximum': 1.0, 'value': 0.95, 'step': 0.01, 'visible': False})
                    self._create_component('lpips_threshold_input', 'slider', {'label': "LPIPS Threshold", 'minimum': 0.0, 'maximum': 1.0, 'value': 0.1, 'step': 0.01, 'visible': False})
                    self._create_component('dedup_visual_diff_input', 'checkbox', {'label': "Enable Visual Diff", 'value': False})
                    self._create_component('visual_diff_image', 'image', {'label': "Visual Diff", 'visible': False})
                    self._create_component('calculate_diff_button', 'button', {'value': "Calculate Diff", 'visible': False})

                metric_configs = {
                    'quality_score': {'open': True}, 'niqe': {'open': False}, 'sharpness': {'open': True},
                    'edge_strength': {'open': True}, 'contrast': {'open': True}, 'brightness': {'open': False},
                    'entropy': {'open': False}, 'face_sim': {'open': False}, 'mask_area_pct': {'open': False},
                    'eyes_open': {'open': True}, 'yaw': {'open': True}, 'pitch': {'open': True}
                }
                for metric_name, metric_config in metric_configs.items():
                    if not hasattr(self.config, f"filter_default_{metric_name}"): continue
                    f_def = getattr(self.config, f"filter_default_{metric_name}")
                    with gr.Accordion(metric_name.replace('_', ' ').title(), open=metric_config['open'], visible=False) as acc:
                        self.components['metric_accs'][metric_name] = acc
                        gr.Markdown(self.get_metric_description(metric_name), elem_classes="metric-description")
                        with gr.Column(elem_classes="plot-and-slider-column"):
                            self.components['metric_plots'][metric_name] = self._create_component(f'plot_{metric_name}', 'html', {'visible': True})
                            self.components['metric_sliders'][f"{metric_name}_min"] = self._create_component(f'slider_{metric_name}_min', 'slider', {'label': "Min", 'minimum': f_def['min'], 'maximum': f_def['max'], 'value': f_def.get('default_min', f_def['min']), 'step': f_def['step'], 'interactive': True, 'visible': True})
                            if 'default_max' in f_def: self.components['metric_sliders'][f"{metric_name}_max"] = self._create_component(f'slider_{metric_name}_max', 'slider', {'label': "Max", 'minimum': f_def['min'], 'maximum': f_def['max'], 'value': f_def['default_max'], 'step': f_def['step'], 'interactive': True, 'visible': True})
                            self.components['metric_auto_threshold_cbs'][metric_name] = self._create_component(f'auto_threshold_{metric_name}', 'checkbox', {'label': "Auto-Threshold this metric", 'value': False, 'interactive': True, 'visible': True})

                            if metric_name == "face_sim": self._create_component('require_face_match_input', 'checkbox', {'label': "Reject if no face", 'value': self.config.default_require_face_match, 'visible': True, 'info': "If checked, any frame without a detected face that meets the similarity threshold will be rejected."})
            with gr.Column(scale=2):
                with gr.Group(visible=False) as results_group:
                    self.components['results_group'] = results_group
                    gr.Markdown("### 🖼️ Step 2: Review Results")
                    with gr.Row():
                        self._create_component('gallery_view_toggle', 'radio', {'choices': self.GALLERY_VIEW_CHOICES, 'value': "Kept", 'label': "Show in Gallery"})
                        self._create_component('show_mask_overlay_input', 'checkbox', {'label': "Show Mask Overlay", 'value': self.config.gradio_show_mask_overlay})
                        self._create_component('overlay_alpha_slider', 'slider', {'label': "Overlay Alpha", 'minimum': 0.0, 'maximum': 1.0, 'value': self.config.gradio_overlay_alpha, 'step': 0.1})
                    self._create_component('results_gallery', 'gallery', {'columns': [4, 6, 8], 'rows': 2, 'height': 'auto', 'preview': True, 'allow_preview': True, 'object_fit': 'contain'})
                with gr.Group(visible=False) as export_group:
                    self.components['export_group'] = export_group
                    gr.Markdown("### 📤 Step 3: Export")
                    with gr.Row():
                        self._create_component('export_button', 'button', {'value': "Export Kept Frames", 'variant': "primary"})
                        self._create_component('dry_run_button', 'button', {'value': "Dry Run Export"})
                    with gr.Accordion("Export Options", open=True):
                        with gr.Row():
                            self._create_component('enable_crop_input', 'checkbox', {'label': "✂️ Crop to Subject", 'value': self.config.export_enable_crop})
                            self._create_component('crop_padding_input', 'slider', {'label': "Padding %", 'value': self.config.export_crop_padding})
                        self._create_component('crop_ar_input', 'textbox', {'label': "Crop ARs", 'value': self.config.export_crop_ars, 'info': "Comma-separated list (e.g., 16:9, 1:1). The best-fitting AR for each subject's mask will be chosen automatically."})

    def get_all_filter_keys(self) -> list[str]:
        """Returns a list of all available metric keys for filtering.

        This method centralizes the list of metrics that can be used in the
        filtering tab, ensuring consistency between data preparation and UI
        construction.

        Returns:
            list[str]: A list of all available metric keys.
        """
        return list(self.config.quality_weights.model_dump().keys()) + ["quality_score", "face_sim", "mask_area_pct", "eyes_open", "yaw", "pitch"]

    def get_metric_description(self, metric_name: str) -> str:
        """Provides a user-friendly description for a given metric name.

        Args:
            metric_name (str): The internal name of the metric.

        Returns:
            str: A string containing the description for display in the UI.
        """
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
        """Initializes all Gradio event handlers and state variables."""
        self.components.update({'extracted_video_path_state': gr.State(""), 'extracted_frames_dir_state': gr.State(""),
                                'analysis_output_dir_state': gr.State(""), 'analysis_metadata_path_state': gr.State(""),
                                'all_frames_data_state': gr.State([]), 'per_metric_values_state': gr.State({}),
                                'scenes_state': gr.State([]), 'selected_scene_id_state': gr.State(None),
                            'scene_gallery_index_map_state': gr.State([]),
                            'gallery_image_state': gr.State(None),
                            'gallery_shape_state': gr.State(None),
                            'yolo_results_state': gr.State({}),
                            'discovered_faces_state': gr.State([]),
                            'resume_state': gr.State(False),
                            'enable_subject_mask_state': gr.State(True),
                            'min_mask_area_pct_state': gr.State(1.0),
                            'sharpness_base_scale_state': gr.State(2500.0),
                            'edge_strength_base_scale_state': gr.State(100.0),
                            'gdino_config_path_state': gr.State("GroundingDINO_SwinT_OGC.py"),
                            'gdino_checkpoint_path_state': gr.State("models/groundingdino_swint_ogc.pth"),
                            })
        self._setup_visibility_toggles(); self._setup_pipeline_handlers(); self._setup_filtering_handlers(); self._setup_bulk_scene_handlers()
        self.components['save_config_button'].click(
            lambda: self.config.save_config('config_dump.json'), [], []
        ).then(lambda: "Configuration saved to config_dump.json", [], self.components['unified_log'])


class EnhancedAppUI(AppUI):
    """Extends the base AppUI with an enhanced footer and progress handling.

    This class inherits from `AppUI` and overrides the `_build_footer` method
    to create a more sophisticated status and logging area. It also introduces
    the `_run_task_with_progress` method, which is a crucial wrapper for
    running all long-running backend tasks in a separate thread while providing
    real-time progress updates to the UI.
    """
    def __init__(self, config: 'Config', logger: 'AppLogger', progress_queue: Queue,
                 cancel_event: threading.Event, thumbnail_manager: 'ThumbnailManager'):
        """Initializes the EnhancedAppUI.

        Args:
            config: The main application configuration.
            logger: The application logger.
            progress_queue: A queue for sending progress updates to the UI.
            cancel_event: An event to signal cancellation of long-running tasks.
            thumbnail_manager: The manager for the thumbnail cache.
        """
        super().__init__(config, logger, progress_queue, cancel_event, thumbnail_manager)
        self.app_logger = logger
        self.performance_metrics, self.log_filter_level, self.all_logs = {}, "INFO", []
        self.last_run_args = None

    def _build_footer(self):
        """Builds the enhanced footer with a unified status panel and detailed logs."""
        with gr.Row():
            with gr.Column(scale=2):
                self._create_component('unified_status', 'markdown', {'label': "📊 Status & Messages", 'value': "Welcome! Ready to start."})
                self.components['progress_bar'] = gr.Progress()
                self._create_component('progress_details', 'html', {'value': '', 'elem_classes': ['progress-details']})
                with gr.Row():
                    self._create_component('pause_button', 'button', {'value': '⏸️ Pause', 'interactive': False})
                    self._create_component('cancel_button', 'button', {'value': '⏹️ Cancel', 'interactive': False})
            with gr.Column(scale=3):
                with gr.Accordion("📋 Verbose Processing Log (for debugging)", open=False):
                    self._create_component('unified_log', 'textbox', {'lines': 15, 'interactive': False, 'autoscroll': True, 'elem_classes': ['log-container']})
                    with gr.Row():
                        self._create_component('log_level_filter', 'dropdown', {'choices': self.LOG_LEVEL_CHOICES, 'value': 'INFO', 'label': 'Log Level', 'scale': 1})
                        self._create_component('clear_logs_button', 'button', {'value': '🗑️ Clear', 'scale': 1})
                        self._create_component('export_logs_button', 'button', {'value': '📥 Export', 'scale': 1})

    def _format_metric_card(self, label: str, value: str) -> str:
        """Formats a simple HTML card for displaying a metric.

        Args:
            label (str): The label for the metric.
            value (str): The value of the metric.

        Returns:
            str: An HTML string representing the metric card.
        """
        return f"""<div class="metric-card"><div class="metric-value">{value}</div><div class="metric-label">{label}</div></div>"""

    def _run_task_with_progress(self, task_func: Callable, output_components: list, progress: Callable, *args) -> Generator[dict, None, None]:
        """Runs a pipeline function in a background thread and updates the UI.

        This is a master wrapper function for all long-running pipeline tasks.
        It handles:
        - Running the target function in a `ThreadPoolExecutor`.
        - Enabling the cancel and pause buttons.
        - Polling a queue for real-time log and progress updates and yielding
          them to the Gradio UI.
        - Handling task completion, cancellation, and exceptions.
        - Formatting the final status message for the UI.

        Args:
            task_func (Callable): The pipeline function to execute.
            output_components (list): A list of Gradio components that the task
                might update.
            progress (Callable): The Gradio `gr.Progress` object.
            *args: Arguments to be passed to the `task_func`.

        Yields:
            dict: Dictionaries of Gradio updates for various UI components.
        """
        self.last_run_args = args
        self.cancel_event.clear()
        tracker_instance = next((arg for arg in args if isinstance(arg, AdvancedProgressTracker)), None)
        if tracker_instance:
            tracker_instance.pause_event.set()

        op_name = getattr(task_func, '__name__', 'Unknown Task').replace('_wrapper', '').replace('_', ' ').title()
        yield {
            self.components['cancel_button']: gr.update(interactive=True),
            self.components['pause_button']: gr.update(interactive=True),
            self.components['unified_status']: f"🚀 **Starting: {op_name}...**"
        }

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(task_func, *args)
            start_time = time.time()
            while future.running():
                if time.time() - start_time > 3600:
                    self.app_logger.error("Task timed out after 1 hour")
                    self.cancel_event.set()
                    future.cancel()
                    break
                if self.cancel_event.is_set():
                    future.cancel()
                    break
                if tracker_instance and not tracker_instance.pause_event.is_set():
                    yield {self.components['unified_status']: f"⏸️ **Paused: {op_name}**"}
                    time.sleep(0.2)
                    continue

                try:
                    msg, update_dict = self.progress_queue.get(timeout=0.1), {}
                    if "log" in msg:
                        self.all_logs.append(msg['log'])
                        log_level_map = {level: i for i, level in enumerate(self.LOG_LEVEL_CHOICES)}
                        current_filter_level = log_level_map.get(self.log_filter_level.upper(), 1)
                        filtered_logs = [
                            l for l in self.all_logs
                            if any(f"[{level}]" in l for level in self.LOG_LEVEL_CHOICES[current_filter_level:])
                        ]
                        update_dict[self.components['unified_log']] = "\n".join(filtered_logs[-1000:])
                    if "progress" in msg:
                        p = ProgressEvent(**msg["progress"])
                        progress(p.fraction, desc=f"{p.stage} ({p.done}/{p.total}) • {p.eta_formatted}")
                        status_md = f"**Running: {op_name}**\n- Stage: {p.stage} ({p.done}/{p.total})\n- ETA: {p.eta_formatted}"
                        if p.substage: status_md += f"\n- Step: {p.substage}"
                        update_dict[self.components['unified_status']] = status_md
                    if update_dict: yield update_dict
                except Empty:
                    pass
                time.sleep(0.05)

        final_updates, final_status, final_label, final_icon = {}, "Task completed successfully.", "Complete", "✅"
        try:
            if self.cancel_event.is_set():
                final_status, final_label, final_icon = "Task cancelled by user.", "Cancelled", "⏹️"
                if tracker_instance: tracker_instance.done_stage(final_label)
            else:
                result_gen = future.result()
                result_dict = deque(result_gen, maxlen=1)[0] if result_gen else {}
                final_status = result_dict.get("unified_log", final_status)
                final_updates = result_dict
        except Exception as e:
            self.app_logger.error("Task execution failed in UI runner", exc_info=True)
            final_status, final_label, final_icon = f"Task failed: {e}", "Failed", "❌"
            if tracker_instance: tracker_instance.set_stage(final_label, substage=str(e))

        status_message = final_updates.pop("status_message", f"{op_name} {final_label.lower()}.")
        error_message = final_updates.pop("error_message", None)
        remediation_hint = final_updates.pop("remediation_hint", None)

        final_markdown = f"{final_icon} **{final_label}: {status_message}**"
        if error_message:
            final_markdown += f"\n- **Details:** `{error_message}`"
        if remediation_hint:
            final_markdown += f"\n- **Suggestion:** {remediation_hint}"

        self.all_logs.append(f"[{final_label.upper()}] {final_status}")
        filtered_logs = [l for l in self.all_logs if self.log_filter_level.upper() == "DEBUG" or f"[{self.log_filter_level.upper()}]" in l]
        final_updates_with_comps = {self.components.get(k): v for k, v in final_updates.items() if self.components.get(k)}
        final_updates_with_comps[self.components['unified_log']] = "\n".join(filtered_logs[-1000:])
        progress(1.0, final_label)
        final_updates_with_comps[self.components['unified_status']] = final_markdown
        final_updates_with_comps[self.components['progress_details']] = ""
        final_updates_with_comps[self.components['cancel_button']], final_updates_with_comps[self.components['pause_button']] = gr.update(interactive=False), gr.update(interactive=False)
        yield final_updates_with_comps

    def _scale_xywh(self, xywh: list[int], src_hw: tuple[int, int], dst_hw: tuple[int, int]) -> list[int]:
        """Scales a bounding box from a source to a destination resolution.

        Args:
            xywh (list[int]): The input bounding box in [x, y, w, h] format.
            src_hw (tuple[int, int]): The source (height, width).
            dst_hw (tuple[int, int]): The destination (height, width).

        Returns:
            list[int]: The scaled bounding box.
        """
        x, y, w, h = map(int, xywh)
        src_h, src_w = src_hw
        dst_h, dst_w = dst_hw
        sx = dst_w / max(1, src_w)
        sy = dst_h / max(1, src_h)
        return [int(round(x * sx)), int(round(y * sy)),
                int(round(w * sx)), int(round(h * sy))]

    def on_select_yolo_subject_wrapper(self, subject_id: str, scenes: list, shot_id: int, outdir: str, view: str, *ana_args) -> tuple:
        """Wrapper for handling subject selection from the YOLO radio buttons.

        Args:
            subject_id (str): The ID of the selected subject from the radio button.
            scenes (list): The list of all scene objects.
            shot_id (int): The ID of the currently selected scene.
            outdir (str): The path to the output directory.
            view (str): The current gallery view.
            *ana_args: The rest of the analysis UI component values.

        Returns:
            tuple: A tuple of Gradio updates for the UI.
        """
        try:
            if not subject_id: return scenes, gr.update(), gr.update(), "Please select a Subject ID."
            subject_idx = int(subject_id) - 1
            scene = next((s for s in scenes if s['shot_id'] == shot_id), None)
            if not scene: return scenes, gr.update(), gr.update(), "Scene not found."
            yolo_boxes = scene.get('yolo_detections', [])
            if not (0 <= subject_idx < len(yolo_boxes)): return scenes, gr.update(), gr.update(), f"Invalid Subject ID. Please enter a number between 1 and {len(yolo_boxes)}."
            masker = _create_analysis_context(self.config, self.logger, self.thumbnail_manager, self.ana_ui_map_keys, ana_args)
            selected_box = yolo_boxes[subject_idx]
            selected_xywh = masker.seed_selector._xyxy_to_xywh(selected_box['bbox'])
            overrides = {"manual_bbox_xywh": selected_xywh, "seedtype": "yolo_manual"}
            scene_idx = scenes.index(scene)
            if 'initial_bbox' not in scenes[scene_idx] or scenes[scene_idx]['initial_bbox'] is None:
                scenes[scene_idx]['initial_bbox'] = selected_xywh
            scenes[scene_idx]['selected_bbox'] = selected_xywh
            initial_bbox = scenes[scene_idx].get('initial_bbox')
            scenes[scene_idx]['is_overridden'] = initial_bbox is not None and selected_xywh != initial_bbox
            _recompute_single_preview(scenes[scene_idx], masker, overrides, self.thumbnail_manager, self.logger)
            save_scene_seeds(scenes, outdir, self.logger)
            gallery_items, index_map = build_scene_gallery_items(scenes, view, outdir)
            return scenes, gr.update(value=gallery_items), gr.update(value=index_map), f"Subject {subject_id} selected and preview recomputed."
        except (ValueError, TypeError):
            gallery_items, index_map = build_scene_gallery_items(scenes, view, outdir)
            return scenes, gr.update(value=gallery_items), gr.update(value=index_map), "Invalid Subject ID. Please enter a number."
        except Exception as e:
            self.logger.error("Failed to select YOLO subject", exc_info=True)
            gallery_items, index_map = build_scene_gallery_items(scenes, view, outdir)
            return scenes, gr.update(value=gallery_items), gr.update(value=index_map), f"Error: {e}"

    def _setup_bulk_scene_handlers(self):
        """Sets up Gradio event handlers for the scene selection and editing tab."""
        c = self.components

        def on_page_change(scenes, view, output_dir, page_num):
            page_num = int(page_num)
            items, index_map, total_pages = build_scene_gallery_items(scenes, view, output_dir, page_num=page_num)
            return gr.update(value=items), index_map, f"/ {total_pages} pages", page_num

        def _refresh_scene_gallery(scenes, view, output_dir):
            items, index_map, total_pages = build_scene_gallery_items(scenes, view, output_dir, page_num=1)
            return gr.update(value=items), index_map, f"/ {total_pages} pages", 1

        # On view toggle change
        c['scene_gallery_view_toggle'].change(
            _refresh_scene_gallery,
            [c['scenes_state'], c['scene_gallery_view_toggle'], c['extracted_frames_dir_state']],
            [c['scene_gallery'], c['scene_gallery_index_map_state'], c['total_pages_label'], c['page_number_input']]
        )

        # Pagination controls
        c['next_page_button'].click(
            lambda scenes, view, output_dir, page_num: on_page_change(scenes, view, output_dir, page_num + 1),
            [c['scenes_state'], c['scene_gallery_view_toggle'], c['extracted_frames_dir_state'], c['page_number_input']],
            [c['scene_gallery'], c['scene_gallery_index_map_state'], c['total_pages_label'], c['page_number_input']]
        )

        c['prev_page_button'].click(
            lambda scenes, view, output_dir, page_num: on_page_change(scenes, view, output_dir, page_num - 1),
            [c['scenes_state'], c['scene_gallery_view_toggle'], c['extracted_frames_dir_state'], c['page_number_input']],
            [c['scene_gallery'], c['scene_gallery_index_map_state'], c['total_pages_label'], c['page_number_input']]
        )

        c['page_number_input'].submit(
            on_page_change,
            [c['scenes_state'], c['scene_gallery_view_toggle'], c['extracted_frames_dir_state'], c['page_number_input']],
            [c['scene_gallery'], c['scene_gallery_index_map_state'], c['total_pages_label'], c['page_number_input']]
        )


        c['scene_gallery'].select(
            self.on_select_for_edit,
            inputs=[c['scenes_state'], c['scene_gallery_view_toggle'], c['scene_gallery_index_map_state'], c['extracted_frames_dir_state'], c['yolo_results_state']],
            outputs=[
                c['scenes_state'], c['scene_filter_status'], c['scene_gallery'], c['scene_gallery_index_map_state'],
                c['selected_scene_id_state'],
                c['sceneeditorstatusmd'], c['sceneeditorpromptinput'], c['sceneeditorboxthreshinput'], c['sceneeditortextthreshinput'],
                c['sceneeditoraccordion'],
                c['gallery_image_state'],
                c['gallery_shape_state'],
                c['scene_editor_yolo_subject_id'],
                c['propagate_masks_button'],
                c['yolo_results_state'],
            ]
        )

        # Wire recompute to use current editor controls and state
        c['scenerecomputebutton'].click(
            fn=lambda scenes, shot_id, outdir, view, txt, bth, tth, subject_id, *ana_args:
                _wire_recompute_handler(
                    self.config, self.app_logger, self.thumbnail_manager, scenes, shot_id, outdir, txt, bth, tth, view,
                    self.ana_ui_map_keys, ana_args, self.cuda_available
                ) if (txt and txt.strip()) else self.on_select_yolo_subject_wrapper(
                    subject_id, scenes, shot_id, outdir, view, *ana_args
                ),
            inputs=[
                c['scenes_state'],
                c['selected_scene_id_state'],
                c['analysis_output_dir_state'],
                c['scene_gallery_view_toggle'],
                c['sceneeditorpromptinput'], c['sceneeditorboxthreshinput'], c['sceneeditortextthreshinput'],
                c['scene_editor_yolo_subject_id'],
                *self.ana_input_components
            ],
            outputs=[
                c['scenes_state'],
                c['scene_gallery'],
                c['scene_gallery_index_map_state'],
                c['sceneeditorstatusmd'],
                c['total_pages_label'],
                c['page_number_input']
            ],
        )

        c['sceneresetbutton'].click(
            self.on_reset_scene_wrapper,
            inputs=[
                c['scenes_state'],
                c['selected_scene_id_state'],
                c['analysis_output_dir_state'],
                c['scene_gallery_view_toggle']
            ] + self.ana_input_components,
            outputs=[
                c['scenes_state'],
                c['scene_gallery'],
                c['scene_gallery_index_map_state'],
                c['sceneeditorstatusmd']
            ]
        )

        c['sceneincludebutton'].click(
            lambda s, sid, out, v: self.on_editor_toggle(s, sid, out, v, "included"),
            inputs=[c['scenes_state'], c['selected_scene_id_state'], c['extracted_frames_dir_state'], c['scene_gallery_view_toggle']],
            outputs=[c['scenes_state'], c['scene_filter_status'], c['scene_gallery'], c['scene_gallery_index_map_state'], c['propagate_masks_button']],
        )
        c['sceneexcludebutton'].click(
            lambda s, sid, out, v: self.on_editor_toggle(s, sid, out, v, "excluded"),
            inputs=[c['scenes_state'], c['selected_scene_id_state'], c['extracted_frames_dir_state'], c['scene_gallery_view_toggle']],
            outputs=[c['scenes_state'], c['scene_filter_status'], c['scene_gallery'], c['scene_gallery_index_map_state'], c['propagate_masks_button']],
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

        bulk_action_outputs = [c['scenes_state'], c['scene_filter_status'], c['scene_gallery'], c['scene_gallery_index_map_state'], c['propagate_masks_button']]

        bulk_filter_inputs = [c['scenes_state'], c['scene_mask_area_min_input'], c['scene_face_sim_min_input'],
                              c['scene_confidence_min_input'], c['enable_face_filter_input'], c['extracted_frames_dir_state'], c['scene_gallery_view_toggle']]

        for comp in [c['scene_mask_area_min_input'], c['scene_face_sim_min_input'], c['scene_confidence_min_input']]:
            comp.release(self.on_apply_bulk_scene_filters_extended, bulk_filter_inputs, bulk_action_outputs)

    def on_reset_scene_wrapper(self, scenes, shot_id, outdir, view, *ana_args):
        try:
            scene_idx = next((i for i, s in enumerate(scenes) if s['shot_id'] == shot_id), None)
            if scene_idx is None:
                return scenes, gr.update(), gr.update(), "Scene not found."
            scene.update({'seed_config': {}, 'seed_result': {}, 'seed_metrics': {}, 'manual_status_change': False, 'status': 'included', 'is_overridden': False, 'selected_bbox': scene.get('initial_bbox')})
            masker = _create_analysis_context(self.config, self.logger, self.thumbnail_manager,
                                              self.ana_ui_map_keys, ana_args)
            scene_idx = scenes.index(scene)
            _recompute_single_preview(scenes[scene_idx], masker, {}, self.thumbnail_manager, self.logger)
            save_scene_seeds(scenes, outdir, self.logger)
            gallery_items, index_map = build_scene_gallery_items(scenes, view, outdir)
            return scenes, gr.update(value=gallery_items), gr.update(value=index_map), f"Scene {shot_id} has been reset to its original state."
        except Exception as e:
            self.logger.error(f"Failed to reset scene {shot_id}", exc_info=True)
            gallery_items, index_map = build_scene_gallery_items(scenes, view, outdir)
            return scenes, gr.update(value=gallery_items), gr.update(value=index_map), f"Error resetting scene: {e}"

    def _empty_selection_response(self, scenes, indexmap):
        status_text, button_update = get_scene_status_text(scenes)
        return (scenes, status_text, gr.update(), indexmap,
                None, "Select a scene from the gallery to edit its properties.", "",
                self.config.gdino_box_threshold, self.config.gdino_text_threshold,
                gr.update(open=False), None, None, gr.update(visible=False, choices=[], value=None),
                button_update, {})

    def on_select_for_edit(self, scenes, view, indexmap, outputdir, yoloresultsstate, event: Optional[gr.EventData] = None, request: Optional[gr.Request] = None):
        sel_idx = getattr(event, "index", None) if event else None
        status_text, button_update = get_scene_status_text(scenes)

        if sel_idx is None:
            return self._empty_selection_response(scenes, indexmap)

        status_text, button_update = get_scene_status_text(scenes)
        if not scenes or not indexmap or not (0 <= sel_idx < len(indexmap)) or not (0 <= (scene_idx_in_state := indexmap[sel_idx]) < len(scenes)):
            self.logger.error(f"Invalid gallery or scene index on selection: gallery_idx={sel_idx}, scene_idx={scene_idx_in_state}")
            return self._empty_selection_response(scenes, indexmap)

        scene = scenes[scene_idx_in_state]
        cfg = scene.get("seed_config") or {}
        shotid = scene.get("shot_id")
        thumb_path_str = scene_thumb(scene, outputdir)
        gallery_image = self.thumbnail_manager.get(Path(thumb_path_str)) if thumb_path_str else None
        gallery_shape = gallery_image.shape[:2] if gallery_image is not None else None
        status_md = f"**Editing Scene {shotid}** (Frames {scene.get('start_frame', '?')}-{scene.get('end_frame', '?')})"
        prompt = cfg.get("text_prompt", "")
        boxth = cfg.get("box_threshold", self.config.gdino_box_threshold)
        textth = cfg.get("text_threshold", self.config.gdino_text_threshold)
        subject_choices = [f"{i+1}" for i in range(len(scene.get('yolo_detections', [])))]
        subject_id_update = gr.update(choices=subject_choices, value=None, visible=bool(subject_choices))

        return (scenes, status_text, gr.update(), indexmap, shotid,
                gr.update(value=status_md), gr.update(value=prompt), gr.update(value=boxth), gr.update(value=textth),
                gr.update(open=True), gallery_image, gallery_shape, subject_id_update, button_update, yoloresultsstate)

    def on_editor_toggle(self, scenes, selected_shotid, outputfolder, view, new_status):
        """Toggles the status of a scene from the scene editor.

        Args:
            scenes: The list of all scene dictionaries.
            selected_shotid: The ID of the currently selected scene.
            outputfolder: The path to the output directory.
            view: The current scene gallery view.
            new_status: The new status to set for the scene.

        Returns:
            A tuple of Gradio updates for the UI.
        """
        scenes, status_text, _, button_update = toggle_scene_status(scenes, selected_shotid, new_status, outputfolder, self.logger)
        items, index_map = build_scene_gallery_items(scenes, view, outputfolder)
        return scenes, status_text, gr.update(value=items), gr.update(value=index_map), button_update

    def _toggle_pause(self, tracker: 'AdvancedProgressTracker') -> str:
        """Toggles the pause state of a running task.

        Args:
            tracker (AdvancedProgressTracker): The progress tracker for the
                current task.

        Returns:
            str: The new label for the pause/resume button.
        """
        if tracker.pause_event.is_set():
            tracker.pause_event.clear()
            return "⏸️ Paused"
        else:
            tracker.pause_event.set()
            return "▶️ Resume"

    def _create_event_handlers(self):
        self.logger.info("Initializing Gradio event handlers...")
        super()._create_event_handlers()
        c = self.components
        c['cancel_button'].click(lambda: self.cancel_event.set(), [], [])
        c['pause_button'].click(
            self._toggle_pause,
            inputs=[gr.State(lambda: next((arg for arg in self.last_run_args if isinstance(arg, AdvancedProgressTracker)), None) if self.last_run_args else None)],
            outputs=c['pause_button']
        )
        c['clear_logs_button'].click(lambda: (self.all_logs.clear(), "")[1], [], c['unified_log'])
        c['log_level_filter'].change(lambda level: (setattr(self, 'log_filter_level', level), "\n".join([l for l in self.all_logs if self.log_filter_level.upper() == "DEBUG" or f"[{level.upper()}]" in l][-1000:]))[1], c['log_level_filter'], c['unified_log'])
        c['scene_editor_yolo_subject_id'].change(
            self.on_select_yolo_subject_wrapper,
            inputs=[c['scene_editor_yolo_subject_id'], c['scenes_state'], c['selected_scene_id_state'], c['extracted_frames_dir_state'], c['scene_gallery_view_toggle']] + self.ana_input_components,
            outputs=[c['scenes_state'], c['scene_gallery'], c['scene_gallery_index_map_state'], c['sceneeditorstatusmd']]
        )
        c['run_diagnostics_button'].click(self.run_system_diagnostics, inputs=[], outputs=[c['unified_log']])

    def run_system_diagnostics(self) -> Generator[str, None, None]:
        """Runs a comprehensive suite of system checks and a dry run.

        This function checks for correct Python and library versions, CUDA
        availability, the presence of necessary assets, and simulates the
        entire end-to-end pipeline using sample data to ensure all components
        are working correctly. The results are yielded as a formatted report.

        Yields:
            str: The formatted diagnostics report.
        """
        self.logger.info("Starting system diagnostics...")
        report = ["\n\n--- System Diagnostics Report ---", "\n[SECTION 1: System & Environment]"]
        try: report.append(f"  - Python Version: OK ({sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro})")
        except Exception as e: report.append(f"  - Python Version: FAILED ({e})")
        try:
            report.append(f"  - PyTorch Version: OK ({torch.__version__})")
            if torch.cuda.is_available(): report.append(f"  - CUDA: OK (Version: {torch.version.cuda}, GPU: {torch.cuda.get_device_name(0)})")
            else: report.append("  - CUDA: NOT AVAILABLE (Running in CPU mode)")
        except Exception as e: report.append(f"  - PyTorch/CUDA Check: FAILED ({e})")
        report.append("\n[SECTION 2: Core Dependencies]")
        for dep in ["cv2", "gradio", "imagehash", "mediapipe", "ultralytics", "groundingdino", "DAM4SAM"]:
            try: __import__(dep.split('.')[0]); report.append(f"  - {dep}: OK")
            except ImportError: report.append(f"  - {dep}: FAILED (Not Installed)")
        report.append("\n[SECTION 3: Paths & Assets]")
        for name, path in {"Models Directory": Path(self.config.models_dir), "Dry Run Assets": Path("dry-run-assets"),
                           "Sample Video": Path("dry-run-assets/sample.mp4"), "Sample Image": Path("dry-run-assets/sample.jpg")}.items():
            report.append(f"  - {name}: {'OK' if path.exists() else 'FAILED'} (Path: {path})")
        report.append("\n[SECTION 4: Model Loading Simulation]")
        try:
            get_person_detector(model_path_str=str(Path(self.config.models_dir) / self.config.default_person_detector_model),
                                device="cuda" if self.cuda_available else "cpu", imgsz=self.config.person_detector_imgsz,
                                conf=self.config.person_detector_conf, logger=self.logger)
            report.append("  - YOLO Model: OK")
        except Exception as e: report.append(f"  - YOLO Model: FAILED ({e})")
        report.append("\n[SECTION 5: E2E Pipeline Simulation]")
        temp_output_dir = Path(self.config.downloads_dir) / "dry_run_output"
        shutil.rmtree(temp_output_dir, ignore_errors=True); temp_output_dir.mkdir(parents=True, exist_ok=True)
        try:
            report.append("  - Stage 1: Frame Extraction...")
            ext_event = ExtractionEvent(source_path="dry-run-assets/sample.mp4", method='interval', interval='1.0', max_resolution="720", thumbnails_only=True, thumb_megapixels=0.2, scene_detect=True)
            ext_result = deque(execute_extraction(ext_event, self.progress_queue, self.cancel_event, self.logger, self.config), maxlen=1)[0]
            if not ext_result.get("done"): raise RuntimeError("Extraction failed")
            report[-1] += " OK"
            report.append("  - Stage 2: Pre-analysis...")
            pre_ana_event = PreAnalysisEvent(output_folder=ext_result['extracted_frames_dir_state'], video_path=ext_result['extracted_video_path_state'], scene_detect=True,
                                             pre_analysis_enabled=True, pre_sample_nth=1, primary_seed_strategy="🧑‍🤝‍🧑 Find Prominent Person")
            pre_ana_result = deque(execute_pre_analysis(pre_ana_event, self.progress_queue, self.cancel_event, self.logger, self.config, self.thumbnail_manager, self.cuda_available), maxlen=1)[0]
            if not pre_ana_result.get("done"): raise RuntimeError(f"Pre-analysis failed: {pre_ana_result}")
            report[-1] += " OK"
            scenes = pre_ana_result['scenes']
            report.append("  - Stage 3: Mask Propagation...")
            prop_event = PropagationEvent(output_folder=pre_ana_result['output_dir'], video_path=ext_result['extracted_video_path_state'], scenes=scenes, analysis_params=pre_ana_event)
            prop_result = deque(execute_propagation(prop_event, self.progress_queue, self.cancel_event, self.logger, self.config, self.thumbnail_manager, self.cuda_available), maxlen=1)[0]
            if not prop_result.get("done"): raise RuntimeError("Propagation failed")
            report[-1] += " OK"
            report.append("  - Stage 4: Frame Analysis...")
            ana_result = deque(execute_analysis(prop_event, self.progress_queue, self.cancel_event, self.logger, self.config, self.thumbnail_manager, self.cuda_available), maxlen=1)[0]
            if not ana_result.get("done"): raise RuntimeError("Analysis failed")
            report[-1] += " OK"
            output_dir = ana_result['output_dir']
            all_frames, _ = load_and_prep_filter_data(output_dir, self.get_all_filter_keys(), self.config)
            report.append("  - Stage 5: Filtering...")
            kept, _, _, _ = apply_all_filters_vectorized(all_frames, {'require_face_match': False, 'dedup_thresh': -1}, self.config, output_dir=ana_result['output_dir'])
            report[-1] += f" OK (kept {len(kept)} frames)"
            report.append("  - Stage 6: Export...")
            export_event = ExportEvent(all_frames_data=all_frames, output_dir=ana_result['output_dir'], video_path=ext_result['extracted_video_path_state'], enable_crop=False, filter_args={'require_face_match': False, 'dedup_thresh': -1})
            export_msg = self.export_kept_frames(export_event)
            if "Error" in export_msg: raise RuntimeError(f"Export failed: {export_msg}")
            report[-1] += " OK"
        except Exception as e:
            error_message = f"FAILED ({e})"
            if "..." in report[-1]: report[-1] += error_message
            else: report.append(f"  - Pipeline Simulation: {error_message}")
            self.logger.error("Dry run pipeline failed", exc_info=True)
        final_report = "\n".join(report)
        self.logger.info(final_report)
        yield final_report

    def _create_pre_analysis_event(self, *args: Any) -> 'PreAnalysisEvent':
        """Creates a `PreAnalysisEvent` from the raw Gradio UI component values.

        This helper function gathers all the arguments from the analysis UI,
        maps them to the correct keys, applies some strategy-specific logic
        (e.g., forcing `enable_face_filter` for the "By Face" strategy), and
        returns a validated `PreAnalysisEvent` object.

        Args:
            *args (Any): The raw values from the Gradio UI components.

        Returns:
            PreAnalysisEvent: A validated `PreAnalysisEvent` object.
        """
        ui_args = dict(zip(self.ana_ui_map_keys, args))
        clean_args = {k: v for k, v in ui_args.items() if v is not None}
        strategy = clean_args.get('primary_seed_strategy', self.config.default_primary_seed_strategy)
        if strategy == "👤 By Face":
            clean_args.update({'enable_face_filter': True, 'text_prompt': ""})
        elif strategy == "📝 By Text":
            clean_args.update({'enable_face_filter': False, 'face_ref_img_path': ""})
        return PreAnalysisEvent.model_validate(clean_args)

    def _run_pipeline(self, pipeline_func: Callable, event: Any, progress: Callable, success_callback: Optional[Callable] = None, *args):
        """A generic wrapper for executing a pipeline function.

        This method handles the boilerplate of running a pipeline function,
        checking for cancellation, and invoking a success callback with the
        final result.

        Args:
            pipeline_func (Callable): The pipeline function to execute.
            event (Any): The event data for the pipeline.
            progress (Callable): The Gradio `gr.Progress` object.
            success_callback (Optional[Callable]): An optional function to call
                with the final result if the pipeline succeeds. Defaults to None.
            *args: Additional arguments for the pipeline function.

        Yields:
            dict: UI updates.
        """
        try:
            for result in pipeline_func(event, self.progress_queue, self.cancel_event, self.app_logger, self.config, self.thumbnail_manager, self.cuda_available, progress=progress):
                if isinstance(result, dict):
                    if self.cancel_event.is_set():
                        yield {"unified_log": f"{pipeline_func.__name__} cancelled."}; return
                    if result.get("done"):
                        if success_callback: yield success_callback(result)
                        return
            yield {"unified_log": f"❌ {pipeline_func.__name__} did not complete successfully."}
        except Exception as e:
            self.app_logger.error(f"{pipeline_func.__name__} execution failed", exc_info=True)
            yield {"unified_log": f"[ERROR] An unexpected error occurred in {pipeline_func.__name__}: {e}"}

    def run_extraction_wrapper(self, *args, progress=gr.Progress(track_tqdm=True)):
        """Wrapper for the extraction pipeline.

        This method gathers UI arguments, creates an `ExtractionEvent`, and
        invokes the `_run_pipeline` helper to execute the extraction.

        Args:
            *args: Raw values from the extraction UI components.
            progress: The Gradio progress object.
        """
        ui_args = dict(zip(self.ext_ui_map_keys, args))
        ui_args['thumbnails_only'] = True
        clean_args = {k: v for k, v in ui_args.items() if v is not None}
        event = ExtractionEvent.model_validate(clean_args)
        yield from self._run_pipeline(execute_extraction, event, progress, self._on_extraction_success)

    def _on_extraction_success(self, result: dict) -> dict:
        """Callback for a successful extraction pipeline run.

        Args:
            result (dict): The result dictionary from the pipeline.

        Returns:
            dict: A dictionary of Gradio UI updates.
        """
        return {
            "unified_log": result.get("unified_log", "✅ Extraction completed."),
            "extracted_video_path_state": result.get("video_path", "") or result.get("extracted_video_path_state", ""),
            "extracted_frames_dir_state": result.get("output_dir", "") or result.get("extracted_frames_dir_state", "")
        }

    def _on_pre_analysis_success(self, result: dict) -> dict:
        """Callback for a successful pre-analysis pipeline run.

        Args:
            result (dict): The result dictionary from the pipeline.

        Returns:
            dict: A dictionary of Gradio UI updates.
        """
        scenes = result.get('scenes', [])
        if scenes: save_scene_seeds(scenes, result['output_dir'], self.app_logger)
        status_text, button_update = get_scene_status_text(scenes)
        log_message = result.get("unified_log", "✅ Pre-analysis completed.") + "\nContinue to the next tab to review scenes."
        updates = {
            "unified_log": log_message, "scenes_state": scenes, "propagate_masks_button": button_update,
            "scene_filter_status": status_text,
            "scene_face_sim_min_input": gr.update(visible=any((s.get("seed_metrics") or {}).get("best_face_sim") is not None for s in (scenes or []))),
            "seeding_results_column": gr.update(visible=True), "main_tabs": gr.update(selected=2)
        }
        gallery_items, index_map = build_scene_gallery_items(scenes, "Kept", result.get('output_dir', ''))
        updates.update({"scene_gallery": gr.update(value=gallery_items), "scene_gallery_index_map_state": index_map})
        if result.get("final_face_ref_path"): updates["face_ref_img_path_input"] = result["final_face_ref_path"]
        return updates

    def run_pre_analysis_wrapper(self, *args, progress=gr.Progress(track_tqdm=True)):
        """Wrapper for the pre-analysis pipeline.

        Args:
            *args: Raw values from the analysis UI components.
            progress: The Gradio progress object.
        """
        event = self._create_pre_analysis_event(*args)
        yield from self._run_pipeline(execute_pre_analysis, event, progress, self._on_pre_analysis_success)

    def run_propagation_wrapper(self, scenes: list, *args, progress=gr.Progress(track_tqdm=True)):
        """Wrapper for the mask propagation pipeline.

        Args:
            scenes (list): The list of scene objects from the UI state.
            *args: Raw values from the analysis UI components.
            progress: The Gradio progress object.
        """
        event = PropagationEvent(output_folder=self._create_pre_analysis_event(*args).output_folder, video_path=self._create_pre_analysis_event(*args).video_path,
                                 scenes=scenes, analysis_params=self._create_pre_analysis_event(*args))
        yield from self._run_pipeline(execute_propagation, event, progress, self._on_propagation_success)

    def _on_propagation_success(self, result: dict) -> dict:
        """Callback for a successful mask propagation run.

        Args:
            result (dict): The result dictionary from the pipeline.

        Returns:
            dict: A dictionary of Gradio UI updates.
        """
        return {
            "unified_log": result.get("unified_log", "✅ Mask propagation completed."),
            "analysis_output_dir_state": result.get('output_dir', "")
        }

    def run_analysis_wrapper(self, scenes: list, *args, progress=gr.Progress(track_tqdm=True)):
        """Wrapper for the frame analysis pipeline.

        Args:
            scenes (list): The list of scene objects from the UI state.
            *args: Raw values from the analysis UI components.
            progress: The Gradio progress object.
        """
        event = PropagationEvent(output_folder=self._create_pre_analysis_event(*args).output_folder, video_path=self._create_pre_analysis_event(*args).video_path,
                                 scenes=scenes, analysis_params=self._create_pre_analysis_event(*args))
        yield from self._run_pipeline(execute_analysis, event, progress, self._on_analysis_success)

    def _on_analysis_success(self, result: dict) -> dict:
        """Callback for a successful frame analysis run.

        Args:
            result (dict): The result dictionary from the pipeline.

        Returns:
            dict: A dictionary of Gradio UI updates.
        """
        return {
            "unified_log": result.get("unified_log", "✅ Analysis complete. You can now filter and export."),
            "analysis_output_dir_state": result.get('output_dir', ""),
            "analysis_metadata_path_state": result.get('metadata_path', ""),
            "filtering_tab": gr.update(interactive=True)
        }

    def run_session_load_wrapper(self, session_path: str) -> dict:
        """Wrapper for the session loading pipeline.

        Args:
            session_path (str): The path to the session directory to load.

        Returns:
            dict: A dictionary of Gradio UI updates.
        """
        try:
            final_result = {}
            for result in execute_session_load(self, SessionLoadEvent(session_path=session_path), self.app_logger, self.config, self.thumbnail_manager):
                if isinstance(result, dict):
                    if 'unified_log' in result: final_result.update(result)
                    else: final_result.update(result)
            return final_result
        except Exception as e: raise

    def _fix_strategy_visibility(self, strategy: str) -> dict:
        """Handler for controlling UI visibility based on the seeding strategy.

        This method is connected to the `primary_seed_strategy_input` radio
        button. It shows and hides the relevant groups of UI components
        (e.g., the face selection group, the text prompt group) based on the
        selected strategy.

        Args:
            strategy (str): The selected seeding strategy from the UI.

        Returns:
            dict: A dictionary of Gradio updates to control component visibility.
        """
        is_face = "By Face" in strategy or "Fallback" in strategy
        is_text = "By Text" in strategy or "Fallback" in strategy
        is_auto = "Prominent Person" in strategy
        return {
            self.components['face_seeding_group']: gr.update(visible=is_face),
            self.components['text_seeding_group']: gr.update(visible=is_text),
            self.components['auto_seeding_group']: gr.update(visible=is_auto),
            self.components['enable_face_filter_input']: gr.update(value=is_face, visible=is_face),
        }

    def _setup_visibility_toggles(self):
        """Sets up Gradio event handlers for dynamically showing/hiding UI components."""
        c = self.components

        def handle_source_change(path):
            is_folder = is_image_folder(path)
            # When a folder is provided, or input is cleared, hide video-specific controls.
            if is_folder or not path:
                return {
                    c['max_resolution']: gr.update(visible=False),
                    c['thumbnail_group']: gr.update(visible=False),
                }
            # When a video path is provided (or any non-folder path), show the default view.
            else:
                return {
                    c['max_resolution']: gr.update(visible=True),
                    c['thumbnail_group']: gr.update(visible=True),
                }

        source_controls = [c['source_input'], c['upload_video_input']]
        video_specific_outputs = [
            c['max_resolution'],
            c['thumbnail_group'],
        ]
        for control in source_controls:
            control.change(handle_source_change, inputs=control, outputs=video_specific_outputs)


        c['method_input'].change(
            lambda m: {
                c['interval_input']: gr.update(visible=m == 'interval'),
                c['nth_frame_input']: gr.update(visible=m in ['every_nth_frame', 'nth_plus_keyframes'])
            },
            c['method_input'],
            [c['interval_input'], c['nth_frame_input']]
        )

        c['primary_seed_strategy_input'].change(
            self._fix_strategy_visibility,
            inputs=c['primary_seed_strategy_input'],
            outputs=[c['face_seeding_group'], c['text_seeding_group'], c['auto_seeding_group'], c['enable_face_filter_input']]
        )


    def _setup_pipeline_handlers(self):
        """Sets up the Gradio event handlers for the main processing pipelines."""
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

        def analysis_handler(scenes, *args, progress: gr.Progress):
            """Handler for the analysis pipeline."""
            # Mirror the other handlers so Gradio gets consistent multi-output updates
            yield from self._run_task_with_progress(self.run_analysis_wrapper, all_outputs, progress, scenes, *args)

        c['load_session_button'].click(
            fn=session_load_handler,
            inputs=[c['session_path_input']],
            outputs=all_outputs,
            show_progress="hidden"
        )
        ext_inputs = [c[{'source_path': 'source_input', 'upload_video': 'upload_video_input', 'max_resolution': 'max_resolution',
                         'scene_detect': 'ext_scene_detect_input', **{k: f"{k}_input" for k in self.ext_ui_map_keys if k not in
                         ['source_path', 'upload_video', 'max_resolution', 'scene_detect']}}[k]] for k in self.ext_ui_map_keys]
        self.ana_input_components = [c.get(k) for k in [{'output_folder': 'extracted_frames_dir_state', 'video_path': 'extracted_video_path_state',
                                                           'resume': 'resume_input', 'enable_face_filter': 'enable_face_filter_input',
                                                           'face_ref_img_path': 'face_ref_img_path_input', 'face_ref_img_upload': 'face_ref_img_upload_input',
                                                               'face_model_name': 'face_model_name_input', 'enable_subject_mask': 'enable_subject_mask_input',
                                                           'dam4sam_model_name': 'dam4sam_model_name_input', 'person_detector_model': 'person_detector_model_input',
                                                               'best_frame_strategy': 'best_frame_strategy_input', 'scene_detect': 'ext_scene_detect_input',
                                                           'enable_dedup': 'enable_dedup_input', 'text_prompt': 'text_prompt_input',
                                                               'box_threshold': 'box_threshold',
                                                               'text_threshold': 'text_threshold',
                                                           'min_mask_area_pct': 'min_mask_area_pct_input',
                                                           'sharpness_base_scale': 'sharpness_base_scale_input',
                                                           'edge_strength_base_scale': 'edge_strength_base_scale_input',
                                                               'gdino_config_path': 'gdino_config_path_input',
                                                               'gdino_checkpoint_path': 'gdino_checkpoint_path_input',
                                                           'pre_analysis_enabled': 'pre_analysis_enabled_input', 'pre_sample_nth': 'pre_sample_nth_input',
                                                           'primary_seed_strategy': 'primary_seed_strategy_input',
                                                           **{f'compute_{m}': f'compute_{m}' for m in [
                                                               'quality_score', 'sharpness', 'edge_strength', 'contrast', 'brightness', 'entropy',
                                                               'eyes_open', 'yaw', 'pitch', 'face_sim', 'subject_mask_area', 'niqe', 'phash'
                                                           ]}
                                                          }[k] for k in self.ana_ui_map_keys]]
        prop_inputs = [c['scenes_state']] + self.ana_input_components
        c['start_extraction_button'].click(fn=extraction_handler,
                                         inputs=ext_inputs, outputs=all_outputs, show_progress="hidden").then(lambda d: gr.update(selected=1) if d else gr.update(), c['extracted_frames_dir_state'], c['main_tabs'])
        c['start_pre_analysis_button'].click(fn=pre_analysis_handler,
                                           inputs=self.ana_input_components, outputs=all_outputs, show_progress="hidden")
        c['propagate_masks_button'].click(fn=propagation_handler,
                                        inputs=prop_inputs, outputs=all_outputs, show_progress="hidden").then(lambda p: gr.update(selected=3) if p else gr.update(), c['analysis_output_dir_state'], c['main_tabs'])

        analysis_inputs = [c['scenes_state']] + self.ana_input_components
        c['start_analysis_button'].click(fn=analysis_handler,
                                         inputs=analysis_inputs, outputs=all_outputs, show_progress="hidden").then(lambda p: gr.update(selected=4) if p else gr.update(), c['analysis_metadata_path_state'], c['main_tabs'])

        c['find_people_button'].click(
            self.on_find_people_from_video,
            inputs=self.ana_input_components,
            outputs=[c['discovered_people_group'], c['discovered_faces_gallery'], c['identity_confidence_slider'], c['discovered_faces_state']]
        )

        c['identity_confidence_slider'].release(
            self.on_identity_confidence_change,
            inputs=[c['identity_confidence_slider'], c['discovered_faces_state']],
            outputs=[c['discovered_faces_gallery']]
        )

        c['discovered_faces_gallery'].select(
            self.on_discovered_face_select,
            inputs=[c['discovered_faces_state'], c['identity_confidence_slider']] + self.ana_input_components,
            outputs=[c['face_ref_img_path_input'], c['face_ref_image']]
        )

    def on_identity_confidence_change(self, confidence: float, all_faces: list) -> gr.update:
        """Handler for when the identity confidence slider is changed.

        This function re-runs the DBSCAN clustering algorithm on the discovered
        face embeddings with a new `eps` value derived from the confidence
        slider. It then updates the gallery of discovered people to reflect the
        new clustering.

        Args:
            confidence (float): The new confidence value from the slider.
            all_faces (list): The list of all discovered face data.

        Returns:
            gr.update: A Gradio update for the discovered faces gallery.
        """
        if not all_faces:
            return []

        from sklearn.cluster import DBSCAN
        embeddings = np.array([face['embedding'] for face in all_faces])
        # The confidence slider is inverted to map to epsilon
        eps = 1.0 - confidence
        clustering = DBSCAN(eps=eps, min_samples=2, metric="cosine").fit(embeddings)
        labels = clustering.labels_

        unique_labels = sorted(list(set(labels)))
        gallery_items = []
        self.gallery_to_cluster_map = {}
        gallery_idx = 0

        for label in unique_labels:
            if label == -1:
                continue # Skip noise

            self.gallery_to_cluster_map[gallery_idx] = label
            gallery_idx += 1

            cluster_faces = [all_faces[i] for i, l in enumerate(labels) if l == label]
            best_face = max(cluster_faces, key=lambda x: x['det_score'])

            thumb_rgb = self.thumbnail_manager.get(Path(best_face['thumb_path']))
            x1, y1, x2, y2 = best_face['bbox'].astype(int)
            face_crop = thumb_rgb[y1:y2, x1:x2]

            gallery_items.append((face_crop, f"Person {label}"))

        return gr.update(value=gallery_items)

    def on_discovered_face_select(self, all_faces: list, confidence: float, *args, evt: gr.EventData = None) -> tuple[str, Optional[np.ndarray]]:
        """Handler for when a face is selected from the discovered faces gallery.

        This function identifies the selected person's cluster, finds the best
        representative frame for that person, extracts the full-resolution
        frame from the video, crops the face, saves it to a file, and updates
        the UI to use this new image as the face reference.

        Args:
            all_faces (list): The list of all discovered face data.
            confidence (float): The current identity confidence value.
            *args: The rest of the analysis UI component values.
            evt (gr.EventData): The Gradio event data, containing the index of
                the selected gallery item.

        Returns:
            tuple[str, Optional[np.ndarray]]: A tuple containing the path to
            the saved face crop and the cropped image data as a NumPy array for
            display.
        """
        if not all_faces or evt is None or evt.index is None:
            return "", None

        selected_person_label = self.gallery_to_cluster_map.get(evt.index)
        if selected_person_label is None:
            self.logger.error(f"Could not find cluster label for gallery index {evt.index}")
            return "", None

        params = self._create_pre_analysis_event(*args)
        video_path = params.video_path

        from sklearn.cluster import DBSCAN
        embeddings = np.array([face['embedding'] for face in all_faces])
        eps = 1.0 - confidence
        clustering = DBSCAN(eps=eps, min_samples=2, metric="cosine").fit(embeddings)
        labels = clustering.labels_


        cluster_faces = [all_faces[i] for i, l in enumerate(labels) if l == selected_person_label]
        if not cluster_faces:
            return "", None

        best_face = max(cluster_faces, key=lambda x: x['det_score'])
        best_frame_num = best_face['frame_num']

        # Extract high-res frame
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, best_frame_num)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            return "", None

        # Get high-res face crop
        x1, y1, x2, y2 = best_face['bbox'].astype(int)
        # Scale bbox to full-res frame
        thumb_rgb = self.thumbnail_manager.get(Path(best_face['thumb_path']))
        h, w, _ = thumb_rgb.shape
        fh, fw, _ = frame.shape
        x1, y1, x2, y2 = int(x1 * fw/w), int(y1 * fh/h), int(x2 * fw/w), int(y2 * fh/h)

        face_crop = frame[y1:y2, x1:x2]

        # Save the crop to a file
        face_crop_path = Path(params.output_folder) / "reference_face.png"
        cv2.imwrite(str(face_crop_path), face_crop)

        return str(face_crop_path), face_crop

    def on_find_people_from_video(self, *args) -> tuple[gr.update, list, float, list]:
        """Handler for the 'Find People From Video' button.

        This function orchestrates a multi-step process:
        1. It samples frames from the video.
        2. For each sampled frame, it runs YOLO to detect people.
        3. For frames with people, it runs InsightFace to detect faces and get
           embeddings.
        4. It clusters all discovered face embeddings using DBSCAN to group
           faces belonging to the same person.
        5. It generates a gallery of the best face crop for each person cluster.

        Args:
            *args: The raw values from the analysis UI components.

        Returns:
            tuple[gr.update, list, float, list]: A tuple of Gradio updates for
            the discovered people group, the gallery, the confidence slider,
            and the state holding all discovered faces.
        """
        try:
            params = self._create_pre_analysis_event(*args)
            output_dir = Path(params.output_folder)
            if not output_dir.exists():
                return gr.update(visible=False), [], 0.5, []

            models = initialize_analysis_models(params, self.config, self.logger)
            person_detector = models['person_detector']
            face_analyzer = models['face_analyzer']

            if not person_detector or not face_analyzer:
                self.logger.error("Person detector or face analyzer not available.")
                return gr.update(visible=False), [], 0.5, []

            frame_map = create_frame_map(output_dir, self.logger)
            if not frame_map:
                self.logger.error("Frame map not found.")
                return gr.update(visible=False), [], 0.5, []

            all_faces = []
            thumb_dir = output_dir / "thumbs"

            for frame_num, thumb_filename in frame_map.items():
                if frame_num % params.pre_sample_nth != 0:
                    continue

                thumb_path = thumb_dir / thumb_filename
                thumb_rgb = self.thumbnail_manager.get(thumb_path)
                if thumb_rgb is None:
                    continue

                people = person_detector.detect_boxes(thumb_rgb)
                if not people:
                    continue

                thumb_bgr = cv2.cvtColor(thumb_rgb, cv2.COLOR_RGB2BGR)
                faces = face_analyzer.get(thumb_bgr)

                for face in faces:
                    all_faces.append({
                        'frame_num': frame_num,
                        'bbox': face.bbox,
                        'embedding': face.normed_embedding,
                        'det_score': face.det_score,
                        'thumb_path': str(thumb_path)
                    })

            if not all_faces:
                self.logger.warning("No faces found in the video.")
                return gr.update(visible=True), [], 0.5, []

            # Simple clustering for now
            from sklearn.cluster import DBSCAN
            embeddings = np.array([face['embedding'] for face in all_faces])
            clustering = DBSCAN(eps=0.5, min_samples=2, metric="cosine").fit(embeddings)
            labels = clustering.labels_

            unique_labels = sorted(list(set(labels)))
            gallery_items = []
            self.gallery_to_cluster_map = {}
            gallery_idx = 0

            for label in unique_labels:
                if label == -1:
                    continue # Skip noise

                self.gallery_to_cluster_map[gallery_idx] = label
                gallery_idx += 1

                cluster_faces = [all_faces[i] for i, l in enumerate(labels) if l == label]
                best_face = max(cluster_faces, key=lambda x: x['det_score'])

                thumb_rgb = self.thumbnail_manager.get(Path(best_face['thumb_path']))
                x1, y1, x2, y2 = best_face['bbox'].astype(int)
                face_crop = thumb_rgb[y1:y2, x1:x2]

                gallery_items.append((face_crop, f"Person {label}"))

            return gr.update(visible=True), gallery_items, 0.5, all_faces

        except Exception as e:
            self.logger.error(f"Error in on_find_people_from_video: {e}", exc_info=True)
            return gr.update(visible=False), [], 0.5, []

    def on_apply_bulk_scene_filters_extended(self, scenes: list, min_mask_area: float, min_face_sim: float, min_confidence: float, enable_face_filter: bool, output_folder: str, view: str) -> tuple:
        """Handler for applying bulk filters to the scene gallery.

        This function iterates through all scenes and automatically sets their
        status to 'excluded' if they do not meet the specified minimum
        thresholds for mask area, face similarity, or confidence. Scenes that
        were manually included or excluded by the user are skipped.

        Args:
            scenes (list): The list of all scene objects.
            min_mask_area (float): The minimum seed mask area percentage.
            min_face_sim (float): The minimum seed face similarity score.
            min_confidence (float): The minimum seed confidence score.
            enable_face_filter (bool): Whether the face similarity filter is active.
            output_folder (str): The path to the output directory.
            view (str): The current scene gallery view.

        Returns:
            tuple: A tuple of Gradio updates for the UI.
        """
        if not scenes:
            status_text, button_update = get_scene_status_text(scenes)
            return [], status_text, gr.update(), [], button_update, "/ 1 pages", 1

        self.logger.info("Applying bulk scene filters", extra={"min_mask_area": min_mask_area, "min_face_sim": min_face_sim, "min_confidence": min_confidence, "enable_face_filter": enable_face_filter})

        for scene in scenes:
            if scene.manual_status_change:
                continue

            rejection_reasons = []
            seed_result = scene.seed_result or {}
            details = seed_result.get('details', {})
            seed_metrics = scene.seed_metrics or {}

            if details.get('mask_area_pct', 101.0) < min_mask_area:
                rejection_reasons.append("Min Seed Mask Area")
            if enable_face_filter and seed_metrics.get('best_face_sim', 1.01) < min_face_sim:
                rejection_reasons.append("Min Seed Face Sim")
            if seed_metrics.get('score', 101.0) < min_confidence:
                rejection_reasons.append("Min Seed Confidence")

            scene.rejection_reasons = rejection_reasons
            if rejection_reasons:
                scene.status = 'excluded'
            else:
                scene.status = 'included'

        save_scene_seeds(scenes, output_folder, self.logger)
        gallery_items, new_index_map, total_pages = build_scene_gallery_items(scenes, view, output_folder, page_num=1)
        status_text, button_update = get_scene_status_text(scenes)
        return scenes, status_text, gr.update(value=gallery_items), new_index_map, button_update, f"/ {total_pages} pages", 1

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
            inputs=[c['scenes_state'], c['scene_gallery_view_toggle'], c['scene_gallery_index_map_state'], c['extracted_frames_dir_state'], c['yolo_results_state']],
            outputs=[
                c['scenes_state'], c['scene_filter_status'], c['scene_gallery'], c['scene_gallery_index_map_state'],
                c['selected_scene_id_state'],
                c['sceneeditorstatusmd'], c['sceneeditorpromptinput'], c['sceneeditorboxthreshinput'], c['sceneeditortextthreshinput'],
                c['sceneeditoraccordion'],
                c['gallery_image_state'],
                c['gallery_shape_state'],
                c['scene_editor_yolo_subject_id'],
                c['propagate_masks_button'],
                c['yolo_results_state'],
            ]
        )

        # Wire recompute to use current editor controls and state
        c['scenerecomputebutton'].click(
            fn=lambda scenes, shot_id, outdir, view, txt, bth, tth, subject_id, *ana_args:
                _wire_recompute_handler(
                    self.config, self.app_logger, self.thumbnail_manager, scenes, shot_id, outdir, txt, bth, tth, view,
                    self.ana_ui_map_keys, ana_args, self.cuda_available
                ) if (txt and txt.strip()) else self.on_select_yolo_subject_wrapper(
                    subject_id, scenes, shot_id, outdir, view, *ana_args
                ),
            inputs=[
                c['scenes_state'],
                c['selected_scene_id_state'],
                c['analysis_output_dir_state'],
                c['scene_gallery_view_toggle'],
                c['sceneeditorpromptinput'], c['sceneeditorboxthreshinput'], c['sceneeditortextthreshinput'],
                c['scene_editor_yolo_subject_id'],
                *self.ana_input_components
            ],
            outputs=[
                c['scenes_state'],
                c['scene_gallery'],
                c['scene_gallery_index_map_state'],
                c['sceneeditorstatusmd'],
            ],
        )

        c['sceneresetbutton'].click(
            self.on_reset_scene_wrapper,
            inputs=[
                c['scenes_state'],
                c['selected_scene_id_state'],
                c['analysis_output_dir_state'],
                c['scene_gallery_view_toggle']
            ] + self.ana_input_components,
            outputs=[
                c['scenes_state'],
                c['scene_gallery'],
                c['scene_gallery_index_map_state'],
                c['sceneeditorstatusmd'],
                c['total_pages_label'],
                c['page_number_input']
            ]
        )

        c['sceneincludebutton'].click(
            lambda s, sid, out, v: self.on_editor_toggle(s, sid, out, v, "included"),
            inputs=[c['scenes_state'], c['selected_scene_id_state'], c['extracted_frames_dir_state'], c['scene_gallery_view_toggle']],
            outputs=[c['scenes_state'], c['scene_filter_status'], c['scene_gallery'], c['scene_gallery_index_map_state'], c['propagate_masks_button'], c['total_pages_label'], c['page_number_input']],
        )
        c['sceneexcludebutton'].click(
            lambda s, sid, out, v: self.on_editor_toggle(s, sid, out, v, "excluded"),
            inputs=[c['scenes_state'], c['selected_scene_id_state'], c['extracted_frames_dir_state'], c['scene_gallery_view_toggle']],
            outputs=[c['scenes_state'], c['scene_filter_status'], c['scene_gallery'], c['scene_gallery_index_map_state'], c['propagate_masks_button'], c['total_pages_label'], c['page_number_input']],
        )

        def init_scene_gallery(scenes, view, outdir):
            if not scenes:
                return gr.update(value=[]), [], "/ 1 pages", 1
            gallery_items, index_map, total_pages = build_scene_gallery_items(scenes, view, outdir)
            return gr.update(value=gallery_items), index_map, f"/ {total_pages} pages", 1

        c['scenes_state'].change(
            init_scene_gallery,
            [c['scenes_state'], c['scene_gallery_view_toggle'], c['extracted_frames_dir_state']],
            [c['scene_gallery'], c['scene_gallery_index_map_state'], c['total_pages_label'], c['page_number_input']]
        )

        bulk_action_outputs = [c['scenes_state'], c['scene_filter_status'], c['scene_gallery'], c['scene_gallery_index_map_state'], c['propagate_masks_button'], c['total_pages_label'], c['page_number_input']]

        bulk_filter_inputs = [c['scenes_state'], c['scene_mask_area_min_input'], c['scene_face_sim_min_input'],
                              c['scene_confidence_min_input'], c['enable_face_filter_input'], c['extracted_frames_dir_state'], c['scene_gallery_view_toggle']]

        for comp in [c['scene_mask_area_min_input'], c['scene_face_sim_min_input'], c['scene_confidence_min_input']]:
            comp.release(self.on_apply_bulk_scene_filters_extended, bulk_filter_inputs, bulk_action_outputs)

    def _setup_filtering_handlers(self):
        c = self.components
        slider_keys, slider_comps = sorted(c['metric_sliders'].keys()), [c['metric_sliders'][k] for k in sorted(c['metric_sliders'].keys())]
        fast_filter_inputs = [c['all_frames_data_state'], c['per_metric_values_state'], c['analysis_output_dir_state'],
                              c['gallery_view_toggle'], c['show_mask_overlay_input'], c['overlay_alpha_slider'],
                              c['require_face_match_input'], c['dedup_thresh_input'], c['dedup_method_input']] + slider_comps
        fast_filter_outputs = [c['filter_status_text'], c['results_gallery']]
        for control in (slider_comps + [c['dedup_thresh_input'], c['gallery_view_toggle'], c['show_mask_overlay_input'],
                                       c['overlay_alpha_slider'], c['require_face_match_input'], c['dedup_method_input']]):
            (control.release if hasattr(control, 'release') else control.input if hasattr(control, 'input') else control.change)(self.on_filters_changed_wrapper, fast_filter_inputs, fast_filter_outputs)

        load_outputs = ([c['all_frames_data_state'], c['per_metric_values_state'], c['filter_status_text'], c['results_gallery'],
                         c['results_group'], c['export_group']] +
                        [c['metric_plots'].get(k) for k in self.get_all_filter_keys() if c['metric_plots'].get(k)] +
                        slider_comps + [c['require_face_match_input']] +
                        [c['metric_accs'].get(k) for k in sorted(c['metric_accs'].keys()) if c['metric_accs'].get(k)])

        def load_and_trigger_update(output_dir):
            if not output_dir:
                # Return an update for every component in load_outputs to avoid errors
                return [gr.update()] * len(load_outputs)
            all_frames, metric_values = load_and_prep_filter_data(output_dir, self.get_all_filter_keys(), self.config)
            svgs = build_all_metric_svgs(metric_values, self.get_all_filter_keys(), self.logger)

            updates = {
                c['all_frames_data_state']: all_frames,
                c['per_metric_values_state']: metric_values,
                c['results_group']: gr.update(visible=True),
                c['export_group']: gr.update(visible=True)
            }

            # Update visibility and content of metric accordions and plots
            for k in self.get_all_filter_keys():
                acc = c['metric_accs'].get(k)
                plot_comp = c['metric_plots'].get(k)
                has_data = k in metric_values and metric_values.get(k)

                if acc:
                    updates[acc] = gr.update(visible=has_data)
                if plot_comp and has_data:
                    updates[plot_comp] = gr.update(value=svgs.get(k, ""))

            # Initial filter application
            slider_values_dict = {key: c['metric_sliders'][key].value for key in slider_keys}
            slider_values_dict['enable_dedup'] = c['enable_dedup_input'].value
            filter_event = FilterEvent(
                all_frames_data=all_frames, per_metric_values=metric_values, output_dir=output_dir,
                gallery_view="Kept Frames", show_overlay=self.config.gradio_show_mask_overlay,
                overlay_alpha=self.config.gradio_overlay_alpha,
                require_face_match=c['require_face_match_input'].value,
                dedup_thresh=c['dedup_thresh_input'].value, slider_values=slider_values_dict
            )
            filter_updates = on_filters_changed(filter_event, self.thumbnail_manager, self.config, self.logger)
            updates.update({
                c['filter_status_text']: filter_updates['filter_status_text'],
                c['results_gallery']: filter_updates['results_gallery']
            })

            # Create a list of updates in the correct order for the outputs
            final_updates_list = [updates.get(comp, gr.update()) for comp in load_outputs]
            return final_updates_list

        c['filtering_tab'].select(
            load_and_trigger_update,
            [c['analysis_output_dir_state']],
            load_outputs
        )

        export_inputs = [c['all_frames_data_state'], c['analysis_output_dir_state'], c['extracted_video_path_state'], c['enable_crop_input'],
                         c['crop_ar_input'], c['crop_padding_input'], c['require_face_match_input'], c['dedup_thresh_input'], c['dedup_method_input']] + slider_comps
        c['export_button'].click(self.export_kept_frames_wrapper, export_inputs, c['unified_log'])
        c['dry_run_button'].click(self.dry_run_export_wrapper, export_inputs, c['unified_log'])

        reset_outputs_comps = (slider_comps + [c['dedup_thresh_input'], c['require_face_match_input'],
                                              c['filter_status_text'], c['results_gallery']] +
                               [c['metric_accs'][k] for k in sorted(c['metric_accs'].keys())] +
                               [c['dedup_method_input']]) # Add the dropdown to the outputs

        c['reset_filters_button'].click(self.on_reset_filters, [c['all_frames_data_state'], c['per_metric_values_state'], c['analysis_output_dir_state']], reset_outputs_comps)

        auto_threshold_checkboxes = [c['metric_auto_threshold_cbs'][k] for k in sorted(c['metric_auto_threshold_cbs'].keys())]
        auto_set_inputs = [c['per_metric_values_state'], c['auto_pctl_input']] + auto_threshold_checkboxes

        c['apply_auto_button'].click(
            self.on_auto_set_thresholds,
            auto_set_inputs,
            [c['metric_sliders'][k] for k in slider_keys]
        ).then(
            self.on_filters_changed_wrapper,
            fast_filter_inputs,
            fast_filter_outputs
        )

        all_accordions = list(c['metric_accs'].values())
        c['expand_all_metrics_button'].click(lambda: {acc: gr.update(open=True) for acc in all_accordions}, [], all_accordions)
        c['collapse_all_metrics_button'].click(lambda: {acc: gr.update(open=False) for acc in all_accordions}, [], all_accordions)

        c['dedup_method_input'].change(
            lambda method: {
                c['dedup_thresh_input']: gr.update(visible=method == 'pHash', label=f"{method} Threshold"),
                c['ssim_threshold_input']: gr.update(visible=method == 'SSIM'),
                c['lpips_threshold_input']: gr.update(visible=method == 'LPIPS')
            },
            c['dedup_method_input'],
            [c['dedup_thresh_input'], c['ssim_threshold_input'], c['lpips_threshold_input']]
        )

        c['dedup_visual_diff_input'].change(
            lambda x: {
                c['visual_diff_image']: gr.update(visible=x),
                c['calculate_diff_button']: gr.update(visible=x)
            },
            c['dedup_visual_diff_input'],
            [c['visual_diff_image'], c['calculate_diff_button']]
        )

        c['calculate_diff_button'].click(
            self.calculate_visual_diff,
            [c['results_gallery'], c['all_frames_data_state'], c['dedup_method_input'], c['dedup_thresh_input'], c['ssim_threshold_input'], c['lpips_threshold_input']],
            [c['visual_diff_image']]
        )

        c['filter_preset_dropdown'].change(
            self.on_preset_changed,
            [c['filter_preset_dropdown']],
            list(c['metric_sliders'].values())
        ).then(
            self.on_filters_changed_wrapper,
            fast_filter_inputs,
            fast_filter_outputs
        )

    def on_preset_changed(self, preset_name: str) -> dict:
        """Applies a filter preset by updating the values of the metric sliders.

        Args:
            preset_name (str): The name of the selected preset from the dropdown.

        Returns:
            dict: A dictionary of Gradio updates for the sliders.
        """
        updates = {}
        slider_keys = sorted(self.components['metric_sliders'].keys())

        if preset_name == "None" or preset_name not in self.FILTER_PRESETS:
            # If "None" is selected, reset all sliders to their defaults.
            for key in slider_keys:
                metric_key = re.sub(r'_(min|max)$', '', key)
                default_key = 'default_max' if key.endswith('_max') else 'default_min'
                f_def = getattr(self.config, f"filter_default_{metric_key}", {})
                default_val = f_def.get(default_key, 0)
                updates[self.components['metric_sliders'][key]] = gr.update(value=default_val)
            return updates

        preset = self.FILTER_PRESETS[preset_name]
        for key in slider_keys:
            if key in preset:
                updates[self.components['metric_sliders'][key]] = gr.update(value=preset[key])
            else:
                # If a slider is not in the preset, reset it to its default
                metric_key = re.sub(r'_(min|max)$', '', key)
                default_key = 'default_max' if key.endswith('_max') else 'default_min'
                f_def = getattr(self.config, f"filter_default_{metric_key}", {})
                default_val = f_def.get(default_key, 0)
                updates[self.components['metric_sliders'][key]] = gr.update(value=default_val)

        return updates


    def on_filters_changed_wrapper(self, all_frames_data: list, per_metric_values: dict, output_dir: str, gallery_view: str, show_overlay: bool, overlay_alpha: float, require_face_match: bool, dedup_thresh: int, dedup_method: str, *slider_values: float) -> tuple[str, gr.update]:
        """Wrapper for the `on_filters_changed` event handler.

        This method collects all the disparate arguments from the Gradio UI,
        packages them into a `FilterEvent` object, and calls the main
        `on_filters_changed` function.

        Args:
            all_frames_data (list): A list of all frame data.
            per_metric_values (dict): A dictionary of metric values.
            output_dir (str): The path to the output directory.
            gallery_view (str): The current gallery view.
            show_overlay (bool): Whether to show the mask overlay.
            overlay_alpha (float): The transparency of the mask overlay.
            require_face_match (bool): Whether to require a face match.
            dedup_thresh (int): The deduplication threshold.
            dedup_method (str): The deduplication method.
            *slider_values (float): The values from the filter sliders.

        Returns:
            tuple[str, gr.update]: A tuple containing the updated status text
            and a Gradio update for the results gallery.
        """
        slider_values_dict = {k: v for k, v in zip(sorted(self.components['metric_sliders'].keys()), slider_values)}

        # "None" in the dropdown disables deduplication
        enable_dedup = dedup_method != "None"

        event_filters = slider_values_dict
        event_filters['enable_dedup'] = enable_dedup
        event_filters['dedup_method'] = dedup_method

        result = on_filters_changed(FilterEvent(all_frames_data, per_metric_values, output_dir, gallery_view, show_overlay, overlay_alpha,
                                                require_face_match, dedup_thresh, event_filters, dedup_method),
                                    self.thumbnail_manager, self.config)
        return result['filter_status_text'], result['results_gallery']

    def calculate_visual_diff(self, gallery: gr.Gallery, all_frames_data: list, dedup_method: str, dedup_thresh: int, ssim_thresh: float, lpips_thresh: float) -> Optional[np.ndarray]:
        """Calculates and displays a visual diff for a selected duplicate frame.

        Args:
            gallery (gr.Gallery): The results gallery component.
            all_frames_data (list): A list of all frame data.
            dedup_method (str): The current deduplication method.
            dedup_thresh (int): The pHash threshold.
            ssim_thresh (float): The SSIM threshold.
            lpips_thresh (float): The LPIPS threshold.

        Returns:
            Optional[np.ndarray]: A side-by-side comparison image of the two
            duplicate frames, or None if no duplicate is found.
        """
        if not gallery or not gallery.selection:
            return None

        selected_image_index = gallery.selection['index']
        selected_frame_data = all_frames_data[selected_image_index]

        # Find the duplicate frame
        duplicate_frame_data = None
        for frame_data in all_frames_data:
            if frame_data['filename'] == selected_frame_data['filename']:
                continue

            if dedup_method == "pHash":
                hash1 = imagehash.hex_to_hash(selected_frame_data['phash'])
                hash2 = imagehash.hex_to_hash(frame_data['phash'])
                if hash1 - hash2 <= dedup_thresh:
                    duplicate_frame_data = frame_data
                    break
            elif dedup_method == "SSIM":
                img1 = self.thumbnail_manager.get(Path(self.config.paths.downloads) / Path(selected_frame_data['filename']).parent.name / "thumbs" / selected_frame_data['filename'])
                img2 = self.thumbnail_manager.get(Path(self.config.paths.downloads) / Path(frame_data['filename']).parent.name / "thumbs" / frame_data['filename'])
                if img1 is not None and img2 is not None:
                    img1_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
                    img2_gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
                    similarity = ssim(img1_gray, img2_gray)
                    if similarity >= ssim_thresh:
                        duplicate_frame_data = frame_data
                        break
            elif dedup_method == "LPIPS":
                loss_fn = lpips.LPIPS(net='alex')
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])
                img1 = self.thumbnail_manager.get(Path(self.config.paths.downloads) / Path(selected_frame_data['filename']).parent.name / "thumbs" / selected_frame_data['filename'])
                img2 = self.thumbnail_manager.get(Path(self.config.paths.downloads) / Path(frame_data['filename']).parent.name / "thumbs" / frame_data['filename'])
                if img1 is not None and img2 is not None:
                    img1_t = transform(img1).unsqueeze(0)
                    img2_t = transform(img2).unsqueeze(0)
                    distance = loss_fn.forward(img1_t, img2_t).item()
                    if distance <= lpips_thresh:
                        duplicate_frame_data = frame_data
                        break

        if duplicate_frame_data:
            img1 = self.thumbnail_manager.get(Path(self.config.paths.downloads) / Path(selected_frame_data['filename']).parent.name / "thumbs" / selected_frame_data['filename'])
            img2 = self.thumbnail_manager.get(Path(self.config.paths.downloads) / Path(duplicate_frame_data['filename']).parent.name / "thumbs" / duplicate_frame_data['filename'])

            if img1 is not None and img2 is not None:
                # Create a side-by-side comparison image
                h, w, _ = img1.shape
                comparison_image = np.zeros((h, w * 2, 3), dtype=np.uint8)
                comparison_image[:, :w] = img1
                comparison_image[:, w:] = img2
                return comparison_image
        return None

    def on_reset_filters(self, all_frames_data: list, per_metric_values: dict, output_dir: str) -> tuple:
        """Handler for the 'Reset Filters' button.

        This function resets all filter sliders and controls to their default
        values as defined in the application configuration. It then re-applies
        the filters and updates the gallery.

        Args:
            all_frames_data (list): A list of all frame data dictionaries.
            per_metric_values (dict): A dictionary of metric values.
            output_dir (str): The path to the output directory.

        Returns:
            tuple: A tuple of Gradio updates for all filter controls and the
            results gallery.
        """
        c = self.components
        slider_keys = sorted(c['metric_sliders'].keys())
        acc_keys = sorted(c['metric_accs'].keys())
        slider_default_values = []
        slider_updates = []
        for key in slider_keys:
            metric_key = re.sub(r'_(min|max)$', '', key)
            default_key = 'default_max' if key.endswith('_max') else 'default_min'
            f_def = getattr(self.config, f"filter_default_{metric_key}", {})
            default_val = f_def.get(default_key, 0)
            slider_updates.append(gr.update(value=default_val))
            slider_default_values.append(default_val)
        face_match_default = self.config.default_require_face_match
        dedup_default = self.config.filter_default_dedup_thresh['default']
        dedup_update = gr.update(value=dedup_default)
        face_match_update = gr.update(value=face_match_default)
        if all_frames_data:
            slider_defaults_dict = {key: val for key, val in zip(slider_keys, slider_default_values)}

            filter_event = FilterEvent(
                all_frames_data=all_frames_data,
                per_metric_values=per_metric_values,
                output_dir=output_dir,
                gallery_view="Kept",
                show_overlay=self.config.gradio_show_mask_overlay,
                overlay_alpha=self.config.gradio_overlay_alpha,
                require_face_match=face_match_default,
                dedup_thresh=dedup_default,
                dedup_method="pHash",
                slider_values=slider_defaults_dict
            )
            filter_updates = on_filters_changed(filter_event, self.thumbnail_manager, self.config)
            status_update = filter_updates['filter_status_text']
            gallery_update = filter_updates['results_gallery']
        else:
            status_update = "Load an analysis to begin."
            gallery_update = gr.update(value=[])
        acc_updates = []
        # When resetting, also reset the visibility of the accordions based on available data
        for key in acc_keys:
            if all_frames_data:
                visible = False
                if key == 'dedup':
                    visible = any('phash' in f for f in all_frames_data)
                elif key == 'face_sim':
                    visible = 'face_sim' in per_metric_values and any(per_metric_values['face_sim'])
                else:
                    visible = key in per_metric_values

                # Determine if it should be open
                preferred_open = next((candidate for candidate in ['quality_score', 'sharpness'] if candidate in per_metric_values), None)
                acc_updates.append(gr.update(visible=visible, open=(key == preferred_open)))
            else:
                acc_updates.append(gr.update(visible=False))

        # Add the update for the dropdown itself to the returned tuple
        dedup_method_update = gr.update(value="pHash")

        return tuple(slider_updates + [dedup_update, face_match_update, status_update, gallery_update] + acc_updates + [dedup_method_update])

    def on_auto_set_thresholds(self, per_metric_values: dict, p: int, *checkbox_values: bool) -> list[gr.update]:
        """Handler for the 'Apply Percentile to Mins' button.

        Args:
            per_metric_values (dict): A dictionary of metric values.
            p (int): The percentile to calculate.
            *checkbox_values (bool): The boolean values from the
                "Auto-Threshold this metric" checkboxes.

        Returns:
            list[gr.update]: A list of Gradio updates for the sliders.
        """
        slider_keys = sorted(self.components['metric_sliders'].keys())
        auto_threshold_cbs_keys = sorted(self.components['metric_auto_threshold_cbs'].keys())
        selected_metrics = [metric_name for metric_name, is_selected in zip(auto_threshold_cbs_keys, checkbox_values) if is_selected]
        updates = auto_set_thresholds(per_metric_values, p, slider_keys, selected_metrics)
        return [updates.get(f'slider_{key}', gr.update()) for key in slider_keys]

    def export_kept_frames_wrapper(self, all_frames_data: list, output_dir: str, video_path: str, enable_crop: bool, crop_ars: str, crop_padding: int, require_face_match: bool, dedup_thresh: int, dedup_method: str, *slider_values: float) -> str:
        """Wrapper for the export function.

        Gathers all filter and export settings from the UI, creates an
        `ExportEvent`, and calls the main `export_kept_frames` method.

        Args:
            all_frames_data (list): A list of all frame data.
            output_dir (str): The path to the output directory.
            video_path (str): The path to the original source video.
            enable_crop (bool): Whether to enable cropping.
            crop_ars (str): Comma-separated list of aspect ratios for cropping.
            crop_padding (int): The padding percentage for cropping.
            require_face_match (bool): Whether to require a face match.
            dedup_thresh (int): The deduplication threshold.
            dedup_method (str): The deduplication method.
            *slider_values (float): The values from the filter sliders.

        Returns:
            str: A status message indicating the result of the export.
        """
        filter_args = {k: v for k, v in zip(sorted(self.components['metric_sliders'].keys()), slider_values)}
        enable_dedup = dedup_method != "None"
        filter_args.update({"require_face_match": require_face_match, "dedup_thresh": dedup_thresh, "dedup_method": dedup_method, "enable_dedup": enable_dedup})
        return self.export_kept_frames(ExportEvent(all_frames_data, output_dir, video_path, enable_crop, crop_ars, crop_padding, filter_args))

    def export_kept_frames(self, event: ExportEvent) -> str:
        """Exports the final selected frames.

        This function performs the final export. It re-applies the current
        filters to get the definitive list of "kept" frames. It then uses FFmpeg
        to extract these specific frames from the original, full-resolution
        video. It also handles renaming the exported files and an optional
        cropping step.

        Args:
            event (ExportEvent): The `ExportEvent` containing all data and
                settings for the export.

        Returns:
            str: A status message indicating the result of the export.
        """
        if not event.all_frames_data:
            return "No metadata to export."
        if not event.video_path or not Path(event.video_path).exists():
            return "[ERROR] Original video path is required for export."

        out_root = Path(event.output_dir)

        try:
            filters = event.filter_args.copy()
            filters.update({
                "face_sim_enabled": any("face_sim" in f for f in event.all_frames_data),
                "mask_area_enabled": any("mask_area_pct" in f for f in event.all_frames_data),
                "enable_dedup": any('phash' in f for f in event.all_frames_data)
            })
            kept, _, _, _ = apply_all_filters_vectorized(event.all_frames_data, filters, self.config, output_dir=event.output_dir)
            if not kept:
                return "No frames kept after filtering. Nothing to export."

            frame_map_path = out_root / "frame_map.json"
            if not frame_map_path.exists():
                return "[ERROR] frame_map.json not found. Cannot export."
            with frame_map_path.open('r', encoding='utf-8') as f:
                frame_map_list = json.load(f)

            sample_name = next((f['filename'] for f in kept if 'filename' in f), None)
            analyzed_ext = Path(sample_name).suffix if sample_name else '.webp'

            fn_to_orig_map = {
                f"frame_{i+1:06d}{analyzed_ext}": orig
                for i, orig in enumerate(sorted(frame_map_list))
            }
            frames_to_extract = sorted([
                fn_to_orig_map.get(f['filename'])
                for f in kept if f.get('filename') in fn_to_orig_map
            ])
            frames_to_extract = [n for n in frames_to_extract if n is not None]

            if not frames_to_extract:
                return "No frames to extract."

            select_filter = f"select='{'+'.join([f'eq(n,{fn})' for fn in frames_to_extract])}'"
            export_dir = out_root.parent / f"{out_root.name}_exported_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            export_dir.mkdir(exist_ok=True, parents=True)
            cmd = ['ffmpeg', '-y', '-i', str(event.video_path), '-vf', select_filter, '-vsync', 'vfr', str(export_dir / "frame_%06d.png")]
            self.logger.info("Starting final export extraction...", extra={'command': ' '.join(cmd)})

            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8')
            stdout, stderr = process.communicate()

            if process.returncode != 0:
                self.logger.error("FFmpeg export failed", extra={'stderr': stderr})
                return f"Error during export: FFmpeg failed. Check logs for details:\n{stderr}"

            self.logger.info("Renaming extracted frames to match original filenames...")
            orig_to_filename_map = {v: k for k, v in fn_to_orig_map.items()}
            plan = []
            for i, orig_frame_num in enumerate(frames_to_extract):
                sequential_filename = f"frame_{i+1:06d}.png"
                target_filename = orig_to_filename_map.get(orig_frame_num)
                if not target_filename:
                    continue
                src = export_dir / sequential_filename
                dst = export_dir / target_filename
                if src != dst:
                    plan.append((src, dst))

            temp_map = {}
            for i, (src, _) in enumerate(plan):
                if not src.exists(): continue
                tmp = export_dir / f"__tmp_{i:06d}__{src.name}"
                j = i
                while tmp.exists():
                    j += 1
                    tmp = export_dir / f"__tmp_{j:06d}__{src.name}"
                try:
                    src.rename(tmp)
                    temp_map[src] = tmp
                except FileNotFoundError:
                    self.logger.warning(f"Could not find {src.name} to rename.", extra={'target': tmp.name})

            for src, dst in plan:
                tmp = temp_map.get(src)
                if tmp and tmp.exists():
                    if dst.exists():
                        stem, ext = dst.stem, dst.suffix
                        k, alt = 1, export_dir / f"{stem} (1){ext}"
                        while alt.exists():
                            k += 1
                            alt = export_dir / f"{stem} ({k}){ext}"
                        dst = alt
                    try:
                        tmp.rename(dst)
                    except FileNotFoundError:
                        self.logger.warning(f"Could not find temp file {tmp.name} to rename.", extra={'target': dst.name})

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
                        
                        feasible_candidates = []
                        for ar_str, r in aspect_ratios:
                            # Determine the smallest crop that contains the subject box at the target AR
                            if w_b / h_b > r:
                                w_c, h_c = w_b, w_b / r
                            else:
                                h_c, w_c = h_b, h_b * r

                            # Apply padding
                            w_padded, h_padded = w_c * padding_factor, h_c * padding_factor

                            # Clamp dimensions to frame boundaries, preserving AR
                            scale = 1.0
                            if w_padded > frame_w:
                                scale = min(scale, frame_w / w_padded)
                            if h_padded > frame_h:
                                scale = min(scale, frame_h / h_padded)
                            
                            w_final, h_final = w_padded * scale, h_padded * scale

                            # Ensure the final crop is still large enough to contain the subject
                            if w_final < w_b or h_final < h_b:
                                # This can happen if padding is negative or scale-down is too aggressive.
                                # As a fallback, ensure subject is contained, which may break padding.
                                if w_final < w_b:
                                    w_final = w_b
                                    h_final = w_final / r
                                if h_final < h_b:
                                    h_final = h_b
                                    w_final = h_final * r
                                # Re-clamp to frame boundaries after this adjustment
                                if w_final > frame_w:
                                    w_final = frame_w
                                    h_final = w_final / r
                                if h_final > frame_h:
                                    h_final = frame_h
                                    w_final = h_final * r
                            
                            # Position the crop to center the subject, then clamp to frame edges
                            center_x_b, center_y_b = x_b + w_b / 2, y_b + h_b / 2
                            x1 = center_x_b - w_final / 2
                            y1 = center_y_b - h_final / 2

                            x1 = max(0, min(x1, frame_w - w_final))
                            y1 = max(0, min(y1, frame_h - h_final))

                            # Final check to ensure subject is contained after all adjustments
                            if (x1 > x_b or y1 > y_b or x1 + w_final < x_b + w_b or y1 + h_final < y_b + h_b):
                                continue # This AR is not feasible with the given constraints

                            feasible_candidates.append({
                                "ar_str": ar_str,
                                "x1": x1, "y1": y1,
                                "w_r": w_final, "h_r": h_final,
                                "area": w_final * h_final,
                            })

                        if not feasible_candidates:
                            # Fallback to native subject box if no AR is feasible
                            cropped_img = frame_img[y_b:y_b+h_b, x_b:x_b+w_b]
                            if cropped_img.size > 0:
                                cv2.imwrite(str(crop_dir / f"{Path(frame_meta['filename']).stem}_crop_native.png"), cropped_img)
                                num_cropped += 1
                            continue

                        # Select the best candidate (smallest area that contains the subject)
                        subject_ar = w_b / h_b if h_b > 0 else 1
                        def sort_key(c):
                            r = c['w_r'] / c['h_r'] if c['h_r'] > 0 else 1
                            ar_diff = abs(r - subject_ar)
                            return (c['area'], ar_diff)

                        best_candidate = min(feasible_candidates, key=sort_key)
                        
                        x1, y1 = int(best_candidate['x1']), int(best_candidate['y1'])
                        w_r, h_r = int(best_candidate['w_r']), int(best_candidate['h_r'])

                        cropped_img = frame_img[y1:y1+h_r, x1:x1+w_r]
                        if cropped_img.size > 0:
                            cv2.imwrite(str(crop_dir / f"{Path(frame_meta['filename']).stem}_crop_{best_candidate['ar_str']}.png"), cropped_img)
                            num_cropped += 1

                    except Exception as e:
                        self.logger.error(f"Failed to crop frame {frame_meta['filename']}", exc_info=True)
                self.logger.info(f"Cropping complete. Saved {num_cropped} cropped images.")
            return f"Exported {len(kept)} frames to {export_dir.name}."
        except subprocess.CalledProcessError as e:
            self.logger.error("FFmpeg export failed", exc_info=True, extra={'stderr': e.stderr})
            return "Error during export: FFmpeg failed. Check logs."
        except Exception as e:
            self.logger.error("Error during export process", exc_info=True)
            return f"Error during export: {e}"

    def dry_run_export_wrapper(self, all_frames_data: list, output_dir: str, video_path: str, enable_crop: bool, crop_ars: str, crop_padding: int, require_face_match: bool, dedup_thresh: int, dedup_method: str, *slider_values: float) -> str:
        """Wrapper for the dry run export function.

        Gathers all filter and export settings from the UI and calls the
        main `dry_run_export` method.

        Args:
            all_frames_data (list): A list of all frame data.
            output_dir (str): The path to the output directory.
            video_path (str): The path to the original source video.
            enable_crop (bool): Whether to enable cropping.
            crop_ars (str): Comma-separated list of aspect ratios for cropping.
            crop_padding (int): The padding percentage for cropping.
            require_face_match (bool): Whether to require a face match.
            dedup_thresh (int): The deduplication threshold.
            dedup_method (str): The deduplication method.
            *slider_values (float): The values from the filter sliders.

        Returns:
            str: A status message containing the dry run information.
        """
        filter_args = {k: v for k, v in zip(sorted(self.components['metric_sliders'].keys()), slider_values)}
        filter_args.update({"require_face_match": require_face_match, "dedup_thresh": dedup_thresh, "enable_dedup": dedup_method != "None"})
        return self.dry_run_export(ExportEvent(all_frames_data=all_frames_data, output_dir=output_dir, video_path=video_path, enable_crop=enable_crop, crop_ars=crop_ars, crop_padding=crop_padding, filter_args=filter_args))

    def dry_run_export(self, event: ExportEvent) -> str:
        """Performs a dry run of the export process.

        This function calculates which frames would be exported based on the
        current filters but does not actually run FFmpeg. Instead, it returns
        a message indicating how many frames would be exported and the FFmpeg
        command that would be used.

        Args:
            event (ExportEvent): The `ExportEvent` containing all data and
                settings for the export.

        Returns:
            str: A status message with the dry run information.
        """
        if not event.all_frames_data:
            return "No metadata to export."
        if not event.video_path or not Path(event.video_path).exists():
            return "[ERROR] Original video path is required for export."

        out_root = Path(event.output_dir)

        try:
            filters = event.filter_args.copy()
            filters.update({
                "face_sim_enabled": any("face_sim" in f for f in event.all_frames_data),
                "mask_area_enabled": any("mask_area_pct" in f for f in event.all_frames_data),
                "enable_dedup": any('phash' in f for f in event.all_frames_data)
            })
            kept, _, _, _ = apply_all_filters_vectorized(event.all_frames_data, filters, self.config, output_dir=event.output_dir)
            if not kept:
                return "No frames kept after filtering. Nothing to export."

            frame_map_path = out_root / "frame_map.json"
            if not frame_map_path.exists():
                return "[ERROR] frame_map.json not found. Cannot export."
            with frame_map_path.open('r', encoding='utf-8') as f:
                frame_map_list = json.load(f)

            sample_name = next((f['filename'] for f in kept if 'filename' in f), None)
            analyzed_ext = Path(sample_name).suffix if sample_name else '.webp'

            fn_to_orig_map = {
                f"frame_{i+1:06d}{analyzed_ext}": orig
                for i, orig in enumerate(sorted(frame_map_list))
            }
            frames_to_extract = sorted([
                fn_to_orig_map.get(f['filename'])
                for f in kept if f.get('filename') in fn_to_orig_map
            ])
            frames_to_extract = [n for n in frames_to_extract if n is not None]

            if not frames_to_extract:
                return "No frames to extract."

            select_filter = f"select='{'+'.join([f'eq(n,{fn})' for fn in frames_to_extract])}'"
            export_dir = out_root.parent / f"{out_root.name}_exported_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            cmd = ['ffmpeg', '-y', '-i', str(event.video_path), '-vf', select_filter, '-vsync', 'vfr', str(export_dir / "frame_%06d.png")]

            return f"Dry Run: {len(frames_to_extract)} frames to be exported.\n\nFFmpeg command:\n{' '.join(cmd)}"
        except Exception as e:
            self.logger.error("Error during dry run process", exc_info=True)
            return f"Error during dry run: {e}"

# --- COMPOSITION & MAIN ---

# Global instance of the model registry, initialized by CompositionRoot.
model_registry: Optional['ModelRegistry'] = None

def cleanup_models():
    """Clears all global model caches and releases GPU memory.

    This function is crucial for ensuring a clean state, especially during
    application shutdown or when re-initializing models, to prevent memory
    leaks and unexpected behavior from stale cached objects.
    """
    global model_registry
    if model_registry:
        model_registry.clear()

    torch.cuda.empty_cache()
    gc.collect()


class CompositionRoot:
    """Manages the application's object graph and lifecycle.

    This class acts as a central point for creating and accessing the core
    components of the application, such as configuration, logging, and the UI.
    It follows the Composition Root pattern to handle dependency injection,
    ensuring that dependencies are created and wired together in one place.

    Attributes:
        config (Config): The application configuration object.
        logger (AppLogger): The application logger.
        model_registry (ModelRegistry): The registry for ML models.
        thumbnail_manager (ThumbnailManager): The manager for the thumbnail cache.
        progress_queue (Queue): A queue for UI progress updates.
        cancel_event (threading.Event): An event to signal task cancellation.
    """
    def __init__(self):
        """Initializes the CompositionRoot and creates core services."""
        global model_registry
        self.config = Config()
        self.logger = AppLogger(config=self.config)
        self.model_registry = ModelRegistry(logger=self.logger)
        model_registry = self.model_registry # Assign to global instance
        self.thumbnail_manager = ThumbnailManager(self.logger, self.config)
        self.progress_queue = Queue()
        self.cancel_event = threading.Event()
        self.logger.set_progress_queue(self.progress_queue)
        self._app_ui = None

    def get_app_ui(self) -> 'EnhancedAppUI':
        """Lazily initializes and returns the main application UI instance.

        Returns:
            The singleton instance of the `EnhancedAppUI`.
        """
        if self._app_ui is None:
            self._app_ui = EnhancedAppUI(config=self.config, logger=self.logger, progress_queue=self.progress_queue,
                                         cancel_event=self.cancel_event, thumbnail_manager=self.get_thumbnail_manager())
        return self._app_ui

    def get_config(self) -> 'Config':
        """Returns the application configuration object."""
        return self.config

    def get_logger(self) -> 'AppLogger':
        """Returns the application logger instance."""
        return self.logger

    def get_thumbnail_manager(self) -> 'ThumbnailManager':
        """Returns the thumbnail manager instance."""
        return self.thumbnail_manager

    def cleanup(self):
        """Performs cleanup of all application resources.

        This method should be called on application shutdown to ensure graceful
        termination. It clears the model registry and thumbnail cache to free
        up memory and resources, and sets the cancel event to stop any running
        background threads.
        """
        cleanup_models()
        self.thumbnail_manager.clear_cache()
        if hasattr(self, '_app_ui'):
            self._app_ui = None
        self.cancel_event.set()


def main():
    """The main entry point for the application.

    This function initializes the `CompositionRoot` to set up all dependencies,
    builds the Gradio UI by calling `build_ui`, launches the web server, and
    ensures that the `cleanup` method is called on exit.
    """
    try:
        composition = CompositionRoot()
        logger = composition.get_logger()
        demo = composition.get_app_ui().build_ui()
        logger.info("Frame Extractor & Analyzer v2.0\nStarting application...")
        demo.launch()
    except KeyboardInterrupt:
        logger.info("\nApplication stopped by user")
    except Exception as e:
        logger.error(f"Error starting application: {e}", exc_info=True)
        sys.exit(1)
    finally:
        if 'composition' in locals():
            composition.cleanup()

if __name__ == "__main__":
    main()