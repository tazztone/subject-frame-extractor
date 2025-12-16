---
Version: 2.0
Last Updated: 2025-12-16
Python: 3.10+
Gradio: 6.x
SAM3: Via submodule
---

# Developer Guidelines & Agent Memory

**⚠️ CRITICAL**: Read this before starting any task.

🔴 CRITICAL | 🟡 WARNING | 🟢 BEST PRACTICE


## 1. Quick Start Guide

### 5-Minute Setup
1. **Clone & Submodules**: `git submodule update --init --recursive`
2. **Environment**: `python3 -m venv venv && source venv/bin/activate`
3. **Dependencies**: `pip install -r requirements.txt` (Installs SAM3 via submodule)
4. **Run App**: `python app.py`

### Essential Commands
- **Test Backend**: `python -m pytest tests/`
- **Test E2E**: `pytest tests/e2e/` (Requires Playwright)
- **Lint/Check**: `.claude/commands/validate.md` (if available)

### Directory Structure
- `app.py`: Entry point.
- `core/`: Business logic (pipelines, config, db).
- `ui/`: Gradio interface components.
- `tests/`: Unit and E2E tests.
- `SAM3_repo/`: **Read-only** submodule.


## 2. Critical Rules

### 🔴 CRITICAL (Must Follow)
- **NEVER** edit files in `SAM3_repo` or `Grounded-SAM-2`. Treat as external libraries.
- **ALWAYS** match Gradio event handler return values count to the `outputs` list. Mismatches crash the app silently.
- **NEVER** use `@lru_cache` on functions taking the `Config` object (it's unhashable). Use `model_registry.get_or_load`.
- **ALWAYS** use `pathlib.Path`, never `os.path`.
- **ALWAYS** mock external dependencies (SAM3, Torch) in unit tests.

### 🟡 WARNING (Potential Bugs)
- **Check Masks**: Verify masks exist on disk before export/processing.
- **Thread Safety**: MediaPipe objects are not thread-safe. Use thread-local storage or one instance per thread.
- **Gradio State**: Do not store locks or file handles in `gr.State`.

### 🟢 BEST PRACTICE
- **Refactoring**: Move logic from `app.py` to `core/`.
- **Typing**: Use Pydantic models (`core/events.py`) instead of untyped dicts.


## 3. Architecture Overview

### Data Flow
`UI (Gradio)` → `Event Object (Pydantic)` → `Pipeline (Core)` → `Database/Files`

### Component Relationship
```
[app.py] (UI Assembly)
   │
   ├─ [core/config.py] (Settings)
   ├─ [core/managers.py] (ModelRegistry, ThumbnailManager)
   └─ [core/pipelines.py] (Logic)
         │
         ├─ ExtractionPipeline (FFmpeg)
         ├─ AnalysisPipeline (SAM3, InsightFace)
         └─ ExportPipeline (Filtering, Rendering)
```

### State Management
- **Session State**: `gr.State` stores mutable data (scene lists, paths).
- **Global State**: `ModelRegistry` (Singleton-like) manages heavy models.
- **Persistence**: `metadata.db` (SQLite) for frame data; `json` for configs.

## 4. Code Skeleton Reference

### `📄 app.py`

```python
"""
Frame Extractor & Analyzer v2.0
"""

import sys
from pathlib import Path
import threading
from queue import Queue
import torch
import gc
from core.config import Config
from core.logger import AppLogger
from core.managers import ModelRegistry, ThumbnailManager
from ui.app_ui import AppUI

project_root = Path(__file__).parent
def cleanup_models(model_registry):
    ...

def main():
    ...

```

### `📄 core/__init__.py`

```python

```

### `📄 core/batch_manager.py`

```python
import threading
import uuid
import time
from typing import List, Optional, Callable, Dict
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor

class BatchStatus(Enum):
    PENDING = 'Pending'
    PROCESSING = 'Processing'
    COMPLETED = 'Completed'
    FAILED = 'Failed'
    CANCELLED = 'Cancelled'

@dataclass
class BatchItem:
    ...

class BatchManager:
    def __init__(self):
        ...
    def add_paths(self, paths: List[str]):
        ...
    def get_queue_snapshot(self) -> List[BatchItem]:
        ...
    def get_status_list(self) -> List[List]:
        ...
    def clear_completed(self):
        ...
    def clear_all(self):
        ...
    def update_progress(self, item_id: str, fraction: float, message: Optional[str]=None):
        ...
    def set_status(self, item_id: str, status: BatchStatus, message: Optional[str]=None):
        ...
    def start_processing(self, processor_func: Callable, max_workers: int=1):
        ...
    def _run_scheduler(self, processor_func, max_workers):
        ...
    def stop_processing(self):
        ...

```

### `📄 core/config.py`

```python
"""
Configuration Management for Frame Extractor & Analyzer
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

def json_config_settings_source() -> Dict[str, Any]:
    """
    Loads settings from a JSON file for Pydantic settings.
    """
    ...

class Config(BaseSettings):
    """
    Manages the application's configuration settings.
    """
    model_config = SettingsConfigDict(env_file='.env', env_prefix='APP_', env_nested_delimiter='_', case_sensitive=False)
    def model_post_init(self, __context: Any) -> None:
        ...
    def _validate_paths(self):
        """
        Ensures critical directories exist.
        """
        ...
    @model_validator(mode='after')
    def _validate_config(self) -> 'Config':
        ...
    @property
    def quality_weights(self) -> Dict[str, int]:
        ...

```

### `📄 core/database.py`

```python
import sqlite3
import json
import threading
from pathlib import Path
from typing import List, Dict, Any

class Database:
    def __init__(self, db_path: Path, batch_size: int=50):
        ...
    def connect(self):
        """
        Connects to the SQLite database.
        """
        ...
    def close(self):
        """
        Closes the database connection.
        """
        ...
    def create_tables(self):
        """
        Creates the necessary tables if they don't exist.
        """
        ...
    def clear_metadata(self):
        """
        Deletes all records from the metadata table.
        """
        ...
    def insert_metadata(self, metadata: Dict[str, Any]):
        """
        Inserts or replaces a metadata record.
        """
        ...
    def flush(self):
        """
        Manually flush the buffer.
        """
        ...
    def _flush_buffer(self):
        ...
    def load_all_metadata(self) -> List[Dict[str, Any]]:
        """
        Loads all metadata from the database.
        """
        ...
    def count_errors(self) -> int:
        """
        Counts the number of records with errors.
        """
        ...

```

### `📄 core/error_handling.py`

```python
"""
Error Handling Infrastructure for Frame Extractor & Analyzer
"""

import functools
import time
import traceback
from enum import Enum
from typing import Any, Callable, Optional

class ErrorSeverity(Enum):
    LOW = 'low'
    MEDIUM = 'medium'
    HIGH = 'high'
    CRITICAL = 'critical'

class RecoveryStrategy(Enum):
    RETRY = 'retry'
    FALLBACK = 'fallback'
    SKIP = 'skip'
    ABORT = 'abort'

class ErrorHandler:
    def __init__(self, logger: 'AppLogger', max_attempts: int, backoff_seconds: list):
        ...
    def with_retry(self, max_attempts: Optional[int]=None, backoff_seconds: Optional[list]=None, recoverable_exceptions: tuple=(Exception,)):
        ...
    def with_fallback(self, fallback_func: Callable):
        ...

```

### `📄 core/events.py`

```python
"""
Event Models for Frame Extractor & Analyzer

Pydantic models representing UI events and data contracts.
"""

from pathlib import Path
from typing import Any, Optional
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

class UIEvent(BaseModel):
    model_config = ConfigDict(validate_assignment=True, extra='ignore', str_strip_whitespace=True, arbitrary_types_allowed=True)

class ExtractionEvent(UIEvent):
    ...

class PreAnalysisEvent(UIEvent):
    @field_validator('face_ref_img_path')
    @classmethod
    def validate_face_ref(cls, v: str, info) -> str:
        ...
    @model_validator(mode='after')
    def validate_strategy_consistency(self) -> 'PreAnalysisEvent':
        ...

class PropagationEvent(UIEvent):
    ...

class FilterEvent(UIEvent):
    ...

class ExportEvent(UIEvent):
    ...

class SessionLoadEvent(UIEvent):
    ...

```

### `📄 core/export.py`

```python
from __future__ import annotations
import subprocess
import shutil
import cv2
import numpy as np
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, TYPE_CHECKING
from core.filtering import apply_all_filters_vectorized
from core.events import ExportEvent

def _perform_ffmpeg_export(video_path: str, frames_to_extract: list, export_dir: Path, logger: 'AppLogger') -> tuple[bool, Optional[str]]:
    ...

def _rename_exported_frames(export_dir: Path, frames_to_extract: list, fn_to_orig_map: dict, logger: 'AppLogger'):
    ...

def _crop_exported_frames(kept_frames: list, export_dir: Path, crop_ars: str, crop_padding: int, masks_root: Path, logger: 'AppLogger', cancel_event) -> int:
    ...

def export_kept_frames(event: ExportEvent, config: 'Config', logger: 'AppLogger', thumbnail_manager, cancel_event) -> str:
    ...

def dry_run_export(event: ExportEvent, config: 'Config') -> str:
    ...

```

### `📄 core/filtering.py`

```python
from __future__ import annotations
import collections
from collections import defaultdict, Counter
import io
import math
from typing import Optional, Union, List, Any, TYPE_CHECKING, Callable
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
from skimage.metrics import structural_similarity as ssim
import lpips
from torchvision import transforms
from core.database import Database
from core.managers import get_lpips_metric

def load_and_prep_filter_data(output_dir: str, get_all_filter_keys: Callable, config: 'Config') -> tuple[list, dict]:
    ...

def histogram_svg(hist_data: tuple, title: str='', logger: Optional['AppLogger']=None) -> str:
    ...

def build_all_metric_svgs(per_metric_values: dict, get_all_filter_keys: Callable, logger: 'AppLogger') -> dict:
    ...

def _extract_metric_arrays(all_frames_data: list[dict], config: 'Config') -> dict:
    ...

def _run_batched_lpips(pairs: list[tuple[int, int]], all_frames_data: list[dict], dedup_mask: np.ndarray, reasons: defaultdict, thumbnail_manager: 'ThumbnailManager', output_dir: str, threshold: float, device: str='cpu'):
    """
    Runs LPIPS deduplication on a list of pairs in batches using GPU if available.
    """
    ...

def _apply_deduplication_filter(all_frames_data: list[dict], filters: dict, thumbnail_manager: 'ThumbnailManager', config: 'Config', output_dir: str) -> tuple[np.ndarray, defaultdict]:
    ...

def _apply_metric_filters(all_frames_data: list[dict], metric_arrays: dict, filters: dict, config: 'Config') -> tuple[np.ndarray, defaultdict]:
    ...

def apply_all_filters_vectorized(all_frames_data: list[dict], filters: dict, config: 'Config', thumbnail_manager: Optional['ThumbnailManager']=None, output_dir: Optional[str]=None) -> tuple[list, list, Counter, dict]:
    ...

def _generic_dedup(all_frames_data: list[dict], dedup_mask: np.ndarray, reasons: defaultdict, thumbnail_manager: 'ThumbnailManager', output_dir: str, compare_fn: Callable[[np.ndarray, np.ndarray], bool]) -> tuple[np.ndarray, defaultdict]:
    ...

def _ssim_compare(img1: np.ndarray, img2: np.ndarray, threshold: float) -> bool:
    ...

def apply_ssim_dedup(all_frames_data: list[dict], filters: dict, dedup_mask: np.ndarray, reasons: defaultdict, thumbnail_manager: 'ThumbnailManager', config: 'Config', output_dir: str) -> tuple[np.ndarray, defaultdict]:
    ...

def apply_lpips_dedup(all_frames_data: list[dict], filters: dict, dedup_mask: np.ndarray, reasons: defaultdict, thumbnail_manager: 'ThumbnailManager', config: 'Config', output_dir: str) -> tuple[np.ndarray, defaultdict]:
    ...

```

### `📄 core/logger.py`

```python
"""
Logging Infrastructure for Frame Extractor & Analyzer
"""

import json
import logging
import traceback
from datetime import datetime
from pathlib import Path
from queue import Queue
from typing import Any, Dict, Optional
from pydantic import BaseModel

SUCCESS_LEVEL_NUM = 25
class LogEvent(BaseModel):
    """
    Represents a structured log entry.
    """
    ...

class ColoredFormatter(logging.Formatter):
    COLORS = {'DEBUG': '\x1b[36m', 'INFO': '\x1b[37m', 'WARNING': '\x1b[33m', 'ERROR': '\x1b[31m', 'CRITICAL': '\x1b[35m', 'SUCCESS': '\x1b[32m', 'RESET': '\x1b[0m'}
    def format(self, record: logging.LogRecord) -> str:
        ...

class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        ...

class AppLogger:
    """
    A comprehensive logger for the application.
    """
    def __init__(self, config: 'Config', log_dir: Optional[Path]=None, log_to_file: bool=True, log_to_console: bool=True):
        ...
    def _setup_console_handler(self):
        ...
    def _setup_file_handlers(self):
        ...
    def set_progress_queue(self, queue: Queue):
        ...
    def _create_log_event(self, level: str, message: str, component: str, **kwargs) -> LogEvent:
        ...
    def _log_event(self, event: LogEvent):
        ...
    def debug(self, message: str, component: str='system', **kwargs):
        ...
    def info(self, message: str, component: str='system', **kwargs):
        ...
    def warning(self, message: str, component: str='system', **kwargs):
        ...
    def error(self, message: str, component: str='system', **kwargs):
        ...
    def success(self, message: str, component: str='system', **kwargs):
        ...
    def critical(self, message: str, component: str='system', **kwargs):
        ...

```

### `📄 core/managers.py`

```python
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
from core.utils import download_model, validate_video_file, safe_resource_cleanup
from core.error_handling import ErrorHandler

build_sam3_video_predictor = None
Sam3VideoPredictor = None
class ThumbnailManager:
    def __init__(self, logger: 'AppLogger', config: 'Config'):
        ...
    def get(self, thumb_path: Path) -> Optional[np.ndarray]:
        ...
    def clear_cache(self):
        ...
    def _cleanup_old_entries(self):
        ...

class ModelRegistry:
    def __init__(self, logger: Optional['AppLogger']=None):
        ...
    def get_or_load(self, key: str, loader_fn: Callable[[], Any]) -> Any:
        ...
    def clear(self):
        ...
    def get_tracker(self, model_name: str, models_path: str, user_agent: str, retry_params: tuple, config: 'Config') -> Optional['SAM3Wrapper']:
        ...
    def _load_tracker_impl(self, model_name: str, models_path: str, user_agent: str, retry_params: tuple, device: str, config: 'Config') -> 'SAM3Wrapper':
        ...

class SAM3Wrapper:
    def __init__(self, checkpoint_path, device='cuda'):
        ...
    def initialize(self, images, init_mask=None, bbox=None, prompt_frame_idx=0):
        """
        Initialize session with images and optional prompt.
        images: List of PIL Images.
        bbox: [x, y, w, h]
        prompt_frame_idx: Index of the frame to apply the prompt to.
        """
        ...
    def propagate_from(self, start_idx, direction='forward'):
        """
        Yields results starting from start_idx in the given direction.
        """
        ...
    def detect_objects(self, image_rgb: np.ndarray, text_prompt: str) -> List[dict]:
        ...

thread_local = threading.local()
def get_face_landmarker(model_path: str, logger: 'AppLogger') -> vision.FaceLandmarker:
    ...

def get_face_analyzer(model_name: str, models_path: str, det_size_tuple: tuple, logger: 'AppLogger', model_registry: 'ModelRegistry', device: str='cpu') -> 'FaceAnalysis':
    ...

def get_lpips_metric(model_name: str='alex', device: str='cpu') -> torch.nn.Module:
    ...

def initialize_analysis_models(params: 'AnalysisParameters', config: 'Config', logger: 'AppLogger', model_registry: 'ModelRegistry') -> dict:
    ...

class VideoManager:
    def __init__(self, source_path: str, config: 'Config', max_resolution: Optional[str]=None):
        ...
    def prepare_video(self, logger: 'AppLogger') -> str:
        ...
    @staticmethod
    def get_video_info(video_path: str) -> dict:
        ...

```

### `📄 core/models.py`

```python
from __future__ import annotations
import math
from typing import Optional, List, Dict, Any, Union, Callable, TYPE_CHECKING
import numpy as np
from pydantic import BaseModel, Field, ConfigDict
import cv2
import torch
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pathlib import Path

def _coerce(val: Any, to_type: type) -> Any:
    ...

def _sanitize_face_ref(kwargs: dict, logger: 'AppLogger') -> tuple[str, bool]:
    ...

class QualityConfig(BaseModel):
    ...

class FrameMetrics(BaseModel):
    ...

class Frame(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    def calculate_quality_metrics(self, thumb_image_rgb: np.ndarray, quality_config: 'QualityConfig', logger: 'AppLogger', mask: Optional[np.ndarray]=None, niqe_metric: Optional[Callable]=None, main_config: Optional['Config']=None, face_landmarker: Optional[Callable]=None, face_bbox: Optional[List[int]]=None, metrics_to_compute: Optional[Dict[str, bool]]=None):
        ...

class Scene(BaseModel):
    ...

class SceneState:
    def __init__(self, scene_data: Union[dict, Scene]):
        ...
    @property
    def data(self) -> dict:
        ...
    @property
    def scene(self) -> Scene:
        ...
    def set_manual_bbox(self, bbox: list[int], source: str):
        ...
    def reset(self):
        ...
    def include(self):
        ...
    def exclude(self):
        ...
    def update_seed_result(self, bbox: Optional[list[int]], details: dict):
        ...

class AnalysisParameters(BaseModel):
    @classmethod
    def from_ui(cls, logger: 'AppLogger', config: 'Config', **kwargs) -> 'AnalysisParameters':
        ...

class MaskingResult(BaseModel):
    ...

```

### `📄 core/pipelines.py`

```python
from __future__ import annotations
import math
import threading
import subprocess
import time
import re
import os
import shutil
import json
import torch
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from queue import Queue, Empty
from typing import Optional, List, Dict, Any, Generator, Callable, TYPE_CHECKING
from dataclasses import fields
from core.models import AnalysisParameters, Scene, Frame
from core.utils import handle_common_errors, estimate_totals, sanitize_filename, _to_json_safe, monitor_memory_usage, validate_video_file, safe_resource_cleanup, create_frame_map
from core.managers import VideoManager, initialize_analysis_models
from core.scene_utils import SubjectMasker, save_scene_seeds, get_scene_status_text, run_scene_detection, make_photo_thumbs
from core.filtering import load_and_prep_filter_data, apply_all_filters_vectorized
from core.database import Database
from core.events import ExtractionEvent, PreAnalysisEvent, PropagationEvent, SessionLoadEvent, ExportEvent
from core.error_handling import ErrorHandler
from core.progress import AdvancedProgressTracker
from ui.gallery_utils import build_scene_gallery_items
import gradio as gr

def _process_ffmpeg_stream(stream, tracker: Optional['AdvancedProgressTracker'], desc: str, total_duration_s: float):
    ...

def _process_ffmpeg_showinfo(stream) -> tuple[list, str]:
    ...

def run_ffmpeg_extraction(video_path: str, output_dir: Path, video_info: dict, params: 'AnalysisParameters', progress_queue: Queue, cancel_event: threading.Event, logger: 'AppLogger', config: 'Config', tracker: Optional['AdvancedProgressTracker']=None):
    ...

class Pipeline:
    def __init__(self, config: 'Config', logger: 'AppLogger', params: 'AnalysisParameters', progress_queue: Queue, cancel_event: threading.Event):
        ...

class ExtractionPipeline(Pipeline):
    def __init__(self, config: 'Config', logger: 'AppLogger', params: 'AnalysisParameters', progress_queue: Queue, cancel_event: threading.Event):
        ...
    def _run_impl(self, tracker: Optional['AdvancedProgressTracker']=None) -> dict:
        ...

class AnalysisPipeline(Pipeline):
    def __init__(self, config: 'Config', logger: 'AppLogger', params: 'AnalysisParameters', progress_queue: Queue, cancel_event: threading.Event, thumbnail_manager: 'ThumbnailManager', model_registry: 'ModelRegistry'):
        ...
    def _initialize_niqe_metric(self):
        ...
    def run_full_analysis(self, scenes_to_process: list['Scene'], tracker: Optional['AdvancedProgressTracker']=None) -> dict:
        ...
    def run_analysis_only(self, scenes_to_process: list['Scene'], tracker: Optional['AdvancedProgressTracker']=None) -> dict:
        ...
    def _filter_completed_scenes(self, scenes: list['Scene'], progress_data: dict) -> list['Scene']:
        ...
    def _save_progress(self, current_scene: 'Scene', progress_file: Path):
        ...
    def _process_reference_face(self):
        ...
    def _run_image_folder_analysis(self, tracker: Optional['AdvancedProgressTracker']=None) -> dict:
        ...
    def _run_analysis_loop(self, scenes_to_process: list['Scene'], metrics_to_compute: dict, tracker: Optional['AdvancedProgressTracker']=None):
        ...
    def _process_batch(self, batch_paths: list[Path], metrics_to_compute: dict) -> int:
        ...
    def _process_single_frame(self, thumb_path: Path, metrics_to_compute: dict):
        ...
    def _analyze_face_similarity(self, frame: 'Frame', image_rgb: np.ndarray) -> Optional[list[int]]:
        ...

@handle_common_errors
def execute_extraction(event: 'ExtractionEvent', progress_queue: Queue, cancel_event: threading.Event, logger: 'AppLogger', config: 'Config', thumbnail_manager: Optional['ThumbnailManager']=None, cuda_available: Optional[bool]=None, progress: Optional[Callable]=None, model_registry: Optional['ModelRegistry']=None) -> Generator[dict, None, None]:
    ...

@handle_common_errors
def execute_pre_analysis(event: 'PreAnalysisEvent', progress_queue: Queue, cancel_event: threading.Event, logger: 'AppLogger', config: 'Config', thumbnail_manager: 'ThumbnailManager', cuda_available: bool, progress: Optional[Callable]=None, model_registry: 'ModelRegistry'=None) -> Generator[dict, None, None]:
    ...

def validate_session_dir(path: Union[str, Path]) -> tuple[Optional[Path], Optional[str]]:
    ...

def execute_session_load(app_ui: 'AppUI', event: 'SessionLoadEvent', logger: 'AppLogger', config: 'Config', thumbnail_manager: 'ThumbnailManager', model_registry: Optional['ModelRegistry']=None) -> Generator[dict, None, None]:
    ...

def execute_propagation(event: PropagationEvent, progress_queue: Queue, cancel_event: threading.Event, logger: AppLogger, config: Config, thumbnail_manager, cuda_available, progress=None, model_registry: 'ModelRegistry'=None) -> Generator[dict, None, None]:
    ...

@handle_common_errors
def execute_analysis(event: PropagationEvent, progress_queue: Queue, cancel_event: threading.Event, logger: AppLogger, config: Config, thumbnail_manager, cuda_available, progress=None, model_registry: 'ModelRegistry'=None) -> Generator[dict, None, None]:
    ...

```

### `📄 core/progress.py`

```python
"""
Progress Tracking Infrastructure for Frame Extractor & Analyzer
"""

import threading
import time
from queue import Queue
from typing import Callable, Optional
from pydantic import BaseModel

class ProgressEvent(BaseModel):
    ...

class AdvancedProgressTracker:
    def __init__(self, progress: Callable, queue: Queue, logger: 'AppLogger', ui_stage_name: str=''):
        ...
    def start(self, total_items: int, desc: Optional[str]=None):
        ...
    def step(self, n: int=1, desc: Optional[str]=None, substage: Optional[str]=None):
        ...
    def set(self, done: int, desc: Optional[str]=None, substage: Optional[str]=None):
        ...
    def set_stage(self, stage: str, substage: Optional[str]=None):
        ...
    def done_stage(self, final_text: Optional[str]=None):
        ...
    def _overlay(self, force: bool=False):
        ...
    def _eta_seconds(self) -> Optional[float]:
        ...
    @staticmethod
    def _fmt_eta(eta_s: Optional[float]) -> str:
        ...

```

### `📄 core/sam3_patches.py`

```python
"""
SAM3 Compatibility Patches for Windows

Provides fallback implementations for SAM3 operations that require Triton,
which is not available on Windows.
"""

import cv2
import numpy as np
import torch

def edt_triton_fallback(data):
    """
    OpenCV-based fallback for Euclidean Distance Transform when Triton unavailable
    """
    ...

def connected_components_fallback(input_tensor):
    """
    CPU-based fallback for connected components when Triton unavailable
    """
    ...

def apply_patches():
    """
    Apply monkey patches to SAM3 if Triton is not available
    """
    ...

```

### `📄 core/scene_utils.py`

```python
from __future__ import annotations
import math
import threading
import json
from typing import Optional, Union, List, Any, TYPE_CHECKING
from queue import Queue
import numpy as np
import cv2
import torch
from pathlib import Path
from PIL import Image
from scenedetect import detect, ContentDetector
from core.utils import safe_resource_cleanup, create_frame_map, rgb_to_pil, postprocess_mask, render_mask_overlay, draw_bbox, _to_json_safe
from core.managers import initialize_analysis_models

def run_scene_detection(video_path: str, output_dir: Path, logger: 'AppLogger') -> list:
    ...

def make_photo_thumbs(image_paths: list[Path], out_dir: Path, params: 'AnalysisParameters', cfg: 'Config', logger: 'AppLogger', tracker: Optional['AdvancedProgressTracker']=None) -> dict:
    ...

class MaskPropagator:
    def __init__(self, params: 'AnalysisParameters', dam_tracker: 'SAM3Wrapper', cancel_event: threading.Event, progress_queue: Queue, config: 'Config', logger: 'AppLogger', device: str='cpu'):
        ...
    def propagate(self, shot_frames_rgb: list[np.ndarray], seed_idx: int, bbox_xywh: list[int], tracker: Optional['AdvancedProgressTracker']=None) -> tuple[list, list, list, list]:
        ...

class SeedSelector:
    def __init__(self, params: 'AnalysisParameters', config: 'Config', face_analyzer: 'FaceAnalysis', reference_embedding: np.ndarray, tracker: 'SAM3Wrapper', logger: 'AppLogger', device: str='cpu'):
        ...
    def _get_param(self, source: Union[dict, object], key: str, default: Any=None) -> Any:
        ...
    def select_seed(self, frame_rgb: np.ndarray, current_params: Optional[dict]=None, scene: Optional['Scene']=None) -> tuple[Optional[list], dict]:
        ...
    def _face_with_text_fallback_seed(self, frame_rgb: np.ndarray, params: Union[dict, 'AnalysisParameters'], scene: Optional['Scene']=None) -> tuple[Optional[list], dict]:
        ...
    def _identity_first_seed(self, frame_rgb: np.ndarray, params: Union[dict, 'AnalysisParameters'], scene: Optional['Scene']=None) -> tuple[Optional[list], dict]:
        ...
    def _object_first_seed(self, frame_rgb: np.ndarray, params: Union[dict, 'AnalysisParameters'], scene: Optional['Scene']=None) -> tuple[Optional[list], dict]:
        ...
    def _find_target_face(self, frame_rgb: np.ndarray) -> tuple[Optional[dict], dict]:
        ...
    def _get_person_boxes(self, frame_rgb: np.ndarray, scene: Optional['Scene']=None) -> list[dict]:
        ...
    def _get_text_prompt_boxes(self, frame_rgb: np.ndarray, params: Union[dict, 'AnalysisParameters']) -> tuple[list[dict], dict]:
        ...
    def _score_and_select_candidate(self, target_face: dict, person_boxes: list[dict], text_boxes: list[dict]) -> tuple[Optional[list], dict]:
        ...
    def _choose_person_by_strategy(self, frame_rgb: np.ndarray, params: Union[dict, 'AnalysisParameters'], scene: Optional['Scene']=None) -> tuple[list, dict]:
        ...
    def _load_image_from_array(self, image_rgb: np.ndarray) -> tuple[np.ndarray, torch.Tensor]:
        ...
    def _calculate_iou(self, box1: list, box2: list) -> float:
        ...
    def _box_contains(self, cb: list, ib: list) -> bool:
        ...
    def _expand_face_to_body(self, face_bbox: list, img_shape: tuple) -> list[int]:
        ...
    def _final_fallback_box(self, img_shape: tuple) -> list[int]:
        ...
    def _xyxy_to_xywh(self, box: list) -> list[int]:
        ...
    def _sam2_mask_for_bbox(self, frame_rgb_small: np.ndarray, bbox_xywh: list) -> Optional[np.ndarray]:
        ...

class SubjectMasker:
    def __init__(self, params: 'AnalysisParameters', progress_queue: Queue, cancel_event: threading.Event, config: 'Config', frame_map: Optional[dict]=None, face_analyzer: Optional['FaceAnalysis']=None, reference_embedding: Optional[np.ndarray]=None, thumbnail_manager: Optional['ThumbnailManager']=None, niqe_metric: Optional[Callable]=None, logger: Optional['AppLogger']=None, face_landmarker: Optional['FaceLandmarker']=None, device: str='cpu', model_registry: 'ModelRegistry'=None):
        ...
    def initialize_models(self):
        ...
    def _initialize_tracker(self) -> bool:
        ...
    def run_propagation(self, frames_dir: str, scenes_to_process: list['Scene'], tracker: Optional['AdvancedProgressTracker']=None) -> dict:
        ...
    def _load_shot_frames(self, frames_dir: str, thumb_dir: Path, start: int, end: int) -> list[tuple[int, np.ndarray, tuple[int, int]]]:
        ...
    def _select_best_frame_in_scene(self, scene: 'Scene', frames_dir: str):
        ...
    def get_seed_for_frame(self, frame_rgb: np.ndarray, seed_config: dict=None, scene: Optional['Scene']=None) -> tuple[Optional[list], dict]:
        ...
    def get_mask_for_bbox(self, frame_rgb_small: np.ndarray, bbox_xywh: list) -> Optional[np.ndarray]:
        ...
    def draw_bbox(self, img_rgb: np.ndarray, xywh: list, color: Optional[tuple]=None, thickness: Optional[int]=None, label: Optional[str]=None) -> np.ndarray:
        ...
    def _create_frame_map(self, output_dir: str):
        ...

def draw_boxes_preview(img: np.ndarray, boxes_xyxy: list[list[int]], cfg: 'Config') -> np.ndarray:
    ...

def save_scene_seeds(scenes_list: list['Scene'], output_dir_str: str, logger: 'AppLogger'):
    ...

def get_scene_status_text(scenes_list: list['Scene']) -> tuple[str, dict]:
    ...

def toggle_scene_status(scenes_list: list['Scene'], selected_shot_id: int, new_status: str, output_folder: str, logger: 'AppLogger') -> tuple[list, str, str, Any]:
    ...

def _create_analysis_context(config: 'Config', logger: 'AppLogger', thumbnail_manager: 'ThumbnailManager', cuda_available: bool, ana_ui_map_keys: list[str], ana_input_components: list, model_registry: 'ModelRegistry') -> 'SubjectMasker':
    ...

def _recompute_single_preview(scene_state: 'SceneState', masker: 'SubjectMasker', overrides: dict, thumbnail_manager: 'ThumbnailManager', logger: 'AppLogger'):
    ...

def _wire_recompute_handler(config: 'Config', logger: 'AppLogger', thumbnail_manager: 'ThumbnailManager', scenes: list['Scene'], shot_id: int, outdir: str, text_prompt: str, view: str, ana_ui_map_keys: list[str], ana_input_components: list, cuda_available: bool, model_registry: 'ModelRegistry') -> tuple:
    ...

```

### `📄 core/utils.py`

```python
from __future__ import annotations
import contextlib
import cv2
import functools
import gc
import hashlib
import json
import logging
import math
import numpy as np
import os
import re
import shutil
import traceback
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any, Callable, Optional, Union, TYPE_CHECKING
import torch
from numba import njit
from pydantic import BaseModel
from PIL import Image

def handle_common_errors(func: Callable) -> Callable:
    ...

def monitor_memory_usage(logger: 'AppLogger', device: str, threshold_mb: int=8000):
    ...

def validate_video_file(video_path: str):
    ...

def estimate_totals(params: 'AnalysisParameters', video_info: dict, scenes: Optional[list['Scene']]) -> dict:
    ...

def sanitize_filename(name: str, config: 'Config', max_length: Optional[int]=None) -> str:
    ...

def _to_json_safe(obj: Any) -> Any:
    ...

@contextlib.contextmanager
def safe_resource_cleanup(device: str='cpu'):
    ...

def is_image_folder(p: Union[str, Path]) -> bool:
    ...

def list_images(p: Union[str, Path], cfg: Config) -> list[Path]:
    ...

@njit
def compute_entropy(hist: np.ndarray, entropy_norm: float) -> float:
    ...

def _compute_sha256(path: Path) -> str:
    ...

def download_model(url: str, dest_path: Union[str, Path], description: str, logger: 'AppLogger', error_handler: 'ErrorHandler', user_agent: str, min_size: int=1000000, expected_sha256: Optional[str]=None, token: Optional[str]=None):
    ...

def postprocess_mask(mask: np.ndarray, config: 'Config', fill_holes: bool=True, keep_largest_only: bool=True) -> np.ndarray:
    ...

def render_mask_overlay(frame_rgb: np.ndarray, mask_gray: np.ndarray, alpha: float, logger: 'AppLogger') -> np.ndarray:
    ...

def rgb_to_pil(image_rgb: np.ndarray) -> Image.Image:
    ...

def create_frame_map(output_dir: Path, logger: 'AppLogger', ext: str='.webp') -> dict:
    ...

def draw_bbox(img_rgb: np.ndarray, xywh: list, config: 'Config', color: Optional[tuple]=None, thickness: Optional[int]=None, label: Optional[str]=None) -> np.ndarray:
    ...

```

### `📄 generate_streamlined.py`

```python
import ast
import os
from pathlib import Path

HEADER = '---\nVersion: 2.0\nLast Updated: 2025-12-16\nPython: 3.10+\nGradio: 6.x\nSAM3: Via submodule\n---\n\n# Developer Guidelines & Agent Memory\n\n**⚠️ CRITICAL**: Read this before starting any task.\n\n🔴 CRITICAL | 🟡 WARNING | 🟢 BEST PRACTICE\n'
QUICK_START = '\n## 1. Quick Start Guide\n\n### 5-Minute Setup\n1. **Clone & Submodules**: `git submodule update --init --recursive`\n2. **Environment**: `python3 -m venv venv && source venv/bin/activate`\n3. **Dependencies**: `pip install -r requirements.txt` (Installs SAM3 via submodule)\n4. **Run App**: `python app.py`\n\n### Essential Commands\n- **Test Backend**: `python -m pytest tests/`\n- **Test E2E**: `pytest tests/e2e/` (Requires Playwright)\n- **Lint/Check**: `.claude/commands/validate.md` (if available)\n\n### Directory Structure\n- `app.py`: Entry point.\n- `core/`: Business logic (pipelines, config, db).\n- `ui/`: Gradio interface components.\n- `tests/`: Unit and E2E tests.\n- `SAM3_repo/`: **Read-only** submodule.\n'
CRITICAL_RULES = "\n## 2. Critical Rules\n\n### 🔴 CRITICAL (Must Follow)\n- **NEVER** edit files in `SAM3_repo` or `Grounded-SAM-2`. Treat as external libraries.\n- **ALWAYS** match Gradio event handler return values count to the `outputs` list. Mismatches crash the app silently.\n- **NEVER** use `@lru_cache` on functions taking the `Config` object (it's unhashable). Use `model_registry.get_or_load`.\n- **ALWAYS** use `pathlib.Path`, never `os.path`.\n- **ALWAYS** mock external dependencies (SAM3, Torch) in unit tests.\n\n### 🟡 WARNING (Potential Bugs)\n- **Check Masks**: Verify masks exist on disk before export/processing.\n- **Thread Safety**: MediaPipe objects are not thread-safe. Use thread-local storage or one instance per thread.\n- **Gradio State**: Do not store locks or file handles in `gr.State`.\n\n### 🟢 BEST PRACTICE\n- **Refactoring**: Move logic from `app.py` to `core/`.\n- **Typing**: Use Pydantic models (`core/events.py`) instead of untyped dicts.\n"
ARCHITECTURE = '\n## 3. Architecture Overview\n\n### Data Flow\n`UI (Gradio)` → `Event Object (Pydantic)` → `Pipeline (Core)` → `Database/Files`\n\n### Component Relationship\n```\n[app.py] (UI Assembly)\n   │\n   ├─ [core/config.py] (Settings)\n   ├─ [core/managers.py] (ModelRegistry, ThumbnailManager)\n   └─ [core/pipelines.py] (Logic)\n         │\n         ├─ ExtractionPipeline (FFmpeg)\n         ├─ AnalysisPipeline (SAM3, InsightFace)\n         └─ ExportPipeline (Filtering, Rendering)\n```\n\n### State Management\n- **Session State**: `gr.State` stores mutable data (scene lists, paths).\n- **Global State**: `ModelRegistry` (Singleton-like) manages heavy models.\n- **Persistence**: `metadata.db` (SQLite) for frame data; `json` for configs.\n'
DEVELOPMENT_WORKFLOWS = '\n## 5. Development Workflows\n\n### Bug Fix Workflow\n1. **Reproduce**: Create a test case in `tests/test_reproduce_issue.py`.\n2. **Log**: Use `logger.debug()` to trace execution.\n3. **Fix**: Implement fix in `core/` or `ui/`.\n4. **Verify**: Run `python -m pytest tests/`.\n5. **Clean**: Remove temporary test files.\n\n### Adding a New Metric\n1. **Config**: Add default thresholds to `Config` in `core/config.py`.\n2. **Extraction**: Update `_extract_metric_arrays()` in `core/filtering.py`.\n3. **UI**: Add slider in `ui/app_ui.py` inside `_create_filtering_tab`.\n4. **Analysis**: Update `calculate_quality_metrics` in `core/models.py`.\n'
TESTING_GUIDE = '\n## 6. Testing & Mocking Guide\n\n### When to Mock\n- **File I/O**: Patch `pathlib.Path.exists`, `open`.\n- **ML Models**: Always mock `SAM3Wrapper`, `FaceAnalysis`, `FaceLandmarker`.\n- **Submodules**: Mock `sam3` package to avoid import errors.\n\n### Common Patterns\n```python\n# Mocking a class method\n@patch("core.managers.ModelRegistry.get_tracker")\ndef test_tracker(mock_get, app_ui):\n    mock_get.return_value = MagicMock()\n    ...\n```\n\n### E2E vs Unit\n- **Unit**: Fast, mocks everything. Run pre-commit.\n- **E2E**: Slower, uses `mock_app.py` to simulate backend. Checks UI flows.\n'
CONFIG_REF = '\n## 7. Configuration Reference\n\nSee `core/config.py` for full list.\n\n| Category | Key Fields | Default |\n|----------|------------|---------|\n| **Paths** | `logs_dir`, `models_dir`, `downloads_dir` | `logs`, `models`, `downloads` |\n| **Models** | `face_model_name`, `tracker_model_name` | `buffalo_l`, `sam3` |\n| **perf** | `analysis_default_workers` | 4 |\n| **UI** | `default_thumb_megapixels` | 0.5 |\n'
TROUBLESHOOTING = '\n## 8. Troubleshooting\n\n### Error: "CUDA out of memory"\n- **Where**: SAM3 initialization, NIQE metric.\n- **Fix**: Set `model_registry.runtime_device_override = \'cpu\'`.\n- **Prevention**: Call `cleanup_models()` between sessions.\n\n### Error: "ModuleNotFoundError: sam3"\n- **Cause**: Submodule not initialized.\n- **Fix**: `git submodule update --init --recursive`.\n- **Check**: Verify `SAM3_repo/` exists and has files.\n\n### Error: "ValueError: ... is not in list" (Gradio)\n- **Cause**: `gr.Radio` or `gr.Dropdown` value updated to something not in `choices`.\n- **Fix**: Update `choices` list *before* setting `value`.\n'
PERFORMANCE = '\n## 9. Performance & Memory\n\n- **SAM3**: Requires ~8GB VRAM. Falls back to CPU (slow).\n- **Thumbnails**: Cached in RAM (`ThumbnailManager`). LRU eviction.\n- **Batch Processing**: Uses `ThreadPoolExecutor`. Limit workers in Config if OOM occurs.\n'
API_REF = '\n## 10. API Quick Reference\n\n### Key Functions\n- `execute_extraction(event: ExtractionEvent) -> Generator`\n- `execute_pre_analysis(event: PreAnalysisEvent) -> Generator`\n- `execute_propagation(event: PropagationEvent) -> Generator`\n\n### Event Models\n- `ExtractionEvent`: Source path, method (interval/scene).\n- `PreAnalysisEvent`: Analysis params, seed strategy.\n- `PropagationEvent`: List of scenes to process.\n'
GIT_DEPLOY = '\n## 11. Git & Deployment\n\n- **Submodules**: Always update recursive.\n- **Requirements**: `requirements.txt` is root.\n- **Validation**: Verify model downloads with SHA256.\n'
def generate_skeleton_section(root_dir):
    ...

def parse_file_to_skeleton(file_path):
    ...

def process_node(node, indent=0):
    ...

def main():
    ...

```

### `📄 tests/e2e/test_app_flow.py`

```python
import pytest
from playwright.sync_api import Page, expect
import subprocess
import time
import os
import signal
import sys

PORT = 7860
BASE_URL = f'http://127.0.0.1:{PORT}'
@pytest.fixture(scope='module')
def app_server():
    """
    Starts the mock app server before tests and kills it after.
    """
    ...

def test_full_user_flow(page: Page, app_server):
    """
    Tests the complete end-to-end workflow:
    Extraction -> Pre-Analysis -> Scene Selection -> Propagation -> Analysis -> Export
    """
    ...

```

### `📄 tests/mock_app.py`

```python
import sys
import os
import threading
import time
from unittest.mock import MagicMock, patch
import app
from app import Config, AppLogger, ThumbnailManager
import core.pipelines
import core.utils
import core.managers
from core.models import Scene

mock_torch = MagicMock(name='torch')
mock_torch.cuda.is_available.return_value = False
mock_torch.__version__ = '2.0.0'
mock_torch.nn.Module = MagicMock
mock_torch.Tensor = MagicMock
mock_sam3 = MagicMock(name='sam3')
mock_sam3.model_builder = MagicMock()
modules_to_mock = {'torch': mock_torch, 'torchvision': MagicMock(), 'torchvision.ops': MagicMock(), 'torchvision.transforms': MagicMock(), 'insightface': MagicMock(), 'insightface.app': MagicMock(), 'sam3': mock_sam3, 'sam3.model_builder': mock_sam3.model_builder, 'sam3.model.sam3_video_predictor': MagicMock(), 'mediapipe': MagicMock(), 'mediapipe.tasks': MagicMock(), 'mediapipe.tasks.python': MagicMock(), 'mediapipe.tasks.python.vision': MagicMock(), 'pyiqa': MagicMock(), 'scenedetect': MagicMock(), 'yt_dlp': MagicMock(), 'ultralytics': MagicMock(), 'groundingdino': MagicMock(), 'numba': MagicMock(), 'lpips': MagicMock()}
def mock_extraction_run(self, tracker=None):
    """
    Mocks the extraction process.
    """
    ...

def mock_pre_analysis_execution(event, progress_queue, cancel_event, logger, config, thumbnail_manager, cuda_available, progress=None, model_registry=None):
    """
    Mocks execute_pre_analysis generator.
    """
    ...

def mock_propagation_execution(event, progress_queue, cancel_event, logger, config, thumbnail_manager, cuda_available, progress=None, model_registry=None):
    ...

def mock_analysis_execution(event, progress_queue, cancel_event, logger, config, thumbnail_manager, cuda_available, progress=None, model_registry=None):
    ...

core.pipelines.ExtractionPipeline._run_impl = mock_extraction_run
core.pipelines.execute_pre_analysis = mock_pre_analysis_execution
core.pipelines.execute_propagation = mock_propagation_execution
core.pipelines.execute_analysis = mock_analysis_execution
core.utils.download_model = MagicMock()
core.managers.download_model = MagicMock()
```

### `📄 tests/test_batch_manager.py`

```python
import time
import pytest
from core.batch_manager import BatchManager, BatchStatus, BatchItem

def test_batch_manager_add():
    ...

def test_batch_manager_processing():
    ...

def test_batch_manager_failure():
    ...

```

### `📄 tests/test_core.py`

```python
import pytest
from pydantic import ValidationError
import sys
import unittest
from unittest.mock import MagicMock, patch, mock_open
from pathlib import Path
import os
import json
import time
import numpy as np
import gradio as gr
import cv2
import datetime
from collections import deque
import pydantic
from core.config import Config
from core.database import Database
from core.logger import AppLogger
from core.models import Scene, Frame, QualityConfig, _coerce
from core.filtering import apply_all_filters_vectorized
from ui.gallery_utils import auto_set_thresholds
from core.events import PreAnalysisEvent

mock_torch = MagicMock(name='torch')
mock_torch.__version__ = '2.0.0'
mock_torch.__path__ = ['fake']
mock_torch.__spec__ = MagicMock()
mock_torch.hub = MagicMock(name='torch.hub')
mock_torch.cuda = MagicMock(name='torch.cuda')
mock_torch.cuda.is_available.return_value = False
mock_torch.distributed = MagicMock(name='torch.distributed')
mock_torch.multiprocessing = MagicMock(name='torch.multiprocessing')
mock_torch.amp = MagicMock(name='torch.amp')
mock_torch_autograd = MagicMock(name='torch.autograd')
mock_torch_autograd.Variable = MagicMock(name='torch.autograd.Variable')
mock_torch_nn = MagicMock(name='torch.nn')
mock_torch_nn.__path__ = ['fake']
class MockNNModule:
    def __init__(self, *args, **kwargs):
        ...
    def __call__(self, *args, **kwargs):
        ...

mock_torch_nn.Module = MockNNModule
mock_torch_nn.attention = MagicMock(name='torch.nn.attention')
mock_torch_nn_init = MagicMock(name='torch.nn.init')
mock_torch_nn_functional = MagicMock(name='torch.nn.functional')
mock_torch_optim = MagicMock(name='torch.optim')
mock_torch_utils = MagicMock(name='torch.utils')
mock_torch_utils.__path__ = ['fake']
mock_torch_utils_data = MagicMock(name='torch.utils.data')
mock_torch_utils_checkpoint = MagicMock(name='torch.utils.checkpoint')
mock_torch_utils_pytree = MagicMock(name='torch.utils._pytree')
mock_torchvision = MagicMock(name='torchvision')
mock_torchvision.ops = MagicMock(name='torchvision.ops')
mock_torchvision.ops.roi_align = MagicMock(name='torchvision.ops.roi_align')
mock_torchvision.ops.misc = MagicMock(name='torchvision.ops.misc')
mock_torchvision.datasets = MagicMock(name='torchvision.datasets')
mock_torchvision.datasets.vision = MagicMock(name='torchvision.datasets.vision')
mock_torchvision.transforms = MagicMock(name='torchvision.transforms')
mock_torchvision.transforms.functional = MagicMock(name='torchvision.transforms.functional')
mock_torchvision.utils = MagicMock(name='torchvision.utils')
mock_insightface = MagicMock(name='insightface')
mock_insightface.app = MagicMock(name='insightface.app')
mock_timm = MagicMock(name='timm')
mock_timm.models = MagicMock(name='timm.models')
mock_timm.models.layers = MagicMock(name='timm.models.layers')
mock_pycocotools = MagicMock(name='pycocotools')
mock_pycocotools.mask = MagicMock(name='pycocotools.mask')
mock_psutil = MagicMock(name='psutil')
mock_psutil.cpu_percent.return_value = 50.0
mock_psutil.virtual_memory.return_value = MagicMock(percent=50.0, available=1024 * 1024 * 1024)
mock_psutil.disk_usage.return_value = MagicMock(percent=50.0)
mock_process = mock_psutil.Process.return_value
mock_process.memory_info.return_value.rss = 100 * 1024 * 1024
mock_process.cpu_percent.return_value = 10.0
mock_matplotlib = MagicMock(name='matplotlib')
mock_matplotlib.__path__ = ['fake']
mock_matplotlib.ticker = MagicMock(name='matplotlib.ticker')
mock_matplotlib.figure = MagicMock(name='matplotlib.figure')
mock_matplotlib.backends = MagicMock(name='matplotlib.backends')
mock_matplotlib.backends.backend_agg = MagicMock(name='matplotlib.backends.backend_agg')
modules_to_mock = {'torch': mock_torch, 'torch.hub': mock_torch.hub, 'torch.distributed': mock_torch.distributed, 'torch.multiprocessing': mock_torch.multiprocessing, 'torch.autograd': mock_torch_autograd, 'torch.nn': mock_torch_nn, 'torch.nn.attention': mock_torch_nn.attention, 'torch.nn.init': mock_torch_nn_init, 'torch.nn.functional': mock_torch_nn_functional, 'torch.optim': mock_torch_optim, 'torch.utils': mock_torch_utils, 'torch.utils.data': mock_torch_utils_data, 'torch.utils.checkpoint': mock_torch_utils_checkpoint, 'torch.utils._pytree': mock_torch_utils_pytree, 'torchvision': mock_torchvision, 'torchvision.ops': mock_torchvision.ops, 'torchvision.ops.roi_align': mock_torchvision.ops.roi_align, 'torchvision.ops.misc': mock_torchvision.ops.misc, 'torchvision.datasets': mock_torchvision.datasets, 'torchvision.datasets.vision': mock_torchvision.datasets.vision, 'torchvision.transforms': mock_torchvision.transforms, 'torchvision.transforms.functional': mock_torchvision.transforms.functional, 'torchvision.utils': mock_torchvision.utils, 'insightface': mock_insightface, 'insightface.app': mock_insightface.app, 'timm': mock_timm, 'timm.models': mock_timm.models, 'timm.models.layers': mock_timm.models.layers, 'onnxruntime': MagicMock(name='onnxruntime'), 'DAM4SAM': MagicMock(name='DAM4SAM'), 'DAM4SAM.utils': MagicMock(name='DAM4SAM.utils'), 'DAM4SAM.dam4sam_tracker': MagicMock(name='DAM4SAM.dam4sam_tracker'), 'GPUtil': MagicMock(getGPUs=lambda: [MagicMock(memoryUtil=0.5)]), 'pycocotools': mock_pycocotools, 'pycocotools.mask': mock_pycocotools.mask, 'psutil': mock_psutil, 'matplotlib': mock_matplotlib, 'matplotlib.ticker': mock_matplotlib.ticker, 'matplotlib.figure': mock_matplotlib.figure, 'matplotlib.backends': mock_matplotlib.backends, 'matplotlib.backends.backend_agg': mock_matplotlib.backends.backend_agg, 'matplotlib.pyplot': MagicMock(), 'scenedetect': MagicMock(), 'yt_dlp': MagicMock(), 'pyiqa': MagicMock(name='pyiqa'), 'mediapipe': MagicMock(), 'mediapipe.tasks': MagicMock(), 'mediapipe.tasks.python': MagicMock(), 'mediapipe.tasks.python.vision': MagicMock(), 'lpips': MagicMock(name='lpips'), 'numba': MagicMock(name='numba'), 'skimage': MagicMock(name='skimage'), 'skimage.metrics': MagicMock(name='skimage.metrics')}
mock_pydantic_settings = MagicMock(name='pydantic_settings')
mock_pydantic_settings.BaseSettings = pydantic.BaseModel
mock_pydantic_settings.SettingsConfigDict = dict
modules_to_mock['pydantic_settings'] = mock_pydantic_settings
@pytest.fixture
def mock_ui_state():
    """
    Provides a dictionary with default values for UI-related event models.
    """
    ...

@pytest.fixture
def sample_frames_data():
    ...

@pytest.fixture
def sample_scenes():
    ...

class TestUtils:
    @pytest.mark.parametrize('value, to_type, expected', [('True', bool, True), ('false', bool, False), ('1', bool, True), ('0', bool, False), ('yes', bool, True), ('no', bool, False), (True, bool, True), (False, bool, False), ('123', int, 123), (123, int, 123), ('123.45', float, 123.45), (123.45, float, 123.45), ('string', str, 'string')])
    def test_coerce(self, value, to_type, expected):
        ...
    def test_coerce_invalid_raises(self):
        ...
    def test_config_init(self):
        ...
    @patch('pathlib.Path.mkdir', MagicMock())
    @patch('os.access', return_value=True)
    def test_validation_error(self, mock_access):
        """
        Test that a validation error is raised for invalid config.
        """
        ...

class TestAppLogger:
    def test_app_logger_instantiation(self):
        """
        Tests that the logger can be instantiated with a valid config.
        """
        ...
    def test_auto_set_thresholds(self):
        ...
    def test_apply_all_filters_with_face_and_mask(self, sample_frames_data):
        """
        Verify filtering by face similarity and mask area.
        """
        ...
    def test_calculate_quality_metrics_with_niqe(self):
        """
        Test quality metrics calculation including NIQE.
        """
        ...

class TestPreAnalysisEvent:
    def test_face_ref_validation(self, tmp_path, mock_ui_state):
        """
        Test the custom validator for face_ref_img_path.
        """
        ...

```

### `📄 tests/test_dedup.py`

```python
import pytest
import numpy as np
import imagehash
import sys
import os
from unittest.mock import MagicMock, patch
from core.filtering import _apply_deduplication_filter, _run_batched_lpips
from core.config import Config
from core.managers import ThumbnailManager

modules_to_mock = {'sam3': MagicMock(), 'sam3.model_builder': MagicMock(), 'sam3.model.sam3_video_predictor': MagicMock(), 'mediapipe': MagicMock(), 'mediapipe.tasks': MagicMock(), 'mediapipe.tasks.python': MagicMock(), 'mediapipe.tasks.python.vision': MagicMock(), 'pyiqa': MagicMock(), 'scenedetect': MagicMock(), 'lpips': MagicMock(), 'yt_dlp': MagicMock(), 'numba': MagicMock(), 'matplotlib': MagicMock(), 'matplotlib.pyplot': MagicMock(), 'matplotlib.ticker': MagicMock(), 'torch': MagicMock(), 'torchvision': MagicMock(), 'torchvision.ops': MagicMock(), 'torchvision.transforms': MagicMock(), 'insightface': MagicMock(), 'insightface.app': MagicMock()}
patcher = patch.dict(sys.modules, modules_to_mock)
@pytest.fixture
def mock_thumbnail_manager():
    ...

@pytest.fixture
def sample_frames_for_dedup():
    ...

def test_dedup_phash_replacement(sample_frames_for_dedup, mock_thumbnail_manager):
    ...

def test_dedup_phash_no_replacement(sample_frames_for_dedup, mock_thumbnail_manager):
    ...

def test_dedup_disabled(sample_frames_for_dedup, mock_thumbnail_manager):
    ...

def test_dedup_threshold(sample_frames_for_dedup, mock_thumbnail_manager):
    ...

def test_run_batched_lpips(mock_thumbnail_manager):
    ...

```

### `📄 tests/test_export.py`

```python
import pytest
from unittest.mock import MagicMock, patch
import sys
import os
from pathlib import Path
import json
from core.events import ExportEvent
from core.export import export_kept_frames

@pytest.fixture
def mock_config():
    ...

@pytest.fixture
def mock_logger():
    ...

@patch('subprocess.Popen')
@patch('core.export.apply_all_filters_vectorized')
def test_export_kept_frames(mock_filter, mock_popen, mock_config, mock_logger, tmp_path):
    ...

```

### `📄 tests/test_pipelines.py`

```python
import pytest
from unittest.mock import MagicMock, patch, ANY, mock_open
import sys
import os
from pathlib import Path
from queue import Queue
import threading
import json
import numpy as np
from core.config import Config
from core.models import AnalysisParameters, Scene
from core.pipelines import ExtractionPipeline, AnalysisPipeline, run_ffmpeg_extraction

@pytest.fixture
def mock_config(tmp_path):
    ...

@pytest.fixture
def mock_logger():
    ...

@pytest.fixture
def mock_params():
    ...

@pytest.fixture
def mock_progress_queue():
    ...

@pytest.fixture
def mock_cancel_event():
    ...

@pytest.fixture
def mock_thumbnail_manager():
    ...

@pytest.fixture
def mock_model_registry():
    ...

class TestExtractionPipeline:
    @patch('core.pipelines.run_ffmpeg_extraction')
    @patch('core.pipelines.VideoManager')
    def test_extraction_video_success(self, mock_vm_cls, mock_ffmpeg, mock_config, mock_logger, mock_params, mock_progress_queue, mock_cancel_event):
        ...
    @patch('core.pipelines.run_ffmpeg_extraction')
    @patch('core.pipelines.VideoManager')
    def test_extraction_video_cancel(self, mock_vm_cls, mock_ffmpeg, mock_config, mock_logger, mock_params, mock_progress_queue, mock_cancel_event):
        ...
    @patch('core.utils.is_image_folder', return_value=True)
    @patch('core.utils.list_images')
    @patch('core.pipelines.make_photo_thumbs')
    def test_extraction_folder(self, mock_thumbs, mock_list_imgs, mock_is_folder, mock_config, mock_logger, mock_params, mock_progress_queue, mock_cancel_event):
        ...
    @patch('subprocess.Popen')
    def test_run_ffmpeg_extraction(self, mock_popen, mock_config, mock_logger, mock_params, mock_progress_queue, mock_cancel_event, tmp_path):
        ...

class TestAnalysisPipeline:
    @patch('core.pipelines.SubjectMasker')
    @patch('core.pipelines.initialize_analysis_models')
    @patch('core.pipelines.Database')
    def test_run_full_analysis_success(self, mock_db_cls, mock_init_models, mock_masker_cls, mock_config, mock_logger, mock_params, mock_progress_queue, mock_cancel_event, mock_thumbnail_manager, mock_model_registry):
        ...
    @patch('core.pipelines.SubjectMasker')
    @patch('core.pipelines.initialize_analysis_models')
    def test_run_full_analysis_cancel(self, mock_init_models, mock_masker_cls, mock_config, mock_logger, mock_params, mock_progress_queue, mock_cancel_event, mock_thumbnail_manager, mock_model_registry):
        ...
    @patch('core.pipelines.initialize_analysis_models')
    @patch('core.pipelines.Database')
    def test_run_analysis_only(self, mock_db_cls, mock_init_models, mock_config, mock_logger, mock_params, mock_progress_queue, mock_cancel_event, mock_thumbnail_manager, mock_model_registry):
        ...

```

### `📄 tests/test_scene_utils.py`

```python
import pytest
from unittest.mock import MagicMock, patch, ANY, mock_open
import sys
import os
import numpy as np
import torch
from pathlib import Path
from queue import Queue
import threading
import json
from core.config import Config
from core.models import AnalysisParameters, Scene
from core.scene_utils import SeedSelector, MaskPropagator, SubjectMasker, run_scene_detection

@pytest.fixture
def mock_config(tmp_path):
    ...

@pytest.fixture
def mock_logger():
    ...

@pytest.fixture
def mock_params():
    ...

class TestSeedSelector:
    def test_select_seed_largest_person(self, mock_config, mock_logger, mock_params):
        ...
    def test_select_seed_text_prompt(self, mock_config, mock_logger, mock_params):
        ...

class TestMaskPropagator:
    @patch('core.scene_utils.postprocess_mask', side_effect=lambda x, **k: x)
    def test_propagate_success(self, mock_post, mock_config, mock_logger, mock_params):
        ...

class TestSubjectMasker:
    @patch('core.scene_utils.create_frame_map', return_value={0: 'frame_0.png'})
    def test_run_propagation(self, mock_create_map, mock_config, mock_logger, mock_params, tmp_path):
        ...

```

### `📄 tests/test_ui.py`

```python
import asyncio
from playwright.async_api import async_playwright

```

### `📄 ui/app_ui.py`

```python
from __future__ import annotations
import threading
import time
import sys
import re
from pathlib import Path
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, List, Dict, Any, Callable, Deque, Generator
from collections import deque
import gradio as gr
import torch
import numpy as np
import cv2
import uuid
import shutil
from core.config import Config
from core.logger import AppLogger
from core.managers import ThumbnailManager, ModelRegistry
from core.models import Scene, SceneState, AnalysisParameters
from core.utils import is_image_folder
from core.scene_utils import toggle_scene_status, save_scene_seeds, _recompute_single_preview, _create_analysis_context, _wire_recompute_handler, get_scene_status_text
from core.pipelines import execute_extraction, execute_pre_analysis, execute_propagation, execute_analysis, execute_session_load, AdvancedProgressTracker
from core.export import export_kept_frames, dry_run_export
from ui.gallery_utils import build_scene_gallery_items, on_filters_changed, auto_set_thresholds, _update_gallery, scene_caption, create_scene_thumbnail_with_badge
from core.events import ExtractionEvent, PreAnalysisEvent, PropagationEvent, SessionLoadEvent, FilterEvent, ExportEvent
from core.batch_manager import BatchManager, BatchStatus, BatchItem

class AppUI:
    def __init__(self, config: 'Config', logger: 'AppLogger', progress_queue: Queue, cancel_event: threading.Event, thumbnail_manager: 'ThumbnailManager', model_registry: 'ModelRegistry'):
        ...
    def preload_models(self):
        """
        Asynchronously preloads heavy models.
        """
        ...
    def _get_stepper_html(self, current_step: int=0) -> str:
        ...
    def build_ui(self) -> gr.Blocks:
        ...
    def _get_comp(self, name: str) -> Optional[gr.components.Component]:
        ...
    def _reg(self, key: str, component: gr.components.Component) -> gr.components.Component:
        ...
    def _create_component(self, name: str, comp_type: str, kwargs: dict) -> gr.components.Component:
        ...
    def _build_header(self):
        ...
    def _build_main_tabs(self):
        ...
    def _build_footer(self):
        ...
    def _create_extraction_tab(self):
        ...
    def _create_define_subject_tab(self):
        ...
    def _create_scene_selection_tab(self):
        ...
    def _create_metrics_tab(self):
        ...
    def _create_filtering_tab(self):
        ...
    def get_all_filter_keys(self) -> list[str]:
        ...
    def get_metric_description(self, metric_name: str) -> str:
        ...
    def _create_event_handlers(self):
        ...
    def update_stepper(self, evt: gr.SelectData):
        ...
    def _push_history(self, scenes: List[Dict], history: Deque) -> Deque:
        ...
    def _undo_last_action(self, scenes: List[Dict], history: Deque, output_dir: str, view: str) -> tuple:
        ...
    def _run_task_with_progress(self, task_func: Callable, output_components: list, progress: Callable, *args) -> Generator[dict, None, None]:
        ...
    def on_select_yolo_subject_wrapper(self, subject_id: str, scenes: list, shot_id: int, outdir: str, view: str, history: Deque, *ana_args) -> tuple:
        """
        Wrapper for handling subject selection from the YOLO radio buttons (now Gallery).
        """
        ...
    def _setup_bulk_scene_handlers(self):
        ...
    def on_reset_scene_wrapper(self, scenes, shot_id, outdir, view, history, *ana_args):
        ...
    def on_select_for_edit(self, scenes, view, indexmap, outputdir, yoloresultsstate, event: Optional[gr.EventData]=None):
        ...
    def on_editor_toggle(self, scenes, selected_shotid, outputfolder, view, new_status, history):
        ...
    def _toggle_pause(self, tracker: 'AdvancedProgressTracker') -> str:
        ...
    def run_system_diagnostics(self) -> Generator[str, None, None]:
        """
        Runs a comprehensive suite of system checks and a dry run.
        """
        ...
    def _create_pre_analysis_event(self, *args: Any) -> 'PreAnalysisEvent':
        ...
    def _run_pipeline(self, pipeline_func: Callable, event: Any, progress: Callable, success_callback: Optional[Callable]=None, *args):
        ...
    def run_extraction_wrapper(self, *args):
        ...
    def add_to_queue_handler(self, *args):
        ...
    def clear_queue_handler(self):
        ...
    def _batch_processor(self, item: BatchItem, progress_callback: Callable):
        ...
    def start_batch_wrapper(self, workers: float):
        ...
    def stop_batch_handler(self):
        ...
    def _on_extraction_success(self, result: dict) -> dict:
        ...
    def _on_pre_analysis_success(self, result: dict) -> dict:
        ...
    def run_pre_analysis_wrapper(self, *args):
        ...
    def run_propagation_wrapper(self, scenes, *args):
        ...
    def _on_propagation_success(self, result: dict) -> dict:
        ...
    def run_analysis_wrapper(self, scenes, *args):
        ...
    def _on_analysis_success(self, result: dict) -> dict:
        ...
    def run_session_load_wrapper(self, session_path: str):
        ...
    def _fix_strategy_visibility(self, strategy: str) -> dict:
        ...
    def _setup_visibility_toggles(self):
        ...
    def get_inputs(self, keys: list[str]) -> list[gr.components.Component]:
        ...
    def _setup_pipeline_handlers(self):
        ...
    def on_identity_confidence_change(self, confidence: float, all_faces: list) -> gr.update:
        ...
    def on_discovered_face_select(self, all_faces: list, confidence: float, *args, evt: gr.EventData=None) -> tuple[str, Optional[np.ndarray]]:
        ...
    def on_find_people_from_video(self, *args) -> tuple[gr.update, list, float, list]:
        ...
    def on_apply_bulk_scene_filters_extended(self, scenes: list, min_mask_area: float, min_face_sim: float, min_confidence: float, enable_face_filter: bool, output_folder: str, view: str) -> tuple:
        ...
    def _get_smart_mode_updates(self, is_enabled: bool) -> list[gr.update]:
        ...
    def _setup_filtering_handlers(self):
        ...
    def on_preset_changed(self, preset_name: str) -> list[Any]:
        ...
    def on_filters_changed_wrapper(self, all_frames_data: list, per_metric_values: dict, output_dir: str, gallery_view: str, show_overlay: bool, overlay_alpha: float, require_face_match: bool, dedup_thresh: int, dedup_method_ui: str, smart_mode_enabled: bool, *slider_values: float) -> tuple[str, gr.update]:
        ...
    def calculate_visual_diff(self, gallery: gr.Gallery, all_frames_data: list, dedup_method_ui: str, dedup_thresh: int, ssim_thresh: float, lpips_thresh: float) -> Optional[np.ndarray]:
        ...
    def on_reset_filters(self, all_frames_data: list, per_metric_values: dict, output_dir: str) -> tuple:
        ...
    def on_auto_set_thresholds(self, per_metric_values: dict, p: int, *checkbox_values: bool) -> list[gr.update]:
        ...
    def export_kept_frames_wrapper(self, all_frames_data: list, output_dir: str, video_path: str, enable_crop: bool, crop_ars: str, crop_padding: int, require_face_match: bool, dedup_thresh: int, dedup_method_ui: str, *slider_values: float) -> str:
        ...
    def dry_run_export_wrapper(self, all_frames_data: list, output_dir: str, video_path: str, enable_crop: bool, crop_ars: str, crop_padding: int, require_face_match: bool, dedup_thresh: int, dedup_method_ui: str, *slider_values: float) -> str:
        ...

```

### `📄 ui/gallery_utils.py`

```python
from __future__ import annotations
import math
import cv2
import numpy as np
import json
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any
import gradio as gr
from collections import Counter
from core.models import Scene
from core.filtering import apply_all_filters_vectorized
from core.utils import render_mask_overlay
from core.events import FilterEvent

def scene_matches_view(scene: Scene, view: str) -> bool:
    ...

def create_scene_thumbnail_with_badge(thumb_img: np.ndarray, scene_idx: int, is_excluded: bool) -> np.ndarray:
    ...

def scene_caption(s: Scene) -> str:
    ...

def build_scene_gallery_items(scenes: list[Union[dict, Scene]], view: str, output_dir: str, page_num: int=1, page_size: int=20) -> tuple[list[tuple], list[int], int]:
    ...

def _update_gallery(all_frames_data: list[dict], filters: dict, output_dir: str, gallery_view: str, show_overlay: bool, overlay_alpha: float, thumbnail_manager: Any, config: Any, logger: Any) -> tuple[str, gr.update]:
    ...

def on_filters_changed(event: FilterEvent, thumbnail_manager: Any, config: Any, logger: Any) -> dict:
    ...

def auto_set_thresholds(per_metric_values: dict, p: int, slider_keys: list[str], selected_metrics: list[str]) -> dict:
    ...

```


## 5. Development Workflows

### Bug Fix Workflow
1. **Reproduce**: Create a test case in `tests/test_reproduce_issue.py`.
2. **Log**: Use `logger.debug()` to trace execution.
3. **Fix**: Implement fix in `core/` or `ui/`.
4. **Verify**: Run `python -m pytest tests/`.
5. **Clean**: Remove temporary test files.

### Adding a New Metric
1. **Config**: Add default thresholds to `Config` in `core/config.py`.
2. **Extraction**: Update `_extract_metric_arrays()` in `core/filtering.py`.
3. **UI**: Add slider in `ui/app_ui.py` inside `_create_filtering_tab`.
4. **Analysis**: Update `calculate_quality_metrics` in `core/models.py`.


## 6. Testing & Mocking Guide

### When to Mock
- **File I/O**: Patch `pathlib.Path.exists`, `open`.
- **ML Models**: Always mock `SAM3Wrapper`, `FaceAnalysis`, `FaceLandmarker`.
- **Submodules**: Mock `sam3` package to avoid import errors.

### Common Patterns
```python
# Mocking a class method
@patch("core.managers.ModelRegistry.get_tracker")
def test_tracker(mock_get, app_ui):
    mock_get.return_value = MagicMock()
    ...
```

### E2E vs Unit
- **Unit**: Fast, mocks everything. Run pre-commit.
- **E2E**: Slower, uses `mock_app.py` to simulate backend. Checks UI flows.


## 7. Configuration Reference

See `core/config.py` for full list.

| Category | Key Fields | Default |
|----------|------------|---------|
| **Paths** | `logs_dir`, `models_dir`, `downloads_dir` | `logs`, `models`, `downloads` |
| **Models** | `face_model_name`, `tracker_model_name` | `buffalo_l`, `sam3` |
| **perf** | `analysis_default_workers` | 4 |
| **UI** | `default_thumb_megapixels` | 0.5 |


## 8. Troubleshooting

### Error: "CUDA out of memory"
- **Where**: SAM3 initialization, NIQE metric.
- **Fix**: Set `model_registry.runtime_device_override = 'cpu'`.
- **Prevention**: Call `cleanup_models()` between sessions.

### Error: "ModuleNotFoundError: sam3"
- **Cause**: Submodule not initialized.
- **Fix**: `git submodule update --init --recursive`.
- **Check**: Verify `SAM3_repo/` exists and has files.

### Error: "ValueError: ... is not in list" (Gradio)
- **Cause**: `gr.Radio` or `gr.Dropdown` value updated to something not in `choices`.
- **Fix**: Update `choices` list *before* setting `value`.


## 9. Performance & Memory

- **SAM3**: Requires ~8GB VRAM. Falls back to CPU (slow).
- **Thumbnails**: Cached in RAM (`ThumbnailManager`). LRU eviction.
- **Batch Processing**: Uses `ThreadPoolExecutor`. Limit workers in Config if OOM occurs.


## 10. API Quick Reference

### Key Functions
- `execute_extraction(event: ExtractionEvent) -> Generator`
- `execute_pre_analysis(event: PreAnalysisEvent) -> Generator`
- `execute_propagation(event: PropagationEvent) -> Generator`

### Event Models
- `ExtractionEvent`: Source path, method (interval/scene).
- `PreAnalysisEvent`: Analysis params, seed strategy.
- `PropagationEvent`: List of scenes to process.


## 11. Git & Deployment

- **Submodules**: Always update recursive.
- **Requirements**: `requirements.txt` is root.
- **Validation**: Verify model downloads with SHA256.
