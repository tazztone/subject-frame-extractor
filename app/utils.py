"""Core utility functions for the frame extractor application."""

import gc
import re
import time
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch


def sanitize_filename(name, max_length=50):
    return re.sub(r'[^\w\-_.]', '_', name)[:max_length]


@contextmanager
def safe_resource_cleanup():
    try:
        yield
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _to_json_safe(obj):
    """Convert objects to JSON-serializable format."""
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
    if hasattr(obj, '__dataclass_fields__'):  # Check if it's a dataclass
        return _to_json_safe(asdict(obj))
    return obj
