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


def safe_execute_with_retry(func, max_retries=3, delay=1.0, backoff=2.0):
    """Execute a function with retry logic and exponential backoff."""
    # Import logger locally to avoid circular imports
    from app.core.logging import UnifiedLogger
    logger = UnifiedLogger()

    last_exception = None
    for attempt in range(max_retries + 1):
        try:
            return func()
        except Exception as e:
            last_exception = e
            if attempt < max_retries:
                retry_msg = (f"Attempt {attempt + 1} failed. "
                           f"Retrying in {delay}s...")
                logger.warning(retry_msg, extra={'error': e})
                time.sleep(delay)
                delay *= backoff
    error_msg = "Function failed after retries."
    raise last_exception if last_exception else RuntimeError(error_msg)


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
    if hasattr(obj, '__dict__'):  # For dataclasses
        return asdict(obj)
    return obj
