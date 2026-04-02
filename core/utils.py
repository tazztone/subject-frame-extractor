from __future__ import annotations

import contextlib
import functools
import gc
import inspect
import traceback
from typing import TYPE_CHECKING, Any, Callable, Optional

import numpy as np
import torch


def _setup_triton_mock():
    """Mocks the Triton library if it's missing (e.g., on Windows or non-CUDA environments)."""
    import sys
    import types
    from importlib.machinery import ModuleSpec
    from unittest.mock import MagicMock

    try:
        import triton  # noqa: F401

        return False
    except ImportError:
        pass

    mock_triton = types.ModuleType("triton")
    mock_triton.__spec__ = ModuleSpec("triton", None)
    mock_triton.__path__ = []
    setattr(mock_triton, "jit", lambda fn: fn)
    setattr(mock_triton, "language", types.ModuleType("triton.language"))

    class MockTL:
        constexpr = lambda x: x
        program_id = MagicMock(return_value=0)
        load = MagicMock(return_value=0)
        store = MagicMock()

    for attr in dir(MockTL):
        if not attr.startswith("_"):
            setattr(mock_triton.language, attr, getattr(MockTL, attr))

    sys.modules["triton"] = mock_triton
    sys.modules["triton.language"] = mock_triton.language
    return True


if TYPE_CHECKING:
    from core.logger import AppLogger
    from core.models import AnalysisParameters, Scene

# Re-export from specialized modules for backward compatibility
from core.image_utils import (  # noqa: F401
    draw_bbox,
    postprocess_mask,
    render_mask_overlay,
    rgb_to_pil,
)
from core.io_utils import (  # noqa: F401
    create_frame_map,
    detect_hwaccel,
    download_model,
    is_image_folder,
    list_images,
    sanitize_filename,
    validate_video_file,
)


def handle_common_errors(func: Callable) -> Callable:
    """Decorator to catch common exceptions and return a standardized error dictionary or yield it if a generator."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        is_gen = inspect.isgeneratorfunction(func)

        # Try to find logger in args or kwargs
        logger = kwargs.get("logger")
        if not logger:
            for arg in args:
                if hasattr(arg, "critical") and hasattr(arg, "error"):
                    logger = arg
                    break

        if is_gen:

            def gen_wrapper():
                try:
                    yield from func(*args, **kwargs)
                except FileNotFoundError as e:
                    msg = f"File not found: {e}"
                    if logger:
                        logger.error(msg)
                    yield {
                        "log": f"[ERROR] {msg}",
                        "status_message": "File not found",
                        "error_message": str(e),
                        "done": False,
                    }
                except (ValueError, TypeError) as e:
                    msg = f"Invalid input: {e}"
                    if logger:
                        logger.error(msg)
                    yield {
                        "log": f"[ERROR] {msg}",
                        "status_message": "Invalid input",
                        "error_message": str(e),
                        "done": False,
                    }
                except RuntimeError as e:
                    if "CUDA out of memory" in str(e):
                        msg = "CUDA OOM"
                        if logger:
                            logger.error(msg)
                        yield {
                            "log": f"[ERROR] {msg}",
                            "status_message": "GPU memory error",
                            "error_message": "CUDA out of memory",
                            "done": False,
                        }
                    else:
                        msg = f"Runtime error: {e}"
                        if logger:
                            logger.error(msg)
                        yield {
                            "log": f"[ERROR] {msg}",
                            "status_message": "Processing error",
                            "error_message": str(e),
                            "done": False,
                        }
                except Exception as e:
                    import traceback

                    msg = f"Unexpected error: {e}\n{traceback.format_exc()}"
                    if logger:
                        logger.critical(msg)
                    else:
                        print(f"CRITICAL: {msg}")
                    yield {
                        "log": f"[CRITICAL] {msg}",
                        "status_message": "Critical error",
                        "error_message": str(e),
                        "done": False,
                    }

            return gen_wrapper()

        try:
            return func(*args, **kwargs)
        except FileNotFoundError as e:
            return {
                "log": f"[ERROR] File not found: {e}",
                "status_message": "File not found",
                "error_message": str(e),
                "done": False,
            }
        except (ValueError, TypeError) as e:
            return {
                "log": f"[ERROR] Invalid input: {e}",
                "status_message": "Invalid input",
                "error_message": str(e),
                "done": False,
            }
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                return {
                    "log": "[ERROR] CUDA OOM",
                    "status_message": "GPU memory error",
                    "error_message": "CUDA out of memory",
                    "done": False,
                }
            return {
                "log": f"[ERROR] Runtime error: {e}",
                "status_message": "Processing error",
                "error_message": str(e),
                "done": False,
            }
        except Exception as e:
            return {
                "log": f"[CRITICAL] Unexpected error: {e}\n{traceback.format_exc()}",
                "status_message": "Critical error",
                "error_message": str(e),
                "done": False,
            }

    return wrapper


def monitor_memory_usage(logger: "AppLogger", device: str, threshold_mb: int = 8000):
    """Logs a warning and clears cache if GPU memory usage exceeds threshold."""
    if device == "cuda" and torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        if allocated > threshold_mb:
            logger.warning(f"High GPU memory usage: {allocated:.1f}MB")
            torch.cuda.empty_cache()


def estimate_totals(params: "AnalysisParameters", video_info: dict, scenes: Optional[list["Scene"]]) -> dict:
    """Estimates the total work items for each pipeline stage."""
    total_frames = int(video_info.get("frame_count") or 0)
    method = params.method

    if method == "every_nth_frame":
        extraction_total = max(1, int(total_frames / max(1, params.nth_frame)))
    elif method == "all":
        extraction_total = total_frames
    elif method == "keyframes":
        # Heuristic: assume ~15% of frames are keyframes/cuts
        extraction_total = max(1, int(total_frames * 0.15))
    else:
        extraction_total = total_frames
    scenes_count = len(scenes or [])
    pre_analysis_total = max(0, scenes_count)
    propagation_total = 0
    if scenes:
        for sc in scenes:
            propagation_total += max(0, sc.end_frame - sc.start_frame + 1)
    return {"extraction": extraction_total, "pre_analysis": pre_analysis_total, "propagation": propagation_total}


def _to_json_safe(obj: Any) -> Any:
    """Recursively converts objects (NumPy types, Path, etc.) to JSON-serializable types."""
    from pathlib import Path

    from pydantic import BaseModel

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


@contextlib.contextmanager
def safe_resource_cleanup(device: str = "cpu"):
    """Context manager to ensure garbage collection and CUDA cache clearing."""
    try:
        yield
    finally:
        gc.collect()
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
