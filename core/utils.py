from __future__ import annotations

import contextlib
import functools
import gc
import hashlib
import json
import re
import shutil
import traceback
import urllib.error
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional, Union

import cv2
import numpy as np
import torch
from numba import njit
from PIL import Image
from pydantic import BaseModel

if TYPE_CHECKING:
    from core.config import Config
    from core.error_handling import ErrorHandler
    from core.logger import AppLogger
    from core.models import AnalysisParameters, Scene


def handle_common_errors(func: Callable) -> Callable:
    """Decorator to catch common exceptions and return a standardized error dictionary."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except FileNotFoundError as e:
            return {"log": f"[ERROR] File not found: {e}", "status_message": "File not found", "error_message": str(e)}
        except (ValueError, TypeError) as e:
            return {"log": f"[ERROR] Invalid input: {e}", "status_message": "Invalid input", "error_message": str(e)}
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                return {
                    "log": "[ERROR] CUDA OOM",
                    "status_message": "GPU memory error",
                    "error_message": "CUDA out of memory",
                }
            return {"log": f"[ERROR] Runtime error: {e}", "status_message": "Processing error", "error_message": str(e)}
        except Exception as e:
            return {
                "log": f"[CRITICAL] Unexpected error: {e}\n{traceback.format_exc()}",
                "status_message": "Critical error",
                "error_message": str(e),
            }

    return wrapper


def monitor_memory_usage(logger: "AppLogger", device: str, threshold_mb: int = 8000):
    """Logs a warning and clears cache if GPU memory usage exceeds threshold."""
    if device == "cuda" and torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        if allocated > threshold_mb:
            logger.warning(f"High GPU memory usage: {allocated:.1f}MB")
            torch.cuda.empty_cache()


def validate_video_file(video_path: str) -> bool:
    """Checks if the video file exists, is not empty, and can be opened by OpenCV."""
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if path.stat().st_size == 0:
        raise ValueError(f"Video file is empty: {video_path}")
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        cap.release()
    except Exception as e:
        raise ValueError(f"Invalid video file: {e}")
    return True


def estimate_totals(params: "AnalysisParameters", video_info: dict, scenes: Optional[list["Scene"]]) -> dict:
    """Estimates the total work items for each pipeline stage."""
    fps = max(1, int(video_info.get("fps") or 30))
    total_frames = int(video_info.get("frame_count") or 0)
    method = params.method
    if method == "interval":
        extraction_total = max(1, int(total_frames / max(0.1, params.interval) / fps))
    elif method == "every_nth_frame":
        extraction_total = max(1, int(total_frames / max(1, params.nth_frame)))
    elif method == "all":
        extraction_total = total_frames
    elif method in ("keyframes", "nth_plus_keyframes"):
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


def sanitize_filename(name: str, config: "Config", max_length: Optional[int] = None) -> str:
    """Sanitizes a string to be safe for use as a filename."""
    max_length = max_length or config.utility_max_filename_length
    return re.sub(r"[^\w\-_.]", "_", name)[:max_length]


def _to_json_safe(obj: Any) -> Any:
    """Recursively converts objects (NumPy types, Path, etc.) to JSON-serializable types."""
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


def is_image_folder(p: Union[str, Path]) -> bool:
    """Checks if the path points to a directory."""
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
    """Lists all valid image files in a directory."""
    p = Path(p)
    exts = {e.lower() for e in cfg.utility_image_extensions}
    return sorted([f for f in p.iterdir() if f.suffix.lower() in exts and f.is_file()])


@njit
def compute_entropy(hist: np.ndarray, entropy_norm: float) -> float:
    """Computes normalized entropy from a histogram using Numba."""
    prob = hist / (np.sum(hist) + 1e-7)
    entropy = -np.sum(prob[prob > 0] * np.log2(prob[prob > 0]))
    return min(max(entropy / entropy_norm, 0), 1.0)


def _compute_sha256(path: Path) -> str:
    """Computes SHA256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def download_model(
    url: str,
    dest_path: Union[str, Path],
    description: str,
    logger: "AppLogger",
    error_handler: "ErrorHandler",
    user_agent: str,
    min_size: int = 1_000_000,
    expected_sha256: Optional[str] = None,
    token: Optional[str] = None,
):
    """
    Downloads a file from a URL with retries, validation, and progress logging.
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
                logger.warning(
                    f"Cached {description} has incorrect SHA256. Re-downloading.",
                    extra={"expected": expected_sha256, "actual": actual_sha256},
                )
                dest_path.unlink()
        elif min_size is None or dest_path.stat().st_size >= min_size:
            logger.info(f"Using cached {description} (SHA not verified): {dest_path}")
            return

    @error_handler.with_retry(recoverable_exceptions=(urllib.error.URLError, TimeoutError, RuntimeError))
    def download_func():
        logger.info(f"Downloading {description}", extra={"url": url, "dest": dest_path})
        headers = {"User-Agent": user_agent}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=180) as resp, open(dest_path, "wb") as out:
            shutil.copyfileobj(resp, out)

        if not dest_path.exists():
            raise RuntimeError(f"Download of {description} failed (file not found after download).")

        if expected_sha256:
            actual_sha256 = _compute_sha256(dest_path)
            if actual_sha256 != expected_sha256:
                raise RuntimeError(
                    f"SHA256 mismatch for {description}. Expected {expected_sha256}, got {actual_sha256}."
                )
        elif dest_path.stat().st_size < min_size:
            raise RuntimeError(f"Downloaded {description} seems incomplete (file size too small).")
        logger.success(f"{description} downloaded and verified successfully.")

    try:
        download_func()
    except Exception as e:
        logger.error(f"Failed to download {description}", exc_info=True, extra={"url": url})
        if dest_path.exists():
            dest_path.unlink()
        raise RuntimeError(f"Failed to download required model: {description}") from e


def postprocess_mask(
    mask: np.ndarray, config: "Config", fill_holes: bool = True, keep_largest_only: bool = True
) -> np.ndarray:
    """Cleans up binary masks using morphological operations and connected components."""
    if mask is None or mask.size == 0:
        return mask
    binary_mask = (mask > 128).astype(np.uint8)
    if fill_holes:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (config.masking_close_kernel_size, config.masking_close_kernel_size)
        )
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    if keep_largest_only and config.masking_keep_largest_only:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
        if num_labels > 1:
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            binary_mask = (labels == largest_label).astype(np.uint8)
    return (binary_mask * 255).astype(np.uint8)


def render_mask_overlay(frame_rgb: np.ndarray, mask_gray: np.ndarray, alpha: float, logger: "AppLogger") -> np.ndarray:
    """overlays a semi-transparent red mask on the image."""
    if mask_gray is None or frame_rgb is None:
        return frame_rgb if frame_rgb is not None else np.array([])
    h, w = frame_rgb.shape[:2]
    if mask_gray.shape[:2] != (h, w):
        mask_gray = cv2.resize(mask_gray, (w, h), interpolation=cv2.INTER_NEAREST)
    m = mask_gray > 128
    red_layer = np.zeros_like(frame_rgb, dtype=np.uint8)
    red_layer[..., 0] = 255
    blended = cv2.addWeighted(frame_rgb, 1.0 - alpha, red_layer, alpha, 0.0)
    if m.ndim == 2:
        m = m[..., np.newaxis]
    elif m.ndim == 3 and m.shape[2] != 1:
        logger.warning("Unexpected mask shape. Skipping overlay.", extra={"shape": m.shape})
        return frame_rgb
    return np.where(m, blended, frame_rgb)


def rgb_to_pil(image_rgb: np.ndarray) -> Image.Image:
    """Converts a NumPy RGB array to a PIL Image."""
    return Image.fromarray(image_rgb)


def create_frame_map(output_dir: Path, logger: "AppLogger", ext: str = ".webp") -> dict:
    """Creates a mapping from original frame numbers to extracted filenames."""
    logger.info("Loading frame map...", component="frames")
    frame_map_path = output_dir / "frame_map.json"
    try:
        with open(frame_map_path, "r", encoding="utf-8") as f:
            frame_map_list = json.load(f)
        sorted_frames = sorted(map(int, frame_map_list))
        return {orig_num: f"frame_{i + 1:06d}{ext}" for i, orig_num in enumerate(sorted_frames)}
    except (FileNotFoundError, json.JSONDecodeError, TypeError) as e:
        logger.error(f"Could not load or parse frame_map.json: {e}. Frame mapping will be empty.", exc_info=False)
        return {}


def draw_bbox(
    img_rgb: np.ndarray,
    xywh: list,
    config: "Config",
    color: Optional[tuple] = None,
    thickness: Optional[int] = None,
    label: Optional[str] = None,
) -> np.ndarray:
    """Draws a bounding box and optional label on an image."""
    color = color or tuple(config.visualization_bbox_color)
    thickness = thickness or config.visualization_bbox_thickness
    x, y, w, h = map(int, xywh or [0, 0, 0, 0])
    img_out = img_rgb.copy()
    cv2.rectangle(img_out, (x, y), (x + w, y + h), color, thickness)
    if label:
        font_scale = 1
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        text_x = x + 5
        text_y = y + text_height + 5
        cv2.rectangle(img_out, (x, y), (x + text_width + 10, y + text_height + 10), color, -1)
        cv2.putText(img_out, label, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)
    return img_out
