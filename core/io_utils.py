"""
I/O and File System Utilities for Subject Frame Extractor
"""

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

import cv2

if TYPE_CHECKING:
    from core.config import Config
    from core.error_handling import ErrorHandler
    from core.logger import LoggerLike

from core.logger import log_with_component


def validate_video_file(video_path: str | Path) -> bool:
    """Checks if the video file exists, is not empty, and can be opened by OpenCV."""
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if path.stat().st_size == 0:
        raise ValueError(f"Video file is empty: {video_path}")
    try:
        cap = cv2.VideoCapture(str(video_path))
        try:
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {video_path}")
        finally:
            cap.release()
    except Exception as e:
        raise ValueError(f"Invalid video file: {e}")
    return True


def atomic_write_text(path: Union[str, Path], content: str, encoding: str = "utf-8"):
    """
    Writes content to a file atomically by writing to a temporary file
    and then renaming it to the target path.
    """
    path = Path(path)
    # Ensure directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    # Create temp file in the SAME directory to ensure atomic rename (same filesystem)
    fd, temp_path = tempfile.mkstemp(dir=str(path.parent), prefix=f"{path.name}.tmp")
    try:
        with os.fdopen(fd, "w", encoding=encoding) as f:
            f.write(content)
        # Final atomic replacement
        os.replace(temp_path, str(path))
    except Exception:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise


def sanitize_filename(name: str, config: "Config", max_length: Optional[int] = None) -> str:
    """Sanitizes a string to be safe for use as a filename."""
    max_length = max_length or config.utility_max_filename_length
    return re.sub(r"[^\w\-_.]", "_", name)[:max_length]


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


def list_images(p: Union[str, Path], cfg: "Config", recursive: bool = False) -> list[Path]:
    """Lists all valid image files in a directory (optionally recursive)."""
    p = Path(p)
    exts = {e.lower() for e in cfg.utility_image_extensions}
    if recursive:
        return sorted([f for f in p.rglob("*") if f.suffix.lower() in exts and f.is_file()])
    return sorted([f for f in p.iterdir() if f.suffix.lower() in exts and f.is_file()])


def detect_hwaccel(logger: "LoggerLike") -> tuple[Optional[str], Optional[str]]:
    """
    Probes FFmpeg for hardware acceleration support.

    Returns:
        tuple: (hwaccel_type, decoder_name) or (None, None)
    """
    try:
        # Check for NVIDIA NVENC/CUVID
        res = subprocess.run(["ffmpeg", "-hide_banner", "-hwaccels"], capture_output=True, text=True)
        hwaccels = res.stdout.splitlines()

        # Check for CUDA (Nvidia)
        if "cuda" in hwaccels or "cuvid" in hwaccels:
            logger.info("Hardware acceleration detected: NVIDIA (CUDA/CUVID)")
            return "cuda", None

        # Check for VAAPI (Intel/AMD on Linux)
        if "vaapi" in hwaccels:
            logger.info("Hardware acceleration detected: VAAPI")
            return "vaapi", None

        # Check for VideoToolbox (macOS)
        if "videotoolbox" in hwaccels:
            logger.info("Hardware acceleration detected: VideoToolbox")
            return "videotoolbox", None

    except Exception as e:
        logger.warning(f"Hardware acceleration detection failed: {e}")

    return None, None


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
    logger: "LoggerLike",
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
        with urllib.request.urlopen(req, timeout=180) as resp:
            content_length = resp.getheader("Content-Length")
            if content_length:
                size = int(content_length)
                logger.info(f"Reported size for {description}: {size / (1024 * 1024):.1f}MB")
                if min_size and size < min_size:
                    raise RuntimeError(
                        f"Remote file for {description} is too small ({size} bytes). "
                        f"Expected at least {min_size} bytes."
                    )

            with open(dest_path, "wb") as out:
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
        logger.info(f"{description} downloaded and verified successfully.")

    try:
        download_func()
    except Exception as e:
        logger.error(f"Failed to download {description}", exc_info=True, extra={"url": url})
        if dest_path.exists():
            dest_path.unlink()
        raise RuntimeError(f"Failed to download required model: {description}") from e


def create_frame_map(output_dir: Path, logger: Optional["LoggerLike"], ext: str = ".webp") -> dict:
    """Creates a mapping from original frame numbers to extracted filenames."""
    log_with_component(logger, "info", "Loading frame map...", component="frames")
    frame_map_path = output_dir / "frame_map.json"
    try:
        with open(frame_map_path, "r", encoding="utf-8") as f:
            frame_map_list = json.load(f)
        sorted_frames = sorted(map(int, frame_map_list))
        # Use % formatting for better performance in tight loops
        fmt = "frame_%06d" + ext
        return {orig_num: fmt % (i + 1) for i, orig_num in enumerate(sorted_frames)}
    except (FileNotFoundError, json.JSONDecodeError, TypeError) as e:
        if logger:
            logger.error(f"Could not load or parse frame_map.json: {e}. Frame mapping will be empty.", exc_info=False)
        return {}
