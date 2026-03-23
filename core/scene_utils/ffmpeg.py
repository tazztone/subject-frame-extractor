"""
FFmpeg-specific utilities for video processing and export.
"""

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from core.logger import AppLogger


def perform_ffmpeg_export(
    video_path: str, frames_to_extract: list[int], export_dir: Path, logger: "AppLogger"
) -> tuple[bool, Optional[str]]:
    """Execute FFmpeg command to extract specific frames from a video."""
    if not frames_to_extract:
        return True, None

    # Construct the select filter for multiple frames
    select_filter = f"select='{'+'.join([f'eq(n,{fn})' for fn in frames_to_extract])}'"

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        select_filter,
        "-vsync",
        "vfr",
        str(export_dir / "frame_%06d.png"),
    ]

    logger.info("Starting final export extraction...", extra={"command": " ".join(cmd)})

    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8")
        stdout, stderr = process.communicate()

        if process.returncode != 0:
            logger.error("FFmpeg export failed", extra={"stderr": stderr})
            return False, stderr

        return True, None
    except Exception as e:
        logger.error(f"Failed to execute FFmpeg: {e}")
        return False, str(e)
