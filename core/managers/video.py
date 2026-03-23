from pathlib import Path
from typing import TYPE_CHECKING, Optional

import cv2
import numpy as np
import yt_dlp as ytdlp

from core.io_utils import validate_video_file

if TYPE_CHECKING:
    from core.config import Config
    from core.logger import AppLogger


class VideoManager:
    """Handles video preparation and metadata extraction."""

    def __init__(self, source_path: str, config: "Config", max_resolution: Optional[str] = None):
        self.source_path = source_path
        self.config = config
        self.max_resolution = max_resolution or self.config.default_max_resolution
        self.is_youtube = "youtube.com/" in source_path or "youtu.be/" in source_path

    def prepare_video(self, logger: "AppLogger") -> str:
        """Downloads or validates the video source."""
        if not self.source_path:
            raise ValueError("No video source path provided.")

        if self.is_youtube:
            logger.info("Downloading video", component="video", user_context={"source": self.source_path})
            tmpl = self.config.ytdl_output_template
            max_h = None if self.max_resolution == "maximum available" else int(self.max_resolution)
            ydl_opts = {
                "outtmpl": str(Path(self.config.downloads_dir) / tmpl),
                "format": self.config.ytdl_format_string.format(max_res=max_h)
                if max_h
                else "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best",
                "merge_output_format": "mp4",
                "noprogress": True,
                "quiet": True,
            }
            try:
                with ytdlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(self.source_path, download=True)
                    return str(Path(ydl.prepare_filename(info)))
            except ytdlp.utils.DownloadError as e:
                raise RuntimeError(f"Download failed: {e}") from e

        local_path = Path(self.source_path)
        validate_video_file(local_path)
        return str(local_path)

    @staticmethod
    def get_video_info(video_path: Optional[str]) -> dict:
        """Extracts metadata from the video file."""
        if not video_path:
            return {"width": 0, "height": 0, "fps": 30.0, "frame_count": 0}
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(f"Could not open video: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        if not np.isfinite(fps) or fps <= 0:
            fps = 30.0
        info = {
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": fps,
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        }
        cap.release()
        return info
