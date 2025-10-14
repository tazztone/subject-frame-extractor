"""Video management and processing utilities."""

import json
import math
import subprocess
import time
from pathlib import Path
import cv2
import yt_dlp as ytdlp
from scenedetect import detect, ContentDetector

from app.config import Config
from app.logging_enhanced import EnhancedLogger


class VideoManager:
    def __init__(self, source_path, max_resolution="maximum available"):
        self.source_path = source_path
        self.max_resolution = max_resolution
        self.is_youtube = ("youtube.com/" in source_path or
                           "youtu.be/" in source_path)

    def prepare_video(self, logger=None):
        """Prepare video for processing (download if YouTube, validate)."""
        config = Config()
        logger = logger or EnhancedLogger()

        if self.is_youtube:
            if not ytdlp:
                raise ImportError("yt-dlp not installed.")
            logger.info("Downloading video",
                       component="video",
                       user_context={'source': self.source_path})

            if self.max_resolution != "maximum available":
                res_filter = f"[height<={self.max_resolution}]"
            else:
                res_filter = ""

            ydl_opts = {
                'outtmpl': str(
                    config.DIRS['downloads'] /
                    '%(id)s_%(title).40s_%(height)sp.%(ext)s'
                ),
                'format': (f'bestvideo{res_filter}[ext=mp4]+'
                           f'bestaudio[ext=m4a]/best{res_filter}[ext=mp4]/'
                           f'best'),
                'merge_output_format': 'mp4',
                'noprogress': True,
                'quiet': True
            }
            try:
                with ytdlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(self.source_path, download=True)
                    return str(Path(ydl.prepare_filename(info)))
            except ytdlp.utils.DownloadError as e:
                raise RuntimeError(
                    f"Download failed. Resolution may not be available. "
                    f"Details: {e}"
                ) from e

        local_path = Path(self.source_path)
        if not local_path.is_file():
            raise FileNotFoundError(f"Video file not found: {local_path}")
        return str(local_path)

    @staticmethod
    def get_video_info(video_path):
        """Extract video metadata using OpenCV."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(f"Could not open video: {video_path}")
            
        info = {
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        }
        cap.release()
        return info


def run_scene_detection(video_path, output_dir, logger=None):
    """Run scene detection and save results."""
    logger = logger or EnhancedLogger()
    
    logger.info("Detecting scenes...", component="video")
    try:
        scene_list = detect(str(video_path), ContentDetector())
        shots = ([(s.frame_num, e.frame_num) for s, e in scene_list]
                if scene_list else [])
        with (output_dir / "scenes.json").open('w', encoding='utf-8') as f:
            json.dump(shots, f)
        logger.success(f"Found {len(shots)} scenes.", component="video")
        return shots
    except Exception as e:
        logger.error("Scene detection failed.", component="video", exc_info=True)
        return []


def run_ffmpeg_extraction(video_path, output_dir, video_info, params,
                          progress_queue, cancel_event, logger=None):
    """Run FFmpeg extraction with progress tracking."""
    import re

    logger = logger or EnhancedLogger()
    log_file_path = output_dir / "ffmpeg_log.txt"

    cmd_base = ['ffmpeg', '-y', '-i', str(video_path), '-hide_banner',
                '-loglevel', 'info']

    if params.thumbnails_only:
        thumb_dir = output_dir / "thumbs"
        thumb_dir.mkdir(exist_ok=True)

        target_area = params.thumb_megapixels * 1_000_000
        w, h = video_info.get('width', 1920), video_info.get('height', 1080)
        scale_factor = math.sqrt(target_area / (w * h))
        vf_scale = (f"scale=w=trunc(iw*{scale_factor}/2)*2:"
                    f"h=trunc(ih*{scale_factor}/2)*2")

        fps = video_info.get('fps', 30)
        vf_filter = f"fps={fps}," + vf_scale + ",showinfo"
        cmd = cmd_base + [
            '-vf', vf_filter,
            '-c:v', 'libwebp',
            '-lossless', '0',
            '-quality', '80',
            '-vsync', 'vfr',
            str(thumb_dir / "frame_%06d.webp")
        ]
    else:  # Legacy full-res extraction
        select_filter_map = {
            'interval': f"fps=1/{max(0.1, float(params.interval))}",
            'keyframes': "select='eq(pict_type,I)'",
            'scene': (f"select='gt(scene,"
                     f"{0.5 if params.fast_scene else 0.4})'"),
            'all': f"fps={video_info.get('fps', 30)}",
            'every_nth_frame': (f"select='not(mod(n,"
                                f"{max(1, int(params.nth_frame))}))'")}
        select_filter = select_filter_map.get(params.method)
        vf_filter = (select_filter + ",showinfo"
                     if select_filter else "showinfo")

        ext = 'png' if params.use_png else 'jpg'
        cmd = cmd_base + [
            '-vf', vf_filter, '-vsync', 'vfr', '-f', 'image2',
            str(output_dir / f"frame_%06d.{ext}")
        ]

    with open(log_file_path, 'w', encoding='utf-8') as stderr_handle:
        process = subprocess.Popen(cmd, stderr=stderr_handle, text=True,
                                  encoding='utf-8', bufsize=1)
        progress_queue.put({
            "total": video_info.get('frame_count', 1),
            "stage": "Extraction"
        })

        while process.poll() is None:
            if cancel_event.is_set():
                process.terminate()
                break
            time.sleep(0.1)
        process.wait()

    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            log_content = f.read()
        frame_map_list = [
            int(m.group(1)) for m in re.finditer(r' n:\s*(\d+)', log_content)
        ]
        with open(output_dir / "frame_map.json", 'w', encoding='utf-8') as f:
            json.dump(frame_map_list, f)
    finally:
        log_file_path.unlink(missing_ok=True)

    if process.returncode != 0 and not cancel_event.is_set():
        raise RuntimeError(f"FFmpeg failed with code {process.returncode}.")