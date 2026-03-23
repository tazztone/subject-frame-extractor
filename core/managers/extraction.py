import json
import math
import subprocess
import threading
from pathlib import Path
from queue import Queue
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from core.config import Config
    from core.logger import AppLogger
    from core.managers import ModelRegistry

import re
import shutil

from core.error_handling import ErrorHandler
from core.managers.video import VideoManager
from core.models import AnalysisParameters
from core.photo_utils import ingest_folder
from core.progress import AdvancedProgressTracker
from core.scene_utils import run_scene_detection
from core.utils import (
    estimate_totals,
    is_image_folder,
)


def _process_ffmpeg_stream(
    stream, tracker: Optional["AdvancedProgressTracker"], desc: str, total_duration_s: float, start_time_s: float = 0
):
    """Parses FFmpeg progress stream and updates the tracker with optional time offset."""
    progress_data = {}
    for line in iter(stream.readline, ""):
        try:
            key, value = line.strip().split("=", 1)
            progress_data[key] = value
            if key == "progress" and value == "end":
                if tracker:
                    tracker.set(tracker.total, desc=desc)
                break
            if key == "out_time_us" and total_duration_s > 0:
                us = int(value)
                current_time_s = start_time_s + (us / 1_000_000)
                fraction = current_time_s / total_duration_s
                if tracker:
                    tracker.set(int(fraction * tracker.total), desc=desc)
            elif key == "frame" and tracker and total_duration_s <= 0:
                current_frame = int(value)
                tracker.set(current_frame, desc=desc)
        except ValueError:
            pass
    stream.close()


def _process_ffmpeg_showinfo(stream, fps: float) -> tuple[list, str]:
    """Parses FFmpeg stderr for 'showinfo' frame timestamps to map back to original frame indices."""
    frame_numbers = []
    stderr_lines = []
    for line in iter(stream.readline, ""):
        stderr_lines.append(line)
        if "[Parsed_showinfo_" in line and "pts_time:" in line:
            match = re.search(r"pts_time:(\d+\.?\d*)", line)
            if match:
                pts_time = float(match.group(1))
                orig_frame_idx = int(round(pts_time * fps))
                frame_numbers.append(orig_frame_idx)
    stream.close()
    return frame_numbers, "".join(stderr_lines)


def run_ffmpeg_extraction(
    video_path: str,
    output_dir: Path,
    video_info: dict,
    params: "AnalysisParameters",
    progress_queue: Queue,
    cancel_event: threading.Event,
    logger: "AppLogger",
    config: "Config",
    tracker: Optional["AdvancedProgressTracker"] = None,
):
    """Executes FFmpeg command to extract frames/thumbnails."""
    from core.utils import detect_hwaccel

    hwaccel_type = None
    if config.ffmpeg_hwaccel != "off":
        if config.ffmpeg_hwaccel == "auto":
            hwaccel_type, _ = detect_hwaccel(logger)
        else:
            hwaccel_type = config.ffmpeg_hwaccel

    thumb_dir = output_dir / "thumbs"
    thumb_dir.mkdir(exist_ok=True)

    target_area = params.thumb_megapixels * 1_000_000
    w, h = video_info.get("width", 1920), video_info.get("height", 1080)
    scale_factor = math.sqrt(target_area / (w * h)) if w * h > 0 else 1.0
    vf_scale_thumb = f"scale=w=trunc(iw*{scale_factor}/2)*2:h=trunc(ih*{scale_factor}/2)*2"
    vf_scale_video = "scale=-2:360"

    fps = max(1, int(video_info.get("fps", 30)))
    N = max(1, int(params.nth_frame or 0))

    start_frame_idx = 0
    existing_frame_map = []
    frame_map_path = output_dir / "frame_map.json"
    if frame_map_path.exists() and params.resume:
        try:
            with open(frame_map_path, "r", encoding="utf-8") as f:
                existing_frame_map = json.load(f)
            if existing_frame_map:
                ext = ".webp" if params.thumbnails_only else ".png"
                files = sorted(list(thumb_dir.glob(f"frame_*{ext}")))
                if len(files) > 0:
                    start_frame_idx = len(files)
                    logger.info(f"Resuming extraction from frame {start_frame_idx + 1}")
        except Exception as e:
            logger.warning(f"Could not load existing frame map for resume: {e}")

    select_map = {
        "keyframes": "select='eq(pict_type,I)'",
        "every_nth_frame": f"select='not(mod(n,{N}))'",
        "all": f"fps={fps}",
    }
    vf_select = select_map.get(params.method, f"fps={fps}")

    ext = ".webp" if params.thumbnails_only else ".png"
    codec = (
        ["-c:v", "libwebp", "-lossless", "0", "-quality", str(config.ffmpeg_thumbnail_quality)]
        if params.thumbnails_only
        else ["-c:v", "png"]
    )

    vf = f"{vf_select},{vf_scale_thumb if params.thumbnails_only else ''},showinfo".strip(",")

    cmd = ["ffmpeg", "-y"]
    if hwaccel_type:
        cmd.extend(["-hwaccel", hwaccel_type])
    cmd.extend(
        ["-i", str(video_path), "-vf", vf]
        + codec
        + [
            "-vsync",
            "vfr",
            "-start_number",
            str(start_frame_idx + 1),
            "-progress",
            "pipe:1",
            "-nostats",
            "-loglevel",
            "info",
            str(thumb_dir / f"frame_%06d{ext}"),
        ]
    )

    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", bufsize=1
    )
    stderr_results = {}

    with process.stdout, process.stderr:
        total_duration_s = video_info.get("frame_count", 0) / max(0.01, video_info.get("fps", 30))
        start_time_s = 0
        if start_frame_idx > 0:
            if params.method == "every_nth_frame":
                start_time_s = (start_frame_idx * N) / fps
            elif params.method == "all":
                start_time_s = start_frame_idx / fps

        stdout_thread = threading.Thread(
            target=lambda: _process_ffmpeg_stream(
                process.stdout, tracker, "Extracting frames", total_duration_s, start_time_s
            )
        )

        def process_stderr_and_store():
            nonlocal stderr_results
            frame_map, full_stderr = _process_ffmpeg_showinfo(process.stderr, fps)
            stderr_results["frame_map"] = frame_map
            stderr_results["full_stderr"] = full_stderr

        stderr_thread = threading.Thread(target=process_stderr_and_store)
        stdout_thread.start()
        stderr_thread.start()
        while process.poll() is None:
            try:
                process.wait(timeout=0.1)
            except subprocess.TimeoutExpired:
                if cancel_event.is_set():
                    process.terminate()
                    break
        stdout_thread.join()
        stderr_thread.join()

    frame_map_list = stderr_results.get("frame_map", [])
    if frame_map_list:
        final_frame_map = sorted(list(set(existing_frame_map + frame_map_list)))
        with open(output_dir / "frame_map.json", "w", encoding="utf-8") as f:
            json.dump(final_frame_map, f)

    if not cancel_event.is_set():
        lowres_video_path = output_dir / "video_lowres.mp4"
        lowres_cmd = ["ffmpeg", "-y"]
        if hwaccel_type:
            lowres_cmd.extend(["-hwaccel", hwaccel_type])
        lowres_cmd.extend(
            [
                "-i",
                str(video_path),
                "-vf",
                vf_scale_video,
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "23",
                "-an",
                str(lowres_video_path),
            ]
        )
        try:
            subprocess.run(lowres_cmd, capture_output=True, timeout=600)
        except Exception as e:
            logger.warning(f"Could not create downscaled video: {e}")


class ExtractionPipeline:
    """Pipeline for extracting frames from video or processing image folders."""

    def __init__(
        self,
        config: "Config",
        logger: "AppLogger",
        params: "AnalysisParameters",
        progress_queue: Queue,
        cancel_event: threading.Event,
        model_registry: Optional["ModelRegistry"] = None,
    ):
        self.config = config
        self.logger = logger
        self.params = params
        self.progress_queue = progress_queue
        self.cancel_event = cancel_event
        self.model_registry = model_registry
        self.error_handler = ErrorHandler(logger, config.retry_max_attempts, config.retry_backoff_seconds)
        self.run = self.error_handler.with_retry()(self._run_impl)

    def _run_impl(self, tracker: Optional["AdvancedProgressTracker"] = None) -> dict:
        source_p = Path(self.params.source_path)
        if is_image_folder(source_p):
            output_dir = Path(self.params.output_folder or Path(self.config.downloads_dir) / source_p.name)
            output_dir.mkdir(exist_ok=True, parents=True)
            self.logger.info(f"Ingesting image folder: {source_p.name}")
            thumb_dir = output_dir / "thumbs"
            thumb_dir.mkdir(exist_ok=True)
            ingested = ingest_folder(source_p, thumb_dir)
            if not ingested:
                return {"done": False, "log": "No images found."}
            scenes, frame_map, source_map = [], {}, {}
            ext = ".webp" if self.params.thumbnails_only else ".png"
            for i, photo in enumerate(ingested):
                frame_idx = i + 1
                target_name = f"frame_{frame_idx:06d}{ext}"
                target_path = thumb_dir / target_name
                if Path(photo["preview"]).exists() and Path(photo["preview"]) != target_path:
                    shutil.copy2(photo["preview"], target_path)
                scenes.append([frame_idx, frame_idx])
                frame_map[frame_idx] = target_name
                source_map[target_name] = str(photo["source"])
            with (output_dir / "scenes.json").open("w") as f:
                json.dump(scenes, f)
            with (output_dir / "frame_map.json").open("w") as f:
                json.dump(list(frame_map.keys()), f)
            with (output_dir / "source_map.json").open("w") as f:
                json.dump(source_map, f)
            return {"done": True, "output_dir": str(output_dir), "video_path": ""}
        else:
            vid_manager = VideoManager(self.params.source_path, self.config, self.params.max_resolution)
            video_path = Path(vid_manager.prepare_video(self.logger))
            output_dir = Path(self.params.output_folder or Path(self.config.downloads_dir) / video_path.stem)
            output_dir.mkdir(exist_ok=True, parents=True)
            video_info = VideoManager.get_video_info(video_path)
            if tracker:
                totals = estimate_totals(self.params, video_info, None)
                tracker.start(totals["extraction"], desc="Extracting frames")
            if self.params.scene_detect:
                run_scene_detection(video_path, output_dir, self.logger)
            run_ffmpeg_extraction(
                video_path,
                output_dir,
                video_info,
                self.params,
                self.progress_queue,
                self.cancel_event,
                self.logger,
                self.config,
                tracker=tracker,
            )
            if self.cancel_event.is_set():
                return {"done": False, "log": "Extraction cancelled"}
            return {"done": True, "output_dir": str(output_dir), "video_path": str(video_path)}
