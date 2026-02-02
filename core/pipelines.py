from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from queue import Queue
from typing import TYPE_CHECKING, Callable, Generator, Optional, Union

import cv2
import gradio as gr
import numpy as np
import torch

# Note: Scene uses Pydantic (Scene.model_fields.keys()), not dataclass.fields()
from PIL import Image

if TYPE_CHECKING:
    from core.config import Config
    from core.logger import AppLogger
    from core.managers import ModelRegistry, ThumbnailManager

from core.database import Database
from core.error_handling import ErrorHandler
from core.events import ExtractionEvent, PreAnalysisEvent, PropagationEvent, SessionLoadEvent
from core.managers import VideoManager, initialize_analysis_models
from core.models import AnalysisParameters, Frame, Scene
from core.progress import AdvancedProgressTracker
from core.scene_utils import (
    SubjectMasker,
    make_photo_thumbs,
    run_scene_detection,
    save_scene_seeds,
)
from core.utils import (
    _to_json_safe,
    create_frame_map,
    estimate_totals,
    handle_common_errors,
    monitor_memory_usage,
    sanitize_filename,
)


def _process_ffmpeg_stream(stream, tracker: Optional["AdvancedProgressTracker"], desc: str, total_duration_s: float):
    """Parses FFmpeg progress stream and updates the tracker."""
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
                fraction = us / (total_duration_s * 1_000_000)
                if tracker:
                    tracker.set(int(fraction * tracker.total), desc=desc)
            elif key == "frame" and tracker and total_duration_s <= 0:
                current_frame = int(value)
                tracker.set(current_frame, desc=desc)
        except ValueError:
            pass
    stream.close()


def _process_ffmpeg_showinfo(stream) -> tuple[list, str]:
    """Parses FFmpeg stderr for 'showinfo' frame numbers."""
    frame_numbers = []
    stderr_lines = []
    for line in iter(stream.readline, ""):
        stderr_lines.append(line)
        match = re.search(r" n:\s*(\d+)", line)
        if match:
            frame_numbers.append(int(match.group(1)))
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
    """
    Executes FFmpeg command to extract frames/thumbnails.

    Constructs complex filter chains based on extraction method (interval, keyframes, etc.).
    Also creates a downscaled video (video_lowres.mp4) for efficient SAM3 processing.
    """
    # TODO: Add configurable audio codec for downscaled video
    # TODO: Support hardware-accelerated encoding (NVENC/VAAPI)
    cmd_base = ["ffmpeg", "-y", "-i", str(video_path), "-hide_banner"]
    progress_args = ["-progress", "pipe:1", "-nostats", "-loglevel", "info"]
    cmd_base.extend(progress_args)

    thumb_dir = output_dir / "thumbs"
    thumb_dir.mkdir(exist_ok=True)

    target_area = params.thumb_megapixels * 1_000_000
    w, h = video_info.get("width", 1920), video_info.get("height", 1080)
    scale_factor = math.sqrt(target_area / (w * h)) if w * h > 0 else 1.0
    vf_scale = f"scale=w=trunc(iw*{scale_factor}/2)*2:h=trunc(ih*{scale_factor}/2)*2"

    fps = max(1, int(video_info.get("fps", 30)))
    N = max(1, int(params.nth_frame or 0))
    interval = max(0.1, float(params.interval or 0.0))

    select_map = {
        "keyframes": "select='eq(pict_type,I)'",
        "every_nth_frame": f"select='not(mod(n,{N}))'",
        "nth_plus_keyframes": f"select='or(eq(pict_type,I),not(mod(n,{N})))'",
        "interval": f"fps=1/{interval}",
        "all": f"fps={fps}",
    }
    vf_select = select_map.get(params.method, f"fps={fps}")

    if params.thumbnails_only:
        vf = f"{vf_select},{vf_scale},showinfo"
        cmd = cmd_base + [
            "-vf",
            vf,
            "-c:v",
            "libwebp",
            "-lossless",
            "0",
            "-quality",
            str(config.ffmpeg_thumbnail_quality),
            "-vsync",
            "vfr",
            str(thumb_dir / "frame_%06d.webp"),
        ]
    else:
        vf = f"{vf_select},showinfo"
        cmd = cmd_base + ["-vf", vf, "-c:v", "png", "-vsync", "vfr", str(thumb_dir / "frame_%06d.png")]

    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", bufsize=1
    )

    frame_map_list = []
    stderr_results = {}

    with process.stdout, process.stderr:
        total_duration_s = video_info.get("frame_count", 0) / max(0.01, video_info.get("fps", 30))
        stdout_thread = threading.Thread(
            target=lambda: _process_ffmpeg_stream(process.stdout, tracker, "Extracting frames", total_duration_s)
        )

        def process_stderr_and_store():
            nonlocal stderr_results
            frame_map, full_stderr = _process_ffmpeg_showinfo(process.stderr)
            stderr_results["frame_map"] = frame_map
            stderr_results["full_stderr"] = full_stderr

        stderr_thread = threading.Thread(target=process_stderr_and_store)
        stdout_thread.start()
        stderr_thread.start()

        while process.poll() is None:
            if cancel_event.is_set():
                process.terminate()
                break
            time.sleep(0.1)

        stdout_thread.join()
        stderr_thread.join()

    process.wait()

    frame_map_list = stderr_results.get("frame_map", [])
    stderr_output = stderr_results.get("full_stderr", "")

    if frame_map_list:
        with open(output_dir / "frame_map.json", "w", encoding="utf-8") as f:
            json.dump(sorted(frame_map_list), f)

    if process.returncode not in [0, -9] and not cancel_event.is_set():
        logger.error("FFmpeg extraction failed", extra={"returncode": process.returncode, "stderr": stderr_output})
        raise RuntimeError(f"FFmpeg failed with code {process.returncode}. Check logs for details.")

    # Create downscaled video for SAM3 (avoids temp JPEG I/O during propagation)
    if not cancel_event.is_set():
        lowres_video_path = output_dir / "video_lowres.mp4"
        logger.info(f"Creating downscaled video for SAM3: {lowres_video_path}")
        lowres_cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-hide_banner",
            "-loglevel",
            "warning",
            "-vf",
            vf_scale,
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "23",
            "-an",  # No audio needed
            str(lowres_video_path),
        ]
        try:
            lowres_proc = subprocess.run(lowres_cmd, capture_output=True, text=True, timeout=600)
            if lowres_proc.returncode == 0:
                logger.success(f"Downscaled video created: {lowres_video_path}")
            else:
                logger.warning(f"Failed to create downscaled video: {lowres_proc.stderr[:500]}")
        except subprocess.TimeoutExpired:
            logger.warning("Downscaled video creation timed out")
        except Exception as e:
            logger.warning(f"Could not create downscaled video: {e}")


class Pipeline:
    """Base class for processing pipelines."""

    def __init__(
        self,
        config: "Config",
        logger: "AppLogger",
        params: "AnalysisParameters",
        progress_queue: Queue,
        cancel_event: threading.Event,
    ):
        self.config = config
        self.logger = logger
        self.params = params
        self.progress_queue = progress_queue
        self.cancel_event = cancel_event


class ExtractionPipeline(Pipeline):
    """Pipeline for extracting frames from video or processing image folders."""

    # TODO: Add support for resumable extraction from checkpoints
    # TODO: Implement parallel video decoding for multi-GPU systems
    # TODO: Add extraction quality validation step
    def __init__(
        self,
        config: "Config",
        logger: "AppLogger",
        params: "AnalysisParameters",
        progress_queue: Queue,
        cancel_event: threading.Event,
    ):
        super().__init__(config, logger, params, progress_queue, cancel_event)
        self.error_handler = ErrorHandler(
            self.logger, self.config.retry_max_attempts, self.config.retry_backoff_seconds
        )
        self.run = self.error_handler.with_retry()(self._run_impl)

    def _run_impl(self, tracker: Optional["AdvancedProgressTracker"] = None) -> dict:
        """Internal execution logic for extraction."""
        source_p = Path(self.params.source_path)
        from core.utils import is_image_folder, list_images

        is_folder = is_image_folder(source_p)

        if is_folder:
            output_dir = (
                Path(self.params.output_folder)
                if self.params.output_folder
                else Path(self.config.downloads_dir) / source_p.name
            )
            output_dir.mkdir(exist_ok=True, parents=True)
            params_dict = self.params.model_dump()
            params_dict["output_folder"] = str(output_dir)
            params_dict["video_path"] = ""
            run_cfg_path = output_dir / "run_config.json"
            try:
                with run_cfg_path.open("w", encoding="utf-8") as f:
                    json.dump(_to_json_safe(params_dict), f, indent=2)
            except OSError as e:
                self.logger.warning(f"Could not write run config to {run_cfg_path}: {e}")

            self.logger.info(f"Processing image folder: {source_p.name}")
            images = list_images(source_p, self.config)
            if not images:
                self.logger.warning("No images found in the specified folder.")
                return {"done": False, "log": "No images found."}

            make_photo_thumbs(images, output_dir, self.params, self.config, self.logger, tracker=tracker)

            num_images = len(images)
            scenes = [[i, i] for i in range(1, num_images + 1)]
            with (output_dir / "scenes.json").open("w", encoding="utf-8") as f:
                json.dump(scenes, f)
            return {"done": True, "output_dir": str(output_dir), "video_path": ""}
        else:
            self.logger.info("Preparing video source...")
            vid_manager = VideoManager(self.params.source_path, self.config, self.params.max_resolution)
            video_path = Path(vid_manager.prepare_video(self.logger))
            output_dir = (
                Path(self.params.output_folder)
                if self.params.output_folder
                else Path(self.config.downloads_dir) / video_path.stem
            )
            output_dir.mkdir(exist_ok=True, parents=True)

            params_dict = self.params.model_dump()
            params_dict["output_folder"] = str(output_dir)
            params_dict["video_path"] = str(video_path)

            run_cfg_path = output_dir / "run_config.json"
            try:
                with run_cfg_path.open("w", encoding="utf-8") as f:
                    json.dump(_to_json_safe(params_dict), f, indent=2)
            except OSError as e:
                self.logger.warning(f"Could not write run config to {run_cfg_path}: {e}")

            self.logger.info("Video ready", user_context={"path": sanitize_filename(video_path.name, self.config)})
            video_info = VideoManager.get_video_info(video_path)
            fps = video_info.get("fps") or 30
            duration = video_info.get("frame_count", 0) / fps
            if duration < self.config.validation_min_duration_secs:
                self.logger.warning(
                    f"Video duration ({duration:.2f}s) is less than the minimum "
                    f"of {self.config.validation_min_duration_secs}s."
                )
                return {
                    "done": False,
                    "log": f"Video is too short ({duration:.2f}s). "
                    f"Minimum duration is {self.config.validation_min_duration_secs}s.",
                }

            if video_info.get("frame_count", 0) < self.config.validation_min_frame_count:
                self.logger.warning(
                    f"Video frame count ({video_info.get('frame_count', 0)}) is less than the minimum "
                    f"of {self.config.validation_min_frame_count}."
                )
                return {
                    "done": False,
                    "log": f"Video has too few frames ({video_info.get('frame_count', 0)}). "
                    f"Minimum is {self.config.validation_min_frame_count}.",
                }

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
                self.logger.info("Extraction cancelled by user.")
                if tracker:
                    tracker.done_stage("Extraction cancelled")
                return {"done": False, "log": "Extraction cancelled"}

            if tracker:
                tracker.done_stage("Extraction complete")
            self.logger.success("Extraction complete.")
            return {"done": True, "output_dir": str(output_dir), "video_path": str(video_path)}


class AnalysisPipeline(Pipeline):
    """Pipeline for analyzing frames (pre-analysis, propagation, full analysis)."""

    # TODO: Add automatic batch size adjustment based on available memory
    # TODO: Implement per-frame error recovery instead of skipping
    # TODO: Add analysis caching to skip already-analyzed frames
    def __init__(
        self,
        config: "Config",
        logger: "AppLogger",
        params: "AnalysisParameters",
        progress_queue: Queue,
        cancel_event: threading.Event,
        thumbnail_manager: "ThumbnailManager",
        model_registry: "ModelRegistry",
    ):
        super().__init__(config, logger, params, progress_queue, cancel_event)
        self.output_dir = Path(self.params.output_folder)
        self.db = Database(self.output_dir / "metadata.db")
        self.thumb_dir = self.output_dir / "thumbs"
        self.masks_dir = self.output_dir / "masks"
        self.processing_lock = threading.Lock()
        self.face_analyzer, self.reference_embedding, self.mask_metadata, self.face_landmarker = None, None, {}, None
        self.scene_map, self.niqe_metric = {}, None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.thumbnail_manager = thumbnail_manager
        self.model_registry = model_registry

    def _initialize_niqe_metric(self):
        """Lazy initialization of the NIQE metric model."""
        if self.niqe_metric is None:
            try:
                import pyiqa

                self.niqe_metric = pyiqa.create_metric("niqe", device=self.device)
                self.logger.info("NIQE metric initialized successfully")
            except ImportError:
                self.logger.warning("pyiqa is not installed, NIQE metric is unavailable.")
            except Exception as e:
                self.logger.warning("Failed to initialize NIQE metric", extra={"error": e})

    def run_full_analysis(
        self, scenes_to_process: list["Scene"], tracker: Optional["AdvancedProgressTracker"] = None
    ) -> dict:
        """
        Runs the mask propagation phase.

        Despite the name, this currently focuses on propagation (subject masking) for video,
        or full processing for image folders.
        """
        is_folder_mode = not self.params.video_path
        if is_folder_mode:
            return self._run_image_folder_analysis(tracker=tracker)

        progress_file = self.output_dir / "progress.json"
        completed_scene_ids = []

        try:
            if progress_file.exists() and self.params.resume:
                with progress_file.open("r", encoding="utf-8") as f:
                    progress_data = json.load(f)
                scenes_to_process = self._filter_completed_scenes(scenes_to_process, progress_data)

            if not scenes_to_process:
                self.logger.info("All scenes already processed, skipping.")
                return {"done": True, "output_dir": str(self.output_dir)}

            self.db.connect()
            self.db.migrate()
            if not self.params.resume:
                self.db.clear_metadata()

            self.scene_map = {s.shot_id: s for s in scenes_to_process}
            self.logger.info("Initializing Models")

            models = initialize_analysis_models(self.params, self.config, self.logger, self.model_registry)
            self.face_analyzer = models["face_analyzer"]
            self.reference_embedding = models["ref_emb"]
            self.face_landmarker = models["face_landmarker"]

            if self.face_analyzer and self.params.face_ref_img_path:
                self._process_reference_face()

            self.params.need_masks_now = True
            self.params.enable_subject_mask = True
            ext = ".webp" if self.params.thumbnails_only else ".png"

            masker = SubjectMasker(
                self.params,
                self.progress_queue,
                self.cancel_event,
                self.config,
                create_frame_map(self.output_dir, self.logger, ext=ext),
                self.face_analyzer,
                self.reference_embedding,
                thumbnail_manager=self.thumbnail_manager,
                niqe_metric=self.niqe_metric,
                logger=self.logger,
                face_landmarker=self.face_landmarker,
                device=models["device"],
                model_registry=self.model_registry,
            )

            for scene in scenes_to_process:
                if self.cancel_event.is_set():
                    self.logger.info("Propagation cancelled by user.")
                    break
                self.mask_metadata.update(masker.run_propagation(str(self.output_dir), [scene], tracker=tracker))
                completed_scene_ids.append(scene.shot_id)

            if self.cancel_event.is_set():
                self.logger.info("Propagation cancelled by user.")
                return {"log": "Propagation cancelled.", "done": False}

            self.logger.success("Propagation complete.", extra={"output_dir": self.output_dir})
            return {"done": True, "output_dir": str(self.output_dir)}

        except Exception as e:
            self.logger.error(
                "Propagation pipeline failed", component="analysis", exc_info=True, extra={"error": str(e)}
            )
            return {"error": str(e), "done": False}
        finally:
            self._save_progress_bulk(completed_scene_ids, progress_file)

    def run_analysis_only(
        self, scenes_to_process: list["Scene"], tracker: Optional["AdvancedProgressTracker"] = None
    ) -> dict:
        """
        Runs the frame analysis phase (calculating quality metrics).

        This phase consumes the masks generated in the propagation phase.
        """
        try:
            self.db.connect()
            self.db.migrate()
            if not self.params.resume:
                self.db.clear_metadata()
            self.scene_map = {s.shot_id: s for s in scenes_to_process}
            self.logger.info("Initializing Models for Analysis")
            models = initialize_analysis_models(self.params, self.config, self.logger, self.model_registry)
            self.face_analyzer = models["face_analyzer"]
            self.reference_embedding = models["ref_emb"]
            self.face_landmarker = models["face_landmarker"]
            if self.face_analyzer and self.params.face_ref_img_path:
                self._process_reference_face()
            mask_metadata_path = self.output_dir / "mask_metadata.json"
            if mask_metadata_path.exists():
                with open(mask_metadata_path, "r", encoding="utf-8") as f:
                    self.mask_metadata = json.load(f)
            else:
                self.mask_metadata = {}
            if tracker:
                tracker.set_stage("Analyzing frames")
            self._initialize_niqe_metric()
            metrics_to_compute = {
                "quality": self.params.compute_quality_score,
                "sharpness": self.params.compute_sharpness,
                "edge_strength": self.params.compute_edge_strength,
                "contrast": self.params.compute_contrast,
                "brightness": self.params.compute_brightness,
                "entropy": self.params.compute_entropy,
                "eyes_open": self.params.compute_eyes_open,
                "yaw": self.params.compute_yaw,
                "pitch": self.params.compute_pitch,
            }
            self._run_analysis_loop(scenes_to_process, metrics_to_compute, tracker=tracker)
            self.db.flush()

            error_count = self.db.count_errors()

            if self.cancel_event.is_set():
                return {"log": "Analysis cancelled.", "done": False}

            msg = "Analysis complete."
            if error_count > 0:
                msg += f" (⚠️ {error_count} frames failed)"
                self.logger.warning(f"Analysis completed with {error_count} errors.")
            else:
                self.logger.success(msg, extra={"output_dir": self.output_dir})

            return {"done": True, "output_dir": str(self.output_dir), "unified_log": msg}
        except Exception as e:
            self.logger.error("Analysis pipeline failed", exc_info=True, extra={"error": str(e)})
            return {"error": str(e), "done": False}

    def _filter_completed_scenes(self, scenes: list["Scene"], progress_data: dict) -> list["Scene"]:
        """Removes scenes that have already been processed when resuming."""
        completed_scenes = progress_data.get("completed_scenes", [])
        return [s for s in scenes if s.shot_id not in completed_scenes]

    def _save_progress_bulk(self, completed_scene_ids: list[int], progress_file: Path):
        """Updates the progress file with a list of completed scene IDs."""
        if not completed_scene_ids:
            return

        progress_data = {"completed_scenes": []}
        if progress_file.exists():
            try:
                with progress_file.open("r", encoding="utf-8") as f:
                    progress_data = json.load(f)
            except (IOError, json.JSONDecodeError):
                self.logger.warning(f"Could not read/parse progress file {progress_file}, overwriting.")
                progress_data = {"completed_scenes": []}

        existing_ids = set(progress_data.get("completed_scenes", []))
        existing_ids.update(completed_scene_ids)
        progress_data["completed_scenes"] = sorted(list(existing_ids))

        try:
            with progress_file.open("w", encoding="utf-8") as f:
                json.dump(progress_data, f)
        except IOError as e:
            self.logger.error(f"Failed to save progress to {progress_file}: {e}")

    def _process_reference_face(self):
        """Computes the embedding for the reference face image."""
        if not self.face_analyzer:
            return
        ref_path = Path(self.params.face_ref_img_path)
        if not ref_path.is_file():
            raise FileNotFoundError(f"Reference face image not found: {ref_path}")
        self.logger.info("Processing reference face...")
        ref_img = cv2.imread(str(ref_path))
        if ref_img is None:
            raise ValueError("Could not read reference image.")
        ref_faces = self.face_analyzer.get(ref_img)
        if not ref_faces:
            raise ValueError("No face found in reference image.")
        self.reference_embedding = max(ref_faces, key=lambda x: x.det_score).normed_embedding
        self.logger.success("Reference face processed.")

    def _run_image_folder_analysis(self, tracker: Optional["AdvancedProgressTracker"] = None) -> dict:
        """Specialized execution path for image folder inputs."""
        self.logger.info("Starting image folder analysis...")
        self.logger.info("Running pre-filter on thumbnails...")
        self.logger.info("Running full analysis on kept images...")
        self.logger.success("Image folder analysis complete.")
        metadata_path = self.output_dir / "metadata.db"
        return {"done": True, "metadata_path": str(metadata_path), "output_dir": str(self.output_dir)}

    def _run_analysis_loop(
        self,
        scenes_to_process: list["Scene"],
        metrics_to_compute: dict,
        tracker: Optional["AdvancedProgressTracker"] = None,
    ):
        """Orchestrates the parallel processing of frames for metric calculation."""
        frame_map = create_frame_map(self.thumb_dir.parent, self.logger)
        all_frame_nums_to_process = {
            fn for scene in scenes_to_process for fn in range(scene.start_frame, scene.end_frame) if fn in frame_map
        }
        image_files_to_process = [
            self.thumb_dir / frame_map[fn] for fn in sorted(list(all_frame_nums_to_process)) if frame_map.get(fn)
        ]
        self.logger.info(f"Analyzing {len(image_files_to_process)} frames")
        num_workers = (
            1 if self.params.disable_parallel else min(os.cpu_count() or 4, self.config.analysis_default_workers)
        )
        batch_size = self.config.analysis_default_batch_size
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            batches = [
                image_files_to_process[i : i + batch_size] for i in range(0, len(image_files_to_process), batch_size)
            ]
            futures = [executor.submit(self._process_batch, batch, metrics_to_compute) for batch in batches]
            for future in as_completed(futures):
                self.model_registry.check_memory_usage(self.config)
                if self.cancel_event.is_set():
                    for f in futures:
                        f.cancel()
                    break
                try:
                    num_processed = future.result()
                    if tracker and num_processed:
                        tracker.step(num_processed)
                except Exception as e:
                    self.logger.error(f"Error processing batch future: {e}")

    def _process_batch(self, batch_paths: list[Path], metrics_to_compute: dict) -> int:
        """Processes a batch of frame files."""
        for path in batch_paths:
            self._process_single_frame(path, metrics_to_compute)
        return len(batch_paths)

    def _process_single_frame(self, thumb_path: Path, metrics_to_compute: dict):
        """
        Analyzes a single frame: computes metrics, face similarity, and stores metadata.
        """
        if self.cancel_event.is_set():
            return
        if not re.search(r"frame_(\d+)", thumb_path.name):
            return
        log_context = {"file": thumb_path.name}
        try:
            thumb_image_rgb = self.thumbnail_manager.get(thumb_path)
            if thumb_image_rgb is None:
                raise ValueError("Could not read thumbnail.")
            frame, base_filename = Frame(image_data=thumb_image_rgb, frame_number=-1), thumb_path.name
            mask_meta = self.mask_metadata.get(base_filename, {})
            mask_thumb = None
            if mask_meta.get("mask_path"):
                mask_full_path = Path(mask_meta["mask_path"])
                if not mask_full_path.is_absolute():
                    mask_full_path = self.masks_dir / mask_full_path.name
                if mask_full_path.exists():
                    mask_full = cv2.imread(str(mask_full_path), cv2.IMREAD_GRAYSCALE)
                    if mask_full is not None:
                        mask_thumb = cv2.resize(
                            mask_full,
                            (thumb_image_rgb.shape[1], thumb_image_rgb.shape[0]),
                            interpolation=cv2.INTER_NEAREST,
                        )
            from core.models import QualityConfig

            quality_conf = QualityConfig(
                sharpness_base_scale=self.config.sharpness_base_scale,
                edge_strength_base_scale=self.config.edge_strength_base_scale,
                enable_niqe=(self.niqe_metric is not None and self.params.compute_niqe),
            )
            face_bbox = None
            if self.params.compute_face_sim and self.face_analyzer:
                face_bbox = self._analyze_face_similarity(frame, thumb_image_rgb)

            if any(metrics_to_compute.values()) or self.params.compute_niqe:
                frame.calculate_quality_metrics(
                    thumb_image_rgb,
                    quality_conf,
                    self.logger,
                    mask=mask_thumb,
                    niqe_metric=self.niqe_metric,
                    main_config=self.config,
                    face_landmarker=self.face_landmarker,
                    face_bbox=face_bbox,
                    metrics_to_compute=metrics_to_compute,
                )

            meta = {"filename": base_filename, "metrics": frame.metrics.model_dump()}
            if self.params.compute_face_sim:
                if frame.face_similarity_score is not None:
                    meta["face_sim"] = frame.face_similarity_score
                if frame.max_face_confidence is not None:
                    meta["face_conf"] = frame.max_face_confidence

            if self.params.compute_subject_mask_area:
                meta.update(mask_meta)

            if (
                meta.get("shot_id") is not None
                and (scene := self.scene_map.get(meta["shot_id"]))
                and scene.seed_metrics
            ):
                meta["seed_face_sim"] = scene.seed_metrics.get("best_face_sim")

            if self.params.compute_phash:
                try:
                    import imagehash

                    meta["phash"] = str(imagehash.phash(Image.fromarray(thumb_image_rgb)))
                except ImportError:
                    pass

            if "dedup_thresh" in self.params.__dict__:
                meta["dedup_thresh"] = self.params.dedup_thresh

            if frame.error:
                meta["error"] = frame.error
            if meta.get("mask_path"):
                meta["mask_path"] = Path(meta["mask_path"]).name

            meta = _to_json_safe(meta)
            self.db.insert_metadata(meta)
        except Exception as e:
            severity = "CRITICAL" if isinstance(e, (RuntimeError, MemoryError)) else "ERROR"
            self.logger.error(f"Error processing frame [{severity}]", exc_info=True, extra={**log_context, "error": e})
            self.db.insert_metadata(
                {"filename": thumb_path.name, "error": f"processing_failed: {e}", "error_severity": severity}
            )

    def _analyze_face_similarity(self, frame: "Frame", image_rgb: np.ndarray) -> Optional[list[int]]:
        """Computes face similarity and confidence against the reference face."""
        face_bbox = None
        try:
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            with self.processing_lock:
                faces = self.face_analyzer.get(image_bgr)
            if faces:
                best_face = max(faces, key=lambda x: x.det_score)
                face_bbox = best_face.bbox.astype(int)
                if self.params.enable_face_filter and self.reference_embedding is not None:
                    distance = 1 - np.dot(best_face.normed_embedding, self.reference_embedding)
                    frame.face_similarity_score, frame.max_face_confidence = (
                        1.0 - float(distance),
                        float(best_face.det_score),
                    )
        except Exception as e:
            frame.error = f"Face similarity failed: {e}"
            if "out of memory" in str(e) and torch.cuda.is_available():
                torch.cuda.empty_cache()
        return face_bbox


def _handle_extraction_uploads(event_dict: dict, config: "Config") -> dict:
    """Handles video file uploads and updates the event dictionary."""
    if event_dict.get("upload_video"):
        source = event_dict.pop("upload_video")
        dest = str(Path(config.downloads_dir) / Path(source).name)
        shutil.copy2(source, dest)
        event_dict["source_path"] = dest
    return event_dict


def _initialize_extraction_params(
    event_dict: dict, config: "Config", logger: "AppLogger"
) -> tuple[AnalysisParameters, AdvancedProgressTracker]:
    """Initializes analysis parameters and progress tracker for extraction."""
    params = AnalysisParameters.from_ui(logger, config, **event_dict)
    return params


@handle_common_errors
def execute_extraction(
    event: "ExtractionEvent",
    progress_queue: Queue,
    cancel_event: threading.Event,
    logger: "AppLogger",
    config: "Config",
    thumbnail_manager: Optional["ThumbnailManager"] = None,
    cuda_available: Optional[bool] = None,
    progress: Optional[Callable] = None,
    model_registry: Optional["ModelRegistry"] = None,
) -> Generator[dict, None, None]:
    """
    Orchestrates the frame extraction process.

    Handlers file uploads, parameter validation, and running the extraction pipeline.
    """
    try:
        event_dict = event.model_dump()
        event_dict = _handle_extraction_uploads(event_dict, config)
        params = _initialize_extraction_params(event_dict, config, logger)

        tracker = AdvancedProgressTracker(progress, progress_queue, logger, ui_stage_name="Extracting")
        pipeline = ExtractionPipeline(config, logger, params, progress_queue, cancel_event)
        result = pipeline.run(tracker=tracker)
        
        if result and result.get("done"):
            yield {
                "unified_log": "Extraction complete. You can now proceed to the next step.",
                "extracted_video_path_state": result.get("video_path", ""),
                "extracted_frames_dir_state": result["output_dir"],
                "done": True,
            }
        else:
            yield {"unified_log": f"Extraction failed. Reason: {result.get('log', 'Unknown error')}", "done": False}
    except Exception as e:
        logger.error("Extraction execution failed", exc_info=True)
        yield {"unified_log": f"[ERROR] Extraction failed unexpectedly: {e}", "done": False}


def _handle_pre_analysis_uploads(event_dict: dict, config: "Config") -> dict:
    """Handles face reference image uploads and updates the event dictionary."""
    if event_dict.get("face_ref_img_upload"):
        ref_upload = event_dict.pop("face_ref_img_upload")
        dest = Path(config.downloads_dir) / Path(ref_upload).name
        shutil.copy2(ref_upload, dest)
        event_dict["face_ref_img_path"] = str(dest)
    return event_dict


def _initialize_pre_analysis_params(
    event_dict: dict, config: "Config", logger: "AppLogger"
) -> tuple[AnalysisParameters, Path]:
    """Initializes analysis parameters and ensures output directory exists."""
    params = AnalysisParameters.from_ui(logger, config, **event_dict)
    output_dir = Path(params.output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save run configuration
    config_to_save = {k: v for k, v in event_dict.items() if k != "face_ref_img_upload"}
    (output_dir / "run_config.json").write_text(json.dumps(config_to_save, indent=4))

    return params, output_dir


def _load_scenes(output_dir: Path) -> list[Scene]:
    """Loads scenes from scenes.json."""
    scenes_path = output_dir / "scenes.json"
    if not scenes_path.exists():
        raise FileNotFoundError("scenes.json not found. Run extraction first.")
    
    with scenes_path.open("r", encoding="utf-8") as f:
        scenes_data = json.load(f)
    
    return [
        Scene(shot_id=i, start_frame=s, end_frame=e)
        for i, (s, e) in enumerate(scenes_data)
    ]


class PreAnalysisPipeline(Pipeline):
    """Pipeline for pre-analyzing scenes (best frame selection, seeding)."""

    def __init__(
        self,
        config: "Config",
        logger: "AppLogger",
        params: "AnalysisParameters",
        progress_queue: Queue,
        cancel_event: threading.Event,
        thumbnail_manager: "ThumbnailManager",
        model_registry: "ModelRegistry",
    ):
        super().__init__(config, logger, params, progress_queue, cancel_event)
        self.thumbnail_manager = thumbnail_manager
        self.model_registry = model_registry
        self.output_dir = Path(self.params.output_folder)

    def run(self, scenes: list[Scene], tracker: Optional["AdvancedProgressTracker"] = None) -> list[Scene]:
        """Runs pre-analysis for a list of scenes."""
        models = initialize_analysis_models(self.params, self.config, self.logger, self.model_registry)
        
        is_folder_mode = not self.params.video_path
        niqe_metric = self._initialize_niqe_if_needed(models["device"], is_folder_mode)

        masker = SubjectMasker(
            self.params,
            self.progress_queue,
            self.cancel_event,
            self.config,
            face_analyzer=models["face_analyzer"],
            reference_embedding=models["ref_emb"],
            niqe_metric=niqe_metric,
            thumbnail_manager=self.thumbnail_manager,
            logger=self.logger,
            face_landmarker=models["face_landmarker"],
            device=models["device"],
            model_registry=self.model_registry,
        )
        masker.frame_map = masker._create_frame_map(str(self.output_dir))
        
        previews_dir = self.output_dir / "previews"
        previews_dir.mkdir(exist_ok=True)

        for scene in scenes:
            if self.cancel_event.is_set():
                break
            
            if tracker:
                tracker.step(1, desc=f"Scene {scene.shot_id}")
            
            self._process_single_scene(scene, masker, previews_dir, is_folder_mode)

        save_scene_seeds(scenes, str(self.output_dir), self.logger)
        return scenes

    def _initialize_niqe_if_needed(self, device: str, is_folder_mode: bool):
        """Lazy initialization of NIQE for seeding."""
        if (not is_folder_mode and 
            self.params.pre_analysis_enabled and 
            self.params.primary_seed_strategy != "🧑‍🤝‍🧑 Find Prominent Person"):
            try:
                import pyiqa
                return pyiqa.create_metric("niqe", device=device)
            except (ImportError, Exception):
                pass
        return None

    def _process_single_scene(self, scene: Scene, masker: SubjectMasker, previews_dir: Path, is_folder_mode: bool):
        """Processes a single scene: selects best frame, generates seed, and preview."""
        if is_folder_mode:
            scene.best_frame = scene.start_frame
        elif not scene.best_frame:
            masker._select_best_frame_in_scene(scene, str(self.output_dir))

        fname = masker.frame_map.get(scene.best_frame)
        if not fname:
            return

        thumb_rgb = self.thumbnail_manager.get(self.output_dir / "thumbs" / f"{Path(fname).stem}.webp")
        if thumb_rgb is None:
            return

        # Get seed bounding box and details
        bbox, details = masker.get_seed_for_frame(thumb_rgb, seed_config=scene.seed_config or self.params, scene=scene)
        scene.seed_result = {"bbox": bbox, "details": details}

        # Handle mask if enabled
        mask = None
        if bbox and self.params.enable_subject_mask:
            mask = masker.get_mask_for_bbox(thumb_rgb, bbox)
            if mask is not None and mask.size > 0:
                h, w = mask.shape[:2]
                area = h * w
                scene.seed_result["details"]["mask_area_pct"] = (np.sum(mask > 0) / area * 100) if area > 0 else 0.0

        # Generate preview overlay
        from core.utils import render_mask_overlay
        overlay_rgb = (
            render_mask_overlay(thumb_rgb, mask, 0.6, logger=self.logger)
            if mask is not None
            else masker.draw_bbox(thumb_rgb, bbox)
        )
        
        preview_path = previews_dir / f"scene_{scene.shot_id:05d}.jpg"
        Image.fromarray(overlay_rgb).save(preview_path)
        scene.preview_path, scene.status = str(preview_path), "included"


@handle_common_errors
def execute_pre_analysis(
    event: "PreAnalysisEvent",
    progress_queue: Queue,
    cancel_event: threading.Event,
    logger: "AppLogger",
    config: "Config",
    thumbnail_manager: "ThumbnailManager",
    cuda_available: bool,
    progress: Optional[Callable] = None,
    model_registry: "ModelRegistry" = None,
) -> Generator[dict, None, None]:
    """
    Orchestrates the pre-analysis phase (scene detection, subject seeding).
    """
    try:
        tracker = AdvancedProgressTracker(progress, progress_queue, logger, ui_stage_name="Pre-Analysis")
        
        event_dict = event.model_dump()
        event_dict = _handle_pre_analysis_uploads(event_dict, config)
        params, output_dir = _initialize_pre_analysis_params(event_dict, config, logger)
        
        try:
            scenes = _load_scenes(output_dir)
        except FileNotFoundError as e:
            yield {"unified_log": f"[ERROR] {e}", "done": False}
            return

        is_folder_mode = not params.video_path
        tracker.start(len(scenes), desc="Analyzing Scenes" if is_folder_mode else "Pre-analyzing Scenes")

        pipeline = PreAnalysisPipeline(
            config, logger, params, progress_queue, cancel_event, thumbnail_manager, model_registry
        )
        processed_scenes = pipeline.run(scenes, tracker=tracker)

        tracker.done_stage("Pre-analysis complete")
        
        final_yield = {
            "unified_log": "Pre-analysis complete. Review scenes in the next tab.",
            "scenes": [s.model_dump() for s in processed_scenes],
            "output_dir": str(output_dir),
            "done": True,
            "seeding_results_column": gr.update(visible=True),
            "propagation_group": gr.update(visible=True),
        }
        if params.face_ref_img_path:
            final_yield["final_face_ref_path"] = params.face_ref_img_path
        yield final_yield

    except Exception as e:
        logger.error("Pre-analysis execution failed", exc_info=True)
        yield {"unified_log": f"[ERROR] Pre-analysis failed unexpectedly: {e}", "done": False}
    except Exception as e:
        logger.error("Pre-analysis execution failed", exc_info=True)
        yield {"unified_log": f"[ERROR] Pre-analysis failed unexpectedly: {e}", "done": False}


def validate_session_dir(path: Union[str, Path]) -> tuple[Optional[Path], Optional[str]]:
    """Checks if the provided path is a valid session directory."""
    try:
        p = Path(path).expanduser().resolve()
        return (
            p if p.exists() and p.is_dir() else None,
            None if p.exists() and p.is_dir() else f"Session directory does not exist: {p}",
        )
    except Exception as e:
        return None, f"Invalid session path: {e}"


def execute_session_load(event: "SessionLoadEvent", logger: "AppLogger") -> dict:
    """
    Loads session state from disk.

    Verifies the directory structure and loads configuration, scenes, and metadata.
    """
    if not event.session_path or not event.session_path.strip():
        logger.error("No session path provided.", component="session_loader")
        return {"error": "Please enter a path to a session directory."}
    session_path, error = validate_session_dir(event.session_path)
    if error:
        logger.error(f"Invalid session path provided: {event.session_path}", component="session_loader")
        return {"error": error}

    config_path = session_path / "run_config.json"
    scene_seeds_path = session_path / "scene_seeds.json"
    metadata_path = session_path / "metadata.db"

    logger.info("Start Load Session", component="session_loader")
    try:
        if not config_path.exists():
            return {"error": f"Could not find 'run_config.json' in {session_path}."}
        try:
            run_config = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            return {"error": f"run_config.json is invalid: {e}"}

        scenes_data = []
        scenes_json_path = session_path / "scenes.json"
        if scenes_json_path.exists():
            try:
                scenes_data = [
                    {"shot_id": i, "start_frame": s, "end_frame": e}
                    for i, (s, e) in enumerate(json.loads(scenes_json_path.read_text(encoding="utf-8")))
                ]
            except Exception as e:
                return {"error": f"Failed to read scenes.json: {e}"}

        if scene_seeds_path.exists():
            try:
                seeds_lookup = {int(k): v for k, v in json.loads(scene_seeds_path.read_text(encoding="utf-8")).items()}
                for scene in scenes_data:
                    if (shot_id := scene.get("shot_id")) in seeds_lookup:
                        rec = seeds_lookup[shot_id]
                        rec["best_frame"] = rec.get("best_frame", rec.get("best_seed_frame"))
                        scene.update(rec)
                    scene.setdefault("status", "included")
            except Exception as e:
                logger.warning(f"Failed to parse scene_seeds.json: {e}")

        logger.success("Session loaded successfully", component="session_loader")
        return {
            "success": True,
            "session_path": str(session_path),
            "run_config": run_config,
            "scenes": scenes_data,
            "metadata_exists": metadata_path.exists(),
        }

    except Exception as e:
        logger.error(f"Failed to load session: {e}", component="session_loader", exc_info=True)
        return {"error": f"Failed to load session: {e}"}


def _load_analysis_scenes(scenes_data: list, is_folder_mode: bool, include_only: bool = True) -> list[Scene]:
    """Converts raw scene data to Scene objects, optionally filtering by status."""
    scene_fields = set(Scene.model_fields.keys())
    return [
        Scene(**{k: v for k, v in s.items() if k in scene_fields})
        for s in scenes_data
        if not include_only or is_folder_mode or s.get("status") == "included"
    ]


def _initialize_analysis_pipeline(
    config: "Config",
    logger: "AppLogger",
    params: AnalysisParameters,
    progress_queue: Queue,
    cancel_event: threading.Event,
    thumbnail_manager: "ThumbnailManager",
    model_registry: "ModelRegistry",
) -> AnalysisPipeline:
    """Creates and returns an AnalysisPipeline instance."""
    return AnalysisPipeline(
        config, logger, params, progress_queue, cancel_event, thumbnail_manager, model_registry
    )


def execute_propagation(
    event: PropagationEvent,
    progress_queue: Queue,
    cancel_event: threading.Event,
    logger: AppLogger,
    config: Config,
    thumbnail_manager,
    cuda_available,
    progress=None,
    model_registry: "ModelRegistry" = None,
) -> Generator[dict, None, None]:
    """Orchestrates the mask propagation stage."""
    try:
        params = AnalysisParameters.from_ui(logger, config, **event.analysis_params.model_dump())
        is_folder_mode = not params.video_path
        
        scenes_to_process = _load_analysis_scenes(event.scenes, is_folder_mode)
        if not scenes_to_process:
            yield {"unified_log": "No scenes were included for processing. Nothing to do."}
            return

        tracker = AdvancedProgressTracker(progress, progress_queue, logger, ui_stage_name="Analysis")
        if is_folder_mode:
            tracker.start(len(scenes_to_process), desc="Analyzing Images")
        else:
            video_info = VideoManager.get_video_info(params.video_path)
            totals = estimate_totals(params, video_info, scenes_to_process)
            tracker.start(totals.get("propagation", 0) + len(scenes_to_process), desc="Propagating Masks & Analyzing")

        pipeline = _initialize_analysis_pipeline(
            config, logger, params, progress_queue, cancel_event, thumbnail_manager, model_registry
        )
        
        result = pipeline.run_full_analysis(scenes_to_process, tracker=tracker)
        if result and result.get("done"):
            masks_dir = Path(result["output_dir"]) / "masks"
            mask_files = list(masks_dir.glob("*.png")) if masks_dir.exists() else []
            if not mask_files:
                yield {
                    "unified_log": "❌ Propagation failed - no masks were generated. Check SAM3 model logs.",
                    "done": False,
                }
                return
            yield {
                "unified_log": f"✅ Propagation complete. Generated {len(mask_files)} masks.",
                "output_dir": result["output_dir"],
                "done": True,
            }
        else:
            yield {
                "unified_log": f"❌ Propagation failed. Reason: {result.get('error', 'Unknown error')}",
                "done": False,
            }
    except Exception as e:
        logger.error("Propagation execution failed", exc_info=True)
        yield {"unified_log": f"[ERROR] Propagation failed unexpectedly: {e}", "done": False}


@handle_common_errors
def execute_analysis(
    event: PropagationEvent,
    progress_queue: Queue,
    cancel_event: threading.Event,
    logger: AppLogger,
    config: Config,
    thumbnail_manager,
    cuda_available,
    progress=None,
    model_registry: "ModelRegistry" = None,
) -> Generator[dict, None, None]:
    """Orchestrates the frame analysis stage."""
    try:
        params = AnalysisParameters.from_ui(logger, config, **event.analysis_params.model_dump())
        is_folder_mode = not params.video_path
        
        scenes_to_process = _load_analysis_scenes(event.scenes, is_folder_mode)
        if not scenes_to_process:
            yield {"unified_log": "No scenes to analyze. Nothing to do."}
            return

        video_info = VideoManager.get_video_info(params.video_path)
        tracker = AdvancedProgressTracker(progress, progress_queue, logger, ui_stage_name="Analyzing")
        tracker.start(sum(s.end_frame - s.start_frame for s in scenes_to_process), desc="Analyzing Frames")

        pipeline = _initialize_analysis_pipeline(
            config, logger, params, progress_queue, cancel_event, thumbnail_manager, model_registry
        )
        
        result = pipeline.run_analysis_only(scenes_to_process, tracker=tracker)
        if result and result.get("done"):
            yield {
                "unified_log": "Analysis complete. You can now proceed to the Filtering & Export tab.",
                "output_dir": result["output_dir"],
                "done": True,
            }
        else:
            yield {"unified_log": f"❌ Analysis failed. Reason: {result.get('error', 'Unknown error')}", "done": False}
    except Exception as e:
        logger.error("Analysis execution failed", exc_info=True)
        yield {"unified_log": f"[ERROR] Analysis failed unexpectedly: {e}", "done": False}
