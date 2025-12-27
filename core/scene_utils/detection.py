"""
Scene detection and thumbnail generation utilities.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import cv2
from PIL import Image
from scenedetect import ContentDetector, detect

if TYPE_CHECKING:
    from core.config import Config
    from core.logger import AppLogger
    from core.models import AnalysisParameters
    from core.progress import AdvancedProgressTracker


def run_scene_detection(video_path: str, output_dir: Path, logger: "AppLogger") -> list:
    """
    Detect scene changes in a video using PySceneDetect.

    Args:
        video_path: Path to the video file
        output_dir: Directory to save scenes.json
        logger: Application logger

    Returns:
        List of (start_frame, end_frame) tuples for each scene
    """
    logger.info("Detecting scenes...", component="video")
    try:
        scene_list = detect(str(video_path), ContentDetector())
        shots = [(s.get_frames(), e.get_frames()) for s, e in scene_list] if scene_list else []
        with (output_dir / "scenes.json").open("w", encoding="utf-8") as f:
            json.dump(shots, f)
        logger.success(f"Found {len(shots)} scenes.", component="video")
        return shots
    except Exception:
        logger.error("Scene detection failed.", component="video", exc_info=True)
        return []


def make_photo_thumbs(
    image_paths: list[Path],
    out_dir: Path,
    params: "AnalysisParameters",
    cfg: "Config",
    logger: "AppLogger",
    tracker: Optional["AdvancedProgressTracker"] = None,
) -> dict:
    """
    Generate thumbnails for a list of images.

    Args:
        image_paths: List of paths to source images
        out_dir: Output directory for thumbnails
        params: Analysis parameters containing thumb_megapixels
        cfg: Application configuration
        logger: Application logger
        tracker: Optional progress tracker

    Returns:
        Dictionary mapping frame numbers to thumbnail filenames
    """
    thumbs_dir = out_dir / "thumbs"
    thumbs_dir.mkdir(parents=True, exist_ok=True)
    target_area = params.thumb_megapixels * 1_000_000
    frame_map, image_manifest = {}, {}

    if tracker:
        tracker.start(len(image_paths), desc="Generating thumbnails")

    for i, img_path in enumerate(image_paths, start=1):
        if tracker and tracker.pause_event.is_set():
            tracker.step()
        try:
            bgr = cv2.imread(str(img_path))
            if bgr is None:
                logger.warning(f"Could not read image file: {img_path}")
                continue

            h, w = bgr.shape[:2]
            scale = math.sqrt(target_area / float(max(1, w * h)))
            if scale < 1.0:
                new_w, new_h = int((w * scale) // 2 * 2), int((h * scale) // 2 * 2)
                bgr = cv2.resize(bgr, (max(2, new_w), max(2, new_h)), interpolation=cv2.INTER_AREA)

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            out_name = f"frame_{i:06d}.webp"
            out_path = thumbs_dir / out_name
            Image.fromarray(rgb).save(out_path, format="WEBP", quality=cfg.ffmpeg_thumbnail_quality)

            frame_map[i] = out_name
            image_manifest[i] = str(img_path.resolve())
        except Exception:
            logger.error(f"Failed to process image {img_path}", exc_info=True)
        finally:
            if tracker:
                tracker.step()

    (out_dir / "frame_map.json").write_text(json.dumps(frame_map, indent=2), encoding="utf-8")
    (out_dir / "image_manifest.json").write_text(json.dumps(image_manifest, indent=2), encoding="utf-8")
    if tracker:
        tracker.done_stage("Thumbnails generated")
    return frame_map
