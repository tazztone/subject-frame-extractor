from __future__ import annotations

import csv
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import cv2

if TYPE_CHECKING:
    from core.config import Config
    from core.logger import AppLogger

from core.events import ExportEvent
from core.filtering import apply_all_filters_vectorized
from core.operators.crop import crop_image_with_subject
from core.scene_utils.ffmpeg import perform_ffmpeg_export
from core.utils import _to_json_safe


def export_kept_frames(
    event: ExportEvent,
    config: "Config",
    logger: "AppLogger",
    progress_queue: Optional[any] = None,
    cancel_event: Optional[any] = None,
) -> str:
    """
    Main export entry point.
    Filters frames and exports them using FFmpeg or by copying/cropping.
    """
    if not event.all_frames_data:
        return "No metadata to export."

    if not event.video_path and not event.output_dir:
        return "[ERROR] Video path or output directory required."

    # 1. Apply Filters
    kept_frames, rejected_frames, stats, reasons = apply_all_filters_vectorized(
        event.all_frames_data, event.filter_args, config, output_dir=event.output_dir
    )

    if not kept_frames:
        return "No frames kept after filtering."

    # 2. Determine Export Mode
    is_folder_mode = not event.video_path

    out_dir = Path(event.output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_dir = out_dir / f"exported_{timestamp}"
    export_dir.mkdir(parents=True, exist_ok=True)

    if is_folder_mode:
        source_map_path = out_dir / "source_map.json"
        if not source_map_path.exists():
            source_map_path = out_dir / "frame_map.json"

        if not source_map_path.exists():
            return "[ERROR] source_map.json not found in output directory. Cannot map frames back to source files."

        with open(source_map_path, "r", encoding="utf-8") as f:
            source_map = json.load(f)

        num_exported = 0
        for frame in kept_frames:
            if cancel_event and cancel_event.is_set():
                break
            fname = frame["filename"]
            src_path = source_map.get(fname)
            if not src_path:
                logger.warning(f"Could not find source path for {fname}")
                continue

            src_p = Path(src_path)
            if src_p.exists():
                shutil.copy2(src_p, export_dir / fname)
                num_exported += 1

        logger.info(f"Exported {num_exported} frames to {export_dir}")

    else:
        video_p = Path(event.video_path)
        if not video_p.exists():
            return f"[ERROR] Video file not found: {event.video_path}"

        frame_map_path = out_dir / "frame_map.json"
        if not frame_map_path.exists():
            return "[ERROR] frame_map.json not found in output directory. Extraction required first."

        with open(frame_map_path, "r", encoding="utf-8") as f:
            frame_nums = json.load(f)

        fn_to_orig = {f"frame_{i + 1:06d}.webp": num for i, num in enumerate(frame_nums)}
        fn_to_orig.update({f"frame_{i + 1:06d}.png": num for i, num in enumerate(frame_nums)})

        frames_to_extract = []
        for f in kept_frames:
            orig_num = fn_to_orig.get(f["filename"])
            if orig_num is not None:
                frames_to_extract.append(orig_num)

        success, msg = perform_ffmpeg_export(str(video_p), frames_to_extract, export_dir, logger)
        if not success:
            return f"FFmpeg failed: {msg}"

        _rename_exported_frames(export_dir, frames_to_extract, fn_to_orig, logger)

    # 3. Handle Cropping
    if event.enable_crop and event.crop_ars:
        masks_root = out_dir / "masks"
        num_cropped = _crop_exported_frames(
            kept_frames, export_dir, event.crop_ars, event.crop_padding, masks_root, logger, cancel_event
        )
        logger.info(f"Created {num_cropped} cropped versions.")

    # 4. Handle Metadata
    _export_metadata(kept_frames, export_dir, logger)

    # 5. Handle XMP sidecars
    if event.enable_xmp_export:
        from core.xmp_writer import export_xmps_for_photos

        photos_to_xmp = []
        for f in kept_frames:
            p = export_dir / f["filename"]
            if p.exists():
                photo_meta = f.copy()
                photo_meta["source"] = p
                photos_to_xmp.append(photo_meta)

        if photos_to_xmp:
            export_xmps_for_photos(photos_to_xmp)

    return f"Export Complete. Results in: {export_dir}"


def _crop_exported_frames(
    kept_frames: list,
    export_dir: Path,
    crop_ars: str,
    crop_padding: int,
    masks_root: Path,
    logger: "AppLogger",
    cancel_event: any,
) -> int:
    logger.info("Starting crop export...")
    crop_dir = export_dir / "cropped"
    crop_dir.mkdir(exist_ok=True)

    aspect_ratios = []
    try:
        if crop_ars:
            for ar_str in crop_ars.split(","):
                ar_str = ar_str.strip()
                if not ar_str:
                    continue
                if ":" not in ar_str:
                    raise ValueError("Invalid aspect ratio format.")
                parts = ar_str.split(":")
                aspect_ratios.append((ar_str.replace(":", "x"), float(parts[0]) / float(parts[1])))
    except (ValueError, ZeroDivisionError, IndexError):
        raise ValueError("Invalid aspect ratio format.")

    if not aspect_ratios:
        return 0

    num_cropped = 0
    padding_factor = 1.0 + (crop_padding / 100.0)

    for frame_meta in kept_frames:
        if cancel_event.is_set():
            break
        try:
            filename = frame_meta["filename"]
            full_frame_path = export_dir / filename
            if not full_frame_path.exists():
                continue

            mask_name = frame_meta.get("mask_path", "")
            if not mask_name:
                continue

            mask_path = masks_root / mask_name
            if not mask_path.exists():
                continue

            frame_img = cv2.imread(str(full_frame_path))
            mask_img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

            if frame_img is None:
                logger.error(f"Could not read frame for cropping: {full_frame_path}")
                continue
            if mask_img is None:
                logger.error(f"Could not read mask for cropping: {mask_path}")
                continue

            cropped_img, ar_label = crop_image_with_subject(frame_img, mask_img, aspect_ratios, padding_factor)

            if cropped_img is not None and cropped_img.size > 0:
                out_name = f"{Path(filename).stem}_crop_{ar_label}.png"
                cv2.imwrite(str(crop_dir / out_name), cropped_img)
                num_cropped += 1

        except Exception as e:
            logger.error(f"Failed to crop {frame_meta.get('filename')}: {e}")

    return num_cropped


def _rename_exported_frames(export_dir: Path, frames_to_extract: list, fn_to_orig: dict, logger: "AppLogger"):
    """Renames FFmpeg output (frame_000001.png) to more descriptive original names if possible."""
    extracted_files = sorted(export_dir.glob("frame_*.webp"))
    if not extracted_files:
        extracted_files = sorted(export_dir.glob("frame_*.png"))

    if len(extracted_files) != len(frames_to_extract):
        logger.warning(
            f"Extracted file count ({len(extracted_files)}) mismatch with expected ({len(frames_to_extract)})"
        )
        return

    orig_to_final = {v: k for k, v in fn_to_orig.items()}

    for i, extracted_p in enumerate(extracted_files):
        orig_num = frames_to_extract[i]
        final_name = orig_to_final.get(orig_num)

        if final_name and final_name != extracted_p.name:
            target_p = export_dir / final_name
            if target_p.exists():
                stem, ext = target_p.stem, target_p.suffix
                target_p = export_dir / f"{stem} (1){ext}"

            try:
                extracted_p.rename(target_p)
            except Exception as e:
                logger.warning(f"Failed to rename {extracted_p.name} to {final_name}: {e}")


def _export_metadata(kept_frames: list, export_dir: Path, logger: "AppLogger"):
    """Exports frame metadata to JSON and CSV formats."""
    if not kept_frames:
        with open(export_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump([], f)
        return

    meta_path = export_dir / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(_to_json_safe(kept_frames), f, indent=4)

    csv_path = export_dir / "metadata.csv"
    try:
        keys = []
        priority = ["filename", "frame_number", "timestamp", "score", "quality_score", "face_sim"]

        all_keys = set()
        for f in kept_frames:
            all_keys.update(f.keys())

        for k in priority:
            if k in all_keys:
                keys.append(k)
                all_keys.remove(k)
        keys.extend(sorted(list(all_keys)))

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(_to_json_safe(kept_frames))

    except Exception as e:
        logger.error(f"Failed to export CSV metadata: {e}")


def dry_run_export(event: ExportEvent, config: "Config", logger: "AppLogger") -> str:
    """Simulates an export and returns a summary of actions."""
    if not event.all_frames_data:
        return "No metadata to export."

    kept_frames, rejected_frames, stats, reasons = apply_all_filters_vectorized(
        event.all_frames_data, event.filter_args, config
    )

    if not kept_frames:
        return "No frames kept after filtering."

    lines = [f"Dry Run: {len(kept_frames)} frames would be exported."]

    if not event.video_path:
        lines.append("Mode: Folder (copying existing frames)")
    else:
        video_p = Path(event.video_path)
        if not video_p.exists():
            lines.append(f"[ERROR] Original video path is required and was not found: {event.video_path}")
        else:
            lines.append(f"Source Video: {video_p.name}")
            lines.append(f"Command: ffmpeg -i ... (extracting {len(kept_frames)} frames)")

    if event.enable_crop:
        lines.append(f"Action: Generate crops for {event.crop_ars}")

    if event.enable_xmp_export:
        lines.append("Action: Write XMP sidecar files")

    return "\n".join(lines)
