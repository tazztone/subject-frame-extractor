from __future__ import annotations

import csv
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import cv2

if TYPE_CHECKING:
    from core.config import Config
    from core.logger import AppLogger

from core.events import ExportEvent
from core.filtering import apply_all_filters_vectorized
from core.operators.crop import crop_image_with_subject
from core.scene_utils.ffmpeg import perform_ffmpeg_export
from core.utils import _to_json_safe
from core.xmp_writer import write_xmp_sidecar


def _rename_exported_frames(export_dir: Path, frames_to_extract: list, fn_to_orig_map: dict, logger: "AppLogger"):
    logger.info("Renaming extracted frames to match original filenames...")
    orig_to_filename_map = {v: k for k, v in fn_to_orig_map.items()}
    plan = []
    for i, orig_frame_num in enumerate(frames_to_extract):
        sequential_filename = f"frame_{i + 1:06d}.png"
        target_filename = orig_to_filename_map.get(orig_frame_num)
        if not target_filename:
            continue
        src = export_dir / sequential_filename
        dst = export_dir / target_filename
        if src != dst:
            plan.append((src, dst))

    temp_map = {}
    for i, (src, _) in enumerate(plan):
        if not src.exists():
            continue
        tmp = export_dir / f"__tmp_{i:06d}__{src.name}"
        j = i
        while tmp.exists():
            j += 1
            tmp = export_dir / f"__tmp_{j:06d}__{src.name}"
        try:
            src.rename(tmp)
            temp_map[src] = tmp
        except FileNotFoundError:
            logger.warning(f"Could not find {src.name} to rename.", extra={"target": tmp.name})

    for src, dst in plan:
        tmp = temp_map.get(src)
        if tmp and tmp.exists():
            if dst.exists():
                stem, ext = dst.stem, dst.suffix
                k, alt = 1, export_dir / f"{stem} (1){ext}"
                while alt.exists():
                    k += 1
                    alt = export_dir / f"{stem} ({k}){ext}"
                dst = alt
            try:
                tmp.rename(dst)
            except FileNotFoundError:
                logger.warning(f"Could not find temp file {tmp.name} to rename.", extra={"target": dst.name})


def _export_metadata(kept_frames: list, export_dir: Path, logger: "AppLogger"):
    logger.info("Exporting metadata...")
    safe_frames = [_to_json_safe(f) for f in kept_frames]

    # JSON export
    try:
        with (export_dir / "metadata.json").open("w", encoding="utf-8") as f:
            json.dump(safe_frames, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save metadata.json: {e}")

    # CSV export
    try:
        if not safe_frames:
            return
        all_keys = set()
        for f in safe_frames:
            all_keys.update(f.keys())
        fieldnames = sorted(list(all_keys))

        # Ensure common important keys are first
        priority_keys = ["filename", "frame_number", "timestamp", "score", "face_sim"]
        fieldnames = [k for k in priority_keys if k in fieldnames] + [k for k in fieldnames if k not in priority_keys]

        with (export_dir / "metadata.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(safe_frames)
    except Exception as e:
        logger.error(f"Failed to save metadata.csv: {e}")


def _crop_exported_frames(
    kept_frames: list,
    export_dir: Path,
    crop_ars: str,
    crop_padding: int,
    masks_root: Path,
    logger: "AppLogger",
    cancel_event,
) -> int:
    logger.info("Starting crop export...")
    crop_dir = export_dir / "cropped"
    crop_dir.mkdir(exist_ok=True)

    try:
        aspect_ratios = [
            (ar_str.replace(":", "x"), float(ar_str.split(":")[0]) / float(ar_str.split(":")[1]))
            for ar_str in crop_ars.split(",")
            if ":" in ar_str
        ]
    except (ValueError, ZeroDivisionError):
        raise ValueError("Invalid aspect ratio format.")

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

            if frame_img is None or mask_img is None:
                continue

            cropped_img, ar_label = crop_image_with_subject(frame_img, mask_img, aspect_ratios, padding_factor)

            if cropped_img is not None and cropped_img.size > 0:
                out_name = f"{Path(filename).stem}_crop_{ar_label}.png"
                cv2.imwrite(str(crop_dir / out_name), cropped_img)
                num_cropped += 1

        except Exception:
            logger.error(f"Failed to crop frame {frame_meta.get('filename')}", exc_info=True)

    return num_cropped


def export_kept_frames(
    event: ExportEvent, config: "Config", logger: "AppLogger", thumbnail_manager, cancel_event
) -> str:
    if not event.all_frames_data:
        return "No metadata to export."

    is_video = bool(event.video_path and Path(event.video_path).exists() and not Path(event.video_path).is_dir())
    out_root = Path(event.output_dir)

    try:
        filters = event.filter_args.copy()
        # Ensure we have data for the filters
        filters.update(
            {
                "face_sim_enabled": any("face_sim" in f for f in event.all_frames_data),
                "mask_area_enabled": any("mask_area_pct" in f for f in event.all_frames_data),
                "enable_dedup": any("phash" in f for f in event.all_frames_data),
            }
        )

        kept, _, _, _ = apply_all_filters_vectorized(
            event.all_frames_data, filters, config, output_dir=event.output_dir
        )

        if not kept:
            return "No frames kept after filtering. Nothing to export."

        export_dir = out_root.parent / f"{out_root.name}_exported_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        export_dir.mkdir(exist_ok=True, parents=True)

        if is_video:
            frame_map_path = out_root / "frame_map.json"
            if not frame_map_path.exists():
                return "[ERROR] frame_map.json not found. Cannot export."
            with frame_map_path.open("r", encoding="utf-8") as f:
                frame_map_list = json.load(f)

            sample_name = next((f["filename"] for f in kept if "filename" in f), None)
            analyzed_ext = Path(sample_name).suffix if sample_name else ".webp"
            fn_to_orig_map = {f"frame_{i + 1:06d}{analyzed_ext}": orig for i, orig in enumerate(sorted(frame_map_list))}

            frames_to_extract = sorted(
                [fn_to_orig_map.get(f["filename"]) for f in kept if f.get("filename") in fn_to_orig_map]
            )
            frames_to_extract = [n for n in frames_to_extract if n is not None]

            if not frames_to_extract:
                return "No frames to extract."

            success, stderr = perform_ffmpeg_export(event.video_path, frames_to_extract, export_dir, logger)
            if not success:
                return f"Error during export: FFmpeg failed. Check logs for details:\n{stderr}"

            _rename_exported_frames(export_dir, frames_to_extract, fn_to_orig_map, logger)
        else:
            # Folder mode: Copy original files
            source_map_path = out_root / "source_map.json"
            source_map = {}
            if source_map_path.exists():
                with source_map_path.open("r", encoding="utf-8") as f:
                    source_map = json.load(f)

            num_copied = 0
            for frame_meta in kept:
                filename = frame_meta.get("filename")
                if filename in source_map:
                    src_path = Path(source_map[filename])
                    if src_path.exists():
                        shutil.copy2(src_path, export_dir / src_path.name)
                        num_copied += 1

                        # Handle XMP Export
                        if event.enable_xmp_export:
                            score = frame_meta.get("score", 0.0)
                            # Simple rating logic: 0-100 -> 0-5 stars
                            rating = int(min(5, max(0, score / 20)))
                            label = "Green"
                            write_xmp_sidecar(src_path, rating, label)

            logger.info(f"Copied {num_copied} original files to export directory.")

        _export_metadata(kept, export_dir, logger)

        if event.enable_crop:
            try:
                num_cropped = _crop_exported_frames(
                    kept, export_dir, event.crop_ars, event.crop_padding, out_root / "masks", logger, cancel_event
                )
                logger.info(f"Cropping complete. Saved {num_cropped} cropped images.")
            except ValueError as e:
                return str(e)

        return f"Exported {len(kept)} items to {export_dir.name}."
    except Exception as e:
        logger.error("Error during export process", exc_info=True)
        return f"Error during export: {e}"


def dry_run_export(event: ExportEvent, config: "Config") -> str:
    if not event.all_frames_data:
        return "No metadata to export."
    if not event.video_path or not Path(event.video_path).exists():
        return "[ERROR] Original video path is required for export."
    out_root = Path(event.output_dir)
    try:
        filters = event.filter_args.copy()
        filters.update(
            {
                "face_sim_enabled": any("face_sim" in f for f in event.all_frames_data),
                "mask_area_enabled": any("mask_area_pct" in f for f in event.all_frames_data),
                "enable_dedup": any("phash" in f for f in event.all_frames_data),
            }
        )

        kept, _, _, _ = apply_all_filters_vectorized(
            event.all_frames_data, filters, config, output_dir=event.output_dir
        )

        if not kept:
            return "No frames kept after filtering. Nothing to export."

        frame_map_path = out_root / "frame_map.json"
        if not frame_map_path.exists():
            return "[ERROR] frame_map.json not found. Cannot export."
        with frame_map_path.open("r", encoding="utf-8") as f:
            frame_map_list = json.load(f)

        sample_name = next((f["filename"] for f in kept if "filename" in f), None)
        analyzed_ext = Path(sample_name).suffix if sample_name else ".webp"
        fn_to_orig_map = {f"frame_{i + 1:06d}{analyzed_ext}": orig for i, orig in enumerate(sorted(frame_map_list))}

        frames_to_extract = sorted(
            [fn_to_orig_map.get(f["filename"]) for f in kept if f.get("filename") in fn_to_orig_map]
        )
        frames_to_extract = [n for n in frames_to_extract if n is not None]

        if not frames_to_extract:
            return "No frames to extract."

        select_filter = f"select='{'+'.join([f'eq(n,{fn})' for fn in frames_to_extract])}'"
        export_dir = out_root.parent / f"{out_root.name}_exported_DATE"
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(event.video_path),
            "-vf",
            select_filter,
            "-vsync",
            "vfr",
            str(export_dir / "frame_%06d.png"),
        ]
        return f"Dry Run: {len(frames_to_extract)} frames to be exported.\n\nFFmpeg command:\n{' '.join(cmd)}"
    except Exception as e:
        return f"Error during dry run: {e}"
