from __future__ import annotations

import csv
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import cv2
import numpy as np

if TYPE_CHECKING:
    from core.config import Config
    from core.logger import AppLogger

from core.events import ExportEvent
from core.filtering import apply_all_filters_vectorized
from core.utils import _to_json_safe
from core.xmp_writer import write_xmp_sidecar


# TODO: Add parallel frame export using multi-threading
# TODO: Support multiple output formats (JPEG, TIFF, EXR)
# TODO: Add export quality/compression options
def _perform_ffmpeg_export(
    video_path: str, frames_to_extract: list, export_dir: Path, logger: "AppLogger"
) -> tuple[bool, Optional[str]]:
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
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8")
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        logger.error("FFmpeg export failed", extra={"stderr": stderr})
        return False, stderr
    return True, None


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
    # TODO: Add smart cropping based on subject center of mass
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
    for frame_meta in kept_frames:
        if cancel_event.is_set():
            break
        try:
            if not (full_frame_path := export_dir / frame_meta["filename"]).exists():
                continue
            mask_name = frame_meta.get("mask_path", "")
            if not mask_name or not (mask_path := masks_root / mask_name).exists():
                continue
            frame_img = cv2.imread(str(full_frame_path))
            mask_img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if frame_img is None or mask_img is None:
                continue
            contours, _ = cv2.findContours(mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            x_b, y_b, w_b, h_b = cv2.boundingRect(np.concatenate(contours))
            if w_b == 0 or h_b == 0:
                continue
            frame_h, frame_w = frame_img.shape[:2]
            padding_factor = 1.0 + (crop_padding / 100.0)
            feasible_candidates = []
            for ar_str, r in aspect_ratios:
                if w_b / h_b > r:
                    w_c, h_c = w_b, w_b / r
                else:
                    h_c, w_c = h_b, h_b * r
                w_padded, h_padded = w_c * padding_factor, h_c * padding_factor
                scale = 1.0
                if w_padded > frame_w:
                    scale = min(scale, frame_w / w_padded)
                if h_padded > frame_h:
                    scale = min(scale, frame_h / h_padded)
                w_final, h_final = w_padded * scale, h_padded * scale
                if w_final < w_b or h_final < h_b:
                    if w_final < w_b:
                        w_final = w_b
                        h_final = w_final / r
                    if h_final < h_b:
                        h_final = h_b
                        w_final = h_final * r
                    if w_final > frame_w:
                        w_final = frame_w
                        h_final = w_final / r
                    if h_final > frame_h:
                        h_final = frame_h
                        w_final = h_final * r
                center_x_b, center_y_b = x_b + w_b / 2, y_b + h_b / 2
                x1 = center_x_b - w_final / 2
                y1 = center_y_b - h_final / 2
                x1 = max(0, min(x1, frame_w - w_final))
                y1 = max(0, min(y1, frame_h - h_final))
                if x1 > x_b or y1 > y_b or x1 + w_final < x_b + w_b or y1 + h_final < y_b + h_b:
                    continue
                feasible_candidates.append(
                    {"ar_str": ar_str, "x1": x1, "y1": y1, "w_r": w_final, "h_r": h_final, "area": w_final * h_final}
                )
            if not feasible_candidates:
                cropped_img = frame_img[y_b : y_b + h_b, x_b : x_b + w_b]
                if cropped_img.size > 0:
                    cv2.imwrite(str(crop_dir / f"{Path(frame_meta['filename']).stem}_crop_native.png"), cropped_img)
                    num_cropped += 1
                continue
            subject_ar = w_b / h_b if h_b > 0 else 1
            best_candidate = min(
                feasible_candidates,
                key=lambda c: (c["area"], abs((c["w_r"] / c["h_r"] if c["h_r"] > 0 else 1) - subject_ar)),
            )
            x1, y1, w_r, h_r = (
                int(best_candidate["x1"]),
                int(best_candidate["y1"]),
                int(best_candidate["w_r"]),
                int(best_candidate["h_r"]),
            )
            cropped_img = frame_img[y1 : y1 + h_r, x1 : x1 + w_r]
            if cropped_img.size > 0:
                cv2.imwrite(
                    str(crop_dir / f"{Path(frame_meta['filename']).stem}_crop_{best_candidate['ar_str']}.png"),
                    cropped_img,
                )
                num_cropped += 1
        except Exception:
            logger.error(f"Failed to crop frame {frame_meta['filename']}", exc_info=True)
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

            success, stderr = _perform_ffmpeg_export(event.video_path, frames_to_extract, export_dir, logger)
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
                            # Lightroom labels: "Green" for kept (default)
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
    except subprocess.CalledProcessError as e:
        logger.error("FFmpeg export failed", exc_info=True, extra={"stderr": e.stderr})
        return "Error during export: FFmpeg failed. Check logs."
    except Exception as e:
        logger.error("Error during export process", exc_info=True)
        return f"Error during export: {e}"
    except subprocess.CalledProcessError as e:
        logger.error("FFmpeg export failed", exc_info=True, extra={"stderr": e.stderr})
        return "Error during export: FFmpeg failed. Check logs."
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
        export_dir = out_root.parent / f"{out_root.name}_exported_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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
