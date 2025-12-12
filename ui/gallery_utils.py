from __future__ import annotations
import math
import cv2
import numpy as np
import json
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any
import gradio as gr
from collections import Counter

from core.models import Scene
from core.filtering import apply_all_filters_vectorized
from core.utils import render_mask_overlay
from events import FilterEvent

def scene_matches_view(scene: Scene, view: str) -> bool:
    status = scene.status
    if view == "All": return status in ("included", "excluded", "pending")
    if view == "Kept": return status == "included"
    if view == "Rejected": return status == "excluded"
    return False

def create_scene_thumbnail_with_badge(thumb_img: np.ndarray, scene_idx: int, is_excluded: bool) -> np.ndarray:
    thumb = thumb_img.copy()
    h, w = thumb.shape[:2]
    if is_excluded:
        border_color = (33, 128, 141)
        cv2.rectangle(thumb, (0, 0), (w-1, h-1), border_color, 4)
        badge_size = int(min(w, h) * 0.15)
        badge_pos = (w - badge_size - 5, 5)
        cv2.circle(thumb, (badge_pos[0] + badge_size//2, badge_pos[1] + badge_size//2), badge_size//2, (255, 255, 255), -1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(thumb, "E", badge_pos, font, 0.5, border_color, 2)
    return thumb

def scene_caption(s: Scene) -> str:
    shot = s.shot_id
    start, end = s.start_frame, s.end_frame
    status_icon = "✅" if s.status == "included" else "❌"
    caption = f"Scene {shot} [{start}-{end}] {status_icon}"
    if s.status == 'excluded' and s.rejection_reasons:
        caption += f"\n({', '.join(s.rejection_reasons)})"
    if s.seed_type:
        caption += f"\nSeed: {s.seed_type}"
    return caption

def build_scene_gallery_items(scenes: list[Union[dict, Scene]], view: str, output_dir: str, page_num: int = 1, page_size: int = 20) -> tuple[list[tuple], list[int], int]:
    items: list[tuple[Optional[str], str]] = []
    index_map: list[int] = []
    if not scenes: return [], [], 1
    # Ensure scenes are Scene objects
    scenes_objs = [Scene(**s) if isinstance(s, dict) else s for s in scenes]
    previews_dir = Path(output_dir) / "previews"
    previews_dir.mkdir(parents=True, exist_ok=True)
    filtered_scenes = [(i, s) for i, s in enumerate(scenes_objs) if scene_matches_view(s, view)]
    total_pages = max(1, (len(filtered_scenes) + page_size - 1) // page_size)
    start_idx = (page_num - 1) * page_size
    end_idx = start_idx + page_size
    page_scenes = filtered_scenes[start_idx:end_idx]

    for i, s in page_scenes:
        thumb_path = previews_dir / f"scene_{s.shot_id:05d}.jpg"
        if not thumb_path.exists():
             continue # Skip if no preview
        else:
            try:
                thumb_img_np = cv2.imread(str(thumb_path))
                if thumb_img_np is None:
                     continue
                thumb_img_np = cv2.cvtColor(thumb_img_np, cv2.COLOR_BGR2RGB)
                badged_thumb = create_scene_thumbnail_with_badge(thumb_img_np, i, s.status == 'excluded')
                items.append((badged_thumb, scene_caption(s)))
            except Exception:
                continue
        index_map.append(i)
    return items, index_map, total_pages

def _update_gallery(all_frames_data: list[dict], filters: dict, output_dir: str, gallery_view: str,
                    show_overlay: bool, overlay_alpha: float, thumbnail_manager: Any,
                    config: Any, logger: Any) -> tuple[str, gr.update]:
    kept, rejected, counts, per_frame_reasons = apply_all_filters_vectorized(all_frames_data, filters or {}, config, thumbnail_manager, output_dir)
    status_parts = [f"**Kept:** {len(kept)}/{len(all_frames_data)}"]
    if counts:
        rejection_reasons = ', '.join([f'{k}: {v}' for k, v in counts.most_common()])
        status_parts.append(f"**Rejections:** {rejection_reasons}")

    status_text, frames_to_show, preview_images = " | ".join(status_parts), rejected if gallery_view == "Rejected" else kept, []
    if output_dir:
        output_path, thumb_dir, masks_dir = Path(output_dir), Path(output_dir) / "thumbs", Path(output_dir) / "masks"
        for f_meta in frames_to_show[:100]:
            thumb_path = thumb_dir / f"{Path(f_meta['filename']).stem}.webp"
            caption = f"Reasons: {', '.join(per_frame_reasons.get(f_meta['filename'], []))}" if gallery_view == "Rejected" else ""
            thumb_rgb_np = thumbnail_manager.get(thumb_path)
            if thumb_rgb_np is None: continue
            if show_overlay and not f_meta.get("mask_empty", True) and (mask_name := f_meta.get("mask_path")):
                mask_path = masks_dir / mask_name
                if mask_path.exists():
                    mask_gray = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                    preview_images.append((render_mask_overlay(thumb_rgb_np, mask_gray, float(overlay_alpha), logger=logger), caption))
                else: preview_images.append((thumb_rgb_np, caption))
            else: preview_images.append((thumb_rgb_np, caption))
    return status_text, gr.update(value=preview_images, rows=1 if gallery_view == "Rejected Frames" else 2)

def on_filters_changed(event: FilterEvent, thumbnail_manager: Any,
                       config: Any, logger: Any) -> dict:
    if not event.all_frames_data: return {"filter_status_text": "Run analysis to see results.", "results_gallery": []}
    filters = event.slider_values.copy()
    filters.update({"require_face_match": event.require_face_match, "dedup_thresh": event.dedup_thresh,
                    "face_sim_enabled": bool(event.per_metric_values.get("face_sim")),
                    "mask_area_enabled": bool(event.per_metric_values.get("mask_area_pct")),
                    "enable_dedup": any('phash' in f for f in event.all_frames_data) if event.all_frames_data else False,
                    "dedup_method": event.dedup_method})
    status_text, gallery_update = _update_gallery(event.all_frames_data, filters, event.output_dir, event.gallery_view,
                                                  event.show_overlay, event.overlay_alpha, thumbnail_manager, config, logger)
    return {"filter_status_text": status_text, "results_gallery": gallery_update}

def auto_set_thresholds(per_metric_values: dict, p: int, slider_keys: list[str], selected_metrics: list[str]) -> dict:
    updates = {}
    if not per_metric_values: return {f'slider_{key}': gr.update() for key in slider_keys}
    pmap = {
        k: float(np.percentile(np.asarray(vals, dtype=np.float32), p))
        for k, vals in per_metric_values.items()
        if not k.endswith('_hist') and vals and k in selected_metrics
    }
    for key in slider_keys:
        metric_name = key.replace('_min', '').replace('_max', '')
        if key.endswith('_min') and metric_name in pmap: updates[f'slider_{key}'] = gr.update(value=round(pmap[metric_name], 2))
        else: updates[f'slider_{key}'] = gr.update()
    return updates
