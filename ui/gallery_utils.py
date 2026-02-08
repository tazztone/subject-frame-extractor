from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import gradio as gr
import numpy as np
from functools import lru_cache

from core.events import FilterEvent
from core.filtering import apply_all_filters_vectorized

# Re-export from core.shared for backward compatibility
from core.shared import build_scene_gallery_items
from core.utils import render_mask_overlay

__all__ = ["build_scene_gallery_items", "render_mask_overlay", "on_filters_changed", "auto_set_thresholds"]

@lru_cache(maxsize=256)
def _load_mask_cached(mask_path: str) -> Optional[np.ndarray]:
    """Loads a mask from disk with LRU caching."""
    if not Path(mask_path).exists():
        return None
    return cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

def clear_mask_cache():
    """Clears the mask LRU cache."""
    _load_mask_cached.cache_clear()


def _update_gallery(
    all_frames_data: list[dict],
    filters: dict,
    output_dir: str,
    gallery_view: str,
    show_overlay: bool,
    overlay_alpha: float,
    thumbnail_manager: Any,
    config: Any,
    logger: Any,
) -> tuple[str, gr.update]:
    """
    Updates the Gradio gallery based on applied filters.

    Returns:
        A tuple containing the status text and a Gradio update object for the gallery.
    """
    # TODO: Add pagination support for large datasets (>1000 frames)
    # TODO: Implement virtual scrolling with lazy image loading
    # TODO: Add gallery sorting options (by score, time, etc.)
    kept, rejected, counts, per_frame_reasons = apply_all_filters_vectorized(
        all_frames_data, filters or {}, config, thumbnail_manager, output_dir
    )
    status_parts = [f"**Kept:** {len(kept)}/{len(all_frames_data)}"]
    if counts:
        rejection_reasons = ", ".join([f"{k}: {v}" for k, v in counts.most_common()])
        status_parts.append(f"**Rejections:** {rejection_reasons}")

    status_text, frames_to_show, preview_images = (
        " | ".join(status_parts),
        rejected if gallery_view == "Rejected" else kept,
        [],
    )
    if output_dir:
        _output_path, thumb_dir, masks_dir = Path(output_dir), Path(output_dir) / "thumbs", Path(output_dir) / "masks"
        MAX_OVERLAY_RENDER = 100
        for i, f_meta in enumerate(frames_to_show[:500]):
            thumb_path = thumb_dir / f"{Path(f_meta['filename']).stem}.webp"
            caption = (
                f"Reasons: {', '.join(per_frame_reasons.get(f_meta['filename'], []))}"
                if gallery_view == "Rejected"
                else ""
            )
            thumb_rgb_np = thumbnail_manager.get(thumb_path)
            if thumb_rgb_np is None:
                continue
                
            use_overlay = show_overlay and i < MAX_OVERLAY_RENDER and not f_meta.get("mask_empty", True)
            if use_overlay and (mask_name := f_meta.get("mask_path")):
                mask_path = masks_dir / mask_name
                mask_gray = _load_mask_cached(str(mask_path))
                if mask_gray is not None:
                    preview_images.append(
                        (render_mask_overlay(thumb_rgb_np, mask_gray, float(overlay_alpha), logger=logger), caption)
                    )
                else:
                    preview_images.append((thumb_rgb_np, caption))
            else:
                preview_images.append((thumb_rgb_np, caption))
    return status_text, gr.update(value=preview_images, rows=1 if gallery_view == "Rejected Frames" else 2)


def on_filters_changed(event: FilterEvent, thumbnail_manager: Any, config: Any, logger: Any) -> dict:
    """
    Event handler for when filter settings are modified.

    Re-filters data and updates the gallery view.
    """
    if not event.all_frames_data:
        return {"filter_status_text": "Run analysis to see results.", "results_gallery": []}
    filters = event.slider_values.copy()
    filters.update(
        {
            "require_face_match": event.require_face_match,
            "dedup_thresh": event.dedup_thresh,
            "face_sim_enabled": bool(event.per_metric_values.get("face_sim")),
            "mask_area_enabled": bool(event.per_metric_values.get("mask_area_pct")),
            "enable_dedup": any("phash" in f for f in event.all_frames_data) if event.all_frames_data else False,
            "dedup_method": event.dedup_method,
        }
    )
    status_text, gallery_update = _update_gallery(
        event.all_frames_data,
        filters,
        event.output_dir,
        event.gallery_view,
        event.show_overlay,
        event.overlay_alpha,
        thumbnail_manager,
        config,
        logger,
    )
    return {"filter_status_text": status_text, "results_gallery": gallery_update}


def auto_set_thresholds(per_metric_values: dict, p: int, slider_keys: list[str], selected_metrics: list[str]) -> dict:
    """
    Calculates threshold values based on data percentiles.

    Args:
        per_metric_values: Dictionary of metric values.
        p: Percentile value.
        slider_keys: List of slider component keys.
        selected_metrics: List of metrics to auto-tune.

    Returns:
        Dictionary of updates for the sliders.
    """

    updates = {}
    if not per_metric_values:
        return {f"slider_{key}": gr.update() for key in slider_keys}
    pmap = {
        k: float(np.percentile(np.asarray(vals, dtype=np.float32), p))
        for k, vals in per_metric_values.items()
        if not k.endswith("_hist") and vals and k in selected_metrics
    }
    for key in slider_keys:
        metric_name = key.replace("_min", "").replace("_max", "")
        if key.endswith("_min") and metric_name in pmap:
            updates[f"slider_{key}"] = gr.update(value=round(pmap[metric_name], 2))
        else:
            updates[f"slider_{key}"] = gr.update()
    return updates
