import json
from pathlib import Path
import re
from collections import Counter, defaultdict

import gradio as gr
import numpy as np

from app.config import Config
from app.logging_enhanced import EnhancedLogger
from app.events import FilterEvent
from app.frames import render_mask_overlay


def load_and_prep_filter_data(metadata_path, get_all_filter_keys):
    """Load and prepare frame data for filtering."""
    if not metadata_path or not Path(metadata_path).exists():
        return [], {}

    with Path(metadata_path).open('r', encoding='utf-8') as f:
        try:
            next(f)  # skip header
        except StopIteration:
            return [], {}
        all_frames = [json.loads(line) for line in f if line.strip()]

    metric_values = {}
    for k in get_all_filter_keys():
        is_face_sim = k == 'face_sim'
        values = np.asarray([
            f.get(k, f.get("metrics", {}).get(f"{k}_score"))
            for f in all_frames
            if (f.get(k) is not None or
                f.get("metrics", {}).get(f"{k}_score") is not None)
        ], dtype=float)

        if values.size > 0:
            hist_range = (0, 1) if is_face_sim else (0, 100)
            counts, bins = np.histogram(values, bins=50, range=hist_range)
            metric_values[k] = values.tolist()
            metric_values[f"{k}_hist"] = (counts.tolist(), bins.tolist())
    return all_frames, metric_values


def build_all_metric_svgs(per_metric_values, get_all_filter_keys, logger):
    """Build SVG histograms for all metrics."""
    svgs = {}
    for k in get_all_filter_keys():
        if (h := per_metric_values.get(f"{k}_hist")):
            svgs[k] = histogram_svg(h, title="", logger=logger)
    return svgs


def histogram_svg(hist_data, title="", logger=None):
    """Generate SVG histogram for metrics."""
    if not hist_data:
        return ""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
        import io

        counts, bins = hist_data
        if (not isinstance(counts, list) or not isinstance(bins, list) or
            len(bins) != len(counts) + 1):
            return ""

        with plt.style.context("dark_background"):
            fig, ax = plt.subplots(figsize=(4.6, 2.2), dpi=120)
            ax.bar(bins[:-1], counts, width=np.diff(bins),
                  color="#7aa2ff", alpha=0.85, align="edge")
            ax.grid(axis="y", alpha=0.2)
            ax.margins(x=0)
            ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            ax.tick_params(labelsize=8)
            ax.set_title(title)
            buf = io.StringIO()
            fig.savefig(buf, format="svg", bbox_inches="tight")
            plt.close(fig)
        return buf.getvalue()
    except Exception as e:
        if logger:
            logger.error("Failed to generate histogram SVG.", exc_info=True)
        return ""


def apply_all_filters_vectorized(all_frames_data, filters, config: 'Config'):
    """Apply all filters to frame data using vectorized operations."""
    import imagehash

    if not all_frames_data:
        return [], [], Counter(), {}

    num_frames = len(all_frames_data)
    filenames = [f['filename'] for f in all_frames_data]
    metric_arrays = {k: np.array([f.get("metrics", {}).get(f"{k}_score", np.nan)
                                for f in all_frames_data], dtype=np.float32)
                     for k in config.QUALITY_METRICS}
    metric_arrays["face_sim"] = np.array([f.get("face_sim", np.nan)
                                         for f in all_frames_data], dtype=np.float32)
    metric_arrays["mask_area_pct"] = np.array([f.get("mask_area_pct", np.nan)
                                              for f in all_frames_data], dtype=np.float32)

    kept_mask = np.ones(num_frames, dtype=bool)
    reasons = defaultdict(list)

    # --- 1. Deduplication ---
    dedup_mask = np.ones(num_frames, dtype=bool)
    dedup_thresh_val = filters.get("dedup_thresh", 5)
    if filters.get("enable_dedup") and dedup_thresh_val != -1:
        all_indices = list(range(num_frames))
        sorted_indices = sorted(all_indices, key=lambda i: filenames[i])
        hashes = {i: imagehash.hex_to_hash(all_frames_data[i]['phash'])
                  for i in sorted_indices if 'phash' in all_frames_data[i]}

        for i in range(1, len(sorted_indices)):
            current_idx, prev_idx = sorted_indices[i], sorted_indices[i - 1]
            if prev_idx in hashes and current_idx in hashes and \
               (hashes[prev_idx] - hashes[current_idx]) <= dedup_thresh_val:
                if dedup_mask[current_idx]:
                    reasons[filenames[current_idx]].append('duplicate')
                dedup_mask[current_idx] = False

    # --- 2. Combined Metric Filters ---
    metric_filter_mask = np.ones(num_frames, dtype=bool)

    for k in config.QUALITY_METRICS:
        min_val = filters.get(f"{k}_min", 0)
        max_val = filters.get(f"{k}_max", 100)
        metric_filter_mask &= (np.nan_to_num(metric_arrays[k], nan=min_val) >= min_val) & \
                              (np.nan_to_num(metric_arrays[k], nan=max_val) <= max_val)

    if filters.get("face_sim_enabled"):
        face_sim_min = filters.get("face_sim_min", 0.5)
        face_sim_values = metric_arrays["face_sim"]
        # Only filter frames that have a face similarity score
        has_face_sim = ~np.isnan(face_sim_values)
        metric_filter_mask[has_face_sim] &= (face_sim_values[has_face_sim] >= face_sim_min)

        if filters.get("require_face_match"):
            metric_filter_mask &= has_face_sim


    if filters.get("mask_area_enabled"):
        mask_area_min = filters.get("mask_area_pct_min", 1.0)
        mask_area_values = np.nan_to_num(metric_arrays["mask_area_pct"], nan=0.0)
        metric_filter_mask &= (mask_area_values >= mask_area_min)

    # --- 3. Final Mask and Reason Assignment ---
    kept_mask = dedup_mask & metric_filter_mask

    # Now, assign reasons ONLY for the frames that were rejected by metrics but not by deduplication
    metric_rejection_mask = ~metric_filter_mask & dedup_mask

    for i in np.where(metric_rejection_mask)[0]:
        # Check each filter condition to see why it was rejected
        for k in config.QUALITY_METRICS:
            min_val, max_val = filters.get(f"{k}_min", 0), filters.get(f"{k}_max", 100)
            if not (min_val <= metric_arrays[k][i] <= max_val):
                reasons[filenames[i]].append(f"{k}_{'low' if metric_arrays[k][i] < min_val else 'high'}")

        if filters.get("face_sim_enabled"):
            face_sim_min = filters.get("face_sim_min", 0.5)
            if metric_arrays["face_sim"][i] < face_sim_min:
                reasons[filenames[i]].append("face_sim_low")
            if filters.get("require_face_match") and np.isnan(metric_arrays["face_sim"][i]):
                reasons[filenames[i]].append("face_missing")

        if filters.get("mask_area_enabled"):
            mask_area_min = filters.get("mask_area_pct_min", 1.0)
            if metric_arrays["mask_area_pct"][i] < mask_area_min:
                reasons[filenames[i]].append("mask_too_small")

    # --- 4. Final Selection ---
    kept_indices = np.where(kept_mask)[0]
    rejected_indices = np.where(~kept_mask)[0]

    kept = [all_frames_data[i] for i in kept_indices]
    rejected = [all_frames_data[i] for i in rejected_indices]

    counts = Counter(r for r_list in reasons.values() for r in r_list)
    return kept, rejected, counts, reasons


def on_filters_changed(event: FilterEvent, thumbnail_manager, config: 'Config', logger=None):
    """Handle filter changes and update gallery."""
    from app.logging_enhanced import EnhancedLogger
    logger = logger or EnhancedLogger()

    if not event.all_frames_data:
        return {"filter_status_text": "Run analysis to see results.", "results_gallery": []}

    filters = event.slider_values.copy()
    filters.update({
        "require_face_match": event.require_face_match,
        "dedup_thresh": event.dedup_thresh,
        "face_sim_enabled": bool(event.per_metric_values.get("face_sim")),
        "mask_area_enabled": bool(event.per_metric_values.get("mask_area_pct")),
        "enable_dedup": (any('phash' in f for f in event.all_frames_data)
                       if event.all_frames_data else False)
    })

    status_text, gallery_update = _update_gallery(
        event.all_frames_data, filters, event.output_dir,
        event.gallery_view, event.show_overlay, event.overlay_alpha,
        thumbnail_manager, config, logger
    )
    return {"filter_status_text": status_text, "results_gallery": gallery_update}


def _update_gallery(all_frames_data, filters, output_dir,
                   gallery_view, show_overlay, overlay_alpha,
                   thumbnail_manager, config: 'Config', logger):
    """Update the results gallery based on current filters."""
    kept, rejected, counts, per_frame_reasons = apply_all_filters_vectorized(
        all_frames_data, filters or {}, config
    )
    status_parts = [f"**Kept:** {len(kept)}/{len(all_frames_data)}"]
    if counts:
        rejections = ', '.join([f'{k}: {v}' for k, v in counts.most_common(3)])
        status_parts.append(f"**Rejections:** {rejections}")
    status_text = " | ".join(status_parts)

    frames_to_show = rejected if gallery_view == "Rejected Frames" else kept
    preview_images = []

    if output_dir:
        output_path = Path(output_dir)
        thumb_dir = output_path / "thumbs"
        masks_dir = output_path / "masks"

        for f_meta in frames_to_show[:100]:
            thumb_path = thumb_dir / f"{Path(f_meta['filename']).stem}.webp"
            caption = ""
            if gallery_view == "Rejected Frames":
                reasons_list = per_frame_reasons.get(f_meta['filename'], [])
                caption = f"Reasons: {', '.join(reasons_list)}"

            thumb_rgb_np = thumbnail_manager.get(thumb_path)
            if thumb_rgb_np is None:
                continue

            if (show_overlay and not f_meta.get("mask_empty", True) and
                (mask_name := f_meta.get("mask_path"))):
                import cv2
                mask_path = masks_dir / mask_name
                if mask_path.exists():
                    mask_gray = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                    thumb_overlay_rgb = render_mask_overlay(
                        thumb_rgb_np, mask_gray, float(overlay_alpha), logger=logger
                    )
                    preview_images.append((thumb_overlay_rgb, caption))
                else:
                    preview_images.append((thumb_rgb_np, caption))
            else:
                preview_images.append((thumb_rgb_np, caption))

    gallery_rows = 1 if gallery_view == "Rejected Frames" else 2
    return status_text, gr.update(value=preview_images, rows=gallery_rows)


def reset_filters(all_frames_data, per_metric_values, output_dir, config,
                  slider_keys, thumbnail_manager):
    """Reset all filters to default values."""
    output_values = {}
    slider_default_values = []

    for key in slider_keys:
        metric_key = re.sub(r'_(min|max)$', '', key)
        default_key = 'default_max' if key.endswith('_max') else 'default_min'
        default_val = config.filter_defaults[metric_key][default_key]
        output_values[f'slider_{key}'] = gr.update(value=default_val)
        slider_default_values.append(default_val)

    face_match_default = config.ui_defaults['require_face_match']
    dedup_default = config.filter_defaults['dedup_thresh']['default']
    output_values['require_face_match_input'] = gr.update(value=face_match_default)
    output_values['dedup_thresh_input'] = gr.update(value=dedup_default)

    if all_frames_data:
        slider_defaults_dict = {key: val for key, val in zip(slider_keys, slider_default_values)}
        filter_event = FilterEvent(
            all_frames_data=all_frames_data,
            per_metric_values=per_metric_values,
            output_dir=output_dir,
            gallery_view="Kept Frames",
            show_overlay=True,
            overlay_alpha=0.6,
            require_face_match=face_match_default,
            dedup_thresh=dedup_default,
            slider_values=slider_defaults_dict
        )
        updates = on_filters_changed(filter_event, thumbnail_manager)
        output_values['filter_status_text'] = updates['filter_status_text']
        output_values['results_gallery'] = updates['results_gallery']
    else:
        output_values['filter_status_text'] = "Load an analysis to begin."
        output_values['results_gallery'] = []

    return output_values


def auto_set_thresholds(per_metric_values, p, slider_keys):
    """Auto-set filter thresholds based on percentiles."""
    updates = {}
    if not per_metric_values:
        return {f'slider_{key}': gr.update() for key in slider_keys}

    pmap = {
        k: float(np.percentile(np.asarray(vals, dtype=np.float32), p))
        for k, vals in per_metric_values.items()
        if not k.endswith('_hist') and vals
    }

    for key in slider_keys:
        updates[f'slider_{key}'] = gr.update()
        if key.endswith('_min'):
            metric = key[:-4]
            if metric in pmap:
                updates[f'slider_{key}'] = gr.update(value=round(pmap[metric], 2))
    return updates