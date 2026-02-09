from __future__ import annotations

import io
from collections import Counter, defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

import cv2
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim
from torchvision import transforms

if TYPE_CHECKING:
    from core.config import Config
    from core.logger import AppLogger
    from core.managers import ThumbnailManager

from core.database import Database
from core.managers import get_lpips_metric


# TODO: Add support for filtering by scene/shot ID
# TODO: Implement filter presets save/load functionality
# TODO: Add real-time histogram updates during filtering
def load_and_prep_filter_data(output_dir: str, get_all_filter_keys: Callable, config: "Config") -> tuple[list, dict]:
    """Loads metadata from the database and prepares histograms for the UI."""
    db_path = Path(output_dir) / "metadata.db"
    if not db_path.exists():
        return [], {}
    db = Database(db_path)
    all_frames = db.load_all_metadata()
    db.close()

    metric_values = {}
    metric_configs = {
        "quality_score": {"path": ("metrics", "quality_score"), "range": (0, 100)},
        "yaw": {
            "path": ("metrics", "yaw"),
            "range": (config.filter_default_yaw["min"], config.filter_default_yaw["max"]),
        },
        "pitch": {
            "path": ("metrics", "pitch"),
            "range": (config.filter_default_pitch["min"], config.filter_default_pitch["max"]),
        },
        "eyes_open": {"path": ("metrics", "eyes_open"), "range": (0, 1)},
        "face_sim": {"path": ("face_sim",), "range": (0, 1)},
    }
    for k in get_all_filter_keys():
        if k not in metric_configs:
            metric_configs[k] = {"path": (k,), "alt_path": ("metrics", f"{k}_score"), "range": (0, 100)}

    for k in get_all_filter_keys():
        config_item = metric_configs.get(k)
        if not config_item:
            continue
        path, alt_path = config_item.get("path"), config_item.get("alt_path")
        values = []
        for f in all_frames:
            val = None
            if path:
                if len(path) == 1:
                    val = f.get(path[0])
                else:
                    val = f.get(path[0], {}).get(path[1])
            if val is None and alt_path:
                if len(alt_path) == 1:
                    val = f.get(alt_path[0])
                else:
                    val = f.get(alt_path[0], {}).get(alt_path[1])
            if val is not None:
                values.append(val)

        values = np.asarray(values, dtype=float)
        if values.size > 0:
            hist_range = config_item.get("range", (0, 100))
            counts, bins = np.histogram(values, bins=50, range=hist_range)
            metric_values[k] = values.tolist()
            metric_values[f"{k}_hist"] = (counts.tolist(), bins.tolist())
    return all_frames, metric_values


def histogram_svg(hist_data: tuple, title: str = "", logger: Optional["AppLogger"] = None) -> str:
    """Generates an SVG string of a histogram plot."""
    if not plt:
        return """<svg width="100" height="20" xmlns="http://www.w3.org/2000/svg"><text x="5" y="15" font-family="sans-serif" font-size="10" fill="orange">Matplotlib missing</text></svg>"""
    if not hist_data:
        return ""
    try:
        counts, bins = hist_data
        if not isinstance(counts, list) or not isinstance(bins, list) or len(bins) != len(counts) + 1:
            return ""
        with plt.style.context("dark_background"):
            fig, ax = plt.subplots(figsize=(4.6, 2.2), dpi=120)
            ax.bar(bins[:-1], counts, width=np.diff(bins), color="#7aa2ff", alpha=0.85, align="edge")
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
    except Exception:
        if logger:
            logger.error("Failed to generate histogram SVG.", exc_info=True)
        return """<svg width="100" height="20" xmlns="http://www.w3.org/2000/svg"><text x="5" y="15" font-family="sans-serif" font-size="10" fill="red">Plotting failed</text></svg>"""


def build_all_metric_svgs(per_metric_values: dict, get_all_filter_keys: Callable, logger: "AppLogger") -> dict:
    """Builds histogram SVGs for all available metrics."""
    svgs = {}
    for k in get_all_filter_keys():
        if h := per_metric_values.get(f"{k}_hist"):
            svgs[k] = histogram_svg(h, title="", logger=logger)
    return svgs


def _extract_metric_arrays(all_frames_data: list[dict], config: "Config") -> dict:
    """Extracts numerical arrays for each metric from the list of frame data dicts."""
    quality_weights_keys = [
        k.replace("quality_weights_", "") for k in config.model_dump().keys() if k.startswith("quality_weights_")
    ]
    metric_sources = {
        **{k: ("metrics", f"{k}_score") for k in quality_weights_keys},
        "quality_score": ("metrics", "quality_score"),
        "face_sim": ("face_sim",),
        "mask_area_pct": ("mask_area_pct",),
        "eyes_open": ("metrics", "eyes_open"),
        "yaw": ("metrics", "yaw"),
        "pitch": ("metrics", "pitch"),
    }
    metric_arrays = {}
    num_frames = len(all_frames_data)
    for key, path in metric_sources.items():
        if len(path) == 1:
            vals = [f.get(path[0], np.nan) for f in all_frames_data]
        else:
            vals = [f.get(path[0], {}).get(path[1], np.nan) for f in all_frames_data]
        
        # Ensure we return a 1D array of floats with correct length
        arr = np.array(vals, dtype=np.float32)
        if arr.size == 0 and num_frames > 0:
             arr = np.full(num_frames, np.nan, dtype=np.float32)
        metric_arrays[key] = arr
    return metric_arrays


def _run_batched_lpips(
    pairs: list[tuple[int, int]],
    all_frames_data: list[dict],
    dedup_mask: np.ndarray,
    reasons: defaultdict,
    thumbnail_manager: "ThumbnailManager",
    output_dir: str,
    threshold: float,
    device: str = "cpu",
):
    """
    Runs LPIPS deduplication on a list of pairs in batches using GPU if available.
    """
    # TODO: Add adaptive batch sizing based on available GPU memory
    # TODO: Implement caching of LPIPS features for repeated comparisons
    # TODO: Consider multi-scale LPIPS for better accuracy
    if not pairs:
        return
    loss_fn = get_lpips_metric(device=device)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    batch_size = 32

    for i in range(0, len(pairs), batch_size):
        batch = pairs[i : i + batch_size]
        img1_batch, img2_batch, valid_indices = [], [], []

        for p_idx, c_idx in batch:
            p_path = Path(output_dir) / "thumbs" / all_frames_data[p_idx]["filename"]
            c_path = Path(output_dir) / "thumbs" / all_frames_data[c_idx]["filename"]
            img1 = thumbnail_manager.get(p_path)
            img2 = thumbnail_manager.get(c_path)

            if img1 is not None and img2 is not None:
                img1_batch.append(transform(img1))
                img2_batch.append(transform(img2))
                valid_indices.append((p_idx, c_idx))

        if not valid_indices:
            continue

        img1_t = torch.stack(img1_batch).to(device)
        img2_t = torch.stack(img2_batch).to(device)

        with torch.no_grad():
            distances = loss_fn.forward(img1_t, img2_t).squeeze()
            if distances.ndim == 0:
                distances = distances.unsqueeze(0)
            distances = distances.cpu().numpy()

        for j, (p_idx, c_idx) in enumerate(valid_indices):
            dist = float(distances[j])
            if dist <= threshold:
                p_score = all_frames_data[p_idx].get("metrics", {}).get("quality_score", 0)
                c_score = all_frames_data[c_idx].get("metrics", {}).get("quality_score", 0)

                if c_score > p_score:
                    if dedup_mask[p_idx]:
                        reasons[all_frames_data[p_idx]["filename"]].append("duplicate")
                    dedup_mask[p_idx] = False
                else:
                    if dedup_mask[c_idx]:
                        reasons[all_frames_data[c_idx]["filename"]].append("duplicate")
                    dedup_mask[c_idx] = False


def _apply_deduplication_filter(
    all_frames_data: list[dict], filters: dict, thumbnail_manager: "ThumbnailManager", config: "Config", output_dir: str
) -> tuple[np.ndarray, defaultdict]:
    """Applies deduplication logic (pHash, SSIM, or LPIPS) to filter out similar frames."""
    import imagehash  # Lazy import or assume available

    num_frames = len(all_frames_data)
    filenames = [f["filename"] for f in all_frames_data]
    dedup_mask = np.ones(num_frames, dtype=bool)
    reasons = defaultdict(list)
    dedup_method = filters.get("dedup_method", "pHash")

    if filters.get("enable_dedup"):
        if dedup_method == "pHash" and imagehash and filters.get("dedup_thresh", -1) != -1:
            sorted_indices = sorted(range(num_frames), key=lambda i: filenames[i])
            hashes = {
                i: imagehash.hex_to_hash(all_frames_data[i]["phash"])
                for i in range(num_frames)
                if "phash" in all_frames_data[i]
            }

            hash_size = 64
            if hashes:
                hash_size = next(iter(hashes.values())).hash.size
            kept_hash_matrix = np.zeros((num_frames, hash_size), dtype=bool)
            kept_indices = np.zeros(num_frames, dtype=int)
            kept_count = 0
            thresh = filters.get("dedup_thresh", 5)

            for i in sorted_indices:
                if i not in hashes:
                    continue
                curr_hash_flat = hashes[i].hash.flatten()
                is_duplicate = False

                if kept_count > 0:
                    valid_hashes = kept_hash_matrix[:kept_count]
                    diffs = np.bitwise_xor(valid_hashes, curr_hash_flat).sum(axis=1)
                    matches = np.where(diffs <= thresh)[0]

                    if len(matches) > 0:
                        is_duplicate = True
                        match_pos = matches[0]
                        kept_idx = kept_indices[match_pos]

                        if all_frames_data[i].get("metrics", {}).get("quality_score", 0) > all_frames_data[
                            kept_idx
                        ].get("metrics", {}).get("quality_score", 0):
                            if dedup_mask[kept_idx]:
                                reasons[filenames[kept_idx]].append("duplicate")
                            dedup_mask[kept_idx] = False
                            kept_hash_matrix[match_pos] = curr_hash_flat
                            kept_indices[match_pos] = i
                        else:
                            if dedup_mask[i]:
                                reasons[filenames[i]].append("duplicate")
                            dedup_mask[i] = False

                if not is_duplicate:
                    kept_hash_matrix[kept_count] = curr_hash_flat
                    kept_indices[kept_count] = i
                    kept_count += 1

        elif dedup_method == "SSIM" and thumbnail_manager:
            dedup_mask, reasons = apply_ssim_dedup(
                all_frames_data, filters, dedup_mask, reasons, thumbnail_manager, config, output_dir
            )
        elif dedup_method == "LPIPS" and thumbnail_manager:
            dedup_mask, reasons = apply_lpips_dedup(
                all_frames_data, filters, dedup_mask, reasons, thumbnail_manager, config, output_dir
            )
        elif dedup_method == "pHash then LPIPS" and thumbnail_manager and imagehash:
            sorted_indices = sorted(range(num_frames), key=lambda i: filenames[i])
            hashes = {
                i: imagehash.hex_to_hash(all_frames_data[i]["phash"])
                for i in range(num_frames)
                if "phash" in all_frames_data[i]
            }
            p_hash_duplicates = []

            hash_size = 64
            if hashes:
                hash_size = next(iter(hashes.values())).hash.size
            kept_hash_matrix = np.zeros((num_frames, hash_size), dtype=bool)
            kept_indices = np.zeros(num_frames, dtype=int)
            kept_count = 0
            thresh = filters.get("dedup_thresh", 5)

            for i in sorted_indices:
                if i not in hashes:
                    continue
                curr_hash_flat = hashes[i].hash.flatten()
                is_duplicate = False

                if kept_count > 0:
                    valid_hashes = kept_hash_matrix[:kept_count]
                    diffs = np.bitwise_xor(valid_hashes, curr_hash_flat).sum(axis=1)
                    matches = np.where(diffs <= thresh)[0]

                    if len(matches) > 0:
                        is_duplicate = True
                        match_pos = matches[0]
                        kept_idx = kept_indices[match_pos]
                        p_hash_duplicates.append((kept_idx, i))

                if not is_duplicate:
                    kept_hash_matrix[kept_count] = curr_hash_flat
                    kept_indices[kept_count] = i
                    kept_count += 1

            if p_hash_duplicates:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                _run_batched_lpips(
                    p_hash_duplicates,
                    all_frames_data,
                    dedup_mask,
                    reasons,
                    thumbnail_manager,
                    output_dir,
                    filters.get("lpips_threshold", 0.1),
                    device=device,
                )
    return dedup_mask, reasons


def _apply_metric_filters(
    all_frames_data: list[dict], metric_arrays: dict, filters: dict, config: "Config"
) -> tuple[np.ndarray, defaultdict]:
    """Applies threshold-based filtering on scalar metrics."""
    num_frames = len(all_frames_data)
    filenames = [f["filename"] for f in all_frames_data]
    reasons = defaultdict(list)
    quality_weights_keys = [
        k.replace("quality_weights_", "") for k in config.model_dump().keys() if k.startswith("quality_weights_")
    ]
    filter_definitions = [
        *[{"key": k, "type": "range"} for k in quality_weights_keys],
        {"key": "quality_score", "type": "range"},
        {
            "key": "face_sim",
            "type": "min",
            "enabled_key": "face_sim_enabled",
            "reason_low": "face_sim_low",
            "reason_missing": "face_missing",
        },
        {"key": "mask_area_pct", "type": "min", "enabled_key": "mask_area_enabled", "reason_low": "mask_too_small"},
        {"key": "eyes_open", "type": "min", "reason_low": "eyes_closed"},
        {"key": "yaw", "type": "range", "reason_range": "yaw_out_of_range"},
        {"key": "pitch", "type": "range", "reason_range": "pitch_out_of_range"},
    ]
    metric_filter_mask = np.ones(num_frames, dtype=bool)
    for f_def in filter_definitions:
        key, f_type = f_def["key"], f_def["type"]
        if f_def.get("enabled_key") and not filters.get(f_def["enabled_key"]):
            continue
        arr = metric_arrays.get(key)
        if arr is None:
            continue
        f_defaults = getattr(config, f"filter_default_{key}", {})
        if not isinstance(f_defaults, dict):
             # Handle cases where it might be a mocked object or other type
             f_defaults = {}

        if f_type == "range":
            def_min = f_defaults.get("default_min")
            if not isinstance(def_min, (int, float, np.number)):
                def_min = -np.inf
            def_max = f_defaults.get("default_max")
            if not isinstance(def_max, (int, float, np.number)):
                def_max = np.inf
                
            min_v = filters.get(f"{key}_min", def_min)
            max_v = filters.get(f"{key}_max", def_max)
            
            # Final sanity check for UI-provided values which might be wrong types
            if not isinstance(min_v, (int, float, np.number)): min_v = -np.inf
            if not isinstance(max_v, (int, float, np.number)): max_v = np.inf
            
            nan_fill = def_min if def_min != -np.inf else 0.0
            mask = (np.nan_to_num(arr, nan=nan_fill) >= min_v) & (np.nan_to_num(arr, nan=nan_fill) <= max_v)
            metric_filter_mask &= mask
        elif f_type == "min":
            def_min = f_defaults.get("default_min")
            if not isinstance(def_min, (int, float, np.number)):
                def_min = -np.inf
                
            min_v = filters.get(f"{key}_min", def_min)
            if not isinstance(min_v, (int, float, np.number)): min_v = -np.inf
            
            nan_fill = def_min if def_min != -np.inf else 0.0
            if key == "face_sim":
                has_face_sim = ~np.isnan(arr)
                mask = np.ones(num_frames, dtype=bool)
                mask[has_face_sim] = arr[has_face_sim] >= min_v
                if filters.get("require_face_match"):
                    mask &= has_face_sim
            else:
                mask = np.nan_to_num(arr, nan=nan_fill) >= min_v
            metric_filter_mask &= mask

    metric_rejection_mask = ~metric_filter_mask
    for i in np.where(metric_rejection_mask)[0]:
        for f_def in filter_definitions:
            key, f_type = f_def["key"], f_def["type"]
            if f_def.get("enabled_key") and not filters.get(f_def["enabled_key"]):
                continue
            arr = metric_arrays.get(key)
            if arr is None:
                continue
            f_defaults = getattr(config, f"filter_default_{key}", {})
            v = arr[i]
            if f_type == "range":
                min_v = filters.get(f"{key}_min", f_defaults.get("default_min", -np.inf))
                max_v = filters.get(f"{key}_max", f_defaults.get("default_max", np.inf))
                
                # Ensure they are scalars for comparison
                if not np.isscalar(min_v): min_v = -np.inf
                if not np.isscalar(max_v): max_v = np.inf

                if not np.isnan(v):
                    reason = f_def.get("reason_range")
                    if v < min_v:
                        reasons[filenames[i]].append(reason or f_def.get("reason_low", f"{key}_low"))
                    if v > max_v:
                        reasons[filenames[i]].append(reason or f_def.get("reason_high", f"{key}_high"))
            elif f_type == "min":
                min_v = filters.get(f"{key}_min", f_defaults.get("default_min", -np.inf))
                if not np.isscalar(min_v): min_v = -np.inf

                if not np.isnan(v) and v < min_v:
                    reasons[filenames[i]].append(f_def.get("reason_low", f"{key}_low"))
                if key == "face_sim" and filters.get("require_face_match") and np.isnan(v):
                    reasons[filenames[i]].append(f_def.get("reason_missing", "face_missing"))
    return metric_filter_mask, reasons


def apply_all_filters_vectorized(
    all_frames_data: list[dict],
    filters: dict,
    config: "Config",
    thumbnail_manager: Optional["ThumbnailManager"] = None,
    output_dir: Optional[str] = None,
) -> tuple[list, list, Counter, dict]:
    """
    Main entry point for filtering frames based on deduplication and metric thresholds.

    Returns:
        Tuple of (kept_frames, rejected_frames, rejection_counts, rejection_reasons)
    """
    if not all_frames_data:
        return [], [], Counter(), {}
    # TODO: These variables are computed but not used - clean up or utilize
    len(all_frames_data)
    [f["filename"] for f in all_frames_data]
    metric_arrays = _extract_metric_arrays(all_frames_data, config)
    dedup_mask, reasons = _apply_deduplication_filter(all_frames_data, filters, thumbnail_manager, config, output_dir)
    metric_filter_mask, metric_reasons = _apply_metric_filters(all_frames_data, metric_arrays, filters, config)
    kept_mask = dedup_mask & metric_filter_mask
    for fname, reason_list in metric_reasons.items():
        reasons[fname].extend(reason_list)
    kept = [all_frames_data[i] for i in np.where(kept_mask)[0]]
    rejected = [all_frames_data[i] for i in np.where(~kept_mask)[0]]
    total_reasons = Counter(r for r_list in reasons.values() for r in r_list)
    return kept, rejected, total_reasons, reasons


def _generic_dedup(
    all_frames_data: list[dict],
    dedup_mask: np.ndarray,
    reasons: defaultdict,
    thumbnail_manager: "ThumbnailManager",
    output_dir: str,
    compare_fn: Callable[[np.ndarray, np.ndarray], bool],
) -> tuple[np.ndarray, defaultdict]:
    """Generic deduplication helper that compares adjacent frames using a custom function."""
    num_frames = len(all_frames_data)
    sorted_indices = sorted(range(num_frames), key=lambda i: all_frames_data[i]["filename"])
    for i in range(1, len(sorted_indices)):
        c_idx, p_idx = sorted_indices[i], sorted_indices[i - 1]
        c_frame_data, p_frame_data = all_frames_data[c_idx], all_frames_data[p_idx]
        c_thumb_path = Path(output_dir) / "thumbs" / c_frame_data["filename"]
        p_thumb_path = Path(output_dir) / "thumbs" / p_frame_data["filename"]
        img1, img2 = thumbnail_manager.get(p_thumb_path), thumbnail_manager.get(c_thumb_path)
        if img1 is not None and img2 is not None:
            if compare_fn(img1, img2):
                if all_frames_data[c_idx].get("metrics", {}).get("quality_score", 0) > all_frames_data[p_idx].get(
                    "metrics", {}
                ).get("quality_score", 0):
                    if dedup_mask[p_idx]:
                        reasons[all_frames_data[p_idx]["filename"]].append("duplicate")
                    dedup_mask[p_idx] = False
                else:
                    if dedup_mask[c_idx]:
                        reasons[all_frames_data[c_idx]["filename"]].append("duplicate")
                    dedup_mask[c_idx] = False
    return dedup_mask, reasons


def _ssim_compare(img1: np.ndarray, img2: np.ndarray, threshold: float) -> bool:
    """Compares two images using SSIM."""
    gray1, gray2 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY), cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    return ssim(gray1, gray2) >= threshold


def apply_ssim_dedup(
    all_frames_data: list[dict],
    filters: dict,
    dedup_mask: np.ndarray,
    reasons: defaultdict,
    thumbnail_manager: "ThumbnailManager",
    config: "Config",
    output_dir: str,
) -> tuple[np.ndarray, defaultdict]:
    """Applies SSIM-based deduplication."""
    threshold = filters.get("ssim_threshold", 0.95)

    def compare_fn(img1, img2):
        return _ssim_compare(img1, img2, threshold)

    return _generic_dedup(all_frames_data, dedup_mask, reasons, thumbnail_manager, output_dir, compare_fn)


def apply_lpips_dedup(
    all_frames_data: list[dict],
    filters: dict,
    dedup_mask: np.ndarray,
    reasons: defaultdict,
    thumbnail_manager: "ThumbnailManager",
    config: "Config",
    output_dir: str,
) -> tuple[np.ndarray, defaultdict]:
    """Applies LPIPS-based deduplication."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_frames = len(all_frames_data)
    sorted_indices = sorted(range(num_frames), key=lambda i: all_frames_data[i]["filename"])

    # Compare adjacent frames
    pairs = [(sorted_indices[i - 1], sorted_indices[i]) for i in range(1, len(sorted_indices))]

    _run_batched_lpips(
        pairs,
        all_frames_data,
        dedup_mask,
        reasons,
        thumbnail_manager,
        output_dir,
        filters.get("lpips_threshold", 0.1),
        device=device,
    )

    return dedup_mask, reasons
