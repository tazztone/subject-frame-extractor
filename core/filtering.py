from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from core.config import Config
    from core.logger import AppLogger
    from core.managers import ThumbnailManager

from core.database import Database
from core.operators.dedup import apply_deduplication_filter
from core.operators.viz import histogram_svg


def _get_nested_value(d: Any, path: tuple[str, ...]) -> Any:
    """Helper to safely fetch a nested value from a dictionary path."""
    for key in path:
        if isinstance(d, dict):
            d = d.get(key)
        else:
            return None
    return d


def load_and_prep_filter_data(output_dir: str, get_all_filter_keys: Callable, config: "Config") -> tuple[list, dict]:
    db_path = Path(output_dir) / "metadata.db"
    if not db_path.exists():
        return [], {}
    db = Database(db_path)
    all_frames = db.load_all_metadata()
    db.close()
    metric_values = {}
    from core.operators.registry import OperatorRegistry

    defs = OperatorRegistry.get_all_filter_definitions(config)
    defs_by_key = {d.key: d for d in defs}

    metric_configs = {}
    for k in get_all_filter_keys():
        d = defs_by_key.get(k)
        if d:
            metric_configs[k] = {"path": d.metadata_path, "range": d.histogram_range}
        else:
            metric_configs[k] = {"path": (k,), "alt_path": ("metrics", f"{k}_score"), "range": (0.0, 100.0)}

    for k in get_all_filter_keys():
        cfg = metric_configs.get(k)
        if not cfg:
            continue
        path, alt = cfg.get("path"), cfg.get("alt_path")

        vals = []
        for f in all_frames:
            val = None
            if path:
                val = _get_nested_value(f, path)
            if val is None and alt:
                val = _get_nested_value(f, alt)
            if val is not None:
                vals.append(val)

        vals = np.asarray(vals, dtype=float)
        if vals.size > 0:
            counts, bins = np.histogram(vals, bins=50, range=cfg.get("range", (0.0, 100.0)))
            metric_values[k], metric_values[f"{k}_hist"] = vals.tolist(), (counts.tolist(), bins.tolist())
    return all_frames, metric_values


def build_all_metric_svgs(per_metric_values: dict, get_all_filter_keys: Callable, logger: "AppLogger") -> dict:
    svgs = {}
    for k in get_all_filter_keys():
        if h := per_metric_values.get(f"{k}_hist"):
            svgs[k] = histogram_svg(h, logger=logger)
    return svgs


def _extract_metric_arrays(all_frames_data: List[Dict[str, Any]], config: "Config") -> Dict[str, np.ndarray]:
    from core.operators.registry import OperatorRegistry

    defs = OperatorRegistry.get_all_filter_definitions(config)

    metric_arrays = {}
    for d in defs:
        path = d.metadata_path
        vals = []
        for f in all_frames_data:
            v = _get_nested_value(f, path)
            vals.append(v if v is not None else np.nan)
        metric_arrays[d.key] = np.array(vals, dtype=np.float32)
    return metric_arrays


def _apply_metric_filters(
    all_frames_data: List[Dict[str, Any]],
    metric_arrays: Dict[str, np.ndarray],
    filters: Dict[str, Any],
    config: "Config",
) -> Tuple[np.ndarray, defaultdict]:
    num_frames = len(all_frames_data)
    filenames = [f["filename"] for f in all_frames_data]
    reasons = defaultdict(list)

    from core.operators.registry import OperatorRegistry

    defs = OperatorRegistry.get_all_filter_definitions(config)

    mask = np.ones(num_frames, dtype=bool)
    for d in defs:
        k, t = d.key, d.filter_type
        if d.enabled_key and not filters.get(d.enabled_key):
            continue
        arr = metric_arrays.get(k)
        if arr is None:
            continue
        defaults = getattr(config, f"filter_default_{k}", {})
        if t == "range":
            min_v = filters.get(f"{k}_min", defaults.get("default_min", d.default_min))
            max_v = filters.get(f"{k}_max", defaults.get("default_max", d.default_max))
            if hasattr(min_v, "tolist"):
                min_v = min_v.tolist()  # Handle mocks
            if hasattr(max_v, "tolist"):
                max_v = max_v.tolist()
            def_min = defaults.get("default_min", d.default_min)
            if hasattr(def_min, "tolist"):
                def_min = def_min.tolist()
            nan_fill = def_min if def_min != -np.inf else 0.0
            mask &= (np.nan_to_num(arr, nan=float(nan_fill)) >= float(min_v)) & (
                np.nan_to_num(arr, nan=float(nan_fill)) <= float(max_v)
            )
        elif t == "min":
            min_v = filters.get(f"{k}_min", defaults.get("default_min", d.default_min))
            if hasattr(min_v, "tolist"):
                min_v = min_v.tolist()
            def_min = defaults.get("default_min", d.default_min)
            if hasattr(def_min, "tolist"):
                def_min = def_min.tolist()
            nan_fill = def_min if def_min != -np.inf else 0.0
            if k == "face_sim":
                has = ~np.isnan(arr)
                m = np.ones(num_frames, dtype=bool)
                m[has] = arr[has] >= float(min_v)
                if filters.get("require_face_match"):
                    m &= has
                mask &= m
            else:
                mask &= np.nan_to_num(arr, nan=float(nan_fill)) >= float(min_v)
    for i in np.where(~mask)[0]:
        for d in defs:
            k, t = d.key, d.filter_type
            arr = metric_arrays.get(k)
            if arr is None:
                continue
            v = arr[i]
            if d.enabled_key and not filters.get(d.enabled_key):
                continue
            defaults = getattr(config, f"filter_default_{k}", {})
            if t == "range":
                min_v, max_v = (
                    filters.get(f"{k}_min", defaults.get("default_min", d.default_min)),
                    filters.get(f"{k}_max", defaults.get("default_max", d.default_max)),
                )
                if not np.isnan(v):
                    if v < min_v:
                        reasons[filenames[i]].append(d.reason_range or d.reason_low or f"{k}_low")
                    if v > max_v:
                        reasons[filenames[i]].append(d.reason_range or d.reason_high or f"{k}_high")
            elif t == "min":
                min_v = filters.get(f"{k}_min", defaults.get("default_min", d.default_min))
                if not np.isnan(v) and v < min_v:
                    reasons[filenames[i]].append(d.reason_low or f"{k}_low")
                elif k == "face_sim" and filters.get("require_face_match") and np.isnan(v):
                    reasons[filenames[i]].append(d.reason_missing or "face_missing")
    return mask, reasons


def apply_all_filters_vectorized(
    all_frames_data: List[Dict[str, Any]],
    filters: Dict[str, Any],
    config: "Config",
    thumbnail_manager: Optional["ThumbnailManager"] = None,
    output_dir: Optional[str] = None,
) -> tuple[list, list, Counter, dict]:
    if not all_frames_data:
        return [], [], Counter(), {}
    metric_arrays = _extract_metric_arrays(all_frames_data, config)

    # Deduplication
    dedup_mask, reasons = apply_deduplication_filter(all_frames_data, filters, thumbnail_manager, config, output_dir)

    metric_mask, metric_reasons = _apply_metric_filters(all_frames_data, metric_arrays, filters, config)
    for fname, r_list in metric_reasons.items():
        reasons[fname].extend(r_list)
    kept_mask = dedup_mask & metric_mask
    return (
        [all_frames_data[i] for i in np.where(kept_mask)[0]],
        [all_frames_data[i] for i in np.where(~kept_mask)[0]],
        Counter(r for r_list in reasons.values() for r in r_list),
        reasons,
    )
