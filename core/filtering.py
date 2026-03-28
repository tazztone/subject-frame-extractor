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


def load_and_prep_filter_data(output_dir: str, get_all_filter_keys: Callable, config: "Config") -> tuple[list, dict]:
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
        cfg = metric_configs.get(k)
        if not cfg:
            continue
        path, alt = cfg.get("path"), cfg.get("alt_path")
        vals = []
        for f in all_frames:
            v = None
            if path:
                v = f.get(path[0]) if len(path) == 1 else f.get(path[0], {}).get(path[1])
            if v is None and alt:
                v = f.get(alt[0]) if len(alt) == 1 else f.get(alt[0], {}).get(alt[1])
            if v is not None:
                vals.append(v)
        vals = np.asarray(vals, dtype=float)
        if vals.size > 0:
            counts, bins = np.histogram(vals, bins=50, range=cfg.get("range", (0, 100)))
            metric_values[k], metric_values[f"{k}_hist"] = vals.tolist(), (counts.tolist(), bins.tolist())
    return all_frames, metric_values


def build_all_metric_svgs(per_metric_values: dict, get_all_filter_keys: Callable, logger: "AppLogger") -> dict:
    svgs = {}
    for k in get_all_filter_keys():
        if h := per_metric_values.get(f"{k}_hist"):
            svgs[k] = histogram_svg(h, logger=logger)
    return svgs


def _extract_metric_arrays(all_frames_data: List[Dict[str, Any]], config: "Config") -> Dict[str, np.ndarray]:
    keys = [k.replace("quality_weights_", "") for k in config.model_dump().keys() if k.startswith("quality_weights_")]
    sources = {
        **{k: ("metrics", f"{k}_score") for k in keys},
        "quality_score": ("metrics", "quality_score"),
        "face_sim": ("face_sim",),
        "mask_area_pct": ("mask_area_pct",),
        "eyes_open": ("metrics", "eyes_open"),
        "yaw": ("metrics", "yaw"),
        "pitch": ("metrics", "pitch"),
    }
    metric_arrays = {}
    for key, path in sources.items():
        vals = [
            f.get(path[0], np.nan) if len(path) == 1 else f.get(path[0], {}).get(path[1], np.nan)
            for f in all_frames_data
        ]
        metric_arrays[key] = np.array(vals, dtype=np.float32)
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
    keys = [k.replace("quality_weights_", "") for k in config.model_dump().keys() if k.startswith("quality_weights_")]
    defs = [
        *[{"key": k, "type": "range"} for k in keys],
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
    mask = np.ones(num_frames, dtype=bool)
    for d in defs:
        k, t = d["key"], d["type"]
        if d.get("enabled_key") and not filters.get(d["enabled_key"]):
            continue
        arr = metric_arrays.get(k)
        if arr is None:
            continue
        defaults = getattr(config, f"filter_default_{k}", {})
        if t == "range":
            min_v = filters.get(f"{k}_min", defaults.get("default_min", -np.inf))
            max_v = filters.get(f"{k}_max", defaults.get("default_max", np.inf))
            if hasattr(min_v, "tolist"):
                min_v = min_v.tolist()  # Handle mocks
            if hasattr(max_v, "tolist"):
                max_v = max_v.tolist()
            def_min = defaults.get("default_min", -np.inf)
            if hasattr(def_min, "tolist"):
                def_min = def_min.tolist()
            nan_fill = def_min if def_min != -np.inf else 0.0
            mask &= (np.nan_to_num(arr, nan=float(nan_fill)) >= float(min_v)) & (
                np.nan_to_num(arr, nan=float(nan_fill)) <= float(max_v)
            )
        elif t == "min":
            min_v = filters.get(f"{k}_min", defaults.get("default_min", -np.inf))
            if hasattr(min_v, "tolist"):
                min_v = min_v.tolist()
            def_min = defaults.get("default_min", -np.inf)
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
            arr = metric_arrays.get(d["key"])
            if arr is None:
                continue
            v = arr[i]
            k, t = d["key"], d["type"]
            if d.get("enabled_key") and not filters.get(d["enabled_key"]):
                continue
            defaults = getattr(config, f"filter_default_{k}", {})
            if t == "range":
                min_v, max_v = (
                    filters.get(f"{k}_min", defaults.get("default_min", -np.inf)),
                    filters.get(f"{k}_max", defaults.get("default_max", np.inf)),
                )
                if not np.isnan(v):
                    if v < min_v:
                        reasons[filenames[i]].append(d.get("reason_range") or d.get("reason_low", f"{k}_low"))
                    if v > max_v:
                        reasons[filenames[i]].append(d.get("reason_range") or d.get("reason_high", f"{k}_high"))
            elif t == "min":
                min_v = filters.get(f"{k}_min", defaults.get("default_min", -np.inf))
                if not np.isnan(v) and v < min_v:
                    reasons[filenames[i]].append(d.get("reason_low", f"{k}_low"))
                elif k == "face_sim" and filters.get("require_face_match") and np.isnan(v):
                    reasons[filenames[i]].append(d.get("reason_missing", "face_missing"))
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
