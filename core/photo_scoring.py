from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
from concurrent.futures import ThreadPoolExecutor

from core.operators import OperatorRegistry, run_operators, OperatorContext
from core.config import Config

logger = logging.getLogger(__name__)

def score_photo(
    preview_path: Path, 
    weights: Dict[str, float], 
    config: Optional[Config] = None,
    model_registry: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Computes quality scores for a photo preview using the Operator engine.

    Args:
        preview_path: Path to the preview image.
        weights: Dictionary of metric weights.
        config: Application configuration.
        model_registry: Model registry for operators.

    Returns:
        Dictionary containing "scores" and "quality_score".
    """
    if not preview_path.exists():
        logger.error(f"Preview not found: {preview_path}")
        return {"scores": {}, "quality_score": 0.0}

    img_bgr = cv2.imread(str(preview_path))
    if img_bgr is None:
        logger.error(f"Failed to load preview: {preview_path}")
        return {"scores": {}, "quality_score": 0.0}
    
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Determine which operators to run based on weights
    # Map weight keys to operator names
    op_map = {
        "sharpness": "sharpness",
        "entropy": "entropy",
        "niqe": "niqe",
        "face": "face_prominence" # Map 'face' weight to prominence operator
    }
    
    operators_to_run = [op_map[k] for k in weights.keys() if weights.get(k, 0) > 0 and k in op_map]
    if "quality_score" not in operators_to_run:
        operators_to_run.append("quality_score")

    # Run Operators
    # We don't have a face_bbox yet, so prominence might be 0 unless we add a detection op first
    # For now, let's include face_sim if face weight is on to get detection
    if "face" in weights and weights["face"] > 0:
        if "face_sim" not in operators_to_run:
            operators_to_run.insert(0, "face_sim")

    results = run_operators(
        image_rgb=img_rgb,
        config=config,
        model_registry=model_registry,
        operators=operators_to_run,
        logger=logger
    )

    scores = {}
    for op_name, res in results.items():
        if res.success:
            # Map back to weights keys
            if op_name == "sharpness": scores["sharpness"] = res.metrics.get("sharpness_score", 0.0)
            elif op_name == "entropy": scores["entropy"] = res.metrics.get("entropy_score", 0.0)
            elif op_name == "niqe": scores["niqe"] = res.metrics.get("niqe_score", 0.0)
            elif op_name == "face_prominence": scores["face"] = res.metrics.get("face_prominence_score", 0.0)
            elif op_name == "quality_score": final_quality_score = res.metrics.get("quality_score", 0.0)

    # If quality_score operator didn't run or we want to use custom weights from photo mode
    # calculate weighted average manually
    total_weight = sum(weights.values())
    if total_weight > 0:
        weighted_sum = sum(scores.get(k, 0.0) * w for k, w in weights.items())
        final_score = weighted_sum / total_weight
    else:
        final_score = 0.0

    return {
        "scores": scores,
        "quality_score": final_score
    }

def apply_scores_to_photos(
    photos: List[Dict[str, Any]], 
    weights: Dict[str, float], 
    config: Optional[Config] = None,
    model_registry: Optional[Any] = None
) -> List[Dict[str, Any]]:
    """Parallel scoring for photos."""
    num_workers = min(len(photos), 4) # Limit workers to avoid OOM
    
    def process_one(p):
        res = score_photo(p["preview"], weights, config, model_registry)
        p["scores"] = res["scores"]
        p["scores"]["quality_score"] = res["quality_score"]
        return p

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(process_one, photos))
        
    return results
