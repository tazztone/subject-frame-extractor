from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
import logging

from core.operators.base import OperatorContext
from core.operators.sharpness import SharpnessOperator
from core.operators.entropy import EntropyOperator
from core.operators.niqe import NIQEOperator
from core.operators.face_metrics import FaceMetricsOperator

logger = logging.getLogger(__name__)

def score_photo(preview_path: Path, weights: Dict[str, float]) -> Dict[str, Any]:
    """
    Computes quality scores for a photo preview using configured weights.

    Args:
        preview_path: Path to the preview image.
        weights: Dictionary of metric weights (e.g., {"sharpness": 0.5}).

    Returns:
        Dictionary containing "scores" (raw metrics) and "quality_score" (weighted).
    """
    if not preview_path.exists():
        logger.error(f"Preview not found: {preview_path}")
        return {"scores": {}, "quality_score": 0.0}

    # Load image
    # cv2.imread loads as BGR, we need RGB for OperatorContext usually
    img_bgr = cv2.imread(str(preview_path))
    if img_bgr is None:
        logger.error(f"Failed to load preview: {preview_path}")
        return {"scores": {}, "quality_score": 0.0}
    
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Create Context (no mask for basic photo mode yet)
    ctx = OperatorContext(image_rgb=img_rgb)

    scores = {}
    
    # helper to run op
    def run_op(op_cls, metric_key, weight_key):
        if weights.get(weight_key, 0) > 0:
            try:
                op = op_cls()
                # Initialize if needed (stateless usually doesn't need it, but good practice)
                op.initialize(None) 
                res = op.execute(ctx)
                if res.success:
                    # Extract the primary metric
                    # Operators return dicts like {"sharpness_score": 88.0}
                    # We need to know the specific key they return or just take the first one
                    if res.metrics:
                        val = next(iter(res.metrics.values()))
                        scores[metric_key] = float(val)
                else:
                    logger.warning(f"{weight_key} operator failed: {res.error}")
            except Exception as e:
                logger.error(f"Failed to run {weight_key}: {e}")

    # Sharpness
    run_op(SharpnessOperator, "sharpness", "sharpness")

    # Entropy
    run_op(EntropyOperator, "entropy", "entropy")
        
    # NIQE (Naturalness)
    # NIQEOperator likely returns "niqe_score"
    run_op(NIQEOperator, "niqe", "niqe")

    # Face Prominence
    # FaceMetricsOperator likely returns "face_prominence" or similar
    # We map 'face' weight to this operator
    run_op(FaceMetricsOperator, "face", "face")

    # Calculate Weighted Score
    total_weight = sum(weights.values())
    if total_weight == 0:
        final_score = 0.0
    else:
        weighted_sum = 0.0
        for metric, w in weights.items():
            val = scores.get(metric, 0.0)
            
            # Normalization logic (if operators don't return 0-100)
            # Assuming standard operators return 0-100 based on their config/impl.
            # SharpnessOperator in existing code does normalize to 0-100.
            # EntropyOperator does too.
            # Check NIQE/Face in future if needed.
            
            weighted_sum += val * w
            
        final_score = weighted_sum / total_weight

    return {
        "scores": scores,
        "quality_score": final_score
    }

def apply_scores_to_photos(photos: list, weights: Dict[str, float]) -> list:
    """Batch scoring for photos."""
    for p in photos:
        res = score_photo(p["preview"], weights)
        p["scores"] = res["scores"]
        p["scores"]["quality_score"] = res["quality_score"]
    return photos
