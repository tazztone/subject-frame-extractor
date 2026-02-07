"""
Sharpness Operator - Measures image sharpness using Laplacian variance.

This is the first concrete operator implementation, demonstrating the
operator pattern for image quality metrics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from core.operators.base import OperatorConfig, OperatorContext, OperatorResult
from core.operators.registry import register_operator

if TYPE_CHECKING:
    pass


@register_operator
class SharpnessOperator:
    """
    Computes sharpness score using Laplacian variance.
    
    Higher variance in the Laplacian indicates more edges and detail,
    which corresponds to a sharper image. Blurry images have low variance.
    """

    @property
    def config(self) -> OperatorConfig:
        """Returns operator configuration."""
        return OperatorConfig(
            name="sharpness",
            display_name="Sharpness Score",
            category="quality",
            default_enabled=True,
            requires_mask=True,  # Benefits from mask to focus on subject
            requires_face=False,
            min_value=0.0,
            max_value=100.0,
            description="Laplacian variance measuring image sharpness. Higher = sharper.",
        )

    def execute(self, ctx: OperatorContext) -> OperatorResult:
        """
        Compute sharpness score from image.
        
        Args:
            ctx: OperatorContext with image_rgb and optional mask
            
        Returns:
            OperatorResult with "sharpness_score" metric (0-100)
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(ctx.image_rgb, cv2.COLOR_RGB2GRAY)
            
            # Compute Laplacian
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            
            # Apply mask if provided
            if ctx.mask is not None and ctx.mask.size > 0:
                # Mask should be same size as image
                if ctx.mask.shape[:2] == gray.shape[:2]:
                    active_mask = ctx.mask > 128
                    if np.sum(active_mask) >= 100:  # Need minimum area for valid stats
                        laplacian = laplacian[active_mask]
            
            # Compute variance
            if laplacian.size == 0:
                return OperatorResult(
                    metrics={"sharpness_score": 0.0},
                    warnings=["Empty region after masking"],
                )
            
            variance = float(np.var(laplacian))
            
            # Get normalization scale from config
            scale = 2500.0  # Default
            if ctx.config is not None and hasattr(ctx.config, "sharpness_base_scale"):
                scale = ctx.config.sharpness_base_scale
            
            # Normalize to 0-100 range
            raw_normalized = min(1.0, variance / scale)
            score = raw_normalized * 100.0
            score = max(0.0, score)  # Ensure non-negative
            
            return OperatorResult(metrics={
                "sharpness": float(raw_normalized),
                "sharpness_score": score
            })

        except Exception as e:
            return OperatorResult(
                metrics={},
                error=f"Sharpness calculation failed: {e}",
            )

    def initialize(self, config: Any) -> None:
        """No initialization needed (stateless operator)."""
        pass

    def cleanup(self) -> None:
        """No cleanup needed (stateless operator)."""
        pass
