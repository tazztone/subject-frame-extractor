import cv2
import numpy as np
import math
from core.operators import OperatorConfig, OperatorResult, OperatorContext, register_operator


@register_operator
class EntropyOperator:
    @property
    def config(self) -> OperatorConfig:
        return OperatorConfig(
            name="entropy",
            display_name="Shannon Entropy",
            category="quality",
            description="Measures image information content (Shannon entropy).",
            min_value=0.0,
            max_value=100.0,
            requires_mask=True,
        )

    def execute(self, ctx: OperatorContext) -> OperatorResult:
        try:
            gray = cv2.cvtColor(ctx.image_rgb, cv2.COLOR_RGB2GRAY)
            
            # Use masking if available
            pixels = gray
            if ctx.mask is not None:
                active_mask = ctx.mask > 128
                if np.sum(active_mask) >= 100:
                    pixels = gray[active_mask]
            
            if pixels.size == 0:
                return OperatorResult(metrics={"entropy_score": 0.0})

            # Histogram
            hist = cv2.calcHist([pixels], [0], None, [256], [0, 256])
            
            # Normalize
            total_pixels = pixels.size
            prob = hist.ravel() / total_pixels
            
            # Remove zero entries
            prob = prob[prob > 0]
            
            # Shannon Entropy: -sum(p * log2(p))
            entropy = -np.sum(prob * np.log2(prob))
            
            # Scale: Max entropy for 8-bit is 8.0. Map 8.0 -> 100.0
            entropy_score = (entropy / 8.0) * 100.0
            
            # Clamp to 100 just in case
            entropy_score = min(100.0, max(0.0, entropy_score))
            
            return OperatorResult(metrics={"entropy_score": entropy_score})
            
        except Exception as e:
            return OperatorResult(success=False, error=str(e))
