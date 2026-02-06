import cv2
import numpy as np
from core.operators import OperatorConfig, OperatorResult, OperatorContext, register_operator
from core.config import Config


@register_operator
class EdgeStrengthOperator:
    @property
    def config(self) -> OperatorConfig:
        return OperatorConfig(
            name="edge_strength",
            display_name="Edge Strength",
            category="quality",
            description="Measures mean edge magnitude using Sobel filter.",
            min_value=0.0,
            max_value=100.0,
            requires_mask=True,
        )

    def execute(self, ctx: OperatorContext) -> OperatorResult:
        try:
            gray = cv2.cvtColor(ctx.image_rgb, cv2.COLOR_RGB2GRAY)
            
            # Match legacy behavior: compute on full image first?
            # Legacy code:
            # sobelx = ...
            # sobely = ...
            # edge_map = ...
            # mean_val = np.mean(edge_map)
            # Legacy code does NOT seem to use mask for Edge Strength calculation itself!
            # Line 202-205 in core/models.py uses `gray` directly, not `masked_gray`.
            # Wait, let's check core/models.py again.
            # Line 194: `masked_lap = laplacian[active_mask]` for Sharpness.
            # Line 202: `sobelx = cv2.Sobel(gray`...
            # Line 205: `mean_val = np.mean(edge_map)` -> NO MASK used in legacy for Edge Strength.
            # This seems like a legacy bug or feature.
            # However, for Parity, I MUST REPLICATE IT.
            # The config says `requires_mask=True` in my draft above, but if legacy doesn't use it, maybe I should set it to False
            # or just ignore it.
            # Actually, let's look at `core/models.py` again.
            # Line 202 `gray` is used. `active_mask` is defined at 181.
            # But `edge_map` calculation doesn't use `active_mask`. `np.mean(edge_map)` uses full image.
            
            # So I will replicate full image calculation.
            
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            edge_map = np.sqrt(sobelx**2 + sobely**2)
            mean_val = np.mean(edge_map)
            edge_strength = float(mean_val)
            
            # Scaling
            base_scale = 100.0 # Default
            if ctx.config and hasattr(ctx.config, "edge_strength_base_scale"):
                base_scale = ctx.config.edge_strength_base_scale
                
            if base_scale:
                edge_strength = min(100.0, (edge_strength / base_scale) * 100.0)
                
            return OperatorResult(metrics={"edge_strength_score": edge_strength})
            
        except Exception as e:
            return OperatorResult(success=False, error=str(e))


@register_operator
class ContrastOperator:
    @property
    def config(self) -> OperatorConfig:
        return OperatorConfig(
            name="contrast",
            display_name="Contrast",
            category="quality",
            description="Measures contrast (std/mean intensity).",
            min_value=0.0,
            max_value=100.0,
            requires_mask=True,
        )

    def execute(self, ctx: OperatorContext) -> OperatorResult:
        try:
            gray = cv2.cvtColor(ctx.image_rgb, cv2.COLOR_RGB2GRAY)
            
            # Legacy logic:
            # pixels = gray[active_mask] if active_mask is not None else gray
            
            pixels = gray
            if ctx.mask is not None:
                # Legacy mask logic: active_mask = (mask > 128)
                # And check if np.sum(active_mask) < 100 -> active_mask = None
                active_mask = ctx.mask > 128
                if np.sum(active_mask) >= 100:
                    pixels = gray[active_mask]
            
            if pixels.size > 0:
                mean_br = np.mean(pixels)
                std_br = np.std(pixels)
            else:
                mean_br, std_br = 0.0, 0.0
                
            contrast = float(std_br) / (mean_br + 1e-7)
            
            # Scaling
            contrast_clamp = 50.0 # Default from mock_config in tests, legacy implies available in config
            if ctx.config and hasattr(ctx.config, "quality_contrast_clamp"):
                contrast_clamp = ctx.config.quality_contrast_clamp
            elif getattr(Config, "quality_contrast_clamp", None): # Fallback if Config class has default
                # But we don't have Config class instance here easily unless passed.
                # We'll rely on ctx.config usually being populated.
                pass
                
            contrast_val = float(contrast)
            contrast_scaled = (min(contrast_val, contrast_clamp) / contrast_clamp)
            
            # Legacy stores as 0-1, verify scaling. 
            # `_calculate_and_store_score` multiplies by 100.
            # So here we return 0-100?
            # Plan 2.1 says "Return: {'contrast_score': 0-100}"
            # Model.py: `setattr(self.metrics, f"{name}_score", float(normalized_value * 100))`
            # So yes, return 0-100.
            
            return OperatorResult(metrics={"contrast_score": contrast_scaled * 100.0})
            
        except Exception as e:
            return OperatorResult(success=False, error=str(e))


@register_operator
class BrightnessOperator:
    @property
    def config(self) -> OperatorConfig:
        return OperatorConfig(
            name="brightness",
            display_name="Brightness",
            category="quality",
            description="Measures mean brightness.",
            min_value=0.0,
            max_value=100.0,
            requires_mask=True,
        )

    def execute(self, ctx: OperatorContext) -> OperatorResult:
        try:
            gray = cv2.cvtColor(ctx.image_rgb, cv2.COLOR_RGB2GRAY)
            
            # Same masking logic as contrast
            pixels = gray
            if ctx.mask is not None:
                active_mask = ctx.mask > 128
                if np.sum(active_mask) >= 100:
                    pixels = gray[active_mask]
            
            mean_br = np.mean(pixels) if pixels.size > 0 else 0.0
            
            # Legacy: float(mean_br) / 255.0
            brightness = float(mean_br) / 255.0
            
            return OperatorResult(metrics={"brightness_score": brightness * 100.0})
            
        except Exception as e:
            return OperatorResult(success=False, error=str(e))
