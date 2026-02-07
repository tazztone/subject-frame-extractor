import torch
import numpy as np
from core.operators import OperatorConfig, OperatorResult, OperatorContext, register_operator

@register_operator
class NiqeOperator:
    def __init__(self):
        self.model = None
        self.device = None

    @property
    def config(self) -> OperatorConfig:
        return OperatorConfig(
            name="niqe",
            display_name="NIQE Score",
            category="quality",
            description="Natural Image Quality Evaluator (no-reference). Score 0-100 (higher is better).",
            min_value=0.0,
            max_value=100.0,
            requires_mask=False, # Masking handled internally if provided
        )

    def initialize(self, config):
        """Load pyiqa model."""
        try:
            import pyiqa
        except ImportError:
            # We don't fail here, but execute will fail if called
            return

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            # Create metric with device
            self.model = pyiqa.create_metric("niqe", device=torch.device(self.device))
        except Exception as e:
            # Log error? Or re-raise?
            # Operator framework swallows exceptions in execute, but initialize?
            # We'll leave self.model as None and handle in execute.
            pass

    def cleanup(self):
        """Release resources."""
        self.model = None
        if self.device == "cuda":
            torch.cuda.empty_cache()

    def execute(self, ctx: OperatorContext) -> OperatorResult:
        if self.model is None:
            # Try initializing if missing? Or fail?
            # Lifecycle implies initialize called explicitly.
            # If still None, it failed or wasn't called.
            return OperatorResult(error="NIQE not initialized or pyiqa missing")

        try:
            # Preprocess: RGB numpy (H, W, C) -> Tensor (1, C, H, W)
            # Normalize to 0-1
            img_rgb = ctx.image_rgb
            
            # Handle masking if provided
            if ctx.mask is not None:
                # Mask is (H, W) uint8 0 or 255
                # We need to zero out non-subject areas?
                # NIQE is global. If we mask, we might introduce artifacts at boundaries.
                # Project legacy logic matches mask by zeroing out background?
                # Legacy: 
                # active_mask_full = (cv2.resize(mask...) > 128)
                # rgb_image = np.where(mask_3ch, rgb_image, 0)
                # Yes, zero out background.
                import cv2
                mask_h, mask_w = ctx.mask.shape
                img_h, img_w = img_rgb.shape[:2]
                
                if (mask_h, mask_w) != (img_h, img_w):
                     # Resize mask to fit image (e.g. if mask is low res)
                     # Although usually they match in frame processing.
                     mask_resized = cv2.resize(ctx.mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
                else:
                     mask_resized = ctx.mask

                active_mask = (mask_resized > 128)
                # Apply mask (broadcasting)
                img_rgb = img_rgb * active_mask[:, :, np.newaxis]

            # Convert to tensor
            img_tensor = torch.from_numpy(img_rgb).float() / 255.0
            # (H, W, C) -> (C, H, W) -> (1, C, H, W)
            img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
            
            # Move to device
            device_obj = self.device 
            # Note: self.model might have its own device concept if pyiqa wraps it?
            # But we passed device to create_metric.
            
            with ctx.model_registry.locked("niqe"): # Use a stable key for NIQE
                with torch.no_grad():
                    # PyIQA forward
                    # Depending on pyiqa version, it might expect different scale.
                    # Usually 0-1 float.
                    res = self.model(img_tensor.to(device_obj))
                    niqe_raw = float(res)

            # Normalization
            # Plan Default: 100 - (raw * 2)
            # Allow overrides
            offset = 100.0 # From plan: implied max
            scale = 2.0    # From plan
            
            if ctx.config:
                if hasattr(ctx.config, "quality_niqe_offset"):
                    offset = ctx.config.quality_niqe_offset
                elif hasattr(ctx.config, "niqe_offset"): # Alternative naming?
                    offset = ctx.config.niqe_offset
                    
                if hasattr(ctx.config, "quality_niqe_scale_factor"):
                    scale = ctx.config.quality_niqe_scale_factor

            niqe_score = max(0.0, min(100.0, (offset - niqe_raw) * scale))
            
            return OperatorResult(metrics={
                "niqe": niqe_score / 100.0,
                "niqe_score": niqe_score
            })

        except Exception as e:
            return OperatorResult(error=str(e))
