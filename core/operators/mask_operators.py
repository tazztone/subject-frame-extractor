import cv2
import numpy as np
from PIL import Image
from core.operators import OperatorConfig, OperatorResult, OperatorContext, register_operator

@register_operator
class PhashOperator:
    @property
    def config(self) -> OperatorConfig:
        return OperatorConfig(
            name="phash",
            display_name="Perceptual Hash",
            category="quality",
            description="Computes a perceptual hash for deduplication.",
        )

    def execute(self, ctx: OperatorContext) -> OperatorResult:
        try:
            import imagehash
            pil_img = Image.fromarray(ctx.image_rgb)
            phash = str(imagehash.phash(pil_img))
            
            return OperatorResult(data={"phash": phash})
        except ImportError:
            return OperatorResult(error="imagehash missing")


@register_operator
class SubjectMaskAreaOperator:
    @property
    def config(self) -> OperatorConfig:
        return OperatorConfig(
            name="mask_area",
            display_name="Subject Mask Area",
            category="quality",
            description="Measures percentage of image covered by the subject mask.",
            requires_mask=True,
        )

    def execute(self, ctx: OperatorContext) -> OperatorResult:
        if ctx.mask is None:
            return OperatorResult(metrics={"mask_area_pct": 0.0}, warnings=["Missing mask"])
            
        h, w = ctx.mask.shape[:2]
        total_area = h * w
        if total_area == 0:
            return OperatorResult(metrics={"mask_area_pct": 0.0})
            
        mask_pixels = np.sum(ctx.mask > 128)
        pct = (mask_pixels / total_area) * 100.0
        
        return OperatorResult(metrics={"mask_area_pct": float(pct)})
