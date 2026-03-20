import torch

from core.operators import OperatorConfig, OperatorContext, OperatorResult, register_operator


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
            requires_mask=False,
            requires_tensor=True,
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
        except Exception:
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
            return OperatorResult(error="NIQE not initialized or pyiqa missing")

        if ctx.image_tensor is None:
            return OperatorResult(error="NIQE requires Torch tensor input (pre-computation failed)")

        try:
            img_tensor = ctx.image_tensor

            # Handle masking if provided
            if ctx.mask_tensor is not None:
                # Optimized Torch masking
                # ctx.mask_tensor is (1, 1, H, W) in range [0, 1]
                # ctx.image_tensor is (1, 3, H, W) in range [0, 1]
                mask_binary = (ctx.mask_tensor > 0.5).float()
                img_tensor = img_tensor * mask_binary

            with torch.no_grad():
                # PyIQA forward
                # It expects 0-1 float on the correct device
                res = self.model(img_tensor)
                niqe_raw = float(res.item())

            # Normalization
            # Plan Default: 100 - (raw * 2)
            offset = 100.0
            scale = 2.0

            if ctx.config:
                if hasattr(ctx.config, "quality_niqe_offset"):
                    offset = ctx.config.quality_niqe_offset
                elif hasattr(ctx.config, "niqe_offset"):
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
