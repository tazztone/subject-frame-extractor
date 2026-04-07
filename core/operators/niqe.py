# Lazy torch import in execute

from core.operators import OperatorConfig, OperatorContext, OperatorResult, register_operator
from core.utils.device import empty_cache


@register_operator
class NiqeOperator:
    def __init__(self):
        self.model = None
        self.device = None

    @property
    def config(self) -> OperatorConfig:
        return OperatorConfig(
            name="niqe",
            description="NIQE (Natural Image Quality Evaluator) no-reference metric",
            display_name="NIQE Quality",
            requires_tensor=True,
        )

    def initialize(self, config: OperatorConfig):
        try:
            import pyiqa

            self.device = getattr(config, "device", "cpu")
            self.model = pyiqa.create_metric("niqe", device=self.device)
        except ImportError:
            # logger is not easily available here, assuming initialized via pipeline
            pass
        except Exception:
            pass

    def cleanup(self):
        self.model = None
        if self.device == "cuda":
            empty_cache()

    def execute(self, ctx: OperatorContext) -> OperatorResult:
        import torch

        if self.model is None:
            return OperatorResult(error="NIQE not initialized or pyiqa missing")

        if ctx.image_tensor is None:
            return OperatorResult(error="NIQE requires Torch tensor input (pre-computation failed)")

        try:
            img_tensor = ctx.image_tensor

            # Handle masking if provided
            if ctx.mask_tensor is not None:
                mask = (ctx.mask_tensor > 0.5).float()
                img_tensor = img_tensor * mask

            with torch.no_grad():
                niqe_raw = self.model(img_tensor).item()

            # Normalize NIQE (lower is better, typically 0-20 for good images)
            # Default mapping: offset=20, scale=5 -> (20 - 5)*5 = 75
            offset = 20.0
            scale = 5.0

            if ctx.config:
                # Fallback to niqe_offset if quality_niqe_offset is missing
                if hasattr(ctx.config, "quality_niqe_offset"):
                    offset = ctx.config.quality_niqe_offset
                elif hasattr(ctx.config, "niqe_offset"):
                    offset = ctx.config.niqe_offset

                if hasattr(ctx.config, "quality_niqe_scale_factor"):
                    scale = ctx.config.quality_niqe_scale_factor

            niqe_score = max(0.0, min(100.0, (offset - niqe_raw) * scale))

            return OperatorResult(metrics={"niqe": niqe_score / 100.0, "niqe_score": niqe_score})

        except Exception as e:
            return OperatorResult(error=str(e))
