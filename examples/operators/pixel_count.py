"""Example operator demonstrating the plugin pattern."""
from core.operators import Operator, OperatorConfig, OperatorContext, OperatorResult
from core.operators import register_operator
import numpy as np

@register_operator
class PixelCountOperator:
    @property
    def config(self) -> OperatorConfig:
        return OperatorConfig(
            name="pixel_count",
            display_name="Non-Black Pixel Count",
            category="debug",
            default_enabled=False,  # Not for production
        )
    
    def execute(self, ctx: OperatorContext) -> OperatorResult:
        # Sum across channels to find non-black pixels
        # Any pixel where sum > 0 is considered non-black
        if ctx.image_rgb.size == 0:
             return OperatorResult(metrics={"pixel_count": 0.0})

        count = np.count_nonzero(ctx.image_rgb.sum(axis=-1))
        return OperatorResult(metrics={"pixel_count": float(count)})
