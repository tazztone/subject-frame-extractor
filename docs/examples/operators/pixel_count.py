"""Example operator demonstrating the plugin pattern."""

import numpy as np

from core.operators import OperatorConfig, OperatorContext, OperatorResult, register_operator


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


if __name__ == "__main__":
    # Example usage
    operator = PixelCountOperator()

    # Access the config property to demonstrate its usage
    print(f"Operator Name: {operator.config.name}")
    print(f"Display Name: {operator.config.display_name}")

    # Create a dummy image (10x10 RGB, half black, half white)
    dummy_image = np.zeros((10, 10, 3), dtype=np.uint8)
    dummy_image[:, 5:] = 255  # Set right half to white

    # Create context and execute
    context = OperatorContext(image_rgb=dummy_image)
    result = operator.execute(context)

    print(f"Non-black pixels: {result.metrics['pixel_count']}")
