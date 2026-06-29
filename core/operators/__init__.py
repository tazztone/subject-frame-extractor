"""
Extensible Operator Framework for Image Analysis.

This package provides a plugin-like architecture for adding new image analysis
metrics without modifying the core pipeline code.

Quick Start:
    # Create a new operator
    from core.operators import Operator, OperatorConfig, OperatorContext, OperatorResult
    from core.operators import register_operator

    @register_operator
    class MyMetricOperator:
        @property
        def config(self) -> OperatorConfig:
            return OperatorConfig(
                name="my_metric",
                display_name="My Custom Metric",
            )

        def execute(self, ctx: OperatorContext) -> OperatorResult:
            score = compute_score(ctx.image_rgb)
            return OperatorResult(metrics={"my_metric_score": score})

    # List all operators
    from core.operators import OperatorRegistry
    for cfg in OperatorRegistry.list_all():
        print(f"{cfg.name}: {cfg.display_name}")

    # Run operators on an image
    from core.operators import OperatorRegistry, OperatorContext
    ctx = OperatorContext(image_rgb=image_rgb, config=app_config)
    results = OperatorRegistry.execute(ctx)
"""

from core.operators.base import (
    FilterDefinition,
    Operator,
    OperatorConfig,
    OperatorContext,
    OperatorResult,
)
from core.operators.registry import (
    OperatorRegistry,
    discover_operators,
    register_operator,
)

# Auto-discover all operators in this package
discover_operators()

__all__ = [
    "Operator",
    "OperatorConfig",
    "OperatorContext",
    "OperatorResult",
    "FilterDefinition",
    "OperatorRegistry",
    "register_operator",
    "discover_operators",
]
