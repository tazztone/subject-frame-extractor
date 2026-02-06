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
    from core.operators import run_operators
    results = run_operators(image_rgb, mask=None, config=app_config)
"""

from core.operators.base import (
    Operator,
    OperatorConfig,
    OperatorContext,
    OperatorResult,
)
from core.operators.registry import (
    OperatorRegistry,
    register_operator,
    run_operators,
)

# Import operators to trigger registration
from core.operators.sharpness import SharpnessOperator
from core.operators.simple_cv import EdgeStrengthOperator, ContrastOperator, BrightnessOperator
from core.operators.entropy import EntropyOperator
from core.operators.niqe import NiqeOperator
from core.operators.face_metrics import EyesOpenOperator, FacePoseOperator

__all__ = [
    "Operator",
    "OperatorConfig",
    "OperatorContext",
    "OperatorResult",
    "OperatorRegistry",
    "register_operator",
    "run_operators",
    "SharpnessOperator",
]
