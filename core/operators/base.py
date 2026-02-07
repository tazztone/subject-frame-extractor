"""
Operator Protocol and Base Types for Analysis Operators.

This module defines the core abstractions for the extensible operator pattern:
- `OperatorConfig`: Metadata and UI configuration for an operator
- `OperatorContext`: Input context passed to operators during execution
- `OperatorResult`: Standardized return type with metrics and error handling
- `Operator`: Protocol defining the operator interface

Example usage:
    from core.operators import Operator, OperatorContext, OperatorResult, OperatorConfig
    
    class MyOperator:
        @property
        def config(self) -> OperatorConfig:
            return OperatorConfig(name="my_metric", display_name="My Metric")
        
        def execute(self, ctx: OperatorContext) -> OperatorResult:
            score = compute_something(ctx.image_rgb)
            return OperatorResult(metrics={"my_metric_score": score})
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    pass


@dataclass
class OperatorConfig:
    """
    Configuration and metadata for an operator.
    
    Attributes:
        name: Machine-readable identifier (e.g., "sharpness")
        display_name: Human-readable label for UI (e.g., "Sharpness Score")
        category: Grouping category ("quality", "face", "composition")
        default_enabled: Whether operator runs by default
        requires_mask: Whether operator benefits from subject mask
        requires_face: Whether operator requires face detection
        min_value: Minimum expected value (for UI sliders)
        max_value: Maximum expected value (for UI sliders)
        description: Tooltip/help text for UI
    """

    name: str
    display_name: str
    category: str = "quality"
    default_enabled: bool = True
    requires_mask: bool = False
    requires_face: bool = False
    min_value: float = 0.0
    max_value: float = 100.0
    description: str = ""


@dataclass
class OperatorContext:
    """
    Execution context passed to operators.
    
    Attributes:
        image_rgb: Input image as RGB numpy array (H, W, 3)
        mask: Optional subject mask as grayscale array (H, W)
        config: Global application Config object
        model_registry: Registry for lazy loading ML models
        logger: Application logger
        shared_data: Dictionary for sharing data between operators (e.g., face detections)
        params: Operator-specific parameters
    """

    image_rgb: np.ndarray
    mask: Optional[np.ndarray] = None
    config: Optional[Any] = None
    model_registry: Optional[Any] = None
    logger: Optional[Any] = None
    shared_data: dict = field(default_factory=dict)
    params: dict = field(default_factory=dict)


@dataclass
class OperatorResult:
    """
    Standardized result from operator execution.
    
    Attributes:
        metrics: Dict of computed numerical values (e.g., {"sharpness_score": 85.2})
        data: Dict of non-numerical data (e.g., {"phash": "a1b2c3d4"})
        error: Error message if execution failed, None otherwise
        warnings: List of non-fatal warning messages
    """

    metrics: dict[str, float] = field(default_factory=dict)
    data: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    warnings: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        """Returns True if no error occurred."""
        return self.error is None


@runtime_checkable
class Operator(Protocol):
    """
    Protocol defining the operator interface.
    
    All analysis operators must implement:
    - config: Property returning OperatorConfig metadata
    - execute: Method computing metrics from OperatorContext
    
    Optionally implement for stateful operators:
    - initialize: Called once with app config for model loading
    - cleanup: Called on shutdown for resource cleanup
    """

    @property
    def config(self) -> OperatorConfig:
        """Returns operator configuration and metadata."""
        ...

    def execute(self, ctx: OperatorContext) -> OperatorResult:
        """
        Execute the operator on the given context.
        
        Args:
            ctx: OperatorContext with image and optional mask/config
            
        Returns:
            OperatorResult with computed metrics or error
        """
        ...

    def initialize(self, config: Any) -> None:
        """
        Optional: Initialize operator with app config.
        
        Called once before first execution. Use for loading ML models.
        Default implementation does nothing.
        """
        ...

    def cleanup(self) -> None:
        """
        Optional: Clean up operator resources.
        
        Called on application shutdown. Use for releasing GPU memory.
        Default implementation does nothing.
        """
        ...
