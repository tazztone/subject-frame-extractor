"""
Operator Registry for managing and discovering operators.

Provides centralized registration, lifecycle management, and a bridge function
for integrating operators with the analysis pipeline.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Optional, Type
import importlib
import pkgutil

from core.operators.base import Operator, OperatorConfig, OperatorContext, OperatorResult

if TYPE_CHECKING:
    pass


class OperatorRegistry:
    """
    Central registry for operator instances.
    
    Manages operator registration, lifecycle (initialize/cleanup),
    and provides discovery mechanisms.
    """

    _operators: dict[str, Operator] = {}
    _initialized: set[str] = set()

    @classmethod
    def register(cls, operator: Operator) -> None:
        """
        Register an operator instance.
        
        Args:
            operator: Operator instance to register
        """
        name = operator.config.name
        cls._operators[name] = operator

    @classmethod
    def get(cls, name: str) -> Optional[Operator]:
        """
        Get an operator by name.
        
        Args:
            name: Operator name (from config.name)
            
        Returns:
            Operator instance or None if not found
        """
        return cls._operators.get(name)

    @classmethod
    def list_all(cls) -> list[OperatorConfig]:
        """
        List all registered operator configs.
        
        Returns:
            List of OperatorConfig for all registered operators
        """
        return [op.config for op in cls._operators.values()]

    @classmethod
    def list_names(cls) -> list[str]:
        """
        List all registered operator names.
        
        Returns:
            List of operator names
        """
        return list(cls._operators.keys())

    @classmethod
    def initialize_all(cls, config: Any) -> None:
        """
        Initialize all registered operators.
        
        Calls initialize() on operators that haven't been initialized yet.
        
        Args:
            config: Application Config object to pass to operators
        """
        for name, operator in cls._operators.items():
            if name not in cls._initialized:
                if hasattr(operator, "initialize"):
                    try:
                        operator.initialize(config)
                    except Exception:
                        pass  # Operators handle their own errors
                cls._initialized.add(name)

    @classmethod
    def cleanup_all(cls) -> None:
        """
        Clean up all registered operators.
        
        Calls cleanup() on all initialized operators.
        """
        for name in list(cls._initialized):
            operator = cls._operators.get(name)
            if operator and hasattr(operator, "cleanup"):
                try:
                    operator.cleanup()
                except Exception:
                    pass
            cls._initialized.discard(name)

    @classmethod
    def clear(cls) -> None:
        """Clear all registrations (useful for testing)."""
        cls._operators.clear()
        cls._initialized.clear()


def discover_operators(package_path: str = "core.operators") -> list[str]:
    """
    Auto-discover and register operators from a package.

    Args:
        package_path: Dotted module path to the package containing operators

    Returns:
        List of names of all discovered operators
    """
    try:
        package = importlib.import_module(package_path)
    except ImportError:
        return []

    if hasattr(package, "__path__"):
        for importer, modname, ispkg in pkgutil.iter_modules(package.__path__):
            # Skip infrastructure modules to avoid circular imports or re-registration loops
            if modname not in ("base", "registry", "__init__"):
                try:
                    importlib.import_module(f"{package_path}.{modname}")
                except Exception:
                    # Log error but continue discovery
                    pass

    return OperatorRegistry.list_names()

def register_operator(cls: Type[Operator]) -> Type[Operator]:
    """
    Decorator to register an operator class.
    
    Instantiates the class and registers it with OperatorRegistry.
    
    Usage:
        @register_operator
        class MyOperator:
            ...
    """
    instance = cls()
    OperatorRegistry.register(instance)
    return cls


def run_operators(
    image_rgb: Any,
    mask: Optional[Any] = None,
    config: Optional[Any] = None,
    operators: Optional[list[str]] = None,
    params: Optional[dict] = None,
    model_registry: Optional[Any] = None,
    logger: Optional[Any] = None,
    shared_data: Optional[dict] = None,
) -> dict[str, OperatorResult]:
    """
    Bridge function to run operators on an image.
    
    This is the integration point for the analysis pipeline.
    
    Args:
        image_rgb: Input image as RGB numpy array
        mask: Optional subject mask
        config: Application Config object
        operators: List of operator names to run (None = all)
        params: Optional operator-specific parameters
        model_registry: Optional model registry for lazy loading
        logger: Optional logger
        shared_data: Optional dictionary for sharing data between operators
        
    Returns:
        Dict mapping operator names to their OperatorResult
    """
    results: dict[str, OperatorResult] = {}
    
    # Build context
    ctx = OperatorContext(
        image_rgb=image_rgb,
        mask=mask,
        config=config,
        model_registry=model_registry,
        logger=logger,
        shared_data=shared_data if shared_data is not None else {},
        params=params or {},
    )
    
    # Determine which operators to run
    if operators is None:
        operator_names = OperatorRegistry.list_names()
    else:
        operator_names = operators
    
    # Sort operator_names to ensure quality_score runs last if present
    if "quality_score" in operator_names:
        operator_names = [n for n in operator_names if n != "quality_score"]
        operator_names.append("quality_score")
    
    # Execute each operator
    for name in operator_names:
        operator = OperatorRegistry.get(name)
        if operator is None:
            results[name] = OperatorResult(
                metrics={},
                error=f"Operator '{name}' not found",
            )
            continue
        
        # Implement retry logic for transient errors (e.g. CUDA timeouts)
        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                result = operator.execute(ctx)
                results[name] = result
                
                # Store normalized metrics for QualityScore computation
                if result.success and result.metrics:
                    if "normalized_metrics" not in ctx.shared_data:
                        ctx.shared_data["normalized_metrics"] = {}
                    
                    # We expect operators to provide raw 0-1 values if they want to contribute to QualityScore
                    # Convention: metric_name_score is 0-100, metric_name is 0-1
                    for m_name, m_val in result.metrics.items():
                        if not m_name.endswith("_score") and m_name not in ["yaw", "pitch", "roll", "blink_prob"]:
                            ctx.shared_data["normalized_metrics"][m_name] = m_val
                break # Success, exit retry loop
                            
            except Exception as e:
                if attempt < max_retries:
                    if logger:
                        logger.warning(f"Operator '{name}' failed (attempt {attempt + 1}), retrying...", extra={"error": e})
                    if "cuda" in str(e).lower() and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    time.sleep(0.5 * (attempt + 1))
                else:
                    results[name] = OperatorResult(
                        metrics={},
                        error=f"Execution failed after {max_retries} retries: {e}",
                    )
    
    return results
