---
phase: 3
plan: 1
wave: 2
---

# Plan 3.1: Operator Documentation & Example

## Objective
Write developer documentation explaining how to add new operators.
Validate the guide by creating a working example operator.

## Context
- `core/operators/base.py` — Operator Protocol definition
- `core/operators/sharpness.py` — Reference implementation

## Tasks

<task type="auto">
  <name>Create HOW_TO_ADD_OPERATOR.md guide</name>
  <files>docs/HOW_TO_ADD_OPERATOR.md</files>
  <action>
    Create a concise Markdown guide (~100 lines) with:
    
    1. **Quick Start** (10 lines): Minimal copy-paste example
    2. **Step-by-Step**:
       - Create `core/operators/my_metric.py`
       - Implement `config` property
       - Implement `execute(ctx)` method
       - Add `@register_operator` decorator
       - That's it! Auto-discovered on import.
    3. **Testing**: Add tests to `tests/unit/test_<name>_operator.py`
    4. **Advanced**: `initialize()`/`cleanup()`, `requires_mask`, error handling
    5. **Reference**: Links to existing operators
    
    Keep examples copy-paste ready. Avoid boilerplate explanations.
  </action>
  <verify>test -f docs/HOW_TO_ADD_OPERATOR.md && head -30 docs/HOW_TO_ADD_OPERATOR.md</verify>
  <done>File exists with Quick Start section visible.</done>
</task>

<task type="auto">
  <name>Create example operator in examples/ directory</name>
  <files>examples/operators/pixel_count.py</files>
  <action>
    Create a simple operator following the guide exactly:
    
    ```python
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
            count = np.count_nonzero(ctx.image_rgb.sum(axis=-1))
            return OperatorResult(metrics={"pixel_count": float(count)})
    ```
    
    Note: This is in `examples/` NOT `core/operators/` to avoid polluting production.
  </action>
  <verify>python -c "import examples.operators.pixel_count; from core.operators import OperatorRegistry; print(OperatorRegistry.get('pixel_count'))"</verify>
  <done>Example operator can be imported and registers correctly.</done>
</task>

<task type="auto">
  <name>Add example operator test</name>
  <files>examples/operators/test_pixel_count.py</files>
  <action>
    Create a test file demonstrating how to test custom operators:
    
    ```python
    import numpy as np
    import pytest
    from examples.operators.pixel_count import PixelCountOperator
    from core.operators import OperatorContext
    
    def test_pixel_count_all_black():
        op = PixelCountOperator()
        ctx = OperatorContext(image_rgb=np.zeros((10, 10, 3), dtype=np.uint8))
        result = op.execute(ctx)
        assert result.metrics["pixel_count"] == 0.0
    
    def test_pixel_count_all_white():
        op = PixelCountOperator()
        ctx = OperatorContext(image_rgb=np.full((10, 10, 3), 255, dtype=np.uint8))
        result = op.execute(ctx)
        assert result.metrics["pixel_count"] == 100.0  # 10x10 pixels
    ```
  </action>
  <verify>pytest examples/operators/test_pixel_count.py -v</verify>
  <done>Both example tests pass.</done>
</task>

## Success Criteria
- [ ] `docs/HOW_TO_ADD_OPERATOR.md` exists and is < 150 lines
- [ ] Example operator in `examples/operators/` works
- [ ] Example tests pass
- [ ] Guide matches actual implementation pattern
