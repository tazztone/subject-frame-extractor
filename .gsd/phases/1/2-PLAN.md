---
phase: 1
plan: 2
wave: 1
---

# Plan 1.2: Sharpness Operator Prototype

## Objective
Implement the first concrete operator (`SharpnessOperator`) to validate the protocol design and demonstrate the pattern for future operators.

## Context
- @.gsd/phases/1/RESEARCH.md — Prototype design and sharpness algorithm
- @core/operators/base.py — Operator Protocol (from Plan 1.1)
- @core/models.py — Current sharpness implementation in `Frame.calculate_quality_metrics()`
- @core/config.py — `sharpness_base_scale` configuration

## Tasks

<task type="auto">
  <name>Implement SharpnessOperator</name>
  <files>
    - core/operators/sharpness.py (NEW)
  </files>
  <action>
    Create `core/operators/sharpness.py` with:
    
    1. Import cv2, numpy, and base classes
    
    2. `SharpnessOperator` class implementing Operator Protocol:
       - `config` property returns:
         ```python
         OperatorConfig(
             name="sharpness",
             display_name="Sharpness Score",
             category="quality",
             requires_mask=True,  # Benefits from mask
         )
         ```
       - `execute(ctx: OperatorContext) -> dict`:
         - Convert image to grayscale: `cv2.cvtColor(ctx.image_rgb, cv2.COLOR_RGB2GRAY)`
         - Compute Laplacian: `cv2.Laplacian(gray, cv2.CV_64F)`
         - If mask provided and valid: apply mask to laplacian
         - Compute variance: `np.var(laplacian)`
         - Normalize using scale from `ctx.config.sharpness_base_scale` (default 2500.0)
         - Clamp to 0-100 range
         - Return `{"sharpness_score": score}`
    
    3. Register with decorator: `@register_operator`
    
    MIRROR existing logic from `Frame.calculate_quality_metrics()` lines 192-199 but:
    - Move logic to standalone operator
    - Use OperatorContext instead of method parameters
    - Return dict instead of setting self.metrics
    
    AVOID:
    - DO NOT refactor existing pipeline yet (that's Phase 2)
    - DO NOT add error handling logic (keep minimal for prototype)
  </action>
  <verify>python -c "from core.operators.sharpness import SharpnessOperator; print(SharpnessOperator().config)"</verify>
  <done>SharpnessOperator importable and registered; config returns expected values</done>
</task>

<task type="auto">
  <name>Add SharpnessOperator tests</name>
  <files>
    - tests/unit/test_operators.py (MODIFY)
  </files>
  <action>
    Add to existing `tests/unit/test_operators.py`:
    
    1. `TestSharpnessOperator` class:
       - `test_config_values`: Verify name, display_name, category, requires_mask
       - `test_execute_returns_score`: Execute on sample image, verify dict with "sharpness_score" key
       - `test_execute_with_mask`: Execute with mask, verify score differs from no-mask
       - `test_execute_score_range`: Verify score is 0-100
       - `test_high_detail_image_scores_higher`: Compare sharp vs blurry synthetic images
    
    2. Fixtures:
       - `sharp_image()` — High-frequency edge pattern (checkerboard)
       - `blurry_image()` — Gaussian blurred version
       - `mock_config()` — Config mock with sharpness_base_scale
    
    Test strategy:
    - Use synthetic images (np.random noise, patterns) not real files
    - Verify relative ordering (sharp > blurry) not absolute values
  </action>
  <verify>uv run pytest tests/unit/test_operators.py::TestSharpnessOperator -v</verify>
  <done>All SharpnessOperator tests pass; demonstrates operator behaves correctly</done>
</task>

<task type="auto">
  <name>Update operators __init__ exports</name>
  <files>
    - core/operators/__init__.py (MODIFY)
  </files>
  <action>
    Update `core/operators/__init__.py` to:
    
    1. Import and export SharpnessOperator
    2. Trigger registration on import (decorator handles this)
    3. Add to __all__ list
    
    Final exports should be:
    ```python
    __all__ = [
        "Operator",
        "OperatorContext", 
        "OperatorConfig",
        "OperatorRegistry",
        "register_operator",
        "SharpnessOperator",
    ]
    ```
  </action>
  <verify>python -c "from core.operators import SharpnessOperator, OperatorRegistry; print(OperatorRegistry.list_all())"</verify>
  <done>SharpnessOperator in registry list_all output</done>
</task>

## Success Criteria
- [ ] `SharpnessOperator` implemented following Protocol
- [ ] Operator registered in `OperatorRegistry`
- [ ] Unit tests validate execution and score range
- [ ] Sharp images score higher than blurry images
- [ ] All tests pass: `uv run pytest tests/unit/test_operators.py -v`
