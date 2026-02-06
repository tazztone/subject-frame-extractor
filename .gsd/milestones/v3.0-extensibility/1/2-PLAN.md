---
phase: 1
plan: 2
wave: 1
---

# Plan 1.2: Sharpness Operator Prototype

## Objective
Implement the first concrete operator (`SharpnessOperator`) to validate the protocol design, demonstrate the pattern, and prove the extensibility goal.

## Context
- @.gsd/phases/1/RESEARCH.md — Prototype design and sharpness algorithm
- @core/operators/base.py — Operator Protocol (from Plan 1.1)
- @core/models.py — Current sharpness implementation (lines 192-199)
- @core/config.py — `sharpness_base_scale` configuration

## Tasks

<task type="auto">
  <name>Implement SharpnessOperator</name>
  <files>
    - core/operators/sharpness.py (NEW)
  </files>
  <action>
    Create `core/operators/sharpness.py`:
    
    1. Imports: cv2, numpy, base classes
    
    2. `SharpnessOperator` class:
       
       a) `config` property:
          ```python
          OperatorConfig(
              name="sharpness",
              display_name="Sharpness Score",
              category="quality",
              requires_mask=True,
              min_value=0.0,
              max_value=100.0,
              description="Laplacian variance measuring image sharpness"
          )
          ```
       
       b) `execute(ctx: OperatorContext) -> OperatorResult`:
          - Convert to grayscale: `cv2.cvtColor(ctx.image_rgb, cv2.COLOR_RGB2GRAY)`
          - Compute Laplacian: `cv2.Laplacian(gray, cv2.CV_64F)`
          - Apply mask if provided: `laplacian[mask > 128]`
          - Compute variance: `np.var(masked_laplacian)`
          - Get scale from config: `ctx.config.sharpness_base_scale` (default 2500.0)
          - Normalize: `min(100.0, (variance / scale) * 100.0)`
          - Return `OperatorResult(metrics={"sharpness_score": score})`
          - On exception: return `OperatorResult(metrics={}, error=str(e))`
       
       c) No `initialize()` needed (stateless, uses OpenCV)
    
    3. Register with `@register_operator` decorator
    
    MIRROR logic from `Frame.calculate_quality_metrics()` lines 192-199
    
    AVOID:
    - DO NOT wire into pipeline yet (Phase 2)
    - DO NOT add complex error handling beyond try/except
  </action>
  <verify>python -c "from core.operators.sharpness import SharpnessOperator; op = SharpnessOperator(); print(op.config)"</verify>
  <done>SharpnessOperator importable; config shows correct values</done>
</task>

<task type="auto">
  <name>Add SharpnessOperator tests</name>
  <files>
    - tests/unit/test_operators.py (MODIFY)
  </files>
  <action>
    Add to `tests/unit/test_operators.py`:
    
    1. New fixtures:
       - `sharp_image()` — checkerboard pattern (high frequency)
       - `blurry_image()` — Gaussian blurred uniform gray
       - `sharpness_operator()` — SharpnessOperator instance
    
    2. `TestSharpnessOperator` class:
       - `test_config_values`: name="sharpness", category="quality", requires_mask=True
       - `test_execute_returns_operator_result`: Check return type is OperatorResult
       - `test_execute_has_sharpness_score`: Check "sharpness_score" in metrics
       - `test_score_in_valid_range`: 0 <= score <= 100
       - `test_sharp_higher_than_blurry`: sharp_image score > blurry_image score
       - `test_with_mask_changes_score`: Score differs with/without mask
       - `test_error_handling`: Invalid input returns OperatorResult with error
    
    Use synthetic images only (no file I/O):
    - Checkerboard: `np.indices((100, 100)).sum(axis=0) % 2 * 255`
    - Blurry: `cv2.GaussianBlur(np.full((100, 100, 3), 128), (21, 21), 0)`
  </action>
  <verify>uv run pytest tests/unit/test_operators.py::TestSharpnessOperator -v</verify>
  <done>All SharpnessOperator tests pass</done>
</task>

<task type="auto">
  <name>Update exports and verify end-to-end</name>
  <files>
    - core/operators/__init__.py (MODIFY)
  </files>
  <action>
    Update `core/operators/__init__.py`:
    
    1. Import SharpnessOperator (triggers registration)
    2. Add to __all__:
       ```python
       __all__ = [
           "Operator",
           "OperatorContext",
           "OperatorConfig",
           "OperatorResult",
           "OperatorRegistry",
           "register_operator",
           "run_operators",
           "SharpnessOperator",
       ]
       ```
    
    3. Verify registration works on import
  </action>
  <verify>python -c "from core.operators import SharpnessOperator, OperatorRegistry, run_operators; import numpy as np; img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8); result = run_operators(img, None, None); print(f'Operators: {list(result.keys())}')"</verify>
  <done>SharpnessOperator registered; run_operators returns results</done>
</task>

## Success Criteria
- [ ] `SharpnessOperator` implements refined Protocol with `OperatorResult`
- [ ] Operator registered in `OperatorRegistry` on import
- [ ] Sharp images score higher than blurry images
- [ ] Error handling returns `OperatorResult` with error field
- [ ] `run_operators()` bridge function works with real operator
- [ ] All tests pass: `uv run pytest tests/unit/test_operators.py -v`
