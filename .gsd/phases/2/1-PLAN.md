---
phase: 2
plan: 1
wave: 1
---

# Plan 2.1: Simple CV Metrics Migration

## Objective
Migrate the simple computer vision metrics (`edge_strength`, `contrast`, `brightness`, `entropy`) from `core/models.py` to the new Operator framework. These are stateless, CPU-bound operators.

## Context
- @core/models.py — Current implementation (lines 201-234)
- @core/operators/base.py — Operator Protocol
- @core/operators/registry.py — Registration logic
- @tests/conftest.py — Shared fixtures (from Plan 2.0)

## Tasks

<task type="auto">
  <name>Implement Edge, Contrast, Brightness Operators</name>
  <files>
    - core/operators/simple_cv.py (NEW)
  </files>
  <action>
    Create `core/operators/simple_cv.py` implementing:
    
    1. `EdgeStrengthOperator` (Sobel magnitude mean):
       - Config: name="edge_strength", display_name="Edge Strength"
       - Logic: Sobel X/Y → magnitude → mean → scale by `edge_strength_base_scale` (from ctx.config)
       - Return: `{"edge_strength_score": 0-100}`
    
    2. `ContrastOperator` (Standard deviation / Mean):
       - Config: name="contrast", display_name="Contrast"
       - Logic: `std_br / (mean_br + 1e-7)` → clamp by `quality_contrast_clamp`
       - Return: `{"contrast_score": 0-100}`
    
    3. `BrightnessOperator` (Mean intensity):
       - Config: name="brightness", display_name="Brightness"
       - Logic: `mean_br / 255.0 * 100`
       - Return: `{"brightness_score": 0-100}`
    
    Port logic exactly from `core/models.py` lines 201-224.
    Register all with `@register_operator`.
  </action>
  <verify>uv run python -c "from core.operators.simple_cv import EdgeStrengthOperator, ContrastOperator, BrightnessOperator; print('OK')"</verify>
  <done>3 new operators implemented and registered</done>
</task>

<task type="auto">
  <name>Implement Entropy Operator</name>
  <files>
    - core/operators/entropy.py (NEW)
  </files>
  <action>
    Create `core/operators/entropy.py`:
    
    1. Note: `core/models.py` line 226-234 has `pass` (incomplete).
    2. Implement proper Shannon entropy:
       - Calculate histogram of grayscale image (256 bins).
       - Normalize to probability distribution.
       - Compute: `-sum(p * log2(p))` where p > 0.
       - Scale: entropy / 8.0 * 100 (max entropy for 8-bit is 8.0).
    3. Config: name="entropy", display_name="Shannon Entropy", min=0, max=100
  </action>
  <verify>uv run python -c "from core.operators.entropy import EntropyOperator; print(EntropyOperator().config.name)"</verify>
  <done>Entropy operator implemented with proper algorithm</done>
</task>

<task type="auto">
  <name>Add Unit Tests for Simple Metrics</name>
  <files>
    - tests/unit/test_simple_cv_operators.py (NEW)
  </files>
  <action>
    Create tests using shared fixtures from conftest.py:
    
    1. `EdgeStrength`: solid color (0) vs random noise (high)
    2. `Brightness`: black (0) vs white (100)
    3. `Contrast`: uniform (0) vs high contrast pattern
    4. `Entropy`: uniform image (0) vs random noise (high)
    
    All tests verify:
    - Returns OperatorResult
    - Score in valid range (0-100)
    - Expected relative ordering (e.g., white brighter than black)
  </action>
  <verify>uv run pytest tests/unit/test_simple_cv_operators.py -v</verify>
  <done>All tests pass</done>
</task>

## Success Criteria
- [ ] 4 operators registered (`edge_strength`, `contrast`, `brightness`, `entropy`)
- [ ] Logic matches existing implementation (except entropy which was incomplete)
- [ ] Unit tests pass
