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

## Tasks

<task type="auto">
  <name>Implement Edge, Contrast, Brightness Operators</name>
  <files>
    - core/operators/simple_cv.py (NEW)
  </files>
  <action>
    Create `core/operators/simple_cv.py` implementing:
    
    1. `EdgeStrengthOperator` (Sobel magnitude mean):
       - Config: name="edge_strength", display_name="Edge Strength", scale defaults to 500.0 (from config)
    
    2. `ContrastOperator` (Standard deviation / Mean):
       - Config: name="contrast", display_name="Contrast", max_value=1.0, clamp in config (default 50.0?)
       - Logic: `std_br / (mean_br + 1e-7)`
    
    3. `BrightnessOperator` (Mean intensity):
       - Config: name="brightness", display_name="Brightness", max_value=1.0
       - Logic: `mean_br / 255.0`
    
    Port logic exactly from `core/models.py` to ensure parity.
    Register all with `@register_operator`.
  </action>
  <verify>python -c "from core.operators.simple_cv import EdgeStrengthOperator, ContrastOperator, BrightnessOperator; print('Imports OK')"</verify>
  <done>3 new operators implemented and registered</done>
</task>

<task type="auto">
  <name>Implement Entropy Operator</name>
  <files>
    - core/operators/entropy.py (NEW)
  </files>
  <action>
    Create `core/operators/entropy.py`:
    
    1. Import `compute_entropy` if available, or reimplement using `skimage.measure.shannon_entropy` or pure numpy.
    2. Note: `core/models.py` line 226 implies it was missing implementation details ("pass").
    3. Implement `EntropyOperator`:
       - Config: name="entropy", display_name="Shannon Entropy", min=0, max=8.0
       - Logic: Calculate histogram of grayscale image, then -sum(p * log2(p)).
       - Use `cv2.calcHist` for speed.
  </action>
  <verify>python -c "from core.operators.entropy import EntropyOperator; print(EntropyOperator().config.name)"</verify>
  <done>Entropy operator implemented</done>
</task>

<task type="auto">
  <name>Add Unit Tests for Simple Metrics</name>
  <files>
    - tests/unit/test_simple_cv_operators.py (NEW)
  </files>
  <action>
    Create tests for all 4 new operators:
    
    1. `EdgeStrength`: Test on solid color (0 edge) vs random noise (high edge).
    2. `Brightness`: Test black image (0.0) vs white image (1.0).
    3. `Contrast`: Test uniform image (0.0) vs high contrast pattern.
    4. `Entropy`: Test uniform image (0.0) vs random noise (high entropy).
    
    Use fixtures from `tests/unit/test_operators.py` if possible (add shared conftest if needed, or duplicate for now).
  </action>
  <verify>uv run pytest tests/unit/test_simple_cv_operators.py -v</verify>
  <done>All tests pass</done>
</task>

## Success Criteria
- [ ] 4 new operators registered (`edge_strength`, `contrast`, `brightness`, `entropy`)
- [ ] Logic matches existing implementation
- [ ] Unit tests pass for all new operators
