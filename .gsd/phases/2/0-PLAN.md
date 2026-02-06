---
phase: 2
plan: 0
wave: 0
---

# Plan 2.0: Regression Safety Setup

## Objective
Create a golden snapshot of current analysis output BEFORE refactoring. This ensures we can verify the operator migration produces identical results.

## Context
- @core/pipelines.py — Current analysis implementation
- @core/models.py — Current metric calculations

## Tasks

<task type="auto">
  <name>Create Shared Test Fixtures</name>
  <files>
    - tests/conftest.py (MODIFY or CREATE)
  </files>
  <action>
    Add shared pytest fixtures to avoid duplication across test files:
    
    1. `sample_image()` — 100x100 RGB random noise
    2. `sample_mask()` — 100x100 grayscale with center active
    3. `sharp_image()` — Checkerboard pattern
    4. `blurry_image()` — Gaussian blurred uniform gray
    5. `mock_config()` — Mock Config with sharpness_base_scale, etc.
    
    These replace duplicates in `test_operators.py` and will be used by all new operator tests.
  </action>
  <verify>uv run pytest tests/unit/test_operators.py -v --collect-only | grep "fixture"</verify>
  <done>Shared fixtures available project-wide</done>
</task>

<task type="auto">
  <name>Create Golden Snapshot Test</name>
  <files>
    - tests/regression/test_metric_parity.py (NEW)
    - tests/regression/golden_metrics.json (NEW - generated)
  </files>
  <action>
    Create a regression test that:
    
    1. Loads a small test image (use `tests/fixtures/test_frame.png` or create one).
    2. Runs the LEGACY `Frame.calculate_quality_metrics()` method.
    3. Saves output to `golden_metrics.json` if not exists.
    4. On subsequent runs, compares new output to golden snapshot.
    
    Test should:
    - Skip gracefully if no test image available.
    - Allow regeneration via `--regenerate-golden` pytest flag or env var.
    - Compare all metric keys with tolerance (1e-5 for floats).
  </action>
  <verify>uv run pytest tests/regression/test_metric_parity.py -v</verify>
  <done>Golden snapshot captured; parity test passes</done>
</task>

## Success Criteria
- [ ] Shared fixtures in `tests/conftest.py`
- [ ] Golden metrics captured from legacy implementation
- [ ] Parity test ready to validate operator migration
