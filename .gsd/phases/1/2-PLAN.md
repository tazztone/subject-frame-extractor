---
phase: 1
plan: 2
wave: 1
---

# Plan 1.2: Resolve Logic Regressions and Environment Verification

## Objective
Fix remaining assertion failures in core logic tests and verify the overall environment health and test suite pass.

## Context
- .gsd/SPEC.md
- core/pipelines.py
- tests/test_mask_propagator_logic.py
- tests/test_pipelines.py
- tests/test_smoke.py

## Tasks

<task type="auto">
  <name>Debug and Fix Logic Regressions</name>
  <files>
    - tests/test_mask_propagator_logic.py
    - tests/test_pipelines.py
    - core/pipelines.py
  </files>
  <action>
    Investigate why `test_propagate_video_success` and `test_run_full_analysis` are failing with bare assertion errors.
    Check if recent changes in pipeline return types (e.g., from boolean to dict/generator) broke the assertions.
    Update tests or implementation to align with the current architecture.
  </action>
  <verify>pytest tests/test_mask_propagator_logic.py tests/test_pipelines.py</verify>
  <done>Assertion failures in propagator and pipeline tests are resolved.</done>
</task>

<task type="auto">
  <name>Environment & Final Smoke Test</name>
  <files>
    - pyproject.toml
    - uv.lock (internal check)
  </files>
  <action>
    Run `uv sync` to ensure dependencies are locked and consistent.
    Run a full smoke test suite to catch any lingering import or signature issues.
    Run the entire test suite (excluding slow/gpu ones as per pyproject.toml defaults).
  </action>
  <verify>uv sync && pytest tests/ -m "not integration and not gpu_e2e"</verify>
  <done>Total test failures = 0 (for the standard unit/smoke suite).</done>
</task>

## Success Criteria
- [ ] 100% pass rate for unit, smoke, and logic tests.
- [ ] Clean `uv` environment.
