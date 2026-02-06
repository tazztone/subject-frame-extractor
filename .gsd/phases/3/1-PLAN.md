---
phase: 3
plan: 1
wave: 1
---

# Plan 3.1: Refactor __init__.py to Use Auto-Discovery

## Objective
Replace manual operator imports in `core/operators/__init__.py` with a call to `discover_operators()`.
This simplifies adding new operators to just creating the file.

## Context
- `core/operators/__init__.py` — Current manual imports (lines 47-52)
- `core/operators/registry.py` — Contains `discover_operators()` from Plan 3.0

## Tasks

<task type="auto">
  <name>Replace manual imports with discover_operators()</name>
  <files>core/operators/__init__.py</files>
  <action>
    1. Remove the explicit operator imports (SharpnessOperator, SimpleCV, etc.)
    2. Add a call to `discover_operators()` at module load time
    3. Keep the public `__all__` exports for framework types
    
    Before:
    ```python
    # Import operators to trigger registration
    from core.operators.sharpness import SharpnessOperator
    from core.operators.simple_cv import EdgeStrengthOperator, ...
    ...
    ```
    
    After:
    ```python
    # Auto-discover and register all operators
    from core.operators.registry import discover_operators
    discover_operators()
    ```
    
    AVOID: Breaking the existing `__all__` exports that downstream code relies on.
  </action>
  <verify>pytest tests/unit/test_operators.py -v</verify>
  <done>All 28+ existing operator tests pass with auto-discovery.</done>
</task>

<task type="auto">
  <name>Verify existing pipeline integration</name>
  <files>core/pipelines.py</files>
  <action>
    Run the regression test suite to confirm `AnalysisPipeline` still works:
    - `run_operators()` is called from `_process_single_frame`
    - All operators are discovered and executed
    
    No code changes expected. This is a verification task.
  </action>
  <verify>pytest tests/regression/test_metric_parity.py -v</verify>
  <done>Metric parity tests pass, confirming operators compute same values.</done>
</task>

## Success Criteria
- [ ] `core/operators/__init__.py` no longer has manual operator imports
- [ ] All 28+ existing tests pass
- [ ] Regression (metric parity) tests pass
