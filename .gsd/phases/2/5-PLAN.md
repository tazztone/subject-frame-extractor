---
phase: 2
plan: 5
wave: 4
---

# Plan 2.5: Switch to Operators & Cleanup

## Objective
Replace legacy `Frame.calculate_quality_metrics()` with `run_operators()` as the sole source of metrics. Clean up deprecated code.

## Context
- @core/pipelines.py — Now has parallel operator execution (from Plan 2.3)
- @core/models.py — Legacy `calculate_quality_metrics` method
- @tests/regression/test_metric_parity.py — Must still pass!

## Tasks

<task type="auto">
  <name>Switch Pipeline to Operators-Only</name>
  <files>
    - core/pipelines.py (MODIFY)
  </files>
  <action>
    Modify `AnalysisPipeline._process_single_frame`:
    
    1. REMOVE the call to `frame.calculate_quality_metrics()`.
    2. Use operator results to populate `frame.metrics`:
       ```python
       for name, result in op_results.items():
           if result.success:
               for metric_name, value in result.metrics.items():
                   setattr(frame.metrics, metric_name, value)
       ```
    3. Remove the temporary comparison/drift logging from Plan 2.3.
    4. Keep error handling for operator failures.
  </action>
  <verify>uv run pytest tests/regression/test_metric_parity.py -v</verify>
  <done>Pipeline uses operators exclusively; parity test passes</done>
</task>

<task type="auto">
  <name>Deprecate Legacy Method</name>
  <files>
    - core/models.py (MODIFY)
  </files>
  <action>
    Modify `Frame.calculate_quality_metrics`:
    
    1. Add deprecation warning at method start:
       ```python
       import warnings
       warnings.warn(
           "calculate_quality_metrics is deprecated. Use core.operators.run_operators() instead.",
           DeprecationWarning,
           stacklevel=2
       )
       ```
    2. Keep the method body intact for now (in case of rollback).
    3. Add docstring note: "DEPRECATED: Will be removed in v4.0".
  </action>
  <verify>grep -n "DeprecationWarning" core/models.py</verify>
  <done>Legacy method marked deprecated</done>
</task>

<task type="auto">
  <name>Update Operator Exports</name>
  <files>
    - core/operators/__init__.py (MODIFY)
  </files>
  <action>
    Ensure all operators are imported to trigger registration:
    
    ```python
    # Import all operators to trigger registration
    from core.operators.sharpness import SharpnessOperator
    from core.operators.simple_cv import EdgeStrengthOperator, ContrastOperator, BrightnessOperator
    from core.operators.entropy import EntropyOperator
    from core.operators.niqe import NiqeOperator
    from core.operators.face_metrics import EyesOpenOperator, FacePoseOperator
    ```
    
    Update `__all__` to include all exports.
  </action>
  <verify>uv run python -c "from core.operators import OperatorRegistry; print(OperatorRegistry.list_names())"</verify>
  <done>All operators registered on import</done>
</task>

## Success Criteria
- [ ] `_process_single_frame` no longer calls legacy method
- [ ] `Frame.calculate_quality_metrics` marked deprecated
- [ ] All operators registered and working
- [ ] Golden snapshot parity test passes
- [ ] Full test suite passes
