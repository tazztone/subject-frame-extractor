---
phase: 2
plan: 3
wave: 2
---

# Plan 2.3: Pipeline Engine Integration

## Objective
Integrate the new `run_operators` bridge into `AnalysisPipeline`. This is a PARALLEL integration: operators run alongside legacy code to verify output matches before switching over.

## Context
- @core/pipelines.py — `_process_single_frame` method (lines 698-797)
- @core/operators/registry.py — `run_operators` function
- @tests/regression/test_metric_parity.py — Golden snapshot test (from Plan 2.0)

## Tasks

<task type="auto">
  <name>Add Operator Initialization to Pipeline</name>
  <files>
    - core/pipelines.py (MODIFY)
  </files>
  <action>
    Modify `AnalysisPipeline.__init__`:
    
    1. Import `OperatorRegistry` from `core.operators`.
    2. After existing model init, call:
       ```python
       OperatorRegistry.initialize_all(self.config)
       ```
    3. Update `core/operators/__init__.py` to import all operators (triggers registration).
  </action>
  <verify>uv run python -c "from core.pipelines import AnalysisPipeline; print('Import OK')"</verify>
  <done>Pipeline initializes operator registry</done>
</task>

<task type="auto">
  <name>Add Parallel Operator Execution</name>
  <files>
    - core/pipelines.py (MODIFY)
  </files>
  <action>
    Modify `AnalysisPipeline._process_single_frame`:
    
    1. AFTER legacy `frame.calculate_quality_metrics()` call (keep it for now!)
    2. Add parallel operator execution:
       ```python
       from core.operators import run_operators, OperatorContext
       
       op_results = run_operators(
           image_rgb=thumb_image_rgb,
           mask=mask_thumb,
           config=self.config,
       )
       
       # Log comparison (debug only)
       for name, result in op_results.items():
           if result.success:
               self.logger.debug(f"Operator {name}: {result.metrics}")
       ```
    3. Do NOT replace legacy code yet — this is for comparison.
  </action>
  <verify>uv run pytest tests/regression/test_metric_parity.py -v</verify>
  <done>Operators run in parallel; parity test still passes</done>
</task>

<task type="auto">
  <name>Add Operator-to-Legacy Comparison</name>
  <files>
    - core/pipelines.py (MODIFY)
  </files>
  <action>
    Add temporary comparison logic in `_process_single_frame`:
    
    1. For each operator result, compare to corresponding legacy metric.
    2. Log WARNING if difference > 1.0 (score points).
    3. This helps catch any implementation drift before switching.
    
    Example:
    ```python
    if "sharpness" in op_results:
        op_score = op_results["sharpness"].metrics.get("sharpness_score", 0)
        legacy_score = frame.metrics.sharpness_score or 0
        if abs(op_score - legacy_score) > 1.0:
            self.logger.warning(f"Sharpness drift: op={op_score:.1f} legacy={legacy_score:.1f}")
    ```
  </action>
  <verify>Manual: Run analysis on test video, check for drift warnings</verify>
  <done>Comparison logging added; no drift warnings</done>
</task>

## Success Criteria
- [ ] `AnalysisPipeline` initializes operators
- [ ] Operators run in parallel with legacy code
- [ ] Comparison logging catches any drift
- [ ] Golden snapshot parity test passes
