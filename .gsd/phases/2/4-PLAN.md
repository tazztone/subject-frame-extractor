---
phase: 2
plan: 4
wave: 2
---

# Plan 2.4: Pipeline Refactor

## Objective
Integrate the new `run_operators` bridge into `AnalysisPipeline`. This switches the engine from the hardcoded `Frame.calculate_quality_metrics` method to the extensible Operator Registry.

## Context
- @core/pipelines.py — `_process_single_frame` method
- @core/operators/registry.py — `run_operators` function

## Tasks

<task type="checkpoint:human-verify">
  <name>Verify Operator Coverage</name>
  <action>
    Ensure all metrics currently used in `AnalysisPipeline` (quality, sharpness, edge, contrast, brightness, entropy, eyes, yaw, pitch) have corresponding registered operators.
  </action>
  <verify>List all registered operators via `OperatorRegistry.list_names()`</verify>
  <done>All legacy metrics covered</done>
</task>

<task type="auto">
  <name>Refactor Pipeline to use Operators</name>
  <files>
    - core/pipelines.py (MODIFY)
    - core/operators/registry.py (MODIFY - optional if needed)
  </files>
  <action>
    Modify `AnalysisPipeline._process_single_frame`:
    
    1. **Pre-calculation**: 
       - Run face detection/landmarking logic FIRST.
       - Store results (landmarks, matrix) in a dict `shared_params`.
    
    2. **Execution**:
       - Replaces calls to `frame.calculate_quality_metrics`.
       - Call `run_operators(img, mask, config, params=shared_params)`.
    
    3. **Result Integration**:
       - Iterate over `OperatorResult` dict.
       - Flatten metrics into `frame.metrics` or a temp dict.
       - Handle errors/warnings (log them).
       - Maintain backward compatibility for `frame.metrics` attribute access if needed elsewhere.
    
    4. **Lifecycle**:
       - Call `OperatorRegistry.initialize_all(self.config)` in pipeline `__init__`.
  </action>
  <verify>Run full end-to-end test (or verify existing UI tests pass with new engine)</verify>
  <done>Pipeline uses operators; metadata output format preserved</done>
</task>

<task type="auto">
  <name>Cleanup Legacy Code</name>
  <files>
    - core/models.py (MODIFY)
  </files>
  <action>
    1. Deprecate `Frame.calculate_quality_metrics`.
    2. Optionally remove the logic body (or leave a stub if risk is high).
    3. Recommendation: Comment out body and add "Use Operator Registry" note.
  </action>
  <verify>grep "calculate_quality_metrics" core/pipelines.py | wc -l # Should be 0 calls</verify>
  <done>Legacy method deprecated/removed</done>
</task>

## Success Criteria
- [ ] `AnalysisPipeline` initializes operators
- [ ] `_process_single_frame` populates context and calls `run_operators`
- [ ] Output metadata matches Phase 0 structure
- [ ] Tests pass
