# Phase 6: Architectural Cleanup & Operator Consolidation

## Objective
Finalize the transition to the Operator pattern by migrating the last remaining legacy metrics and removing redundant code from the core models.

## Tasks

### 1. Identify & Migrate Legacy Metrics
- Analyze `Frame.calculate_quality_metrics` in `core/models.py`.
- Ensure all logic (e.g., contrast, brightness, entropy) is fully covered by existing or new Operators.
- Move any unique logic (like specific normalization) into the relevant Operator classes.

### 2. Refactor Analysis Pipeline (Unified Context)
- Define an `OperatorContext` that includes:
  - `model_registry` (for lazy loading)
  - `thumbnail_manager`
  - `config`
  - Cached data (like face detections) to avoid redundant calculations.
- Update `run_operators` to accept this context.
- Simplify `AnalysisPipeline._process_single_frame` to stop pre-calculating face data.

### 3. Decommission Legacy Code
- Remove `calculate_quality_metrics` from the `Frame` class.
- Cleanup unused imports and utility functions in `core/models.py`.
- Verify that `Database` serialization still works correctly with the new structure.

## Success Criteria
- [ ] No occurrences of `calculate_quality_metrics` in the codebase.
- [ ] Operators successfully handle their own model dependencies via the context.
- [ ] All tests pass with 100% metric parity compared to the legacy implementation.
