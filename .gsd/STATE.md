# STATE.md — Project Memory

> **Last Updated**: 2026-03-20T18:15:00+01:00
> **Session Status**: Active

## Current Position
- **Milestone**: v0.9.1-stabilization
- **Phase**: 2 (Refactoring & Architecture)
- **Task**: Completed AppUI refactoring and I/O optimization.
- **Status**: FINALIZED

## Last Session Summary
Performed a major architectural cleanup of the `AppUI` class and optimized the I/O data flow for ML operators.
- **Domain Extraction**: 
    - Moved face clustering and representative selection to `core/face_clustering.py`.
    - Moved system health diagnostics and E2E simulation to `core/system_health.py`.
- **UI Componentization**:
    - Extracted the log viewer, background update timer, and dispatching logic to a new `LogViewer` component in `ui/components/log_viewer.py`.
    - Integrated `LogViewer` into `AppUI`, reducing redundant log handling code in `_run_task_with_progress`.
- **I/O Optimization**:
    - Standardized `OperatorContext` to include optional `image_tensor` and `mask_tensor` (Torch tensors).
    - Updated `run_operators` in `core/operators/registry.py` to lazily compute and share these tensors once per frame if any operator requires them.
    - Refactored `NiqeOperator` to use pre-computed Torch tensors and Torch-based masking, eliminating redundant CPU-GPU transfers and Numpy-Tensor conversions.
- **Architecture**:
    - Implemented a basic **State Reducer** in `AppUI.get_ui_updates_from_state` to centralize state-to-UI transitions.

## In-Progress Work
- Transitioning to v1.0-photo-mode foundation.

## Blockers
- None.

## Context Dump

### Decisions Made
- **Lazy Tensor Computation**: Tensors are only created if at least one operator in the current run-set has `requires_tensor=True`.
- **Component-Level Handlers**: `LogViewer` manages its own timer and event handlers, reducing the clutter in `AppUI._create_event_handlers`.

### Approaches Tried
- **Universal Tensor Conversion**: Rejected due to overhead for pure Numpy/CV operators. Lazy conversion is more efficient.

### Files of Interest
- `core/face_clustering.py`: New face domain logic.
- `core/system_health.py`: New diagnostic domain logic.
- `ui/components/log_viewer.py`: New reusable UI component.
- `core/operators/base.py` & `registry.py`: Updated for I/O standardization.

## Next Steps
1. Final verification of refactored UI behavior.
2. Begin Phase 1 of v1.0-photo-mode: Foundation (Ingest & Interop).
