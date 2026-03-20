# STATE.md — Project Memory

> **Last Updated**: 2026-03-20T13:30:00+01:00
> **Session Status**: Active

## Current Position
- **Milestone**: v0.9.1-stabilization
- **Phase**: 1 (Planning & Final Fixes)
- **Task**: Resolved critical UI crashes and model loading issues.
- **Status**: Verified

## Last Session Summary
Resolved a critical `ValueError` in the Gradio UI and addressed several stabilization issues identified in the audit.
- **UI Stabilization**: Fixed `ValueError` caused by the log timer yielding component updates not specified in `outputs`. Implemented `self.all_outputs` to handle dynamic UI updates globally.
- **Handler Contract Alignment**: Updated `_run_task_with_progress` to use component-object keys instead of string keys, ensuring Gradio correctly routes updates.
- **SAM3 Resiliency (Submodule Clean)**: Implemented `torch.float32` forcing via monkey-patching in `core/sam3_patches.py` instead of direct edits to `SAM3_repo`. This ensures stability on Ampere+ GPUs while keeping the submodule directory clean.
- **Batch Processing**: Fixed potential `ValueError` in `stop_batch_handler` by adding missing `outputs`.
- **Verification**: Verified fixes via `tests/ui/test_handler_contracts.py` and `tests/unit/test_pipeline_result_schemas.py` with 100% pass rate.

## In-Progress Work
- None. Stabilization fixes are complete and verified.

## Blockers
- None.

## Context Dump

### Decisions Made
- **Global All Outputs**: Defined `self.all_outputs` containing all UI components to simplify dynamic updates from background threads/timers.
- **Strict Component Keys**: Enforced using `self.components[key]` instead of string keys in all UI-update dictionaries.

### Approaches Tried
- **Mocking for Contract Tests**: Used fast Python-level tests to verify handler contracts without needing a browser or GPU.

### Files of Interest
- `ui/app_ui.py`: Core UI logic and event handlers.
- `SAM3_repo/sam3/model/sam3_video_predictor.py`: SAM3 model initialization.
- `SAM3_repo/sam3/model_builder.py`: SAM3 model builder.

## Next Steps
1. Final end-to-end manual verification.
2. Update documentation if any public APIs changed (none so far).
3. Proceed with v1.0-photo-mode foundation if no regressions are found.