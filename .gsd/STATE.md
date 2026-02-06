## Current Position
- **Phase**: 0 (State Cleanup)
- **Task**: Plan 0.3 - Remove Legacy States & Final Cleanup
- **Status**: Active (resumed 2026-02-06 22:38)
- **Mode**: Verification

## Last Session Summary
- **Accomplished**:
    - Refactored `SceneHandler` to use `ApplicationState` exclusively.
    - Extracted `ApplicationState` to `core/application_state.py` to fix circular dependencies.
    - Removed all legacy `gr.State` components from `app_ui.py`.
    - Fixed multiple syntax/indentation errors in `scene_handler.py`.
- **Roadblocks**:
    - **Circular Imports**: Caused `NameError` on startup. Fixed by moving model to `core`.
    - **Syntax Errors**: `IndentationError`, `SyntaxError` (missing brackets, return outside function) in `scene_handler.py` prevented app startup, leading to `ERR_CONNECTION_REFUSED` in E2E tests.

## In-Progress Work
- **Verification**: `tests/ui/test_app_flow.py` was running. Last run status was ambiguous (no output captured). Needs definitive pass confirmation.
- **Files Modified**:
    - `core/application_state.py` (New)
    - `ui/app_ui.py` (Legacy state removal)
    - `ui/handlers/scene_handler.py` (Refactor & Fixes)

## Blockers
- **Verification Confirmation**: Need to confirm E2E tests pass consistently now that syntax errors are resolved.

## Context Dump

### Decisions Made
- **Extract ApplicationState**: Necessary to break the import cycle between `AppUI` and `SceneHandler`.
- **Remove Legacy Returns**: Updated `on_select_for_edit` and other handlers to stop returning legacy state tuples, simplifying the data flow.

### Approaches Tried
- **Grep Verification**: Used `grep` to find legacy key usages (`scenes_state` etc.).
- **App.py Debugging**: Ran `app.py` directly to catch syntax errors that `pytest` masked with generic connection errors.

### Current Hypothesis
- The `ERR_CONNECTION_REFUSED` was purely due to the Python syntax errors crashing the app on startup. Now that `app.py` runs (or at least valid syntax is restored), the E2E tests should pass.

### Files of Interest
- `ui/handlers/scene_handler.py`: Recently fixed syntax errors here. Watch for logic regressions.
- `tests/ui/test_app_flow.py`: Primary verification suite.

## Next Steps
1. **Run E2E Tests**: `uv run pytest tests/ui/test_app_flow.py` to confirm everything is green.
2. **Commit Phase 0 Code**: Once tests pass, `git commit` the removal of legacy states.
3. **Verify Roadmap**: Mark Phase 0 as complete in `task.md` and `ROADMAP.md` (if separate).
4. **Start Phase 1**: Move to `Operator Pattern` implementation.