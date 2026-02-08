# Research: Phase 1 (UI & Workflow Stabilization)

## Context
This phase addresses critical UI bugs and workflow disruptions identified in the v6.0 audit.

## Findings (from Audit Report)

### 1. Initialization Hang ("System Initializing...")
- **Issue**: `update_logs` in `ui/app_ui.py` discards UI status updates (like model loading indicators) because it only looks for "log" keys in the progress queue.
- **Fix**: Modify `update_logs` to yield `ui_update` messages correctly to the `model_status_indicator` and `unified_status` components.

### 2. Auto-Tab Switching
- **Issue**: Hardcoded `gr.update(selected=X)` in success callbacks forces the user to switch tabs, disrupting workflow (e.g., when batch processing).
- **Locations**: `_on_extraction_success`, `_on_pre_analysis_success`, `_on_propagation_success`, `_on_analysis_success` in `ui/app_ui.py`.
- **Fix**: Remove these updates.

### 3. "Propagate Masks" Button Crash
- **Issue**: The button is visible even for image-only folders where it shouldn't be valid (or needs different logic).
- **Crash**: `execute_propagation` -> `VideoManager.get_video_info(None)` causes `IOError`.
- **UI Mismatch**: The button's click handler (`_propagation_button_handler`) yields a generator, but the UI component expects a specific number of outputs defined in `all_outputs`.
- **Fix**:
    - Hide/Disable button for image folders in `ui/tabs/scene_tab.py`.
    - Fix `execute_propagation` in Phase 2 (pipeline safety).
    - Ensure `_propagation_button_handler` returns a list/tuple matching `all_outputs` or uses `gr.update` correctly for the mapped components.

## Implementation Strategy
- Modify `ui/app_ui.py` to fix the log/status loop.
- Remove tab switching logic.
- Update `ui/tabs/scene_tab.py` to conditionally render/update the propagation button.
