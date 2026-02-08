---
phase: 1
plan: 1
wave: 1
---

# Plan 1.1: Fix Initialization Hang & Auto-Tab Switching

## Objective
Resolve the "System Initializing..." hang on startup and remove auto-tab switching behavior that frustrates users.

## Context
- .gsd/phases/1/RESEARCH.md
- ui/app_ui.py (lines 896-936: `_run_pipeline`, 1008-1127: success callbacks)

## Tasks

<task type="auto">
  <name>Remove Auto-Tab Switching</name>
  <files>ui/app_ui.py</files>
  <action>
    Delete `self.components["main_tabs"]: gr.update(selected=N)` from:
    - Line ~1025: `_on_extraction_success`
    - Line ~1051: `_on_pre_analysis_success`
    - Line ~1094: `_on_propagation_success`
    - Line ~1126: `_on_analysis_success`
    Keep all other return dict entries. Do NOT remove the success messages.
  </action>
  <verify>grep -n "selected=" ui/app_ui.py | wc -l  # Should return 0</verify>
  <done>No `selected=` entries remain in success callbacks.</done>
</task>

<task type="auto">
  <name>Fix Status Indicator Updates</name>
  <files>ui/app_ui.py</files>
  <action>
    In `update_logs` (around line 700-750), add handling for `ui_update` keys from `progress_queue`:
    ```python
    if "ui_update" in msg:
        updates = msg["ui_update"]
        for comp_key, value in updates.items():
            if comp_key in self.components:
                yield {self.components[comp_key]: value}
    ```
    Ensure the timer yields these updates alongside log messages.
  </action>
  <verify>python -c "from ui.app_ui import AppUI; print('Import OK')"</verify>
  <done>`model_status_indicator` updates correctly during model loading.</done>
</task>

## Success Criteria
- [ ] `grep -n "selected=" ui/app_ui.py` returns only non-callback lines (e.g., Gradio defaults)
- [ ] App launches and shows "Ready" status within 5 seconds
