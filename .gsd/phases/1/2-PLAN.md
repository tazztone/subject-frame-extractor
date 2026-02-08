---
phase: 1
plan: 2
wave: 1
---

# Plan 1.2: Fix Propagation Button for Image Mode

## Objective
Prevent the "Propagate Masks" button from crashing when processing image-only folders.

## Context
- .gsd/phases/1/RESEARCH.md
- ui/tabs/scene_tab.py
- ui/app_ui.py (lines 1062-1082: `_propagation_button_handler`, `run_propagation_wrapper`)
- core/utils.py (`is_image_folder`)

## Tasks

<task type="auto">
  <name>Hide Propagation Button for Image Folders</name>
  <files>ui/tabs/scene_tab.py, ui/app_ui.py</files>
  <action>
    1. In `SceneTabBuilder.build()`, wrap `propagate_masks_button` creation in a conditional or set `visible=False` by default.
    2. In `ui/app_ui.py`, after successful pre-analysis:
       - Check if `is_image_folder(params.output_folder)` is True.
       - If so, yield `{self.components["propagate_masks_button"]: gr.update(visible=False)}`.
       - If video mode, ensure button is visible.
    This prevents the crash by hiding the button before the user can click it.
  </action>
  <verify>python -c "from ui.tabs.scene_tab import SceneTabBuilder; print('OK')"</verify>
  <done>Button is hidden when loading image folders but visible for videos.</done>
</task>

<task type="auto">
  <name>Add Defensive Guard in Handler</name>
  <files>ui/app_ui.py</files>
  <action>
    In `_propagation_button_handler`, add an early return if `current_state` indicates image-only mode:
    ```python
    if not current_state.extracted_video_path:
        yield {self.components["unified_log"]: "ℹ️ Propagation is not needed for image folders."}
        return
    ```
    This acts as a safety net if the UI fails to hide the button.
  </action>
  <verify>grep -n "not needed for image folders" ui/app_ui.py</verify>
  <done>Handler includes defensive check for image mode.</done>
</task>

## Success Criteria
- [ ] Load an image folder and confirm button is not visible
- [ ] If button is somehow clicked, no crash occurs
