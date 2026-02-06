---
phase: 0
plan: 3
wave: 3
---

# Plan 0.3: Remove Legacy States & Final Cleanup

## Objective
Remove all legacy `gr.State` declarations from `app_ui.py` except for the unified `application_state`. Verify the app functions correctly with only the unified state.

## Context
- [ui/app_ui.py](file:///home/tazztone/_coding/subject-frame-extractor/ui/app_ui.py) — Lines 539-552 (legacy states to remove)

## Tasks

<task type="auto">
  <name>Remove legacy state declarations</name>
  <files>ui/app_ui.py</files>
  <action>
    Delete the following lines from `_create_event_handlers`:
    - `self.components["scenes_state"] = gr.State([])`
    - `self.components["extracted_frames_dir_state"] = gr.State("")`
    - `self.components["scene_gallery_index_map_state"] = gr.State([])`
    - `self.components["selected_scene_id_state"] = gr.State(None)`
    - `self.components["gallery_image_state"] = gr.State(None)`
    - `self.components["gallery_shape_state"] = gr.State(None)`
    - `self.components["analysis_output_dir_state"] = gr.State("")`
    - `self.components["extracted_video_path_state"] = gr.State("")`
    - `self.components["analysis_metadata_path_state"] = gr.State("")`
    - `self.components["scene_history_state"] = gr.State(deque(...))`
    - `self.components["smart_filter_state"] = gr.State(False)`
    
    Keep ONLY: `self.components["application_state"] = gr.State(ApplicationState())`
  </action>
  <verify>`grep -c "gr.State" ui/app_ui.py` returns 1 (only application_state)</verify>
  <done>Only `application_state` remains as a `gr.State` component</done>
</task>

<task type="checkpoint:human-verify">
  <name>Manual E2E verification</name>
  <files>N/A</files>
  <action>
    Launch the application and perform:
    1. Load a video source
    2. Navigate to Scenes tab
    3. Toggle a scene status
    4. Export a frame
    
    Verify no errors and state persists correctly.
  </action>
  <verify>User confirms app works</verify>
  <done>User signs off on state migration</done>
</task>

## Success Criteria
- [ ] Only 1 `gr.State` call in `app_ui.py`
- [ ] Full E2E workflow completes without errors
