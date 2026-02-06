---
phase: 0
plan: 1
wave: 1
---

# Plan 0.1: Audit & Extend ApplicationState Model

## Objective
Ensure the `ApplicationState` Pydantic model in `app_ui.py` contains ALL fields currently tracked by legacy `gr.State` components, enabling a full migration.

## Context
- [ui/app_ui.py](file:///home/tazztone/_coding/subject-frame-extractor/ui/app_ui.py) — Lines 60-80 (ApplicationState), Lines 536-552 (legacy states)

## Tasks

<task type="auto">
  <name>Extend ApplicationState with missing fields</name>
  <files>ui/app_ui.py</files>
  <action>
    Compare the existing `ApplicationState` model (lines 60-80) with the legacy `gr.State` declarations (lines 539-552).
    
    Add any missing fields to `ApplicationState`:
    - `scene_history: List[List[dict]]` (for undo/redo, replaces deque)
    - `smart_filter_enabled: bool` (already exists, verify)
    
    Ensure all types are JSON-serializable (Pydantic constraint).
  </action>
  <verify>`python -c "from ui.app_ui import ApplicationState; print(ApplicationState.model_fields.keys())"` lists all required fields</verify>
  <done>ApplicationState contains: scenes, extracted_frames_dir, scene_gallery_index_map, selected_scene_id, gallery_image, gallery_shape, analysis_output_dir, extracted_video_path, analysis_metadata_path, scene_history, smart_filter_enabled</done>
</task>

<task type="auto">
  <name>Add helper methods to ApplicationState</name>
  <files>ui/app_ui.py</files>
  <action>
    Add convenience methods to `ApplicationState` for common operations:
    - `push_history(self, scenes: list)` — Append current scenes to history
    - `pop_history(self) -> Optional[list]` — Pop last scenes from history (undo)
    
    This encapsulates the deque logic inside the model.
  </action>
  <verify>Unit test: `ApplicationState` can push/pop history correctly</verify>
  <done>`push_history` and `pop_history` methods exist and are tested</done>
</task>

## Success Criteria
- [ ] `ApplicationState` model contains all fields from legacy states
- [ ] Model is importable and JSON-serializable
- [ ] Helper methods exist for history management
