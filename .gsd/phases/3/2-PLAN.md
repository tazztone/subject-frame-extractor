---
phase: 3
plan: 2
wave: 1
---

# Plan 3.2: Photo Tab UI & State Integration

## Objective
Create the "Photo Culling" tab in the UI with a paginated gallery, scoring weight sliders, and status display.

## Context
- .gsd/SPEC.md (P2: Photo Mode MVP)
- .gsd/ARCHITECTURE.md (UI Layer, ApplicationState)
- .gsd/phases/3/RESEARCH.md (Gradio Gallery, SceneHandler patterns)
- ui/tabs/scene_tab.py (reference for gallery patterns)
- ui/handlers/scene_handler.py (reference for pagination)
- core/application_state.py

## Tasks

<task type="auto">
  <name>Extend ApplicationState for Photos</name>
  <files>core/application_state.py</files>
  <action>
    - Add new fields to `ApplicationState`:
      ```python
      photos: List[Dict[str, Any]] = []  # {"id": str, "source": Path, "preview": Path, "scores": Dict, "status": str}
      photo_filter_settings: Dict[str, float] = {}  # Metric weights
      photo_page: int = 0
      photo_page_size: int = 50
      photo_index_map: Dict[int, str] = {}  # gallery_idx -> photo_id
      ```
  </action>
  <verify>python -c "from core.application_state import ApplicationState; s = ApplicationState(); print(s.photos, s.photo_page)"</verify>
  <done>ApplicationState includes `photos`, `photo_filter_settings`, `photo_page`, `photo_page_size`, and `photo_index_map` fields.</done>
</task>

<task type="auto">
  <name>Create PhotoTabBuilder</name>
  <files>ui/tabs/photo_tab.py (NEW)</files>
  <action>
    - Create `ui/tabs/photo_tab.py` with `PhotoTabBuilder` class.
    - The `build()` method should create:
      1. A `gr.Gallery` component named `photo_gallery`.
      2. An "Import Folder" textbox + button.
      3. An Accordion "Scoring Weights" with sliders for: `sharpness`, `niqe`, `face`, `entropy`.
      4. A status label.
      5. Pagination buttons (prev/next).
    - Follow the pattern from `SceneTabBuilder`.
  </action>
  <verify>python -c "from ui.tabs.photo_tab import PhotoTabBuilder; print('PhotoTabBuilder imported')"</verify>
  <done>`PhotoTabBuilder` class exists and is importable.</done>
</task>

<task type="auto">
  <name>Integrate Photo Tab into Main UI</name>
  <files>ui/app_ui.py</files>
  <action>
    - Import `PhotoTabBuilder` from `ui.tabs.photo_tab`.
    - In `_build_main_tabs`, add a new `gr.Tab("Photo Culling", id=...)` and call `PhotoTabBuilder(self).build()`.
  </action>
  <verify>python -c "from ui.app_ui import AppUI; print('AppUI with photo tab imported')"</verify>
  <done>The main UI includes a "Photo Culling" tab.</done>
</task>

## Success Criteria
- [ ] `ApplicationState` has photo-related fields.
- [ ] `PhotoTabBuilder` renders a gallery and scoring sliders.
- [ ] The main app shows a "Photo Culling" tab.
