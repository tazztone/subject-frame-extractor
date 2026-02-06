---
phase: 0
plan: 2
wave: 2
---

# Plan 0.2: Migrate SceneHandler to ApplicationState

## Objective
Refactor `SceneHandler` to read/write from the unified `ApplicationState` instead of individual legacy `gr.State` components.

## Context
- [ui/handlers/scene_handler.py](file:///home/tazztone/_coding/subject-frame-extractor/ui/handlers/scene_handler.py) — Main handler class
- [ui/app_ui.py](file:///home/tazztone/_coding/subject-frame-extractor/ui/app_ui.py) — ApplicationState model

## Tasks

<task type="auto">
  <name>Refactor SceneHandler state access</name>
  <files>ui/handlers/scene_handler.py</files>
  <action>
    1. Change all handler methods to accept `application_state: ApplicationState` instead of individual state parameters.
    2. Read scene data from `application_state.scenes` instead of `scenes_state`.
    3. Write scene updates by returning a modified `application_state` object.
    
    AVOID: Breaking existing event handler signatures abruptly. Use a wrapper pattern if needed for backward compatibility during transition.
  </action>
  <verify>Grep for `scenes_state` in `scene_handler.py` returns 0 matches</verify>
  <done>SceneHandler uses only `application_state.scenes` for scene access</done>
</task>

<task type="auto">
  <name>Update event handler wiring in app_ui.py</name>
  <files>ui/app_ui.py</files>
  <action>
    Update `scene_handler.setup_handlers()` calls to:
    1. Include `application_state` in inputs.
    2. Include `application_state` in outputs.
    
    Ensure Gradio event bindings pass the unified state object.
  </action>
  <verify>App launches without errors: `python app.py --dry-run` (if available) or manual launch test</verify>
  <done>Scene handlers receive and return `application_state`</done>
</task>

## Success Criteria
- [ ] Zero references to `scenes_state` in `scene_handler.py`
- [ ] App launches and scene operations work via `application_state`
