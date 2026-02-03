# ROADMAP.md

> **Current Phase**: Phase 4: Final Verification & Delivery
> **Milestone**: v2.1-stabilized

## Must-Haves (from SPEC)
- [x] Passing test suite (0 failures)
- [x] Memory watchdog in `ModelRegistry`
- [x] Refactored pipelines
- [x] Verified GUI workflow

## Phases
...
### Phase 3: Resource Management & UI State
**Status**: ✅ Completed
**Objective**: Prevent OOM crashes and stabilize the Gradio interface.
**Tasks**:
- [x] Implement `ModelRegistry` memory watchdog/unloader.
- [x] Consolidate `gr.State` usage in `ui/app_ui.py` into a more predictable application state model.
- [x] Improve error handling in the GUI to prevent silent crashes.

### Phase 4: Final Verification & Delivery
**Status**: ✅ Completed
**Objective**: Empirical validation of all features via the GUI.
**Tasks**:
- [x] Perform E2E browser-based run using `sample.mp4` and `sample.jpg`.
- [x] Verify every UI tab: Source, Subject, Scenes, Metrics, Export.
- [x] Validate export artifacts (metadata, cropped frames).
- [x] Update documentation to reflect final changes.
- [x] Create Linux run script (`scripts/linux_run_app.sh`).

### Phase 5: UI/UX Polish (User Feedback)
**Status**: ✅ Completed
**Objective**: Address usability friction points identified during user testing.
**Tasks**:
- [x] Remove redundant "Stepper" component (obsolete).
- [x] Implement auto-refresh for System Logs (remove manual refresh button).
- [x] Implement auto-save of logs to session output folder (remove manual export button).
- [x] Add explanatory info for "System Diagnostics".
- [x] Ensure robust auto-navigation between tabs on step completion.
