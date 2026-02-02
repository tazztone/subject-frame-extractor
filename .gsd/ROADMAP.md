# ROADMAP.md

> **Current Phase**: Phase 2: Pipeline Refactoring & Cleanup
> **Milestone**: v2.1-stabilized

## Must-Haves (from SPEC)
- [x] Passing test suite (0 failures)
- [x] Memory watchdog in `ModelRegistry`
- [x] Refactored pipelines
- [ ] Verified GUI workflow

## Phases

### Phase 1: Core Regressions & Infrastructure
**Status**: ✅ Completed
**Objective**: Restore the codebase to a "Green" state where all existing tests pass and basic pathing errors are resolved.
**Tasks**:
- [x] Fix `Database` API in tests (Migrate `create_tables()` calls to `migrate()`).
- [x] Fix `PermissionError: [Errno 13]` by removing hardcoded `/out` paths in tests.
- [x] Verify `uv.lock` consistency and environment health.
- [x] Run full test suite and confirm core logic failures are resolved.

### Phase 2: Pipeline Refactoring & Cleanup
**Status**: ✅ Completed
**Objective**: Improve maintainability of core logic and remove legacy artifacts.
**Tasks**:
- [x] Decompose `execute_pre_analysis` in `core/pipelines.py` into smaller logic blocks.
- [x] Refactor common pipeline patterns into a base class or utility functions.
- [x] Remove `ui/app_ui.py.bak` and other identified orphans.
- [x] Consolidate manual verification scripts under a single standard `pytest` pattern.

### Phase 3: Resource Management & UI State
**Status**: ⬜ Not Started
**Objective**: Prevent OOM crashes and stabilize the Gradio interface.
**Tasks**:
- [ ] Implement `ModelRegistry` memory watchdog/unloader.
- [ ] Consolidate `gr.State` usage in `ui/app_ui.py` into a more predictable application state model.
- [ ] Improve error handling in the GUI to prevent silent crashes.

### Phase 4: Final Verification & Delivery
**Status**: ⬜ Not Started
**Objective**: Empirical validation of all features via the GUI.
**Tasks**:
- [ ] Perform E2E browser-based run using `sample.mp4` and `sample.jpg`.
- [ ] Verify every UI tab: Source, Subject, Scenes, Metrics, Export.
- [ ] Validate export artifacts (metadata, cropped frames).
- [ ] Update `AGENTS.md` and documentation to reflect changes.
