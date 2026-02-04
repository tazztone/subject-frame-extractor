# TODO

## Ongoing / Post-Release Improvements
- [ ] Investigate seeding mismatch: 2 scenes found but 6 seeds generated (likely redundant calls or logic error) `medium` — 2026-02-03
- [ ] Refactor `AppUI` to remove legacy state keys and fully migrate to `ApplicationState`.
- [ ] Monitor documentation generation script to ensure it doesn't leak secrets or grow excessively.
- [ ] Create a unified `Makefile` or `justfile` for cross-platform task management (shortcut for run, test, layout).

## Completed (v2.1-stabilized)
- [x] Investigate E2E failures: Fixed SAM3 tracking loss via text hints and 360p resolution.
- [x] Fix `test_database.py` failures (AttributeError: create_tables).
- [x] Fix `test_pipelines_extended.py` (AttributeError: create_tables).
- [x] Fix `PermissionError` for `/out` in `tests/test_app_ui_logic.py`.
- [x] Fix `test_mask_propagator_logic.py` assertion failure.
- [x] Decompose `execute_pre_analysis`.
- [x] Remove `ui/app_ui.py.bak`.
- [x] Implement `ModelRegistry` watchdog.
- [x] Update `AGENTS.md` with new stabilization instructions.
- [x] Create Linux run script (`scripts/linux_run_app.sh`).
