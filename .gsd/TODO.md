# TODO

## Phase 1: Core Regressions
- [ ] Fix `test_database.py` failures (AttributeError: create_tables).
- [ ] Fix `test_pipelines_extended.py` (AttributeError: create_tables).
- [ ] Fix `PermissionError` for `/out` in `tests/test_app_ui_logic.py`.
- [ ] Fix `test_mask_propagator_logic.py` assertion failure.

## Phase 2: Refactoring
- [ ] Decompose `execute_pre_analysis`.
- [ ] Remove `ui/app_ui.py.bak`.

## Phase 3: Resource Management
- [ ] Implement `ModelRegistry` watchdog.

## Miscellaneous
- [ ] Update `AGENTS.md` with new stabilization instructions.
