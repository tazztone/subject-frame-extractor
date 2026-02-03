---
phase: 1
plan: 1
wave: 1
---

# Plan 1.1: Fix Database and UI Action Regressions

## Objective
Restore basic test stability by fixing the Database API mismatch and resolving file permission errors in UI tests.

## Context
- .gsd/SPEC.md
- .gsd/ARCHITECTURE.md
- core/database.py
- tests/test_database.py
- tests/test_app_ui_logic.py
- tests/test_pipelines_extended.py

## Tasks

<task type="auto">
  <name>Fix Database API in Tests</name>
  <files>
    - tests/test_database.py
    - tests/test_pipelines_extended.py
    - tests/mock_app.py (if applicable)
  </files>
  <action>
    Search for all occurrences of `.create_tables()` in the test suite and replace them with `.migrate()`.
    Ensure the Database initialization in tests correctly handles the transition to migration-based schema setup.
  </action>
  <verify>pytest tests/test_database.py tests/test_pipelines_extended.py</verify>
  <done>AttributeError related to 'create_tables' is resolved in all referenced files.</done>
</task>

<task type="auto">
  <name>Fix UI Permission Errors</name>
  <files>
    - tests/test_app_ui_logic.py
  </files>
  <action>
    Identify hardcoded references to `/out` or other root-level protected directories.
    Update them to use a temporary directory or a project-relative path like `Path("tests/temp_out")`.
    Ensure paths are handled using `pathlib.Path` as per conventions in AGENTS.md.
  </action>
  <verify>pytest tests/test_app_ui_logic.py</verify>
  <done>PermissionError: [Errno 13] is resolved in UI logic tests.</done>
</task>

## Success Criteria
- [ ] `tests/test_database.py` passes or shows zero AttributeError for `create_tables`.
- [ ] `tests/test_app_ui_logic.py` passes or shows zero PermissionError for `/out`.
