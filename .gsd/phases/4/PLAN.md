---
phase: 4
plan: stabilize-ui-tests
wave: 1
gap_closure: true
---

# Fix: Stabilize UI Tests

## Problem
The UI test suite (formerly E2E) is experiencing high failure rates (~12%) and slow execution. This is primarily due to:
1. Selector/label divergence between the tests and the current Gradio UI.
2. Slow execution of accessibility and UX audits during failures (timeouts).
3. Incomplete mocking in `mock_app.py` for certain UI states.

## Root Cause
- Gradio labels were updated in the `ui/tabs/` refactor but tests were not fully synchronized.
- Default Playwright timeouts make failing tests appear "unrealistically slow".

## Tasks

<task type="auto">
  <name>Update UI Test Selectors</name>
  <files>
    <file>tests/ui/conftest.py</file>
    <file>tests/ui/test_app_flow.py</file>
    <file>tests/ui/test_advanced_workflow.py</file>
  </files>
  <action>Synchronize all button names, tab names, and input labels with the current `ui/tabs/*.py` definitions.</action>
  <verify>uv run pytest tests/ui/test_app_flow.py</verify>
  <done>Main workflow test passes.</done>
</task>

<task type="auto">
  <name>Isolate Slow Audits</name>
  <files>
    <file>tests/ui/test_accessibility.py</file>
    <file>tests/ui/test_ai_ux_audit.py</file>
    <file>pyproject.toml</file>
  </files>
  <action>Add `@pytest.mark.slow` or `@pytest.mark.audit` to heavy tests. Update `pyproject.toml` default `addopts` to exclude them by default.</action>
  <verify>uv run pytest tests/ui/</verify>
  <done>Default UI test run is fast and excludes audits.</done>
</task>

<task type="auto">
  <name>Update mock_app State Logic</name>
  <files>
    <file>tests/mock_app.py</file>
  </files>
  <action>Ensure `mock_app` correctly populates `application_state` keys that the UI expects after extraction and pre-analysis.</action>
  <verify>uv run pytest tests/ui/test_app_flow.py</verify>
  <done>Mock app state matches production UI expectations.</done>
</task>
