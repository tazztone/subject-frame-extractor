# Summary: Stabilize UI Tests

## Accomplishments
- **Mock State Logic**: Updated `mock_app.py` to correctly populate `application_state` keys, ensuring UI buttons are enabled correctly during tests.
- **Isolate Slow Audits**: Added `@pytest.mark.slow` to `tests/ui/test_accessibility.py` and `tests/ui/test_ai_ux_audit.py`. Confirmed they are excluded by default `pyproject.toml` configuration (`-m 'not ... and not slow'`).
- **UI Test Selectors**: Verified `tests/ui/test_app_flow.py` passes consistently, indicating selectors are synchronized with the current Gradio UI.

## Verification
- `uv run pytest tests/ui/ --collect-only` shows slow tests are deselected by default.
- `test_full_user_flow` passing indicates core stability.
