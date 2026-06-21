# Agent Rules & Test Commands

## Test Execution

- Unit tests: `bash scripts/linux_test_unit.sh`
- UI/Playwright tests: `bash scripts/linux_test_ui.sh`
- Regression tests: `uv run --no-sync pytest tests/regression/ -v`
- Integration tests (requires `export PYTEST_INTEGRATION_MODE=true` and `-n 1` to prevent GPU OOM): `uv run --no-sync pytest tests/integration/ -m "integration or gpu_e2e" -n 1 -v --no-cov`
- Update visual baselines (serial only): `uv run pytest -n 0 --update-baselines tests/ui/test_visual_regression.py`

## Active Rules

- **SAM3.1 Multiplex**: The project's default tracker model and baseline for all integration tests.
- **Architectural Isolation**: NEVER import from `ui/` inside `core/`.
- **UI Safety Contract**: Event methods in `app_ui.py` MUST be wrapped in `@AppUI.safe_ui_callback`.
- **Unhashable Config**: NEVER use `@lru_cache` on functions taking `Config`.
