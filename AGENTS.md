# Agent Context

> See `docs/CONVENTIONS.md` for detailed architectural rationale, UI patterns, and mock infrastructure rules.

## Core Invariants

- **Architecture**: NEVER import from `ui/` inside `core/`.
- **UI Contract**: Wrap `app_ui.py` event methods in `@AppUI.safe_ui_callback`.
- **Cache Safety**: NEVER use `@lru_cache` on functions taking `Config`.
- **Immutable Repos**: NEVER edit files in `SAM3_repo`.
- **Default State**: `SAM2.1 Hiera Tiny` is the default tracker. `sam3` is an experimental alternative.

## Testing

- **Full Suite**: `bash scripts/linux_test_all.sh`
- **Unit**: `bash scripts/linux_test_unit.sh`
- **UI**: `bash scripts/linux_test_ui.sh`
- **Regression**: `uv run --no-sync pytest tests/regression/ -v`
- **Visual Baselines**: `uv run pytest -n 0 --update-baselines tests/ui/test_visual_regression.py`
- **GPU / Integration**: 
  ```bash
  export PYTEST_INTEGRATION_MODE=true
  uv run --no-sync pytest tests/integration/ -m "integration or gpu_e2e" -n 1
  ```
