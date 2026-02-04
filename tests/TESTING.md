# Testing Guide

## 📚 Overview

This project uses a tiered testing strategy to ensure stability across unit logic, UI interactions, and heavy ML pipelines.

### Structure

| Layer | Directory | Description | Runner |
|-------|-----------|-------------|--------|
| **Unit** | `tests/unit/` | Fast tests for core logic. Function-level, heavily mocked. | `pytest` |
| **Integration** | `tests/integration/` | Real backend pipeline tests. Uses real PyTorch models (or mock registry) but executes full logic. | `pytest` |
| **UI (E2E)** | `tests/ui/` | Browser automation using Playwright. Mocks the backend to test UI state/flows instanty. | `pytest + playwright` |
| **Verification**| `scripts/verification/` | Manual scripts to run against a *live* server for ad-hoc quality checks. | `python` |

---

## 🚀 Running Tests

All tests should be run using `uv` to ensure the correct environment.

### 1. Unit Tests (Fast)
Run these frequently during development.
```bash
uv run pytest tests/unit/
```

### 2. Integration Tests (Slow)
Run these before committing changes to core pipelines.
```bash
uv run pytest tests/integration/
```

### 3. UI / E2E Tests (Browser)
Requires Playwright browsers installed.
```bash
# First time setup
uv run playwright install chromium

# Run all UI tests
uv run pytest tests/ui/
```

### 4. Coverage Report
```bash
uv run pytest --cov=core --cov=ui tests/
```

---

## 🧪 Verification Scripts

Located in `scripts/verification/`. These are NOT run by pytest.
They are used to verify the quality of outputs or debug specific issues on a running server.

**Example: Simple UI Check**
1. Start the app: `uv run scripts/linux_run_app.sh`
2. Run the check: `uv run python scripts/verification/verify_simple.py`

---

## 🧩 Writing Tests

### Where to add your test?

- **New Helper Function?** -> `tests/unit/test_utils.py` (or similar)
- **New UI Button?** -> `tests/ui/test_ui_interactions.py` (Mock the backend action!)
- **New ML Pipeline Step?** -> `tests/integration/test_real_workflow.py` (Verify it actually processes data)

### Mocks & Fixtures
Common fixtures are defined in `tests/conftest.py`.
- `mock_config`: A Config object with temp directories.
- `mock_ui_state`: A valid dictionary for `AnalysisParameters`.
- `mock_torch`: Fakes PyTorch for unit tests.

---

## ⚠️ Known Issues

### Numpy Boolean Assertions
**Problem**: `assert mask[0] is True` fails.
**Solution**: Use `assert mask[0]` or `assert mask[0] == True`.

### Flaky Tests
`tests/unit/test_sam3_wrapper.py` can be flaky if run in parallel due to global patching. Run it in isolation if needed:
```bash
uv run pytest tests/unit/test_sam3_wrapper.py
```
