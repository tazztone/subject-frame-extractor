# Testing Guide

## Overview
This project uses `pytest` for testing and `playwright` for end-to-end (E2E) UI testing.

## Running Tests

### 1. Unit & Integration Tests
These tests mock heavy dependencies like PyTorch and SAM3.
```bash
python -m pytest tests/test_core.py tests/test_ui_unit.py
```

### 2. End-to-End (E2E) Tests
These tests launch a mock version of the application (`mock_app.py`) that simulates ML operations, allowing full UI workflow verification without a GPU.

**Prerequisites:**
```bash
pip install pytest-playwright
playwright install chromium
```

**Run all E2E tests:**
```bash
python -m pytest tests/e2e/
```

**Run specific meaningful E2E tests:**
*   `tests/e2e/test_filters_real.py`: Verifies real filtering logic with sample video.
*   `tests/e2e/test_export_flow.py`: Verifies export workflow and UI visibility.
*   `tests/e2e/test_full_workflow_mocked.py`: Full user journey (Source -> Export) against mock backend.

### 3. Real Integration & GPU Tests (New)
These tests run the **real** application logic (no mocks) with actual models and data. They are perfect for final verification before release.

**Prerequisites:**
- Valid video files in `downloads/` (e.g., `example clip (2).mp4`).
- A GPU with CUDA support (for GPU tests).

**Run the Full Real-World Pipeline:**
This replaces the manual `verification/e2e_run.py` script.
```bash
python -m pytest tests/integration/ -m integration
```

**Run GPU Hardware Tests:**
Verifies that models load and run on the GPU without OOM or dtype errors.
```bash
python -m pytest tests/test_gpu_e2e.py -m gpu_e2e
```

## New Tests (Dec 2024)
*   `tests/test_export_advanced.py`: Covers advanced export logic like cropping and error handling.
*   `tests/test_managers_extended.py`: Covers caching and model loading retries.

## Mock App
The `tests/mock_app.py` script mimics the real `app.py` but replaces:
- `execute_extraction` with a dummy file generator.
- `execute_pre_analysis` with random scene generation.
- `execute_analysis` with fake metric generation.

This ensures tests run fast and deterministic.

## Writing New Tests
- **UI Logic**: Add to `tests/test_ui_unit.py` using mocks.
- **Backend Logic**: Add to `tests/test_core.py`.
- **User Flows**: Add to `tests/e2e/` using `page` fixture from Playwright. Use `app_server` fixture to automatically start/stop the mock app.

## Coverage
To check coverage:
```bash
python -m pytest --cov=app --cov=core --cov=ui tests/
```

## Known Issues & Gotchas

### Numpy Boolean Assertions
**Problem**: Using `is True` or `is False` in assertions fails with numpy arrays.
```python
# ❌ Fails - numpy.bool_ is not Python's True/False singleton
assert mask[0] is True

# ✅ Works
assert mask[0] == True
assert mask[0]  # for truthy check
```

**Why**: Numpy returns `numpy.bool_` objects, not Python's `bool`. The `is` operator checks identity, not value.

### Test Isolation Issues
Some tests in `test_sam3_wrapper.py` may fail when run with the full suite but pass individually. This is due to mock state leaking between test files via `conftest.py` patches.

**Workaround**: Run flaky tests in isolation:
```bash
python -m pytest tests/test_sam3_wrapper.py -v
```

### Ruff Linting
The project uses Ruff for linting. Some lint rules are ignored in tests:
- `E722`: Bare `except:` (acceptable for defensive cleanup)
- `E712`: Equality comparisons to `True`/`False` (needed for numpy bools)

See `pyproject.toml` `[tool.ruff.lint.per-file-ignores]` for details.

