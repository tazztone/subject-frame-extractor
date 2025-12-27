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
*   `tests/e2e/test_with_sample_data.py`: Full extraction and scene detection workflow.

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
