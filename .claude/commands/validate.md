---
description: Run comprehensive validation including linting, type checking, unit tests, and end-to-end tests.
---

# Validate Codebase

This workflow validates the `subject-frame-extractor` application.

## Prerequisite: Environment Setup
Ensure you have the virtual environment activated and dependencies installed.
```bash
source venv/bin/activate
pip install -r requirements.txt
pip install -r tests/requirements-test.txt
playwright install chromium
```

## Phase 1: Linting
Check for code style and potential errors using `ruff`.
```bash
ruff check .
```

## Phase 2: Type Checking
Verify type safety using `mypy`.
```bash
mypy . --ignore-missing-imports --exclude venv --exclude SAM3_repo
```

## Phase 3: Unit Testing
Run the backend unit tests to verify logic in isolation.
```bash
python -m pytest tests/test.py
```

## Phase 4: End-to-End Testing
Run the comprehensive E2E suite using Playwright against a mocked application instance.
This verifies the full user journey: Extraction -> Pre-Analysis -> Propagation -> Analysis -> Export.
```bash
python -m pytest tests/e2e/test_app_flow.py
```
