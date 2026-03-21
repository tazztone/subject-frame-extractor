# Testing Patterns

**Analysis Date:** 2026-03-21

## Test Framework

**Runner:**
- `pytest` (configured via `pyproject.toml` and `pytest.ini` settings)
- `pytest-playwright` for UI/E2E testing

**Assertion Library:**
- Standard Python `assert` statements.
- `unittest.mock` for verification of calls and side effects.

**Run Commands:**
```bash
uv run pytest tests/unit/             # Run fast unit tests
uv run pytest tests/integration/      # Run heavy integration tests (requires GPU)
uv run pytest tests/ui/               # Run browser E2E tests (Playwright)
uv run pytest tests/smoke/            # Run import/smoke tests
uv run pytest --cov=core tests/       # Run coverage report
```

## Test File Organization

**Location:**
- Dedicated `tests/` directory at the project root.
- Organized by test tier: `unit/`, `integration/`, `ui/`, `signature/`, `smoke/`.

**Naming:**
- `test_*.py` for all test files.
- Grouped by the component being tested (e.g., `test_database.py`, `test_pipelines.py`).

## Test Structure

**Fixture Usage:**
Shared fixtures are defined in `tests/conftest.py` to ensure consistency and reduce setup code.

**Mocking Patterns:**
```python
from unittest.mock import MagicMock, patch

def test_example_with_mock(mock_config):
    # Arrange
    mock_service = MagicMock()
    mock_service.process.return_value = {"status": "ok"}
    
    # Act
    with patch("core.module.Service", return_value=mock_service):
        result = some_function(mock_config)
    
    # Assert
    assert result == "expected"
    mock_service.process.assert_called_once()
```

## Fixtures and Factories

**Common Fixtures (`tests/conftest.py`):**
- `mock_config`: Pre-configured mock of the `Config` Pydantic model.
- `sample_image`: 100x100 RGB numpy array for image processing tests.
- `mock_ui_state`: Dictionary of default UI event parameters.
- `mock_params`: `AnalysisParameters` instance populated from `mock_ui_state`.
- `mock_model_registry`: Mock for the heavy model management singleton.

## Test Tiers

**Unit Tests (`tests/unit/`):**
- **Scope:** Testing individual functions, operators, and small class interactions.
- **Mocking:** All heavy ML models (SAM3, InsightFace) and I/O (FFmpeg, Filesystem) MUST be mocked.
- **Speed:** Must be extremely fast (<10s for the whole suite).

**Integration Tests (`tests/integration/`):**
- **Scope:** Testing full pipelines (`AnalysisPipeline`, `ExportPipeline`) with realistic data flow.
- **Requirements:** Often requires a GPU and local model weights. Large artifacts (videos) may be used.

**UI / Browser Tests (`tests/ui/`):**
- **Scope:** Testing the Gradio interface using Playwright.
- **Strategy:** Often uses `mock_app.py` to run a headless version of the UI with mocked backend calls to ensure fast, deterministic UI verification.

## Common Patterns

**Async Testing:**
- Use `@pytest.mark.asyncio` for testing async functions or Gradio async handlers.

**GPU/Hardware Testing:**
- Use `@pytest.mark.gpu_e2e` or `@pytest.mark.integration` to skip heavy tests in speed-focused CI runs.
- Check `torch.cuda.is_available()` within tests to provide clean fallback or skip messaging.

---

*Testing analysis: 2026-03-21*
*Update when test patterns change*
