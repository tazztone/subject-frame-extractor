# Testing Guide

**Last Updated**: 2025-12-23  
**Python**: 3.10+ | **Pytest**: 7.x+ | **Playwright**: 1.40+

---

## Quick Start

```bash
# Activate virtual environment first
. venv/Scripts/activate.ps1   # Windows
source venv/bin/activate       # Linux/Mac

# Run all unit tests (fast, ~30s)
python -m pytest tests/

# Run smoke tests only (import validation)
python -m pytest tests/test_smoke.py -v

# Run GPU E2E tests (requires CUDA + models, ~5min)
python -m pytest tests/test_gpu_e2e.py -v -m ""

# Run Playwright E2E tests (requires mock app)
python tests/mock_app.py &
python -m pytest tests/e2e/ -v
```

> **⚠️ Note**: GPU E2E tests require `-m ""` to override the default marker filter in `setup.cfg`.

---

## Test Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Test Pyramid                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│     ┌─────────────┐                                             │
│     │  E2E Tests  │  ← Playwright (browser-based, uses mock)   │
│     └──────┬──────┘                                             │
│            │                                                    │
│     ┌──────▼──────┐                                             │
│     │  GPU E2E    │  ← Real CUDA inference (SAM3, InsightFace) │
│     └──────┬──────┘                                             │
│            │                                                    │
│     ┌──────▼──────┐                                             │
│     │ Integration │  ← No mocks, real imports, GPU optional    │
│     └──────┬──────┘                                             │
│            │                                                    │
│     ┌──────▼──────┐                                             │
│     │   Smoke     │  ← Import validation, no mocks              │
│     └──────┬──────┘                                             │
│            │                                                    │
│  ┌─────────▼─────────┐                                          │
│  │    Unit Tests     │  ← Fast, isolated, use conftest mocks   │
│  └───────────────────┘                                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Test Categories

| Marker | File Pattern | GPU? | Mock? | Run Command |
|--------|-------------|------|-------|-------------|
| `unit` | `test_*.py` | No | Yes | `pytest tests/` |
| `smoke` | `test_smoke.py` | No | No | `pytest tests/test_smoke.py` |
| `signature` | `test_signatures.py` | No | No | `pytest tests/test_signatures.py` |
| `integration` | `test_integration.py` | Yes | No | `pytest -m integration` |
| `gpu_e2e` | `test_gpu_e2e.py` | Yes | No | `pytest tests/test_gpu_e2e.py -m ""` |
| `e2e` | `tests/e2e/` | No | Yes | `pytest tests/e2e/` |

### Unit Tests
Fast, isolated tests using mocked dependencies. These run on every commit.

**Files**: `test_core.py`, `test_pipelines.py`, `test_database.py`, `test_filtering.py`, etc.

**Mocks Applied**: torch, torchvision, sam3, insightface, mediapipe, pyiqa

```bash
# Run all unit tests
python -m pytest tests/ -v

# Run specific test class
python -m pytest tests/test_core.py::TestUtils -v
```

### Smoke Tests
Validate that all modules import without errors. These catch missing imports before runtime.

```bash
python -m pytest tests/test_smoke.py -v
```

### Integration Tests
Run without mocks to catch real integration issues. Require GPU and all dependencies.

```bash
python -m pytest tests/test_integration.py -v -m integration
```

### GPU E2E Tests
Real model inference tests. Catch dtype mismatches, CUDA OOM, and model loading failures.

```bash
# All GPU E2E tests
python -m pytest tests/test_gpu_e2e.py -v -m ""

# Single test class
python -m pytest tests/test_gpu_e2e.py::TestSAM3Inference -v -m ""

# Single test
python -m pytest tests/test_gpu_e2e.py::TestSAM3Inference::test_sam3_wrapper_initialization -v -m ""
```

### Playwright E2E Tests
Browser-based tests using a mock application server.

```bash
# Start mock server
python tests/mock_app.py &

# Run Playwright tests
python -m pytest tests/e2e/ -v -s
```

---

## Fixtures Reference

All fixtures are defined in `tests/conftest.py`. Use these for consistent test setup.

### Configuration Fixtures

| Fixture | Scope | Description |
|---------|-------|-------------|
| `mock_config` | function | Config with temp directories |
| `mock_config_simple` | function | MagicMock config for flexibility |
| `mock_logger` | function | AppLogger for testing |

### Manager Fixtures

| Fixture | Scope | Description |
|---------|-------|-------------|
| `mock_thumbnail_manager` | function | ThumbnailManager mock |
| `mock_model_registry` | function | ModelRegistry mock |
| `mock_progress_queue` | function | Queue for progress updates |
| `mock_cancel_event` | function | Threading Event for cancellation |

### Data Fixtures

| Fixture | Scope | Description |
|---------|-------|-------------|
| `sample_frames_data` | function | Frame metadata for filtering tests |
| `sample_scenes` | function | Scene objects for scene tests |
| `sample_image_rgb` | function | 100x100 RGB numpy array |
| `sample_mask` | function | Binary mask array |
| `mock_ui_state` | function | Default UI state dict |
| `mock_params` | function | AnalysisParameters for pipelines |

### GPU E2E Fixtures

| Fixture | Scope | Description |
|---------|-------|-------------|
| `test_image` | function | 256x256 test image with object |
| `test_image_with_face` | function | Image with face-like pattern |
| `test_frames_dir` | function | Directory with test frames |

---

## Mocking Guidelines

### When to Mock

| Component | Mock? | Reason |
|-----------|-------|--------|
| PyTorch/CUDA | ✓ | Slow, GPU-dependent |
| SAM3 | ✓ | Large model, GPU-only |
| InsightFace | ✓ | Model download required |
| MediaPipe | ✓ | Model download required |
| File I/O | ✓ | For unit test isolation |
| Network | ✓ | Avoid external dependencies |
| Config | ✗ | Use `mock_config` fixture |
| Logger | ✗ | Use `mock_logger` fixture |

### Mock Patterns

```python
# Mocking a class method
from unittest.mock import patch, MagicMock

@patch("core.managers.ModelRegistry.get_tracker")
def test_tracker(mock_get, app_ui):
    mock_get.return_value = MagicMock()
    # ...

# Skip if dependency unavailable
import pytest

@pytest.mark.skipif(not _is_sam3_available(), reason="SAM3 not installed")
def test_sam3_feature(...):
    # ...

# Using conftest fixtures
def test_filtering(sample_frames_data, mock_config):
    kept, rejected, _, _ = apply_all_filters_vectorized(
        sample_frames_data, filters, mock_config
    )
    assert len(kept) > 0
```

---

## Adding New Tests

### 1. Unit Test
```python
# tests/test_my_feature.py
import pytest
from core.my_module import my_function

class TestMyFeature:
    def test_basic_case(self, mock_config):
        result = my_function(mock_config)
        assert result is not None
```

### 2. GPU E2E Test
```python
# In tests/test_gpu_e2e.py
@requires_sam3
def test_new_sam3_feature(self, test_frames_dir):
    import torch
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    from core.managers import SAM3Wrapper
    wrapper = SAM3Wrapper(device="cuda")
    try:
        # Test implementation
        pass
    finally:
        wrapper.cleanup()
```

### 3. Playwright E2E Test
```python
# tests/e2e/test_my_flow.py
def test_my_flow(page, app_server):
    page.goto(BASE_URL)
    page.get_by_role("button", name="My Button").click()
    expect(page.locator("#result")).to_contain_text("Success")
```

---

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Tests

on: [push, pull_request]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          submodules: recursive
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.10'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov
      
      - name: Run unit tests
        run: python -m pytest tests/ -v --cov=core --cov=ui

  smoke-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          submodules: recursive
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.10'
      
      - name: Install dependencies
        run: pip install -r requirements.txt
      
      - name: Run smoke tests
        run: python -m pytest tests/test_smoke.py -v
```

---

## Troubleshooting

### Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `ModuleNotFoundError: sam3` | Submodule not installed | `pip install -e SAM3_repo` |
| `CUDA OOM` | Insufficient GPU memory | Set `device="cpu"` or cleanup models |
| `Marker 'gpu_e2e' not found` | Default marker filter | Use `-m ""` flag |
| Playwright timeout | Mock app not running | Start `python tests/mock_app.py` first |

### Debugging Tests

```bash
# Run with output visible
python -m pytest tests/test_core.py -v -s

# Run with pdb on failure
python -m pytest tests/test_core.py --pdb

# Run with full traceback
python -m pytest tests/test_core.py --tb=long

# Run specific test by name
python -m pytest tests/ -k "test_config"
```

### GPU Memory Issues

```python
# In test cleanup
import torch
torch.cuda.empty_cache()

# Or use wrapper.cleanup() for SAM3
wrapper.cleanup()
```

---

## File Structure

```
tests/
├── conftest.py              # Shared fixtures and module mocks
├── mock_app.py              # Mock server for Playwright E2E
├── assets/                  # Test assets (images, videos)
│
├── test_batch_manager.py    # BatchManager unit tests
├── test_core.py             # Config, Logger, Filtering tests
├── test_database.py         # Database CRUD tests
├── test_dedup.py            # Deduplication tests
├── test_error_handling.py   # ErrorHandler tests
├── test_export.py           # Export functionality tests
├── test_gallery_utils.py    # Gallery utility tests
├── test_integration.py      # Integration tests (GPU)
├── test_managers.py         # ModelRegistry, ThumbnailManager tests
├── test_pipelines.py        # Pipeline execution tests
├── test_progress.py         # Progress tracking tests
├── test_scene_utils.py      # Scene processing tests
├── test_signatures.py       # Function signature validation
├── test_smoke.py            # Import smoke tests
├── test_ui_unit.py          # UI component tests
│
├── test_gpu_e2e.py          # GPU E2E tests (SAM3, InsightFace)
│
└── e2e/                     # Playwright E2E tests
    ├── test_app_flow.py     # Main workflow test
    ├── test_export_flow.py  # Export workflow tests
    └── test_session_lifecycle.py  # Session management tests
```

---

## Coverage Targets

| Category | Target | Current |
|----------|--------|---------|
| Core modules | 80% | ~70% |
| UI handlers | 60% | ~50% |
| Pipeline execution | 70% | ~65% |
| Error paths | 50% | ~40% |

Run coverage report:
```bash
python -m pytest tests/ --cov=core --cov=ui --cov-report=html
open htmlcov/index.html
```
