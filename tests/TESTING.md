# Testing Guide

**Last Updated**: 2025-02-12
**Python**: 3.10+ | **Pytest**: 7.x+ | **Playwright**: 1.40+

---

## Quick Start

> **🔴 CRITICAL**: Always activate the virtual environment before running commands! Use `uv pip` for faster installs.

```bash
# Activate virtual environment first
.\venv\Scripts\activate.ps1   # Windows
source venv/bin/activate       # Linux/Mac

# Run unit tests (fast, ~30s, mocks heavy dependencies)
python -m pytest tests/ -k "not integration"

# Run integration tests (slower, requires real dependencies/imports)
python -m pytest tests/test_integration*

# Run smoke tests only (import validation)
python -m pytest tests/test_smoke.py -v

# Run GPU E2E tests (requires CUDA + models, ~5min)
python -m pytest tests/test_gpu_e2e.py -v -m ""

# Run Playwright E2E tests (requires mock app)
python tests/mock_app.py &
python -m pytest tests/e2e/ -v

# Run full UX audit suite
python scripts/run_ux_audit.py
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
│     │   UX Tests  │  ← Visual regression, accessibility, AI    │
│     └──────┬──────┘                                             │
│            │                                                    │
│     ┌──────▼──────┐                                             │
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

## Execution Strategy: Unit vs Integration

The test suite is divided into **Unit Tests** and **Integration Tests** to handle heavy dependencies (Torch, CUDA, SAM3) efficiently.

### Unit Tests
- **Files**: `test_core.py`, `test_managers.py`, `test_pipelines_extended.py`, `test_filtering.py`, etc.
- **Behavior**: Aggressively mock heavy libraries (`torch`, `sam3`, `insightface`) via `tests/conftest.py`.
- **Command**: `python -m pytest tests/ -k "not integration"`
- **Why**: Allows fast feedback loops without loading GBs of models.

### Integration Tests
- **Files**: `test_integration.py`, `test_integration_sam3_patches.py`, `test_integration_sam3_patches_unit.py`.
- **Behavior**: Bypass global mocks to use real libraries. Essential for testing tensor operations, file I/O, and model fallback logic.
- **Command**: `python -m pytest tests/test_integration*`

> **Note**: Running `pytest tests/` blindly may fail because integration tests might receive mocked dependencies. Use the split commands above.

---

## Test Categories

| Marker | File Pattern | GPU? | Mock? | Run Command |
|--------|-------------|------|-------|-------------|
| `unit` | `test_*.py` | No | Yes | `pytest tests/ -k "not integration"` |
| `integration` | `test_integration*.py` | Yes | No | `pytest tests/test_integration*` |
| `smoke` | `test_smoke.py` | No | No | `pytest tests/test_smoke.py` |
| `signature` | `test_signatures.py` | No | No | `pytest tests/test_signatures.py` |
| `gpu_e2e` | `test_gpu_e2e.py` | Yes | No | `pytest tests/test_gpu_e2e.py -m ""` |
| `e2e` | `tests/e2e/` | No | Yes | `pytest tests/e2e/` |
| `bug_regression` | `test_bug_regression.py` | No | Yes | `pytest tests/e2e/test_bug_regression.py` |
| `component` | `test_component_verification.py` | No | Yes | `pytest -m component` |
| `visual` | `test_visual_regression.py` | No | Yes | `pytest -m visual` |
| `accessibility` | `test_accessibility.py` | No | Yes | `pytest -m accessibility` |
| `ux_audit` | `test_ai_ux_audit.py` | No | Yes | `pytest -m ux_audit` |

---

## Unit Tests

Fast, isolated tests using mocked dependencies. These run on every commit.

**Files**: `test_core.py`, `test_pipelines.py`, `test_database.py`, `test_filtering.py`, `test_ui_unit.py`, `test_managers.py`.

**Mocks Applied**: torch, torchvision, sam3, insightface, mediapipe, pyiqa

```bash
# Run all unit tests
python -m pytest tests/ -k "not integration" -v

# Run specific test class
python -m pytest tests/test_core.py::TestUtils -v

# Run UI unit tests (Phase 0 fixes)
python -m pytest tests/test_ui_unit.py -v
```

### Key UI Unit Tests

| Test Class | Purpose |
|------------|---------|
| `TestMinConfidenceFilter` | Verifies min_confidence slider filters correctly |
| `TestTextStrategyWarning` | Verifies TEXT strategy shows warning label |

---

## Integration Tests

Tests that verify interaction with real libraries or file systems, bypassing the global mocks.

**Files**:
- `tests/test_integration_sam3_patches.py`: Verifies patching logic for SAM3 on non-Triton systems.
- `tests/test_integration_sam3_patches_unit.py`: Unit-style tests for SAM3 patch logic using real Torch tensors.

---

## Smoke Tests

Validate that all modules import without errors. These catch missing imports before runtime.

```bash
python -m pytest tests/test_smoke.py -v
```

---

## GPU E2E Tests

Real model inference tests. Catch dtype mismatches, CUDA OOM, and model loading failures.

```bash
# All GPU E2E tests
python -m pytest tests/test_gpu_e2e.py -v -m ""

# Single test class
python -m pytest tests/test_gpu_e2e.py::TestSAM3Inference -v -m ""
```

---

## Playwright E2E Tests

Browser-based tests using a mock application server.

```bash
# Start mock server
python tests/mock_app.py &

# Run all Playwright tests
python -m pytest tests/e2e/ -v -s
```

### E2E Test Files

| File | Purpose |
|------|---------|
| `test_app_flow.py` | Main workflow from extraction to export |
| `test_export_flow.py` | Export workflow and options |
| `test_session_lifecycle.py` | Session management and state |

---

## UX Testing Framework

A comprehensive suite of tests to catch UI/UX issues automatically.

### Visual Regression Tests

Capture screenshots and compare against baselines using perceptual hashing.

```bash
# Run visual regression tests
python -m pytest tests/e2e/test_visual_regression.py -v

# Update baselines (after intentional UI changes)
python -m pytest tests/e2e/test_visual_regression.py -v --update-baselines
```

**Key Files**:
- `tests/e2e/visual_test_utils.py` - Screenshot capture and comparison utilities
- `tests/e2e/test_visual_regression.py` - Visual regression test cases
- `tests/e2e/baselines/` - Reference screenshots (auto-created on first run)

**States Captured**:
- Initial app load
- Each tab (Source, Subject, Scenes, Metrics, Export)
- Expanded accordions
- Active workflows

### Component Verification Tests

Verify that each UI component actually works, not just renders.

```bash
python -m pytest tests/e2e/test_component_verification.py -v
```

**Test Classes**:

| Class | What It Tests |
|-------|---------------|
| `TestSliderFunctionality` | Sliders change values when interacted |
| `TestDropdownFunctionality` | Dropdowns can be opened and selected |
| `TestFiltersFunctionality` | View toggles change displayed content |
| `TestLogsFunctionality` | Logs are visible and have content |
| `TestPaginationFunctionality` | Page dropdown and prev/next buttons work |
| `TestButtonsFunctionality` | Critical buttons are visible and clickable |
| `TestStrategyVisibility` | Strategy selection shows correct UI groups |

### AI-Powered UX Analysis

Uses vision AI to analyze screenshots against a UX checklist.

```bash
# Without AI (uses heuristic checks)
python -m pytest tests/e2e/test_ai_ux_audit.py -v

# With AI (requires OpenAI API key)
OPENAI_API_KEY=sk-xxx python -m pytest tests/e2e/test_ai_ux_audit.py -v
```

**Key Files**:
- `tests/e2e/ai_ux_analyzer.py` - UX analysis engine and checklist
- `tests/e2e/test_ai_ux_audit.py` - AI-powered audit tests

**UX Checklist Categories**:
- Layout & alignment
- Usability (interactive elements, state visibility)
- Feedback (loading states, error messages)
- Controls (sliders, dropdowns, buttons)
- Accessibility (contrast, labels)
- Consistency (icons, terminology)

### Accessibility Audits

Automated accessibility testing using axe-core.

```bash
python -m pytest tests/e2e/test_accessibility.py -v
```

**What It Checks**:
- WCAG 2.0 AA compliance
- Color contrast
- Form labels
- Keyboard navigation
- ARIA roles and attributes

---

## Running the Full UX Audit

Use the audit runner script to execute all UX tests and generate a report:

```bash
# Full audit with report
python scripts/run_ux_audit.py

# Quick check (component tests only)
python scripts/run_ux_audit.py --quick-check

# Update visual baselines
python scripts/run_ux_audit.py --update-baselines
```

**Output**: `ux_reports/ux_audit_YYYYMMDD_HHMMSS.md`

---

## Fixtures Reference

All fixtures are defined in `tests/conftest.py` and `tests/e2e/conftest.py`.

### Configuration Fixtures

| Fixture | Scope | Description |
|---------|-------|-------------|
| `mock_config` | function | Config with temp directories |
| `mock_config_simple` | function | MagicMock config for flexibility |
| `mock_logger` | function | AppLogger for testing |

### E2E Fixtures

| Fixture | Scope | Description |
|---------|-------|-------------|
| `app_server` | module | Starts/stops mock Gradio server |
| `page` | function | Playwright page instance |

### Data Fixtures

| Fixture | Scope | Description |
|---------|-------|-------------|
| `sample_frames_data` | function | Frame metadata for filtering tests |
| `sample_scenes` | function | Scene objects for scene tests |
| `sample_image_rgb` | function | 100x100 RGB numpy array |

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
@pytest.mark.skipif(not _is_sam3_available(), reason="SAM3 not installed")
def test_sam3_feature(...):
    # ...
```

---

## CI/CD Integration

### GitHub Actions Workflows

**Unit Tests** (`.github/workflows/tests.yml`):
```yaml
- name: Run unit tests
  run: python -m pytest tests/ -k "not integration" -v --cov=core --cov=ui
```

**UX Tests** (`.github/workflows/ux-testing.yml`):
```yaml
- name: Run visual regression
  run: python -m pytest tests/e2e/test_visual_regression.py -v

- name: Run accessibility
  run: python -m pytest tests/e2e/test_accessibility.py -v
```

The UX testing workflow:
- Runs on PRs touching `ui/**` or `tests/e2e/**`
- Captures and stores visual baselines as artifacts
- Generates UX audit reports

---

## Troubleshooting

### Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `ModuleNotFoundError: sam3` | Submodule not installed | `pip install -e SAM3_repo` |
| `CUDA OOM` | Insufficient GPU memory | Set `device="cpu"` or cleanup |
| `Marker 'gpu_e2e' not found` | Default marker filter | Use `-m ""` flag |
| Playwright timeout | Mock app not running | Start `python tests/mock_app.py` |
| `ImportError: relative import` | Package structure | Add `__init__.py` to e2e/ |
| `axe-core not found` | CDN issue | Check network connectivity |
| `AttributeError: MagicMock object has no attribute 'dim'` | Unit test running with mocked torch | Ensure test file starts with `test_integration` |

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

---

## File Structure

```
tests/
├── conftest.py              # Shared fixtures and module mocks
├── mock_app.py              # Mock server for Playwright E2E
├── assets/                  # Test assets (images, videos)
│
├── test_core.py             # Config, Logger, Filtering tests
├── test_pipelines.py        # Pipeline execution tests (Basic)
├── test_pipelines_extended.py # Pipeline execution tests (Detailed)
├── test_handlers.py         # UI handler tests (Analysis, Extraction, Filtering)
├── test_managers.py         # Model and Resource Manager tests
├── test_scene_detection.py  # Scene detection and management tests
├── test_ui_unit.py          # UI component unit tests
├── test_smoke.py            # Import smoke tests
├── test_gpu_e2e.py          # GPU E2E tests (SAM3, InsightFace)
├── test_integration_sam3_patches.py # Integration tests for SAM3 patching
├── test_integration_sam3_patches_unit.py # Unit-style tests for SAM3 patches using real torch
│
└── e2e/                     # Playwright E2E tests
    ├── __init__.py          # Package init
    ├── conftest.py          # E2E fixtures (app_server, BASE_URL)
    │
    ├── test_app_flow.py           # Main workflow tests
    ├── test_export_flow.py        # Export workflow tests
    ├── test_session_lifecycle.py  # Session management tests
    │
    ├── test_component_verification.py  # Component functionality tests
    ├── test_visual_regression.py       # Visual regression tests
    ├── test_ai_ux_audit.py             # AI-powered UX analysis
    ├── test_accessibility.py           # axe-core accessibility audits
    │
    ├── visual_test_utils.py       # Screenshot utilities
    ├── ai_ux_analyzer.py          # UX analysis engine
    └── baselines/                 # Visual regression baselines
```

---

## Coverage Targets

| Category | Target | Current (Feb 2025) |
|----------|--------|--------------------|
| Total Project| 80%   | **61%** (Up from 50%) |
| Core modules | 80%   | **60-80%** |
| UI handlers  | 60%   | **75%** |
| Pipeline execution | 70% | **70%** (Improved with extended tests) |
| Error paths  | 50%   | **50%** (Managers & Pipelines) |
| UX regression | 100% | 100% (All tabs covered) |

Run coverage report:
```bash
python -m pytest tests/ -k "not integration" --cov=core --cov=ui --cov-report=html
open htmlcov/index.html
```
