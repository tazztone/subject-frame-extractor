# Testing Strategy & Patterns

**Analysis Date:** 2026-03-24
**Deep Dive Refinement:** Standardized mocking, SAM2.1 migration patterns, and coverage enforcement.

## The "Mock-First" Philosophy

To ensure fast execution and hardware independence, all **Unit Tests** must completely mock the following:
- **ML Models**: Mock `ModelRegistry.get_tracker`, `get_face_analyzer`, and `TrackerFactory`.
- **GPU/Torch**: Use the `ModuleType` spoofing pattern in `conftest.py` to prevent double-init errors.
- **File I/O**: Use `sample_image` and `sample_mask` fixtures instead of reading from disk.

## Core Fixtures (`tests/conftest.py`)

Always reuse these base fixtures to ensure consistency across the suite:
- `mock_config`: Returns a `MagicMock` with calibrated quality weights and directory paths.
- `mock_torch`: A fixture that ensures `torch` is correctly stubbed for unit tests.
- `sample_image`: A stable 100x100 RGB noise image (seed 42). Use for basic operator tests.
- `sharp_image` / `blurry_image`: Predetermined patterns for validating sharpness metrics.
- `mock_ui_state`: A baseline dictionary for validating `UIEvent` pydantic models.

## Torch & Heavy Dependency Mocking

Unit tests must never import real `torch`, `sam2`, or `insightface`.
Use the `ModuleType` stub pattern in `conftest.py`:

```python
import sys, types
sys.modules["torch"] = types.ModuleType("torch")
sys.modules["torch.cuda"] = types.ModuleType("torch.cuda")
```

This must be applied **before any core module is imported** to prevent double-init errors.

## Coverage Requirements

- **Target**: 80% total coverage (`--cov-fail-under=80` enforced in CI)
- **Run locally**: `uv run pytest --cov=core --cov=ui tests/unit/`
- CI will reject PRs that drop below 80%.

## Testing Tracker-Agnostic Components

Since `SubjectMasker`, `MaskPropagator`, and `SeedSelector` are now tracker-agnostic, always inject via `TrackerFactory` mock — never instantiate `SAM3Wrapper` or `SAM21Wrapper` directly in unit tests.

## "Golden" Test Case Example: Operator Execution

A "Golden" unit test follows the Arrange-Act-Assert pattern with strict mocking:

```python
def test_my_metric_operator_golden(mock_config, sample_image, mock_logger):
    # 1. Arrange: Setup registry and context
    OperatorRegistry.register(MyMetricOperator)
    ctx = OperatorContext(
        image_rgb=sample_image,
        config=mock_config,
        logger=mock_logger,
        params={}
    )
    
    # 2. Act: Execute the operator
    op = MyMetricOperator()
    result = op.execute(ctx)
    
    # 3. Assert: Verify results and metric ranges
    assert "my_metric_score" in result.metrics
    assert 0.0 <= result.metrics["my_metric_score"] <= 100.0
    mock_logger.error.assert_not_called()
```

## Regression Testing (`--capture-golden`)

The system supports a "Golden Reference" workflow for ML metrics:
1.  **Capture**: Run `pytest --capture-golden` to save current metric outputs to `tests/data/golden_metrics.json`.
2.  **Verify**: Subsequent runs compare new outputs against these stored values to detect drift in quality analysis logic.

## E2E Testing (Playwright)

Located in `tests/ui/`, these tests use Playwright to:
- Verify that clicking "Run Extraction" triggers the `unified_status` update.
- Ensure that the "Cancel" button successfully sends the `cancel_event`.
- **Note**: Always use `page.wait_for_selector` for dynamic Gradio elements which may take time to render.

## Performance Benchmarking

Integration tests in `tests/integration/` track execution time and VRAM usage.
- **CI/CD**: These tests are typically skipped in standard GitHub Actions unless a runner with a GPU is specified.

## Integration Smoke Tests

Use `test_integration_smoke.py` to verify manager wiring without real ML inference. This catches "wiring" bugs (e.g., incorrect argument passing to pipelines) without requiring a GPU.

---

*Refined testing: 2026-03-24*
