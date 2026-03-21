# Testing Strategy & Patterns

**Analysis Date:** 2026-03-21
**Deep Dive Refinement:** Standardized mocking and regression testing patterns.

## The "Mock-First" Philosophy

To ensure fast execution and hardware independence, all **Unit Tests** must completely mock the following:
- **ML Models**: Mock `ModelRegistry.get_tracker` and `get_face_analyzer`.
- **GPU/Torch**: Use `unittest.mock.patch` to simulate `torch.cuda.is_available()` returning `False` if hardware is not needed.
- **File I/O**: Use `sample_image` and `sample_mask` fixtures instead of reading from disk.

## Core Fixtures (`tests/conftest.py`)

Always reuse these base fixtures to ensure consistency across the suite:
- `mock_config`: Returns a `MagicMock` with calibrated quality weights and directory paths.
- `sample_image`: A stable 100x100 RGB noise image (seed 42). Use for basic operator tests.
- `sharp_image` / `blurry_image`: Predetermined patterns for validating sharpness metrics.
- `mock_ui_state`: A baseline dictionary for validating `UIEvent` pydantic models.

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

---

*Refined testing: 2026-03-21*
