# Testing Strategy & Patterns

**Analysis Date:** 2026-03-25
**Deep Dive Refinement:** Standardized mocking, SAM2.1 migration patterns, UI interaction stability, and refactoring for testability.

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

### Refactoring for Testability
Avoid complex `sys.modules` hacking to test functions that perform inline imports. Instead, **extract the logic into a standalone function** that takes simple types (like `Path` or `str`) and test that function directly. 

**Example**: Instead of mocking the entire `import sam3` chain to check a file hash in `apply_patches()`, extract `_check_sam3_version(path)` as a pure function and test it with a real temp file and a single `patch("logger")`.

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

- **Use Stable ID Selectors**: Always prefer ID-based CSS selectors (`#elem_id`) over fragile text labels. Target nested elements specifically:
    - Textboxes/Logs: `#unified_log textarea`
    - Sliders: `#my_slider input[type=range]`
- **Navigation Optimization**: Use `wait_until="domcontentloaded"` in `page.goto()`. Avoid `networkidle` as Gradio's persistent WebSockets and heavy payloads cause timeouts.
- **Fail Fast**: Set a short global timeout (e.g., 5000ms) in `conftest.py` to ensure fast feedback.
- **Accordion Orchestration**: Explicitly click accordion headers to open them before interacting with nested components. Check `is_visible()` first to avoid accidental toggling.
- **Flexible Text Selectors**: Avoid `exact=True` for text-based selectors that include emojis or dynamic labels (e.g., Gradio's accordion headers). Use `exact=False` substring matching to improve robustness.
- **Strict Mode**: Use `get_by_label` or scoped locators to avoid multiple matches for common terms (e.g., "By Face" in both labels and help text).

## Performance Benchmarking

Integration tests in `tests/integration/` track execution time and VRAM usage.
- **CI/CD**: These tests are typically skipped in standard GitHub Actions unless a runner with a GPU is specified.

## Integration Smoke Tests

Use `test_integration_smoke.py` to verify manager wiring without real ML inference. This catches "wiring" bugs (e.g., incorrect argument passing to pipelines) without requiring a GPU.

---

*Refined testing: 2026-03-25*
