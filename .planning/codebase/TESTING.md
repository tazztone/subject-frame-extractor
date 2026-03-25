# Testing Strategy & Patterns

**Analysis Date:** 2026-03-25
**Deep Dive Refinement:** Standardized mocking, SAM2.1 migration patterns, UI interaction stability, and refactoring for testability.

## The "Mock-First" Philosophy

To ensure fast execution and hardware independence, all **Unit Tests** must completely mock the following:
- **ML Models**: Mock `ModelRegistry.get_tracker`, `get_face_analyzer`, and `TrackerFactory`.
- **GPU/Torch**: Use `pytest_sessionstart` in `tests/conftest.py` to globally intercept `torch`, `sam3`, and `insightface` before any imports occur.
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
We use a global interception pattern in `tests/conftest.py` via the `pytest_sessionstart` hook:

```python
def pytest_sessionstart(session):
    # 1. Create robust mock shells for heavy modules
    mock_torch = MagicMock(name="torch")
    # 2. Inject OutOfMemoryError for registry robustness
    class OutOfMemoryError(RuntimeError): pass
    mock_torch.cuda.OutOfMemoryError = OutOfMemoryError
    
    # 3. Use sys.modules to shadow the real libraries
    sys.modules["torch"] = mock_torch
    sys.modules["sam3"] = types.ModuleType("sam3")
    ...
```

This ensures that even third-party libraries that import `torch` at top-level will receive the mock instead of triggering a VRAM allocation.

### Gradio & Matplotlib Mocking
When mocking `matplotlib` for a Gradio-based application (as in `tests/mock_app.py`), you MUST satisfy Gradio's internal `MatplotlibBackendMananger`. The mock must include at minimum:
- `matplotlib.get_backend()` (return a string like "agg")
- `matplotlib.use()`
- `matplotlib.rcParams` (dictionary)

Failure to provide these will cause silent "Error" prefixes in UI status components.

### Exception Identity & Sub-module Mocks
If a module uses `from scenedetect import VideoOpenFailure`, and you mock `scenedetect` globally, the class identity might mismatch if the module was loaded prematurely or if the mock structure doesn't match the library's re-exports.
- **Best Practice**: Use a scoped `patch("module.under.test.VideoOpenFailure", VideoOpenFailure)` in the test method to ensure the `except` block in the target module perfectly matches the exception raised by the mock.

### Robust `MockTensor` Support
For tests involving mask processing (e.g., SAM3/SAM2.1), `conftest.py` provides a `MockTensor` class that simulates `torch.Tensor` behavior:
- **Shape Propagation**: Returns correctly shaped numpy arrays via `.cpu().numpy()`.
- **Dunder Support**: Implements `__len__`, `__getitem__`, and `__gt__` for filtering logic.
- **Arithmetic**: Supports scalar operations used in quality metrics.

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

## Deep Property-Based Testing (Hypothesis)

Don't settle for "no-crash" tests. Use `hypothesis` to define mathematical invariants for geometry logic.

**Example: BBox Normalization**
When converting pixel coordinates to relative `[0, 1]` coordinates for SAM3:
- **Invariant**: The output coordinates *must* stay within `[0, 1]` regardless of the image size or input pixel value (even if input is slightly out of bounds).
- **Outcome**: This pattern surfaced a critical division-by-zero risk in the early SAM3 implementation.

```python
@given(
    x=st.floats(min_value=-1000, max_value=5000),
    img_w=st.integers(min_value=1, max_value=8000)
)
def test_bbox_normalization_invariant(x, img_w):
    rel_x = normalize_coord(x, img_w)
    assert 0.0 <= rel_x <= 1.0  # Mathematical guarantee
```
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

### Mandatory E2E Patterns

- **Log Persistence vs. UI Updates**:
    - **Observation**: In Gradio, updating a component via a return value in an event handler (like `.click()`) replaces the value.
    - **Mandate**: Never return log strings directly to `unified_log` for long-running processes. Instead, use `logger.info()`. The `LogViewer` component periodically polls the log queue and appends content, preserving the history that E2E tests rely on for "Log Trail" verification.
- **Mock Side-Effect Consistency**:
    - **Requirement**: Mocks for pipelines (Extraction, Analysis, etc.) must not only return the expected dictionary but also trigger a `logger.info()` call. Without this, the UI log viewer remains empty or missing stages during E2E runs, leading to assertion failures in tests that verify the workflow's history.
- **Status Bar Priority**:
    - **Design Pattern**: The `unified_status` component is the primary target for E2E assertions. Any major backend operation should update this component immediately upon completion/failure to provide a clear signal to Playwright's `expect()` locators.
- **Case Sensitivity in E2E**:
    - **Rule**: E2E tests are extremely fragile regarding status strings. "Pre-Analysis" (Title Case) vs "Pre-analysis" (Sentence Case) will cause a test failure if the log scraper is looking for a specific marker. Always use **Title Case** for major pipeline milestones.

- **Standardized Wait Strategy**:
    - **Rule**: Always use `page.wait_for_timeout(ms)` for arbitrary buffers in Playwright tests. Never use `time.sleep(s)` inside test functions or Playwright-managed fixtures as it blocks the Python event loop and contributes to flakiness.
    - **Exception**: `time.sleep()` is permitted in infrastructure setup (like `wait_for_server` or `cleanup_port`) where no browser context is active.

## Signature Validation & Mock Sync

To prevent "mock drift" where `mock_app.py` or unit test mocks become outdated relative to production code:
- **Mandate**: Use `inspect.signature()` in `tests/unit/test_signatures.py` to compare mocked interfaces against real implementations.
- **Scope**: All `execute_*` pipeline functions and `SAM3Wrapper` methods must be validated for parameter-for-parameter matching.

## Performance Benchmarking

Integration tests in `tests/integration/` track execution time and VRAM usage.
- **CI/CD**: These tests are typically skipped in standard GitHub Actions unless a runner with a GPU is specified.

## Integration Smoke Tests

Use `test_integration_smoke.py` to verify manager wiring without real ML inference. This catches "wiring" bugs (e.g., incorrect argument passing to pipelines) without requiring a GPU.

---

*Refined testing: 2026-03-25*
