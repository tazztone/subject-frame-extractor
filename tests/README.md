# Testing Strategy & Patterns

**Note:** This is the primary developer-facing source of truth for running and writing tests.

**Analysis Date:** 2026-03-28
**Deep Dive Refinement:** UI stability, accessibility hardening, and CI/CD integration.

## Tiered Testing Strategy

| Layer | Directory | Description | Runner |
|-------|-----------|-------------|--------|
| **Unit** | `tests/unit/` | Fast tests for core logic. Function-level, heavily mocked. | `pytest` |
| **Integration** | `tests/integration/` | Real backend pipeline tests. Uses real PyTorch models. | `pytest` |
| **UI (E2E)** | `tests/ui/` | Playwright automation. Mocks backend to test Gradio flows. | `pytest + playwright` |
| **UX Audit**| `scripts/run_ux_audit.py` | Accessibility (Axe), Visuals, and Performance. | `python` |
| **GPU E2E** | `tests/integration/` | Heavy-duty SAM2/SAM3 propagation on real hardware. | `pytest (serial)` |

## Setup & Execution

All tests should be run using `uv` to ensure the correct environment.

### Primary Entry Points (`scripts/`)

| Script | Purpose | Usage |
| :--- | :--- | :--- |
| `scripts/test.sh` | **Standard Quality Pass**. Runs Ruff, Unit Tests, and Integration Smoke. | `./scripts/test.sh` |
| `scripts/linux_test_ui.sh` | Runs Playwright tests in `tests/ui/` with xdist. | `./scripts/linux_test_ui.sh` |
| `scripts/linux_test_all.sh` | Runs the full suite with **duration logging**. | `./scripts/linux_test_all.sh` |
| `scripts/linux_test_integration.sh` | Specialized runner for GPU/Backend tests. | `./scripts/linux_test_integration.sh` |

### GPU Concurrency Warning

> [!CAUTION]
> **NEVER run GPU-heavy integration tests with `pytest -n auto`.**
> Loading multiple 3GB+ models (SAM2, SAM3, InsightFace) concurrently in parallel workers will exhaust VRAM and system RAM instantly, leading to a hard kernel-level system freeze. Always run `tests/integration/test_gpu_e2e.py` serially or with `-n 1`.

## CI/CD Pipeline

The project uses GitHub Actions (`.github/workflows/ci.yml`) for automated verification:
1. **Unit & Lint**: Runs on every push. Checks types (Pyright), lint (Ruff), and unit coverage.
2. **UI / E2E Tests**: Runs after unit tests pass. 
   - **Environment**: Ubuntu Latest, Chromium.
   - **Timeout**: 15 minutes.
   - **Workers**: Pinned to `-n 4` for deterministic mock server performance.
   - **Artifacts**: Screenshots of failures are uploaded automatically.
3. **GPU Integration Tests (Dry Run)**: A `gpu-tests` job runs on standard runners to verify the `gpu_e2e` suite.
   - **Verification**: Confirms file collection, dependency imports, and `pytest.mark.skipif` logic without requiring real hardware.

## The "Mock-First" Philosophy

To ensure fast execution and hardware independence, all **Unit Tests** must completely mock the following:
- **ML Models**: Mock `ModelRegistry.get_tracker`, `get_face_analyzer`, and `TrackerFactory`.
- **GPU/Torch**: `tests/conftest.py` promotes `torch.cuda` to a real `ModuleType` instance (not just a `MagicMock`). This provides stable access to `OutOfMemoryError` and `is_available` across parallel workers.
- **Gradio Choice Validation**: When mocking `Config` defaults for Dropdown components, always use the **Internal ID** (e.g., `"all"`) instead of the **Display Label** (e.g., `"All Frames"`). Gradio 5+ will issue `UserWarning` if the value doesn't match the internal choice list exactly.
- **Requirement**: Always include `create=True` in `patch("torch.cuda.is_available", ...)` to prevent worker collisions in parallel runs.

## Gradio & Pyright Resiliency

- **SAM3 Experimental Status**: `sam3` is an experimental tracker. Integration tests (`tests/integration/test_gpu_e2e.py`) verify its logic, but it is **not** the baseline for regressions. The default is `sam2`.
- **SAM3 Checkpoints**: SAM3 requires a local `models/sam3.pt` checkpoint. The wrapper includes a `RuntimeError` guard if the model fails to load. Never patch `download_ckpt_from_hf` as it prevents the resolver from finding local files and causes `NoneType` attribute errors.
- **HuggingFace Access**: To avoid `401 GatedRepoError` in tests, always ensure `checkpoint_path` is explicitly passed to constructors or point registries to a directory containing the real `.pt` file.
- **Type Hints**: For Gradio event handlers, use `Any` or `dict[str, Any]` for return type hints instead of `gr.update`. Gradio 5+ treats updates as dynamic dictionaries, and `gr.update` often causes Pyright noise.
- **Optional Members**: Components like `AdvancedProgressTracker` should use `Optional[Queue]` and `Optional[AppLogger]` with explicit null-checks to prevent Pyright "attribute not found on None" errors.

## Accessibility (Axe-core)

- **Pattern**: Always provide a `label` to interactive Gradio components, even if `show_label=False`. Gradio renders this as an ARIA label in the accessibility tree.
- **Rule Filtering**: Accessibility tests in `tests/ui/test_accessibility.py` are configured to skip Gradio-internal violations (e.g., `aria-hidden-focus`, `color-contrast`) that cannot be fixed at the application level.

## E2E Testing (Playwright)

- **Port Isolation**: To prevent collisions in parallel runs (`xdist`), mock servers use the formula `8765 + worker_id`. Each worker gets a unique port (8765, 8766, etc.).
- **Locator Protocol**: Never hardcode strings in tests. Always use `ui_locators.py` (`Labels` and `Selectors`).
- **Emoji Sensitivity**: Playwright is sensitive to emojis. If a UI button adds an emoji (e.g., `🚀 Start`), the `Labels` entry must match exactly.
- **Accordion Handling**: Use the `open_accordion(page, label)` helper in `conftest.py`. It uses `elem_id` (#system_logs_accordion) and JS state checks to handle Gradio's complex DOM.
- **Wait Strategy**: Use `page.wait_for_timeout(ms)` for UI animations. Use `expect(locator).to_be_visible()` for backend state changes.

## Visual Regression (`tests/ui/test_visual_regression.py`)

The suite compares current UI states against baseline screenshots using perceptual hashing.
- **Update Baselines**: To refresh baselines (e.g., after changing labels), run:
  `uv run pytest -n 0 --update-baselines tests/ui/test_visual_regression.py`
- **Constraint**: Always use `-n 0` when updating baselines to prevent race conditions during file writes. Parallel execution (`-n > 0`) is strictly forbidden for updates.

## Coverage Requirements

- **Target**: 80% total coverage (`--cov-fail-under=80` enforced in CI).
- **Manual Verification**: `scripts/linux_test_cov.sh`.

## Performance Monitoring

The `scripts/linux_test_all.sh` script automatically tracks test durations to help identify bottlenecks.

- **Log File**: Timings are saved to `tests/results/logs/test_performance.log` in a clean `[duration]s [test_name]` format.
- **Background Tracking**: Every test is timed automatically (using `--durations=0`), but the results are kept in the log to avoid terminal clutter.
- **Interpreting Results**: Use the "slowest durations" section at the end of each test stage to identify tests that may need better mocking or optimization.

## Common Gotchas & Troubleshooting

### Tests are "Deselected" (0 selected)
If you try to run a specific test file (e.g., `tests/integration/test_gpu_e2e.py`) and see `0 selected` or many tests being deselected:
- **Cause**: The `pyproject.toml` contains a default filter: `-m 'not integration and not gpu_e2e and not audit and not slow'`.
- **Fix**: You **MUST** explicitly include the marker in your command.
  ```bash
  # Correct way to run GPU tests
  uv run pytest tests/integration/test_gpu_e2e.py -m "gpu_e2e"
  ```

### Mocks are not disabling in Integration tests
- **Cause**: The global mock system in `tests/conftest.py` only disables when `PYTEST_INTEGRATION_MODE=true` is set.
- **Requirement**: Always `export` the variable before running integration/E2E tests.
  ```bash
  export PYTEST_INTEGRATION_MODE=true
  uv run pytest tests/integration/ -m "integration"
  ```

---

*Last Updated: 2026-03-31 (Duration Logging & Troubleshooting)*
