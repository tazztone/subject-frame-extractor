# Testing Strategy & Patterns

**Note:** `tests/README.md` is the primary developer-facing source of truth for running and writing tests. This file mirrors that content for architectural reference.

**Analysis Date:** 2026-03-28
**Deep Dive Refinement:** UI stability, accessibility hardening, and CI/CD integration.

## Tiered Testing Strategy

| Layer | Directory | Description | Runner |
|-------|-----------|-------------|--------|
| **Unit** | `tests/unit/` | Fast tests for core logic. Function-level, heavily mocked. | `pytest` |
| **Integration** | `tests/integration/` | Real backend pipeline tests. Uses real PyTorch models. | `pytest` |
| **UI (E2E)** | `tests/ui/` | Playwright automation. Mocks backend to test Gradio flows. | `pytest + playwright` |
| **UX Audit**| `scripts/run_ux_audit.py` | Accessibility (Axe), Visuals, and Performance. | `python` |

## Setup & Execution

All tests should be run using `uv` to ensure the correct environment.

### Primary Entry Points (`scripts/`)

| Script | Purpose | Usage |
| :--- | :--- | :--- |
| `scripts/test.sh` | **Standard Quality Pass**. Runs Ruff, Unit Tests, and Integration Smoke. | `./scripts/test.sh` |
| `scripts/linux_test_ui.sh` | Runs Playwright tests in `tests/ui/` with xdist. | `./scripts/linux_test_ui.sh` |
| `scripts/linux_test_all.sh` | Runs the full suite including slow integration tests. | `./scripts/linux_test_all.sh` |

## CI/CD Pipeline

The project uses GitHub Actions (`.github/workflows/ci.yml`) for automated verification:
1. **Unit & Lint**: Runs on every push. Checks types (Pyright), lint (Ruff), and unit coverage.
2. **UI / E2E Tests**: Runs after unit tests pass. 
   - **Environment**: Ubuntu Latest, Chromium.
   - **Timeout**: 15 minutes.
   - **Workers**: Pinned to `-n 4` for deterministic mock server performance.
   - **Artifacts**: Screenshots of failures are uploaded automatically.

## The "Mock-First" Philosophy

To ensure fast execution and hardware independence, all **Unit Tests** must completely mock the following:
- **ML Models**: Mock `ModelRegistry.get_tracker`, `get_face_analyzer`, and `TrackerFactory`.
- **GPU/Torch**: `tests/conftest.py` promotes `torch.cuda` to a `ModuleType` to allow stable patching.
- **Requirement**: Always include `create=True` in `patch("torch.cuda.is_available", ...)` to prevent worker collisions in parallel runs.

## Gradio & Pyright Resiliency

- **Type Hints**: For Gradio event handlers, use `Any` or `dict[str, Any]` for return type hints instead of `gr.update`. Gradio 5+ treats updates as dynamic dictionaries, and `gr.update` often causes Pyright noise.
- **Optional Members**: Components like `AdvancedProgressTracker` should use `Optional[Queue]` and `Optional[AppLogger]` with explicit null-checks to prevent Pyright "attribute not found on None" errors.

## Accessibility (Axe-core)

- **Pattern**: Always provide a `label` to interactive Gradio components, even if `show_label=False`. Gradio renders this as an ARIA label in the accessibility tree.
- **Rule Filtering**: Accessibility tests in `tests/ui/test_accessibility.py` are configured to skip Gradio-internal violations (e.g., `aria-hidden-focus`, `color-contrast`) that cannot be fixed at the application level.

## E2E Testing (Playwright)

- **Locator Protocol**: Never hardcode strings in tests. Always use `ui_locators.py` (`Labels` and `Selectors`).
- **Emoji Sensitivity**: Playwright is sensitive to emojis. If a UI button adds an emoji (e.g., `🚀 Start`), the `Labels` entry must match exactly.
- **Accordion Handling**: Use the `open_accordion(page, label)` helper in `conftest.py`. It uses `elem_id` (#system_logs_accordion) and JS state checks to handle Gradio's complex DOM.
- **Wait Strategy**: Use `page.wait_for_timeout(ms)` for UI animations. Use `expect(locator).to_be_visible()` for backend state changes.

## Visual Regression (`tests/ui/test_visual_regression.py`)

The suite compares current UI states against baseline screenshots using perceptual hashing.
- **Update Baselines**: To refresh baselines (e.g., after changing labels), run:
  `uv run pytest -n 0 --update-baselines tests/ui/test_visual_regression.py`
- **Constraint**: Always use `-n 0` when updating baselines to prevent race conditions during file writes.

## Coverage Requirements

- **Target**: 80% total coverage (`--cov-fail-under=80` enforced in CI).
- **Manual Verification**: `scripts/linux_test_cov.sh`.

---

*Last Updated: 2026-03-28 (Infrastructure Stabilization)*
