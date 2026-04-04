# 🧑‍💻 Agent Logic & Constraints (High-Signal Context)

This document contains **strict behavioral constraints** for agents. Every rule here fixes a known hallucination and is evaluated by our test suite (The Evals).

---

## 1. Validated Architectural Rules (Must Follow)

These rules map directly to automated CI checks.

*   **Architectural Isolation**: NEVER import from `ui/` inside `core/`. (Eval: `tests/unit/test_context_adherence.py`)
*   **UI Safety Contract**: Event methods in `app_ui.py` MUST be wrapped in `@AppUI.safe_ui_callback`. (Eval: `tests/unit/test_context_adherence.py`)
*   **Unhashable Config**: NEVER use `@lru_cache` on functions taking `Config`. (Eval: `tests/unit/test_context_adherence.py`)
*   **Mock Integrity**: Sync `mock_app.py` stubs directly with real pipeline signatures. (Eval: `tests/unit/test_signatures.py`)

## 2. Hard-Learned Behavioral Patterns (Source of Truth)

These are complex, opinionated patterns that correct behaviors the agent naturally gets wrong.

*   **Gradio 5+ Protocol**: Prefer **dictionary returns** for component updates.
*   **Accordion Visibility**: Components inside collapsed `gr.Accordion` are INVISIBLE to Playwright. ALWAYS use `open_accordion(page, Label)` before asserting values inside.
*   **Slider Selectors**: Gradio Sliders contain multiple inputs. Use `#elem_id input[data-testid='number-input']` for numeric values.
*   **Playwright UI Tests**: ALWAYS use `page.wait_for_timeout()` instead of web-first assertions to account for Gradio's reactive delay.
*   **Operator Plugin Pattern**: When adding a metric, you MUST add an entry to `core.filtering._extract_metric_arrays`, otherwise it silently fails filtering.
*   **Thread-Safe Model Access**: Models (like InsightFace) are NOT thread-safe. You MUST use specific locks if running inside `ThreadPoolExecutor`.
*   **Milestone Logging**: Use **Title Case** for major pipeline milestones or the downstream E2E scraper logs will break.
*   **Immutable Submodule**: NEVER edit files in `SAM3_repo`. It is an external dependency.
*   **SAM2.1 Default Baseline**: **SAM2.1 Hiera Tiny** is the project's default tracker and baseline for all integration tests. SAM3 is considered an experimental alternative.
*   **Visual Baseline Updates**: When UI labels or layouts change, visual regression baselines MUST be updated using `uv run pytest -n 0 --update-baselines tests/ui/test_visual_regression.py`. Parallel execution (`-n > 0`) is forbidden during updates.
*   **Feature Status**: `sam3` is an experimental tracker. The default is `sam2`. Do not switch the default without explicit instruction.
*   **Reactive State Safety**: NEVER use falsy defaults (`[]`, `""`, `0`) for `ApplicationState` fields that trigger Gradio `.change()` listeners. Use `Optional` with `None` and check `is None` to differentiate between "not loaded" and "empty".
*   **Orchestrator Strip-Done Protocol**: When chaining generators (e.g., in `execute_analysis_orchestrator`), intermediate stages MUST have their `done: True` flag stripped before yielding. Only the final stage or the orchestrator itself should yield `done: True` to prevent premature consumer termination.

---

## 3. Canonical Test Commands (Use These Exactly)

> **CRITICAL**: Always use the commands below. Never invent pytest invocations.
> Running tests without the correct flags causes deselection, mock leakage, or GPU OOM crashes.

### Run Everything (Recommended)
```bash
bash scripts/linux_test_all.sh
```
This is the only correct way to run the full suite. It exports `PYTEST_INTEGRATION_MODE=true`,
runs stages in the correct order (unit → integration → UI → regression), and uses `-n 1`
for GPU tests to prevent VRAM exhaustion.

### Unit Tests Only (fast, no GPU)
```bash
bash scripts/linux_test_unit.sh
```

### Integration / GPU E2E Tests Only
```bash
export PYTEST_INTEGRATION_MODE=true
uv run --no-sync pytest tests/integration/ -m "integration or gpu_e2e" -n 1 -v --no-cov
```
**Rules that MUST be followed for integration tests:**
- `export PYTEST_INTEGRATION_MODE=true` MUST be set before running. An inline prefix (`VAR=value pytest ...`) is NOT sufficient — xdist workers are spawned as subprocesses and do not inherit inline-prefixed variables.
- `-n 1` (or no `-n`) MUST be used for `gpu_e2e` tests. Never use `-n auto` with GPU tests. SAM3 and SAM2 each require 3–4 GB VRAM; parallel workers will OOM and freeze the desktop.
- Do NOT add `--ignore` flags or `-k` filters that exclude `sam3` or `sam2` markers without explicit instruction.

### Run a Single Integration Test
```bash
export PYTEST_INTEGRATION_MODE=true
uv run --no-sync pytest tests/integration/test_gpu_e2e.py::TestMaskPropagatorE2E::test_mask_propagator_propagate -m "gpu_e2e" -n 1 -v --no-cov
```
Note the `-m "gpu_e2e"` flag — without it the test is **deselected** (shows `0 selected`) because `pyproject.toml` requires the marker for GPU tests.

### UI / Playwright Tests
```bash
bash scripts/linux_test_ui.sh
```

### Regression Tests
```bash
uv run --no-sync pytest tests/regression/ -v
```

---

## 4. Test Infrastructure Rules (Prevent Mock Leakage)

*   **SAM3 is NOT in global mocks**: `tests/conftest.py` intentionally excludes all `sam3.*` entries from `modules_to_mock`. Unit tests that need a mock SAM3 MUST create a local mock in their own file using `patch("sam3.model_builder.build_sam3_predictor")`.
*   **`PYTEST_INTEGRATION_MODE` propagation**: `conftest.py` disables global mocks when this env var is `true`. xdist workers inherit env vars only if the parent shell used `export`, not an inline prefix. Always use `export`.
*   **`gpu_e2e` tests are serial-only**: The `module_model_registry` fixture uses `scope="module"` to share model weights within a worker, but provides no cross-worker isolation. Multiple workers loading SAM3 + SAM2 simultaneously causes GPU OOM.
*   **Blank model dirs are forbidden in E2E fixtures**: The `module_model_registry` fixture in `test_gpu_e2e.py` must point to the project's real `models/` directory. Never use `tempfile.TemporaryDirectory()` for the models path in GPU tests — it triggers a 3.3 GB HuggingFace download race.

---
*Rule Validation Status: Evals are enforced via `uv run pytest tests/unit/test_context_adherence.py`*
