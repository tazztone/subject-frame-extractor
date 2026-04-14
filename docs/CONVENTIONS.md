# Coding Conventions

**Analysis Date:** 2026-03-21
**Deep Dive Refinement:** Standardized UI safety and data validation patterns.

## UI Event Patterns

- **Behavior**: Catches exceptions, logs them with `self.logger.error`, and returns a standardized error message to the "Unified Status" and "Unified Log" components.

### 2. Async Orchestration (`_run_pipeline`)
Any UI action that triggers a long-running process (Extraction, Analysis, Session Loading) MUST use the `self.app._run_pipeline` async wrapper.
- **Responsiveness**: This ensures the Gradio UI remains interactive and displays a "Loading" or "Running" status instead of freezing.
- **Standard**: Even non-ML tasks like `Load Session` should be adapted to this generator-based pattern.

### 3. Anti-Loop State Initialization
Reactive state fields in `ApplicationState` (like `all_frames_data`) that trigger `.change()` listeners MUST NOT use "falsy" defaults that indicate "not yet loaded" (e.g., `[]`, `""`, `0`).
- **Pattern**: Initialize with `None`.
- **Condition**: Listeners should check `if state.all_frames_data is None` to trigger first-time loading.
- **Success State**: After a load (even if 0 items found), the field becomes `[]`. Since `[] is not None`, the reactive loop terminates.

## Logging & Progress Tracking

### 1. Mandatory Level Tags
To ensure logs are correctly filtered and visible in the UI `LogViewer`, all log messages MUST contain a bracketed level tag.
- **Allowed Levels**: `[INFO]`, `[ERROR]`, `[DEBUG]`, `[SUCCESS]`, `[WARNING]`.
- **Filtering**: The UI component uses these tags to toggle visibility (e.g., hiding DEBUG logs by default).
- **Example**: `self.logger.info("[SUCCESS] Session loaded correctly")`.

### 2. Queue-First Principle
Directly yielding to `unified_log` is an anti-pattern. Always prefer the background queue via the logger to ensure the UI refresh timer captures the update atomically.

## Metadata & Validation (Pydantic)

- **Strict Validation**: Use `model_config = ConfigDict(validate_assignment=True)` in base events.
- **Cross-Field Checks**: Use `@model_validator(mode='after')` for logic that depends on multiple fields (e.g., ensuring a face reference image exists if `enable_face_filter` is True).
- **Auto-Stripping**: String fields should automatically strip whitespace.

## Gradio Update Protocol (Gradio 5+)

When a function returns updates to multiple components, it MUST return a dictionary mapping the component object (or its key) to its new value or `gr.update(...)`.
- **Consistency**: Always prefer updating the `value` through `gr.update(value=...)` for clarity.
- **Visibility**: Toggle visibility via `gr.update(visible=True/False)`.
- **State Persistence**: In Gradio 5, state objects (like `ApplicationState`) must be explicitly yielded or returned in every event handler that modifies them. If a handler triggers a sequence of pipeline steps, each success callback must merge and re-yield the state to ensure it persists across subsequent UI interactions.

## Byte-Level Emoji Parity (ZWJ)

To avoid "Choice not in list" errors in Gradio `Radio` or `Dropdown` components:
- **Explicit Unicode**: Always use explicit unicode escape sequences (e.g., `\u200d` for Zero Width Joiners) for complex emojis in constants and comparison logic.
- **Synchronization**: Ensure the exact same string literal (including invisible characters) is used across `core/config.py`, `ui/app_ui.py`, and any backend filtering logic.
- **Example**: `"🧑\u200d🤝\u200d🧑 Find Prominent Subject"` should be the source of truth for all references.

## Concurrency & Threading

- **Background Tasks**: Long-running pipelines must run in a background thread to keep the UI responsive.
- **Yielding Updates**: Generators should `yield` update dictionaries frequently. The UI logic uses a `Queue` to bridge between the background thread and the Gradio event loop.
- **Singleton Locks**: Always acquire the appropriate lock from `ModelRegistry` or `ThumbnailManager` before modifying shared model states.

## SAM3 Interaction Conventions

### 1. Persistent Tracking Protocol
To maintain object identity across manual prompts, never use the high-level `boxes_xywh` argument in the predictor.
- **Pattern**: Always map BBoxes to the `points` argument with `labels=[2, 3]` (Top-Left, Bottom-Right).
- **Coordinate Mapping**: [[x, y], [x+w, y+h]]. Ensure all values are clipped to [0.0, 1.0].

### 2. Immediate Feedback Initialization
For interactive subject selection, the model must provide immediate feedback on the seed frame.
- **hotstart_delay**: Must be set to `0` in `SAM3Wrapper` to bypass the default 15-frame stabilization delay.
- **masklet_confirmation**: Set to `False` to prevent the tracker from "doubting" and suppressing initial segments.
- **"Subject" / "Anchor"**: Use these terms instead of "Person" to refer to the object being tracked, reflecting the multi-class (80 COCO classes) capabilities.
- **"Extraction"**: The process of saving frames to disk.

## Typing & Documentation

- **Type Hints**: Mandatory for all function signatures and class members.
- **Docstrings**: Google-style docstrings used throughout.
- **Forward References**: Use `"ClassName"` or `from __future__ import annotations` to handle circular dependencies in type hints.

## Development & Testing Workflows

### 1. Mock Integrity & API Signatures
The `tests/mock_app.py` stubs MUST be kept in perfect synchronization with the real `PipelineHandler` and `core/pipelines.py` signatures.
- **Verification**: Always run `uv run pytest tests/unit/test_signatures.py` after changing any public pipeline API.
- **Failure**: Failure to sync mocks will cause CI regressions in unit tests that rely on the mock app.

### 2. Bug Fix Workflow
1. **Reproduce**: Create a minimal test case in `tests/unit/` or `tests/integration/`.
2. **Trace**: Use `logger.debug()` or `logger.info()` to inspect state during pipeline execution.
3. **Fix**: Implement the fix in the appropriate `core/` or `ui/` module.
4. **Verify**: Run the full suite with `uv run pytest tests/`.
5. **Clean**: Remove any temporary debug artifacts or print statements.

---

## Agent & CI Knowledge Base (Migrated from AGENTS.md)

### Validated Architectural Rules
*   **Architectural Isolation**: NEVER import from `ui/` inside `core/`.
*   **UI Safety Contract**: Event methods in `app_ui.py` MUST be wrapped in `@AppUI.safe_ui_callback`.
*   **Unhashable Config**: NEVER use `@lru_cache` on functions taking `Config`.
*   **Mock Integrity**: Sync `mock_app.py` stubs directly with real pipeline signatures.

### Hard-Learned Behavioral Patterns
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
*   **Orchestrator Strip-Done Protocol**: When chaining generators (e.g., in `execute_analysis_orchestrator`), intermediate stages MUST have their `done: True` flag stripped before yielding. Only the final stage or the orchestrator itself should yield `done: True` to prevent premature consumer termination.

### Test Infrastructure Rules (Mock Safety & Parity)

*   **Deterministic Mock Injection**: All mock module patching must be handled centrally through `tests/helpers/sys_mock_modules.py`. Manually overriding `sys.modules` in local test files or `mock_app.py` is strictly prohibited as it leads to non-deterministic state leakage across background threads.
*   **Signature Parity Contract**: The stubs in `tests/mock_app.py` MUST mirror the signatures of `core/pipelines.py` and `PipelineHandler` exactly (verified by `test_signatures.py`).
*   **SAM3 is globally mocked in unit mode**: `sam3` and `sam3.model_builder` are included in `tests/helpers/sys_mock_modules.py`. Unit tests do not need local patches. Integration tests requiring the real SAM3 install must set `PYTEST_INTEGRATION_MODE=true`.
*   **`PYTEST_INTEGRATION_MODE` propagation**: `conftest.py` disables global mocks when this env var is `true`. Always use `export` to ensure inheritance by parallel workers.
*   **`gpu_e2e` tests are serial-only**: Parallel loading of multiple SAM models (SAM2 + SAM3) will cause VRAM OOM.

*Refined conventions: 2026-03-21*
