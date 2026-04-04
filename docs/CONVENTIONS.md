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


*Refined conventions: 2026-03-21*
