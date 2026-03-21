# Coding Conventions

**Analysis Date:** 2026-03-21
**Deep Dive Refinement:** Standardized UI safety and data validation patterns.

## UI Event Patterns

### 1. The "Safe Callback" Decorator
All UI logic in `AppUI` should be wrapped with `@AppUI.safe_ui_callback("Context Name")`.
- **Purpose**: Prevents the entire Gradio app from crashing on unhandled exceptions.
- **Behavior**: Catches exceptions, logs them with `self.logger.error`, and returns a standardized error message to the "Unified Status" and "Unified Log" components.

### 2. The event-Pipeline Pattern
Never pass raw Gradio arguments directly to core logic.
- **UI**: Collects `*args`.
- **Mapping**: Create a dictionary and validate it into a Pydantic `UIEvent` model (e.g., `ExtractionEvent`).
- **Pipeline**: Processes the validated event object.

## Metadata & Validation (Pydantic)

- **Strict Validation**: Use `model_config = ConfigDict(validate_assignment=True)` in base events.
- **Cross-Field Checks**: Use `@model_validator(mode='after')` for logic that depends on multiple fields (e.g., ensuring a face reference image exists if `enable_face_filter` is True).
- **Auto-Stripping**: String fields should automatically strip whitespace.

## Gradio Update Protocol (Gradio 5+)

When a function returns updates to multiple components, it MUST return a dictionary mapping the component object (or its key) to its new value or `gr.update(...)`.
- **Consistency**: Always prefer updating the `value` through `gr.update(value=...)` for clarity.
- **Visibility**: Toggle visibility via `gr.update(visible=True/False)`.

## Concurrency & Threading

- **Background Tasks**: Long-running pipelines must run in a background thread to keep the UI responsive.
- **Yielding Updates**: Generators should `yield` update dictionaries frequently. The UI logic uses a `Queue` to bridge between the background thread and the Gradio event loop.
- **Singleton Locks**: Always acquire the appropriate lock from `ModelRegistry` or `ThumbnailManager` before modifying shared model states.

## Typing & Documentation

- **Type Hints**: Mandatory for all function signatures and class members.
- **Docstrings**: Google-style docstrings used throughout.
- **Forward References**: Use `"ClassName"` or `from __future__ import annotations` to handle circular dependencies in type hints.

---

*Refined conventions: 2026-03-21*
