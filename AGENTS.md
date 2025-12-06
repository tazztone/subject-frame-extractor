# Developer Guidelines & Agent Memory

This file serves as a guide for developers and AI agents working on this repository. It contains architectural insights, coding standards, and recurring patterns/gotchas.

## 🏗️ Architecture

The application is a **Gradio-based desktop application** for video processing.

*   **Monolithic Core**: `app.py` contains the main application logic, UI definition (`AppUI`), and pipeline orchestration.
*   **Modules**:
    *   `config.py`: Configuration management (`Config` class, Pydantic `BaseSettings`).
    *   `logger.py`: Structured logging (`AppLogger`, `JsonFormatter`).
    *   `error_handling.py`: Error handling decorators (`@with_retry`, `@handle_common_errors`) and utilities.
    *   `events.py`: Pydantic models for UI-to-backend communication (`*Event` classes).
    *   `progress.py`: Progress tracking (`AdvancedProgressTracker`).
    *   `database.py`: SQLite storage for frame metadata.
*   **Data Models**: Pydantic `BaseModel` is preferred over `dataclasses`.
*   **State Management**:
    *   **Gradio State**: `gr.State` components store session data (file paths, scene lists).
    *   **Model Registry**: `ModelRegistry` (in `app.py`) is a thread-safe singleton for lazy-loading heavy ML models (SAM3, InsightFace).

## 🧪 Testing & Verification

*   **Backend Tests**: `python -m pytest tests/`
    *   Use `unittest.mock` extensively.
    *   Note: `tests/requirements-test.txt` may be incomplete. Ensure `pydantic-settings`, `numba`, `scikit-learn` are installed.
*   **Frontend Tests**: E2E tests using **Playwright**.
    *   Located in `tests/e2e/`.
    *   Run with `pytest tests/e2e/`.
    *   Requires `playwright install`.
*   **Mocking**:
    *   **Files**: Patch `io.open` or `pathlib.Path.open`. Mock `pathlib.Path.exists` and `is_file`.
    *   **Models**: Mock `sam3`, `timm`, `pycocotools` if needed. Use `autospec=True` for instance methods.
    *   **Gradio**: Tests for event handlers must strictly match return value counts.

## 🚨 Common Pitfalls (Gotchas)

### Gradio
*   **Return Values**: Event handlers must return **exactly** the number of values expected by the `outputs` list. Mismatches cause silent failures or crashes.
*   **Component Order**: The `inputs` list in `gr.on(...)` must match the function signature **exactly**.
*   **State**: `gr.State` initial values must be deep-copyable. Avoid storing complex objects (locks, file handles).
*   **Updates**: When returning `gr.update()`, use dictionary keys (e.g., `res['value']`) in tests, not attributes.

### ML & Resources
*   **Model Loading**: Use `model_registry.get_or_load()`. Do not use `@lru_cache` on methods taking the full `Config` object (it's unhashable).
*   **Submodules**: **DO NOT EDIT** files in `SAM3_repo` or `Grounded-SAM-2`. They are git submodules.
*   **Memory**: Large models (SAM3) are memory-intensive.

### Coding Patterns
*   **Error Handling**: Return structured error payloads (dict with `status_message`, `error_message`) from UI handlers. Do not return empty dicts on failure.
*   **File I/O**: Use `pathlib`.
*   **Logging**: Use `AppLogger`.

## 🛠️ Development Setup

1.  **Submodules**: `git submodule update --init --recursive`
2.  **Environment**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    pip install -r tests/requirements-test.txt
    pip install git+https://github.com/facebookresearch/sam3.git
    ```
