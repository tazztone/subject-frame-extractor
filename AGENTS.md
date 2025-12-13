# Developer Guidelines & Agent Memory

This file is the **primary source of truth** for developers and AI agents working on this repository. It consolidates architectural knowledge, coding standards, testing protocols, and lessons learned ("memories") from previous development cycles.

**⚠️ CRITICAL INSTRUCTION**: Before starting any task, read this file to avoid repeating past mistakes.

## 🏗️ Architecture & Design Patterns

The application is a **Gradio-based desktop application** for video processing, currently transitioning from a monolithic script to a modular architecture.

### Core Structure
*   **Monolithic Entry Point**: `app.py` contains the main `AppUI` class, `AnalysisPipeline` orchestration, and legacy logic.
*   **Modular Components**:
    *   **Core Logic**: The `core/` directory contains business logic decoupled from the UI (e.g., `core/filtering.py`, `core/batch_manager.py`).
    *   **Configuration**: `core/config.py` uses Pydantic `BaseSettings`. It is a flat configuration model (preferred over nested).
    *   **Logging**: `core/logger.py` implements structured JSONL logging via `AppLogger`.
    *   **Error Handling**: `core/error_handling.py` provides decorators like `@with_retry` and `@handle_common_errors`.
    *   **Data Models**: `core/events.py` defines Pydantic `BaseModel` classes for UI-Backend communication (e.g., `ExtractionEvent`).
    *   **Persistence**: `core/database.py` manages SQLite storage for frame metadata.
    *   **Progress**: `core/progress.py` handles multi-stage progress tracking.

### Key Patterns
*   **Model Management**: heavy ML models (SAM3, InsightFace) are managed by the **`ModelRegistry`** singleton in `core/managers.py` (instantiated in `app.py`).
    *   *Rule*: Never use `@lru_cache` for model loaders that take the full `Config` object (it is unhashable). Use `model_registry.get_or_load()`.
    *   *Rule*: Models should be lazy-loaded.
*   **State Management**:
    *   **Gradio State**: Use `gr.State` for session-specific data (paths, scene lists).
    *   **Global State**: Avoid global variables. Use the `Config` singleton or `ModelRegistry`.
*   **Concurrency**:
    *   **Batch Processing**: `core/batch_manager.py` handles queueing and parallel execution.
    *   **Thread Safety**: `Database` uses a buffering mechanism with explicit `flush()`. `ModelRegistry` is thread-safe.
*   **Frontend/Backend Decoupling**:
    *   The UI (`AppUI`) should only handle presentation and event triggering.
    *   Business logic resides in pipeline classes (`ExtractionPipeline`, `AnalysisPipeline`) and `core/` modules.
    *   Communication is done via typed events (`core/events.py`) and standardized return payloads.

## 📝 Coding Standards

*   **Data Classes**: Prefer **Pydantic `BaseModel`** over Python `dataclasses` for better validation and serialization.
*   **Path Handling**: Use **`pathlib.Path`** exclusively. Avoid `os.path`.
*   **Type Hinting**: Fully type-hint all new functions and classes.
*   **Docstrings**: Use **Google Style** Python docstrings.
*   **Refactoring**:
    *   **Simplify**: Remove unused code and wrappers. Flatten nested structures.
    *   **Standardize**: Replace custom implementations with standard library features where possible.
    *   **Defensive Coding**: Use `getattr` for optional attributes. Validate inputs early.

## 🧪 Testing & Verification

### Test Suite Structure
*   **Backend Tests**: `tests/` (e.g., `test.py`, `test_utils.py`). Run with:
    ```bash
    python -m pytest tests/
    ```
*   **Frontend (E2E) Tests**: `tests/e2e/` using **Playwright**. Run with:
    ```bash
    pytest tests/e2e/
    ```

### Mocking Guidelines (Crucial)
*   **File I/O**:
    *   Patch `io.open` (or `app.io.open` if imported) instead of `builtins.open`.
    *   Mock `pathlib.Path.exists`, `is_file`, `stat` (for size), and `os.access` (for permissions).
*   **External Libraries**:
    *   **SAM3 / ML**: Mock deep dependencies (`timm`, `pycocotools`, `torchvision`) to avoid import errors.
    *   **Submodules**: When mocking packages like `sam3.model_builder`, mock the parent package first and set `__path__` and `__spec__`.
*   **Instance Methods**: Use `autospec=True` when patching class methods to ensure `self` is handled correctly.
*   **Comparisons**: `MagicMock` objects are not comparable. Implement `__lt__` on mocks if they are sorted in the code.

### Pre-Commit Checklist
1.  **Run Backend Tests**: Ensure all logic changes are verified.
2.  **Skip Frontend Verification**: Unless you touched the UI layout.
3.  **Review**: Self-review for "Common Pitfalls" below.

## 🚨 Common Pitfalls (The "Do Not Do" List)

### Gradio Specifics
*   **Return Mismatch**: Event handlers MUST return exactly the number of values expected by `outputs`. Mismatches cause silent crashes.
*   **Input Order**: The `inputs` list in `gr.on()` MUST match the function arguments exactly.
*   **State Initialization**: Do NOT initialize `gr.State` with non-picklable objects (locks, file handles).
*   **Component Visibility**: `gr.Gallery` does not accept `None`. `Radio` components crash if set to a value not in `choices`.
*   **CSS**: The `css` argument in `gr.Blocks` is deprecated in Gradio 6.x.

### ML & Performance
*   **Memory Leaks**: SAM3 is memory-intensive. Ensure `cleanup_models()` is called.
*   **Thread Safety**: MediaPipe objects are not thread-safe; create one instance per thread.
*   **Vectors**: Use NumPy for heavy lifting (deduplication, filtering). Avoid Python loops for pixel operations.

### Git & Environment
*   **Submodules**: **NEVER** edit files in `SAM3_repo` or `Grounded-SAM-2`. Treat them as read-only libraries.
*   **Dependencies**: Install `requirements.txt` AND `tests/requirements-test.txt`.
*   **Validation**: Verify model downloads with **SHA256 checksums**, not just file size.

## 🧠 Workflows

*   **Deep Planning**: Before coding, analyze the request and codebase. Ask clarifying questions. Create a detailed plan using `set_plan`.
*   **Bug Fixing**:
    1.  Reproduce with a test case.
    2.  Fix the root cause (don't just patch the symptom).
    3.  Verify with the test.
    4.  Ensure no regressions.
*   **New Features**:
    1.  Update `Config` and `Events`.
    2.  Add UI components (in `AppUI`).
    3.  Implement backend logic.
    4.  Wire them up.

## 📂 File System
*   **`config_dump.json`**: Where configuration is saved. Do not use YAML.
*   **`structured_log.jsonl`**: Machine-readable logs.
*   **`metadata.db`**: SQLite database for frame data.
