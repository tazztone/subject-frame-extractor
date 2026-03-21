# Coding Conventions

**Analysis Date:** 2026-03-21

## Naming Patterns

**Files:**
- `snake_case.py` for all Python modules (e.g., `pipelines.py`, `gallery_utils.py`).
- `test_*.py` for test files, located in the `tests/` hierarchy.
- `AGENTS.md`, `README.md` (UPPERCASE) for root-level documentation.

**Classes:**
- `PascalCase` for all classes (e.g., `ExtractionPipeline`, `SAM3Wrapper`, `AppUI`).
- `[Component]Pipeline` for orchestration logic.
- `[Component]Manager` for resource lifecycle management.

**Functions:**
- `snake_case` for all functions and methods.
- `_prefix` for private methods and internal helper functions (e.g., `_build_header`, `_process_ffmpeg_stream`).
- `on_event_name` for UI event handlers in `ui/gallery_utils.py`.

**Variables:**
- `snake_case` for local variables and instance attributes.
- `UPPER_SNAKE_CASE` for class-level constants and global configuration defaults.

**Types:**
- Use Python Type Hints for all function signatures and complex variables.
- Pydantic models for data schemas (e.g., `Frame`, `Scene`).

## Code Style

**Formatting:**
- **Tool:** `ruff` is the primary linter and formatter.
- **Line Length:** 120 characters (configured in `pyproject.toml`).
- **Quotes:** Double quotes preferred for strings (Ruff default).
- **Indentation:** 4 spaces.

**Linting:**
- **Tool:** `ruff` with `E`, `F`, `W`, `I` rules enabled.
- **Strictness:** Type checking is encouraged but not enforced via MyPy yet (though type hints are prevalent).

## Import Organization

**Order:**
1. `from __future__ import annotations` (if needed)
2. Standard library imports (e.g., `json`, `os`)
3. Third-party library imports (e.g., `gradio`, `numpy`, `torch`)
4. Local application imports (e.g., `core.models`, `ui.app_ui`)

**Grouping:**
- Blank line between standard, third-party, and local blocks.
- Alphabetical sorting within groups (handled by Ruff `I` rules).

## Error Handling

**Strategy:**
- **Pipelines:** Use the `ErrorHandler` wrapper for automated retries on transient failures.
- **UI:** Use the `@AppUI.safe_ui_callback` decorator to catch and log exceptions, returning user-friendly error messages to the Gradio log.
- **Validation:** Use Pydantic `model_validator` and `field_validator` for configuration and event data.

**Logging:**
- Use the centralized `AppLogger` for structured logging.
- Log levels: `DEBUG`, `INFO`, `SUCCESS`, `WARNING`, `ERROR`, `CRITICAL`.
- Avoid naked `print()` statements; use `logger.info()` instead.

## Function Design

**Size:**
- Aim for modularity. Highly complex logic (like FFmpeg filter construction) is extracted into specialized helper functions.
- Entry point methods (like `_run_impl`) handle orchestration, while helpers handle mechanical steps.

**Parameters:**
- Use Pydantic objects (`AnalysisParameters`, `ExtractionEvent`) for functions taking more than 4-5 arguments.
- Prefer explicit type hints for all parameters.

---

*Convention analysis: 2026-03-21*
*Update when patterns change*
