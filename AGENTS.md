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

*   **Gradio 5+ Protocol**: Prefer **dictionary returns** for component updates instead of positional tuples to prevent silent app crashes during UI refactors.
*   **Playwright UI Tests**: ALWAYS use `page.wait_for_timeout()` instead of web-first assertions to account for Gradio's reactive delay.
*   **Operator Plugin Pattern**: When adding a metric, you MUST add an entry to `core.filtering._extract_metric_arrays`, otherwise it silently fails filtering.
*   **Thread-Safe Model Access**: Models (like InsightFace) are NOT thread-safe. You MUST use specific locks if running inside `ThreadPoolExecutor`.
*   **Milestone Logging**: Use **Title Case** for major pipeline milestones or the downstream E2E scraper logs will break.
*   **Immutable Submodule**: NEVER edit files in `SAM3_repo`. It is an external dependency.

---
*Rule Validation Status: Evals are enforced via `uv run pytest tests/unit/test_context_adherence.py`*
