# Codebase Structure

**Analysis Date:** 2026-03-21

## Directory Layout

```
subject-frame-extractor/
├── core/               # Business logic, pipelines, and managers
│   ├── operators/      # Quality metric plugins
│   ├── config.py       # Pydantic settings
│   ├── database.py     # SQLite management
│   ├── managers.py     # Resource lifecycles (ModelRegistry, SAM3)
│   ├── models.py       # Pydantic schemas and domain models
│   └── pipelines.py    # Orchestration logic
├── ui/                 # Gradio interface and UI helpers
│   ├── app_ui.py       # Tab-specific layouts
│   └── gallery_utils.py # Image gallery and state utilities
├── SAM3_repo/          # Segment Anything Model 3 submodule (Read-only)
├── tests/              # Multi-tier test suite
│   ├── unit/           # Fast, mocked logic tests
│   ├── integration/    # Real backend/pipeline tests
│   └── ui/             # Browser-based Playwright tests
├── scripts/            # Verification and maintenance scripts
├── docs/               # Technical and user documentation
├── models/             # Local cache for ML model weights
├── logs/               # Application runtime logs
├── app.py              # Main Gradio entry point
└── cli.py              # Headless CLI entry point
```

## Directory Purposes

**core/**:
- Purpose: The "Brain" of the application. Handles all ML processing, data management, and orchestration.
- Contains: `*.py` logic files, sub-packages like `operators/`.
- Key files: `pipelines.py` (orchestration), `managers.py` (resource management), `database.py` (persistence).

**ui/**:
- Purpose: The "Face" of the application. Manages layout, state updates, and visual components.
- Contains: `*.py` Gradio-specific code.
- Key files: `app_ui.py` (layout), `gallery_utils.py` (image grid management).

**tests/**:
- Purpose: Quality assurance across all layers.
- Contains: `pytest` suites organized by test type.
- Subdirectories: `unit/`, `integration/`, `ui/`, `signature/`, `smoke/`.

**scripts/**:
- Purpose: Developer utilities and automated verification tasks.
- Contains: `python` scripts for UX audits, E2E runs, and metadata updates.

## Key File Locations

**Entry Points:**
- `app.py`: Main Gradio web application.
- `cli.py`: Command-line interface for batch processing.

**Configuration:**
- `core/config.py`: Primary application settings and quality weights.
- `pyproject.toml`: Dependency management and tool configuration (Ruff, Pytest).
- `.env_example`: Template for environment-specific secrets.

**Core Logic:**
- `core/pipelines.py`: Workflow definitions (Extraction, Analysis, Export).
- `core/database.py`: SQLite schema and migration logic.

**Testing:**
- `tests/conftest.py`: Shared pytest fixtures and mocks.
- `pytest.ini` (inline in `pyproject.toml`): Test runner configuration.

## Naming Conventions

**Files:**
- `snake_case.py`: All Python modules.
- `test_*.py`: Pytest discovery pattern.
- `*.md`: Documentation and planning artifacts.

**Directories:**
- `snake_case`: Standard Python package and resource directories.
- `__pycache__`: Python bytecode cache (ignored).

**Special Patterns:**
- `[Component]Pipeline`: Classes in `core/pipelines.py`.
- `[Component]Manager`: Classes in `core/managers.py`.

## Where to Add New Code

**New Quality Metric:**
- Implementation: `core/operators/[metric_name].py`.
- Registration: Use `@register_operator` decorator.

**New UI Tab:**
- Layout: Add to `ui/app_ui.py`.
- Callbacks: Define in `ui/app_ui.py` or `CORE` depending on complexity.

**New Core Logic:**
- Module: `core/[domain].py`.
- Integration: Add to appropriate pipeline in `core/pipelines.py`.

## Special Directories

**SAM3_repo/**:
- Purpose: External dependency managed as a submodule.
- Status: **READ-ONLY**. Under no circumstances should this be modified directly.
- Committed: Yes (as submodule reference).

**.planning/**:
- Purpose: GSD (Get Shit Done) context and roadmap documents.
- Status: Internal to development process.
- Committed: Yes.

---

*Structure analysis: 2026-03-21*
*Update when directory structure changes*
