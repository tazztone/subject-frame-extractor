# Architecture

**Analysis Date:** 2026-03-21

## Pattern Overview

**Overall:** Core/UI Separated ML Application (Gradio)

**Key Characteristics:**
- **Modular Pipeline-Driven:** Heavy tasks are encapsulated in pipelines (Extraction, Analysis, Propagation, Export).
- **Lazy Loading Singleton:** `ModelRegistry` manages heavy PyTorch/ONNX models to avoid VRAM OOM.
- **Async-Aware UI:** Gradio event handlers interact with long-running core processes via state and progress updates.
- **Plugin-Based Metrics:** Quality metrics are implemented as "Operators" using auto-discovery.

## Layers

**UI Layer (`ui/`, `app.py`):**
- Purpose: User interaction, layout management, and visualization.
- Contains: Gradio components, `gallery_utils.py` for image display logic, `app_ui.py` for tab layouts.
- Depends on: `core/` (via pipelines and shared state).

**Orchestration Layer (`core/pipelines.py`):**
- Purpose: Coordinating complex multi-step workflows.
- Contains: `ExtractionPipeline`, `PreAnalysisPipeline`, `AnalysisPipeline`, `ExportPipeline`.
- Depends on: `core/managers.py`, `core/models.py`, `core/database.py`.

**Resource Management Layer (`core/managers.py`):**
- Purpose: Handling lifecycle and thread-safety of external assets.
- Contains: `ModelRegistry` (singleton), `SAM3Wrapper` (predictor abstraction), `ThumbnailManager` (LRU cache).
- Depends on: ML libraries (`sam3`, `insightface`), filesystem.

**Data & Logic Layer (`core/`):**
- Purpose: Pure business logic, persistence, and domain models.
- Contains: `database.py` (SQLite), `models.py` (Pydantic schemas), `config.py` (settings), `filtering.py` (logic).
- Depends on: Python standard libraries, `pydantic`.

## Data Flow

**Frame Propagation Flow:**
1. **Trigger:** User provides a box/point prompt on a frame in the UI.
2. **Setup:** `AnalysisPipeline` prepares `SAM3Wrapper` with the selected frames and prompts.
3. **Execution:** `SAM3Wrapper.propagate()` yields masks and metrics frame-by-frame.
4. **Persistence:** `Database.insert_metadata()` buffers and flushes results to `metadata.db`.
5. **Update:** UI receives completion signals and refreshes the gallery using `GalleryManager`.

**State Management:**
- **Ephemeral State:** `gr.State` handles session-specific parameters (current scene, selected object ID).
- **Persistent State:** `metadata.db` (SQLite) stores all per-frame metrics and processing status.
- **Global State:** `ModelRegistry` maintains loaded model instances across the application lifetime.

## Key Abstractions

**Pipeline:**
- Purpose: High-level task execution with progress reporting.
- Pattern: Strategy/Command-like classes in `core/pipelines.py`.

**Operator:**
- Purpose: Discrete image/frame analysis logic (e.g., Blur Detection, Face Detection).
- Pattern: Plugin architecture with decorator-based registration (`@register_operator`).

**Wrapper:**
- Purpose: Isolating complex external APIs (like SAM3) from core business logic.
- Example: `SAM3Wrapper` in `core/managers.py`.

## Entry Points

**Web UI (`app.py`):**
- Location: Root `app.py`.
- Triggers: `python app.py` or `gradio app.py`.
- Responsibilities: Building the UI tree, registering callbacks, launching server.

**CLI (`cli.py`):**
- Location: Root `cli.py`.
- Triggers: `python cli.py [COMMAND]`.
- Responsibilities: Headless execution of pipelines (e.g., batch extraction or export).

## Error Handling

**Strategy:** 
- Centralized logging via `logging`.
- Processing errors are caught within pipelines and recorded in the database (`error` and `error_severity` fields) to allow "Export skipping" of failed frames.

---

*Architecture analysis: 2026-03-21*
*Update when major patterns change*
