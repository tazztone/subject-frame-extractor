# Project Structure

**Analysis Date:** 2026-03-21
**Deep Dive Refinement:** Detailed internal data structures and file roles.

## Directory Layout

```text
.
├── core/                   # Business logic and heavy lifting
│   ├── config.py           # Pydantic BaseSettings (Global Config)
│   ├── pipelines.py        # Pipeline orchestration (Extraction, Analysis)
│   ├── managers.py         # Resource/Model lifecycle management
│   ├── events.py           # Pydantic UIEvent models
│   ├── models.py           # Pydantic data models (Frame, Scene)
│   ├── enums.py            # [NEW] Status and Direction enums
│   ├── filtering.py        # Vectorized metric filtering and deduplication
│   ├── export.py           # FFmpeg extraction, cropping, and XMP export
│   ├── operators/          # Quality metric plugins (Face, Sharpness, etc.)
│   ├── scene_utils/        # Low-level ML wrappers (SubjectMasker, SAM3Wrapper)
│   ├── database.py         # SQLite metadata persistence
│   ├── utils.py            # Path, image, and hash utilities
│   ├── shared.py           # Cross-cutting shared utilities
│   ├── error_handling.py   # Retry/fallback decorators
│   ├── progress.py         # Progress tracking infrastructure
│   ├── fingerprint.py      # Run-skipping logic
│   ├── system_health.py    # Diagnostic runner
│   ├── application_state.py# Unified UI state reducer
│   ├── batch_manager.py    # Multi-video batch queue management
│   └── sam3_patches.py     # SAM3 submodule runtime patches
├── ui/                     # Modular Gradio interface
│   ├── app_ui.py           # Main layout and tab composition
│   ├── gallery_utils.py    # Gallery logic and filtering handlers
│   ├── handlers/           # Tab-specific event handlers (SceneHandler)
│   ├── tabs/               # Modular Tab builders (ExtractionTabBuilder, etc.)
│   └── components/         # Reusable Gradio components
├── tests/                  # Multi-tier test suite
│   ├── unit/               # Fast, mocked logic tests
│   ├── integration/        # Hardware-dependent pipeline tests
│   └── ui/                 # Playwright E2E browser tests
├── scripts/                # Utility and verification scripts
│   └── verification/       # E2E health check scripts
├── SAM3_repo/              # READ-ONLY Git Submodule (SAM3 official) 🔴
└── .planning/              # GSD Project Management (Context & Roadmaps)
```

## Key File Roles

### Orchestration
- **`app.py`**: The "Glue". Initializes the `ModelRegistry`, `ThumbnailManager`, and `Database`. Sets up the Gradio event loop.
- **`core/pipelines.py`**: The "Brain". Orchestrates FFmpeg, SAM3, and Operator loops. It is the primary consumer of `AnalysisParameters`.

### Resources
- **`core/managers.py`**: The "Gatekeeper". Contains `ModelRegistry` (prevents VRAM leaks) and `SAM3Wrapper` (adapts the submodule API).
- **`core/config.py`**: The "Manifest". Defines all thresholds, weights, and model URLs.

### Data & State
- **`metadata.db`**: Persistent SQLite storage for frame metrics. Allows the Gallery to filter thousands of frames instantly.
- **`run_config.json`**: Generated in each output folder; stores the exact parameters used for that run (essential for resumability).
- **`frame_map.json`**: The translation layer between filesystem image names (`frame_000001.png`) and original video indices.

## Naming Conventions (Refined)

- **Pipelines**: Class names should end in `Pipeline` (e.g., `ExtractionPipeline`).
- **Managers**: Class names should end in `Manager` (e.g., `ThumbnailManager`).
- **Mocks**: In tests, use the `mock_` prefix (e.g., `mock_config`, `mock_logger`).
- **Events**: Event models in `core/events.py` should follow the `[Action]Event` pattern.

## Rule of Thirds (Code Placement)

1.  **If it touches Gradio components**: Put it in `ui/`.
2.  **If it touches ML models or high-volume data**: Put it in `core/`.
3.  **If it's a reusable math or path utility**: Put it in `core/utils.py`.

---

*Refined structure: 2026-03-21*
