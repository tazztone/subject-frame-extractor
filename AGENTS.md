# 🧑‍💻 Developer Guide & Technical Reference

This document provides advanced technical details, architectural overview, and development workflows for the Subject Frame Extractor.

## 1. Core Architecture & Data Flow

The application enforces a strict **Core/UI separation** to ensure maintainability and testability.

### Data Path
`UI (Gradio)` → `Event Object (Pydantic)` → `Pipeline (Core)` → `Database/Files`

### Directory Structure & Responsibilities
- **`app.py`**: The main entry point. Assembles the Gradio interface and manages layout.
- **`core/`**: Business logic and heavy lifting.
    - `config.py`: Centralized configuration via Pydantic `BaseSettings`.
    - `pipelines.py`: Orchestrates Extraction, Pre-Analysis, Propagation, and Export.
    - `managers.py`: Resource lifecycle management (`ModelRegistry`, `ThumbnailManager`).
    - `models.py`: Pydantic schemas, frame objects, and metric calculation logic.
    - `database.py`: SQLite interface for frame metadata.
- **`ui/`**: Modular UI components.
    - `app_ui.py`: Tab-specific layouts.
    - `gallery_utils.py`: UI logic and gallery generation.
- **`SAM3_repo/`**: **Read-only** official SAM3 submodule.
- **`tests/`**: Comprehensive test suite (Unit, Integration, E2E).

### Component Relationships
- **ExtractionPipeline**: FFmpeg-based frame extraction with complex filter chain construction.
- **PreAnalysisPipeline**: Scene-by-scene "Best Frame" selection and seeding.
- **AnalysisPipeline**: Manages SAM3 propagation and InsightFace analysis.
- **ExportPipeline**: Handles filtering logic, deduplication, and final rendering.

### State Management
- **Session State**: `gr.State` handles ephemeral UI data (scene lists, current paths).
- **Global State**: `ModelRegistry` acts as a thread-safe singleton for lazy-loading heavy ML models.
- **Persistence**: `metadata.db` (SQLite) stores all frame metrics; `run_config.json` saves session parameters; `run_fingerprint.json` enables run-skipping logic.

## 2. Critical Rules (The "Agent Memory")

### 🔴 CRITICAL (Must Follow)
- **Immutable Submodule**: **NEVER** edit files in `SAM3_repo`. Treat it as an external package.
- **Architectural Isolation**: **NEVER** import from `ui/` inside `core/`. Use callbacks or `core/shared.py` for cross-cutting needs.
- **Gradio Protocol**: **ALWAYS** ensure the number of return values in event handlers matches the `outputs` list. Mismatches cause silent app crashes.
- **Unhashable Config**: **NEVER** use `@lru_cache` on functions taking the `Config` object. Use `model_registry.get_or_load` instead.
- **Path Hygiene**: **ALWAYS** use `pathlib.Path`. Do not use string concatenation or `os.path`.
- **Mock Integrity**: **NEVER** update a core pipeline signature without updating its `mock_app.py` equivalent. Run `test_signatures.py` to verify (Zero Drift Policy).
- **Environmental Isolation**: **ALWAYS** use `types.ModuleType` for injecting heavy mocks (Torch/SAM3) into `sys.modules`. Never inject raw `MagicMock` objects directly, as it breaks type-checking in downstream libraries.
- **UI Wait Strategy**: **ALWAYS** use `page.wait_for_timeout()` in Playwright tests. **NEVER** use `time.sleep()`, which blocks the Python event loop and causes flakiness.
- **Milestone Reporting**: **ALWAYS** use **Title Case** for major pipeline milestones (e.g., "Extraction Complete."). E2E status scrapers are case-sensitive.

### 🟡 WARNING (Potential Pitfalls)
- **Mask Validation**: Always verify masks exist on disk before triggering export or batch processing.
- **Thread Safety**: MediaPipe and InsightFace are NOT thread-safe. Use thread-local storage or unique instances per thread.
- **State Serialization**: Avoid storing locks, file handles, or large complex objects in `gr.State`.
- **Patching Targets**: Always patch the module where the function/class is *imported*, not where it is defined.

### 🟢 BEST PRACTICES
- **Logic Migration**: Continuously move complex logic from `app.py` into specialized `core/` modules.
- **Strong Typing**: Use Pydantic models (`core/events.py`) instead of untyped dictionaries for event data.
- **Test Fixtures**: Add shared mocks to `tests/conftest.py` for reuse across the suite.

## 3. Development Workflows

### Bug Fix Workflow
1. **Reproduce**: Create a minimal test case in `tests/`.
2. **Trace**: Use `logger.debug()` to inspect state during pipeline execution.
3. **Fix**: Implement the fix in the appropriate `core/` or `ui/` module.
4. **Verify**: Run the full suite with `uv run pytest tests/`.
5. **Clean**: Remove any temporary debug artifacts.

### Adding a New Quality Metric
1. **Config**: Add default thresholds and toggle to `Config` in `core/config.py`.
2. **Models**: Update `calculate_quality_metrics` in `core/models.py`.
3. **Filtering**: Update `_extract_metric_arrays()` in `core/filtering.py`.
4. **UI**: Add a slider/control in `ui/app_ui.py` within the filtering tab.

## 4. Testing & Validation Guide

This project uses a tiered testing strategy to ensure stability across unit logic, UI interactions, and heavy ML pipelines.

### Structure

| Layer | Directory | Description | Runner |
|-------|-----------|-------------|--------|
| **Unit** | `tests/unit/` | Fast tests for core logic. Function-level, heavily mocked. | `pytest` |
| **Integration** | `tests/integration/` | Real backend pipeline tests. Uses real PyTorch models but executes full logic. | `pytest` |
| **UI (E2E)** | `tests/ui/` | Browser automation using Playwright. Mocks the backend to test UI flows. | `pytest + playwright` |
| **Verification**| `scripts/verification/` | Manual scripts to run against a live server for ad-hoc checks. | `python` |

### Running Tests

All tests should be run using `uv` to ensure the correct environment.

1. **Unit Tests (Fast)**: `uv run pytest tests/unit/`
2. **Integration Tests (Slow)**: `uv run pytest tests/integration/`
3. **UI / E2E Tests (Browser)**: 
   - Setup: `uv run playwright install chromium`
   - Run: `uv run pytest tests/ui/`
4. **Coverage Report**: `uv run pytest --cov=core --cov=ui tests/`

### Writing Tests

- **New Helper Function?** -> `tests/unit/test_utils.py`
- **New UI Button?** -> `tests/ui/test_ui_interactions.py` (Mock the backend action!)
- **New ML Pipeline Step?** -> `tests/integration/test_real_workflow.py`

### Mocks & Fixtures
Common fixtures are defined in `tests/conftest.py` (`mock_config`, `mock_ui_state`, `mock_torch`).

---

## 5. SAM3 API Guide (Developer Reference)

The `SAM3Wrapper` in `core/managers.py` encapsulates the official `Sam3VideoPredictor` API. For the primary reference implementation this project follows, see:
`SAM3_repo/examples/sam3_video_predictor_example.ipynb`

### Core API Usage
```python
from core.managers import SAM3Wrapper
wrapper = SAM3Wrapper(device="cuda")

# 1. Initialize session
wrapper.init_video("/path/to/frames")

# 2. Add Prompt (Wrapper handles absolute -> relative normalization)
mask = wrapper.add_bbox_prompt(frame_idx=0, obj_id=1, bbox_xywh=[x,y,w,h], img_size=(W,H))

# 3. Propagate (Generator yielding frame_idx, obj_id, mask_array)
# direction can be "forward", "backward", or "both"
for frame_idx, obj_id, mask in wrapper.propagate(start_idx=0, direction="both"):
    ...
```

### Complete API Reference
| Method | Description |
|--------|-------------|
| `init_video(path)` | Initialize session with video or frame folder. |
| `add_bbox_prompt(...)` | Add a bounding box. Auto-converts pixels to relative 0-1. |
| `detect_objects(...)` | Open-vocabulary text detection on single frames. |
| `add_text_prompt(...)` | Add text prompts for video tracking (new API). |
| `add_point_prompt(...)` | Refine masks with positive/negative points. |
| `propagate(...)` | **Generator** yielding propagation results. Supports `direction="both"`. |
| `close_session()` | Release GPU resources. |
| `shutdown()` | Shutdown the predictor, release memory, and clean up all resources. |

## 6. Performance & Memory Management

- **VRAM Requirements**: SAM3 typically requires ~8GB VRAM. If exceeded, the system falls back to CPU (extremely slow).
- **Caching**: `ThumbnailManager` implements an LRU cache in RAM. Max size is configurable in `core/config.py`.
- **Processing**: We use `ThreadPoolExecutor` for batch analysis. Limit `analysis_default_workers` if CPU RAM OOM occurs.
- **Optimization**: Extraction generates `video_lowres.mp4` at thumbnail resolution. `MaskPropagator` uses this directly during propagation to eliminate redundant JPEG I/O.

## 7. Configuration Reference

See `core/config.py` for the full schema.

| Category | Key Fields | Default |
|----------|------------|---------|
| **Paths** | `logs_dir`, `models_dir`, `downloads_dir` | `logs`, `models`, `downloads` |
| **Models** | `face_model_name`, `tracker_model_name` | `buffalo_l`, `sam3` |
| **Performance** | `analysis_default_workers`, `cache_size` | `4`, `200` |
| **Quality** | `quality_weights_*` | (Variable Weights) |

## 8. Troubleshooting & Known Issues

### Common Errors
- **"CUDA out of memory"**: Triggered during SAM3 init or NIQE analysis. **Fix**: Use `ModelRegistry.clear()` or set `APP_HUGGINGFACE_TOKEN` correctly.
- **"ModuleNotFoundError: sam3"**: Submodule missing. **Fix**: `git submodule update --init --recursive`.
- **"ValueError: ... is not in list" (Gradio)**: Occurs when updating `gr.Dropdown` values. **Fix**: Update `choices` list before the `value`.

### Environment Gotchas
- **PyTorch 2.9+**: May show TF32 deprecation warnings; these are safe to ignore.
- **venv pip**: Some environments may have path issues with venv `pip`. Use `python -m pip` or `uv pip`.
- **Ruff Format**: Ruff is aggressive. If code "disappears," check `__all__` re-exports or logic that might look like dead code to a static analyzer.

## 9. Security & Compliance

- **Integrity**: All model downloads are verified via SHA256 hashes (`core/utils._compute_sha256`).
- **Input Sanitization**: Video paths are validated via `validate_video_file()`. Export filenames are sanitized for path traversal.
- **Model Loading**: We prefer `.safetensors` over `.pt` for improved security.

---

## 10. Verification Framework

To ensure system stability, we employ a multi-layered verification strategy.

### 🟢 Layer 1: Unit & Integration Tests
Run these frequently during development to catch regressions early.
```bash
uv run pytest tests/
```

### 🟡 Layer 2: End-to-End (E2E) Verification
Validates the full processing pipeline using real video data and ML models (no mocks).
```bash
uv run python tests/e2e/e2e_run.py
```

### 🔵 Layer 3: UX Audit & Visual Regression
Automated browser testing using Playwright to verify UI components and visual consistency.
```bash
uv run python scripts/run_ux_audit.py
```

---

---

## 11. Extensibility: Adding New Quality Metrics (Operators)

The system uses an **auto-discovery** plugin pattern for quality metrics. You can add new analysis logic by creating a file in `core/operators/` and registering your class.

### Quick Start (Copy-Paste)

Create a new file `core/operators/my_metric.py`:

```python
from core.operators import Operator, OperatorConfig, OperatorContext, OperatorResult
from core.operators import register_operator

@register_operator
class MyMetricOperator:
    @property
    def config(self) -> OperatorConfig:
        return OperatorConfig(
            name="my_metric",
            display_name="My Custom Metric",
            category="quality",
            description="Measures something amazing.",
            min_value=0.0,
            max_value=100.0,
        )

    def execute(self, ctx: OperatorContext) -> OperatorResult:
        # Access image as numpy array (RGB)
        image = ctx.image_rgb
        
        # Compute your score (simulated here)
        score = 85.0 
        
        return OperatorResult(metrics={"my_metric_score": score})
```

The system will automatically find and load it on startup.

### Step-by-Step Implementation

1. **Create the File**: Add a Python file in `core/operators/` (e.g., `exposure.py`).
2. **Implement the Protocol**: Implementation must include a `config` property and an `execute(ctx)` method.
3. **Register**: Use the `@register_operator` decorator.
4. **Context Access**: Use `ctx.image_rgb`, `ctx.mask`, `ctx.config`, and `ctx.params` (e.g., face landmarks).

### Advanced: Heavy Models
If your operator requires a heavy model (neural net), implement `initialize(self, config)` to load it once and `cleanup(self)` to release it.

---

## 📄 References
- [AGENTS_CODE_REFERENCE.md](docs/AGENTS_CODE_REFERENCE.md): Auto-generated code skeletons.
- [TESTS_CODE_REFERENCE.md](docs/TESTS_CODE_REFERENCE.md): Auto-generated test skeletons.
