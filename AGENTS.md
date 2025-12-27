---
Version: 2.2
Last Updated: 2025-12-26
Python: 3.10+
Gradio: 6.x
SAM3: Via submodule
---

# Developer Guidelines & Agent Memory

**⚠️ CRITICAL**: Read this before starting any task.

🔴 CRITICAL | 🟡 WARNING | 🟢 BEST PRACTICE


## 1. Quick Start Guide

### 5-Minute Setup
1. **Clone & Submodules**: `git clone --recursive` or `git submodule update --init --recursive`
2. **Create venv**: `python -m venv venv`
3. **Activate venv**:
   - Windows: `. venv\Scripts\activate.ps1`
   - Linux/Mac: `source venv/bin/activate`
4. **Install dependencies (use uv for speed)**:
   ```bash
   uv pip install -r requirements.txt
   uv pip install -e SAM3_repo
   ```
5. **Run App**: `python app.py`

### Essential Commands
```bash
# Activate venv first!
. venv\Scripts\activate.ps1  # Windows
source venv/bin/activate       # Linux/Mac

# Unit tests (fast, uses mocks)
python -m pytest tests/

# GPU E2E tests (requires CUDA + SAM3)
python -m pytest tests/test_gpu_e2e.py -v -m "" --tb=short

# Run specific E2E tests (requires Playwright)
python -m pytest tests/e2e/test_filters_real.py
```

> **⚠️ NOTE**: GPU tests require `-m ""` to override the default marker filter in `pyproject.toml`.

### Directory Structure
- `app.py`: Entry point.
- `core/`: Business logic (pipelines, config, db).
- `ui/`: Gradio interface components.
- `tests/`: Unit and E2E tests.
- `SAM3_repo/`: **Read-only** submodule (install with `uv pip install -e SAM3_repo`).


## 2. Critical Rules

### 🔴 CRITICAL (Must Follow)
- **NEVER** edit files in `SAM3_repo`. Treat as an external library.
- **NEVER** import from `ui/` in `core/` modules. Use callbacks or `core/shared.py` for shared functionality.
- **ALWAYS** match Gradio event handler return values count to the `outputs` list. Mismatches crash the app silently.
- **NEVER** use `@lru_cache` on functions taking the `Config` object (it's unhashable). Use `model_registry.get_or_load`.
- **ALWAYS** use `pathlib.Path`, never `os.path` or `os.access`.
- **ALWAYS** mock external dependencies (SAM3, Torch) in unit tests.
- **ALWAYS** use `tests/conftest.py` fixtures for mock objects in tests.

### 🟡 WARNING (Potential Bugs)
- **Check Masks**: Verify masks exist on disk before export/processing.
- **Thread Safety**: MediaPipe objects are not thread-safe. Use thread-local storage or one instance per thread.
- **Gradio State**: Do not store locks or file handles in `gr.State`.
- **Circular Imports**: If you need UI functions in core, add them to `core/shared.py`.

### 🟢 BEST PRACTICE
- **Refactoring**: Move logic from `app.py` to `core/`.
- **Typing**: Use Pydantic models (`core/events.py`) instead of untyped dicts.
- **Testing**: Add fixtures to `tests/conftest.py` for reuse across test files.
- **E2E Testing**: Use `tests/e2e/test_filters_real.py` as a reference for meaningful E2E tests using sample data.


## 3. Architecture Overview

### Data Flow
`UI (Gradio)` → `Event Object (Pydantic)` → `Pipeline (Core)` → `Database/Files`

### Component Relationship
```
[app.py] (UI Assembly)
   │
   ├─ [core/config.py] (Settings)
   ├─ [core/managers.py] (ModelRegistry, ThumbnailManager)
   └─ [core/pipelines.py] (Logic)
         │
         ├─ ExtractionPipeline (FFmpeg)
         ├─ AnalysisPipeline (SAM3, InsightFace)
         └─ ExportPipeline (Filtering, Rendering)
```

### State Management
- **Session State**: `gr.State` stores mutable data (scene lists, paths).
- **Global State**: `ModelRegistry` (Singleton-like) manages heavy models.
- **Persistence**: `metadata.db` (SQLite) for frame data; `json` for configs.


## 4. Development Workflows

### Bug Fix Workflow
1. **Reproduce**: Create a test case in `tests/test_reproduce_issue.py`.
2. **Log**: Use `logger.debug()` to trace execution.
3. **Fix**: Implement fix in `core/` or `ui/`.
4. **Verify**: Run `python -m pytest tests/`.
5. **Clean**: Remove temporary test files.

### Adding a New Metric
1. **Config**: Add default thresholds to `Config` in `core/config.py`.
2. **Extraction**: Update `_extract_metric_arrays()` in `core/filtering.py`.
3. **UI**: Add slider in `ui/app_ui.py` inside `_create_filtering_tab`.
4. **Analysis**: Update `calculate_quality_metrics` in `core/models.py`.


## 5. Testing & Mocking Guide

### Test Types
| Test Type | File | GPU? | Run With |
|-----------|------|------|----------|
| Unit | `test_*.py` | No | `pytest tests/` |
| Smoke | `test_smoke.py` | No | `pytest tests/test_smoke.py` |
| UI Handlers | `test_handlers.py` | No | `pytest tests/test_handlers.py` |
| Scene Detection | `test_scene_detection.py` | No | `pytest tests/test_scene_detection.py` |
| Signature | `test_signatures.py` | No | `pytest tests/test_signatures.py` |
| Integration | `test_integration.py` | Yes | `pytest -m integration` |
| **GPU E2E** | `test_gpu_e2e.py` | Yes | `pytest tests/test_gpu_e2e.py -v -m ""` |
| E2E | `tests/e2e/` | No* | `pytest tests/e2e/` (*Uses Mock App) |

> **⚠️ CRITICAL**: GPU E2E tests require `-m ""` flag to override `pyproject.toml` marker filter!

### GPU E2E Test Classes
- `TestSAM3Inference`: Tests SAM3 API (init, prompts, propagation)
- `TestMaskPropagatorE2E`: Tests mask propagation integration
- `TestInsightFaceInference`: Tests face analysis
- `TestQualityMetricsE2E`: Tests NIQE and other metrics

### E2E Tests (Playwright)
Located in `tests/e2e/`. Use `app_server` fixture to run against a mock backend.
- `test_filters_real.py`: Tests filtering logic with sample data.
- `test_export_flow.py`: Tests export UI flow.

### When to Mock
- **File I/O**: Patch `pathlib.Path.exists`, `open`.
- **ML Models**: Always mock `SAM3Wrapper`, `FaceAnalysis`, `FaceLandmarker`.
- **Submodules**: Mock `sam3` package to avoid import errors.

### Common Patterns
```python
# Mocking a class method
@patch("core.managers.ModelRegistry.get_tracker")
def test_tracker(mock_get, app_ui):
    mock_get.return_value = MagicMock()
    ...

# Skip if SAM3 not installed
@pytest.mark.skipif(not _is_sam3_available(), reason="SAM3 not installed")
def test_sam3_feature(...):
    ...

### Common Testing Pitfalls (Found Dec 2024)
- **Patching Targets**: Always patch the module where the function/class is *imported*, not where it is defined. For example, if `ui/handlers/filtering_handler.py` imports `on_filters_changed` from `ui.gallery_utils`, you must patch `ui.gallery_utils.on_filters_changed`.
- **Private vs Public Attributes**: Some managers (e.g., `ModelRegistry`, `ThumbnailManager`) use public attributes for `logger` and `cache` instead of private ones. Check the `__init__` before asserting on `self._logger`!
- **Mocking Loggers**: `AppLogger` instances are often real objects in tests, not `MagicMock`s. Don't use `.assert_called()` on them; instead, use `@patch` if you need to verify logging calls.
- **JSON Structure**: When verifying saved state (like `scene_seeds.json`), remember it might be a dictionary keyed by IDs, not a list.
```

### E2E vs Unit
- **Unit**: Fast, mocks everything. Run pre-commit.
- **GPU E2E**: Real model inference. Catches dtype/CUDA errors.
- **E2E**: Full app flow with Playwright.


## 6. SAM3 API Guide

### Official API Pattern (as of Dec 2024)
> **Source of Truth**: The implementation is based on [sam3_video_predictor_example.ipynb](file:///c:/_coding/subject-frame-extractor2/SAM3_repo/examples/sam3_video_predictor_example.ipynb).
The `SAM3Wrapper` class uses the official `Sam3VideoPredictor` API via `build_sam3_video_predictor`:

```python
from core.managers import SAM3Wrapper

wrapper = SAM3Wrapper(device="cuda")

# 1. Initialize with video/frames directory
wrapper.init_video("/path/to/frames_dir")

# 2. Add bbox prompt (absolute coords, auto-normalized to 0-1)
mask = wrapper.add_bbox_prompt(
    frame_idx=0,
    obj_id=1,
    bbox_xywh=[50, 50, 100, 150],  # x, y, width, height (pixels)
    img_size=(width, height)
)

# 3. Propagate masks (generator)
for frame_idx, obj_id, mask in wrapper.propagate(start_idx=0, reverse=False):
    process_mask(mask)

# 4. Clean up when done
wrapper.close_session()
```

### Complete API Reference

| Method | Description | Returns |
|--------|-------------|---------|
| `init_video(video_path)` | Initialize session with video/frames | session_id |
| `add_bbox_prompt(frame_idx, obj_id, bbox_xywh, img_size)` | Add bounding box prompt | np.ndarray (mask) |
| `detect_objects(frame_rgb, prompt)` | Text-based object detection (single image) | list[dict] with bbox, conf, type |
| `add_text_prompt(frame_idx, text)` | Add text prompt for video detection | dict with obj_ids, masks, boxes |
| `add_point_prompt(frame_idx, obj_id, points, labels, img_size)` | Mask refinement with clicks | np.ndarray (refined mask) |
| `propagate(start_idx, max_frames, reverse)` | Generator for mask propagation | yields (frame_idx, obj_id, mask) |
| `clear_prompts()` | Clear all prompts in session | None |
| `reset_session()` | Reset without closing session | None |
| `close_session()` | Close session, free GPU memory | None |
| `remove_object(obj_id)` | Remove specific object from session | None |
| `shutdown()` | Shutdown predictor (multi-GPU cleanup) | None |

### Text-Based Detection (Open Vocabulary)
```python
# Single image detection (using legacy processor)
results = wrapper.detect_objects(frame_rgb, "person")
for r in results:
    print(f"Found at {r['bbox']} with confidence {r['conf']}")

# Video text prompt (for tracking using new API)
wrapper.init_video(video_path)
result = wrapper.add_text_prompt(frame_idx=0, text="person")
# Then propagate detected objects
```

### Point Prompts for Mask Refinement
```python
# Positive click (include this region)
mask = wrapper.add_point_prompt(
    frame_idx=0, obj_id=1,
    points=[(150, 200)], labels=[1],  # 1=positive
    img_size=(width, height)
)

# Negative click (exclude this region)
mask = wrapper.add_point_prompt(
    frame_idx=0, obj_id=1,
    points=[(50, 50)], labels=[0],  # 0=negative
    img_size=(width, height)
)
```

### Key Differences from Legacy API
| Old (handle_request) | New (Official Sam3VideoPredictor) |
|---------------------|----------------|
| `handle_request(type="start_session")` | `init_video(video_path)` (uses `start_session` internally) |
| `handle_request(type="add_prompt")` | `add_bbox_prompt()` (uses `add_prompt` internally) |
| `handle_request(type="propagate")` → dict | `propagate()` → **generator** (uses `propagate_in_video` stream) |
| Absolute pixel coords | **Relative (0-1) coords** (wrapper handles conversion) |

### 🔴 CRITICAL: Coordinate Handling
- SAM3 official API expects **relative coordinates (0-1)**
- `SAM3Wrapper.add_bbox_prompt()` handles conversion automatically
- Pass absolute pixel coords + image size, it converts internally to the required format (e.g. `[x, y, w, h]` normalized)

### Performance: Downscaled Video
During extraction, `video_lowres.mp4` is created at thumbnail resolution. When available, `MaskPropagator.propagate_video()` uses this directly, eliminating temp JPEG I/O.


## 7. Configuration Reference

See `core/config.py` for full list.

| Category | Key Fields | Default |
|----------|------------|---------|
| **Paths** | `logs_dir`, `models_dir`, `downloads_dir` | `logs`, `models`, `downloads` |
| **Models** | `face_model_name`, `tracker_model_name` | `buffalo_l`, `sam3` |
| **perf** | `analysis_default_workers` | 4 |
| **UI** | `default_thumb_megapixels` | 0.5 |


## 8. Troubleshooting

### Error: "CUDA out of memory"
- **Where**: SAM3 initialization, NIQE metric.
- **Fix**: Set `model_registry.runtime_device_override = 'cpu'`.
- **Prevention**: Call `cleanup_models()` between sessions.

### Error: "ModuleNotFoundError: sam3"
- **Cause**: Submodule not initialized.
- **Fix**: `git submodule update --init --recursive`.
- **Check**: Verify `SAM3_repo/` exists and has files.

### Error: "ValueError: ... is not in list" (Gradio)
- **Cause**: `gr.Radio` or `gr.Dropdown` value updated to something not in `choices`.
- **Fix**: Update `choices` list *before* setting `value`.


## 9. Performance & Memory

- **SAM3**: Requires ~8GB VRAM. Falls back to CPU (slow).
- **Thumbnails**: Cached in RAM (`ThumbnailManager`). LRU eviction.
- **Batch Processing**: Uses `ThreadPoolExecutor`. Limit workers in Config if OOM occurs.


## 10. API Quick Reference

### Key Functions
- `execute_extraction(event: ExtractionEvent) -> Generator`
- `execute_pre_analysis(event: PreAnalysisEvent) -> Generator`
- `execute_propagation(event: PropagationEvent) -> Generator`

### Event Models
- `ExtractionEvent`: Source path, method (interval/scene).
- `PreAnalysisEvent`: Analysis params, seed strategy.
- `PropagationEvent`: List of scenes to process.


## 11. Git & Deployment

- **Submodules**: Always update recursive.
- **Requirements**: `requirements.txt` is root.
- **Validation**: Verify model downloads with SHA256.


## 12. Contribution Guidelines

### Code Style
- **Linting & Formatting**: Use `ruff` (replaces black, isort, flake8). Run `ruff check .` and `ruff format .`
- **Naming**: snake_case for functions/variables, PascalCase for classes
- **Line Length**: 120 characters max (configured in `pyproject.toml`)
- **Docstrings**: Use Google-style docstrings for public APIs

### Pull Request Process
1. Create feature branch from `main`
2. Run `python -m pytest tests/` before committing
3. Update AGENTS.md if adding new modules: `python scripts/update_agents_md.py`
4. Request review from maintainers

### Adding New Modules
- Place business logic in `core/`
- Place UI components in `ui/`
- Update imports in `__init__.py` files
- Add test fixtures to `tests/conftest.py`


## 13. Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `APP_HUGGINGFACE_TOKEN` | HuggingFace token for gated models (SAM3) | None |
| `APP_MODELS_DIR` | Override models download location | `models/` |
| `APP_LOGS_DIR` | Override logs location | `logs/` |
| `CUDA_VISIBLE_DEVICES` | Limit which GPUs are used | All |

### .env File
Copy `.env_example` to `.env` and configure:
```bash
APP_HUGGINGFACE_TOKEN=hf_your_token_here
```


## 14. Known Issues & Gotchas

### GPU E2E Tests Require `-m ""`
The `pyproject.toml` has a default marker filter that skips GPU tests. Override with:
```bash
python -m pytest tests/test_gpu_e2e.py -v -m ""
```

### SAM3 Coordinate System
- SAM3 API expects **relative coordinates (0-1)**, not pixels
- `SAM3Wrapper.add_bbox_prompt()` handles conversion automatically
- Always pass `img_size=(width, height)` for correct normalization

### PyTorch 2.9+ BFloat16 Warning
You may see TF32 deprecation warnings. These are informational only and don't affect functionality.

### venv pip May Be Broken
If `pip` fails with path errors, use `python -m pip` or `uv pip` instead.

### Forward References and TYPE_CHECKING
When using string type hints like `logger: 'AppLogger'`, Ruff's F821 check flags them as undefined. Fix by importing under TYPE_CHECKING:
```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.logger import AppLogger

class MyClass:
    def __init__(self, logger: 'AppLogger'):  # Now valid
        pass
```

### Ruff Format vs Code Style
`ruff format` aggressively reformats code. It does **NOT** delete functions or change logic. If code disappears after formatting, check:
1. Git status for uncommitted changes
2. Missing re-exports (add to `__all__` to prevent F401 removal suggestions)
3. File was overwritten by another operation

### Re-exporting Modules
When re-exporting functions from another module for backward compatibility, add `__all__` to prevent F401 warnings:
```python
# ui/gallery_utils.py
from core.shared import build_scene_gallery_items

__all__ = ["build_scene_gallery_items"]  # Prevents F401 "unused import"
```


## 15. Debugging Workflows

### Tracing SAM3 Issues
1. Check SAM3 is installed: `python -c "from sam3.model_builder import build_sam3_video_predictor"`
2. Verify CUDA: `python -c "import torch; print(torch.cuda.is_available())"`
3. Check VRAM: `nvidia-smi`
4. Run single test: `pytest tests/test_gpu_e2e.py::TestSAM3Inference::test_sam3_wrapper_initialization -v -m ""`

### Debugging Mask Propagation
1. Check `MaskPropagator.propagate()` receives valid frames
2. Verify bbox format is `[x, y, width, height]` (not xyxy)
3. Ensure temp directory is created and cleaned up

### Common Log Locations
- Application logs: `logs/app.log`
- GPU memory issues: Check `nvidia-smi` during inference
- Model downloads: `models/` directory


## 16. Windows-Specific Notes

### Triton Not Available
SAM3 uses Triton for some operations. On Windows, fallback implementations are used automatically via `core/sam3_patches.py`.

### Path Handling
- Always use `pathlib.Path`, never string concatenation
- Backslashes are handled automatically

### venv Activation
```powershell
. venv\Scripts\activate.ps1
```

### Long Paths
If you encounter path length issues, enable long paths in Windows:
```powershell
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```


## 17. Security Considerations

### Model Downloads
- All model downloads are verified via SHA256 hash
- Never disable hash verification in production
- Use `core/utils.download_model()` which has built-in retry and validation

### User Inputs
- Video paths are validated via `validate_video_file()`
- Text prompts are sanitized before use with external models
- File exports use `sanitize_filename()` to prevent path traversal
- Never construct paths from user input without validation

### Session Data
- Session directories are validated before loading (`validate_session_dir()`)
- JSON files are parsed with error handling to prevent injection



---

> **Code Reference**: For detailed code skeletons, see [AGENTS_CODE_REFERENCE.md](AGENTS_CODE_REFERENCE.md)
