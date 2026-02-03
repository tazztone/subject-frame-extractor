# 🎬 Subject Frame Extractor

**An AI-powered powerhouse for extracting, analyzing, and filtering high-quality frames from video.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-CUDA%20Supported-ee4c2c.svg)](https://pytorch.org/)
[![Gradio](https://img.shields.io/badge/UI-Gradio-ff5000.svg)](https://gradio.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Designed for content creators, dataset builders (LoRA/Dreambooth), and researchers. This tool bridges the gap between raw video footage and curated, high-quality image datasets using state-of-the-art AI.

---

## ✨ Overview

Traditional frame extraction is noisy. **Subject Frame Extractor** uses advanced segmentation and quality heuristics to ensure you only keep the frames that matter.

*   **Intelligent Extraction**: Beyond simple intervals—use scene detection and keyframe awareness.
*   **Subject Centric**: Automatically track and mask specific people or objects using **SAM 3**.
*   **Quality First**: Filter by sharpness, contrast, and perceptual quality (**NIQE**).
*   **Face Matching**: Find every frame of a specific person using **InsightFace**.

---

## 🚀 Key Features

### 🎯 Smart Extraction
*   **Extraction Strategies**: Keyframes, fixed intervals, scene-based, or every Nth frame.
*   **YouTube Integration**: Direct URL processing with resolution control.
*   **Scene Intelligence**: Automatically segments video into shots to optimize analysis.

### 🧠 Advanced AI Analysis
*   **SAM 3 Integration**: Precise subject segmentation and tracking across scenes.
*   **Open-Vocabulary Detection**: Describe what you want to find (e.g., "a golden retriever") and let the AI find it.
*   **Face Analysis**: Similarity matching, blink detection, and head pose estimation (yaw/pitch/roll).
*   **Perceptual Metrics**: Real-time quality scoring to surface the "best" frames automatically.

### 🔍 Filtering & Export
*   **Interactive Sliders**: Filter thousands of frames in real-time based on AI-calculated metrics.
*   **Smart Deduplication**: Uses pHash and LPIPS to remove near-identical frames.
*   **AR-Aware Cropping**: Export subject-centered crops in 1:1, 9:16, 16:9, or custom ratios.

---

## 🛠️ Tech Stack

*   **Segmentation**: [Segment Anything Model 3 (SAM 3)](https://github.com/facebookresearch/sam3)
*   **Face Analysis**: [InsightFace](https://github.com/deepinsight/insightface)
*   **UI Framework**: [Gradio 6.x](https://gradio.app/)
*   **Data Science**: PyTorch, NumPy, OpenCV, Pydantic
*   **Media Handling**: FFmpeg, yt-dlp
*   **Database**: SQLite (for lightning-fast metadata filtering)

---

## 💻 Installation & Setup

### Prerequisites
*   **Python 3.10+** (3.12 recommended)
*   **FFmpeg** installed and in your system PATH.
*   **CUDA-capable GPU** (highly recommended; ~8GB VRAM for SAM 3).

### Quick Start (using `uv`)
We highly recommend [uv](https://astral.sh/uv) for its speed and reliability.

1.  **Clone with Submodules**
    ```bash
    git clone --recursive https://github.com/tazztone/subject-frame-extractor.git
    cd subject-frame-extractor
    ```
    *Note: Use `git submodule update --init --recursive` if already cloned.*

2.  **Sync Environment**
    ```bash
    uv sync
    ```

3.  **Launch**
    ```bash
    uv run python app.py
    ```
    Access the UI at `http://127.0.0.1:7860`.

### Manual Setup (vEnv)
1. `python -m venv venv`
2. Activate: `. venv/bin/activate` (Linux/Mac) or `. venv\Scripts\activate.ps1` (Windows)
3. `pip install -r requirements.txt`
4. `pip install -e SAM3_repo`

---

## 📖 Usage Guide

1.  **Source**: Upload a video or paste a YouTube URL. Choose your extraction resolution.
2.  **Extract**: Run the extraction. The tool identifies scenes and generates thumbnails.
3.  **Define Subject**: 
    *   **By Face**: Upload a reference photo for similarity matching.
    *   **By Text**: Enter a description (e.g., "cat", "person in red").
    *   **Auto**: Let the AI select the most prominent subject.
4.  **Analyze**: Review "Scene Seeds". Run **Propagation** to track subjects through the video.
5.  **Filter**: Use sliders in the **Metrics & Filtering** tab to curate your dataset.
6.  **Export**: Select your crop settings and aspect ratio, then hit **Export**.

---

# 🧑‍💻 Advanced Technical Reference & Developer Guide

**⚠️ CRITICAL**: Read this section before contributing or making significant changes.

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
- **Persistence**: `metadata.db` (SQLite) stores all frame metrics; `run_config.json` saves session parameters.

## 2. Critical Rules (The "Agent Memory")

### 🔴 CRITICAL (Must Follow)
- **Immutable Submodule**: **NEVER** edit files in `SAM3_repo`. Treat it as an external package.
- **Architectural Isolation**: **NEVER** import from `ui/` inside `core/`. Use callbacks or `core/shared.py` for cross-cutting needs.
- **Gradio Protocol**: **ALWAYS** ensure the number of return values in event handlers matches the `outputs` list. Mismatches cause silent app crashes.
- **Unhashable Config**: **NEVER** use `@lru_cache` on functions taking the `Config` object. Use `model_registry.get_or_load` instead.
- **Path Hygiene**: **ALWAYS** use `pathlib.Path`. Do not use string concatenation or `os.path`.
- **Hermetic Testing**: **ALWAYS** mock external dependencies (SAM3, InsightFace, Torch) in unit tests.

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

### Test Categories
| Test Type | Target File(s) | GPU? | Command |
|-----------|----------------|------|---------|
| **Unit** | `tests/test_*.py` | No | `pytest tests/` |
| **Pipelines** | `test_pipelines_extended.py` | No | `pytest tests/test_pipelines_extended.py` |
| **GPU E2E** | `test_gpu_e2e.py` | **Yes** | `pytest tests/test_gpu_e2e.py -v -m ""` |
| **Integration** | `test_integration.py` | **Yes** | `pytest -m integration` |
| **UI E2E** | `tests/e2e/` | No | `pytest tests/e2e/` (Uses Playwright) |

> **⚠️ NOTE**: GPU E2E tests require the `-m ""` flag to override default markers in `pyproject.toml`.

### GPU E2E Coverage
- `TestSAM3Inference`: Low-level API validation (init, prompts, propagation).
- `TestMaskPropagatorE2E`: Full integration of tracking logic.
- `TestInsightFaceInference`: Face detection and similarity matching.
- `TestQualityMetricsE2E`: Real-world NIQE and perceptual metric validation.

### Common Testing Patterns
```python
# Patching an imported class method
@patch("core.managers.ModelRegistry.get_tracker")
def test_tracker(mock_get, app_ui):
    mock_get.return_value = MagicMock()
    ...

# Skipping GPU tests on CPU-only machines
@pytest.mark.skipif(not _is_sam3_available(), reason="SAM3 not installed")
def test_sam3_feature(...):
    ...
```

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

### 🔴 Coordinate Handling Nuance
SAM3's official API strictly requires **relative coordinates (0.0 to 1.0)**. `SAM3Wrapper` performs this conversion internally. Always pass absolute pixel coordinates and the current image dimensions to the wrapper.

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
Validates the full processing pipeline using real video data and ML models (no mocks). This ensures that FFmpeg, SAM3, and InsightFace are working correctly together.
```bash
uv run python verification/e2e_run.py
```
*   **Requires**: `downloads/example clip (2).mp4` and `downloads/example face.png`.
*   **Output**: Checks for valid `metadata.db` and generated masks in `verification/e2e_output`.

### 🔵 Layer 3: UX Audit & Visual Regression
Automated browser testing using Playwright to verify UI components and visual consistency.
```bash
uv run python scripts/run_ux_audit.py
```
*   **Generates**: A comprehensive report in `ux_reports/`.
*   **Options**: Use `--update-baselines` to approve visual changes.

---

## 📄 License
MIT License. See [LICENSE](LICENSE) for details.

> **Internal Reference**: For detailed code skeletons, see [AGENTS_CODE_REFERENCE.md](AGENTS_CODE_REFERENCE.md)
