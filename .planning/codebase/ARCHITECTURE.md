# Architecture Detail

**Analysis Date:** 2026-03-21
**Deep Dive Refinement:** Added detailed data flow and resource management logic.

## Core/UI Separation

The system follows a strict isolation pattern:
- **UI (Gradio)**: Collects user parameters and displays results. It is prohibited from executing heavy logic directly.
- **Core (Pipelines)**: Receives an `AnalysisParameters` object (Pydantic) and executes long-running tasks in background threads.

## Pipeline Orchestration

The processing flow is split into three distinct phases to allow for checkpoints and resumability:

### 1. Extraction Phase (`ExtractionPipeline`)
- **Input**: Video file or Image folder.
- **FFmpeg Integration**: Uses a complex filter chain (`scale`, `showinfo`) to extract frames.
- **Frame Mapping**: Captures `pts_time` from FFmpeg stderr to create `frame_map.json`. This ensures that even with variable frame rates, the pipeline can accurately map extracted images back to original video timestamps.
- **Optimization**: Automatically generates a `video_lowres.mp4` (360p) for SAM3. This avoids slow JPEG I/O during the propagation phase.

### 2. Propagation Phase (`AnalysisPipeline` via `SubjectMasker`)
- **Model**: SAM3 (Segment Anything Model v3).
- **Temporal Memory**: SAM3 uses a session-based approach where it tracks objects across frames. 
- **PVS vs. PCS Protocol**: The system strictly uses the **PVS (Promptable Visual Segmentation)** tracker path instead of the PCS (Semantic) path. This ensures tracking state persistence across frames and avoids the automatic `reset_state()` calls triggered by high-level BBox APIs.
- **Coordinate System**: UI coordinates (pixels) are normalized to [0.0, 1.0] before being passed to the `SAM3Wrapper`.
- **Output**: Binary masks stored as compressed `.png` or `.webp` files in the `masks/` directory.

### 3. Pre-Analysis Phase (`execute_pre_analysis`)
- **Seeding**: Automatically selects the "Best Frame" per scene based on quality metrics and face similarity.
- **Subject Discovery**: Runs face detection or open-vocabulary object detection (SAM3) to initialize tracking seeds.

### 4. Analysis Phase (`AnalysisPipeline` via `Operators`)
- **Metric Loop**: Parallel execution of "Operators" (Action/Quality metrics).
- **Concurrency**: Uses `ThreadPoolExecutor` with a pool size limited by `analysis_default_workers` to prevent CPU RAM exhaustion.
- **Persistence**: Final metrics are flushed to a SQLite `metadata.db` for fast filtering and export.

### 5. Filtering & Export Phase (`core/export.py`)
- **Refinement**: Applies vectorized thresholds and deduplication (pHash/LPIPS) to selection.
- **Rendering**: Final high-res extraction via FFmpeg and sidecar generation (XMP, JSON, CSV).

## Resource Management & Stability

### Thread-Safe Model Loading (`ModelRegistry`)
Large ML models (SAM3, InsightFace) are managed by a central registry:
- **Lazy Loading**: Models are only loaded into VRAM when first requested.
- **Locking**: Uses a reentrant `RLock` for the registry state and individual `threading.Lock` per model to prevent race conditions during initialization.
- **Path Safety**: The registry strictly validates `models_path`. If `models_path` is `None` (common in misconfigured environments), it logs an error and returns `None` instead of crashing with a `Path` instantiation error.
- **OOM Recovery**: If a `RuntimeError` with "out of memory" is caught during initialization, the registry automatically clears the CUDA cache and retries the load on the CPU.

### Image Caching (`ThumbnailManager`)
- **Pattern**: Least Recently Used (LRU) cache.
- **Cleanup**: Triggers eviction when the cache exceeds `cache_size * cache_cleanup_threshold` (default 90%).

## Data Flow Contract

| Artifact | Producer | Consumer | Purpose |
|----------|----------|-----------|---------|
| `AnalysisParameters` | UI | All Pipelines | Configuration and user intent. |
| `frame_map.json` | Extraction | Analysis, UI | Maps file index to original video frame index. |
| `scenes.json` | Extraction | Analysis, UI | Defines shot boundaries and status (`pending`, `included`). |
| `metadata.db` | Analysis | UI, Export | Queryable frame quality and metadata. |
| `mask_metadata.json` | Analysis | Pipeline | Mapping frame indices to mask file paths and metrics. |
| `video_lowres.mp4`| Extraction | SAM3 | High-speed frame source for mask propagation. |
| `run_config.json` | Core | Core | Persistent storage of session parameters. |
| `run_fingerprint.json`| Core | Core | Enables hash-based skipping of already processed videos. |

## Directory Responsibilities

- **`app.py`**: The main entry point. Assembles the Gradio interface and manages layout.
- **`core/`**: Business logic and heavy lifting (Pipelines, Managers, Models, Database).
- **`ui/`**: Modular UI components and gallery generation logic.
- **`SAM3_repo/`**: **Read-only** official SAM3 submodule.
- **`tests/`**: Comprehensive test suite (Unit, Integration, E2E).

---

## Extensibility: Adding New Quality Metrics (Operators)

The system uses an **auto-discovery** plugin pattern for quality metrics. You can add new analysis logic by creating a file in `core/operators/` and registering your class.

### Implementation Checklist

1. **Create the File**: Add a Python file in `core/operators/` (e.g., `exposure.py`).
2. **Implement the Protocol**: Implementation must include a `config` property and an `execute(ctx)` method.
3. **Register**: Use the `@register_operator` decorator.
4. **Context Access**: Use `ctx.image_rgb`, `ctx.mask`, `ctx.config`, and `ctx.params` (e.g., face landmarks).

### Quick Start (Copy-Paste Template)

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

### Advanced: Heavy Models
If your operator requires a heavy model (neural net), implement `initialize(self, config)` to load it once and `cleanup(self)` to release it.


---

*Refined architecture: 2026-03-21*
