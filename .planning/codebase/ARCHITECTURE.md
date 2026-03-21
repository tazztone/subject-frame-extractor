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
- **Coordinate System**: UI coordinates (pixels) are normalized to [0.0, 1.0] before being passed to the `SAM3Wrapper`.
- **Output**: Binary masks stored as compressed `.png` or `.webp` files in the `masks/` directory.

### 3. Analysis Phase (`AnalysisPipeline` via `Operators`)
- **Metric Loop**: Parallel execution of "Operators" (Action/Quality metrics).
- **Concurrency**: Uses `ThreadPoolExecutor` with a pool size limited by `analysis_default_workers` to prevent CPU RAM exhaustion.
- **Persistence**: Final metrics are flushed to a SQLite `metadata.db` for fast filtering and export.

## Resource Management & Stability

### Thread-Safe Model Loading (`ModelRegistry`)
Large ML models (SAM3, InsightFace) are managed by a central registry:
- **Lazy Loading**: Models are only loaded into VRAM when first requested.
- **Locking**: Uses a reentrant `RLock` for the registry state and individual `threading.Lock` per model to prevent race conditions during initialization.
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
| `video_lowres.mp4`| Extraction | SAM3 | High-speed frame source for mask propagation. |

---

*Refined architecture: 2026-03-21*
