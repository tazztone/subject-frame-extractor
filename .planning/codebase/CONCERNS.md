# Technical Concerns & Risks

**Analysis Date:** 2026-03-21
**Deep Dive Refinement:** Detailed resource constraints and pipeline fragilities.

## Resource Management (VRAM / RAM)

### 1. SAM3 VRAM Fragmentation
- **Concern**: SAM3's video predictor maintains a temporal memory of objects. Long videos or multiple tracked objects can lead to VRAM fragmentation and eventual "CUDA Out of Memory".
- **Mitigation**: The `ModelRegistry` triggers `torch.cuda.empty_cache()` and attempts a CPU fallback for heavy models. However, CPU fallback is ~10-20x slower.
- **Warning**: Users with < 8GB VRAM will struggle with propagation on high-resolution videos.

### 2. Face Analysis Thread-Safety
- **Concern**: `InsightFace` and `MediaPipe` are not natively thread-safe. Concurrent calls to `execute_analysis` with multiple workers can cause segfaults or corrupted results.
- **Mitigation**: The system uses a `ModelRegistry` lock, but high-concurrency analysis (`analysis_default_workers > 4`) is risky on low-RAM systems.

## Performance Bottlenecks

### 1. FFmpeg `showinfo` Overhead
- **Concern**: Precisely mapping frames back to timestamps requires parsing FFmpeg stderr from the `showinfo` filter.
- **Impact**: For extremely long videos (1hr+), this parsing overhead can become significant, potentially slowing down the extraction phase.

### 2. JPEG Decompression (Legacy Mode)
- **Concern**: In "Legacy Full-Frame" mode, the `SubjectMasker` reads high-res JPEGs for every frame.
- **Impact**: I/O bound. The "Recommended Thumbnails" mode mitigates this by using `video_lowres.mp4`.

## Pipeline Fragility

### 1. Gradio Event IO Mismatch 🔴
- **Concern**: Gradio requires that the number of inputs/outputs in `.click()` or `.change()` exactly matches the function signature.
- **Impact**: Adding a new UI control without updating the corresponding pipeline wrapper results in a silent crash where the UI simply stops responding.

### 2. SAM3 Submodule Dependency
- **Concern**: The project treats `SAM3_repo` as read-only, but its internal imports are sensitive to the Python path.
- **Risk**: Environment changes (different `PYTHONPATH`) can break the link between `core/managers.py` and the submodule.

### 3. Mask Persistence Integrity
- **Concern**: Masks are stored as separate `.png` files. If the user deletes the `masks/` directory manually, the `metadata.db` becomes inconsistent, leading to "File Not Found" errors during export.

## Maintenance Debt

- **String-based Status**: Scene and Task statuses (e.g., `"pending"`, `"success"`) are strings. Refactoring to Enums is needed to prevent typos in core logic.
- **XMP Complexity**: Exporting to XMP (Adobe/Lightroom) involves XML manipulation which is currently fragile and lacks comprehensive unit testing.

---

*Refined concerns: 2026-03-21*
