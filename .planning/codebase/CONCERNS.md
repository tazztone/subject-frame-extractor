# Technical Concerns & Risks

**Analysis Date:** 2026-03-21
**Deep Dive Refinement:** Detailed resource constraints and pipeline fragilities.

## Resource Management (VRAM / RAM)

### 1. SAM3 VRAM Fragmentation
- **Concern**: SAM3's video predictor maintains a temporal memory of objects. Long videos or multiple tracked objects can lead to VRAM fragmentation and eventual "CUDA Out of Memory".
- **Mitigation**: The `ModelRegistry` triggers `torch.cuda.empty_cache()` and attempts a CPU fallback for heavy models. However, CPU fallback is ~10-20x slower.
- **Warning**: Users with < 8GB VRAM will struggle with propagation on high-resolution videos.

### 2. Face Analysis Thread-Safety 🔴
- **Concern**: `InsightFace` (FaceAnalysis) is not natively thread-safe. While the `ModelRegistry` has a lock, the `AnalysisPipeline` uses a `ThreadPoolExecutor` and calls `analyzer.get()` inside operators without a shared lock.
- **Impact**: Concurrent inference calls on the same instance can cause segfaults or corrupted metadata.
- **Fix**: Consolidate face detection to run once in the pipeline and pass results to operators.

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

- **String-based Status**: Scene and Task statuses (e.g., `"pending"`, `"success"`) are currently strings. Refinement to Enums is needed to ensure type safety.
- **Duplicate Logic (Maintenance)**: `core/export.py` contains duplicate unreachable `except` blocks for FFmpeg errors.
- **XMP Robustness**: XMP export uses `xml.etree.ElementTree`, which is robust, but the logic lacks comprehensive unit tests for different sidecar scenarios.

## Performance & Memory Management

- **VRAM Requirements**: SAM3 typically requires ~8GB VRAM. If exceeded, the system falls back to CPU (extremely slow).
- **Caching**: `ThumbnailManager` implements an LRU cache in RAM. Max size is configurable in `core/config.py`.
- **Processing**: We use `ThreadPoolExecutor` for batch analysis. Limit `analysis_default_workers` if CPU RAM OOM occurs.
- **Optimization**: Extraction generates `video_lowres.mp4` at thumbnail resolution. `MaskPropagator` uses this directly during propagation to eliminate redundant JPEG I/O.

## Troubleshooting & Known Issues

### Common Errors
- **"CUDA out of memory"**: Triggered during SAM3 init or NIQE analysis. **Fix**: Use `ModelRegistry.clear()` or set `APP_HUGGINGFACE_TOKEN` correctly.
- **"ModuleNotFoundError: sam3"**: Submodule missing. **Fix**: `git submodule update --init --recursive`.
- **"ValueError: ... is not in list" (Gradio)**: Occurs when updating `gr.Dropdown` values. **Fix**: Update `choices` list before the `value`.

### Environment Gotchas
- **PyTorch 2.9+**: May show TF32 deprecation warnings; these are safe to ignore.
- **venv pip**: Some environments may have path issues with venv `pip`. Use `python -m pip` or `uv pip`.
- **Ruff Format**: Ruff is aggressive. If code "disappears," check `__all__` re-exports or logic that might look like dead code to a static analyzer.

## Security & Compliance

- **Integrity**: All model downloads are verified via SHA256 hashes (`core/utils._compute_sha256`).
- **Input Sanitization**: Video paths are validated via `validate_video_file()`. Export filenames are sanitized for path traversal.
- **Model Loading**: We prefer `.safetensors` over `.pt` for improved security.

---


*Refined concerns: 2026-03-21*
