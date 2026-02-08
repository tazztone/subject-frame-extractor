# Research: Phase 2 (ARW Resource Optimization & Pipeline Safety)

## Context
This phase focuses on reducing the massive file sizes of extracted ARW previews and preventing crashes when processing image-only folders.

## Findings (from Audit Report)

### 1. ARW Resource Optimization
- **Issue**: `extract_preview` in `core/photo_utils.py` prioritizes `-JpgFromRaw`, which extracts full-resolution embedded JPEGs (often 10MB+). This wastes disk space and slows down loading.
- **Goal**: Extract smaller previews (~0.5MP) for the UI, while keeping the original ARW for processing.
- **Fix**:
    - Modify `extract_preview` to accept a `thumbnails_only=True` flag.
    - If `thumbnails_only` is set, prioritize `-ThumbnailImage` or `-PreviewImage`.
    - If extracted image is > 1MB or > 1000px, resize it to max 1000px long edge.

### 2. Pipeline Safety (Image-Only Mode)
- **Issue**: `VideoManager.get_video_info(video_path)` raises `IOError` if `video_path` is None.
- **Scenario**: When loading a folder of images, `video_path` is correctly `None`, but `execute_propagation` and `execute_analysis` still try to call `get_video_info` to estimate progress totals.
- **Fix**:
    - In `core/pipelines.py`: Add `if params.video_path:` check before calling `VideoManager.get_video_info`.
    - If `video_path` is None, skip video-specific total estimation or use file count.

## Implementation Strategy
- Update `core/photo_utils.py` with resizing logic.
- Update `core/pipelines.py` with null guards.
