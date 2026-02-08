# Research: Phase 3 (Tab 3 UX & Performance)

## Context
This phase addresses the severe UI lag experienced in Tab 3 (Filtering) when adjusting sliders, caused by inefficient gallery rendering.

## Findings (from Audit Report)

### 1. Tab 3 Gallery Lag
- **Issue**: `_update_gallery` in `ui/gallery_utils.py` is called whenever filters change. It iterates through all kept frames and calls `render_mask_overlay` for each.
- **Bottleneck**: `render_mask_overlay` reads the mask image from disk *synchronously* on the main thread for every frame in the gallery view.
- **Fix**:
    - Implement a simple `lru_cache` for mask loading in `ui/gallery_utils.py`.
    - Optimize `render_mask_overlay` to reuse loaded masks.
    - Ensure `_update_gallery` only renders visible items (pagination) or optimizes the loop.

### 2. Vectorized Filtering
- **Status**: The filtering logic in `core/filtering.py` uses `pandas` and `numpy` and appears efficient.
- **Action**: Verify that `apply_all_filters` is indeed vectorized and not the source of lag. The audit confirmed the lag is in *rendering*, not *filtering*.

## Implementation Strategy
- specific optimization of `ui/gallery_utils.py`.
- Validation of `core/filtering.py`.
