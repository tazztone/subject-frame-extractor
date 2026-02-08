---
phase: 3
plan: 1
wave: 1
---

# Plan 3.1: Tab 3 Performance Optimization

## Objective
Eliminate UI lag in the Filtering tab by caching masks and optimizing gallery rendering.

## Context
- .gsd/phases/3/RESEARCH.md
- ui/gallery_utils.py (entire file, ~140 lines)
- core/utils.py (`render_mask_overlay` function)

## Tasks

<task type="auto">
  <name>Implement Mask LRU Cache</name>
  <files>ui/gallery_utils.py</files>
  <action>
    Add a module-level cached mask loader:
    ```python
    from functools import lru_cache
    import cv2
    
    @lru_cache(maxsize=256)
    def _load_mask_cached(mask_path: str):
        if not Path(mask_path).exists():
            return None
        return cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    ```
    Update `_update_gallery` or `render_mask_overlay` to use this cached loader.
    Add a `clear_mask_cache()` function that calls `_load_mask_cached.cache_clear()`.
  </action>
  <verify>grep -n "lru_cache" ui/gallery_utils.py</verify>
  <done>Mask loading uses LRU cache.</done>
</task>

<task type="auto">
  <name>Limit Gallery Render Count</name>
  <files>ui/gallery_utils.py</files>
  <action>
    In `_update_gallery`, limit the number of items rendered with overlays to 100 (or configurable):
    ```python
    MAX_OVERLAY_RENDER = 100
    for i, item in enumerate(gallery_items[:MAX_OVERLAY_RENDER]):
        # render with overlay
    # Remaining items: use thumbnail without overlay
    ```
    This prevents rendering 1000+ overlays on every filter change.
  </action>
  <verify>grep -n "MAX_OVERLAY_RENDER" ui/gallery_utils.py</verify>
  <done>Gallery rendering is capped to prevent lag.</done>
</task>

## Success Criteria
- [ ] Moving sliders in Tab 3 feels responsive (< 200ms update)
- [ ] Scrolling through 500+ images doesn't freeze the UI
