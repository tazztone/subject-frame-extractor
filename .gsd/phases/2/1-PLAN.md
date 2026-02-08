---
phase: 2
plan: 1
wave: 1
---

# Plan 2.1: ARW Preview Optimization

## Objective
Reduce the file size of extracted RAW previews from multi-MB to ~150KB for faster UI loading.

## Context
- .gsd/phases/2/RESEARCH.md
- core/photo_utils.py (entire file, 100 lines)
- core/managers.py (`EmbeddedPreviewExtractor` class, around line 200)

## Tasks

<task type="auto">
  <name>Prioritize ThumbnailImage Extraction</name>
  <files>core/photo_utils.py</files>
  <action>
    Modify `extract_preview` to reorder tag priority when `thumbnails_only=True`:
    - Current order: `JpgFromRaw` > `PreviewImage` > `ThumbnailImage`
    - New order for thumbnails: `ThumbnailImage` > `PreviewImage` > `JpgFromRaw`
    Add a `thumbnails_only: bool = True` parameter with this as the default.
  </action>
  <verify>grep -n "ThumbnailImage" core/photo_utils.py</verify>
  <done>Tag priority is inverted when thumbnails_only is True.</done>
</task>

<task type="auto">
  <name>Add Post-Extraction Resize</name>
  <files>core/photo_utils.py</files>
  <action>
    After extracting the embedded preview, check its dimensions.
    If the long edge exceeds 1000px, resize it to max 1000px using PIL:
    ```python
    from PIL import Image
    img = Image.open(output_path)
    if max(img.size) > 1000:
        img.thumbnail((1000, 1000), Image.Resampling.LANCZOS)
        img.save(output_path, quality=85, optimize=True)
    ```
  </action>
  <verify>python -c "from core.photo_utils import extract_preview; print('OK')"</verify>
  <done>Extracted previews are resized if larger than 1000px.</done>
</task>

## Success Criteria
- [ ] `ls -lh output/thumbs/` shows files averaging 100-300KB
- [ ] No preview file exceeds 500KB
