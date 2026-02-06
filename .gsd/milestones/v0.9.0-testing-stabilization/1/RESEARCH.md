---
phase: 1
level: 2
researched_at: 2026-02-06
---

# Phase 1 Research: Foundation (Ingest & Interop)

## Questions Investigated
1. How to extract embedded JPEG previews from RAW files without demosaicing?
2. How to read/write XMP sidecar files for ratings and color labels?
3. Can Gradio's Gallery component handle 500+ images with acceptable performance?

## Findings

### 1. RAW Preview Extraction (rawpy)

**What was learned:**
- `rawpy` provides `extract_thumb()` method that returns embedded JPEG/BITMAP preview
- Returns JPEG data directly (no demosaicing needed) for most cameras
- Handles two formats: `ThumbFormat.JPEG` (bytes) and `ThumbFormat.BITMAP` (numpy array)
- Raises `LibRawNoThumbnailError` if no thumbnail exists

**Code Pattern:**
```python
import rawpy

with rawpy.imread(path) as raw:
    thumb = raw.extract_thumb()
if thumb.format == rawpy.ThumbFormat.JPEG:
    with open('thumb.jpeg', 'wb') as f:
        f.write(thumb.data)  # Already JPEG bytes
elif thumb.format == rawpy.ThumbFormat.BITMAP:
    imageio.imwrite('thumb.jpeg', thumb.data)  # RGB numpy array
```

**Recommendation:** Use rawpy for embedded preview extraction.

**Source:** https://github.com/letmaik/rawpy

---

### 2. XMP Sidecar Handling (pyexiv2)

**What was learned:**
- `pyexiv2` is a Python binding to the `exiv2` C++ library
- Supports reading/writing EXIF, IPTC, and **XMP** metadata
- Can handle XMP embedded in images AND external `.xmp` sidecar files
- BSD-licensed (commercial safe)

**Key XMP Fields for Culling:**
| XMP Key | Purpose | Type |
|---------|---------|------|
| `Xmp.xmp.Rating` | Star rating (0-5) | Integer |
| `Xmp.xmp.Label` | Color label | String |

**Recommendation:** Use pyexiv2 for XMP sidecar R/W.

**Source:** https://pyexiv2.readthedocs.io

---

### 3. Gradio Gallery Performance

**What was learned:**
- Gradio Gallery renders **all images in DOM** at once (no native lazy loading)
- No built-in pagination—must be implemented manually
- `allow_preview=True` enables modal zoom

**Performance Implications:**
| Image Count | Expected Behavior |
|-------------|-------------------|
| <100 | Smooth |
| 100-500 | Usable with load time |
| 500+ | Sluggish, needs pagination |

**Recommendation:** Target 100 images per page with pagination controls.

---

## Decisions Made

| Decision | Choice | Rationale |
|----------|--------|-----------|
| RAW preview library | `rawpy` | MIT-like license, direct JPEG extraction |
| XMP library | `pyexiv2` | BSD license, full XMP support |
| Gallery pagination | Manual (100/page) | Gradio has no native lazy loading |

## Dependencies to Add

| Package | Version | Purpose | License |
|---------|---------|---------|---------|
| `rawpy` | >=0.18 | RAW embedded preview extraction | MIT-like |
| `pyexiv2` | >=2.8 | XMP sidecar read/write | BSD |

## Risks

| Risk | Mitigation |
|------|------------|
| Small embedded previews (<1600px) | Fallback to `rawpy.postprocess()` |
| pyexiv2 requires libexiv2 | Document install requirement |

## Ready for Planning
- [x] Questions answered
- [x] Approach selected
- [x] Dependencies identified
