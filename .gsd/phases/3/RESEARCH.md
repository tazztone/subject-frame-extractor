---
phase: 3
level: 2
researched_at: 2026-02-07
---

# Phase 3 Research: Photo Mode (MVP)

## Questions Investigated
1. How to extract embedded previews from RAW files using FFmpeg?
2. How to write XMP sidecars (Ratings/Labels) without heavy dependencies?
3. How to implement a large-scale photo gallery in Gradio with selection?

## Findings

### RAW Extraction via ExifTool
User feedback and further research indicate `ExifTool` is superior for extracting embedded preview images from RAW files (CR2, NEF, ARW, etc.) as it reliably accesses the specialized binary tags without decoding the full RAW data.

**Command:**
```bash
# Extract JpgFromRaw to output file
exiftool -b -JpgFromRaw -w %d%f_preview.jpg input.CR2

# Or recursive for a directory
exiftool -b -JpgFromRaw -w %d%f_preview.jpg -ext CR2 -r ./directory
```
*Note: Some cameras use different tags like `PreviewImage`. We should probe for `JpgFromRaw`, `PreviewImage`, then `ThumbnailImage` in priority order.*

**Recommendation:**
Use `subprocess` to call `exiftool`.
- Check if `exiftool` is installed (`shutil.which('exiftool')`).
- Fallback to FFmpeg ONLY if ExifTool is missing? Or just require ExifTool for Photo Mode. 
- **Decision:** Require ExifTool for Photo Mode.

### Lightweight XMP Sidecar Generation
We explored using `xml.etree.ElementTree` to generate XMP sidecars instead of adding `pyexiv2` (which has system dependencies) or `python-xmp-toolkit` (depends on Exempi).

**Prototype Result:**
We successfully generated valid XMP with Rating and Label using standard library `xml` tools.
```xml
<x:xmpmeta xmlns:x="adobe:ns:meta/">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about=""
    xmlns:xmp="http://ns.adobe.com/xap/1.0/"
    xmp:Rating="5"
    xmp:Label="Select"/>
 </rdf:RDF>
</x:xmpmeta>
```
**Decision:** Use `xml.etree.ElementTree` for the MVP to keep dependencies low.

### Gradio Gallery & Selection
Analyzed `ui/handlers/scene_handler.py`.
- **Mechanism:** `gr.Gallery` supports `.select()` event.
- **Event Data:** `evt.index` gives the index of the selected item in the *current page*.
- **Pagination:** Essential for performance. `SceneHandler` implements pagination by manually slicing the list of items and maintaining a `index_map` to map gallery indices back to global scene IDs.
- **Pattern:** We will reuse the `SceneHandler` pagination pattern for the Photo Mode tab.

## Decisions Made
| Decision | Choice | Rationale |
|----------|--------|-----------|
| RAW Extraction | **ExifTool** | Superior reliability for embedded previews over FFmpeg. |
| XMP Library | `xml.etree` | Lightweight, sufficient for simple Ratings/Labels. |
| UI Component | `gr.Gallery` | reused with pagination logic from SceneHandler. |
| Pipeline | New `PhotoPipeline`? | No, adapt `ExtractionPipeline` to handle directory inputs and treat images as "frames". |

## Patterns to Follow
- **Pagination:** Always paginate galleries (max 50-100 items per page) to prevent browser crashing.
- **State Mapping:** Use an `index_map` in `ApplicationState` to translate UI indices to Data indices.
- **Directory as Video:** Treat a folder of images conceptually similar to a video timeline for the `ApplicationState`.

## Dependencies Identified
- **ExifTool**: Must be installed on the system (CLI).
- None new python packages.

## Risks
- **ExifTool Missing:** User might not have it installed.
  - *Mitigation:* Add check in `diagnostics` and show warning in UI.
- **Gradio Latency:** Sending large base64 strings for gallery.
  - *Mitigation:* Use `ThumbnailManager` to serve small cached JPEGs, not full frames.

## Ready for Planning
- [x] Questions answered
- [x] Approach selected
- [x] Dependencies identified
