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

### RAW Extraction via FFmpeg
FFmpeg can extract embedded I-frames (thumbnails/previews) from many RAW formats.
Ref: `ffmpeg -i input.CR2` typically exposes a video stream (mjpeg) for the embedded preview.

**Command:**
```bash
ffmpeg -i input.CR2 -vf "select='eq(pict_type,I)'" -vsync vfr -q:v 2 output.jpg
```
*Note: If multiple streams exist, mapping the largest MJPEG stream is preferred.*

**Recommendation:**
Use `ExtractionPipeline`'s existing FFmpeg wrapper but adapt it for single-file processing or batch processing of a folder of RAWs.

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
| RAW Extraction | FFmpeg | Requirement in Roadmap; avoids new `rawpy` dependency for now. |
| XMP Library | `xml.etree` | Lightweight, sufficient for simple Ratings/Labels. |
| UI Component | `gr.Gallery` | reused with pagination logic from SceneHandler. |
| Pipeline | New `PhotoPipeline`? | No, adapt `ExtractionPipeline` to handle directory inputs and treat images as "frames". |

## Patterns to Follow
- **Pagination:** Always paginate galleries (max 50-100 items per page) to prevent browser crashing.
- **State Mapping:** Use an `index_map` in `ApplicationState` to translate UI indices to Data indices.
- **Directory as Video:** Treat a folder of images conceptually similar to a video timeline for the `ApplicationState`.

## Dependencies Identified
- None new required. (FFmpeg is already a system dep, standard lib for XML).

## Risks
- **FFmpeg Speed:** Extracting 1000 RAWs might be slow if shelling out 1000 times.
  - *Mitigation:* Use `find ... | xargs ffmpeg` or python `subprocess` with parallelism.
- **Gradio Latency:** Sending large base64 strings for gallery.
  - *Mitigation:* Use `ThumbnailManager` to serve small cached JPEGs, not full frames.

## Ready for Planning
- [x] Questions answered
- [x] Approach selected
- [x] Dependencies identified
