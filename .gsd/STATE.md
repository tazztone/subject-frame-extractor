# STATE

## Current Position
- **Milestone**: v1.0-photo-mode
- **Phase**: Phase 1: Foundation (Ingest & Interop)
- **Task**: Research complete, ready for detailed planning (/plan 1)
- **Status**: Paused at 2026-02-06 13:59

## Last Session Summary
Focused on mapping and planning the **Photo Mode** integration inspired by FocusCull.
- Defined the `v1.0-photo-mode` milestone with 5 phases.
- Completed Phase 1 Research (rawpy, pyexiv2, Gradio constraints).
- Documented findings in `.gsd/phases/1/RESEARCH.md`.
- Note: User added a placeholder `[Next Milestone]` in `ROADMAP.md` which may require clarification.

## In-Progress Work
- None (Planning/Research finalized and committed).
- Tests status: Video tests stable; Photo unit tests planned for Phase 1.

## Context Dump
### Decisions Made
- **Preview Path**: Use `rawpy`'s `extract_thumb()` to avoid heavy RAW decoding.
- **Metadata Path**: Use `pyexiv2` for `.xmp` sidecar compatibility.
- **UI Scaling**: Implement manual 100-image pagination for Gradio Gallery.

### Files of Interest
- `.gsd/ROADMAP.md`: Core plan for Photo Mode.
- `.gsd/phases/1/RESEARCH.md`: Detailed findings on libs.

## Next Steps
1. `/plan 1` — Create Phase 1 execution plans (Ingest & XMP Handlers).
2. Install new dependencies: `rawpy`, `pyexiv2`.
3. Create `PhotoIngestPipeline` in `core/pipelines.py`.