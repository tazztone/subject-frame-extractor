# STATE

## Current Position
- **Milestone**: v3.0-extensibility
- **Phase**: 0 (State Cleanup)
- **Task**: Not started
- **Status**: Milestone planned

## Last Session Summary
- **Roadmap Cleanup**: Marked all pending stabilization tasks as complete.
- **Archive**: Moved all planned milestones to Archived status.
- **Decision**: Deferred Photo Mode was preserved for future use, but the active roadmap is now considered 100% complete per user request.

## In-Progress Work
- [x] Captured architectural inspirations from `OVERVIEW_comparison.md` into `TODO.md`.
- [x] Verified status of pre-existing todos:
    - Seeding mismatch: Investigated; logic found (3 per scene), but remains open.
    - AppUI refactor: Not done; legacy states still active.
    - Makefile: Not done.
    - Doc Monitoring: Ongoing; secret redaction implemented.
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