# STATE.md — Project Memory

> **Last Updated**: 2026-02-07T18:00:00+01:00
> **Session Status**: Active (resumed 2026-02-07T16:51:47+01:00)

## Current Position
- **Milestone**: v4.0-cli-first
- **Phase**: 3 (Photo Mode MVP)
- **Status**: Verified

## Last Session Summary
Executed Phase 3 (Photo Mode MVP).
- Implemented `core/photo_utils.py` for RAW ingest via ExifTool.
- Added "Photo Culling" tab to UI (`ui/tabs/photo_tab.py`).
- Implemented `core/photo_scoring.py` with custom IQA weights (Sharpness, Entropy, NISQE).
- Implemented `core/xmp_writer.py` for Lightroom-compatible sidecars.
- Verified all components with `uv run python`.

## In-Progress Work
- None. (Phase 3 complete).

## Blockers
- None.

## Next Steps
1. Proceed to Phase 4 (Polish & Verification).