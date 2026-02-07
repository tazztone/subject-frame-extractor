# STATE.md — Project Memory

> **Last Updated**: 2026-02-07T16:05:00+01:00
> **Session Status**: Active

## Current Position
- **Milestone**: v4.0-cli-first
- **Phase**: 0 (Triage - UI Blocker)
- **Status**: In Progress

## Last Action
Strategic refinement completed. SPEC.md updated with CLI-first vision:
- P0: Fix Tab 1→2 UI blocker, formalize CLI
- P1: Implement caching/fingerprinting for fast re-runs
- P2: Photo Mode MVP (RAW preview extraction, XMP export)

User began fixing the Tab 1→2 blocker by updating `_run_pipeline` and `_on_extraction_success` in `app_ui.py`:
- Fixed string-key to component-object mapping in pipeline yields
- Added explicit visibility updates for Subject tab components

## Next Steps
1. Verify Tab 1→2 fix by running the app and completing extraction
2. If fixed, commit and mark Phase 0 complete
3. Proceed to Phase 1 (CLI Foundation)

## Blockers
- None currently

## Context
- Operator docs already exist at `docs/HOW_TO_ADD_OPERATOR.md`
- ffmpeg confirmed capable of extracting embedded JPEGs from RAW files
- Trade-off accepted: Photo Mode is MVP only (no compare view, burst grouping)