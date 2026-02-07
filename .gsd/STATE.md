# STATE.md — Project Memory

> **Last Updated**: 2026-02-07T18:55:00+01:00
> **Session Status**: Active (resumed 2026-02-07T18:52:56+01:00)

## Current Position
- **Milestone**: v4.0-cli-first
- **Phase**: 5 (Post-Release Polish)
- **Status**: Completed

## Last Session Summary
Finalized Phase 5 Polish & UI Hotfixes.
- Implemented `filter` CLI subcommand (completing P0 SPEC goal).
- Standardized logging via `dictConfig`, silencing framework noise and unifying library outputs (SAM3, SceneDetect).
- Decoupled UI progress queue from core `AppLogger` using a custom `GradioQueueHandler`.
- Added `allowed_paths` configuration to fix `InvalidPathError` when using external drives in the UI.
- Purged all references to the disproven "Seeding Stability" bug.
- Updated `SPEC.md` and `ROADMAP.md` to reflect `ExifTool` implementation.

## In-Progress Work
- None (Phase 5 complete).
- Files modified (this session): `core/logger.py`, `cli.py`, `app.py`, `.gsd/STATE.md`, `.gsd/JOURNAL.md`, `.gsd/phases/5/LOGGING_PLAN.md`.

## Blockers
- None.

## Context Dump

### Decisions Made
- **Standard Logging**: Refactored `AppLogger` to use `dictConfig`. Specifically targeted `sam3` logger to force propagation and unified formatting.
- **UI Decoupling**: Moved UI updates to a dedicated `GradioQueueHandler`, allowing backend code to be log-agnostic.

### Approaches Tried
- **Framework Silencing**: Successfully suppressed TF/MediaPipe noise using env vars and logger level overrides.

### Files of Interest
- `core/logger.py`: Now contains the centralized logging logic.
- `cli.py`: Updated to use `setup_logging`.
- `app.py`: Updated to use `setup_logging` with the Gradio queue.

## Next Steps
1. Tag Release v4.0.0.
2. Archive Milestone v4.0-cli-first.