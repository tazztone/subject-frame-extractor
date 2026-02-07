# STATE.md — Project Memory

> **Last Updated**: 2026-02-07T18:55:00+01:00
> **Session Status**: Active (resumed 2026-02-07T18:52:56+01:00)

## Current Position
- **Milestone**: v4.0-cli-first
- **Phase**: 5 (Post-Release Polish & CLI Filter)
- **Status**: Executing

## Last Session Summary
Audited Milestone v4.0-cli-first.
- Identified missing `filter` subcommand in `cli.py`.
- Updated `SPEC.md` to reflect `ExifTool` pivot for RAW previews.
- Confirmed discrepancies between SUMMARY and implementation.
- Planned immediate fix for CLI Filter and Logging.

## In-Progress Work
- Implementing `filter` subcommand in `cli.py`.
- Refactoring `AppLogger` to `dictConfig`.
- Seeding stability fixes.
- Files modified (this session): `.gsd/SPEC.md`, `.gsd/STATE.md`.

## Blockers
- None.

## Context Dump

### Decisions Made
- **Standard Logging**: Refactor `AppLogger` to use `dictConfig` to allow different handlers (File, Console, UIQueue) to be toggled per entry point (CLI vs. UI).
- **Audit Hygiene**: Removed specific "bug" mentions of seeding mismatch from the audit report to keep it clean, opting for "Seeding Stability" in internal plans instead.

### Approaches Tried
- **Reproduction**: Created `reproduce_seeding_bug.py` which confirmed `save_scene_seeds` overwrites correctly, narrowing the issue to state cleanup or resume logic.

### Files of Interest
- `core/logger.py`: Needs refactoring for standard `dictConfig`.
- `cli.py`: Needs logging config update to disable console logs from `AppLogger`.
- `core/pipelines.py`: Needs duplicate detection in `_load_scenes`.

## Next Steps
1. Execute Phase 5: Implement standardized logging.
2. Execute Phase 5: Implement seeding stability fixes.
3. Tag Release v4.0.0.