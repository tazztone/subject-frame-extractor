# SPEC.md — Project Specification

> **Status**: FINALIZED

## Vision
A robust, automated Subject Frame Extraction platform for Video and Photo datasets.
The system prioritizes **testability** (via a headless CLI) and **composability** (via a plugin architecture for metrics).

## Core Principles
1. **CLI-First**: The UI is a presentation layer atop a fully scriptable CLI.
2. **Idempotent**: Re-running the same job on the same input is fast (skips completed work).
3. **Extensible**: New metrics are added as drop-in "Metric Plugins" (the existing Operator pattern).

---

## Goals (Prioritized)

### P0: Stabilization (Must Fix Before Anything Else)
1. **Repair UI State Flow**: Fix the Tab 1 → Tab 2 blocker.
   - Root cause: Component visibility not updating after extraction success.
   - Verification: Can complete a full Extraction → Pre-Analysis → Propagation → Analysis → Filter cycle via the UI.
2. **Formalize CLI**: Create `cli.py` with subcommands: `extract`, `analyze`, `filter`.
   - Verification: `python cli.py extract --video path/to/video.mp4 --output path/to/output` runs to completion.

### P1: Performance & Caching
3. **Implement Run Fingerprinting**: Hash inputs (video path, settings) to detect re-runs.
   - If fingerprint matches existing output, skip to results.
   - Verification: Second run on same video takes <10% of original time.
4. **Consolidate Progress Tracking**: Use a single `run_state.json` file to track which stages are complete.
   - Enables `--resume` flag for interrupted runs.

### P2: Photo Mode (MVP)
5. **RAW/JPEG Import**: Use `ExifTool` to extract high-resolution embedded JPEGs from RAW files.
   - Treat each photo as a "seed frame".
   - Verification: Import a folder of 50 RAW files, extract previews.
6. **XMP Sidecar Export**: Write Lightroom-compatible XMP files with rating/label.
   - Verification: Imported into Lightroom, ratings visible.

---

## Non-Goals
- Replacing Gradio (we fix the existing UI).
- Full RAW demosaicing (we extract embedded previews only).
- Cloud integration or multi-user features.
- Renaming "Operator" → "Plugin" in code (cosmetic, defer).

---

## Success Criteria (Measurable)
| # | Criterion | Verification Method |
|---|-----------|---------------------|
| 1 | UI Unblocked | Manual: Complete full UI cycle on sample video. |
| 2 | CLI Works | `python cli.py extract --video sample.mp4 --output ./out` exits 0. |
| 3 | Re-run Fast | Timed: `time python cli.py analyze ...` on same input, <10% of first run. |
| 4 | Photo Import | CLI: `python cli.py import-photos --dir ./raws --output ./out` extracts previews. |
| 5 | XMP Export | Lightroom: Open sidecar, see rating. |
