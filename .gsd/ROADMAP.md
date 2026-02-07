# ROADMAP

> **Current Milestone**: v4.0-cli-first

## Milestone: v4.0-cli-first
> **Goal**: Stabilize the application, formalize a CLI for automated testing, improve performance via caching, and add minimal Photo Mode support.

### Must-Haves
- [x] Fix Tab 1 → Tab 2 UI blocker
- [x] Create `cli.py` with `extract`, `analyze`, `filter` subcommands
- [x] Implement run fingerprinting for cache hits
- [x] RAW/JPEG import via ffmpeg embedded preview extraction
- [x] XMP sidecar export for Lightroom

### Phases

#### Phase 0: Triage (UI Blocker)
**Status**: ✅ Complete
**Objective**: Fix the Tab 1 → Tab 2 visibility bug preventing full UI workflow.
- [ ] Investigate `ApplicationState` propagation after extraction
- [ ] Fix component visibility updates in `_on_extraction_success`
- [ ] Manual verification: Complete full UI cycle on sample video

#### Phase 1: CLI Foundation
**Status**: ✅ Complete
**Objective**: Create a formal CLI for scriptable, headless operation.
- [ ] Create `cli.py` with Click or argparse
- [ ] Implement `extract` subcommand (wraps `execute_extraction`)
- [ ] Implement `analyze` subcommand (wraps pre-analysis + propagation + analysis)
- [ ] Implement `filter` subcommand (applies filter presets, exports)
- [ ] Verification: `python cli.py extract --video sample.mp4 --output ./out` exits 0

#### Phase 2: Caching & Idempotency
**Status**: ✅ Complete
**Objective**: Make re-runs fast by detecting already-completed work.
- [ ] Implement `run_state.json` to track completed stages
- [ ] Add `--resume` flag to CLI
- [ ] Add fingerprinting (hash video path + key settings)
- [ ] Verification: Second run on same video takes <10% of first run time

#### Phase 3: Photo Mode (MVP)
**Status**: ✅ Complete
**Objective**: Minimal viable photo culling support.
- [x] Implement RAW preview extraction via ffmpeg (Changed to ExifTool)
- [x] Treat extracted previews as "seed frames" for subject detection
- [x] Implement XMP sidecar export (rating, label)
- [x] Add "Photo Mode" preset with suitable IQA settings
- [x] Verification: Import 50 RAWs, export XMPs, verify in Lightroom

#### Phase 4: Polish & Verification
**Status**: ✅ Complete
**Objective**: Finalize, document, and verify.
- [x] Full E2E test via CLI
- [x] Full E2E test via UI
- [x] Update README with CLI usage
- [x] Commit and tag release

#### Phase 5: Post-Release Polish (Gap Closure)
**Status**: ✅ Complete
**Objective**: Stabilize and unify the logging architecture for v4.0.0.
- [x] Standardize Python Logging ([5-2-PLAN.md](file:///.gsd/phases/5/5-2-PLAN.md))
- [ ] Tag Release v4.0.0

---


## Milestone: v3.0-extensibility (ARCHIVED)
> **Goal**: Establish clean state management, then refactor pipelines into an extensible "Operator" pattern.

### Must-Haves
- [x] Migrate `AppUI` fully to `ApplicationState` (remove legacy states)
- [x] Define `Operator` Protocol/ABC
- [x] Refactor `AnalysisPipeline` to use operators
- [x] Migrate 3+ existing metrics to operators
- [x] "How to add an Operator" documentation

### Phases
See [.gsd/milestones/v3.0-extensibility-SUMMARY.md](.gsd/milestones/v3.0-extensibility-SUMMARY.md)

---

---

## Milestone: v0.9.1-stabilization (ARCHIVED)
> **Goal**: Final stabilization and UI test cleanup.

### Must-Haves
- [x] UI Test parity with Gradio updates
- [x] Robust mock application state

### Phases

### Phase 1: Planning & Final Fixes
**Status**: ✅ Complete
**Objective**: Address UI test regressions and stabilization issues identified in @v0.9.0 audit.

**Tasks:**
- [x] Synchronize UI test selectors with current Gradio labels
- [x] Isolate slow accessibility/UX audits with `@pytest.mark.slow`
- [x] Update `mock_app.py` for full `ApplicationState` compatibility
- [x] Verify full unit + fast-ui suite pass with zero failures


---

---

## Milestone: v2.1-stabilized (ARCHIVED)

### Phase 1: Test Suite Stabilization
- [x] Fix all unit and integration test failures.
- [x] Implement automated Quality Verification script.
- [x] Ensure 100% test pass rate.

### Phase 2: Pipeline Refactoring & Optimization
- [x] Decompose monolithic pipelines into testable components.
- [x] Implement memory watchdog for heavy models.
- [x] Optimize mask propagation using downscaled video.

### Phase 3: Linux Support & UX
- [x] Create native Linux run script (`scripts/linux_run_app.sh`).
- [x] Configure `uv` for seamless environment management.

### Phase 4: Release Verification (FINAL)
- [x] Resolve SAM3 tracking loss in grounding mode.
- [x] Verify 100% mask yield in real-world E2E workflow.
- [x] Complete release documentation.

---

## Deferred Milestones

### Milestone: v1.0-photo-mode
> **Goal**: Integrate dedicated photo culling capabilities for still images (RAW/JPEG), adapting FocusCull's cascading reject pipeline and visual explainability features.

#### Must-Haves
- [ ] RAW/JPEG directory import with embedded preview extraction
- [ ] XMP sidecar read/write for Lightroom/Capture One interop
- [ ] Dedicated "Photo Culling" UI tab
- [ ] Cascading analysis pipeline (Metadata → Fast CV → Deep AI)
- [ ] Basic culling UX (ratings, traffic-light borders)

#### Out of Scope (v1.0)
- RAW demosaicing (embedded preview only)
- Cloud/sync features
- Tethering
- Custom-trained aesthetic models (use existing `pyiqa` NIQE)

#### Phase 1: Foundation (Ingest & Interop)
**Status**: ⬜ Not Started
**Objective**: RAW/JPEG ingest, XMP R/W, Photo tab with paginated grid.
- [ ] Add `rawpy` and `pyexiv2` dependencies
- [ ] Implement `EmbeddedPreviewExtractor` for RAW files
- [ ] Implement `XMPHandler` for ratings/labels read/write
- [ ] Create "Photo Culling" tab with paginated gallery (max 500)
- [ ] Unit tests for preview extraction and XMP round-trip

#### Phase 2: Technical Cull (Fast CV)
**Status**: ⬜ Not Started
**Objective**: Stage 1/2 pipeline with metadata + sharpness scoring.
- [ ] Implement Stage 1: Metadata extraction (EXIF, histogram)
- [ ] Implement Stage 2: Sharpness/blur detection (Laplacian variance)
- [ ] Add "Traffic Light" visual feedback (green/yellow/red borders)
- [ ] Implement auto-reject logic for technically unusable images
- [ ] Verification: Import 1k photos, blur auto-detected correctly

#### Phase 3: Performance Spike
**Status**: ⬜ Not Started
**Objective**: Evaluate Gradio limits, implement viewport-priority queue.
- [ ] Benchmark Gradio Gallery with 2k+ images
- [ ] Implement viewport-priority analysis queue (visible images first)
- [ ] Document known limitations if Gradio cannot scale
- [ ] Verification: 2k images, scroll latency < 100ms

#### Phase 4: Deep AI (Smart Grouping)
**Status**: ⬜ Not Started
**Objective**: Face detection, burst grouping, smart deduplication.
- [ ] Integrate MediaPipe face detection (commercial-safe)
- [ ] Implement burst grouping by timestamp + embedding similarity
- [ ] Implement "Best of Burst" auto-selection
- [ ] Add Face Focus heatmap overlay
- [ ] Verification: Burst of 10 photos grouped correctly

#### Phase 5: UX Polish
**Status**: ⬜ Not Started
**Objective**: Compare view, bulk export, keyboard shortcuts.
- [ ] Implement "Compare View" for burst comparison
- [ ] Add bulk rating/export actions
- [ ] Add keyboard shortcuts (best-effort in Gradio)
- [ ] E2E verification: Full cull workflow on sample set
