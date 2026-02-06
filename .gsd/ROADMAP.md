# ROADMAP

## Milestone: v0.9.0-testing-stabilization
> **Goal**: Clarify testing taxonomy, restructure test folders, and stabilize the suite structure.

## Must-Haves
- [ ] Clear folder structure (`unit`, `integration`, `e2e`).
- [ ] `TESTING.md` rewritten.
- [ ] All root-level `tests/test_*.py` files organized into subfolders.

## Phases

### Phase 1: Research & Definition
**Status**: ✅ Complete
**Objective**: Audit current verify/integrate/e2e folders and define clear boundaries (Integration vs Verification).

### Phase 2: Restructuring
**Status**: ✅ Complete
**Objective**: Move files into strict `unit/`, `integration/`, `e2e/` hierarchy and fix imports.

### Phase 3: Documentation & Verification
**Status**: ✅ Complete
**Objective**: Rewrite `TESTING.md` and ensure 100% test pass rate in new structure.

### Phase 4: Gap Closure
**Status**: ⬜ Not Started
**Objective**: Address UI test regressions and stabilization issues identified in @v0.9.0 audit.

**Tasks:**
- [ ] Synchronize UI test selectors with current Gradio labels
- [ ] Isolate slow accessibility/UX audits with `@pytest.mark.slow`
- [ ] Update `mock_app.py` for full `ApplicationState` compatibility
- [ ] Verify full unit + fast-ui suite pass with zero failures


---

## Milestone: v1.0-photo-mode
> **Goal**: Integrate dedicated photo culling capabilities for still images (RAW/JPEG), adapting FocusCull's cascading reject pipeline and visual explainability features.

### Must-Haves
- [ ] RAW/JPEG directory import with embedded preview extraction
- [ ] XMP sidecar read/write for Lightroom/Capture One interop
- [ ] Dedicated "Photo Culling" UI tab
- [ ] Cascading analysis pipeline (Metadata → Fast CV → Deep AI)
- [ ] Basic culling UX (ratings, traffic-light borders)

### Out of Scope (v1.0)
- RAW demosaicing (embedded preview only)
- Cloud/sync features
- Tethering
- Custom-trained aesthetic models (use existing `pyiqa` NIQE)

### Phase 1: Foundation (Ingest & Interop)
**Status**: ⬜ Not Started
**Objective**: RAW/JPEG ingest, XMP R/W, Photo tab with paginated grid.
- [ ] Add `rawpy` and `pyexiv2` dependencies
- [ ] Implement `EmbeddedPreviewExtractor` for RAW files
- [ ] Implement `XMPHandler` for ratings/labels read/write
- [ ] Create "Photo Culling" tab with paginated gallery (max 500)
- [ ] Unit tests for preview extraction and XMP round-trip

### Phase 2: Technical Cull (Fast CV)
**Status**: ⬜ Not Started
**Objective**: Stage 1/2 pipeline with metadata + sharpness scoring.
- [ ] Implement Stage 1: Metadata extraction (EXIF, histogram)
- [ ] Implement Stage 2: Sharpness/blur detection (Laplacian variance)
- [ ] Add "Traffic Light" visual feedback (green/yellow/red borders)
- [ ] Implement auto-reject logic for technically unusable images
- [ ] Verification: Import 1k photos, blur auto-detected correctly

### Phase 3: Performance Spike
**Status**: ⬜ Not Started
**Objective**: Evaluate Gradio limits, implement viewport-priority queue.
- [ ] Benchmark Gradio Gallery with 2k+ images
- [ ] Implement viewport-priority analysis queue (visible images first)
- [ ] Document known limitations if Gradio cannot scale
- [ ] Verification: 2k images, scroll latency < 100ms

### Phase 4: Deep AI (Smart Grouping)
**Status**: ⬜ Not Started
**Objective**: Face detection, burst grouping, smart deduplication.
- [ ] Integrate MediaPipe face detection (commercial-safe)
- [ ] Implement burst grouping by timestamp + embedding similarity
- [ ] Implement "Best of Burst" auto-selection
- [ ] Add Face Focus heatmap overlay
- [ ] Verification: Burst of 10 photos grouped correctly

### Phase 5: UX Polish
**Status**: ⬜ Not Started
**Objective**: Compare view, bulk export, keyboard shortcuts.
- [ ] Implement "Compare View" for burst comparison
- [ ] Add bulk rating/export actions
- [ ] Add keyboard shortcuts (best-effort in Gradio)
- [ ] E2E verification: Full cull workflow on sample set

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
