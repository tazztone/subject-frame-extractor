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

---

## Milestone: v2.1-stabilized

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
