# JOURNAL

## 2026-02-02: Project Kickoff & Initialization
- Completed brownfield mapping.
- Identified 9 major test failures/errors related to Database refactor.
- Established a stabilization-focused roadmap.
- Initialized GSD environment.

## 2026-02-02 (Session 2): Stabilization & E2E Verification
- **Stabilization**: Resolved critical bugs in `SAM3Wrapper` weight loading by switching back to `.pt` files and submodule-native loading. Fixed `ValueError` and `AttributeError` in frame quality analysis.
- **Resource Management**: Verified that the memory watchdog correctly triggers and cleans up models during high-load periods.
- **E2E Success**: Verified the entire application workflow via `verification/e2e_run.py`. Confirmed that masks are generated correctly and metadata is accurately stored in SQLite.
- **Dependency Hygiene**: Fixed `onnx` and `sam3` version conflicts via `pyproject.toml` updates.
- **Status**: The project is now stable, resource-safe, and passes comprehensive E2E verification.

## 2026-02-03: Linux Support Enhancement
- **UX/Ease of Use**: Created `scripts/linux_run_app.sh` to mirror the functionality of the Windows `.bat` script.
- **Environment Optimization**: Configured the script to use `uv run`, ensuring that the correct virtual environment and project settings are always applied without manual activation.
- **Status**: Improved accessibility for Linux-based developers and users.

## 2026-02-04: Critical SAM3 Tracking Fix & Milestone Completion
- **The Problem**: Discovered that SAM3 grounding mode (used for box prompts) was losing tracking immediately after the seed frame, resulting in only a 1.3% mask yield.
- **The Fix**: 
    1. Unlocked tracking by providing a default text hint (`"person"`) in the `add_prompt` call. Grounding mode requires a text prompt to enable `allow_new_detections` on non-prompted frames.
    2. Increased lowres video resolution from 240p to 360p to provide better tracking features.
    3. Optimized `MaskPropagator` to use a single bidirectional pass.
- **Verification**: Integration tests now pass with **100.0% mask yield** (149/149 frames) on real video data.
- **Status**: Milestone `v2.1-stabilized` is now fully complete and verified.

## 2026-02-06: UI Test Stabilization (Phase 4.1)
- **Objective**: Fix persistent timeouts and failures in E2E UI tests (`test_full_user_flow`).
- **Accomplished**:
    - **Test Robustness**: Implemented robust "Wait & Retry" (tab switching) strategy in `tests/ui/test_app_flow.py` to handle Gradio's lazy rendering, fixing timeouts in Steps 2-5.
    - **Bug Fix**: Identified and fixed a critical bug in `ui/app_ui.py` where propagation/analysis button handlers were returning generators (lambda issue), causing silent failures.
    - **Mock Verification**: Confirmed `tests/mock_app.py` correctly populates state to enable UI buttons.
- **Verification**: `test_full_user_flow` now passes 100% consistently.
- **Status**: Phase 4.1 UI Debugging Complete. Proceeding to Documentation/Verification.

## Session: 2026-02-06 13:58

### Objective
Incorporate best ideas from FocusCull PRD into `subject-frame-extractor`, specifically a dedicated "Photo Mode" for culling RAW/JPEG files.

### Accomplished
- **Milestone Defined**: Created `v1.0-photo-mode` milestone in `ROADMAP.md` with 5 clear phases.
- **Phase 1 Research**:
    - Identified `rawpy` for 10x-100x faster preview extraction.
    - Selected `pyexiv2` for industry-standard XMP sidecar R/W.
    - Solved Gradio Gallery scaling via manual pagination (100/page).
- **Documentation**: Generated `.gsd/phases/1/RESEARCH.md`.

### Verification
- [x] Research documented and approach selected.
- [ ] Implementation plan for Phase 1 pending.

### Paused Because
Session end.

### Handoff Notes
Ready to start Phase 1. Planning involves designing the `PhotoIngestPipeline` and the `XMPHandler`. Dependencies (`rawpy`, `pyexiv2`) need to be added to `pyproject.toml`.
