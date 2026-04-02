# Changelog

All notable changes to the Subject Frame Extractor project.

## [1.7.0] - 2026-04-02
### Added
- **Multi-Class Subject Architecture**: Transitioned the entire detection and analysis pipeline from "person-only" to a flexible, COCO-compliant "Subject" system supporting all 80 YOLO classes.
- **Dynamic Class Selection**: Added a "Subject Class" dropdown to the UI, allowing users to target specific object types (e.g., cars, dogs, bicycles) for tracking.
- **YOLO26 Family Support**: Integrated the full YOLO26 model family (n, s, m, l, x) into the subject detection stage.

### Changed
- **Nomenclature Synchronization**: Standardized on "Subject" terminology across the entire codebase, including Pydantic models, managers, events, and Gradio UI components.
- **Core Refactor**: Renamed the `PersonDetector` class to `SubjectDetector` and migrated its logic to `core/managers/subject_detector.py`.
- **UI Labeling**: Updated all "Person" references in the interface to "Subject" for architectural consistency.
- **Test Infrastructure Sync**: Updated all E2E Playwright locators and regenerated visual regression baselines to align with the new nomenclature and layout.

## [1.6.0] - 2026-03-31
### Added
- **Performance Optimization Suite**: Comprehensive audit and refactor of core processing pipelines.
- **Parallel Analysis**: Narrowed locking strategy in `AnalysisPipeline` to enable true multi-threaded execution of image quality operators.
- **Batch Pre-loading**: Implemented batch-level image and mask pre-loading to minimize disk I/O and decoding overhead during analysis.
- **Parallel Post-processing**: Added `ThreadPoolExecutor` to mask propagation for concurrent post-processing of results.

### Changed
- **Orchestrator Refactor**: `execute_analysis_orchestrator` now initializes models once and shares them across all stages (Pre-Analysis, Propagation, Analysis), eliminating redundant model loads.
- **Optimized Defaults**: Increased `analysis_default_batch_size` to 50 and database `batch_size` to 100 for better throughput.
- **Resource Management**: Reduced blocking `torch.cuda.empty_cache()` calls in `SAM3Wrapper` to improve frame-to-frame transition speed.
- **Test Infrastructure**: Optimized UI test setup by reducing polling intervals and mock extraction delays, resulting in a >75% reduction in total test suite duration.

## [1.5.0] - 2026-02-07
### Added
- **`/refine` Workflow** — Strategic Course Correction and Roadmap Reconciliation.
- **Linux-First Optimization** — Complete removal of PowerShell dependencies.

### Changed
- **Workflows:** All 26 workflows converted to pure Bash.
- **Skills:** All 8 skills updated with Linux-specific examples.
- **Repo Structure:** Removed `scripts/*.ps1` files.
- **Documentation:** Updated README.md and quick-reference.md for Linux users.

## [2026-02-03] - SAM3 API Alignment
### Changed
- **Unified Propagation**: Refactored `SAM3Wrapper` and `MaskPropagator` to use the official `propagation_direction="both"` API.
- Optimized propagation efficiency by eliminating redundant forward/backward passes.
- Updated `README.md` documentation to reflect the new `propagate` signature.

## [1.4.0] - 2026-01-17
### Added
- **Template Parity** — 8 new templates (22 total): `architecture.md`, `decisions.md`, `journal.md`, `stack.md`, `phase-summary.md`, `sprint.md`, `todo.md`, `spec.md`.
- `validate-templates.sh` — template validation scripts.
- `validate-all` now includes template validation.

## [1.3.0] - 2026-01-17
### Added
- **Validation Scripts** — expanded testing infrastructure: `validate-skills.sh`, `validate-all.sh`.
- **VERSION file** — single source of truth for version.
- `/help` now displays current version.

### Changed
- README.md updated with Testing section.

## [1.2.0] - 2026-01-17
### Added
- **Cross-Platform Support** — All 16 workflow files now have Bash equivalents.
- `/web-search` — Search the web for technical research.

### Changed
- README.md updated with Getting Started guidance.
- README.md added Cross-Platform Support section.
- Git commands in workflows use `bash` syntax.

## [1.1.0] - 2026-01-17
### Added
- **Template Parity** — 14 templates aligned with original repository (`DEBUG.md`, `UAT.md`, `discovery.md`, etc.).
- **Examples** — `.gsd/examples/` directory.
- `/add-todo` — Quick capture workflow.
- `/check-todos` — List pending items workflow.
- `/whats-new` — Show recent changes.

### Changed
- Workflows now have "Related" sections for discoverability.
- Cross-linked workflows and skills.

## [1.0.0] - 2026-01-17
### Added
- **Core Workflows (21)**: `/map`, `/plan`, `/execute`, `/verify`, `/debug`, `/new-project`, etc.
- **Skills (8)**: `planner`, `executor`, `verifier`, `debugger`, `codebase-mapper`, `plan-checker`, `context-health-monitor`, `empirical-validation`.
- **Methodology Documentation**: README.md with full methodology explanation.

## [2025-12-27] - UI & UX Overhaul
### Changed
- Significantly improved UI visual hierarchy, clarity, and accessibility in `app_ui.py`.
- Refactored UI architecture for better maintainability.
- Migrate project configuration to `pyproject.toml`.
- Fix SAM3 mask generation error ("Unexpected error in mask generation").

## [2025-12-26] - SAM3 High-Level API Integration
### Changed
- Refactored `SAM3Wrapper` to use the robust `Sam3VideoPredictor` high-level API.
- Fixed object detection in `SeedSelector` using `Sam3Processor`.
- Updated documentation generation scripts.

## [2025-12-25]
- Optimize SAM3 integration performance and cleanup.
- Fix "Find People" tests and integration suite issues.
- Fix queue drain and system logs visibility bugs.

## [2025-12-23] - Comprehensive Testing Framework
### Added
- Implement comprehensive test coverage framework (Unit & E2E).
- Add detailed `TESTING.md` documentation.

## [2025-12-20] - Legacy Cleanup & SAM3 Rewrite
### Changed
- **Removed legacy dependencies**: YOLO, DAM4SAM, Grounded-SAM-2.
- Rewrote SAM3 integration to strictly follow official API patterns.
- Update Setup Scripts for full SAM3 and Jules compatibility.

## [2025-12-16] - Modular Architecture Refactor
### Changed
- **Major Refactor**: Split monolithic `app.py` into modular `core/` package architecture.
- Split monolithic `AppUI` class into specialized handlers (`extraction`, `analysis`, `filtering`).
- Refactored `scene_utils` into focused modules.
- Fixed critical architecture issues including circular imports.

## [2025-09-17] - SAM2 Integration
### Added
- **New Model**: Integrated **SAM2** for advanced segmentation and mask propagation.
- Add `scenedetect` for automatic scene boundary detection.
- Implement real-time FFmpeg progress monitoring.

## [2025-09-15] - Face Analysis Upgrade
### Changed
- **Major Upgrade**: Replaced DeepFace with **InsightFace** for superior face analysis.
- Add live analysis statistics reporting and progress bars.

## [2025-06-13] - Initial Foundation
- Integrated **DeepFace** for face similarity (later replaced by InsightFace).
- Refactor frame extraction to use FFmpeg.
- Initial commit: YouTube screenshot extractor script (2024-09-07).

---
### Attribution
Adapted from [glittercowboy/get-shit-done](https://github.com/glittercowboy/get-shit-done) for Google Antigravity.
