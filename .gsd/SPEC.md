# SPEC.md — Project Specification

> **Status**: `FINALIZED`

## Vision
Stabilize and modernize the Subject Frame Extractor by resolving critical regressions, refactoring core pipelines for maintainability, and implementing automated memory management to ensure a robust user experience.

## Goals
1. **Fix Test Suite**: Resolve all 9+ failing tests and errors, primarily focusing on the `Database` API mismatch and hardcoded path issues.
2. **Refactor Pipelines**: Decompose monolithic functions in `core/pipelines.py` into smaller, testable, pure-logic components.
3. **Memory Management**: Implement an auto-unloader/watchdog in `ModelRegistry` to prevent OOM by clearing models when resource thresholds are exceeded.
4. **UI Stability & State**: Consolidate transient UI states to reduce reliance on `gr.State` and fix common GUI glitches.
5. **Repo Hygiene**: Remove orphaned files (`.bak`) and consolidate manual verification scripts.

## Non-Goals (Out of Scope)
- Adding new ML model architectures (e.g., swapping SAM3 for something else).
- Rewriting the entire UI in a different framework (staying with Gradio).
- Cloud integration or multi-user features.

## Users
- Developers and researchers needing to curate high-quality subject-focused datasets from video.
- AI engineers requiring precise frame extraction and quality filtering.

## Constraints
- **Technical**: Must maintain compatibility with Python 3.10+ and CUDA 11.x/12.x.
- **Dependency**: Must respect the boundaries of the `SAM3_repo` submodule (read-only).
- **Environment**: Must run reliably on Linux environments using `uv` for package management.

## Success Criteria
- [ ] **Zero Failing Tests**: All unit, smoke, and integration tests pass.
- [ ] **Successful E2E Run**: Complete a full processing run (Extraction -> Analysis -> Export) via the GUI using the provided `sample.mp4` and `sample.jpg`.
- [ ] **Refactored Codebase**: Monolithic pipeline functions are broken down; `ModelRegistry` handles its own memory cleanup.
- [ ] **Clean Repository**: No legacy artifacts or redundant configuration files remaining.
