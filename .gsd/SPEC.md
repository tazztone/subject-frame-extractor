# SPEC.md — Project Specification

> **Status**: 

## Vision
Stabilize the Subject Frame Extractor by resolving critical regressions, refactoring core pipelines for maintainability, and implementing automated memory management to ensure a robust user experience and best practice coding standards.

## Goals
1. **Test Suite**: ensure end to end testing and automated analysis is in place for catching issues or report room for improvements early.
2. **Refactor Pipelines**: Decompose monolithic functions in `core/pipelines.py` into smaller, testable, pure-logic components.
3. **UI Stability & State**: fix or improve gradio GUI.
4. **Performance Optimizations**: certain tasks like face detection and quality metrics should be done on the extracted thumbnail images while others like propagate masks in video can be done on very low res downscaled video. investigate and improve.

## Non-Goals (Out of Scope)
- swapping SAM3 for something else.
- Rewriting the entire UI in a different framework (staying with Gradio).
- Cloud integration or multi-user features.

## Users
- Developers and researchers needing to filter the highest-quality subject-focused image datasets from a video.

## Constraints
- **Dependency**: Must respect the boundaries of the `SAM3_repo` submodule (read-only! no edits allowed). implementation should follow example code located in SAM3_repo/examples/sam3_video_predictor_example.ipynb where possible.
- **Environment**: should run on linux and windows using `uv` for package management.

## Success Criteria
- [x] **Zero Failing Tests**: All unit, smoke, and integration tests pass.
- [x] **Successful E2E Run**: Complete a full run via the GUI. ensure all outputs are as expected according to the settings used. use the downloads/example clip 720p 2x.mov and downloads/example face.png as inputs
- [x] **Clean Repository**: No legacy or redundant files remaining.
