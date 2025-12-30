# Changelog

## 2025-12-27
### UI & UX Overhaul
* Significantly improved UI visual hierarchy, clarity, and accessibility in `app_ui.py`.
* Refactored UI architecture for better maintainability.

* Migrate project configuration to `pyproject.toml`.
* Fix SAM3 mask generation error ("Unexpected error in mask generation").

## 2025-12-26
### SAM3 High-Level API Integration
* Refactored `SAM3Wrapper` to use the robust `Sam3VideoPredictor` high-level API.
* Fixed object detection in `SeedSelector` using `Sam3Processor`.

* Update documentation generation scripts.

## 2025-12-25
* Optimize SAM3 integration performance and cleanup.
* Fix "Find People" tests and integration suite issues.
* Fix queue drain and system logs visibility bugs.

## 2025-12-23
### Comprehensive Testing Framework
* Implement comprehensive test coverage framework (Unit & E2E).
* Add detailed `TESTING.md` documentation.

## 2025-12-20
### Legacy Cleanup & SAM3 Rewrite
* **Removed legacy dependencies**: YOLO, DAM4SAM, Grounded-SAM-2.
* Rewrote SAM3 integration to strictly follow official API patterns.
* Update Setup Scripts for full SAM3 and Jules compatibility.

## 2025-12-17
* Add extensive unit and integration tests for core functionality.
* Fix SAM3 and Triton imports for Windows compatibility.
* Increase maximum range for Sharpness and Edge Strength sliders.

## 2025-12-16
### Modular Architecture Refactor
* **Major Refactor**: Split monolithic `app.py` into modular `core/` package architecture.
* Split monolithic `AppUI` class into specialized handlers (`extraction`, `analysis`, `filtering`).
* Refactored `scene_utils` into focused modules (`detection`, `mask_propagator`, `seed_selector`).
* Fixed critical architecture issues including circular imports.

## 2025-12-13
### Vectorized Performance & Security
* **Performance**: Vectorized LPIPS deduplication logic for faster processing.
* **Security**: Add authentication token support for secure model downloads.

## 2025-09-17
### SAM2 Integration
* **New Model**: Integrated **SAM2** for advanced segmentation and mask propagation.
* Add `scenedetect` for automatic scene boundary detection.
* Implement real-time FFmpeg progress monitoring.
* Enhance mask propagation with re-identification and identity ambiguity checks.

## 2025-09-15
### Face Analysis Upgrade
* **Major Upgrade**: Replaced DeepFace with **InsightFace** for superior face analysis.
* Add live analysis statistics reporting and progress bars.

## 2025-06-29
### Configuration Architecture
* Refactor configuration into a dedicated, type-safe `Config` class.

## 2025-06-26
### Application Foundation
* Add initial `app.py` structure.
* Centralize configuration with `CONFIG` dictionary.
* Implement frame export and video download enhancements.

## 2025-06-25
### Performance Optimization
* Optimize frame quality metrics with Numba.
* Implement batch processing for face similarity.
* Enhance FFmpeg error handling and GPU synchronization.

## 2025-06-24
* Implement producer-consumer pattern for frame extraction.
* Replace blur score with improved sharpness metric.

## 2025-06-23
* Refactor logging to use structured JSON messages.
* Optimize Laplacian variance calculation by caching.

## 2025-06-22
* Refactor GPU processing to use PyTorch.

## 2025-06-20
* Add live frame preview functionality during extraction.
* Add process management for extraction with stop functionality.

## 2025-06-19
* Add support for video upload and face image upload.
* Update extraction method default to keyframes and enable PNG output.

## 2025-06-13
### DeepFace Integration
* Add face similarity filter using **DeepFace** (later replaced by InsightFace).
* Refactor frame extraction to use FFmpeg.

## 2024-09-07
* Initial commit: Add YouTube screenshot extractor script.
