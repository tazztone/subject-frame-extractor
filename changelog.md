# Changelog

## 2025-12-27
* Fix empty error logging in mask generation and add E2E tests
* Refactor app_ui.py to improve UX and visual hierarchy
* Enhance UI clarity and accessibility in app_ui.py
* Improve test coverage with meaningful E2E and Unit tests
* Migrate project configuration to `pyproject.toml`
* Fix SAM3 mask generation error ("Unexpected error in mask generation")

## 2025-12-26
* Refactor SAM3Wrapper to use Sam3VideoPredictor high-level API
* Improve test coverage with Advanced Workflow E2E and Core Unit Tests
* Update documentation generation scripts
* Fix object detection call in SeedSelector using Sam3Processor

## 2025-12-25
* Improve test coverage with meaningful SAM3 pipeline and E2E tests
* Optimize and clean up SAM3 integration code
* Condense AGENTS_CODE_REFERENCE.md and add file tree
* Fix "Find People" tests and resolve integration suite issues
* Fix queue drain and system logs visibility bugs

## 2025-12-24
* Improve test coverage and fix unit test issues

## 2025-12-23
* Improve test coverage for core components and UI logic
* Add comprehensive testing documentation (TESTING.md)
* Implement comprehensive test coverage framework

## 2025-12-20
* Cleanup legacy code and dependencies (YOLO, DAM4SAM, Grounded-SAM-2)
* Rewrite SAM3 integration to follow official API patterns
* Update Setup Scripts for SAM3 and Jules Compatibility
* Restructure AGENTS.md and move code skeleton to separate file

## 2025-12-17
* Add extensive unit and integration tests for core functionality
* Fix log selector in headless environment for E2E tests
* Increase maximum range for Sharpness and Edge Strength sliders
* Fix SAM3 and Triton imports for Windows compatibility
* Add docstrings to application codebase

## 2025-12-16
* Refactor core architecture and remove legacy shim
* Improve test coverage for core and UI modules
* Fix and improve E2E tests
* Refactor scene_utils and split monolithic UI class
* Implement handler architecture (extraction, analysis, filtering)
* Fix critical architecture issues (circular imports, os usage)
* Automate AGENTS.md generation and streamline content

## 2025-12-15
* Resolve NameError in SAM3Wrapper and handle import errors
* Refactor UI for simplified UX and improved scene selection

## 2025-12-13
* Vectorize LPIPS deduplication logic
* Improve test coverage for core logic and fix test infrastructure
* Add authentication token support to model downloader
* Update auto-merge workflow

## 2025-09-18
* Update SAM2 model name and add DAM4SAM dependency
* Remove reid_fallback and related fields

## 2025-09-17
* Correct insightface import and improve error handling
* Implement real-time FFmpeg progress monitoring
* Add scenedetect to project dependencies
* Restructure UI for improved workflow
* Add SAM2 model and update requirements
* Add feature status check and refactor masking metrics
* Add IOU threshold and improve SAM2 error handling
* Add mask quality metrics and error tracking
* Enhance mask propagation with re-identification and identity ambiguity checks
* Optimize SAM2 mask propagation by removing temporary video
* Enhance SAM2 video predictor initialization

## 2025-09-16
* Improve subject masking logic and error handling
* Implement SAM2 for mask propagation
* Initialize SAM2 predictor in SubjectMasker
* Remove unused extraction and analysis tab methods
* Improve face analysis initialization and quality thresholding
* Improve scene detection and filtering logic
* Reorder state creation and event handler wiring

## 2025-09-15
* Refine image quality metrics and add GPU lock
* Add progress bars for extraction and analysis
* Add live analysis statistics reporting
* Replace DeepFace with InsightFace for face analysis

## 2025-06-29
* Refactor configuration into a dedicated Config class

## 2025-06-28
* Rename app2.py to app.py
* Refactor code structure for improved readability

## 2025-06-27
* Simplify clinerules and enhance frame filtering
* Update video download options and enhance logging
* Enhance UI layout and refactor parameter handling
* Add UI defaults and parameter mapping for video analysis

## 2025-06-26
* Adjust face detection thresholds and UI settings
* Implement frame export and video download enhancements
* Centralize configuration with `CONFIG` dictionary
* Add initial `app.py`
* Enhance face detection and processing in frame extraction

## 2025-06-25
* Enhance FFmpeg error handling and GPU synchronization
* Limit displayed frames to 100 and refine quality filter UI
* Enable interactive sliders for quality weights in GUI
* Implement batch processing for face similarity
* Optimize frame quality metrics with Numba
* Remove CUDA test script and enhance frame extractor with pre-filter options

## 2025-06-24
* Enhance frame extraction with GPU lock
* Add JSON logging for GUI integration
* Load quality weights and thresholds from shared config
* Replace blur score with sharpness metric
* Implement producer-consumer pattern for frame extraction

## 2025-06-23
* Refactor logging to use structured JSON messages
* Remove verbose logging parameter
* Apply post-processing filter only to filtered images
* Optimize Laplacian variance calculation by caching
* Refactor face filtering logic and update slider label

## 2025-06-22
* Add output directory logging to frame extraction process
* Add quality weights configuration to GUI and JSON
* Refactor GPU processing to use PyTorch
* Add ffmpeg installation script and unit tests

## 2025-06-20
* Update configuration for source URL and face reference image
* Refactor extraction functions and improve error handling
* Enhance frame processing feedback with detailed rejection reasons
* Add live frame preview functionality during extraction
* Add process management for extraction with stop functionality

## 2025-06-19
* Add configuration management features to frame extractor GUI
* Add support for video upload and face image upload
* Enhance GUI tooltips for settings
* Update extraction method default to keyframes and enable PNG output

## 2025-06-13
* Refactor frame extraction to use ffmpeg for improved color handling
* Add function to convert limited range images to full range
* Add face similarity filter for selective frame extraction using DeepFace

## 2025-06-12
* Add YouTube ID extraction and check for existing videos
* Count saved keyframes during extraction

## 2024-09-07
* Initial commit: Add YouTube screenshot extractor script and documentation
