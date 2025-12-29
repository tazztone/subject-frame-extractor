# Changelog

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
