# Requirements Document

## Introduction

The Subject Frame Extraction application is a sophisticated AI-powered video processing system that automatically extracts, analyzes, and filters high-quality frames containing specific subjects from video content. Built on advanced computer vision models including SAM3 (Segment Anything Model 3) and InsightFace, the system provides professional-grade tools for content creators, dataset builders (LoRA/Dreambooth), researchers, and media professionals to efficiently curate high-quality image datasets from video footage with intelligent subject tracking, quality assessment, and batch processing capabilities.

## Glossary

- **Subject**: A person, object, or entity of interest that the user wants to extract and track from video frames
- **Frame_Extractor**: The core pipeline component responsible for FFmpeg-based video processing and frame extraction
- **Quality_Scorer**: The component that evaluates frames using multiple metrics including sharpness, contrast, NIQE, and perceptual quality
- **SAM3_Segmenter**: The Segment Anything Model 3 component that performs precise semantic segmentation and mask propagation
- **InsightFace_Analyzer**: The face recognition component that performs similarity matching, blink detection, and pose estimation
- **Scene_Detector**: The component that identifies scene boundaries and transitions using FFmpeg scene detection
- **Mask_Propagator**: The component that tracks subject masks across video sequences using temporal consistency
- **Export_Manager**: The component that handles filtered output, aspect ratio cropping, and batch export operations
- **Batch_Manager**: The component that orchestrates processing of multiple videos or large datasets
- **Thumbnail_Manager**: The component that manages efficient thumbnail generation and caching
- **NIQE_Metric**: No-Reference Image Quality Evaluator for perceptual quality assessment
- **Pre_Analysis_Pipeline**: The component that performs initial scene analysis and seed frame selection
- **Analysis_Pipeline**: The component that executes full AI analysis including segmentation and face recognition

## Requirements

### Requirement 1: Video Processing and Frame Extraction

**User Story:** As a content creator, I want to upload videos and configure extraction parameters, so that I can efficiently extract frames using intelligent sampling strategies.

#### Acceptance Criteria

1. WHEN a user uploads a video file, THE Frame_Extractor SHALL process all supported formats (MP4, AVI, MOV, MKV, WebM)
2. WHEN processing video content, THE Frame_Extractor SHALL provide multiple extraction strategies (keyframes, every Nth frame, scene-based, all frames)
3. WHEN YouTube URLs are provided, THE Frame_Extractor SHALL download and process videos with configurable resolution limits
4. WHEN extraction is configured, THE Frame_Extractor SHALL generate both full-resolution frames and optimized thumbnails
5. WHERE thumbnails-only mode is enabled, THE Frame_Extractor SHALL process at reduced resolution for faster analysis while maintaining quality assessment accuracy

### Requirement 2: Advanced AI-Powered Subject Analysis

**User Story:** As a dataset builder, I want precise subject segmentation and tracking, so that I can create high-quality training datasets with accurate subject masks.

#### Acceptance Criteria

1. WHEN processing frames, THE SAM3_Segmenter SHALL generate pixel-perfect segmentation masks for identified subjects
2. WHEN propagating across video sequences, THE Mask_Propagator SHALL maintain temporal consistency and subject identity
3. WHEN face-based subject selection is used, THE InsightFace_Analyzer SHALL perform similarity matching with configurable thresholds
4. WHEN text prompts are provided, THE SAM3_Segmenter SHALL detect subjects using open-vocabulary descriptions
5. WHEN multiple subjects are present, THE SAM3_Segmenter SHALL provide individual tracking and mask generation for each subject

### Requirement 3: Comprehensive Quality Assessment

**User Story:** As a media professional, I want multi-dimensional quality scoring, so that I can automatically identify the highest quality frames based on technical and perceptual metrics.

#### Acceptance Criteria

1. WHEN evaluating frame quality, THE Quality_Scorer SHALL calculate sharpness, edge strength, contrast, brightness, entropy, and NIQE scores
2. WHEN face analysis is enabled, THE Quality_Scorer SHALL assess blink probability, eye openness, and head pose (yaw, pitch, roll)
3. WHEN quality weights are configured, THE Quality_Scorer SHALL compute weighted composite scores based on user preferences
4. WHEN mask areas are evaluated, THE Quality_Scorer SHALL calculate subject visibility percentages and apply area-based filtering
5. THE Quality_Scorer SHALL normalize all metrics to 0-100 scales for consistent comparison and filtering

### Requirement 4: Intelligent Scene Detection and Pre-Analysis

**User Story:** As a video editor, I want automatic scene detection and smart frame selection, so that I can efficiently review content without processing redundant similar frames.

#### Acceptance Criteria

1. WHEN processing video content, THE Scene_Detector SHALL identify scene boundaries using configurable sensitivity thresholds
2. WHEN scenes are detected, THE Pre_Analysis_Pipeline SHALL select representative "seed" frames from each scene
3. WHEN seed selection occurs, THE Pre_Analysis_Pipeline SHALL apply intelligent strategies (largest person, prominent person detection, balanced scoring)
4. WHEN scene analysis is complete, THE Pre_Analysis_Pipeline SHALL provide scene-by-scene navigation and preview capabilities
5. WHERE face matching is enabled, THE Pre_Analysis_Pipeline SHALL prioritize frames containing the target subject during seed selection

### Requirement 5: Interactive User Interface and Workflow Management

**User Story:** As a user, I want an intuitive multi-tab interface with real-time feedback, so that I can efficiently configure, monitor, and control the extraction and analysis process.

#### Acceptance Criteria

1. WHEN the application launches, THE User_Interface SHALL display organized tabs for Extraction, Subject Definition, Scene Analysis, Metrics, and Filtering
2. WHEN processing is active, THE User_Interface SHALL provide real-time progress tracking with detailed status updates and memory monitoring
3. WHEN configuring analysis parameters, THE User_Interface SHALL offer preset configurations and advanced customization options
4. WHEN reviewing results, THE User_Interface SHALL provide interactive galleries with thumbnail previews, quality scores, and filtering controls
5. WHERE batch processing is enabled, THE User_Interface SHALL display batch queue status and allow individual item management

### Requirement 6: Advanced Filtering and Export Capabilities

**User Story:** As a researcher, I want sophisticated filtering and export options, so that I can curate datasets with precise quality criteria and export in formats suitable for my workflow.

#### Acceptance Criteria

1. WHEN filtering frames, THE Export_Manager SHALL provide real-time slider-based filtering across all quality metrics
2. WHEN deduplication is enabled, THE Export_Manager SHALL remove near-identical frames using perceptual hashing (pHash) and LPIPS similarity
3. WHEN exporting frames, THE Export_Manager SHALL support multiple aspect ratios (1:1, 16:9, 9:16, custom) with intelligent subject-centered cropping
4. WHEN batch exporting, THE Export_Manager SHALL maintain consistent naming conventions, preserve metadata, and provide progress tracking
5. WHERE custom export settings are specified, THE Export_Manager SHALL apply compression settings, resolution scaling, and format conversion

### Requirement 7: Performance Optimization and Resource Management

**User Story:** As a professional user processing large video files, I want efficient resource utilization and memory management, so that I can handle substantial content without system instability.

#### Acceptance Criteria

1. WHEN processing high-resolution content, THE Frame_Extractor SHALL maintain processing speeds of at least 30 FPS for thumbnail generation
2. WHEN system resources are constrained, THE Batch_Manager SHALL implement intelligent memory management with configurable worker thread limits
3. WHEN GPU memory is available, THE SAM3_Segmenter SHALL utilize CUDA acceleration with automatic fallback to CPU processing
4. WHEN caching thumbnails, THE Thumbnail_Manager SHALL implement LRU eviction with configurable cache sizes and cleanup thresholds
5. WHERE memory monitoring is enabled, THE Frame_Extractor SHALL track VRAM and system memory usage with automatic cleanup when approaching limits

### Requirement 8: Robust Error Handling and Session Management

**User Story:** As a user processing long videos, I want reliable error recovery and session persistence, so that I can resume work after interruptions without losing progress.

#### Acceptance Criteria

1. WHEN file corruption or processing errors occur, THE Frame_Extractor SHALL skip problematic frames and continue processing with detailed error logging
2. WHEN segmentation or analysis fails, THE SAM3_Segmenter SHALL log specific failure reasons and attempt graceful degradation
3. WHEN sessions are interrupted, THE Frame_Extractor SHALL save processing state to SQLite database for reliable session recovery
4. WHEN resuming sessions, THE Frame_Extractor SHALL validate existing data integrity and continue from the last successful checkpoint
5. WHERE critical errors occur, THE Frame_Extractor SHALL provide clear user feedback with actionable recovery suggestions

### Requirement 9: Database Integration and Metadata Management

**User Story:** As a power user managing large datasets, I want persistent metadata storage and fast querying capabilities, so that I can efficiently organize and retrieve processed content.

#### Acceptance Criteria

1. WHEN frames are processed, THE Database SHALL store all quality metrics, face analysis results, and mask metadata in SQLite format
2. WHEN filtering operations are performed, THE Database SHALL provide sub-second query response times for real-time UI updates
3. WHEN exporting data, THE Database SHALL maintain referential integrity between frames, scenes, and analysis results
4. WHEN session data is saved, THE Database SHALL preserve complete processing configurations and user preferences
5. WHERE data migration is needed, THE Database SHALL provide schema versioning and backward compatibility support