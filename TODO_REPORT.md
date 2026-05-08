# Technical Debt / TODO Report
Generated on: 2026-05-08 23:57:10
Total items: 42

## Top Priority Items

- [Medium] **core/batch_manager.py:35**: Add resource-aware scheduling (wait for GPU or RAM availability)
- [Medium] **core/batch_manager.py:152**: Store output path from result for later retrieval
- [Medium] **core/batch_manager.py:155**: Add retry logic for transient failures
- [Medium] **core/batch_manager.py:156**: Capture full stack trace for debugging
- [Medium] **core/config.py:21**: Log config loading errors instead of silently ignoring
- [Medium] **core/config.py:180**: Implement actual memory monitoring using these thresholds
- [Medium] **core/config.py:181**: Add automatic memory cleanup when approaching critical threshold
- [Medium] **core/error_handling.py:22**: Consider adding NOTIFY strategy for non-blocking alerts
- [Medium] **core/events.py:144**: Add scene status validation (enum instead of string)
- [Medium] **core/progress.py:33**: Support nested progress stages (sub-operations)
- [Medium] **core/progress.py:161**: Add locale-aware formatting
- [Medium] **core/progress.py:162**: Support different precision levels (coarse/fine)
- [Medium] **core/scene_utils/detection.py:24**: Support multiple detection algorithms (content, histogram, motion)
- [Medium] **core/scene_utils/detection.py:39**: Make ContentDetector threshold configurable
- [Medium] **core/scene_utils/detection.py:82**: Support thumbnail caching to skip already-processed images
- [Medium] **core/scene_utils/helpers.py:32**: Add box drawing style options (dashed, rounded corners, etc.)
- [Medium] **core/scene_utils/helpers.py:33**: Support drawing box labels with confidence scores
- [Medium] **core/scene_utils/helpers.py:138**: Add undo/redo support for status changes
- [Medium] **core/scene_utils/helpers.py:139**: Support batch status changes for multiple scenes
- [Medium] **core/scene_utils/helpers.py:140**: Add status change history for audit trail
- [Medium] **core/scene_utils/helpers.py:214**: Add preview caching to avoid redundant regeneration
- [Medium] **core/scene_utils/helpers.py:215**: Support async preview generation for faster UI response
- [Medium] **core/scene_utils/helpers.py:216**: Add comparison view (before/after) for seed changes
- [Medium] **core/scene_utils/mask_propagator.py:34**: Add temporal consistency smoothing between frames
- [Medium] **core/scene_utils/mask_propagator.py:35**: Implement bidirectional propagation merging (not just forward+backward)
- [Medium] **core/scene_utils/mask_propagator.py:36**: Add adaptive quality thresholds based on propagation distance from seed
- [Medium] **core/scene_utils/mask_propagator.py:278**: Consider deprecating legacy method in favor of video-based propagation
- [Medium] **core/scene_utils/mask_propagator.py:279**: Add memory-mapped frame loading for very large sequences
- [Medium] **core/scene_utils/seed_selector.py:39**: Add adaptive face similarity thresholds based on video quality
- [Medium] **core/scene_utils/seed_selector.py:40**: Implement multi-person tracking with ID assignment
- [Medium] **core/scene_utils/seed_selector.py:41**: Add strategy confidence scoring for automatic fallback ordering
- [Medium] **core/scene_utils/seed_selector.py:501**: Cache transform for reuse across multiple frames
- [Medium] **core/scene_utils/subject_masker.py:46**: Add support for tracking multiple subjects simultaneously
- [Medium] **core/scene_utils/subject_masker.py:47**: Implement confidence-based mask rejection
- [Medium] **core/scene_utils/subject_masker.py:48**: Add mask quality assessment metrics
- [Medium] **core/shared.py:23**: Add caching for repeated scene status checks
- [Medium] **core/shared.py:60**: Add additional status badges (pending, error, etc.)
- [Medium] **core/shared.py:61**: Consider SVG overlay for better scaling
- [Medium] **core/shared.py:202**: Log the specific exception for debugging
- [Medium] **ui/gallery_utils.py:51**: Add pagination support for large datasets (>1000 frames)
- [Medium] **ui/gallery_utils.py:52**: Implement virtual scrolling with lazy image loading
- [Medium] **ui/gallery_utils.py:53**: Add gallery sorting options (by score, time, etc.)