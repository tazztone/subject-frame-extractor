# app.py Documentation

This document provides a detailed explanation of the `app.py` script, which is a monolithic application for frame extraction and analysis from videos.


## 1. Overview

`app.py` is a comprehensive tool for extracting frames from videos, analyzing them for various quality and content metrics, and providing a user interface for filtering and exporting the results. It integrates several machine learning models for tasks like subject detection, face analysis, and image quality assessment. The application is built using Gradio to provide an interactive web-based UI.

## 2. Configuration (`Config` class)

The application's behavior is controlled by the `Config` dataclass, which can be initialized from a `config.json` file and can be overridden by environment variables. A `config_dump.json` with the current settings can be saved from the UI.

### 2.1. `Paths`

Defines the directory structure for logs, models, and downloads.

*   [`logs`](app.py:117): Directory to store log files.
*   [`models`](app.py:118): Directory to store downloaded machine learning models.
*   [`downloads`](app.py:119): Directory for downloaded videos and other temporary files.
*   [`grounding_dino_config`](app.py:120): Path to the GroundingDINO model configuration file.
*   [`grounding_dino_checkpoint`](app.py:121): Path to the GroundingDINO model checkpoint file.

### 2.2. `Models`

Contains URLs and settings for downloading the required machine learning models.

*   [`user_agent`](app.py:125): User agent to use for downloading models.
*   [`grounding_dino`](app.py:126): URL for the GroundingDINO model.
*   [`face_landmarker`](app.py:127): URL for the MediaPipe Face Landmarker model.
*   [`dam4sam`](app.py:128): Dictionary of URLs for different DAM4SAM models.
*   [`yolo`](app.py:134): Base URL for YOLO models.

### 2.3. `YouTubeDL`

Configuration for downloading videos from YouTube using `yt-dlp`.

*   [`output_template`](app.py:138): Template for the output filename of downloaded videos.
*   [`format_string`](app.py:139): `yt-dlp` format string to select video and audio quality.

### 2.4. `Ffmpeg`

Settings for `ffmpeg` used for frame extraction.

*   [`log_level`](app.py:143): Logging level for `ffmpeg`.
*   [`thumbnail_quality`](app.py:144): Quality setting for thumbnail extraction.
*   [`scene_threshold`](app.py:145): Threshold for scene detection.

### 2.5. `Cache`

Parameters for the thumbnail cache.

*   [`size`](app.py:150): Maximum number of thumbnails to keep in the cache.
*   [`eviction_factor`](app.py:151): Percentage of the cache to evict when it's full.
*   [`cleanup_threshold`](app.py:152): Cache fullness threshold to trigger cleanup.

### 2.6. `Retry`

Default settings for retry logic.

*   [`max_attempts`](app.py:156): Maximum number of retry attempts.
*   [`backoff_seconds`](app.py:157): List of backoff delays between retries.

### 2.7. `QualityScaling`

Parameters for scaling quality metrics.

*   [`entropy_normalization`](app.py:161): Normalization factor for entropy calculation.
*   [`resolution_denominator`](app.py:162): Denominator for resolution scaling.
*   [`contrast_clamp`](app.py:163): Maximum value for contrast.
*   [`niqe_offset`](app.py:164): Offset for NIQE score calculation.
*   [`niqe_scale_factor`](app.py:165): Scaling factor for NIQE score.

### 2.8. `Masking`

Settings for mask post-processing.

*   [`keep_largest_only`](app.py:169): If true, only the largest connected component in a mask is kept.
*   [`close_kernel_size`](app.py:170): Kernel size for morphological closing.
*   [`open_kernel_size`](app.py:171): Kernel size for morphological opening.

### 2.9. `UIDefaults`

Default values for the Gradio user interface. This class defines the initial state of the UI components.

### 2.10. `FilterDefaults`

Default values for the filter sliders in the UI.

### 2.11. `QualityWeights`

Weights for combining different quality metrics into a single quality score.

### 2.12. `Choices`

Lists of choices for dropdowns in the UI.

### 2.13. `GroundingDinoParams`

Parameters for the GroundingDINO model.

*   [`box_threshold`](app.py:238): Box confidence threshold.
*   [`text_threshold`](app.py:239): Text confidence threshold.

### 2.14. `PersonDetector`

Configuration for the YOLO person detector.

*   [`model`](app.py:243): Model file to use.
*   [`imgsz`](app.py:244): Image size for inference.
*   [`conf`](app.py:245): Confidence threshold.

### 2.15. `Logging`

Configuration for the logging system.

*   [`log_level`](app.py:249): The minimum log level to output.
*   [`log_format`](app.py:250): The format string for log messages.
*   [`colored_logs`](app.py:251): Whether to use colored logs in the console.
*   [`structured_log_path`](app.py:252): Path for the structured JSONL log file.


## 3. Logging (`AppLogger`)

The application uses a custom logging setup to provide detailed and structured information about its execution.

### 3.1. `LogEvent`

The [`LogEvent`](app.py:441) dataclass defines the structure for a single log entry. It includes not only the standard log message and level but also performance metrics, error information, and other context.

### 3.2. `AppLogger`

The [`AppLogger`](app.py:456) is the core of the logging system. It provides the following features:

*   **Colored Console Output**: For improved readability during development.
*   **Session-based Log Files**: A new log file is created for each application run.
*   **Structured Logging**: Writes logs in JSONL format to a separate file ([`structured_log.jsonl`](app.py:252)), which is useful for programmatic analysis.
*   **Operation Context**: The `operation` context manager ([`operation()`](app.py:539)) can be used to automatically log the start, success, or failure of a block of code, including its duration.
*   **Progress Queue Integration**: Can be connected to a `Queue` to send log messages to the UI.

## 4. Error Handling

The [`ErrorHandler`](app.py:621) class provides decorators for common error recovery strategies.

*   [`with_retry()`](app.py:628): A decorator that automatically retries a function if it fails with a recoverable exception. The number of attempts and backoff delays are configurable.
*   [`with_fallback()`](app.py:655): A decorator that executes a fallback function if the primary function fails.

## 5. Events

The application uses a series of dataclasses to represent events triggered by the user interface. These events carry the necessary data from the UI to the backend processing pipelines. Examples include [`ExtractionEvent`](app.py:683), [`PreAnalysisEvent`](app.py:697), and [`FilterEvent`](app.py:731).

## 6. Utilities

The script contains several utility functions:

*   [`sanitize_filename()`](app.py:758): Cleans a string to make it a valid filename.
*   [`safe_resource_cleanup()`](app.py:775): A context manager to ensure that garbage collection is run and the CUDA cache is emptied, which is crucial for managing memory in PyTorch.
*   [`_to_json_safe()`](app.py:782): Converts various Python objects, including dataclasses and NumPy arrays, into a JSON-serializable format.


## 7. Core Data Structures

The application uses several dataclasses to represent the data it works with.

*   [`FrameMetrics`](app.py:825): Holds all the calculated quality scores for a single frame.
*   [`Frame`](app.py:836): Represents a single extracted frame, containing the image data, frame number, and its associated metrics. It also has a method [`calculate_quality_metrics()`](app.py:844) to perform the quality analysis.
*   [`Scene`](app.py:901): Represents a single scene (shot) in the video, defined by a start and end frame. It also stores information about the best seed frame within that scene.
*   [`AnalysisParameters`](app.py:916): A dataclass that aggregates all the parameters from the UI that are needed for the analysis pipeline. The `from_ui` class method is used to construct an instance from the raw UI values.

## 8. Model Loading and Management

The script includes several functions for downloading and initializing the machine learning models.

*   [`download_model()`](app.py:1078): A generic function to download a file from a URL if it doesn't already exist locally. It includes retry logic.
*   [`get_face_analyzer()`](app.py:1100): Loads the face analysis model from `insightface`. It's cached using `@lru_cache` to avoid reloading the model.
*   [`PersonDetector`](app.py:1125): A class that wraps a YOLO model for person detection.
*   [`get_person_detector()`](app.py:1147): A cached function to get an instance of the `PersonDetector`.
*   [`get_grounding_dino_model()`](app.py:1153): A cached function to download and load the GroundingDINO model.
*   [`get_dam4sam_tracker()`](app.py:1189): A cached function to download and initialize the DAM4SAM tracker for video object segmentation.
*   [`initialize_analysis_models()`](app.py:1222): This function initializes all the necessary models for the analysis pipeline based on the current parameters.

## 9. Video and Frame Processing

This section covers the classes and functions responsible for handling the video input and processing the frames.

*   [`VideoManager`](app.py:1257): Handles the initial video preparation. It can either download a video from YouTube or use a local file. The [`prepare_video()`](app.py:1264) method returns the path to the local video file that can be used by `ffmpeg`.
*   [`run_scene_detection()`](app.py:1299): Uses `scenedetect` to find the scene cuts in the video.
*   [`run_ffmpeg_extraction()`](app.py:1313): Constructs and runs the `ffmpeg` command to extract frames based on the selected method (e.g., keyframes, interval).
*   [`postprocess_mask()`](app.py:1349): Cleans up the masks generated by the segmentation models using morphological operations.
*   [`render_mask_overlay()`](app.py:1362): Renders a mask on top of a frame for visualization.


## 10. Masking and Propagation

This is the core logic for identifying a subject in a video and tracking it across a scene.

### 10.1. `SeedSelector`

The [`SeedSelector`](app.py:1447) class is responsible for finding the initial bounding box (the "seed") for the subject of interest in a single frame. It supports several strategies:

*   **Identity-First**: [`_identity_first_seed()`](app.py:1507) uses a reference face image to find the target person.
*   **Object-First**: [`_object_first_seed()`](app.py:1521) uses a text prompt and GroundingDINO to find an object.
*   **Face with Text Fallback**: [`_face_with_text_fallback_seed()`](app.py:1489) tries the identity-first approach and falls back to the object-first approach if no matching face is found.
*   **Automatic**: [`_choose_person_by_strategy()`](app.py:1613) uses a person detector (YOLO) to find either the largest or most central person in the frame.

### 10.2. `MaskPropagator`

The [`MaskPropagator`](app.py:1396) class takes the initial seed from the `SeedSelector` and uses the DAM4SAM tracker to propagate the segmentation mask forwards and backwards through all the frames in a scene.

### 10.3. `SubjectMasker`

The [`SubjectMasker`](app.py:1672) class orchestrates the whole masking process. It iterates through the scenes, uses the `SeedSelector` to find the best seed frame in each scene, and then uses the `MaskPropagator` to generate the masks for all frames in that scene.

## 11. Pipelines

The application logic is organized into pipelines, which are classes that encapsulate a sequence of processing steps.

*   [`Pipeline`](app.py:1007): The base class for all pipelines.
*   [`ExtractionPipeline`](app.py:1813): The pipeline for the initial frame extraction. It prepares the video, runs scene detection, and then runs `ffmpeg` to extract the frames.
*   [`AnalysisPipeline`](app.py:1851): This pipeline runs the main analysis. It initializes the models, and for each frame, it calculates quality metrics and face similarity, and then writes all the metadata to a `.jsonl` file.

## 12. Filtering and Scene Logic

The script includes functions for filtering the extracted frames based on their metrics and for managing scenes.

*   [`load_and_prep_filter_data()`](app.py:2011): Loads the `metadata.jsonl` file and prepares the data for the filter controls in the UI.
*   [`apply_all_filters_vectorized()`](app.py:2053): A vectorized function that efficiently applies all the active filters to the frame data.
*   [`on_filters_changed()`](app.py:2093): The event handler that is called whenever a filter control is changed in the UI. It re-applies the filters and updates the results gallery.
*   The script also contains several functions for managing scenes, such as `toggle_scene_status()` and `build_scene_gallery_items()`.


## 13. User Interface (`AppUI` and `EnhancedAppUI`)

The user interface is built using Gradio.

*   [`AppUI`](app.py:2688): This class is responsible for building the Gradio UI. It defines all the UI components (buttons, sliders, etc.) and lays them out in tabs.
*   [`EnhancedAppUI`](app.py:2946): This class inherits from `AppUI` and adds more advanced features to the UI, such as an enhanced logger with filtering, a progress bar with more details, and pause/cancel buttons. It also contains the main event handling logic, wiring the UI components to the backend pipeline functions. The `_run_task_with_progress` method is a generic wrapper for running a pipeline task in a separate thread and updating the UI with progress.

## 14. Main Execution

The `main()` function at the end of the script is the entry point of the application.

*   It first checks for the presence of `ffmpeg` and other required dependencies.
*   It then creates a `CompositionRoot` instance, which is a dependency injection container that creates and holds the instances of the main classes like `Config`, `EnhancedLogger`, and `ThumbnailManager`.
*   It gets the `AppUI` instance from the composition root and calls the `build_ui()` method to create the Gradio interface.
*   Finally, it calls `demo.launch()` to start the Gradio web server.
