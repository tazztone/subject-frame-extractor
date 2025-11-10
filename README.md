# 🎬 Subject Frame Extractor

An intelligent AI-powered tool for extracting, analyzing, and filtering high-quality frames from videos or YouTube URLs. Designed for content creators, dataset builders, and anyone needing precise video frame analysis with advanced subject detection and quality metrics.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-CUDA%20Supported-green)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Gradio](https://img.shields.io/badge/UI-Gradio-orange)

## ✨ What This Tool Does

This application revolutionizes video frame extraction by combining traditional computer vision with cutting-edge AI models to:

- **Extract frames intelligently** from any video using multiple extraction strategies
- **Analyze frame quality** using comprehensive metrics (sharpness, contrast, entropy, NIQE)
- **Detect and track subjects** automatically using state-of-the-art segmentation models
- **Filter by face similarity** to find frames of specific people
- **Export curated datasets** with smart cropping and aspect ratio options

Perfect for creating training datasets, finding thumbnail candidates, or analyzing video content at scale.

## 🚀 Key Features

### 🎯 Intelligent Frame Extraction
- **Multiple extraction methods**: keyframes, intervals, scene detection, or every frame
- **YouTube integration**: Direct URL processing with resolution control
- **Smart scene detection**: Automatically identify unique shots and transitions
- **Flexible timing**: Custom intervals or N-th frame extraction

### 🧠 Advanced AI Analysis
- **Subject Segmentation**: Uses DAM4SAM (SAM 2.1) for precise subject tracking and masking
- **Face Recognition**: InsightFace-powered similarity matching with reference photos
- **Quality Assessment**: Multi-metric scoring including NIQE perceptual quality
- **Person Detection**: YOLO-based human detection for seeding subject tracking
- **Text-to-Object**: Use text prompts with Grounded-DINO to identify subjects.

### 🔍 Powerful Filtering System
- **Real-time filtering**: Interactive sliders for all quality metrics
- **Face similarity matching**: Find frames containing specific people
- **Subject-focused analysis**: Quality metrics calculated only on main subjects
- **Duplicate detection**: Perceptual hash-based near-duplicate removal

### 📤 Smart Export Options
- **Intelligent cropping**: Automatic subject-centered cropping with padding
- **Multiple aspect ratios**: 16:9, 1:1, 9:16, or custom ratios
- **Batch processing**: Export hundreds of frames with consistent formatting
- **Resume capability**: Pause and resume analysis without losing progress

## 🛠️ Technical Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **UI Framework** | Gradio | Web-based interface |
| **Computer Vision** | OpenCV, PyTorch | Image processing |
| **Subject Tracking** | DAM4SAM + SAM 2.1 | Zero-shot object segmentation |
| **Face Recognition** | InsightFace | High-accuracy face detection/matching |
| **Object Detection** | YOLO | Person detection for tracking seed |
| **Text-to-Object** | Grounded-DINO | Grounding subjects with text prompts |
| **Video Processing** | FFmpeg, yt-dlp | Frame extraction and video handling |
| **Quality Assessment** | PyIQA (NIQE) | Perceptual image quality metrics |
| **Performance** | Numba, CUDA | Optimized computation |

## 📋 Prerequisites

Before installation, ensure you have:

1. **Python 3.10 or newer**
2. **Git** (for cloning submodules)
3. **NVIDIA GPU** (recommended for full functionality)
4. **CUDA toolkit** (for GPU acceleration)
5. **FFmpeg** installed and in system PATH

### FFmpeg Installation
- **Windows**: Download from [ffmpeg.org](https://ffmpeg.org/download.html)
- **macOS**: `brew install ffmpeg`
- **Ubuntu/Debian**: `sudo apt install ffmpeg`

## 📖 How to Use

The application provides a guided, five-tab workflow. Each tab represents a clear step in the process, with options progressively revealed to keep the UI clean and intuitive. During long-running tasks like extraction and analysis, a progress bar will appear to provide real-time feedback on the status and ETA. The application will also automatically switch you to the next tab after a major step is complete.

### Tab 1: 📹 Frame Extraction

This tab is for getting frames from your video source.

1.  **Provide a Video Source**: Paste a YouTube URL, enter a local file path, or upload a video file directly.
2.  **Configure Extraction Method**:
    -   **Recommended Method (Default)**: Use the "Thumbnail Extraction" for a fast and efficient pre-analysis workflow. This extracts lightweight thumbnails that are used in Tab 2 to find the best scenes *before* you commit to extracting full-resolution frames.
    -   **Advanced Method**: If you have specific needs, you can uncheck the "Recommended" option to access legacy full-frame extraction methods like `keyframes`, `interval`, or `every_nth_frame`. This is slower and not recommended for most workflows.
3.  **Start Extraction**: Click the "Start Extraction" button to begin. The results will be saved to the `downloads/` folder and will be automatically available for the next step.

### Tab 2: 👩🏼‍🦰 Define Subject

This tab is for identifying your subject within the extracted scenes.

1.  **Choose Your Seeding Strategy**: First, decide *how* you want the AI to find your subject. The UI dynamically shows only the relevant options for your choice:
    -   **👤 By Face**: The most precise method. Upload a reference photo, and the system will find that specific person.
    -   **📝 By Text**: Describe your subject using a text prompt (e.g., "a man in a blue shirt").
    -   **🤖 Automatic**: Let the AI find the most prominent person in each scene automatically. This is a great general-purpose starting point.
2.  **Find & Preview Scene Seeds**: Click the **"Find & Preview Best Frames"** button. The app runs a pre-analysis to find the best "seed frame" in each scene—the single frame where your subject is clearest.

### Tab 3: 🎞️ Scene Selection

This tab becomes active after you complete the subject definition. It allows you to refine your selection before the heavy processing begins.

1.  **Review & Refine Seeds**: A gallery of these seed frames will appear, along with controls to:
    -   Quickly include or exclude entire scenes.
    -   Use the **Scene Editor** to fine-tune the detection for a specific scene. You can select a different person from the YOLO detections or provide a text prompt to override the initial seed.
    -   Apply **Bulk Filters** to remove scenes that don't meet a minimum quality standard.
2.  **Propagate Masks**: Once you're happy with your seeds, click **"Propagate Masks on Kept Scenes"**. The AI uses the seed frame to track the subject through all other frames in each selected scene.

### Tab 4: 📝 Metrics

Choose which metrics to calculate during the analysis phase. More metrics provide more filtering options but may increase processing time.

### Tab 5: 📊 Filtering & Export

This tab becomes active after you complete mask propagation. It allows you to refine your selection and export the final frames.

1.  **Load Analysis & View Metrics**: When you select this tab, it automatically loads the results from the previous step. You can now use the **Filter Controls** to:
    -   Adjust sliders for quality metrics like `sharpness`, `contrast`, and `NIQE`.
    -   Set a `deduplication` threshold to remove visually similar frames.
    -   View histograms to understand the distribution of each metric.
2.  **Review Results**: As you adjust the filters, the **Results Gallery** updates in real-time. You can toggle between viewing "Kept" and "Rejected" frames to see the impact of your changes.
3.  **Export**: Once you are satisfied with your filtered selection, you can configure the **Export Options**:
    -   Enable **"Crop to Subject"** for automatic, intelligent cropping.
    -   Define a list of desired **Aspect Ratios** (e.g., `16:9, 1:1`).
    -   Click the **"Export Kept Frames"** button to save your final, curated dataset.

## ⚙️ Configuration

The application uses a `config.json` file for fine-tuning, which can be saved from the UI.

### Quality Metric Weights
```json
"quality_weights": {
  "sharpness": 25,
  "edge_strength": 15,
  "contrast": 15,
  "brightness": 10,
  "entropy": 15,
  "niqe": 20
}
```

### UI Defaults
```json
"ui_defaults": {
  "enable_face_filter": true,
  "enable_subject_mask": true,
  "dam4sam_model_name": "sam21pp-L",
  "person_detector_model": "yolo11x.pt",
  "face_model_name": "buffalo_l"
}
```

## 🔧 Advanced Usage

### Custom Text Prompts
Ground subjects using natural language with Grounded-DINO:
```
"a person wearing a red jacket"
"the main speaker on stage"  
"woman with long hair"
```

### Batch Processing
Process multiple videos by running the analysis pipeline programmatically. Note that the `app.py` is not structured as a library, so this requires refactoring.
```python
from app import AnalysisPipeline, PreAnalysisEvent # Fictional import

params = PreAnalysisEvent(
    output_folder="path/to/frames",
    video_path="video.mp4",
    enable_subject_mask=True,
    enable_face_filter=True,
    face_ref_img_path="reference.jpg"
)

# This is a conceptual example; direct import is not supported.
pipeline = AnalysisPipeline(params, queue, cancel_event)
result = pipeline.run_full_analysis(scenes_to_process)
```

### Model Selection
Choose models based on your hardware and accuracy needs:

| Model | Size | Speed | Accuracy | GPU Memory |
|-------|------|-------|----------|------------|
| sam21pp-T | Tiny | Fast | Good | 2GB |
| sam21pp-S | Small | Medium | Better | 4GB |
| sam21pp-L | Large | Slow | Best | 8GB+ |

## 📁 Project Structure

```
subject-frame-extractor/
├── app.py                     # Main application
├── requirements.txt           # Python dependencies
├── DAM4SAM/                   # Subject tracking submodule
├── Grounded-SAM-2/            # Grounding/segmentation submodule
├── downloads/                 # Output directory (created at runtime)
│   └── [video_name]/
│       ├── frame_000001.png
│       ├── metadata.jsonl     # Analysis results
│       ├── masks/             # Subject masks
│       └── thumbs/            # Preview thumbnails
├── models/                    # Cached AI models
└── logs/                      # Application logs
```

## 🔍 Troubleshooting

### Common Issues

**"FFmpeg not found"**
- Ensure FFmpeg is installed and added to system PATH
- Windows: Add FFmpeg bin directory to environment variables

**"CUDA out of memory"**
- Reduce DAM4SAM model size (`sam21pp-S` or `sam21pp-T`)
- Enable "Disable Parallelism" option
- Process fewer frames at once

**"No faces found in reference image"**
- Ensure reference photo shows clear, well-lit face
- Try different reference image with frontal view
- Check face model compatibility

**Slow performance**
- Enable GPU acceleration (CUDA)
- Use smaller AI models
- Reduce video resolution for processing
- Enable parallel processing

### Performance Tips

1. **For large videos**: Use scene detection to focus on unique shots
2. **For face filtering**: Use high-quality, well-lit reference photos  
3. **For quality analysis**: Process at lower resolution first, then refine
4. **For GPU memory**: Start with smaller models and scale up as needed

## 🤝 Contributing

Contributions welcome! This project combines multiple cutting-edge AI models and would benefit from:

- Additional quality metrics
- More efficient processing pipelines  
- Better UI/UX improvements
- Support for additional video formats
- Performance optimizations

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

This project builds upon several excellent open-source projects:

- **[DAM4SAM](https://github.com/jovanavidenovic/DAM4SAM)**: Dynamic object tracking
- **[Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2)**: Text-grounded segmentation
- **[InsightFace](https://github.com/deepinsight/insightface)**: Face recognition models
- **[Ultralytics YOLO](https://github.com/ultralytics/ultralytics)**: Object detection
- **[PyIQA](https://github.com/chaofengc/IQA-PyTorch)**: Image quality assessment

## 📞 Support

- 🐛 **Bug Reports**: [Open an issue](https://github.com/tazztone/subject-frame-extractor/issues)
- 💡 **Feature Requests**: [Start a discussion](https://github.com/tazztone/subject-frame-extractor/discussions)
- 📖 **Documentation**: Check this README for technical details.

---

<p align="center">
  <strong>Built with ❤️ for the computer vision community.</strong>
</p>

---

# Technical Documentation

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
