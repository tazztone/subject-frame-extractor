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

## 🚀 Quick Start

### Windows (Automated Install)

 **download and Run the installer bat file**:
   ```bash
   windows_STANDALONE_install.bat
   ```
   
   This script will:
   - Initialize git submodules (DAM4SAM, Grounded-SAM-2)
   - Create a Python virtual environment
   - Install PyTorch with CUDA 12.8 support
   - Install all required dependencies

2. **Launch the application**:
   ```bash
   windows_run_app.bat
   ```

## 📖 How to Use

The application provides a guided, three-tab workflow. Each tab represents a clear step in the process, with options progressively revealed to keep the UI clean and intuitive. During long-running tasks like extraction and analysis, a progress bar will appear to provide real-time feedback on the status and ETA. The application will also automatically switch you to the next tab after a major step is complete.

### Tab 1: 📹 Frame Extraction

This tab is for getting frames from your video source.

1.  **Provide a Video Source**: Paste a YouTube URL, enter a local file path, or upload a video file directly.
2.  **Configure Extraction Method**:
    -   **Recommended Method (Default)**: Use the "Thumbnail Extraction" for a fast and efficient pre-analysis workflow. This extracts lightweight thumbnails that are used in Tab 2 to find the best scenes *before* you commit to extracting full-resolution frames.
    -   **Advanced Method**: If you have specific needs, you can uncheck the "Recommended" option to access legacy full-frame extraction methods like `keyframes`, `interval`, or `every_nth_frame`. This is slower and not recommended for most workflows.
3.  **Start Extraction**: Click the "Start Extraction" button to begin. The results will be saved to the `downloads/` folder and will be automatically available for the next step.

### Tab 2: 🎯 Define Subject

This tab is for identifying your subject within the extracted scenes.

1.  **Choose Your Seeding Strategy**: First, decide *how* you want the AI to find your subject. The UI dynamically shows only the relevant options for your choice:
    -   **👤 By Face**: The most precise method. Upload a reference photo, and the system will find that specific person.
    -   **📝 By Text**: Describe your subject using a text prompt (e.g., "a man in a blue shirt").
    -   **🤖 Automatic**: Let the AI find the most prominent person in each scene automatically. This is a great general-purpose starting point.
2.  **Find & Preview Scene Seeds**: Click the **"Find & Preview Scene Seeds"** button. The app runs a pre-analysis to find the best "seed frame" in each scene—the single frame where your subject is clearest.

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
- 📖 **Documentation**: Check this README and inline code comments

---

<p align="center">
  <strong>Built with ❤️ for the computer vision community. (readme by gemini)</strong>
</p>
