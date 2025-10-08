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

The application provides a three-step workflow through an intuitive web interface:

### Step 1: 📹 Frame Extraction

1. **Input your video**:
   - Paste a YouTube URL (e.g., `https://youtube.com/watch?v=...`)
   - Provide local file path (e.g., `C:\videos\my_video.mp4`)
   - Or upload directly through the interface

2. **Configure extraction**:
   - **Method**: Choose extraction strategy
     - `keyframes`: Extract I-frames only (fast, good coverage)
     - `interval`: Extract every N seconds (customizable)
     - `scene`: Detect scene changes automatically
     - `all`: Extract every frame (comprehensive but large)
   - **Resolution**: Limit YouTube download quality if needed
   - **Format**: PNG (lossless) or JPG (compressed)

3. **Start extraction**: Frames are saved to `downloads/[video_name]/`

### Step 2: 🔍 Seeding & Scene Selection

1. **Configure AI analysis**:
   - **Face Similarity**: Enable to filter by specific person
     - Upload reference photo of the target person
     - Choose face recognition model (buffalo_l recommended)
   - **Subject Masking**: Enable for subject-focused quality metrics
     - Select DAM4SAM model size (larger = more accurate)
     - Choose seeding strategy (face-based or person detection)

2. **Advanced options**:
   - **Scene Detection**: Use video scene boundaries for better tracking
   - **Text Prompts**: Ground subjects using natural language (e.g., "woman in red dress")
   - **Deduplication**: Remove perceptually similar frames

3. **Start analysis**: Creates `metadata.jsonl` with all computed metrics

### Step 3: 🎯 Filtering & Export

1. **Filter frames**:
   - Use quality sliders to set minimum thresholds
   - Adjust face similarity requirements
   - View histogram distributions for each metric
   - Toggle between kept and rejected frames

2. **Preview results**:
   - Interactive gallery with mask overlays
   - Real-time filter updates
   - Frame count and rejection statistics

3. **Export selection**:
   - **Cropping**: Automatic subject-centered cropping
   - **Aspect Ratios**: 16:9, 1:1, 9:16, or custom (e.g., "4:3,2:3")
   - **Padding**: Adjustable margin around subject
   - **Batch Export**: All kept frames with consistent formatting

## ⚙️ Configuration

The application uses YAML configuration for fine-tuning:

### Quality Metric Weights (`configs/config.yaml`)
```yaml
quality_weights:
  sharpness: 25      # Edge clarity
  edge_strength: 15  # Edge density  
  contrast: 15       # Dynamic range
  brightness: 10     # Luminance distribution
  entropy: 15        # Information content
  niqe: 20          # Perceptual quality
```

### UI Defaults
```yaml
ui_defaults:
  enable_face_filter: true
  enable_subject_mask: true
  dam4sam_model_name: "sam21pp-L"  # Largest, most accurate
  person_detector_model: "yolo11x.pt"
  face_model_name: "buffalo_l"
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
Process multiple videos by running the analysis pipeline programmatically:
```python
from app.pipelines.analyze import AnalysisPipeline
from app.logic.events import PreAnalysisEvent

params = PreAnalysisEvent(
    output_folder="path/to/frames",
    video_path="video.mp4",
    enable_subject_mask=True,
    enable_face_filter=True,
    face_ref_img_path="reference.jpg"
)

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
├── main.py                    # Main application
├── configs/
│   └── config.yaml          # Configuration settings
├── requirements.txt          # Python dependencies
├── DAM4SAM/                 # Subject tracking submodule
├── Grounded-SAM-2/          # Grounding/segmentation submodule
├── downloads/               # Output directory (created at runtime)
│   └── [video_name]/
│       ├── frame_000001.png
│       ├── metadata.jsonl   # Analysis results
│       ├── masks/           # Subject masks
│       └── thumbs/          # Preview thumbnails
├── models/                  # Cached AI models
└── logs/                    # Application logs
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
