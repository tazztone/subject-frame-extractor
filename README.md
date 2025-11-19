# 🎬 Subject Frame Extractor v2.0

An AI-powered tool for extracting, analyzing, and filtering high-quality frames from videos or YouTube URLs. Designed for content creators, dataset builders, and anyone needing precise video frame analysis with advanced subject detection and quality metrics.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-CUDA%20Supported-green)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Gradio](https://img.shields.io/badge/UI-Gradio-orange)

## ✨ What This Tool Does

This application revolutionizes video frame extraction by combining traditional computer vision with cutting-edge AI models to:

- **Extract frames intelligently** from any video using multiple extraction strategies
- **Analyze frame quality** using comprehensive metrics (sharpness, contrast, entropy, NIQE)
- **Detect and track subjects** automatically using state-of-the-art segmentation models (SAM 2.1)
- **Filter by face similarity** to find frames of specific people
- **Export curated datasets** with smart cropping and aspect ratio options

Perfect for creating training datasets (LoRA/Dreambooth), finding thumbnail candidates, or analyzing video content at scale.

## 🚀 Key Features

### 🎯 Intelligent Frame Extraction
- **Multiple extraction methods**: keyframes, intervals, scene detection, or every frame
- **YouTube integration**: Direct URL processing with resolution control
- **Smart scene detection**: Automatically identify unique shots and transitions
- **Flexible timing**: Custom intervals or N-th frame extraction

### 🧠 Advanced AI Analysis
- **Subject Segmentation**: Uses **DAM4SAM (SAM 2.1)** for precise subject tracking and masking
- **Face Recognition**: **InsightFace**-powered similarity matching with reference photos
- **Quality Assessment**: Multi-metric scoring including **NIQE** perceptual quality
- **Person Detection**: **YOLOv11**-based human detection for seeding subject tracking
- **Text-to-Object**: Use text prompts with **Grounded-DINO** to identify subjects

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
| **Object Detection** | YOLOv11 | Person detection for tracking seed |
| **Text-to-Object** | Grounded-DINO | Grounding subjects with text prompts |
| **Video Processing** | FFmpeg, yt-dlp | Frame extraction and video handling |
| **Quality Assessment** | PyIQA (NIQE) | Perceptual image quality metrics |

## 📋 Prerequisites

Before installation, ensure you have:

1. **Python 3.10 or newer**
2. **Git** (for cloning submodules)
3. **NVIDIA GPU** (Highly recommended for full functionality)
4. **CUDA toolkit** (for GPU acceleration)
5. **FFmpeg** installed and in system PATH

### FFmpeg Installation
- **Windows**: Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH.
- **macOS**: `brew install ffmpeg`
- **Ubuntu/Debian**: `sudo apt install ffmpeg`

## 💻 Installation

### 🪟 Windows (Automated Method)

We provide batch scripts to automate the setup process.

1.  **Install / Update**:
    Run `windows_STANDALONE_install.bat` to clone the repo and set up the environment automatically.
    *Note: This script is designed to be run from a parent folder to create a new installation.*

2.  **Run the App**:
    Double-click `windows_run_app.bat`. This will activate the virtual environment and launch the UI in your browser.

3.  **Update**:
    Run `windows_update.bat` to pull the latest changes and update dependencies.

### 🐧 Linux / macOS / Manual Windows

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/tazztone/subject-frame-extractor.git
    cd subject-frame-extractor
    ```

2.  **Initialize Submodules:**
    ```bash
    git submodule update --init --recursive
    ```

3.  **Create Virtual Environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

4.  **Install PyTorch (CUDA support):**
    *Visit [pytorch.org](https://pytorch.org/get-started/locally/) for your specific command.*
    ```bash
    # Example for CUDA 11.8
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
    ```

5.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

6.  **Install DAM4SAM (SAM 2.1):**
    *Note: On Windows, you may need to set `SAM2_BUILD_CUDA=0` to avoid compilation errors if you lack build tools.*
    ```bash
    cd DAM4SAM
    # Windows: set SAM2_BUILD_CUDA=0 && pip install -e .
    # Linux/Mac: pip install -e .
    cd ..
    ```

## 📖 How to Use

The application provides a guided, five-tab workflow.

### Tab 1: 📹 Frame Extraction
**Get frames from your video source.**
1.  **Source**: Paste a YouTube URL or upload a video file.
2.  **Method**: Use "Thumbnail Extraction" (Recommended) for fast pre-analysis, or legacy methods for full extraction.
3.  **Start**: Results are saved to `downloads/` and auto-loaded for the next step.

### Tab 2: 👩🏼‍🦰 Define Subject
**Identify your subject within the scenes.**
1.  **Strategy**:
    -   **👤 By Face**: Upload a reference photo.
    -   **📝 By Text**: Describe the subject (e.g., "man in red shirt").
    -   **🤖 Automatic**: Finds the most prominent person.
2.  **Find Seeds**: Click **"Find & Preview Best Frames"** to identify the best "seed frame" per scene.

### Tab 3: 🎞️ Scene Selection
**Refine selection before heavy processing.**
1.  **Review**: Check the gallery of seed frames.
2.  **Edit**: Override detections (change person, use text) or exclude scenes.
3.  **Propagate**: Click **"Propagate Masks"** to track the subject through all frames in selected scenes using SAM 2.1.

### Tab 4: 📝 Metrics
**Configure analysis.**
Choose which metrics to calculate (Sharpness, NIQE, Face Similarity, etc.).

### Tab 5: 📊 Filtering & Export
**Curate and save.**
1.  **Filter**: Use sliders to filter by quality metrics.
2.  **Deduplicate**: Remove similar frames using pHash/SSIM.
3.  **Export**: Enable **"Crop to Subject"**, set aspect ratios (e.g., `1:1, 9:16`), and save your dataset.

## ⚙️ Configuration

The application uses a `config.json` file for fine-tuning. Key settings include:

-   **Quality Weights**: Adjust the importance of sharpness, contrast, etc. in the global score.
-   **Model Selection**: Choose between `sam21pp-T` (Tiny/Fast), `sam21pp-S` (Small), or `sam21pp-L` (Large/Best) based on your VRAM.

## 📁 Project Structure

```
subject-frame-extractor/
├── app.py                     # Main application entry point
├── requirements.txt           # Python dependencies
├── DAM4SAM/                   # Subject tracking submodule (SAM 2.1)
├── downloads/                 # Output directory
│   └── [video_name]/
│       ├── frame_000001.png
│       ├── metadata.jsonl     # Analysis results
│       ├── masks/             # Subject masks
│       └── thumbs/            # Preview thumbnails
├── models/                    # Cached AI models
├── logs/                      # Application logs
├── windows_run_app.bat        # Launcher for Windows
└── windows_update.bat         # Updater for Windows
```

## 🔍 Troubleshooting

-   **FFmpeg not found**: Ensure it's in your system PATH.
-   **CUDA OOM**: Switch to a smaller SAM model (`sam21pp-T`) or process fewer frames.
-   **Installation Issues**: Try the `windows_STANDALONE_install.bat` for a clean setup.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <strong>Built with ❤️ for the computer vision community.</strong>
</p>
