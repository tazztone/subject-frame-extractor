# 🎬 Subject Frame Extractor

An AI-powered tool for extracting, analyzing, and filtering high-quality frames from videos or YouTube URLs. Designed for content creators, dataset builders, and anyone needing precise video frame analysis with advanced subject detection and quality metrics.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-CUDA%20Supported-green)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Gradio](https://img.shields.io/badge/UI-Gradio-orange)

## ✨ What This Tool Does

This application revolutionizes video frame extraction by combining traditional computer vision with cutting-edge AI models to:

- **Extract frames intelligently** from any video using multiple extraction strategies.
- **Analyze frame quality** using comprehensive metrics (sharpness, contrast, entropy, NIQE).
- **Detect and track subjects** automatically using state-of-the-art segmentation models (**SAM 2.1**).
- **Filter by face similarity** to find frames of specific people.
- **Export curated datasets** with smart cropping and aspect ratio options.

Perfect for creating training datasets (LoRA/Dreambooth), finding thumbnail candidates, or analyzing video content at scale.

## 🚀 Key Features

### 🎯 Intelligent Frame Extraction
- **Multiple Methods**: Keyframes, intervals, scene detection, or every frame.
- **YouTube Integration**: Direct URL processing with resolution control.
- **Smart Scene Detection**: Automatically identifies unique shots and transitions.

### 🧠 Advanced AI Analysis
- **Subject Segmentation**: Uses **SAM3** for precise subject tracking and masking.
- **Face Recognition**: **InsightFace**-powered similarity matching.
- **Quality Assessment**: Multi-metric scoring including **NIQE** perceptual quality.
- **Text-to-Object**: Use text prompts with **Grounded-DINO** to identify subjects.

### 🔍 Powerful Filtering & Export
- **Real-time Filtering**: Interactive sliders for all quality metrics.
- **Deduplication**: Perceptual hash (pHash) and SSIM-based near-duplicate removal.
- **Smart Cropping**: Automatic subject-centered cropping with padding.
- **Aspect Ratios**: Export in 16:9, 1:1, 9:16, or custom ratios.

## 💻 Installation

### Prerequisites
- Python 3.10+
- FFmpeg installed and in your system PATH.
- CUDA-capable GPU (recommended for AI features).

### Setup Guide

1.  **Clone the Repository**
    ```bash
    git clone --recursive https://github.com/tazztone/subject-frame-extractor.git
    cd subject-frame-extractor
    ```
    *Note: The `--recursive` flag is critical to fetch the SAM3 submodule.*

2.  **Create a Virtual Environment**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    # Install SAM3 directly from the submodule
    pip install git+https://github.com/facebookresearch/sam3.git
    ```

4.  **Install FFmpeg**
    - **Ubuntu/Debian**: `sudo apt install ffmpeg`
    - **macOS**: `brew install ffmpeg`
    - **Windows**: Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH.

## 📖 Usage

Run the application:
```bash
python app.py
```
Open your browser to the URL displayed (usually `http://127.0.0.1:7860`).

### Workflow

1.  **Input**: Upload a video or paste a YouTube URL.
2.  **Extraction**: Choose "Thumbnail Extraction" for speed or "Full Extraction" for precision.
3.  **Define Subject**:
    -   **By Face**: Upload a reference photo.
    -   **By Text**: Describe the subject (e.g., "cat", "man in suit").
    -   **Auto**: Let the AI find the most prominent subject.
4.  **Refine**: Review scene seeds, adjust selections, and run **Propagation** (SAM3) to track the subject.
5.  **Filter & Export**: Use sliders to filter by quality, remove duplicates, and export your final dataset.

## 🏗️ Architecture

The application is built on a modular architecture:

*   **UI Layer**: Built with **Gradio**, separating presentation from logic.
*   **Core Logic**: `app.py` orchestrates the workflow.
*   **Configuration**: `config.py` handles settings via Pydantic.
*   **Data Storage**: SQLite (`database.py`) for frame metadata; JSONL for logs.
*   **AI Models**: Managed by a thread-safe `ModelRegistry` for lazy loading.

### Directory Structure
```
subject-frame-extractor/
├── app.py                     # Main entry point
├── config.py                  # Configuration management
├── logger.py                  # Structured logging
├── database.py                # SQLite database interface
├── events.py                  # Event data models
├── SAM3_repo/                 # SAM3 Submodule (Do not edit)
├── tests/                     # Unit and E2E tests
└── AGENTS.md                  # Developer documentation
```

## 👨‍💻 Development

See **[AGENTS.md](AGENTS.md)** for detailed developer guidelines, testing instructions, and architectural insights.

### Running Tests
```bash
# Backend tests
python -m pytest tests/

# Frontend E2E tests (requires Playwright)
pytest tests/e2e/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
