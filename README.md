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
- **Detect and track subjects** automatically using state-of-the-art segmentation models (**SAM 3**).
- **Filter by face similarity** to find frames of specific people.
- **Export curated datasets** with smart cropping and aspect ratio options.

Perfect for creating training datasets (LoRA/Dreambooth), finding thumbnail candidates, or analyzing video content at scale.

## 🚀 Key Features

### 🎯 Intelligent Frame Extraction
- **Multiple Methods**: Keyframes, intervals, scene detection, or every frame.
- **YouTube Integration**: Direct URL processing with resolution control.
- **Smart Scene Detection**: Automatically identifies unique shots and transitions.

### 🧠 Advanced AI Analysis
- **Subject Segmentation**: Uses **SAM 3** for precise subject tracking and masking.
- **Face Recognition**: **InsightFace**-powered similarity matching.
- **Quality Assessment**: Multi-metric scoring including **NIQE** perceptual quality.
- **Text-to-Object**: Use text prompts with **SAM 3** to identify subjects.

### 🔍 Powerful Filtering & Export
- **Real-time Filtering**: Interactive sliders for all quality metrics.
- **Deduplication**: Perceptual hash (pHash) and LPIPS-based near-duplicate removal.
- **Smart Cropping**: Automatic subject-centered cropping with padding.
- **Aspect Ratios**: Export in 16:9, 1:1, 9:16, or custom ratios.

## 💻 Installation

### Prerequisites
- Python 3.10+
- FFmpeg installed and in your system PATH.
- CUDA-capable GPU (highly recommended for AI features).

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
    # Install main dependencies
    pip install -r requirements.txt

    # Install SAM3 from the local submodule
    pip install -e SAM3_repo
    ```

4.  **Install FFmpeg**
    - **Ubuntu/Debian**: `sudo apt install ffmpeg`
    - **macOS**: `brew install ffmpeg`
    - **Windows**: Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH.

5.  **Configure Environment Variables (Optional but Recommended)**
    To download gated models (like SAM3) automatically, you need to set up your Hugging Face token:
    
    1.  Copy the example environment file:
        ```bash
        cp .env_example .env
        ```
    2.  Edit `.env` and add your Hugging Face token:
        ```bash
        APP_HUGGINGFACE_TOKEN=your_token_here
        ```

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
4.  **Refine**: Review scene seeds, adjust selections, and run **Propagation** (SAM 3) to track the subject.
5.  **Filter & Export**: Use sliders to filter by quality, remove duplicates, and export your final dataset.

## 🏗️ Architecture

The application is built on a modular architecture:

*   **UI Layer**: `ui/` (Gradio components), separating presentation from logic.
*   **Core Logic**: `core/` contains business logic, pipelines, and managers.
*   **Configuration**: `core/config.py` handles settings via Pydantic.
*   **Data Storage**: SQLite (`core/database.py`) for frame metadata; JSONL for logs.
*   **AI Models**: Managed by a thread-safe `ModelRegistry` for lazy loading.

### Performance Optimizations

*   **Downscaled Video for SAM3**: During extraction, a `video_lowres.mp4` is created at thumbnail resolution. SAM3 reads this directly during propagation, eliminating per-scene temp JPEG I/O overhead.

### Directory Structure
```
subject-frame-extractor/
├── app.py                     # Main entry point
├── requirements.txt           # Python dependencies
├── core/                      # Core business logic
│   ├── config.py              # Configuration
│   ├── pipelines.py           # Processing pipelines
│   ├── database.py            # Database interface
│   ├── managers.py            # Model & Resource managers
│   └── events.py              # Pydantic event models
├── ui/                        # Gradio UI components
├── SAM3_repo/                 # SAM3 Submodule (Read-only)
├── tests/                     # Unit and E2E tests
└── AGENTS.md                  # Developer documentation
```

## 👨‍💻 Development

See **[AGENTS.md](AGENTS.md)** for detailed developer guidelines, testing instructions, and architectural insights.

### Running Tests
```bash
# Backend unit tests (fast, uses mocks)
python -m pytest tests/

# Integration tests (no mocks, requires GPU)
python -m pytest tests/test_integration.py -m integration

# Frontend E2E tests (requires App running + Playwright)
python -m pytest tests/e2e/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
