# 🎬 Subject Frame Extractor

**An AI-powered powerhouse for extracting, analyzing, and filtering high-quality frames from video.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-CUDA%20Supported-ee4c2c.svg)](https://pytorch.org/)
[![Gradio](https://img.shields.io/badge/UI-Gradio-ff5000.svg)](https://gradio.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Designed for content creators, dataset builders (LoRA/Dreambooth), and researchers. This tool bridges the gap between raw video footage and curated, high-quality image datasets using state-of-the-art AI.

---

## ✨ Overview

Traditional frame extraction is noisy. **Subject Frame Extractor** uses advanced segmentation and quality heuristics to ensure you only keep the frames that matter.

*   **Intelligent Extraction**: Beyond simple intervals—use scene detection and keyframe awareness.
*   **Subject Centric**: Automatically track and mask specific people or objects using **SAM 3**.
*   **Quality First**: Filter by sharpness, contrast, and perceptual quality (**NIQE**).
*   **Face Matching**: Find every frame of a specific person using **InsightFace**.

---

## 🚀 Key Features

### 🎯 Smart Extraction
*   **Extraction Strategies**: Keyframes, fixed intervals, scene-based, or every Nth frame.
*   **YouTube Integration**: Direct URL processing with resolution control.
*   **Scene Intelligence**: Automatically segments video into shots to optimize analysis.

### 🧠 Advanced AI Analysis
*   **SAM 3 Integration**: Precise subject segmentation and tracking across scenes.
*   **Open-Vocabulary Detection**: Describe what you want to find (e.g., "a golden retriever") and let the AI find it.
*   **Face Analysis**: Similarity matching, blink detection, and head pose estimation (yaw/pitch/roll).
*   **Perceptual Metrics**: Real-time quality scoring to surface the "best" frames automatically.

### 🔍 Filtering & Export
*   **Interactive Sliders**: Filter thousands of frames in real-time based on AI-calculated metrics.
*   **Smart Deduplication**: Uses pHash and LPIPS to remove near-identical frames.
*   **AR-Aware Cropping**: Export subject-centered crops in 1:1, 9:16, 16:9, or custom ratios.

### 📸 Photo Mode (Culling)
*   **RAW Support**: Extract high-resolution embedded previews from RAW files (CR2, NEF, ARW, DNG, ORF, etc.) using **ExifTool**. No demosaicing required for ultra-fast ingestion.
*   **Quality Culling**: AI-powered scoring for focus, composition, and technical quality.
    *   **Sharpness**: Laplacian variance based edge-detection to identify focused shots.
    *   **Naturalness (NIQE)**: Perceptual quality score that measures how \"natural\" an image looks without needing a reference.
    *   **Information (Entropy)**: Measures the complexity/detail density of the image.
    *   **Face Prominence**: Uses InsightFace to detect faces and score them based on confidence and size.
*   **Lightroom/C1 Interop**: Export internal scores as 1-5 star ratings directly to **non-destructive XMP sidecars**.

---

## 🛠️ Tech Stack

*   **Segmentation**: [Segment Anything Model 3 (SAM 3)](https://github.com/facebookresearch/sam3)
*   **Face Analysis**: [InsightFace](https://github.com/deepinsight/insightface)
*   **UI Framework**: [Gradio 6.x](https://gradio.app/)
*   **Data Science**: PyTorch, NumPy, OpenCV, Pydantic
*   **Media Handling**: FFmpeg, yt-dlp
*   **Database**: SQLite (for lightning-fast metadata filtering)

---

## 💻 Installation & Setup

### Prerequisites
*   **Python 3.10+** (3.12 recommended)
*   **FFmpeg** installed and in your system PATH.
*   **CUDA-capable GPU** (highly recommended; ~8GB VRAM for SAM 3).

### Quick Start (using `uv`)
We highly recommend [uv](https://astral.sh/uv) for its speed and reliability.

1.  **Clone with Submodules**
    ```bash
    git clone --recursive https://github.com/tazztone/subject-frame-extractor.git
    cd subject-frame-extractor
    ```
    *Note: Use `git submodule update --init --recursive` if already cloned.*

2.  **Sync Environment**
    ```bash
    uv sync
    ```

3.  **Launch**
    ```bash
    uv run python app.py
    ```
    *Alternatively, on Linux, use:* `./scripts/linux_run_app.sh`

    Access the UI at `http://127.0.0.1:7860`.

### Manual Setup (vEnv)
1. `python -m venv venv`
2. Activate: `. venv/bin/activate` (Linux/Mac) or `. venv\Scripts\activate.ps1` (Windows)
3. `pip install -r requirements.txt`
4. `pip install -e SAM3_repo`

---

## ⌨️ CLI Usage

The application provides a powerful CLI for automated extraction, analysis, and headless operation. Always use `uv run` to ensure the correct environment.

### Extraction
Extract thumbnails and detect scenes from a video:
```bash
uv run python cli.py extract --video path/to/video.mp4 --output ./results --nth-frame 10
```
- **Caching**: Subsequent runs with identical settings will skip automatically using fingerprints.
- **Force**: Use `--force` to re-extract even if a fingerprint match is found.
- **Clean**: Use `--clean` to delete the output directory before starting.

### Analysis
Run the full AI pipeline (seeding, tracking, metrics) on an existing extraction:
```bash
uv run python cli.py analyze --session ./results --video path/to/video.mp4 --face-ref person.png --resume
```
- **Resume**: Use `--resume` to skip already completed scenes (uses `progress.json`).

### Full Pipeline
Run extraction and analysis in one command:
```bash
uv run python cli.py full --video video.mp4 --output ./results --face-ref person.png
```

### Status
Check the progress and metadata of a session:
```bash
uv run python cli.py status --session ./results
```

### Photo Mode (Culling)
Process image folders and sync ratings to sidecars:
```bash
# 1. Ingest folder (crawls images, extracts RAW previews)
uv run python cli.py photo ingest --folder /path/to/raws --output ./photo_session

# 2. Score photos (sharpness, naturalness, face prominence, etc.)
uv run python cli.py photo score --session ./photo_session

# 3. Export XMP sidecars (Ratings & Labels compatible with Lightroom)
uv run python cli.py photo export --session ./photo_session
```


---

## 📖 Usage Guide

1.  **Source**: Upload a video or paste a YouTube URL. Choose your extraction resolution.
2.  **Extract**: Run the extraction. The tool identifies scenes and generates thumbnails.
3.  **Define Subject**: 
    *   **By Face**: Upload a reference photo for similarity matching.
    *   **By Text**: Enter a description (e.g., "cat", "person in red").
    *   **Auto**: Let the AI select the most prominent subject.
4.  **Analyze**: Review "Scene Seeds". Run **Propagation** to track subjects through the video.
5.  **Filter**: Use sliders in the **Metrics & Filtering** tab to curate your dataset.
6.  **Export**: Select your crop settings and aspect ratio, then hit **Export**.

---

# 🧑‍💻 Developer Guide

For detailed information on architecture, critical rules (Agent Memory), development workflows, and testing, please refer to the [AGENTS.md](AGENTS.md).

---

## 📄 License
MIT License. See [LICENSE](LICENSE) for details.
