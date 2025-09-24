# 🎬 Frame Extractor & Analyzer

This is a comprehensive, user-friendly tool built with Gradio for intelligently extracting, analyzing, and filtering frames from video files or YouTube links. It leverages powerful AI models to go beyond simple frame grabs, allowing you to find the highest-quality shots of a specific person or subject.

Whether you're creating a dataset, searching for the perfect thumbnail, or analyzing video content, this tool provides a streamlined workflow from raw video to a curated set of high-quality images.

-----

## ✨ Key Features

  * **Flexible Frame Extraction**:
      * Supports both local video files and YouTube URLs.
      * Multiple extraction methods: every frame, fixed intervals, keyframes only, or automatic scene change detection.
  * **Advanced Frame Analysis**:
      * **Quality Scoring**: Each frame is scored on multiple metrics (sharpness, contrast, brightness, edge strength, entropy) which are combined into a weighted "Overall Quality" score.
      * **Face Similarity Filtering**: Provide a reference image of a person, and the tool will use the `insightface` model to find frames containing that specific person, scoring each by similarity.
      * **AI-Powered Subject Masking**: Utilizes the **DAM4SAM** tracking model (powered by SAM 2.1) to automatically generate a segmentation mask for the main subject in each scene. This allows for subject-focused quality analysis.
  * **Intuitive Filtering & Export**:
      * An interactive gallery allows you to filter frames in real-time using sliders for quality, face similarity, and individual metrics.
      * Export the curated set of frames with a single click.
      * **Smart Cropping**: Automatically crop the exported frames to the subject's mask with padding and target aspect ratios (e.g., 16:9, 1:1, 9:16).
  * **Efficient & Resumable**:
      * The analysis pipeline is resumable. If you adjust filtering parameters, you don't need to re-process the frames.
      * Uses parallel processing and Numba-optimized functions for faster analysis.
      * Automatically downloads required AI models on first run.

-----

## ⚙️ Core Technologies

This application integrates several state-of-the-art libraries and models:

  * **Backend & UI**: Python, Gradio
  * **Video Processing**: FFmpeg, OpenCV
  * **AI Models**:
      * **Subject Tracking**: `DAM4SAM` with a `SAM 2.1` backend for zero-shot tracking and segmentation.
      * **Face Recognition**: `insightface` (`buffalo_l` model) for high-accuracy face detection and embedding comparison.
      * **Person Detection**: `YOLOv11` for quickly identifying people to help seed the subject tracker.
  * **Performance**: PyTorch (CUDA-accelerated), ONNX Runtime, Numba

-----

## 🚀 Getting Started

### Prerequisites

Before you begin, ensure you have the following installed on your system:

1.  **Python**: Version 3.10 or newer.
2.  **Git**: Required for cloning repositories during setup.
3.  **NVIDIA GPU (Highly Recommended)**: For full functionality (Subject Masking, Face Analysis), a modern NVIDIA GPU with **CUDA** is required. The tool will run in a limited, CPU-only mode without it.
4.  **FFmpeg**: Must be installed and accessible from your system's PATH. The application will check for it on startup. You can download it from [ffmpeg.org](https://ffmpeg.org/download.html).

### Installation

This repository includes a `windows_install.bat` script to automate the setup process on Windows.

1.  **Clone the Repository**:

    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```

2.  **Run the Installation Script**:
    Simply double-click the `windows_install.bat` file.

    This script will perform the following steps:

      * Clone the required `DAM4SAM` repository.
      * Create a Python virtual environment (`venv`) to keep dependencies isolated.
      * Install all necessary Python packages, including PyTorch with CUDA 12.9 support.

    > **Note:** The first time you run the script, it may take several minutes to download all the packages.

-----

### Custom Installation (CPU-only or different CUDA version)

If you don't have an NVIDIA GPU or have a different CUDA version, you must modify the `windows_install.bat` script **before** running it.

1.  Open `windows_install.bat` in a text editor.
2.  Find this line:
    ```bat
    uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    ```
3.  **For CPU-only installation**, replace it with:
    ```bat
    uv pip install torch torchvision
    ```
4.  **For a different CUDA version**, visit the [PyTorch website](https://pytorch.org/get-started/locally/) to find the correct command for your setup and replace the line accordingly.

-----

## 🖥️ How to Use

1.  **Activate the Virtual Environment**:

      * On **Windows**: Open a command prompt in the project folder and run `venv\Scripts\activate.bat`
      * On **Linux/macOS**: Open a terminal and run `source venv/bin/activate`

2.  **Launch the Application**:

    ```bash
    python app_small.py
    ```

3.  Open your web browser and navigate to the local URL provided (usually `http://127.0.0.1:7860`).

### The 3-Step Workflow

The UI is organized into three tabs that guide you through the process.

#### 탭 1: 📹 Frame Extraction

This is where you provide your video source.

1.  **Video Source**: Paste a YouTube URL or a local file path (e.g., `C:\videos\my_video.mp4`). You can also use the upload button.
2.  **Extraction Settings**:
      * **Method**: Choose how to select frames. `interval` is good for general use, while `scene` is excellent for finding unique shots.
      * **DL Res**: If using a YouTube URL, you can limit the download resolution to save time and space.
      * **Save as PNG**: PNG is lossless but creates larger files. JPG is smaller but has compression artifacts.
3.  **Start Extraction**: Click the button to begin. `ffmpeg` will run in the background. The output frames will be saved to a new folder inside the `downloads` directory.

#### Tab 2: 🔍 Frame Analysis

Once extraction is complete, the output folder is automatically passed to this tab. Here, you configure the AI-powered analysis.

1.  **Input**: The frames folder and original video path should be pre-filled.
2.  **Analysis Settings**:
      * **Enable Face Similarity**: Check this to activate filtering by person. You must provide a clear **Reference Image** of the person you want to find.
      * **Enable Subject-Only Metrics**: Check this to use the DAM4SAM model. This ensures quality metrics (like sharpness) are calculated only on the main subject, ignoring blurry backgrounds.
3.  **Start Analysis**: This is the most computationally intensive step. The application will process each frame to calculate quality scores, find faces, and generate subject masks. The results are saved to a `metadata.jsonl` file in the frames folder.

#### Tab 3: 🎯 Filtering & Export

After analysis, this tab becomes active. It's your workspace for finding the best frames.

1.  **Filter Controls**:
      * Use the **Min Quality** and **Min Face Sim** sliders to narrow down the results. The gallery will update automatically.
      * You can switch to "Individual Metrics" mode to filter by specific criteria like sharpness or contrast.
      * You can also adjust the weights that contribute to the overall quality score.
2.  **Results Gallery**: Shows a preview of the frames that match your current filter settings.
3.  **Export Kept Frames**: Once you are satisfied, click this button.
      * A new folder named `..._exported_...` will be created.
      * If **Crop to Subject** is enabled, each exported image will be intelligently cropped around the detected subject mask, perfect for creating portraits or consistently framed datasets.

-----

## 🛠️ Technical Details

  * **Project Structure**: The application will create several directories in its root folder upon first run:
      * `DAM4SAM/`: A clone of the DAM4SAM repository.
      * `venv/`: The isolated Python virtual environment.
      * `downloads/`: Where YouTube videos and extracted frames are stored.
      * `models/`: Caches for the downloaded AI models (YOLO, InsightFace, SAM).
      * `logs/`: Contains the main application log file.
      * `configs/`: Stores saved UI setting presets.
  * **Metadata**: The `AnalysisPipeline` generates a `metadata.jsonl` file. This is a text file where each line is a JSON object containing all the calculated data for a single frame (metrics, face similarity, mask path, etc.). The Filtering tab reads this file directly, making the filtering process fast and efficient without reloading images or models.
  * **Caching and Resuming**: The application is designed to be stateful. The `config_hash` in the metadata header allows the app to verify if the existing analysis results are compatible with the current settings, enabling you to safely resume your work.