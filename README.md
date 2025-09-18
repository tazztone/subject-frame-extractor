# Advanced Frame Extractor & Filter

### Overview
Extract and score high-quality still frames from YouTube links or local videos, with an interactive Gradio UI for one-click analysis, live filtering, and export of the best shots. The pipeline streams frames from **FFmpeg**, computes multi-metric quality scores, optionally applies face presence and face similarity filters via **insightface**, and now includes advanced **Subject Masking with SAM2** for subject-focused analysis. It writes frames plus JSONL metadata for reproducible workflows.

### Highlights
- Download from YouTube via yt-dlp or open local files; resumes and caches analysis when metadata is present.
- Four extraction modes: keyframes (I-frames), interval, scene-change, or all frames, selected through FFmpeg filters.
- Quality metrics per frame: sharpness, edge strength, contrast, brightness, entropy, with configurable weights and pre-filters during extraction.
- Optional face tools: face presence pre-filter and face similarity distance to a reference image using **insightface** embeddings, CPU or CUDA providers via ONNX Runtime.
- **Advanced Subject Masking:** Utilize SAM2 (Segment Anything Model v2) to automatically detect and track a primary subject across video shots, enabling subject-focused quality analysis and filtering. Requires CUDA-enabled GPU.
- Multithreaded producer-consumer pipeline with optional Numba acceleration for metric computation, and a live table and gallery preview in the UI.
- Export of kept frames to a timestamped folder alongside the run directory for downstream use.

### Requirements
- System: **FFmpeg** on PATH is mandatory; the app aborts at startup if missing.
- Python: this app uses type unions like FrameMetrics | None and requires Python 3.10+ semantics in practice.
- Packages: gradio, opencv-python-headless, numpy, insightface, onnxruntime, numba, yt-dlp, **torch**, **sam2**; `insightface`, `yt-dlp`, `torch`, and `sam2` are optional but enable face features, URL downloads, and subject masking respectively. `numba` improves performance when available.

### Installation
- Create and activate a virtual environment, then install the packages noted at the top of the script.
- Example:
  - `pip install gradio opencv-python-headless numpy insightface onnxruntime numba yt-dlp torch sam2`
- Ensure **FFmpeg** is installed and discoverable on PATH before launching.

### Quick start
- Launch the UI:
  - `python app.py`
- In the app:
  - Paste a YouTube URL or choose a local file or upload a video in “Video Source,” choose a method, adjust pre-filters, and click “Extract & Analyze Frames”.
  - Watch logs, progress, and the live frame-status table during processing; when done, switch to “Interactive Filtering & Export” to refine kept frames and export.

### Extraction methods
- Keyframes: `select='eq(pict_type,I)'` to grab I-frames only; efficient for edited content.
- Interval: `fps=1/N` to sample frames every N seconds; good for long, steady footage.
- Scene: `select='gt(scene,T)'` with T=0.5 if Fast Scene Detect is on, else 0.4; picks cuts or large content changes.
- All: no select filter; processes every decoded frame, which can be heavy for high-FPS sources.

### Quality metrics and weights
- Metrics computed per frame: sharpness (variance of Laplacian), edge strength (Sobel magnitude), brightness (mean), contrast (std/mean), entropy (histogram-based).
- Weights default to sharpness 30, edge 20, contrast 20, brightness 10, entropy 20, with normalization constants for sharpness and edge strength, and a combined quality score.
- Adjust weights under “Customize Quality Weights,” and optionally enable overall or per-metric pre-filters to discard poor frames during extraction.

### Face analysis and similarity
- Face pre-filter: requires `insightface`; a frame passes if it contains a face above Min Face Confidence and above Min Face Area (% of frame), both user-configurable.
- Similarity: if enabled and a reference face is provided, the app computes a cosine distance between L2-normalized embeddings; lower distance is more similar, and the filtering tab provides a Max Face Distance slider (default 0.5).
- Models: `buffalo_l`, `buffalo_s`, `buffalo_m`, `antelopev2`; providers default to CPU, with optional CUDA provider when “Use GPU (Faces/Masks)” is enabled and a compatible ONNX Runtime is installed.

### Subject Masking
- **Enable Subject-Only Metrics:** When enabled, the application uses SAM2 to identify and track a primary subject throughout the video. Quality metrics are then calculated *only* within the masked region of the subject, providing more focused analysis.
- **Requirements:** Requires `torch`, `sam2` libraries, and a **CUDA-enabled GPU**.
- **Masking Model:** Currently supports `sam2`.
- **SAM2 Model Variant:** Choose between `hiera_large` and `hiera_base`.
- **SAM2 Weights Path:** Specify the path to your downloaded SAM2 model weights (e.g., `C:\models\sam2_hiera_large.pt`).
- **Scene Detection:** Leverages `scenedetect` to break the video into shots, improving masking accuracy and robustness across scene changes.
- **ReID Fallback:** If the primary subject's identity is lost during tracking, the system attempts to re-identify it using face embeddings (if face analysis is enabled) or IoU (Intersection over Union) with previous masks.
- **IoU Fallback Threshold:** Configurable threshold for IoU fallback, determining the minimum overlap to keep tracking an object if its ID is lost.

### Outputs
- Run folder: `downloads/<video_stem>/` contains frames (JPG by default, PNG if toggled) and a `metadata.jsonl` file with one JSON record per processed frame.
- Metadata includes: `frame_number`, `filename`, per-metric scores, overall quality score, face similarity distance if computed, max face confidence if detected, plus any error string per frame. Masking metadata (mask path, shot ID, ReID similarity, mask area percentage, etc.) is also included if subject masking is enabled.
- Logs: `logs/frame_extractor.log` captures run-time diagnostics and errors for troubleshooting.

### Interactive filtering and export
- Open “3. Interactive Filtering & Export” to filter by overall quality or per-metric thresholds, and optionally by face distance threshold for similarity, and by identity similarity from the masking stage.
- The gallery previews up to 100 kept frames; the stats box shows kept/total counts and primary discard reasons from the filter pass (quality, face, identity, or first failing metric).
- Click “Export Kept Frames” to copy all filtered frames to `<run>_exported_<timestamp>` beside the analysis folder.

### Performance tips
- Prefer keyframes or scene modes for edited/cinematic content and interval for surveillance/time-lapse; “all” may be expensive for high-FPS videos.
- Enable quality and/or face pre-filters during extraction to avoid writing low-value frames to disk and reduce downstream I/O.
- Use GPU only if a CUDA-capable system with a compatible ONNX Runtime build is present; otherwise keep the default CPU providers.
- **Subject Masking:** This feature is computationally intensive and requires a powerful CUDA-enabled GPU for reasonable performance.

### YouTube downloads and resolution
- If a URL is provided, the app uses `yt-dlp` and merges AVC video with M4A audio into MP4 by default; a max height can be selected for downloads to reduce size.
- Already-downloaded videos are reused from the downloads directory using the video ID pattern when present, enabling faster resumes.

### What changed vs the old README
- Face library: switched from “DeepFace + TensorFlow CUDA” to **insightface + ONNX Runtime** (CPU or CUDA providers) for detection and embeddings.
- **Subject Masking:** Added advanced subject masking capabilities using SAM2 for subject-focused analysis.
- App form: this code is a single-file Gradio app (`app.py`) without a separate CLI package or python -m entry points; launch via `python app.py`.
- Exports: frame export is supported; montage/thumbnail sheet export is not implemented in this build of the app.
- Presets and resume: UI includes Config Presets (save/load/delete/reset) and “Resume/Use Cache” defaults enabled to reuse `metadata.jsonl` when present.

### Mapping old CLI flags to the UI
- `--method {interval,all,keyframes,scene}` → Method dropdown.
- `--interval N` → Interval (s) input (visible when Method=interval).
- `--quality-weights S E C B ENT` → “Customize Quality Weights” sliders.
- `--png` → “Save as PNG” toggle.
- `--enable-face-filter --face-ref-img IMG` → “Enable Face Similarity Analysis” plus reference image path or upload.
- `--resume` → “Resume/Use Cache” toggle.
- **New:** `--enable-subject-mask` → “Enable Subject-Only Metrics” checkbox.
- **New:** `--masking-model sam2` → “Masking Model” dropdown.
- **New:** `--sam2-model-variant {hiera_large, hiera_base}` → “SAM2 Model Variant” dropdown.
- **New:** `--sam2-weights-path PATH` → “SAM2 Weights Path” textbox.
- **New:** `--scene-detect` (for masking) → “Scene Detection” checkbox within Subject Masking options.
- **New:** `--reid-fallback` → “Enable ReID Fallback” checkbox.
- **New:** `--iou-threshold N` → “IoU Fallback Threshold” slider.

### Troubleshooting
- “FFMPEG is not installed or not in PATH”: install FFmpeg and ensure the shell can find it, then relaunch.
- “yt-dlp not found”: YouTube URL downloads will be disabled; install yt-dlp to enable URL-based sources.
- “insightface not found” or face errors: face features will be disabled; install insightface and ONNX Runtime, and toggle GPU only if CUDA is available.
- **“SAM2 dependencies (torch, sam2) are not installed” or “CUDA not available”**: Subject masking will be disabled; install `torch` and `sam2`, and ensure you have a CUDA-enabled GPU with appropriate drivers.
- Stuck or long runs: reduce resolution for downloads, prefer keyframes/scene instead of all, and enable pre-filters to cut low-value frames.

### Contributing
- The current repository structure is a single-file `app.py`; contributions can focus on bug fixes, UI/UX improvements, new filters, or optional montage export, with PRs that keep the `app.py` monolithic per the header comment.

### License
- MIT