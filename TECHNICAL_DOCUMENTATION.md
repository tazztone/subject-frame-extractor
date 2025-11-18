# Technical Documentation: Frame Extractor & Analyzer v2.0

## 1. System Overview
`app.py` is a monolithic, event-driven application designed for high-fidelity video frame extraction, subject tracking, and quality analysis. Built on **Gradio**, it integrates state-of-the-art computer vision models (SAM2, GroundingDINO, YOLO, InsightFace) to automate the curation of video datasets.

The system follows a **Composition Root** pattern for dependency injection and utilizes a generator-based pipeline to stream real-time progress to the UI.

## 2. Core Infrastructure

### 2.1 Configuration & State
*   **`Config` (Pydantic):** Centralizes settings for paths, model URLs, thresholds, and UI defaults. Supports loading from `config.json` and environment variables.
*   **`ModelRegistry`:** A thread-safe singleton that handles lazy loading, caching, and GPU resource management for heavy ML models.
*   **`AppLogger`:** A custom logging engine that simultaneously outputs colored console logs, structured JSONL files for programmatic audit, and streams log events to the UI via a `Queue`.

### 2.2 Data Structures
*   **`Frame`:** Represents a single image with associated pixel data and computed `FrameMetrics` (sharpness, contrast, NIQE, etc.).
*   **`Scene`:** Defines a temporal shot (start/end frames) and stores "Seeding" results (bounding boxes, masks) used for propagation.
*   **`AnalysisParameters`:** A consolidated Dataclass mapping UI inputs to backend pipeline arguments.

## 3. Processing Pipelines

The application logic is segmented into distinct processing stages managed by the `Pipeline` base class.

### 3.1 Stage I: Ingestion & Extraction
*   **`VideoManager`:** Handles input validation and `yt-dlp` integration for YouTube downloads.
*   **`ExtractionPipeline`:**
    *   Uses **PySceneDetect** to segment the video into logical shots.
    *   Wraps **FFmpeg** to extract frames based on strategies (Keyframes, Interval, Every Nth).
    *   Generates lightweight thumbnails for the UI to minimize memory overhead during the selection phase.

### 3.2 Stage II: Subject Definition (Seeding)
This stage identifies *what* to track in a specific scene. The `SeedSelector` class implements four strategies:
1.  **Identity-First:** Matches subjects against a reference face using **InsightFace**.
2.  **Text-Prompt:** Uses **GroundingDINO** to find objects matching a text description.
3.  **Automatic (Prominent Person):** Uses **YOLO** to identify subjects based on heuristics (Largest, Center-most, Highest Confidence).
4.  **Hybrid:** Attempts Face matching, falling back to Text or Auto if confidence is low.

### 3.3 Stage III: Propagation & Masking
Once a "Seed" (bounding box) is selected for a scene:
*   **`MaskPropagator`:** Initializes **DAM4SAM (SAM2)** on the seed frame.
*   **Bi-Directional Tracking:** Propagates the segmentation mask forward and backward through time to cover the entire shot.
*   **`SubjectMasker`:** Orchestrates the interaction between the `SeedSelector` and `MaskPropagator`.

### 3.4 Stage IV: Analysis & Metrics
*   **`AnalysisPipeline`:** Iterates through processed frames to compute:
    *   **Reference-less Quality:** NIQE (via `pyiqa`), Laplacian Sharpness, Entropy.
    *   **Content Metrics:** Face similarity, Subject Mask Area percentage.
    *   **Deduplication Hashes:** pHash generation.

## 4. Post-Processing & Export

*   **Vectorized Filtering:** Uses `numpy` for high-performance filtering of thousands of frames based on metric thresholds (e.g., "Keep frames with Sharpness > 50 AND Face Similarity > 0.6").
*   **Deduplication:** Filters redundant frames using pHash, SSIM, or LPIPS comparisons.
*   **Export:** Uses FFmpeg to save the final filtered dataset. Includes an optional **Cropping Engine** that calculates optimal crops (16:9, 1:1) centered on the subject mask.

## 5. User Interface Architecture

The UI is built using `gradio.Blocks` and is separated into two layers:

1.  **`AppUI` (Layout):** Defines the visual structure, tabs (Extraction, Subject, Scene Selection, Filtering), and component instantiation.
2.  **`EnhancedAppUI` (Logic):** Inherits from `AppUI` to add:
    *   **Event Wiring:** Connects UI components to backend pipelines.
    *   **Thread Management:** Uses `ThreadPoolExecutor` to run pipelines without freezing the UI.
    *   **Progress Streaming:** Consumes the logger queue to update the progress bar and status console in real-time.

## 6. Error Handling & Utilities
*   **Resilience:** The `ErrorHandler` class provides `@with_retry` and `@with_fallback` decorators to handle transient failures (e.g., network timeouts during model download).
*   **Resource Management:** Context managers like `safe_resource_cleanup` ensure CUDA cache is emptied and Garbage Collection is triggered to prevent VRAM leaks between pipeline stages.