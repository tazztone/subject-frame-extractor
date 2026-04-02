# Technology Stack

**Analysis Date:** 2026-03-21

## Languages

**Primary:**
- Python 3.10 - 3.12 - All application code (Core and UI)

**Secondary:**
- Shell/PowerShell - Build scripts, environment setup (implied by uv/venv usage)
- JavaScript - Custom Gradio components (if any, though none explicitly found yet)

## Runtime

**Environment:**
- Python 3.10+
- GPU with CUDA support (Highly recommended for ML model performance)

**Package Manager:**
- `uv` - Primary package and environment manager
- Lockfile: `uv.lock` present

## Frameworks

**Core:**
- Gradio - Web interface and application assembly
- Pydantic - Data validation, settings management (`BaseSettings`), and state schemas

**Testing:**
- pytest - Unit, integration, and signature testing
- Playwright - Browser-based E2E and UI verification (via `pytest-playwright`)

**Build/Dev:**
- ruff - Fast linting and formatting
- setuptools - Used for package building and egg-info generation

## Key Dependencies

**Critical:**
- **SAM3**: Segment Anything Model 3 (PVS-first tracking path).
- **YOLO26**: Multi-class (80 COCO classes) detection and segmentation (ONNX format).
- **InsightFace**: `buffalo_l` / `buffalo_s` models for high-accuracy face embedding and quality scoring.
- **MediaPipe**: Used for face mesh analysis (eyes-open validation, yaw/pitch estimation).
- `opencv-python` - Essential image and video processing operations
- `scenedetect` - Automated scene change detection for frame extraction
- `onnxruntime-gpu` - High-performance inference engine for ML models

**Infrastructure:**
- `pydantic-settings` - Environment-aware configuration management
- `huggingface_hub` - Model weight distribution and management
- `numba` / `numpy` - High-performance numerical computing

## Configuration

**Environment:**
- `.env` files (via `python-dotenv`)
- `core/config.py` - Centralized Pydantic `BaseSettings`

**Build:**
- **ONNX Runtime**: Executes YOLO26 and InsightFace models with CUDA acceleration support.
- **DuckDB**: Used for fast, local persistent metadata storage and query during filtering.
- `pyproject.toml` - Project metadata and dependency definitions
- `uv.lock` - Deterministic dependency resolution

## Platform Requirements

**Development:**
- Linux (Primary development environment)
- Windows (Supported via `triton-windows` conditional dependency)
- Python 3.10-3.12 installed

**Production:**
- Local execution/Self-hosted (Gradio app)
- NVIDIA GPU with 8GB+ VRAM (Recommended for SAM3 propagation)

---

*Stack analysis: 2026-03-21*
*Update after major dependency changes*
