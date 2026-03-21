# External Integrations

**Analysis Date:** 2026-03-21

## APIs & External Services

**Model Repositories:**
- Hugging Face Hub - used for downloading pre-trained model weights for SAM3, InsightFace, and other analysis models.
  - Client: `huggingface_hub` Python package
  - Auth: Optional `APP_HUGGINGFACE_TOKEN` environment variable for private models or higher rate limits.

## Data Storage

**Databases:**
- SQLite (Local) - Primary metadata store for frame analysis results, quality metrics, and session state.
  - Connection: Local file path (usually `metadata.db` in the run directory).
  - Client: standard `sqlite3` library with WAL mode enabled for concurrent access.
  - Schema: Managed via `core/database.py` with built-in migration logic.

**File Storage:**
- Local Filesystem - Used for all media assets and intermediate processing artifacts.
  - Structure:
    - `frames/`: Extracted video frames (JPEG).
    - `masks/`: SAM3 propagation results (PNG/NPY).
    - `thumbnails/`: Low-resolution previews for UI gallery.
    - `models/`: Cached model weights from Hugging Face.
    - `exports/`: Final rendered video and metadata exports.

## ML Frameworks & Submodules

**Tracking & Segmentation:**
- SAM3 (Segment Anything Model 3) - Core engine for subject tracking.
  - Integration: Git submodule at `SAM3_repo`.
  - Wrapper: `core/managers.SAM3Wrapper` encapsulates the predictor API.

**Biometrics:**
- InsightFace - Used for face detection, embedding generation, and subject consistency.
  - Client: `insightface` Python package.
  - Models: `buffalo_l` or similar, auto-downloaded to `models/`.

## CI/CD & Automation

**CI Pipeline:**
- GitHub Actions - Automated testing and verification workflows.
  - Workflows:
    - `ux-testing.yml`: Runs unit tests, visual regression, and accessibility audits.
    - `auto-merge-jules.yml`: Automation for dependency updates.
    - `update_docs.yml`: Keeps documentation in sync.

## Environment Configuration

**Development:**
- Required env vars: `PYTHONPATH` (often set to include project root), `CUDA_VISIBLE_DEVICES` (optional).
- Secrets: `.env` file for local development (e.g., Hugging Face tokens).

---

*Integration audit: 2026-03-21*
*Update when adding/removing external services*
