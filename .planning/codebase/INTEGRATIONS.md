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

### SAM3 API Guide (Developer Reference)

The `SAM3Wrapper` in `core/managers.py` encapsulates the official `Sam3VideoPredictor` API. For the primary reference implementation this project follows, see:
`SAM3_repo/examples/sam3_video_predictor_example.ipynb`

#### Core API Usage
```python
from core.managers import SAM3Wrapper
wrapper = SAM3Wrapper(device="cuda")

# 1. Initialize session
wrapper.init_video("/path/to/frames")

# 2. Add Prompt (Wrapper handles absolute -> relative normalization)
mask = wrapper.add_bbox_prompt(frame_idx=0, obj_id=1, bbox_xywh=[x,y,w,h], img_size=(W,H))

# 3. Propagate (Generator yielding frame_idx, obj_id, mask_array)
# direction can be "forward", "backward", or "both"
for frame_idx, obj_id, mask in wrapper.propagate(start_idx=0, direction="both"):
    ...
```

#### Complete API Reference
| Method | Description |
|--------|-------------|
| `init_video(path)` | Initialize session with video or frame folder. |
| `add_bbox_prompt(...)` | Add a bounding box. Auto-converts pixels to relative 0-1. |
| `detect_objects(...)` | Open-vocabulary text detection on single frames. |
| `add_text_prompt(...)` | Add text prompts for video tracking (new API). |
| `add_point_prompt(...)` | Refine masks with positive/negative points. |
| `propagate(...)` | **Generator** yielding propagation results. Supports `direction="both"`. |
| `close_session()` | Release GPU resources. |
| `shutdown()` | Shutdown the predictor, release memory, and clean up all resources. |

---


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
