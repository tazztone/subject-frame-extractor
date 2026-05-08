# Subject Frame Extractor

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-CUDA-ee4c2c)](https://pytorch.org/)
[![Gradio](https://img.shields.io/badge/UI-Gradio%206.x-ff5000)](https://gradio.app/)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow)](LICENSE)

An AI-powered tool for extracting, analyzing, and filtering high-quality frames from video footage. Designed for dataset builders (LoRA / Dreambooth training), content creators, and researchers who need curated image sets from raw video — not just raw frame dumps.

Also includes a **Photo Culling** mode for scoring and rating RAW photo libraries.

## Tech Stack

| Layer | Technology |
|---|---|
| Runtime | Python 3.10+ (3.12 recommended) |
| UI | Gradio 6.x |
| Segmentation | SAM 3 (Segment Anything Model 3, Facebook Research) |
| Object detection | YOLO (80 COCO classes) |
| Face analysis | InsightFace (similarity matching, blink detection, head pose) |
| Quality scoring | NIQE (perceptual), Laplacian variance (sharpness), pHash + LPIPS (dedup) |
| Video / media | FFmpeg, yt-dlp |
| RAW processing | ExifTool (embedded preview extraction — no demosaicing) |
| Data | PyTorch, NumPy, OpenCV, SQLite, Pydantic |
| Dependency management | uv |

## Architecture

```
/
├── app.py                  # Gradio UI entry point
├── cli.py                  # Headless CLI (extract / analyze / full / status / photo)
├── core/
│   ├── config.py           # Full configuration schema (Pydantic)
│   ├── extractor.py        # Extraction strategies (keyframe, interval, scene, Nth)
│   ├── analyzer.py         # AI analysis pipeline (SAM seeding, tracking, metrics)
│   ├── tracker.py          # Subject tracking across scenes
│   ├── face.py             # InsightFace integration
│   ├── quality.py          # NIQE, sharpness, entropy, LPIPS scoring
│   ├── dedup.py            # pHash + LPIPS deduplication
│   ├── photo.py            # Photo culling: RAW ingest, scoring, XMP sidecar export
│   └── database.py         # SQLite session metadata
├── SAM3_repo/              # SAM 3 submodule
└── scripts/
    ├── linux_run_app.sh
    └── setup scripts
```

**Pipeline:** Extract frames → scene segmentation → AI seeding (face ref / text / YOLO) → SAM 3 propagation → quality metrics → interactive filtering → AR-aware crop export.

## Key Features

- **Extraction strategies** — keyframes, fixed intervals, scene-based, every Nth frame; YouTube URL support
- **Multi-class tracking** — find and track any of 80 COCO objects via YOLO + SAM 3; open-vocabulary text descriptions
- **Face matching** — find every frame of a specific person using InsightFace reference photo
- **Quality filtering** — interactive sliders for sharpness, contrast, NIQE perceptual score
- **Smart deduplication** — pHash + LPIPS removes near-identical frames per scene
- **AR-aware export** — subject-centred crops in 1:1, 9:16, 16:9, or custom ratios
- **Photo culling mode** — RAW preview extraction (CR2, NEF, ARW, DNG, ORF…), AI scoring, export to Lightroom/Capture One XMP sidecar star ratings

## Quick Start

**Prerequisites:** Python 3.10+, FFmpeg in PATH, CUDA GPU recommended (~8 GB VRAM for SAM 3)

```bash
git clone --recursive https://github.com/tazztone/subject-frame-extractor.git
cd subject-frame-extractor
uv sync

# Launch Gradio UI
uv run python app.py
# → http://127.0.0.1:7860
```

## CLI Usage

```bash
# Extract frames
uv run python cli.py extract --video video.mp4 --output ./results --nth-frame 10

# Run AI analysis (with face reference)
uv run python cli.py analyze --session ./results --video video.mp4 --face-ref person.png --resume

# Full pipeline in one command
uv run python cli.py full --video video.mp4 --output ./results --face-ref person.png

# Photo culling workflow
uv run python cli.py photo ingest --folder /path/to/raws --output ./photo_session
uv run python cli.py photo score --session ./photo_session
uv run python cli.py photo export --session ./photo_session   # → XMP sidecars
```

## Configuration

See `core/config.py` for the full Pydantic schema. Key settings:

| Category | Key Fields | Default |
|---|---|---|
| Paths | `logs_dir`, `models_dir`, `downloads_dir` | `logs`, `models`, `downloads` |
| Models | `face_model_name`, `tracker_model_name` | `buffalo_l`, `sam3` |
| Performance | `analysis_default_workers`, `cache_size` | `4`, `200` |

See [AGENTS.md](AGENTS.md) for architecture details, critical rules, and development guidelines.

## License

MIT — see [LICENSE](LICENSE).

---

## Technical Debt & Roadmap

This project uses a semi-automated TODO tracking system to prioritize refactors and features.

- **Check Current Debt**: Run `uv run python scripts/generate_todo_report.py` to generate `TODO_REPORT.md`.
- **Top 20 Summary**:
    1.  [High] Refactor `core/pipelines.py` to use modular `core/managers`. (In Progress)
    2.  [High] Implement thread-safe model access for InsightFace.
    3.  [Medium] Add temporal consistency smoothing between frames in `MaskPropagator`.
    4.  [Medium] Add adaptive quality thresholds based on propagation distance.
    5.  [Low] Support demosaicing for RAW photo ingest (currently uses previews).

