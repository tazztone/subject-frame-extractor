# TODO

## Ongoing / Post-Release Improvements
- [x] ~~Refactor `AppUI` to remove legacy state keys and fully migrate to `ApplicationState`.~~ *(Done in v3.0 Phase 0)*
- [ ] Monitor documentation generation script to ensure it doesn't leak secrets or grow excessively.
- [ ] Create a unified `Makefile` or `justfile` for cross-platform task management (shortcut for run, test, layout).
- [ ] Implement Embeddings Visualization (Cluster Panel): 2D projection (UMAP or t-SNE) of LPIPS/CLIP embeddings for visual frame clustering. `medium` — 2026-02-06
- [ ] Integrate Semantic Search (CLIP-powered Filtering): Text-to-image search field using OpenAI's CLIP for ad-hoc filtering. `medium` — 2026-02-06
- [x] ~~Refactor `AnalysisPipeline` into "Operators" pattern (Extensible Pipelines) for plugin-based metrics and UI updates.~~ *(Done in v3.0 Phase 2)*
- [x] ~~Implement Multi-View Synchronization (State Mirroring)~~ *(Partially done via `ApplicationState` in v3.0 Phase 0)*
- [ ] Add better guidance for agents to avoid using system `python` (use `python3` or `uv run`) and ensure venv awareness. `high` — 2026-02-06
- [x] ~~Sync `requirements.txt` with development and testing dependencies (Playwright, pytest-playwright).~~ *(Consolidated into pyproject.toml)* `low` — 2026-02-07
- [ ] Fix the top right UI "🟡 System Initializing..." never finishing. `high` — 2026-02-08
- [ ] remove the auto tab switching after step completion, because it inhibits UI elements from loading (state issue or smth). `medium` — 2026-02-08
- [ ] **Optimize ARW Ingestion**: Use `ExifTool` to extract efficient `PreviewImage` (150KB) or `ThumbnailImage` (4KB) instead of `rawpy`'s huge `extract_thumb()`. `high` — 2026-02-08
- [ ] **Eliminate Redundant Storage**: stop saving 3x copies (`frame`, `_preview`, `scene`) or ARW images; use 1 base image and overlay masks in-memory or dynamically. `high` — 2026-02-08
- [ ] **Implement Smart Downscaling**: Ensure all thumbnails/previews are ~0.5MP max, regardless of source format or extraction method. `medium` — 2026-02-08
- [ ] **ARW Algorithm**: Algorithm that decides which embedded JPEG to take (Preview vs Thumbnail vs JpgFromRaw) based on size requirements. `medium` — 2026-02-08
- [ ] **Tab 3 Performance**: Analyze why UI recompute is slow after slider changes; optimize for instant filtering. `high` — 2026-02-08
- [ ] **Tab 3 UI Enhancement**: Replace barely visible red bounding boxes on preview scenes with semi-transparent mask overlays. `medium` — 2026-02-08
- [ ] **Fix Obsolete Propagate Button**: Disable or hide "Propagate Masks" button when loading individual images (RAW folders) and fix `ValueError` in `_propagation_button_handler` (return mismatch on failure). `high` — 2026-02-08



## Completed (v2.1-stabilized)
- [x] Investigate E2E failures: Fixed SAM3 tracking loss via text hints and 360p resolution.
- [x] Fix `test_database.py` failures (AttributeError: create_tables).
- [x] Fix `test_pipelines_extended.py` (AttributeError: create_tables).
- [x] Fix `PermissionError` for `/out` in `tests/test_app_ui_logic.py`.
- [x] Fix `test_mask_propagator_logic.py` assertion failure.
- [x] Decompose `execute_pre_analysis`.
- [x] Remove `ui/app_ui.py.bak`.
- [x] Implement `ModelRegistry` watchdog.
- [x] Update `AGENTS.md` with new stabilization instructions.
- [x] Create Linux run script (`scripts/linux_run_app.sh`).
