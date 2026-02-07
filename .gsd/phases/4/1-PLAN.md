# Phase 4: Polish & Verification

**Goal**: Finalize v4.0 release by ensuring feature parity in CLI, robust testing for new Photo Mode, and consistent documentation.

## User Review Required
> [!IMPORTANT]
> **Versioning**: This plan bumps the project version to **4.0.0** across `pyproject.toml` and `cli.py`.
> **CLI**: New `photo` command group will be added.

## Proposed Changes

### 1. CLI Enhancements (Photo Mode)
**Files**: `cli.py`
- [NEW] Add `photo` command group.
- [NEW] `photo ingest --folder <path>`: Wraps `ingest_folder`.
- [NEW] `photo score --weights <json>`: Wraps `score_photo` logic (reusing `PhotoTab` logic where possible).
- [NEW] `photo export`: Wraps `write_xmp_sidecar`.

### 2. Testing Infrastructure
**Files**: `tests/`
- [NEW] `tests/ui/test_photo_flow.py`: Playwright tests for Photo Mode UI (mocking backend).
- [NEW] `tests/e2e/test_photo_cli.py`: E2E tests for `cli.py photo` commands (requires ExifTool).
- [MODIFY] `scripts/run_ux_audit.py`: Fix path to `tests/ui/test_app_flow.py`.

### 3. Documentation & Versioning
**Files**: `README.md`, `pyproject.toml`
- [MODIFY] `README.md`: Add "📸 Photo Culling" section and CLI usage.
- [MODIFY] `pyproject.toml`: Bump version to `4.0.0`.

## Verification Plan

### Automated Tests
1. **CLI Tests**: `uv run pytest tests/e2e/test_photo_cli.py`
2. **UI Tests**: `uv run pytest tests/ui/test_photo_flow.py`
3. **UX Audit**: `uv run python scripts/run_ux_audit.py` (Verify it runs without error)

### Manual Verification
1. **CLI Round-trip**:
   ```bash
   uv run python cli.py photo ingest --folder ./downloads/photos
   uv run python cli.py photo score --weights '{"sharpness": 1.0}'
   uv run python cli.py photo export
   ```
2. **Verify XMP**: Check `.xmp` sidecars are generated in `./downloads/photos`.
