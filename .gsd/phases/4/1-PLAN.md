# Phase 4: Polish & Verification

**Goal**: Finalize v4.0 release by ensuring CLI feature parity for Photo Mode, robust test coverage, and consistent documentation.

---

## User Review Required

> [!IMPORTANT]
> **Versioning**: This plan bumps the project version to **4.0.0** across `pyproject.toml` and `cli.py`.
> **CLI**: New `photo` command group will be added using existing core functions.

---

## Proposed Changes

### Plan 4.1: CLI Enhancements (Photo Mode)

#### [MODIFY] [cli.py](file:///home/tazztone/_coding/subject-frame-extractor/cli.py)

Add a `photo` command group with three subcommands:

| Command | Wraps Core Function | Description |
|---------|---------------------|-------------|
| `photo ingest` | `photo_utils.ingest_folder()` | Scan folder and extract RAW previews |
| `photo score` | `photo_scoring.apply_scores_to_photos()` | Compute quality scores using IQA operators |
| `photo export` | `xmp_writer.export_xmps_for_photos()` | Write XMP sidecars for all photos |

**CLI Interface**:
```bash
# Ingest photos from a folder
uv run python cli.py photo ingest --folder ./downloads/photos --output ./session

# Score ingested photos (requires ingest first)
uv run python cli.py photo score --session ./session --weights '{"sharpness": 0.5, "entropy": 0.3, "niqe": 0.2}'

# Export XMP sidecars
uv run python cli.py photo export --session ./session
```

**Key Design**:
- Reuse `_setup_runtime` for config/logging.
- Session state stored as `photos.json` in output dir.
- No need for mask propagation (single-frame analysis).

---

### Plan 4.2: Testing Infrastructure

#### [NEW] [test_photo_flow.py](file:///home/tazztone/_coding/subject-frame-extractor/tests/ui/test_photo_flow.py)

Playwright UI tests for Photo Mode tab. **Backend is mocked** to avoid ExifTool dependency in CI.

| Test Case | Verifies |
|-----------|----------|
| `test_photo_tab_visible` | Tab renders correctly |
| `test_ingest_button_triggers_backend` | Click invokes handler |
| `test_scoring_sliders_update_state` | Slider changes propagate |
| `test_export_button_enabled_after_score` | Button state logic |

---

#### [NEW] [test_photo_cli.py](file:///home/tazztone/_coding/subject-frame-extractor/tests/e2e/test_photo_cli.py)

E2E tests for CLI commands. **Requires ExifTool** (skip in CI if unavailable).

| Test Case | Verifies |
|-----------|----------|
| `test_ingest_creates_photos_json` | Session file written |
| `test_score_updates_photos_json` | Scores populated |
| `test_export_creates_xmp_files` | `.xmp` sidecars exist |

**CI Safety Pattern**:
```python
import shutil
import pytest

pytestmark = pytest.mark.skipif(
    shutil.which("exiftool") is None,
    reason="ExifTool not found"
)
```

---

#### [MODIFY] [run_ux_audit.py](file:///home/tazztone/_coding/subject-frame-extractor/scripts/run_ux_audit.py)

Fix broken path reference:
```diff
- code, output = run_tests("tests/e2e/test_app_flow.py")
+ code, output = run_tests("tests/ui/test_app_flow.py")
```

---

### Plan 4.3: Documentation & Versioning

#### [MODIFY] [pyproject.toml](file:///home/tazztone/_coding/subject-frame-extractor/pyproject.toml)

Bump version:
```diff
- version = "0.1.0"
+ version = "4.0.0"
```

---

#### [MODIFY] [README.md](file:///home/tazztone/_coding/subject-frame-extractor/README.md)

Add Photo Culling section after "CLI Usage":

```markdown
## 📸 Photo Culling (CLI)

Photo Mode enables quality-based culling for RAW/JPEG image collections.

### Workflow
1. **Ingest** images from a folder:
   ```bash
   uv run python cli.py photo ingest --folder ./photos --output ./session
   ```
2. **Score** images using AI metrics:
   ```bash
   uv run python cli.py photo score --session ./session
   ```
3. **Export** XMP sidecars for Lightroom:
   ```bash
   uv run python cli.py photo export --session ./session
   ```
```

---

## Execution Order

1. **CLI** (Plan 4.1) — Implement core functionality first.
2. **Tests** (Plan 4.2) — Write tests against the new CLI.
3. **Docs** (Plan 4.3) — Polish everything last.

---

## Verification Plan

### Automated Tests

| Command | Suite | CI-Safe |
|---------|-------|---------|
| `uv run pytest tests/ui/test_photo_flow.py -v` | UI (Playwright) | ✅ Yes |
| `uv run pytest tests/e2e/test_photo_cli.py -v` | E2E (CLI) | ⚠️ Requires ExifTool |
| `uv run python scripts/run_ux_audit.py --quick` | UX Audit | ✅ Yes |

### Manual Verification

1. **CLI Round-trip** (with sample photos):
   ```bash
   mkdir -p ./downloads/photos
   # Copy 2-3 RAW/JPEG files into ./downloads/photos

   uv run python cli.py photo ingest --folder ./downloads/photos --output ./session
   uv run python cli.py photo score --session ./session
   uv run python cli.py photo export --session ./session

   # Verify:
   ls ./session/photos.json  # Should exist
   ls ./downloads/photos/*.xmp  # Should have XMP sidecars
   ```

2. **Lightroom Import** (optional):
   - Open Lightroom and import the folder `./downloads/photos`.
   - XMP ratings and labels should appear automatically.

---

## Success Criteria

- [ ] `cli.py photo --help` shows `ingest`, `score`, `export` subcommands.
- [ ] `photos.json` persists session state between CLI calls.
- [ ] `.xmp` sidecars are written next to source images.
- [ ] All unit tests pass: `uv run pytest tests/ -v --ignore=tests/e2e/`
- [ ] UI tests pass: `uv run pytest tests/ui/test_photo_flow.py -v`
- [ ] Version is `4.0.0` in both `pyproject.toml` and `cli.py`.
