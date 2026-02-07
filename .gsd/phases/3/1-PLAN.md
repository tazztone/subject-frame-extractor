---
phase: 3
plan: 1
wave: 1
---

# Plan 3.1: Photo Ingest & Preview Extraction

## Objective
Implement the ability to ingest a folder of images (RAW/JPEG) and extract previews using ExifTool.

## Context
- .gsd/SPEC.md (P2: Photo Mode MVP)
- .gsd/ARCHITECTURE.md (Pipelines, DataLayer)
- .gsd/phases/3/RESEARCH.md (ExifTool findings)
- core/pipelines.py
- core/diagnostics.py

## Tasks

<task type="auto">
  <name>Add ExifTool Check to Diagnostics</name>
  <files>ui/app_ui.py</files>
  <action>
    - Import `shutil`
    - Add a new check for `exiftool`:
      ```python
      exiftool_path = shutil.which("exiftool")
      results["exiftool"] = {
          "available": exiftool_path is not None,
          "path": exiftool_path,
          "version": subprocess.run([exiftool_path, "-ver"], capture_output=True, text=True).stdout.strip() if exiftool_path else None
      }
      ```
    - Log a warning if `exiftool` is not found.
  </action>
  <verify>python -c "from ui.app_ui import AppUI; print('AppUI imported')"</verify>
  <done>Diagnostics output includes an "exiftool" key with availability status.</done>
</task>

<task type="auto">
  <name>Implement PreviewExtractor Utility</name>
  <files>core/photo_utils.py (NEW)</files>
  <action>
    - Create a new module `core/photo_utils.py`.
    - Implement `extract_preview(raw_path: Path, output_dir: Path) -> Optional[Path]`:
      - Use `subprocess.run` to call `exiftool -b -JpgFromRaw -PreviewImage -ThumbnailImage ...`
      - Save the first found preview to `output_dir / (raw_path.stem + "_preview.jpg")`.
      - Return the path to the extracted preview, or None if extraction failed.
    - Implement `ingest_folder(folder_path: Path, output_dir: Path) -> List[Dict]`:
      - Scan `folder_path` for image files (RAW extensions: CR2, NEF, ARW, DNG; JPEG: jpg, jpeg).
      - For each file, call `extract_preview` or copy JPEG directly.
      - Return a list of dicts: `{"source": original_path, "preview": preview_path, "type": "raw"|"jpeg"}`.
  </action>
  <verify>python -c "from core.photo_utils import ingest_folder; print(ingest_folder.__doc__)"</verify>
  <done>`core/photo_utils.py` exists and both functions are importable.</done>
</task>

## Success Criteria
- [ ] `run_system_diagnostics()` reports ExifTool availability.
- [ ] `ingest_folder(Path("test_raws"), Path("test_out"))` returns a list of ingested photos.
