---
phase: 2
plan: 1
wave: 1
---

# Plan 2.1: Fingerprinting Infrastructure

## Objective
Create the fingerprinting system that enables fast detection of repeated runs. When a user runs the same extraction twice with identical settings, the second run should skip to results in <2 seconds.

## Context
- .gsd/SPEC.md
- .gsd/phases/2/RESEARCH.md
- cli.py
- core/pipelines.py

## Tasks

<task type="auto">
  <name>Create Fingerprint Module</name>
  <files>core/fingerprint.py</files>
  <action>
    Create a new module with:
    
    1. `RunFingerprint` dataclass with fields:
       - video_path: str
       - video_size: int
       - video_mtime: float
       - extraction_hash: str (MD5 of extraction settings)
       - analysis_hash: str (MD5 of analysis settings, optional)
       - created_at: str (ISO timestamp)
    
    2. `create_fingerprint(video_path, extraction_settings, analysis_settings=None) -> RunFingerprint`
       - Hash relevant settings dicts with MD5
       - Get file stats for video
    
    3. `save_fingerprint(fingerprint, output_dir)` -> writes `run_fingerprint.json`
    
    4. `load_fingerprint(output_dir) -> Optional[RunFingerprint]`
    
    5. `fingerprints_match(new, existing) -> bool`
       - Compare video path, size, mtime, extraction_hash
       - Return True if all match
    
    Use `hashlib.md5` and `json` from stdlib. No new dependencies.
  </action>
  <verify>python -c "from core.fingerprint import create_fingerprint, save_fingerprint, load_fingerprint, fingerprints_match; print('✓ Module imports')"</verify>
  <done>core/fingerprint.py exists and all 4 functions are importable</done>
</task>

<task type="auto">
  <name>Integrate Fingerprinting into Extraction</name>
  <files>core/pipelines.py</files>
  <action>
    In `execute_extraction`:
    
    1. After successful extraction, create fingerprint:
       ```python
       from core.fingerprint import create_fingerprint, save_fingerprint
       
       fingerprint = create_fingerprint(
           video_path=str(params.source_path),
           extraction_settings={
               "method": params.method,
               "nth_frame": params.nth_frame,
               "max_resolution": params.max_resolution,
               "scene_detect": params.scene_detect,
               "thumb_megapixels": params.thumb_megapixels,
           }
       )
       save_fingerprint(fingerprint, output_dir)
       ```
    
    2. Add this at the END of successful extraction, just before the final yield.
    
    Do NOT add fingerprint checking here — that's for the CLI layer.
  </action>
  <verify>rm -rf cli_test_output && python cli.py extract --video "downloads/example clip 720p 2x.mp4" --output ./cli_test_output --nth-frame 5 && cat cli_test_output/run_fingerprint.json</verify>
  <done>run_fingerprint.json is created after extraction with correct structure</done>
</task>

## Success Criteria
- [ ] `core/fingerprint.py` module exists with 4 functions
- [ ] Extraction creates `run_fingerprint.json` in output directory
- [ ] Fingerprint contains video path, size, mtime, and settings hash
