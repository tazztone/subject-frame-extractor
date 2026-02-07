---
phase: 3
plan: 3
wave: 2
---

# Plan 3.3: XMP Export & Scoring Logic

## Objective
Implement the IQA scoring pipeline for photos and the XMP sidecar export functionality.

## Context
- .gsd/SPEC.md (P2: Photo Mode MVP)
- .gsd/phases/3/RESEARCH.md (XMP generation, Custom Scoring)
- core/operators/ (existing IQA metrics)
- core/photo_utils.py (from Plan 3.1)

## Tasks

<task type="auto">
  <name>Implement Photo Scoring Logic</name>
  <files>core/photo_scoring.py (NEW)</files>
  <action>
    - Create `core/photo_scoring.py`.
    - Implement `score_photo(preview_path: Path, weights: Dict[str, float]) -> Dict[str, Any]`:
      - Load the preview image.
      - Call existing operators: `SharpnessOperator`, `EntropyOperator`, `NIQEOperator`.
      - Calculate a weighted `quality_score` using the formula from RESEARCH.md.
      - Return `{"scores": {metric: value, ...}, "quality_score": float}`.
    - Implement `apply_scores_to_photos(photos: List[Dict], weights: Dict) -> List[Dict]`:
      - For each photo, call `score_photo` and update its `scores` field.
  </action>
  <verify>python -c "from core.photo_scoring import score_photo; print(score_photo.__doc__)"</verify>
  <done>`score_photo` and `apply_scores_to_photos` are importable.</done>
</task>

<task type="auto">
  <name>Implement XMP Sidecar Writer</name>
  <files>core/xmp_writer.py (NEW)</files>
  <action>
    - Create `core/xmp_writer.py`.
    - Implement `write_xmp_sidecar(source_path: Path, rating: int, label: str) -> Path`:
      - Use `xml.etree.ElementTree` to generate XMP (as prototyped in tests/research).
      - Write to `source_path.with_suffix(".xmp")`.
      - Return the path to the XMP file.
    - Implement `export_xmps_for_photos(photos: List[Dict], star_thresholds: List[int]) -> int`:
      - For each photo, map `quality_score` to `rating` (1-5 stars).
      - Map `status` to `label` (Green=Kept, Red=Rejected).
      - Call `write_xmp_sidecar`.
      - Return count of files written.
  </action>
  <verify>python -c "from core.xmp_writer import write_xmp_sidecar; print(write_xmp_sidecar.__doc__)"</verify>
  <done>`write_xmp_sidecar` and `export_xmps_for_photos` are importable and tested.</done>
</task>

<task type="checkpoint:human-verify">
  <name>Verify XMP in Lightroom</name>
  <action>
    - User imports a folder of RAWs using the UI.
    - User clicks "Sync XMP".
    - User opens the folder in Lightroom/Capture One.
    - User confirms that ratings and labels are visible.
  </action>
  <done>User confirms XMPs are correctly read by Lightroom.</done>
</task>

## Success Criteria
- [ ] `score_photo` returns a dict with `quality_score` and individual metric scores.
- [ ] `export_xmps_for_photos` writes `.xmp` files next to source RAWs.
- [ ] User verifies XMPs in Lightroom.
