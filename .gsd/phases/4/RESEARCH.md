---
phase: 4
level: 2
researched_at: 2026-02-07
---

# Phase 4 Research: Polish & Verification

## Questions Investigated
1. What is the current state of E2E verification for Photo Mode?
2. Does the CLI support the new Photo Mode features (ingest/score/export)?
3. Are the UX audit scripts compatible with the latest changes?
4. What documentation is missing?

## Findings

### 1. Verification Gap: Photo Mode Untested by E2E
- The current E2E script (`tests/e2e/e2e_run.py`) only tests video extraction and analysis.
- **Gap**: No automated test for:
    - `ingest_folder` (ExifTool preview extraction).
    - `score_photo` (IQA metrics).
    - `write_xmp_sidecar` (Export).
- **Recommendation**: Create `tests/e2e/test_photo_mode.py` or extend `e2e_run.py` to support a `--photo` mode.

### 2. CLI Feature Parity Gap
- `cli.py` (v4.0.0) has commands: `extract`, `analyze`, `full`, `status`.
- **Gap**: No commands for Photo Mode.
- **Recommendation**: 
    - Add `photo` command group.
    - `photo ingest --folder <path>`
    - `photo score --threshold <N>`
    - `photo export --format xmp`

### 3. Broken Verification Scripts
- `scripts/run_ux_audit.py` references `tests/e2e/test_app_flow.py`.
- **Finding**: Actual file is at `tests/ui/test_app_flow.py`. The script is broken.
- **Action**: Fix path in `run_ux_audit.py`.

### 4. Documentation & Version Mismatch
- `README.md`: No mention of Photo Mode usage.
- `pyproject.toml`: Version is `0.1.0`.
- `cli.py`: Version is `4.0.0`.
- `app.py`: Mentions "Subject Frame Extractor v4.0 (Dev)"? (Assumed based on CLI).
- **Action**: Unify version numbers (Recommend bump to `0.4.0` or `4.0.0` depending on user preference, probably `0.4.0` as it is still dev?). The CLI says "4.0.0". Let's standardize on **4.0.0** to match CLI and generic "v4.0" naming in `STATE.md` ("Milestone: v4.0-cli-first").

## Decisions Made
| Decision | Choice | Rationale |
|----------|--------|-----------|
| E2E Strategy | New `test_photo_e2e.py` | Separate concern from video pipeline; faster to run independently. |
| CLI Strategy | New `photo` command | Keep broad capabilities distinct (video vs photo). |
| Versioning | **4.0.0** | Align `pyproject.toml` with `cli.py` and milestone name. |

## Risks
- **ExifTool Dependency**: E2E tests for photo mode will require ExifTool installed in the test environment (CI/CD).
- **FileSystem Access**: Photo mode tests involve file creation/deletion; ensure cleanup to avoid artifact buildup.

## Ready for Planning
- [x] Questions answered
- [x] Approach selected
- [x] Dependencies identified
