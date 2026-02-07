---
phase: 4
level: 3
researched_at: 2026-02-07
---

# Phase 4 Research: Polish & Verification

## Questions Investigated
1. What is the current state of E2E verification for Photo Mode?
2. Does the CLI support the new Photo Mode features (ingest/score/export)?
3. Are the UX audit scripts compatible with the latest changes?
4. What documentation is missing?

## Findings

### 1. Verification Gap: Photo Mode Untested
The current test suite focuses entirely on video workflows.
- **UI Tests**: `tests/ui/test_app_flow.py` tests the Video Tab.
    - **Gap**: No `tests/ui/test_photo_flow.py` to verify:
        - Ingest folder selection.
        - Gallery rendering (lazy loading).
        - Scoring slider interactions.
        - Export button state.
- **E2E/CLI Tests**: `tests/e2e/e2e_run.py` tests the video pipeline.
    - **Gap**: No automated backend test for `ingest_folder` -> `score_photo` -> `write_xmp_sidecar`.

**Recommendation**:
1. Create `tests/ui/test_photo_flow.py` using Playwright (mocking the backend to avoid real RAW file dependencies in CI).
2. Create `tests/e2e/test_photo_cli.py` to test the new CLI commands with a small set of sample assets.

### 2. CLI Feature Parity Gap
`cli.py` (v4.0.0) lacks Photo Mode commands.

**Proposed Architecture**:
Use `click` groups to nest photo commands. Reuse `_setup_runtime` for logging/config.

```python
@cli.group()
def photo():
    """Photo Mode: Ingest, Score, and Export."""
    pass

@photo.command()
@click.option("--folder", required=True, type=click.Path(exists=True))
def ingest(folder):
    ...

@photo.command()
@click.option("--input", required=True)
@click.option("--weights", type=str, help="JSON string of weights")
def score(input, weights):
    ...
```

### 3. Broken Verification Scripts
`scripts/run_ux_audit.py` fails because it references `tests/e2e/test_app_flow.py`.
- **Actual Path**: `tests/ui/test_app_flow.py`.
- **Fix**: Update the path in `run_ux_audit.py`.

### 4. Documentation & Version Mismatch
- **README.md**: Completely missing Photo Mode documentation. Needs a new section "📸 Photo Culling".
- **Versioning**: `pyproject.toml` (0.1.0) vs `cli.py` (4.0.0).
- **Decision**: Bump `pyproject.toml` to **0.4.0** (Pre-release for 4.0) or **4.0.0** to match CLI. Given the "v4.0-cli-first" milestone, **4.0.0** is the intended target.

## Decisions Made
| Decision | Choice | Rationale |
|----------|--------|-----------|
| Verification | **Split Suites** | Separate `test_video_flow.py` and `test_photo_flow.py` for specialized UI testing. |
| CLI Structure | **Nested Group** | `python cli.py photo <cmd>` keeps the namespace clean. |
| Versioning | **4.0.0** | Unify all version numbers to match the milestone goal. |

## Risks
- **ExifTool in CI**: UI tests should mock the *result* of ingest (stateless), but E2E CLI tests will fail without `exiftool`.
    - *Mitigation*: Mark E2E tests with `@pytest.mark.skipif(shutil.which("exiftool") is None)`.

## Ready for Planning
- [x] Questions answered
- [x] Approach selected
- [x] Dependencies identified
