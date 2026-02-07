---
phase: 2
level: 2
researched_at: 2026-02-07
---

# Phase 2 Research: Caching & Idempotency

## Questions Investigated
1. What caching mechanisms already exist?
2. What's the fastest way to detect if a run can be skipped?
3. How to expose `--resume` in the CLI?

## Findings

### Existing Caching Infrastructure

The project already has partial caching support:

| File | Purpose | Created By |
|------|---------|------------|
| `progress.json` | Tracks completed scene IDs | AnalysisPipeline |
| `run_config.json` | Saves run parameters | execute_pre_analysis |
| `frame_map.json` | Maps frame numbers to files | ExtractionPipeline |
| `scene_seeds.json` | Stores seeding results | PreAnalysisPipeline |
| `scenes.json` | Scene boundaries | ExtractionPipeline |
| `metadata.db` | Analysis results (SQLite) | AnalysisPipeline |

**Current Resume Behavior:**
- `params.resume = True` causes:
  - `progress.json` to be loaded
  - Completed scenes to be filtered out
  - `metadata.db` NOT cleared

**Source:** `/core/pipelines.py` lines 475-487

### Fingerprinting Strategy

For fast re-run detection, we need to hash:
1. Video file path (absolute)
2. Video file size OR modification time
3. Key extraction settings: `nth_frame`, `max_resolution`, `scene_detect`
4. Key analysis settings: `primary_seed_strategy`, `enable_face_filter`

**Fingerprint File:** `run_fingerprint.json`
```json
{
  "video_path": "...",
  "video_mtime": 1234567890.0,
  "video_size": 12345678,
  "extraction_settings_hash": "abc123...",
  "analysis_settings_hash": "def456...",
  "created_at": "2026-02-07T16:00:00"
}
```

**Decision:** Use MD5 hash of settings dict for simplicity. Not cryptographic, but fast.

### CLI Integration

Add to existing commands:
- `--resume` flag on `analyze` and `full` commands
- Auto-detect if fingerprint matches for quick skip

```python
@cli.command()
@click.option("--resume", is_flag=True, help="Resume from previous run, skip completed stages")
def analyze(..., resume):
    pre_event.resume = resume
```

## Decisions Made

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Fingerprint storage | JSON file | Simple, human-readable, matches existing pattern |
| Hash algorithm | MD5 | Fast, sufficient for non-security use |
| Skip detection | Compare fingerprint before loading models | Fail fast = save time |
| Resume granularity | Per-scene | Already supported by `progress.json` |

## Patterns to Follow
- Check fingerprint BEFORE initializing expensive models
- Log "skipping" message when fingerprint matches
- Write fingerprint at START of run, not end (so interrupted runs can resume)

## Anti-Patterns to Avoid
- Hashing video file contents: Too slow for large videos
- Storing fingerprint in SQLite: Adds complexity

## Tasks Required

1. **Create `run_fingerprint.json` writer** — Write at start of extraction
2. **Create fingerprint checker** — Compare on subsequent runs
3. **Add `--resume` flag to CLI** — Wire to `params.resume`
4. **Add `--force` flag to CLI** — Ignore fingerprint, re-run everything
5. **Update status command** — Show fingerprint info

## Verification Criteria

| Criterion | Test |
|-----------|------|
| Fingerprint created | `cli.py extract ...` creates `run_fingerprint.json` |
| Fast skip | Second run with same settings exits in <2 seconds |
| Resume works | Interrupted run completes missing scenes |
| Force works | `--force` ignores fingerprint |

## Dependencies Identified

None new — uses stdlib `hashlib` and `json`.

## Ready for Planning
- [x] Questions answered
- [x] Approach selected
- [x] Dependencies identified
