---
phase: 1
level: 2
researched_at: 2026-02-07
---

# Phase 1 Research: CLI Foundation

## Questions Investigated
1. How are the pipeline functions currently called in headless mode?
2. What are the correct function signatures for the pipeline executors?
3. What verification has already been done?

## Findings

### Pipeline Function Signatures

All pipeline executors follow this pattern:
```python
def execute_*(
    event,                     # The Event model (pydantic)
    progress_queue: Queue,     # For progress updates
    cancel_event: Event,       # threading.Event for cancellation
    logger: AppLogger,         # Logger
    config: Config,            # Application config
    thumbnail_manager,         # ThumbnailManager
    cuda_available: bool,      # GPU flag
    progress=None,             # Optional Gradio progress callback
    model_registry=None,       # ModelRegistry for model caching
) -> Generator[dict, None, None]
```

**Critical Finding:** The `progress` and `model_registry` parameters MUST be passed as **keyword arguments** when called from CLI, not positional. The E2E test (`tests/e2e/e2e_run.py`) demonstrates the correct calling pattern.

**Sources:**
- `/core/pipelines.py` lines 1076-1086, 1234-1244, 1293-1303
- `/tests/e2e/e2e_run.py` lines 150-154, 181-184, 194-197

**Recommendation:** Always use keyword args for optional parameters in CLI.

### Existing E2E Test Pattern

The project already has a working headless execution pattern in `tests/e2e/e2e_run.py`:
- Uses `deque(generator, maxlen=1)[0]` to consume pipeline generators
- Passes CUDA availability as keyword arg
- Handles errors by checking `result.get("done")`

**Recommendation:** CLI should mirror this pattern exactly.

### CLI Created and Verified

A `cli.py` was created during this session using Click with:
- `extract` command — ✅ Works
- `analyze` command — ✅ Works (after kwarg fix)
- `full` command — Wraps both
- `status` command — Shows session state

**Verification Result:**
```
📋 SESSION STATUS: cli_test_output
   ✓ Extraction complete (frame_map.json)
   ✓ Pre-analysis complete (scene_seeds.json)
   ✓ Propagation complete (masks/ 16 items)
   ✓ Analysis complete (metadata.db 36KB)
```

## Decisions Made

| Decision | Choice | Rationale |
|----------|--------|-----------|
| CLI Framework | Click | Clean decorator syntax, good help generation, used widely |
| Pipeline Calling | Keywords-only for optional params | Matches E2E test pattern, avoids positional arg bugs |
| Error Handling | `sys.exit(1)` on failure | Standard CLI behavior, clean for scripting |

## Patterns to Follow
- Always pass `progress=None` and `model_registry=model_registry` as keyword args
- Always check `result.get("done")` before proceeding
- Print progress updates from `unified_log` key

## Anti-Patterns to Avoid
- Positional args for optional pipeline parameters: Causes TypeErrors
- Swallowing errors: Always exit non-zero on failure

## Dependencies Identified

| Package | Version | Purpose |
|---------|---------|---------|
| click | latest | CLI framework (already in project) |

## Risks
- **Face landmark warning:** Non-fatal bug in `_process_single_frame`, doesn't block analysis
  - **Mitigation:** Log as warning, continue processing

## Ready for Planning
- [x] Questions answered
- [x] Approach selected (CLI using Click, mirroring E2E patterns)
- [x] Dependencies identified (none new needed)
- [x] **CLI already implemented and verified working**
