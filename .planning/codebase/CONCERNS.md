# Codebase Concerns

**Analysis Date:** 2026-03-21

## Tech Debt

**Status Field Typing:**
- Issue: Scene status (`pending`, `included`, `excluded`) is currently handled as strings.
- Files: `core/models.py`, `core/events.py`
- Impact: Lack of type safety can lead to silent bugs or typos in status transitions.
- Fix approach: Transition to a Python `Enum`.

**Resource-Aware Scheduling:**
- Issue: Batch manager starts tasks without checking available GPU RAM or system memory.
- Files: `core/batch_manager.py`
- Impact: Risk of OOM (Out of Memory) crashes on machines with lower specs.
- Fix approach: Implement a resource-aware queue that waits for sufficient VRAM/RAM availability.

**Parallel Export:**
- Issue: Frame export is currently serial (one frame at a time).
- Files: `core/export.py`
- Impact: Slow export for large jobs (e.g., thousands of frames).
- Fix approach: Implement multi-threaded frame saving and processing.

## Known Bugs

**Gradio Dropdown "Value Error":**
- Symptoms: App crashes or UI hangs when updating `gr.Dropdown` with a value not in the initial `choices`.
- Trigger: Dynamically updating lists (like scene names) before setting the value.
- Mitigation: Always update the `choices` list *before* or *simultaneously* with the `value`.

## Security Considerations

**Path Traversal:**
- Risk: While video paths are validated, export folder creation and filename sanitization must be strictly enforced to prevent writing outside designated project folders.
- Files: `core/export.py`, `core/utils.py`
- Recommendations: Audit `sanitize_filename` and ensure all absolute paths are resolved against a restricted root.

## Performance Bottlenecks

**Hardware Acceleration:**
- Problem: Downscaling and thumbnail extraction currently rely on CPU-based FFmpeg.
- Measurement: Extraction takes ~15-30% of total processing time on high-res videos.
- Cause: Lack of NVENC/VAAPI configuration in the command-line builder.
- Improvement path: Detect GPU type and add `-hwaccel` flags to the FFmpeg command in `core/pipelines.py`.

**Preview Generation:**
- Problem: UI previews for scenes are generated synchronously during tab transitions.
- Measurement: 100-500ms lag when switching to the gallery if previews aren't cached.
- Improvement path: Move preview generation to an async background task or use the `ThumbnailManager` more aggressively.

## Fragile Areas

**Gradio Protocol (IO Mismatch):**
- Why fragile: Gradio event handlers MUST return the exact number of values specified in the `outputs` list.
- Common failures: Adding a new UI component and forgetting to update the return tuple in `app_ui.py` or handlers.
- Safe modification: Always verify the function signature against the `.then()` or `.click()` call.

**SAM3 Submodule Dependency:**
- Why fragile: `SAM3_repo` is an external dependency. Updating it might break the `SAM3Wrapper` if the internal API changes.
- Safe modification: Treat it as read-only. Use heavy mocking in unit tests to avoid dependency on the submodule state.

## Test Coverage Gaps

**Integration Testing for Multi-GPU:**
- What's not tested: Behavior when multiple NVIDIA GPUs are present (device selection logic).
- Risk: Logic might default to `cuda:0` regardless of configuration.
- Priority: Medium.

---

*Concerns audit: 2026-03-21*
*Update as issues are fixed or new ones discovered*
