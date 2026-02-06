---
phase: 4
level: 2
researched_at: 2026-02-04
---

# Phase 4 Debug Research: UI Test Timeout

## Problem Statement
The `test_full_user_flow` UI test consistently times out waiting for "Extraction Complete" in `#unified_status`, despite the mock backend printing "[Mock] Running Extraction...".

## Questions Investigated
1. How does extraction status reach the UI?
2. Why doesn't the mock's tracker update trigger `unified_status`?
3. What is the correct mock strategy for Gradio 5 generators?

## Findings

### 1. The UI Update Chain (Root Cause Identified)

The real extraction flow is:

```
Button Click
  → run_extraction_wrapper()
  → _run_pipeline(execute_extraction, ..., success_callback=_on_extraction_success)
  → execute_extraction() yields {"done": True, ...}
  → _run_pipeline catches "done" and calls success_callback()
  → _on_extraction_success() returns {"unified_status": "✅ Frame Extraction Complete..."}
```

**The current mock patches the wrong layer:**

```python
# mock_app.py (LINE 167)
core.pipelines.ExtractionPipeline._run_impl = mock_extraction_run
```

But `execute_extraction()` calls `pipeline.run(tracker=tracker)` which:
1. Still uses the patched `_run_impl` ✓
2. Returns `{"done": True, "output_dir": ..., "video_path": ...}`
3. `execute_extraction` then yields: `{"unified_log": "...", "extracted_*_state": ..., "done": True}`

The **real problem** is that `mock_extraction_run` returns instantly but doesn't put any result in the format `execute_extraction` expects. The mock's result lacks the required keys (`output_dir`, `video_path`).

### 2. Mock Result Format Mismatch

Looking at `mock_extraction_run` (line 56-80):

```python
def mock_extraction_run(self, tracker=None):
    # ... simulation ...
    return {"done": True, "output_dir": output_dir, "video_path": "mock_video.mp4"}
```

But `execute_extraction` (line 866-872) expects:
```python
if result and result.get("done"):
    yield {
        "extracted_video_path_state": result.get("video_path", ""),
        "extracted_frames_dir_state": result["output_dir"],
        "done": True,
    }
```

**This looks correct!** So the mock format is fine.

### 3. The Real Issue: _run_pipeline → success_callback

Looking at `run_extraction_wrapper` (line 959-969):
```python
def run_extraction_wrapper(self, current_state: ApplicationState, *args, progress=None):
    # ...
    yield from self._run_pipeline(
        execute_extraction, event, progress or gr.Progress(), 
        lambda res: self._on_extraction_success(res, current_state)
    )
```

And `_run_pipeline` (line 917-958):
```python
def _run_pipeline(self, pipeline_func, event, progress, success_callback=None, *args):
    for result in pipeline_func(...):
        if result.get("done"):
            if success_callback:
                yield success_callback(result)  # ← This yields the unified_status update!
            return
```

**The key insight:** `_run_pipeline` iterates through `pipeline_func` generator and when it sees `done=True`, it calls `success_callback` which returns the dict containing `unified_status`.

But `run_extraction_wrapper` uses `yield from self._run_pipeline(...)`, meaning it yields whatever `_run_pipeline` yields.

So the flow should be:
1. `execute_extraction` yields `{..., "done": True}`
2. `_run_pipeline` catches this, calls `_on_extraction_success(result)` 
3. `_on_extraction_success` returns `{..., "unified_status": "✅ Frame Extraction Complete..."}`
4. `_run_pipeline` yields this dict
5. `run_extraction_wrapper` yields this via `yield from`

### 4. The Actual Bug: `run_extraction_wrapper` is NOT wrapped in `_run_task_with_progress`

Looking at `_setup_pipeline_handlers` - we need to find how the button is wired.

**Key discovery:** The extraction button click handler likely uses either:
- Direct call to `run_extraction_wrapper` (then UI updates work)
- Wrapped in `_run_task_with_progress` (then UI updates via ThreadPoolExecutor + queue)

The test expects `#unified_status` to contain "Extraction Complete", but `_on_extraction_success` returns:
```python
msg = f"""<div class="success-card">
    <h3>✅ Frame Extraction Complete</h3>
    ...
</div>"""
```

So it says "**Frame** Extraction Complete", not just "Extraction Complete".

## Decisions Made

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Mock strategy | Mock `execute_extraction` directly | Patching `_run_impl` works but the test selector is wrong |
| Test selector | Update to match "Frame Extraction Complete" | The UI says "Frame Extraction Complete" not "Extraction Complete" |

## Debug Plan

### Step 1: Verify Test Selector Match
The test expects:
```python
expect(page.locator("#unified_status")).to_contain_text("Extraction Complete", timeout=30000)
```

But the UI renders:
```html
<h3>✅ Frame Extraction Complete</h3>
```

This **should match** since "Extraction Complete" is a substring. So selector is correct.

### Step 2: Verify Mock Yields Reach the UI
Run the test with verbose output to see if the mock generator's yield reaches `_run_pipeline`.

**Command:**
```bash
uv run pytest tests/ui/test_app_flow.py::TestMainWorkflow::test_full_user_flow -v -s --headed 2>&1 | head -100
```

### Step 3: Check if `_run_task_with_progress` is Used
If extraction uses `_run_task_with_progress`, then:
- The generator runs in a thread
- Updates go through `progress_queue`
- Main thread polls queue and yields updates

This could cause timing issues if the mock returns too fast.

### Step 4: Inspect Button Wiring
Find how the extraction button is connected:
```bash
grep -n "Start Extraction" ui/*.py ui/tabs/*.py
```

### Step 5: Add Mock Delay if Needed
If the issue is timing, the mock should simulate realistic async behavior:
```python
def mock_extraction_run(self, tracker=None):
    time.sleep(0.5)  # Give UI time to start listening
    # ... rest of mock
```

## Complete Flow Analysis

The extraction flow is:
```
Button Click (start_extraction_button)
  → run_extraction_wrapper(state, *args)
  → _run_pipeline(execute_extraction, event, progress, _on_extraction_success)
    → for result in execute_extraction(...):
        → ExtractionPipeline.run(tracker) → _run_impl(tracker) [MOCKED]
        → yields {"extracted_video_path_state": ..., "done": True}
    → success_callback(result) → _on_extraction_success(result)
        → returns {components["unified_status"]: "<div>✅ Frame Extraction Complete...</div>", ...}
  → yield from passes this to Gradio
```

**Key insight:** The yield dict from `_on_extraction_success` uses component OBJECTS as keys:
```python
return {
    self.components["unified_status"]: msg,  # ← component object, not string
    ...
}
```

This is correct for Gradio's output mapping to work.

## Risks
- **Gradio 5 Threading Model**: May differ from Gradio 4 in how generators are handled
- **Mock Speed**: Too-fast mocks may complete before UI renders the component

## Concrete Debug Plan

### Step 1: Run the Failing Test (Observe)
```bash
uv run pytest tests/ui/test_app_flow.py::TestMainWorkflow::test_full_user_flow -v -s --headed 2>&1 | tail -50
```
Look for:
- "[Mock] Running Extraction..." in test output
- What `#unified_status` actually contains when timeout occurs

### Step 2: Browser Visual Debug
If Step 1 fails, use browser subagent to:
1. Start the mock app manually
2. Click extraction
3. Observe what `#unified_status` shows

### Step 3: Check Mock Output Format
Verify `mock_extraction_run` returns correct keys:
```python
# Expected: {"done": True, "output_dir": ..., "video_path": ...}
# Then execute_extraction yields: {"extracted_video_path_state": ..., "done": True}
```

### Step 4: Check extracted_session Fixture
The `conftest.py` has a fixture `extracted_session` that expects `"Extraction complete"` (lowercase) at line 127:
```python
expect(page.get_by_text("Extraction complete")).to_be_visible(timeout=20000)
```
But `test_full_user_flow` expects `"Extraction Complete"` (capitalized) in `#unified_status`.

**Potential Mismatch:** Test uses `to_contain_text("Extraction Complete")` but the actual UI HTML is:
```html
<h3>✅ Frame Extraction Complete</h3>
```
This should match as substring. Need to verify with browser.

### Step 5: Fix if Needed
Based on findings:
- If mock is too fast: Add `time.sleep(0.2)` after mock
- If key mismatch: Fix the mock to yield correct format
- If selector wrong: Update test selector

## Ready for Debugging
- [x] Root cause analyzed
- [x] Mock architecture understood  
- [x] Debug steps defined
- [ ] Test executed with fresh context
- [ ] Fix implemented and verified
