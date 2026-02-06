---
phase: 4
plan: 1
wave: 1
---

# Plan 4.1: Fix UI Test Timeout in Extraction Flow

## Objective
Resolve the `test_full_user_flow` timeout that blocks Phase 4 Gap Closure. The test times out waiting for "Extraction Complete" in `#unified_status` despite the mock running correctly.

## Context
- [.gsd/phases/4/RESEARCH.md](file:///home/tazztone/_coding/subject-frame-extractor/.gsd/phases/4/RESEARCH.md)
- [tests/ui/test_app_flow.py](file:///home/tazztone/_coding/subject-frame-extractor/tests/ui/test_app_flow.py)
- [tests/mock_app.py](file:///home/tazztone/_coding/subject-frame-extractor/tests/mock_app.py)
- [ui/app_ui.py](file:///home/tazztone/_coding/subject-frame-extractor/ui/app_ui.py)

## Tasks

<task type="auto">
  <name>Observe failure with fresh context</name>
  <files>tests/ui/test_app_flow.py</files>
  <action>
    Run the failing test with --headed flag to observe what happens:
    ```bash
    uv run pytest tests/ui/test_app_flow.py::TestMainWorkflow::test_full_user_flow -v -s --headed 2>&1 | tail -100
    ```
    
    Observe:
    1. Does "[Mock] Running Extraction..." appear in output?
    2. What does `#unified_status` show when timeout occurs?
    3. Does the success card appear?
    
    If timeout occurs, take screenshot of browser state.
  </action>
  <verify>Test output captured showing current failure state</verify>
  <done>Clear understanding of what the UI shows at timeout</done>
</task>

<task type="auto">
  <name>Fix mock or test selector based on observations</name>
  <files>tests/mock_app.py, tests/ui/test_app_flow.py</files>
  <action>
    Based on Task 1 observations, apply one of these fixes:
    
    **If mock returns too fast** (UI hasn't rendered):
    - Add `time.sleep(0.5)` before returning from `mock_extraction_run`
    
    **If mock output format is wrong**:
    - Ensure mock returns `{"done": True, "output_dir": "...", "video_path": "..."}`
    - These keys are expected by `execute_extraction` to yield correct state
    
    **If selector is wrong**:
    - Update test to match actual HTML: "Frame Extraction Complete" vs "Extraction Complete"
    - Or wait for `.success-card` visibility instead of text content
    
    **If components not in outputs**:
    - Verify `unified_status` is included in the handler's output list
  </action>
  <verify>
    ```bash
    uv run pytest tests/ui/test_app_flow.py::TestMainWorkflow::test_full_user_flow -v -s 2>&1
    ```
  </verify>
  <done>Test passes extraction stage and proceeds to Pre-Analysis</done>
</task>

## Success Criteria
- [ ] `test_full_user_flow` extraction stage completes without timeout
- [ ] Mock extraction properly updates `#unified_status`
- [ ] Test proceeds to subsequent workflow stages
