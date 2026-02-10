# Stabilization & Comprehensive Testing Plan

## Background

Three user-reported errors and a deeper audit revealed **5 distinct bugs** and a fundamental gap in the test suite: no tests verify that handler return values match what Gradio event wiring expects.

---

## Issue Catalog

| # | Issue | Root Cause | Severity |
|---|-------|-----------|----------|
| 1 | `ValueError: too many/few values` on page navigation | `on_page_change` returns 2 values; wiring expects 4. Lines 52-64 return early; the correct 4-value return on lines 72-83 is **dead code** (unreachable). | 🔴 Critical |
| 2 | `KeyError: 'metadata_path'` after analysis | `execute_analysis` yields `"output_dir"` but not `"metadata_path"`. `_on_analysis_success` accesses `result["metadata_path"]`. | 🔴 Critical |
| 3 | Silent failure on cancel/error | `_run_pipeline` yields `{"unified_log": "Cancelled."}` (string key), but `outputs=all_outputs` means Gradio expects component-object keys. String keys are silently ignored → user sees nothing. | 🟡 High |
| 4 | Silent failure on "No scenes" | `run_propagation_wrapper` and `run_analysis_wrapper` yield `{"unified_log": "No scenes."}` with string keys — same problem as #3. | 🟡 High |
| 5 | SAM3 `BFloat16` / `float32` bias mismatch | Model loads with default dtype on Ampere+ GPUs → `addmm` fails mixing `BFloat16` weights with `float32` bias. | 🔴 Critical (GPU only) |

---

## Proposed Changes

### Fix 1: `on_page_change` dead code → [scene_handler.py](file:///home/tazztone/_coding/subject-frame-extractor/ui/handlers/scene_handler.py)

Delete the early return (lines 52-64) and the stale comments. Keep only the correct 4-value return:

```diff
 def on_page_change(app_state, view, page_num):
     ...
     items, index_map, total_pages = build_scene_gallery_items(...)
     page_choices = [str(i) for i in range(1, total_pages + 1)] ...
-    return (
-        gr.update(value=items),
-        index_map,
-    )
-    # RE-THINK: ...dead comments...
     app_state.scene_gallery_index_map = index_map
     return (
         app_state,
         gr.update(value=items),
         f"/ {total_pages} pages",
         gr.update(choices=page_choices, value=str(current_page)),
     )
```

---

### Fix 2: Missing `metadata_path` → [pipelines.py](file:///home/tazztone/_coding/subject-frame-extractor/core/pipelines.py)

In `execute_analysis`, add `metadata_path` to the success yield:

```diff
 if result and result.get("done"):
     yield {
         "unified_log": "Analysis complete...",
         "output_dir": result["output_dir"],
+        "metadata_path": str(Path(result["output_dir"]) / "metadata.db"),
         "done": True,
     }
```

---

### Fix 3 & 4: String-keyed yields → [app_ui.py](file:///home/tazztone/_coding/subject-frame-extractor/ui/app_ui.py)

Convert all string-keyed yield dicts in `_run_pipeline`, `run_propagation_wrapper`, and `run_analysis_wrapper` to use component-object keys:

```diff
 # _run_pipeline (lines 929, 935, 938)
-yield {"unified_log": "Cancelled."}
+yield {self.components["unified_log"]: "Cancelled."}

 # run_propagation_wrapper (line 1087)
-yield {"unified_log": "No scenes."}
+yield {self.components["unified_log"]: "No scenes."}

 # run_analysis_wrapper (line 1113)
-yield {"unified_log": "No scenes."}
+yield {self.components["unified_log"]: "No scenes."}
```

---

### Fix 5: SAM3 dtype → [sam3_video_predictor.py](file:///home/tazztone/_coding/subject-frame-extractor/SAM3_repo/sam3/model/sam3_video_predictor.py)

Force `float32` when moving model to GPU:

```diff
-model.to(device)
+model.to(device=device, dtype=torch.float32)
```

---

## Comprehensive Testing Strategy

> [!IMPORTANT]
> The current test suite only has **browser-level Playwright E2E tests** and **operator unit tests**. There's a critical missing layer: **fast Python-level tests** that verify handler contracts without launching a browser.

### Testing Pyramid for This Project

```
            ┌─────────────────────┐
            │   E2E (Playwright)  │  ← Existing: test_app_flow, test_bug_regression
            │   Slow, high-level  │
            └────────┬────────────┘
                     │
         ┌───────────┴───────────┐
         │  Handler Contract     │  ← NEW: Verifies return counts & key types
         │  Tests (fast, no GPU) │     match Gradio wiring WITHOUT a browser
         └───────────┬───────────┘
                     │
    ┌────────────────┴────────────────┐
    │  Pipeline Result Schema Tests   │  ← NEW: Verifies each pipeline yields
    │  (unit-level, mocked deps)      │     dicts with expected keys
    └────────────────┬────────────────┘
                     │
    ┌────────────────┴────────────────┐
    │  Operator Unit Tests            │  ← Existing: tests/operators/
    │  (fast, isolated)               │
    └─────────────────────────────────┘
```

### New Test Files

#### [NEW] [test_handler_contracts.py](file:///home/tazztone/_coding/subject-frame-extractor/tests/ui/test_handler_contracts.py)

**Purpose**: Verify every handler function returns the exact number of values its event wiring expects. Catches the exact class of bug that caused Issues #1, #3, #4.

**Approach**:
1. Import `AppUI` and instantiate with mocks (no GPU, no browser).
2. Call `setup_handlers()` to register all event listeners.
3. Introspect Gradio's internal `demo.config["dependencies"]` to get the `outputs` list for each listener.
4. For each listener, invoke the handler with `ApplicationState()` + mock inputs.
5. Assert: `len(returned_tuple) == len(expected_outputs)`.
6. Assert: if handler is a generator and wired to `outputs=all_outputs`, each yielded dict uses component-object keys (not strings).

**Key tests**:
- `test_pagination_handlers_return_4_values` — directly prevents Issue #1 regression
- `test_pipeline_wrappers_yield_component_keyed_dicts` — prevents Issues #3 & #4
- `test_success_callbacks_return_component_keyed_dicts` — validates `_on_*_success`

#### [NEW] [test_pipeline_result_schemas.py](file:///home/tazztone/_coding/subject-frame-extractor/tests/unit/test_pipeline_result_schemas.py)

**Purpose**: Verify pipeline generators (`execute_extraction`, `execute_analysis`, etc.) yield dicts with required keys. Catches Issue #2.

**Approach**:
1. Mock all heavy dependencies (SAM3, ffmpeg, DB).
2. Call each `execute_*` function with mock inputs.
3. Collect all yielded dicts.
4. Assert the final "success" dict contains all keys the corresponding `_on_*_success` callback accesses.

**Key tests**:
- `test_execute_analysis_yields_metadata_path` — directly prevents Issue #2 regression
- `test_execute_extraction_yields_output_dir` — validates extraction result schema
- `test_all_pipelines_yield_unified_log` — every yield must have a log message

---

## Verification Plan

### Automated Tests
```bash
# New contract tests (fast, no browser needed)
pytest tests/ui/test_handler_contracts.py -v

# New pipeline schema tests (fast, mocked)
pytest tests/unit/test_pipeline_result_schemas.py -v

# Existing E2E (requires mock app server)
pytest tests/ui/test_app_flow.py -v
pytest tests/ui/test_bug_regression.py -v
```

### Manual Verification
1. **Start app** → navigate scenes → confirm no `ValueError` in console
2. **Run full analysis pipeline** → confirm `metadata.db` path appears in state
3. **Cancel mid-pipeline** → confirm "Cancelled" appears in the log panel (not silently swallowed)
