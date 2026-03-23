# Plan: Fix Manager Tests and Boost Coverage

This plan focuses on resolving failures in `tests/unit/test_analysis_manager.py` and implementing Phases 3, 4, and 5 of the hardened test plan to achieve >80% coverage across core managers.

## 1. Debug and Fix `test_run_analysis_only_success`
- **Symptom**: `test_run_analysis_only_success` fails with `assert False is True`.
- **Action**: Modify the test temporarily to print `result["error"]` or `result["log"]` on failure.
- **Hypothesis**: The failure is likely due to `self.db.flush()` being called on a `MagicMock` that isn't fully configured, or a missing file/directory check in the analysis loop.
- **Fix**: Update mocks in `tests/unit/test_analysis_manager.py` to ensure all database and filesystem interactions are covered.

## 2. Expand `core/managers/analysis.py` Coverage (Target 80%)
- **Test Image Folder Analysis**: Add `test_run_image_folder_analysis` to cover the branch where `video_path` is empty.
- **Test Progress Resumption**: Add a test that mocks `progress.json` to verify `_filter_completed_scenes`.
- **Test Batch Processing Logic**: Add a test that specifically triggers `_process_batch` and `_process_single_frame` with varied `metrics_to_compute`.
- **Test NIQE Initialization**: Verify that the bare `except` blocks I added logging to are actually triggered when `pyiqa` is missing or fails.

## 3. Implement Phase 4: `core/managers/extraction.py` (Target 80%)
- **Test Frame Extraction Loop**: 
    - Mock `cv2.VideoCapture` to return a sequence of dummy frames.
    - Verify `nth_frame` skipping logic.
    - Verify `max_resolution` resizing using `cv2.resize` mocks.
- **Test Scene Boundary Handling**:
    - Pass specific `Scene` objects and verify only frames within `[start_frame, end_frame]` are processed.
- **Test Cancellation**:
    - Set `cancel_event` mid-loop and verify the generator yields a "cancelled" state and exits.
- **Test Hardware Acceleration Detection**:
    - Mock `subprocess.run` to simulate different FFmpeg outputs for CUDA/VAAPI.

## 4. Implement Phase 5: `core/managers/sam3.py` (Target 75%)
- **Test Post-processing Utilities**:
    - Test `MaskPostProcessor` (if it exists or is part of SAM3 manager) with synthetic numpy masks.
- **Mock SAM3 Lifecycle**:
    - Mock `sam3` library to avoid heavy dependencies.
    - Test session `open()` and `close()` logic.
    - Test `predict()` wrapper and mask conversion.

## 5. Global Verification
- Run all tests: `uv run pytest tests/unit/test_pipelines.py tests/unit/test_analysis_manager.py tests/unit/test_operators_registry.py`.
- Check global coverage: `uv run pytest --cov=core`.

## Success Criteria
- All tests in `tests/unit/` pass.
- `core/managers/analysis.py` coverage > 80%.
- `core/managers/extraction.py` coverage > 80%.
- `core/managers/sam3.py` coverage > 75%.
- `core/pipelines.py` coverage > 75%.
