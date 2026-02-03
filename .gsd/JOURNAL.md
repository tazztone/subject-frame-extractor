# JOURNAL

## 2026-02-02: Project Kickoff & Initialization
- Completed brownfield mapping.
- Identified 9 major test failures/errors related to Database refactor.
- Established a stabilization-focused roadmap.
- Initialized GSD environment.

## 2026-02-02 (Session 2): Stabilization & E2E Verification
- **Stabilization**: Resolved critical bugs in `SAM3Wrapper` weight loading by switching back to `.pt` files and submodule-native loading. Fixed `ValueError` and `AttributeError` in frame quality analysis.
- **Resource Management**: Verified that the memory watchdog correctly triggers and cleans up models during high-load periods.
- **E2E Success**: Verified the entire application workflow via `verification/e2e_run.py`. Confirmed that masks are generated correctly and metadata is accurately stored in SQLite.
- **Dependency Hygiene**: Fixed `onnx` and `sam3` version conflicts via `pyproject.toml` updates.
- **Status**: The project is now stable, resource-safe, and passes comprehensive E2E verification.

## 2026-02-03: Linux Support Enhancement
- **UX/Ease of Use**: Created `scripts/linux_run_app.sh` to mirror the functionality of the Windows `.bat` script.
- **Environment Optimization**: Configured the script to use `uv run`, ensuring that the correct virtual environment and project settings are always applied without manual activation.
- **Status**: Improved accessibility for Linux-based developers and users.
