# Developer Utility Scripts

This directory contains scripts and utility programs for setting up the environment, running tests, launching the application, and auditing quality metrics.

## Primary Entry Points & Test Runners

| Script | Platform | Purpose | Usage |
| :--- | :--- | :--- | :--- |
| [linux_run_app.sh](file:///home/tazztone/_coding/subject-frame-extractor/scripts/linux_run_app.sh) | Linux | Launches the Gradio web UI application. | `bash scripts/linux_run_app.sh` |
| [linux_test_all.sh](file:///home/tazztone/_coding/subject-frame-extractor/scripts/linux_test_all.sh) | Linux | Runs the complete test suite (Unit, Integration, UI, Regression) and logs timings to `tests/results/logs/test_performance.log`. | `bash scripts/linux_test_all.sh` |
| [linux_test_unit.sh](file:///home/tazztone/_coding/subject-frame-extractor/scripts/linux_test_unit.sh) | Linux | Runs fast unit tests (completely mocked, no GPU dependency). | `bash scripts/linux_test_unit.sh` |
| [linux_test_ui.sh](file:///home/tazztone/_coding/subject-frame-extractor/scripts/linux_test_ui.sh) | Linux | Runs parallel UI/E2E browser automation tests using Playwright + `xdist`. | `bash scripts/linux_test_ui.sh` |
| [linux_test_integration.sh](file:///home/tazztone/_coding/subject-frame-extractor/scripts/linux_test_integration.sh) | Linux | Runs serial integration tests (requires real PyTorch models and GPU). | `bash scripts/linux_test_integration.sh` |
| [linux_test_sam3.sh](file:///home/tazztone/_coding/subject-frame-extractor/scripts/linux_test_sam3.sh) | Linux | Runs heavyweight integration/E2E tests specifically targeting SAM3 tracking. | `bash scripts/linux_test_sam3.sh` |
| [linux_test_cov.sh](file:///home/tazztone/_coding/subject-frame-extractor/scripts/linux_test_cov.sh) | Linux | Runs unit tests and generates a local HTML coverage report (`htmlcov/index.html`). | `bash scripts/linux_test_cov.sh` |

---

## Setup & Updates

| Script | Platform | Purpose | Usage |
| :--- | :--- | :--- | :--- |
| [linux_setup_playwright.sh](file:///home/tazztone/_coding/subject-frame-extractor/scripts/linux_setup_playwright.sh) | Linux | Installs and caches the Chromium browser binaries required for Playwright tests. | `bash scripts/linux_setup_playwright.sh` |
| [windows_STANDALONE_install.bat](file:///home/tazztone/_coding/subject-frame-extractor/scripts/windows_STANDALONE_install.bat) | Windows | Installs dependencies, sets up virtual environments, and installs required packages on Windows. | `windows_STANDALONE_install.bat` |
| [windows_run_app.bat](file:///home/tazztone/_coding/subject-frame-extractor/scripts/windows_run_app.bat) | Windows | Launches the Gradio web UI on Windows. | `windows_run_app.bat` |
| [windows_update.bat](file:///home/tazztone/_coding/subject-frame-extractor/scripts/windows_update.bat) | Windows | Updates the application dependencies and packages on Windows. | `windows_update.bat` |

---

## Auditing, Reference, & Analysis

| Script | Type | Purpose | Usage |
| :--- | :--- | :--- | :--- |
| [run_ux_audit.py](file:///home/tazztone/_coding/subject-frame-extractor/scripts/run_ux_audit.py) | Python | Performs UX accessibility auditing (Axe-core), layout diagnostics, and responsiveness checks. | `uv run python scripts/run_ux_audit.py` |
| [verify_quality.py](file:///home/tazztone/_coding/subject-frame-extractor/scripts/verify_quality.py) | Python | Evaluates pipeline outputs (mask yield rates, face similarities, average NIQE metrics) and logs them to `tests/integration/HISTORY.csv`. | `uv run python scripts/verify_quality.py <output_directory>` |
| [update_agents_md.py](file:///home/tazztone/_coding/subject-frame-extractor/scripts/update_agents_md.py) | Python | Scans codebase ASTs and regenerates the auto-generated code reference skeletons (`docs/AGENTS_CODE_REFERENCE.md` and `docs/TESTS_CODE_REFERENCE.md`). | `uv run python scripts/update_agents_md.py` |
| [generate_todo_report.py](file:///home/tazztone/_coding/subject-frame-extractor/scripts/generate_todo_report.py) | Python | Scans the codebase for TODO/FIXME annotations and outputs a structured report. | `uv run python scripts/generate_todo_report.py` |
| [take_screenshot.py](file:///home/tazztone/_coding/subject-frame-extractor/scripts/take_screenshot.py) | Python | Uses Playwright browser automation to take a visual snapshot of the running web page. | `uv run python scripts/take_screenshot.py` |
| [diff_upstream_sam3.sh](file:///home/tazztone/_coding/subject-frame-extractor/scripts/diff_upstream_sam3.sh) | Bash | Compares modifications made to the vendored SAM3 library against upstream components. | `bash scripts/diff_upstream_sam3.sh` |

> [!NOTE]
> All python-based scripts should be executed using the `uv` toolchain (e.g. `uv run python scripts/<script_name>`) to ensure they use the correct virtual environment.
