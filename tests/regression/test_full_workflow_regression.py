import os
import re
import sys
import time
from pathlib import Path

import pytest
from playwright.sync_api import Page, expect

# Add project root to sys.path for app.py imports
sys.path.insert(0, str(Path(__file__).parents[2]))


@pytest.fixture(scope="module")
def real_app_url():
    """Start the real app and return the URL."""
    import signal
    import subprocess

    port = 8766
    url = f"http://127.0.0.1:{port}"
    log_file = Path("tests/results/logs/real_app_hang_reproduction.log")
    log_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nStarting REAL app on port {port}...")
    with open(log_file, "w") as f:
        # We use uv run to ensure the correct environment
        process = subprocess.Popen(
            ["uv", "run", "python", "app.py", "--server-port", str(port), "--debug"],
            stdout=f,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )

    # Wait for the app to be ready
    import requests

    max_wait = 60
    start_time = time.time()
    while time.time() - start_time < max_wait:
        try:
            response = requests.get(url, timeout=1)
            if response.status_code == 200:
                print(f"✓ Real App started successfully (Logs: {log_file.absolute()})")
                break
        except requests.exceptions.ConnectionError:
            time.sleep(1)
    else:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        pytest.fail("Real App failed to start in time.")

    yield url

    print("\nStopping real app...")
    os.killpg(os.getpgid(process.pid), signal.SIGTERM)


def switch_to_tab(page: Page, tab_name: str):
    """Robustly switch tabs in Gradio."""
    tab_btn = page.get_by_role("tab", name=tab_name, exact=False)
    expect(tab_btn).to_be_visible(timeout=10000)

    # Only click if not already selected
    if tab_btn.get_attribute("aria-selected") != "true":
        tab_btn.click(force=True)
        expect(tab_btn).to_have_attribute("aria-selected", "true", timeout=5000)
        page.wait_for_timeout(1500)  # Gradio animation/content load buffer


def test_full_workflow_regression(page: Page, real_app_url: str):
    """
    Test the full workflow in the real app to verify stability and fixes.
    Covers: Extraction -> Pre-Analysis -> Analysis
    """
    # Navigate to app
    print("Navigating to App...")
    page.goto(real_app_url)

    # Step 1: Input
    print("Step 1: Frame Extraction")
    # Check if we are on the Source tab (default)
    expect(page.get_by_text("Step 1: Input & Extraction")).to_be_visible()

    # Select video file
    video_path = "downloads/example clip 720p 2x.mp4"
    if not os.path.exists(video_path):
        pytest.fail(f"Video file not found: {video_path}")

    print(f"  - Using video: {video_path}")
    page.get_by_label("Input Path or URL").fill(video_path)
    page.get_by_label("Input Path or URL").press("Enter")

    # Click Start Extraction
    page.get_by_text("🚀 Start Extraction").click()

    # Wait for extraction completion
    print("  - Waiting for extraction...")
    expect(page.locator("body")).to_contain_text("Extraction Complete", timeout=60000)
    print("  ✓ Extraction Complete")

    # Step 2: Subject (it should automatically switch)
    print("Step 2: Define Subject")

    # Ensure Automatic Detection is selected (default)
    btn_confirm = page.get_by_text("Confirm Subject & Find Scenes (Next Step)")
    expect(btn_confirm).to_be_visible()

    # Click Start Pre-Analysis
    print("  - Clicking Start Pre-Analysis...")
    btn_confirm.click()

    # Wait for Pre-Analysis completion
    print("  - Waiting for Pre-Analysis...")
    expect(page.locator("body")).to_contain_text("Pre-Analysis Complete", timeout=120000)
    print("  ✓ Pre-Analysis Complete")

    # Step 3: Analysis
    print("Step 3: Run Analysis")

    # Manual tab switch just in case automatic doesn't work
    switch_to_tab(page, "Scenes")

    # The button text is dynamic: "Propagate Masks on X Ready Scenes"
    btn_propagate = page.get_by_text(re.compile(r"Propagate Masks on \d+ Ready Scenes"))
    expect(btn_propagate).to_be_visible()
    btn_propagate.click()

    # Wait for propagation to complete (optional but good for stability)
    print("  - Waiting for propagation...")
    expect(page.locator("body")).to_contain_text("Mask Propagation Complete", timeout=120000)
    print("  ✓ Propagation Complete")

    # Switch to Metrics tab
    switch_to_tab(page, "Metrics")

    # Ensure we are on the Metrics/Analysis tab
    expect(page.get_by_text("Step 4: Analysis Metrics")).to_be_visible()

    # Click "Run Analysis"
    print("  - Clicking Run Analysis...")
    page.get_by_text("Run Analysis").click()

    # Wait for Analysis completion
    # This verifies the FaceProminenceOperator fix and the logger serialization fix
    print("  - Waiting for Analysis...")
    expect(page.locator("body")).to_contain_text("Analysis Complete", timeout=180000)
    print("  ✓ Analysis Complete")

    # Step 4: Export
    print("Step 4: Export Results")

    switch_to_tab(page, "Export")

    # WORKAROUND: Sometimes clicking the tab once doesn't trigger the select event properly
    # in a headless test environment. We'll wait and click it again.
    # This ensures load_and_trigger_update runs and enables the export button.
    page.wait_for_timeout(2000)
    print("  - Clicking Export tab again to force select event...")
    page.get_by_role("tab", name="Export").click()

    # Wait for the export button to become interactive/enabled
    # (it becomes enabled once state.all_frames_data is populated by load_and_trigger_update)
    print("  - Waiting for export button to be enabled...")
    export_btn = page.locator("#export_button")
    try:
        expect(export_btn).to_be_enabled(timeout=35000)
    except Exception:
        page.screenshot(path="tests/results/failures/export_btn_fail.png", full_page=True)
        print("  - [DEBUG] Export button (#export_button) not enabled. Screenshot saved.")
        raise

    # Click Export
    print("  - Clicking Export...")
    export_btn.click()

    # Wait for Export completion
    print("  - Waiting for Export...")
    expect(page.locator("body")).to_contain_text("Export Complete", timeout=60000)
    print("  ✓ Export Complete. Full workflow verified from end-to-end!")
