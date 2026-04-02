import os
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
            ["uv", "run", "python", "app.py", "--server-port", str(port)],
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


def test_reproduce_hang_workflow(page: Page, real_app_url: str):
    """
    Test the pre-analysis workflow in the real app to reproduce and verify the hang fix.
    Specifically tests transition after Scene 0's analysis.
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

    # Wait for extraction completion (visible text in the log/status)
    print("  - Waiting for extraction...")
    expect(page.locator("body")).to_contain_text("Extraction Complete", timeout=60000)
    print("  ✓ Extraction Complete")

    # Step 2: Subject (it should automatically switch, but let's click just in case)
    print("Step 2: Define Subject")

    # Ensure Automatic Detection is selected (default)
    # Check "Confirm Subject & Find Scenes (Next Step)" button
    btn_confirm = page.get_by_text("Confirm Subject & Find Scenes (Next Step)")
    expect(btn_confirm).to_be_visible()

    # Click Start Pre-Analysis
    print("  - Clicking Start Pre-Analysis...")
    btn_confirm.click()

    # Wait for Pre-Analysis completion
    # If the bug is present, it will hang here or before this.
    print("  - Waiting for Pre-Analysis... (This is where it should hang)")
    # We expect "Pre-Analysis Complete" to appear in the body/unified_log
    expect(page.locator("body")).to_contain_text("Pre-Analysis Complete", timeout=120000)
    print("  ✓ Pre-Analysis Complete. Hang resolved!")
