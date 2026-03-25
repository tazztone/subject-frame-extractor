"""
Shared fixtures for Playwright E2E tests.

Provides the app_server fixture that starts/stops the mock Gradio server.
"""

import subprocess
import sys
import time
from os import environ
from pathlib import Path

import pytest
import requests
from playwright.sync_api import Page, expect

# Constants
# Use an isolated port for UI tests to avoid collisions with the real app (7860)
PORT = 8765
BASE_URL = f"http://127.0.0.1:{PORT}"


@pytest.fixture(autouse=True)
def setup_playwright_timeout(page: Page):
    """Set a baseline timeout for all Playwright actions."""
    page.set_default_timeout(10000)
    yield


def wait_for_server(url, timeout=60):
    """Wait for the server to be responsive."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(url, timeout=1)
            if response.status_code == 200:
                return True
        except requests.RequestException:
            pass
        time.sleep(1)
    return False


def switch_to_tab(page: Page, tab_name: str):
    """Robustly switch tabs in Gradio."""
    # Click the tab button
    tab_btn = page.get_by_role("tab", name=tab_name)
    expect(tab_btn).to_be_visible(timeout=5000)
    tab_btn.click(force=True)

    # Wait for the tab to be selected
    expect(tab_btn).to_have_attribute("aria-selected", "true", timeout=5000)
    time.sleep(0.5)  # Shorter animation wait


def cleanup_port(port: int):
    """Forcefully kill any process using the specified port."""
    try:
        if sys.platform != "win32":
            subprocess.run(["fuser", "-k", f"{port}/tcp"], capture_output=True, check=False)
    except Exception:
        pass


@pytest.fixture(scope="module")
def app_server():
    """
    Starts the mock app server before tests and kills it after.

    The mock app replaces heavy ML operations with fast stubs,
    allowing E2E tests to run quickly without GPU.
    """

    # Aggressively clean up the test port before starting
    cleanup_port(PORT)
    time.sleep(1)  # Give the OS a moment to free the socket

    print(f"Starting mock app on port {PORT}...")

    mock_app_path = str(Path(__file__).parent.parent / "mock_app.py")

    # Start the mock server process
    log_dir = Path(__file__).parent.parent.parent / "tests" / "results" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file_path = log_dir / "mock_app_e2e.log"

    log_file = open(log_file_path, "w")
    process = subprocess.Popen(
        [sys.executable, mock_app_path],
        stdout=log_file,
        stderr=subprocess.STDOUT,
        env={**environ, "APP_SERVER_PORT": str(PORT), "PYTHONUNBUFFERED": "1"},
    )

    # Wait for server startup using HTTP check with increased timeout
    if wait_for_server(BASE_URL, timeout=60):
        print(f"✓ Server started successfully (Logs: {log_file_path})")
    else:
        print("❌ Server failed to start within timeout")
        # Ensure cleanup before failing
        process.terminate()
        with open(log_file_path, "r") as f:
            print(f"Mock App Logs:\n{f.read()[-2000:]}")
        pytest.fail("Mock server failed to start")

    yield process

    # Cleanup
    print("Stopping mock app...")
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
    log_file.close()


@pytest.fixture
def extracted_session(page, app_server):
    """
    Fixture that provides a page with extraction already completed.

    Useful for tests that need to start from a specific workflow stage.
    """
    page.goto(BASE_URL)

    # Ensure app is loaded
    expect(page.get_by_text("Frame Extractor & Analyzer v2.0")).to_be_visible(timeout=10000)

    # Run extraction
    page.get_by_placeholder("Paste YouTube URL or local path").fill("dummy_video.mp4")

    extract_btn = page.get_by_role("button", name="🚀 Start Extraction")
    expect(extract_btn).to_be_visible(timeout=5000)
    extract_btn.click()

    # Wait for completion
    expect(page.get_by_text("Extraction complete")).to_be_visible(timeout=20000)
    time.sleep(1)

    return page


@pytest.fixture
def analyzed_session(extracted_session):
    """
    Fixture that provides a page with pre-analysis completed.
    """
    page = extracted_session

    # Navigate to Subject tab
    switch_to_tab(page, "Subject")

    # Click find frames
    find_btn = page.get_by_role("button", name="✅ Confirm Subject & Find Scenes (Next Step)")
    # Sometimes it takes time to switch tabs or render
    time.sleep(1)
    if not find_btn.is_visible():
        page.reload()
        switch_to_tab(page, "Subject")
        time.sleep(1)

    expect(find_btn).to_be_visible(timeout=10000)
    find_btn.click()

    # Wait for completion (check status)
    expect(page.locator("#unified_status")).to_contain_text("Pre-Analysis Complete", timeout=30000)
    time.sleep(1)

    return page


@pytest.fixture
def full_analysis_session(analyzed_session):
    """
    Fixture that provides a page with full analysis completed (ready for export).
    """
    page = analyzed_session

    # Propagate
    propagate_btn = page.get_by_role("button", name="⚡ Propagate Masks to All Frames")
    if propagate_btn.is_visible():
        propagate_btn.click()
        expect(page.locator("#unified_status")).to_contain_text("Mask Propagation Complete", timeout=30000)
        time.sleep(1)

    # Analyze (Metrics tab)
    switch_to_tab(page, "Metrics")

    analyze_btn = page.get_by_role("button", name="⚡ Run Analysis")
    expect(analyze_btn).to_be_visible(timeout=5000)
    analyze_btn.click()

    expect(page.locator("#unified_status")).to_contain_text("Analysis Complete", timeout=30000)
    time.sleep(1)

    return page
