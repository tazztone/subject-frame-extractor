"""
Shared fixtures for Playwright E2E tests.

Provides the app_server fixture that starts/stops the mock Gradio server.
"""

import re
import socket
import subprocess
import sys
import time
from os import environ
from pathlib import Path

import pytest
import requests
from playwright.sync_api import Page, expect

# Constants
PORT = 7860
BASE_URL = f"http://127.0.0.1:{PORT}"


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
    expect(tab_btn).to_be_visible()
    tab_btn.click(force=True)

    # Wait for the tab to be selected - checking class using regex
    expect(tab_btn).to_have_class(re.compile(r"selected|active"))
    time.sleep(1)  # Animation wait


@pytest.fixture(scope="module")
def app_server():
    """
    Starts the mock app server before tests and kills it after.

    The mock app replaces heavy ML operations with fast stubs,
    allowing E2E tests to run quickly without GPU.

    If the real app is already running on port 7860, uses that instead.
    """

    # Check if something is already running on the port
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(("127.0.0.1", PORT))
    sock.close()

    if result == 0:
        # Port is in use - assume real app is running
        print(f"✓ Using existing app on port {PORT}")
        yield None  # No process to manage
        return

    print(f"Starting mock app on port {PORT}...")

    mock_app_path = str(Path(__file__).parent.parent / "mock_app.py")

    # Start the mock server process
    log_file = open("mock_app_e2e.log", "w")
    process = subprocess.Popen(
        [sys.executable, mock_app_path],
        stdout=log_file,
        stderr=subprocess.STDOUT,
        env={**environ, "GRADIO_SERVER_PORT": str(PORT), "PYTHONUNBUFFERED": "1"},
    )

    # Wait for server startup using HTTP check
    if wait_for_server(BASE_URL, timeout=30):
        print("✓ Server started successfully")
    else:
        print("❌ Server failed to start within timeout")
        # Print last few lines of log
        with open("mock_app_e2e.log", "r") as f:
            print(f.read()[-1000:])

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
    expect(page.get_by_text("Frame Extractor & Analyzer")).to_be_visible(timeout=10000)

    # Run extraction
    # The label is hidden in the UI, so we use the placeholder or a more robust selector
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
    find_btn = page.get_by_role("button", name=re.compile("Find & Preview Scenes"))
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
    propagate_btn = page.get_by_role("button", name="🔬 Propagate Masks")
    if propagate_btn.is_visible():
        propagate_btn.click()
        expect(page.locator("#unified_status")).to_contain_text("Mask Propagation Complete", timeout=30000)
        time.sleep(1)

    # Analyze (Metrics tab)
    switch_to_tab(page, "Metrics")

    analyze_btn = page.get_by_role("button", name="Analyze Selected Frames")
    expect(analyze_btn).to_be_visible(timeout=5000)
    analyze_btn.click()

    expect(page.locator("#unified_status")).to_contain_text("Analysis Complete", timeout=30000)
    time.sleep(1)

    return page
