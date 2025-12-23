"""
Shared fixtures for Playwright E2E tests.

Provides the app_server fixture that starts/stops the mock Gradio server.
"""
import pytest
import subprocess
import sys
import time
from pathlib import Path
from os import environ

# Constants
PORT = 7860
BASE_URL = f"http://127.0.0.1:{PORT}"


@pytest.fixture(scope="module")
def app_server():
    """
    Starts the mock app server before tests and kills it after.
    
    The mock app replaces heavy ML operations with fast stubs,
    allowing E2E tests to run quickly without GPU.
    """
    print(f"Starting mock app on port {PORT}...")

    mock_app_path = str(Path(__file__).parent.parent / 'mock_app.py')

    # Start the mock server process
    log_file = open("mock_app_e2e.log", "w")
    process = subprocess.Popen(
        [sys.executable, mock_app_path],
        stdout=log_file,
        stderr=subprocess.STDOUT,
        env={**environ, "GRADIO_SERVER_PORT": str(PORT), "PYTHONUNBUFFERED": "1"}
    )

    # Wait for server startup
    time.sleep(8)  # Gradio can be slow to initialize

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
    time.sleep(2)
    
    # Run extraction
    page.get_by_label("Video URL or Local Path").fill("dummy_video.mp4")
    page.get_by_role("button", name="🚀 Start Single Extraction").click()
    
    # Wait for completion
    from playwright.sync_api import expect
    expect(page.get_by_text("Extraction complete")).to_be_visible(timeout=20000)
    time.sleep(2)
    
    return page


@pytest.fixture
def analyzed_session(extracted_session):
    """
    Fixture that provides a page with pre-analysis completed.
    
    Builds on extracted_session to provide further workflow progress.
    """
    page = extracted_session
    
    # Navigate to Subject tab and run pre-analysis
    page.get_by_role("tab", name="Subject").click(force=True)
    time.sleep(1)
    page.get_by_role("button", name="🌱 Find & Preview Best Frames").click()
    
    from playwright.sync_api import expect
    expect(page.locator("#unified_log")).to_contain_text("Pre-analysis complete", timeout=10000)
    time.sleep(2)
    
    return page
