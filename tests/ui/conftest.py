import subprocess
import sys
import time
from os import environ
from pathlib import Path

import pytest
import requests
from playwright.sync_api import Page, expect

from .ui_locators import Labels, Selectors


# Constants
# Use an isolated port for UI tests to avoid collisions with the real app (7860)
# Support parallel execution via pytest-xdist worker IDs
def get_test_port():
    base_port = 8765
    worker_id = environ.get("PYTEST_XDIST_WORKER")
    if worker_id:
        # worker_id is 'gw0', 'gw1', etc.
        try:
            worker_num = int("".join(filter(str.isdigit, worker_id)))
            return base_port + worker_num
        except ValueError:
            pass
    return base_port


PORT = get_test_port()
BASE_URL = f"http://127.0.0.1:{PORT}"
FAILURES_DIR = Path(__file__).parent.parent.parent / "tests" / "results" / "failures"


@pytest.fixture(autouse=True)
def setup_playwright_timeout(page: Page):
    """Set a baseline timeout for all Playwright actions."""
    page.set_default_timeout(15000)  # Increased for slow CI/container runs
    yield


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Capture screenshot on test failure."""
    outcome = yield
    report = outcome.get_result()

    if report.when == "call" and report.failed:
        page = item.funcargs.get("page")
        if page:
            FAILURES_DIR.mkdir(parents=True, exist_ok=True)
            screenshot_path = FAILURES_DIR / f"{item.name}.png"
            try:
                page.screenshot(path=str(screenshot_path), full_page=True)
                print(f"\n[DEBUG] Failure screenshot saved to: {screenshot_path}")
            except Exception as e:
                print(f"\n[DEBUG] Failed to capture screenshot: {e}")


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


def wait_for_app_ready(page: Page):
    """
    Robustly wait for the Gradio app to be interactive.
    Avoids 'networkidle' as it's unreliable with Gradio WebSockets.
    """
    # Wait for the main heading to be visible as a proxy for app load
    expect(page.get_by_text("Frame Extractor & Analyzer")).to_be_visible(timeout=30000)
    # Wait for the status area to be present
    expect(page.locator(Selectors.UNIFIED_STATUS)).to_be_attached(timeout=5000)
    # Small buffer for Gradio's JS to settle
    page.wait_for_timeout(1500)


def open_accordion(page: Page, text: str):
    """Robustly opens an accordion by partial text match, no-ops if already open."""
    # Try multiple ways to find the accordion toggle
    accordion = page.get_by_text(text, exact=False).first

    # Check if a sibling or parent contains the expanded state
    # Gradio accordions typically have an .open or aria-expanded state on a child or parent
    is_open = False
    try:
        # Check if the text matches something inside an open accordion
        # In Gradio 5, we can often just click safely if we don't know the state,
        # but double-clicking closes it. Let's look for the svg chevron direction if possible,
        # or just check visibility of a known child if provided.
        # For now, we'll use a pragmatic approach: if the log textarea is already visible,
        # then the "System Logs" accordion is already open.
        if text == Labels.SYSTEM_LOGS and page.locator(Selectors.LOG_TEXTAREA).is_visible():
            is_open = True
    except Exception:
        pass

    if not is_open:
        accordion.click()
        page.wait_for_timeout(500)


def switch_to_tab(page: Page, tab_name: str):
    """Robustly switch tabs in Gradio."""
    # Click the tab button
    tab_btn = page.get_by_role("tab", name=tab_name, exact=False)
    expect(tab_btn).to_be_visible(timeout=10000)

    # Only click if not already selected
    if tab_btn.get_attribute("aria-selected") != "true":
        tab_btn.click(force=True)
        expect(tab_btn).to_have_attribute("aria-selected", "true", timeout=5000)
        page.wait_for_timeout(1000)  # Gradio animation/content load buffer


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

    log_dir = Path(__file__).parent.parent.parent / "tests" / "results" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    worker_id = environ.get("PYTEST_XDIST_WORKER", "master")
    log_file_path = log_dir / f"mock_app_e2e_{worker_id}.log"

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
    wait_for_app_ready(page)

    # Run extraction
    source_input = page.get_by_placeholder(Labels.SOURCE_PLACEHOLDER)
    if not source_input.is_visible():
        source_input = page.locator(Selectors.SOURCE_INPUT)

    source_input.fill("dummy_video.mp4")

    extract_btn = page.locator(Selectors.START_EXTRACTION)
    expect(extract_btn).to_be_visible(timeout=5000)
    extract_btn.click()

    # Wait for completion
    expect(page.locator(Selectors.UNIFIED_STATUS)).to_contain_text("Extraction Complete", timeout=30000)
    page.wait_for_timeout(1000)

    return page


@pytest.fixture
def analyzed_session(extracted_session):
    """
    Fixture that provides a page with pre-analysis completed.
    """
    page = extracted_session

    # Navigate to Subject tab
    switch_to_tab(page, Labels.TAB_SUBJECT)

    # Click find frames
    find_btn = page.locator(Selectors.START_PRE_ANALYSIS)

    # Sometimes it takes time to switch tabs or render
    time.sleep(1)
    if not find_btn.is_visible():
        switch_to_tab(page, Labels.TAB_SUBJECT)
        time.sleep(1)

    expect(find_btn).to_be_visible(timeout=10000)
    find_btn.click()

    # Wait for completion (check status)
    expect(page.locator(Selectors.UNIFIED_STATUS)).to_contain_text("Pre-Analysis Complete", timeout=30000)
    page.wait_for_timeout(1000)

    return page


@pytest.fixture
def full_analysis_session(analyzed_session):
    """
    Fixture that provides a page with full analysis completed (ready for export).
    """
    page = analyzed_session

    # Propagation happens in Scenes tab
    switch_to_tab(page, Labels.TAB_SCENES)

    # Click Propagate
    # For now, use text matching since I don't have a specific ID for Propagate button in Selectors yet
    propagate_btn = page.get_by_role("button", name="Propagate Masks", exact=False)
    expect(propagate_btn).to_be_visible(timeout=10000)
    propagate_btn.click()

    expect(page.locator(Selectors.UNIFIED_STATUS)).to_contain_text("Propagation Complete", timeout=30000)
    page.wait_for_timeout(1000)

    # Analyze (Metrics tab)
    switch_to_tab(page, Labels.TAB_METRICS)

    analyze_btn = page.locator(Selectors.START_ANALYSIS)
    expect(analyze_btn).to_be_visible(timeout=5000)
    analyze_btn.click()

    expect(page.locator(Selectors.UNIFIED_STATUS)).to_contain_text("Analysis Complete", timeout=30000)
    page.wait_for_timeout(1000)

    return page
