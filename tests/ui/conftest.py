import subprocess
import sys
import time
from os import environ
from pathlib import Path

import pytest
import requests
from playwright.sync_api import Page, expect

from .app_driver import BASE_URL, PORT, AppDriver
from .ui_locators import Labels, Selectors

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
        time.sleep(0.1)
    return False


def cleanup_port(port: int):
    """Forcefully kill any process using the specified port."""
    try:
        if sys.platform != "win32":
            subprocess.run(["fuser", "-k", f"{port}/tcp"], capture_output=True, check=False)
    except Exception:
        pass


@pytest.fixture(scope="session")
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
def app_instance(app_server):
    """
    Retrieve the live AppUI instance from the running server.
    Note: This only works if the server is running in the same process.
    For subprocess-based servers, this will return None unless captured via IPC.
    """
    try:
        from tests.mock_app import get_active_app

        return get_active_app()
    except ImportError:
        return None


@pytest.fixture
def app_driver(page: Page, app_server):
    """
    Canonical E2E entry point: a Playwright page with the mock app loaded and
    reset to idle, wrapped in an ``AppDriver``.

    Tests consume this instead of the raw ``(page, app_server)`` pair and drive
    the UI through driver verbs (``extract``, ``expect_status``, ...). The
    driver is the single owner of Gradio DOM-synchronization knowledge; see
    ``app_driver.py`` for the contract.
    """
    driver = AppDriver(page)
    driver.goto_app()
    return driver


@pytest.fixture(scope="module")
def shared_page(browser):
    """
    Provides a module-scoped page to amortize browser/context startup.
    Useful for tests that share heavy setup state.
    """
    context = browser.new_context()
    page = context.new_page()
    page.set_default_timeout(15000)
    yield page
    context.close()


@pytest.fixture(scope="module")
def shared_analysis_session(shared_page, app_server):
    """
    Provides a shared analysis session (module-scoped).
    Runs the full analysis pipeline once and shares the resulting state.
    """
    page = shared_page
    driver = AppDriver(page)
    page.goto(BASE_URL)
    driver.goto_app()

    # 1. Extraction
    source_input = page.get_by_placeholder(Labels.SOURCE_PLACEHOLDER)
    if not source_input.is_visible():
        source_input = page.locator(Selectors.SOURCE_INPUT)
    source_input.fill("dummy_video.mp4")
    page.locator(Selectors.START_EXTRACTION).click()
    expect(page.locator(Selectors.UNIFIED_STATUS)).to_contain_text(Selectors.STATUS_SUCCESS_EXTRACTION, timeout=30000)
    page.wait_for_timeout(500)

    # 2. Pre-Analysis
    driver.navigate(Labels.TAB_SUBJECT)
    page.locator(Selectors.START_PRE_ANALYSIS).click()
    expect(page.locator(Selectors.UNIFIED_STATUS)).to_contain_text(Selectors.STATUS_SUCCESS_PRE_ANALYSIS, timeout=30000)
    page.wait_for_timeout(500)

    # 3. Propagation
    driver.navigate(Labels.TAB_SCENES)
    propagate_btn = page.locator(Selectors.PROPAGATE_MASKS)
    # The button becomes visible only after pre-analysis results are populated
    expect(propagate_btn).to_be_visible(timeout=10000)
    propagate_btn.click(force=True)
    expect(page.locator(Selectors.UNIFIED_STATUS)).to_contain_text(Selectors.STATUS_SUCCESS_PROPAGATION, timeout=30000)
    page.wait_for_timeout(1000)

    # 4. Analysis
    driver.navigate(Labels.TAB_METRICS)
    page.locator(Selectors.START_ANALYSIS).click()
    expect(page.locator(Selectors.UNIFIED_STATUS)).to_contain_text(Selectors.STATUS_SUCCESS_ANALYSIS, timeout=30000)
    page.wait_for_timeout(500)

    return page
