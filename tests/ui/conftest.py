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
FAILURES_DIR = Path(__file__).parent.parent / "results" / "failures"


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


def wait_for_app_ready(page: Page):
    """
    Blocks until the Gradio application is fully hydrated and ready for interaction.
    Uses framework-native signals (loading indicators) instead of fixed timeouts.
    """
    # 1. Wait for any initial Gradio loading indicators to disappear
    # These are present during the initial JS bundle load and hydration gap.
    page.wait_for_selector(".generating, .loading, [data-testid='loading']", state="hidden", timeout=20000)

    # 2. Wait for the main heading to be visible as a proxy for app load
    expect(page.get_by_text("Frame Extractor & Analyzer")).to_be_visible(timeout=30000)

    # 3. Wait for the status area to be attached
    expect(page.locator(Selectors.UNIFIED_STATUS)).to_be_attached(timeout=5000)

    # 4. Optional: Clear state if the mock-only reset button is present
    reset_btn = page.locator(Selectors.RESET_STATE_BUTTON)
    if not reset_btn.is_visible():
        try:
            open_accordion(page, "Tests (Experimental)")
        except Exception:
            pass

    if reset_btn.is_visible():
        reset_btn.click()

        # CRITICAL: Wait for the reset status to appear to ensure clean state
        status_locator = page.locator(Selectors.UNIFIED_STATUS)
        expect(status_locator).to_contain_text(Selectors.STATUS_READY, timeout=15000)
        expect(page.get_by_text("System Reset Ready.")).to_be_visible(timeout=5000)

    # Small settle for Gradio JS hydration stability
    page.wait_for_timeout(200)



def open_accordion(page: Page, text: str):
    """Robustly open an accordion if it's closed."""
    # Try multiple ways to find the accordion header
    # 1. By text (with fuzzy match for emojis)
    # 2. By elem_id (if we know it)
    elem_id = None
    if "Log" in text:
        elem_id = "#system_logs_accordion"
    elif "Help" in text:
        elem_id = "#help_accordion"
    elif "Deduplication" in text:
        elem_id = "#dedup_accordion"
    elif "Advanced Model Configuration" in text:
        elem_id = "#subject_advanced_config_accordion"

    if elem_id:
        accordion = page.locator(elem_id)
    else:
        accordion = page.get_by_text(text, exact=False).first

    expect(accordion).to_be_visible(timeout=5000)

    # Check if already open using multiple strategies (Gradio 5+ details/button)
    is_open = False
    try:
        # 1. Check if it's a 'details' element with 'open' attribute
        if accordion.evaluate("el => el.tagName === 'DETAILS' && el.open"):
            is_open = True
        # 2. Check aria-expanded on the accordion element or its buttons
        elif accordion.get_attribute("aria-expanded") == "true":
            is_open = True
        elif accordion.locator("button[aria-expanded='true']").count() > 0:
            is_open = True
    except Exception:
        pass

    if not is_open:
        # Click the header/accordion to toggle
        accordion.click()
        # Wait for animation
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
        # Wait for any loading indicator to disappear after tab switch
        page.wait_for_selector(".generating, [data-testid='loading']", state="hidden", timeout=2000)


def cleanup_port(port: int):
    """Forcefully kill any process using the specified port."""
    try:
        if sys.platform != "win32":
            subprocess.run(["fuser", "-k", f"{port}/tcp"], capture_output=True, check=False)
    except Exception:
        pass


@pytest.fixture(scope="module")
def app_server():
    """Starts the mock app server (Legacy alias for mock_only tests)."""
    for process_data in _start_app_server(use_mock=True):
        yield process_data


@pytest.fixture(scope="module")
def live_server():
    """
    Recommended fixture for E2E tests.
    Starts real app.py if PYTEST_INTEGRATION_MODE=true, otherwise runs mock_app.py.
    """
    use_mock = environ.get("PYTEST_INTEGRATION_MODE") != "true"
    for process_data in _start_app_server(use_mock=use_mock):
        yield process_data


def _start_app_server(use_mock: bool):
    """Internal helper to start either real or mock server."""
    cleanup_port(PORT)
    time.sleep(1)

    app_type = "MOCK" if use_mock else "REAL"
    print(f"Starting {app_type} app on port {PORT}...")

    if use_mock:
        app_path = str(Path(__file__).parent.parent / "mock_app.py")
    else:
        app_path = str(Path(__file__).parent.parent.parent / "app.py")

    log_dir = Path(__file__).parent.parent.parent / "tests" / "results" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    worker_id = environ.get("PYTEST_XDIST_WORKER", "master")
    log_file_path = log_dir / f"{app_type.lower()}_app_e2e_{worker_id}.log"

    log_file = open(log_file_path, "w")
    cmd = [sys.executable, app_path]
    if not use_mock:
        # Pass test port to real app
        cmd += ["--server-port", str(PORT)]

    process = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        env={**environ, "APP_SERVER_PORT": str(PORT), "PYTHONUNBUFFERED": "1"},
    )

    if wait_for_server(BASE_URL, timeout=60):
        print(f"✓ {app_type} Server started successfully (Logs: {log_file_path})")
    else:
        print(f"❌ {app_type} Server failed to start within timeout")
        process.terminate()
        with open(log_file_path, "r") as f:
            print(f"App Logs:\n{f.read()[-2000:]}")
        pytest.fail(f"{app_type} server failed to start")

    yield process

    # Cleanup
    print(f"Stopping {app_type} app...")
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
    if log_file and not log_file.closed:
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
        app = get_active_app()
        if app is None:
            pytest.skip("app_instance is not available for subprocess-based servers")
        return app
    except ImportError:
        return None


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
    expect(page.locator(Selectors.UNIFIED_STATUS)).to_contain_text(Selectors.STATUS_SUCCESS_EXTRACTION, timeout=30000)

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
    expect(find_btn).to_be_visible(timeout=15000)
    find_btn.click()

    # Wait for completion (check status)
    expect(page.locator(Selectors.UNIFIED_STATUS)).to_contain_text(Selectors.STATUS_SUCCESS_PRE_ANALYSIS, timeout=30000)

    return page


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
    page.goto(BASE_URL)
    wait_for_app_ready(page)

    # 1. Extraction
    source_input = page.get_by_placeholder(Labels.SOURCE_PLACEHOLDER)
    if not source_input.is_visible():
        source_input = page.locator(Selectors.SOURCE_INPUT)
    source_input.fill("dummy_video.mp4")
    page.locator(Selectors.START_EXTRACTION).click()
    expect(page.locator(Selectors.UNIFIED_STATUS)).to_contain_text(Selectors.STATUS_SUCCESS_EXTRACTION, timeout=30000)
    page.wait_for_timeout(500)

    # 2. Pre-Analysis
    switch_to_tab(page, Labels.TAB_SUBJECT)
    page.locator(Selectors.START_PRE_ANALYSIS).click()
    expect(page.locator(Selectors.UNIFIED_STATUS)).to_contain_text(Selectors.STATUS_SUCCESS_PRE_ANALYSIS, timeout=30000)
    page.wait_for_timeout(500)

    # 3. Propagation
    switch_to_tab(page, Labels.TAB_SCENES)
    propagate_btn = page.locator(Selectors.PROPAGATE_MASKS)
    # The button becomes visible only after pre-analysis results are populated
    expect(propagate_btn).to_be_visible(timeout=10000)
    propagate_btn.click(force=True)
    expect(page.locator(Selectors.UNIFIED_STATUS)).to_contain_text(Selectors.STATUS_SUCCESS_PROPAGATION, timeout=30000)
    page.wait_for_timeout(1000)

    # 4. Analysis
    switch_to_tab(page, Labels.TAB_METRICS)
    page.locator(Selectors.START_ANALYSIS).click()
    expect(page.locator(Selectors.UNIFIED_STATUS)).to_contain_text(Selectors.STATUS_SUCCESS_ANALYSIS, timeout=30000)
    page.wait_for_timeout(500)

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
    propagate_btn = page.locator(Selectors.PROPAGATE_MASKS)
    expect(propagate_btn).to_be_visible(timeout=10000)
    propagate_btn.click(force=True)

    expect(page.locator(Selectors.UNIFIED_STATUS)).to_contain_text(Selectors.STATUS_SUCCESS_PROPAGATION, timeout=30000)
    page.wait_for_timeout(1000)

    # Analyze (Metrics tab)
    switch_to_tab(page, Labels.TAB_METRICS)

    analyze_btn = page.locator(Selectors.START_ANALYSIS)
    expect(analyze_btn).to_be_visible(timeout=5000)
    analyze_btn.click()

    expect(page.locator(Selectors.UNIFIED_STATUS)).to_contain_text(Selectors.STATUS_SUCCESS_ANALYSIS, timeout=30000)
    page.wait_for_timeout(1000)

    return page
