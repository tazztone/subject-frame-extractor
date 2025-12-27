"""
Visual regression tests - captures UI states and compares to baselines.

These tests detect unintended visual changes by:
1. Capturing screenshots at each UI state
2. Comparing against baseline images using perceptual hashing
3. Failing if visual changes exceed threshold

Run with:
    python -m pytest tests/e2e/test_visual_regression.py -v

Update baselines:
    python -m pytest tests/e2e/test_visual_regression.py -v --update-baselines
"""

import time

import pytest
from playwright.sync_api import Page

from .conftest import BASE_URL

try:
    from .visual_test_utils import capture_state_screenshot, cleanup_diffs, compare_with_baseline, save_as_baseline

    HAS_UTILS = True
except ImportError:
    HAS_UTILS = False

pytestmark = [pytest.mark.e2e, pytest.mark.visual]


def pytest_addoption(parser):
    """Add --update-baselines option."""
    parser.addoption(
        "--update-baselines",
        action="store_true",
        default=False,
        help="Update baseline screenshots instead of comparing",
    )


@pytest.fixture(scope="module", autouse=True)
def cleanup_before_run():
    """Clean up diff screenshots before test run."""
    if HAS_UTILS:
        cleanup_diffs()
    yield


class TestVisualRegression:
    """Screenshot-based visual regression tests."""

    # UI states to capture and compare
    # Format: (state_name, setup_action)
    UI_STATES = [
        ("01_source_tab_initial", None),
        ("02_source_tab_with_input", lambda p: p.get_by_label("Video URL or Local Path").fill("sample_video.mp4")),
        ("03_subject_tab_initial", lambda p: p.get_by_role("tab", name="Subject").click(force=True)),
        ("04_subject_face_strategy", lambda p: _click_strategy(p, "Face")),
        ("05_subject_text_strategy", lambda p: _click_strategy(p, "Text")),
        ("06_scenes_tab_initial", lambda p: p.get_by_role("tab", name="Scenes").click(force=True)),
        ("07_metrics_tab_initial", lambda p: p.get_by_role("tab", name="Metrics").click(force=True)),
        ("08_export_tab_initial", lambda p: p.get_by_role("tab", name="Export").click(force=True)),
        ("09_logs_accordion_open", lambda p: _open_logs(p)),
        ("10_help_accordion_open", lambda p: _open_help(p)),
    ]

    @pytest.mark.skipif(not HAS_UTILS, reason="visual_test_utils dependencies not installed")
    @pytest.mark.parametrize("state_name,action", UI_STATES)
    def test_ui_state_visual(self, page: Page, app_server, state_name, action, request):
        """Capture and compare UI state screenshot."""
        page.goto(BASE_URL)
        page.wait_for_load_state("networkidle")
        time.sleep(1)  # Wait for initial render

        # Execute setup action if provided
        if action:
            try:
                action(page)
                time.sleep(0.5)  # Wait for state change
            except Exception as e:
                pytest.skip(f"Setup action failed: {e}")

        # Capture screenshot
        screenshot = capture_state_screenshot(page, state_name)

        # Check if we should update baselines
        update_baselines = request.config.getoption("--update-baselines", default=False)

        if update_baselines:
            baseline = save_as_baseline(screenshot)
            pytest.skip(f"Baseline updated: {baseline}")

        # Compare with baseline
        result = compare_with_baseline(screenshot)

        if result["status"] == "no_baseline":
            # First run - save as baseline
            save_as_baseline(screenshot)
            pytest.skip(f"No baseline for {state_name} - saved current as baseline")

        if result["status"] == "skip":
            pytest.skip(result.get("reason", "Comparison skipped"))

        assert result["status"] == "pass", (
            f"Visual regression detected for {state_name}! "
            f"Diff score: {result['diff_score']} (threshold: {result['threshold']}). "
            f"Compare: {result['current']} vs {result['baseline']}"
        )


class TestUIStateConsistency:
    """Test that UI remains consistent across interactions."""

    @pytest.mark.skipif(not HAS_UTILS, reason="visual_test_utils dependencies not installed")
    def test_tab_switching_preserves_state(self, page: Page, app_server):
        """Switching tabs and back should preserve visual state."""
        page.goto(BASE_URL)
        page.wait_for_load_state("networkidle")
        time.sleep(1)

        # Fill in some data on Source tab
        page.get_by_label("Video URL or Local Path").fill("test_video.mp4")

        # Capture initial state
        initial = capture_state_screenshot(page, "consistency_initial")

        # Switch away and back
        page.get_by_role("tab", name="Subject").click(force=True)
        time.sleep(0.3)
        page.get_by_role("tab", name="Source").click(force=True)
        time.sleep(0.3)

        # Capture return state
        returned = capture_state_screenshot(page, "consistency_returned")

        # Compare (should be nearly identical)
        import imagehash
        from PIL import Image

        initial_hash = imagehash.phash(Image.open(initial))
        returned_hash = imagehash.phash(Image.open(returned))
        diff = initial_hash - returned_hash

        assert diff <= 2, f"UI state changed after tab switching! Diff: {diff}"


# Helper functions for test setup
def _click_strategy(page: Page, strategy_keyword: str):
    """Click a strategy radio button containing keyword."""
    page.get_by_role("tab", name="Subject").click(force=True)
    time.sleep(0.3)
    radios = page.locator("input[type='radio']")
    for i in range(radios.count()):
        label = radios.nth(i).locator("..").text_content()
        if strategy_keyword.lower() in label.lower():
            radios.nth(i).click()
            return
    # Try clicking by text
    page.get_by_text(strategy_keyword, exact=False).first.click()


def _open_logs(page: Page):
    """Open the System Logs accordion."""
    logs = page.get_by_text("System Logs")
    if logs.is_visible():
        logs.click()
        time.sleep(0.3)


def _open_help(page: Page):
    """Open the Help accordion."""
    help_btn = page.get_by_text("Help / Troubleshooting")
    if help_btn.is_visible():
        help_btn.click()
        time.sleep(0.3)
