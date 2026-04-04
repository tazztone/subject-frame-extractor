"""
Visual regression tests - captures UI states and compares to baselines.
Standardized to use the new unified Selectors and Labels contract.
"""

import pytest
from playwright.sync_api import Page

from .conftest import BASE_URL, open_accordion, switch_to_tab, wait_for_app_ready
from .ui_locators import Labels, Selectors

try:
    from .visual_test_utils import capture_state_screenshot, compare_with_baseline, save_as_baseline

    HAS_UTILS = True
except ImportError:
    HAS_UTILS = False

pytestmark = [pytest.mark.e2e, pytest.mark.visual]


class TestVisualRegression:
    """Screenshot-based visual regression tests."""

    # UI states to capture and compare
    # Format: (state_name, setup_action)
    UI_STATES = [
        ("01_source_tab_initial", None),
        (
            "02_source_tab_with_input",
            lambda p: p.get_by_placeholder(Labels.SOURCE_PLACEHOLDER).fill("sample_video.mp4"),
        ),
        ("03_subject_tab_initial", lambda p: switch_to_tab(p, Labels.TAB_SUBJECT)),
        ("04_subject_face_strategy", lambda p: _click_strategy(p, Labels.STRATEGY_FACE)),
        ("05_subject_text_strategy", lambda p: _click_strategy(p, Labels.STRATEGY_TEXT)),
        ("06_scenes_tab_initial", lambda p: switch_to_tab(p, Labels.TAB_SCENES)),
        ("07_metrics_tab_initial", lambda p: switch_to_tab(p, Labels.TAB_METRICS)),
        ("08_export_tab_initial", lambda p: switch_to_tab(p, Labels.TAB_EXPORT)),
        ("09_logs_accordion_open", lambda p: open_accordion(p, Labels.SYSTEM_LOGS)),
        ("10_help_accordion_open", lambda p: open_accordion(p, Labels.HELP_ACCORDION)),
    ]

    @pytest.mark.skipif(not HAS_UTILS, reason="visual_test_utils dependencies not installed")
    @pytest.mark.parametrize("state_name,action", UI_STATES)
    def test_ui_state_visual(self, page: Page, app_server, state_name, action, request):
        """Capture and compare UI state screenshot."""
        page.goto(BASE_URL)
        wait_for_app_ready(page)

        # Execute setup action if provided
        if action:
            try:
                action(page)
                page.wait_for_timeout(1000)  # Wait for state change/animation
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
        wait_for_app_ready(page)

        # Fill in some data on Source tab
        page.get_by_placeholder(Labels.SOURCE_PLACEHOLDER).fill("test_video.mp4")

        # Capture initial state
        initial = capture_state_screenshot(page, "consistency_initial")

        # Switch away and back
        switch_to_tab(page, Labels.TAB_SUBJECT)
        switch_to_tab(page, Labels.TAB_METRICS) # Switch to a mid-flow tab
        switch_to_tab(page, Labels.TAB_SOURCE)

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
def _click_strategy(page: Page, label: str):
    """Select a subject discovery strategy."""
    switch_to_tab(page, Labels.TAB_SUBJECT)
    page.get_by_label(label).check(force=True)
    page.wait_for_timeout(500)
