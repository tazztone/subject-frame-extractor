import pytest
from playwright.sync_api import Page

from .conftest import BASE_URL, switch_to_tab, wait_for_app_ready
from .ui_locators import Labels
from .visual_test_utils import capture_state_screenshot, compare_with_baseline, save_as_baseline

pytestmark = [pytest.mark.e2e, pytest.mark.visual]


@pytest.mark.parametrize(
    "tab_label, snapshot_name",
    [
        (Labels.TAB_SOURCE, "tab_source_empty"),
    ],
)
def test_visual_baselines(page: Page, app_server, request, tab_label, snapshot_name):
    """
    Visual regression test across main tabs in clean state.
    """
    page.goto(BASE_URL)
    wait_for_app_ready(page)

    if tab_label != Labels.TAB_SOURCE:
        switch_to_tab(page, tab_label)
        page.wait_for_timeout(1000)

    screenshot_path = capture_state_screenshot(page, snapshot_name, wait_ms=1000)

    if request.config.getoption("--update-baselines"):
        save_as_baseline(screenshot_path)
        pytest.skip(f"Updated baseline for {snapshot_name}")
    else:
        result = compare_with_baseline(screenshot_path)
        if result["status"] == "no_baseline":
            pytest.skip("No baseline found. Run with --update-baselines to generate one.")
        elif result["status"] == "skip":
            pytest.skip(f"Skipping visual test: {result.get('reason')}")
        else:
            assert result["status"] == "pass", (
                f"Visual mismatch on {snapshot_name}: score {result.get('diff_score', 'N/A')}"
            )
