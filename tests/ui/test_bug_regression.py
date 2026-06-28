"""
Playwright E2E Tests for Bug Regression Prevention.
Driven through ``AppDriver``.
"""

import pytest
from playwright.sync_api import expect

from .app_driver import AppDriver
from .ui_locators import Labels, Selectors

# Mark all tests as e2e
pytestmark = pytest.mark.e2e


class TestPaginationBugRegression:
    """Tests to prevent pagination crash regression (Bug 2)."""

    def test_next_button_on_empty_gallery_no_crash(self, app_driver: AppDriver):
        """Clicking Next on empty/single-page gallery should not crash."""
        app_driver.navigate(Labels.TAB_SCENES)

        # Even if disabled, clicking should not crash the application
        app_driver.page.locator(Selectors.NEXT_PAGE_BUTTON).click(force=True)
        app_driver.page.wait_for_timeout(500)

        # App should still be responsive
        expect(app_driver.page.locator("body")).to_be_visible()

    def test_prev_button_on_page_one_no_crash(self, app_driver: AppDriver):
        """Clicking Previous on page 1 should not crash."""
        app_driver.navigate(Labels.TAB_SCENES)

        app_driver.page.locator(Selectors.PREV_PAGE_BUTTON).click(force=True)
        app_driver.page.wait_for_timeout(500)
        expect(app_driver.page.locator("body")).to_be_visible()


class TestFindPeopleButtonRegression:
    """Tests to prevent Find People button regression (Bug 3)."""

    def test_find_people_button_visible_in_face_strategy(self, app_driver: AppDriver):
        """Find People button should be visible when 'By Face' strategy selected."""
        app_driver.navigate(Labels.TAB_SUBJECT)
        app_driver.select_strategy(Labels.STRATEGY_FACE)
        app_driver.page.wait_for_timeout(500)

        # Click the Scan Video tab
        app_driver.page.get_by_role("tab", name=Labels.TAB_SCAN_VIDEO, exact=False).click()

        # Find People button should be visible
        btn = app_driver.page.get_by_role("button", name=Labels.SCAN_VIDEO_BUTTON, exact=False)
        expect(btn).to_be_visible(timeout=5000)


class TestFilterSlidersRegression:
    """Tests to prevent filter slider bugs (Bug 4)."""

    def test_scenes_tab_has_filter_sliders(self, app_driver: AppDriver):
        """Scenes tab should have properly ranged filter sliders."""
        app_driver.navigate(Labels.TAB_SCENES)
        app_driver.open_accordion("Batch Filter Scenes")

        expect(app_driver.page.locator(f"{Selectors.SCENE_FACE_SIM_MIN} input[type=range]")).to_be_attached(
            timeout=5000
        )
        expect(app_driver.page.locator(f"{Selectors.SCENE_QUALITY_SCORE_MIN} input[type=range]")).to_be_attached(
            timeout=5000
        )

    def test_export_tab_has_filter_sliders(self, app_driver: AppDriver):
        """Export tab filtering should have proper metric sliders."""
        app_driver.navigate(Labels.TAB_EXPORT)

        sliders = app_driver.page.locator("input[type='range']")
        expect(sliders.first).to_be_attached(timeout=5000)


class TestGallerySizeControlsRegression:
    """Tests to prevent gallery sizing issues (Bug 5)."""

    def test_gallery_size_controls_exist(self, app_driver: AppDriver):
        """Scene gallery should have columns and height controls."""
        app_driver.navigate(Labels.TAB_SCENES)
        app_driver.open_accordion("Display Settings")

        expect(app_driver.page.get_by_label("Columns").first).to_be_visible()
        expect(app_driver.page.get_by_label("Height").first).to_be_visible()


class TestSystemLogsRegression:
    """Tests to prevent system log visibility issues (Bug 6)."""

    def test_logs_accordion_exists(self, app_driver: AppDriver):
        """System Logs accordion should be present."""
        logs_accordion = app_driver.page.get_by_role("button", name="📋 System Logs", exact=False)
        expect(logs_accordion).to_be_visible(timeout=5000)

    def test_refresh_logs_button_exists(self, app_driver: AppDriver):
        """Refresh Logs button should be present for manual log updates."""
        app_driver.open_accordion(Labels.SYSTEM_LOGS)

        refresh_btn = app_driver.page.get_by_role("button", name="Refresh", exact=False)
        expect(refresh_btn).to_be_visible(timeout=5000)

    def test_clear_logs_button_works(self, app_driver: AppDriver):
        """Clear Logs button should clear the log display."""
        app_driver.open_accordion(Labels.SYSTEM_LOGS)

        clear_btn = app_driver.page.get_by_role("button", name="Clear", exact=False)
        if clear_btn.is_visible():
            clear_btn.click()
            # Log area should be empty — exact-value assertion (not a substring poll).
            app_driver.expect_log_equals("", timeout=5000)


class TestPropagationErrorHandling:
    """Tests for propagation error handling (Bug 1)."""

    def test_propagate_button_found(self, analyzed_session):
        """Propagate button should be present on the Scenes tab."""
        driver = AppDriver(analyzed_session)
        driver.navigate(Labels.TAB_SCENES)

        expect(driver.page.locator(Selectors.PROPAGATE_MASKS)).to_be_attached(timeout=5000)
