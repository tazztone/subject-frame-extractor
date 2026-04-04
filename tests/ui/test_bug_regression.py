"""
Playwright E2E Tests for Bug Regression Prevention.
Standardized to use the new unified Selectors and Labels contract.
"""

import pytest
from playwright.sync_api import Page, expect

from .conftest import BASE_URL, open_accordion, switch_to_tab, wait_for_app_ready
from .ui_locators import Labels, Selectors

# Mark all tests as e2e
pytestmark = pytest.mark.e2e


class TestPaginationBugRegression:
    """Tests to prevent pagination crash regression (Bug 2)."""

    def test_next_button_on_empty_gallery_no_crash(self, page: Page, app_server):
        """Clicking Next on empty/single-page gallery should not crash."""
        page.goto(BASE_URL)
        wait_for_app_ready(page)

        switch_to_tab(page, Labels.TAB_SCENES)

        # Try to click Next - should not crash application
        next_button = page.locator(Selectors.NEXT_PAGE_BUTTON)
        # Even if disabled, clicking should not crash
        next_button.click(force=True)
        page.wait_for_timeout(500)

        # App should still be responsive
        expect(page.locator("body")).to_be_visible()

    def test_prev_button_on_page_one_no_crash(self, page: Page, app_server):
        """Clicking Previous on page 1 should not crash."""
        page.goto(BASE_URL)
        wait_for_app_ready(page)

        switch_to_tab(page, Labels.TAB_SCENES)

        prev_button = page.locator(Selectors.PREV_PAGE_BUTTON)
        prev_button.click(force=True)
        page.wait_for_timeout(500)
        expect(page.locator("body")).to_be_visible()


class TestFindPeopleButtonRegression:
    """Tests to prevent Find People button regression (Bug 3)."""

    def test_find_people_button_visible_in_face_strategy(self, page: Page, app_server):
        """Find People button should be visible when 'By Face' strategy selected."""
        page.goto(BASE_URL)
        wait_for_app_ready(page)

        switch_to_tab(page, Labels.TAB_SUBJECT)

        # Select Face strategy - use label-based click
        page.get_by_label(Labels.STRATEGY_FACE, exact=False).click()
        page.wait_for_timeout(500)

        # Click the Scan Video tab
        page.get_by_role("tab", name=Labels.TAB_SCAN_VIDEO, exact=False).click()

        # Find People button should be visible
        find_people_btn = page.get_by_role("button", name=Labels.SCAN_VIDEO_BUTTON, exact=False)
        expect(find_people_btn).to_be_visible(timeout=5000)


class TestFilterSlidersRegression:
    """Tests to prevent filter slider bugs (Bug 4)."""

    def test_scenes_tab_has_filter_sliders(self, page: Page, app_server):
        """Scenes tab should have properly ranged filter sliders."""
        page.goto(BASE_URL)
        wait_for_app_ready(page)

        switch_to_tab(page, Labels.TAB_SCENES)

        # Open Scene Filtering accordion
        open_accordion(page, "Batch Filter Scenes")

        # Check for Face Similarity slider
        face_sim_slider = page.locator(f"{Selectors.SCENE_FACE_SIM_MIN} input[type=range]")
        expect(face_sim_slider).to_be_attached(timeout=5000)

        # Check for Quality Score slider
        quality_slider = page.locator(f"{Selectors.SCENE_QUALITY_SCORE_MIN} input[type=range]")
        expect(quality_slider).to_be_attached(timeout=5000)

    def test_export_tab_has_filter_sliders(self, page: Page, app_server):
        """Export tab filtering should have proper metric sliders."""
        page.goto(BASE_URL)
        wait_for_app_ready(page)

        switch_to_tab(page, Labels.TAB_EXPORT)

        # Should have at least some range inputs (sliders)
        sliders = page.locator("input[type='range']")
        expect(sliders.first).to_be_attached(timeout=5000)


class TestGallerySizeControlsRegression:
    """Tests to prevent gallery sizing issues (Bug 5)."""

    def test_gallery_size_controls_exist(self, page: Page, app_server):
        """Scene gallery should have columns and height controls."""
        page.goto(BASE_URL)
        wait_for_app_ready(page)

        switch_to_tab(page, Labels.TAB_SCENES)
        open_accordion(page, "Display Settings")

        # Check for Columns slider
        columns_slider = page.get_by_label("Columns").first
        expect(columns_slider).to_be_visible()

        # Check for Height slider
        height_slider = page.get_by_label("Height").first
        expect(height_slider).to_be_visible()


class TestSystemLogsRegression:
    """Tests to prevent system log visibility issues (Bug 6)."""

    def test_logs_accordion_exists(self, page: Page, app_server):
        """System Logs accordion should be present."""
        page.goto(BASE_URL)
        wait_for_app_ready(page)

        logs_accordion = page.get_by_role("button", name="📋 System Logs", exact=False)
        expect(logs_accordion).to_be_visible(timeout=5000)

    def test_refresh_logs_button_exists(self, page: Page, app_server):
        """Refresh Logs button should be present for manual log updates."""
        page.goto(BASE_URL)
        wait_for_app_ready(page)

        open_accordion(page, Labels.SYSTEM_LOGS)

        # Check for Refresh button
        refresh_btn = page.get_by_role("button", name="Refresh", exact=False)
        expect(refresh_btn).to_be_visible(timeout=5000)

    def test_clear_logs_button_works(self, page: Page, app_server):
        """Clear Logs button should clear the log display."""
        page.goto(BASE_URL)
        wait_for_app_ready(page)

        open_accordion(page, Labels.SYSTEM_LOGS)

        # Click Clear button
        clear_btn = page.get_by_role("button", name="Clear", exact=False)
        if clear_btn.is_visible():
            clear_btn.click()
            # Log area should be empty or show initial message
            expect(page.locator(Selectors.LOG_TEXTAREA)).to_have_value("", timeout=5000)


class TestPropagationErrorHandling:
    """Tests for propagation error handling (Bug 1)."""

    def test_propagate_button_found(self, page: Page, app_server):
        """Propagate button should be present on the Scenes tab."""
        page.goto(BASE_URL)
        wait_for_app_ready(page)

        switch_to_tab(page, Labels.TAB_SCENES)

        # Propagate button should exist
        propagate_btn = page.locator(Selectors.PROPAGATE_MASKS)
        expect(propagate_btn).to_be_attached(timeout=5000)
