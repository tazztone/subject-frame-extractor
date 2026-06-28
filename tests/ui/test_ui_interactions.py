"""
Playwright E2E Tests for specific UI interactions.
Driven through ``AppDriver``; raw ``page`` retained for the idioms the driver
does not model (sub-tab clicks, slider fill+Enter, console listeners).
"""

import pytest
from playwright.sync_api import expect

from .app_driver import AppDriver
from .ui_locators import Labels, Selectors

pytestmark = pytest.mark.e2e


class TestInteractiveComponents:
    """Tests for individual interactive components."""

    def test_find_people_button_visibility(self, app_driver: AppDriver):
        """Verify the Find People button appears when face strategy is selected."""
        page = app_driver.navigate(Labels.TAB_SUBJECT).page

        app_driver.select_strategy(Labels.STRATEGY_FACE)
        expect(page.get_by_role("tab", name=Labels.TAB_SCAN_VIDEO, exact=False)).to_be_visible(timeout=5000)

        # Click the "Scan Video for Subjects" sub-tab
        page.get_by_role("tab", name=Labels.TAB_SCAN_VIDEO, exact=False).click()

        # Find People button should be visible
        btn = page.get_by_role("button", name=Labels.SCAN_VIDEO_BUTTON, exact=False)
        expect(btn).to_be_visible(timeout=5000)

    def test_logs_accordion_works(self, app_driver: AppDriver):
        """Verify the System Logs accordion opens and shows content."""
        app_driver.open_accordion(Labels.SYSTEM_LOGS)
        app_driver.expect_visible(Selectors.LOG_TEXTAREA, timeout=5000)


class TestWorkflowInteractions:
    """Tests for cross-component interactions."""

    def test_face_strategy_shows_upload(self, app_driver: AppDriver):
        """Verify that selecting Face strategy shows upload component."""
        app_driver.navigate(Labels.TAB_SUBJECT).select_strategy(Labels.STRATEGY_FACE)

        expect(app_driver.page.get_by_text("Upload Reference Photo", exact=False)).to_be_visible()

    def test_text_strategy_shows_prompt(self, app_driver: AppDriver):
        """Verify that selecting Text strategy shows prompt input."""
        app_driver.navigate(Labels.TAB_SUBJECT).select_strategy(Labels.STRATEGY_TEXT)

        expect(app_driver.page.get_by_placeholder("e.g., 'a man in a blue suit'")).to_be_visible()


class TestErrorScenarios:
    """Tests for graceful error handling in UI."""

    def test_find_people_graceful_error_handling(self, app_driver: AppDriver):
        """Verify that Scan Video Now handles errors gracefully when no source is set."""
        page = app_driver.navigate(Labels.TAB_SUBJECT).page

        app_driver.select_strategy(Labels.STRATEGY_FACE)
        expect(page.get_by_role("tab", name=Labels.TAB_SCAN_VIDEO, exact=False)).to_be_visible(timeout=5000)

        # Ensure we are on Scan Video tab, then click without setting an input.
        page.get_by_role("tab", name=Labels.TAB_SCAN_VIDEO, exact=False).click()
        page.get_by_role("button", name=Labels.SCAN_VIDEO_BUTTON, exact=False).click()

        # Should show informational status
        app_driver.expect_status("Face Discovery Finished", timeout=10000)


class TestGallerySliderInteractions:
    """Tests for gallery size sliders."""

    def test_columns_slider_exists_and_interactive(self, app_driver: AppDriver):
        """Columns slider should exist and be draggable."""
        page = app_driver.navigate(Labels.TAB_SCENES).page
        app_driver.open_accordion("Display Settings")

        # Gradio 5+ range sliders often have an input[type=range]; columns is first
        columns_slider = page.locator("input[type=range]").first
        expect(columns_slider).to_be_visible(timeout=5000)

        columns_slider.fill("4")
        columns_slider.press("Enter")
        page.wait_for_timeout(500)

    def test_height_slider_exists_and_interactive(self, app_driver: AppDriver):
        """Height slider should exist and be adjustable."""
        page = app_driver.navigate(Labels.TAB_SCENES).page
        app_driver.open_accordion("Display Settings")

        # In scene_handler.py, height is the second range slider in that group
        height_slider = page.locator("input[type=range]").nth(1)
        expect(height_slider).to_be_visible(timeout=5000)

        height_slider.fill("400")
        height_slider.press("Enter")
        page.wait_for_timeout(500)


class TestLogRefreshMechanism:
    """Tests for log display."""

    def test_refresh_button_updates_logs(self, app_driver: AppDriver):
        """Clicking Refresh should drain log queue and update display."""
        app_driver.open_accordion(Labels.SYSTEM_LOGS)
        app_driver.expect_visible(Selectors.LOG_TEXTAREA, timeout=5000)

        refresh_btn = app_driver.page.get_by_role("button", name="Refresh", exact=False)
        expect(refresh_btn).to_be_visible(timeout=5000)
        refresh_btn.click()
        app_driver.page.wait_for_timeout(1000)

        app_driver.expect_visible(Selectors.LOG_TEXTAREA)


class TestUIConsoleErrors:
    """Tests that monitor browser console for JavaScript errors."""

    def test_no_console_errors_on_load(self, page, app_server):
        """Page should load without JavaScript errors.

        Uses raw ``page`` (not ``app_driver``) because the console listener must
        be attached *before* navigation — the driver's ``goto_app`` would miss
        load-time errors otherwise.
        """
        console_errors = []
        page.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None)

        from .app_driver import AppDriver as _AD

        _AD(page).goto_app()

        # Filter out known benign errors (e.g. Gradio internal warnings)
        critical_errors = [e for e in console_errors if "gradio" not in e.lower() and "favicon" not in e.lower()]
        assert len(critical_errors) == 0, f"Critical console errors found: {critical_errors}"

    def test_no_errors_during_tab_navigation(self, app_driver: AppDriver):
        """Navigating through tabs should not cause errors."""
        console_errors = []
        app_driver.page.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None)

        tabs = [Labels.TAB_SOURCE, Labels.TAB_SUBJECT, Labels.TAB_SCENES, Labels.TAB_METRICS, Labels.TAB_EXPORT]
        for tab_name in tabs:
            app_driver.navigate(tab_name)

        critical_errors = [e for e in console_errors if "gradio" not in e.lower()]
        assert len(critical_errors) == 0, f"Critical console errors during nav: {critical_errors}"
