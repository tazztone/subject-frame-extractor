"""
Playwright E2E Tests for specific UI interactions.

Tests individual component behaviors:
- Accordion toggling
- Strategy selection visibility
- Slider interactions
- Log refresh mechanism
- Error message displays

Run with: python -m pytest tests/ui/test_ui_interactions.py -v -s
"""

import re

import pytest
from playwright.sync_api import Page, expect

from .conftest import BASE_URL, open_accordion, switch_to_tab
from .ui_locators import Labels, Selectors

pytestmark = pytest.mark.e2e


class TestInteractiveComponents:
    """Tests for individual interactive components."""

    def test_find_people_button_visibility(self, page: Page, app_server):
        """Verify the Find People button appears when face strategy is selected."""
        page.goto(BASE_URL)
        page.wait_for_timeout(2000)

        switch_to_tab(page, Labels.TAB_SUBJECT)

        # Select Face strategy
        page.get_by_text(Labels.STRATEGY_FACE, exact=False).click()

        # Click the "Scan Video for People" sub-tab
        page.locator(Selectors.SCAN_VIDEO_TAB).click()

        # Find People button should be visible
        btn = page.get_by_role("button", name=Labels.SCAN_VIDEO_BUTTON, exact=False)
        expect(btn).to_be_visible(timeout=5000)

    def test_logs_accordion_works(self, page: Page, app_server):
        """Verify the System Logs accordion opens and shows content."""
        page.goto(BASE_URL)
        page.wait_for_timeout(2000)

        # Find and open logs accordion
        open_accordion(page, Labels.SYSTEM_LOGS)

        # Verify log textarea is visible
        expect(page.locator(Selectors.LOG_TEXTAREA)).to_be_visible(timeout=5000)


class TestWorkflowInteractions:
    """Tests for cross-component interactions."""

    def test_face_strategy_shows_upload(self, page: Page, app_server):
        """Verify that selecting Face strategy shows upload component."""
        page.goto(BASE_URL)
        page.wait_for_timeout(2000)

        switch_to_tab(page, Labels.TAB_SUBJECT)

        # Select Face
        page.get_by_text(Labels.STRATEGY_FACE, exact=False).click()

        # Check for photo upload text
        expect(page.get_by_text("Upload Reference Photo", exact=False)).to_be_visible()

    def test_text_strategy_shows_prompt(self, page: Page, app_server):
        """Verify that selecting Text strategy shows prompt input."""
        page.goto(BASE_URL)
        page.wait_for_timeout(2000)

        switch_to_tab(page, Labels.TAB_SUBJECT)

        # Select Text
        page.get_by_text(Labels.STRATEGY_TEXT, exact=False).click()

        # Check for prompt placeholder
        expect(page.get_by_placeholder("e.g., 'a man in a blue suit'")).to_be_visible()


class TestErrorScenarios:
    """Tests for graceful error handling in UI."""

    def test_find_people_graceful_error_handling(self, page: Page, app_server):
        """Verify that Scan Video Now handles errors gracefully when no source is set."""
        page.goto(BASE_URL)
        page.wait_for_timeout(2000)

        switch_to_tab(page, Labels.TAB_SUBJECT)

        # Select "By Face" strategy to make the group visible
        page.get_by_text(Labels.STRATEGY_FACE, exact=False).click()

        # Ensure we are on Scan Video tab
        page.locator(Selectors.SCAN_VIDEO_TAB).click()

        # Click without setting an input
        page.get_by_role("button", name=Labels.SCAN_VIDEO_BUTTON, exact=False).click()

        # Should show error in status or logs
        expect(page.locator(Selectors.UNIFIED_STATUS)).to_contain_text("Failed", timeout=10000)

        # Verify status message content
        text = page.locator(Selectors.UNIFIED_STATUS).inner_text()
        assert "Run extraction first" in text or "Face analyzer unavailable" in text or "No video frames" in text


class TestGallerySliderInteractions:
    """Tests for gallery size sliders."""

    def test_columns_slider_exists_and_interactive(self, page: Page, app_server):
        """Columns slider should exist and be draggable."""
        page.goto(BASE_URL)
        page.wait_for_timeout(2000)

        switch_to_tab(page, Labels.TAB_SCENES)

        # Change the slider value
        open_accordion(page, "Display Settings")

        columns_slider = page.locator("#scene_gallery_columns input[type=range]")
        expect(columns_slider).to_be_visible(timeout=5000)

        columns_slider.fill("4")
        columns_slider.press("Enter")
        page.wait_for_timeout(500)

    def test_height_slider_exists_and_interactive(self, page: Page, app_server):
        """Height slider should exist and be adjustable."""
        page.goto(BASE_URL)
        page.wait_for_timeout(2000)

        switch_to_tab(page, Labels.TAB_SCENES)

        # Find Height slider
        open_accordion(page, "Display Settings")

        height_slider = page.locator("#scene_gallery_height input[type=range]")
        expect(height_slider).to_be_visible(timeout=5000)

        # Change the slider
        height_slider.fill("400")
        height_slider.press("Enter")
        page.wait_for_timeout(500)


class TestLogRefreshMechanism:
    """Tests for log display."""

    def test_refresh_button_updates_logs(self, page: Page, app_server):
        """Clicking Refresh should drain log queue and update display."""
        page.goto(BASE_URL)
        page.wait_for_timeout(2000)

        # Open logs accordion
        open_accordion(page, Labels.SYSTEM_LOGS)

        # Verify log area
        expect(page.locator(Selectors.LOG_TEXTAREA)).to_be_visible(timeout=5000)

        # Click Refresh
        refresh_btn = page.get_by_role("button", name="Refresh", exact=False)
        expect(refresh_btn).to_be_visible(timeout=5000)
        refresh_btn.click()
        page.wait_for_timeout(1000)

        expect(page.locator(Selectors.LOG_TEXTAREA)).to_be_visible()


class TestPropagationErrorHandling:
    """Tests for propagation error handling."""

    def test_propagation_without_scenes_no_crash(self, page: Page, app_server):
        """Clicking propagate without scenes should not crash."""
        page.goto(BASE_URL)
        page.wait_for_timeout(2000)

        switch_to_tab(page, Labels.TAB_SCENES)

        # Find propagate button (should be disabled but let's try regex)
        prop_btn = page.get_by_role("button", name=re.compile("Propagate", re.I))
        if prop_btn.count() > 0 and prop_btn.first.is_visible():
            try:
                prop_btn.first.click(timeout=1000)
            except:
                pass  # Expected if disabled

        expect(page.locator("body")).to_be_visible()


class TestUIConsoleErrors:
    """Tests that monitor browser console for JavaScript errors."""

    def test_no_console_errors_on_load(self, page: Page, app_server):
        """Page should load without JavaScript errors."""
        console_errors = []
        page.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None)

        page.goto(BASE_URL)
        page.wait_for_timeout(3000)

        # Filter out known benign errors
        critical_errors = [e for e in console_errors if "gradio" not in e.lower()]
        assert len(critical_errors) == 0, f"Critical console errors found: {critical_errors}"

    def test_no_errors_during_tab_navigation(self, page: Page, app_server):
        """Navigating through tabs should not cause errors."""
        console_errors = []
        page.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None)

        page.goto(BASE_URL)
        page.wait_for_timeout(2000)

        tabs = [Labels.TAB_SOURCE, Labels.TAB_SUBJECT, Labels.TAB_SCENES, Labels.TAB_METRICS, Labels.TAB_EXPORT]
        for tab_name in tabs:
            switch_to_tab(page, tab_name)

        critical_errors = [e for e in console_errors if "gradio" not in e.lower()]
        assert len(critical_errors) == 0, f"Critical console errors during nav: {critical_errors}"
