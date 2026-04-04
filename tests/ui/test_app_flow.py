"""
Playwright E2E Tests for main application workflow.
Restructured for Gradio 5 and robust status tracking using centralized locators.
"""

import pytest
from playwright.sync_api import Page, expect

from .conftest import BASE_URL, open_accordion, switch_to_tab, wait_for_app_ready
from .ui_locators import Labels, Selectors

# Mark all tests as e2e
pytestmark = pytest.mark.e2e


class TestMainWorkflow:
    """Complete end-to-end workflow tests."""

    @pytest.mark.flaky(reruns=3)
    def test_full_user_flow(self, page: Page, app_server):
        """
        Tests the complete end-to-end workflow:
        Extraction -> Pre-Analysis -> Scene Selection -> Propagation -> Analysis -> Export
        """
        page.goto(BASE_URL)
        wait_for_app_ready(page)
        
        unified_status = page.locator(Selectors.UNIFIED_STATUS)

        # 1. Frame Extraction
        page.locator(Selectors.SOURCE_INPUT).fill("dummy_video.mp4")
        page.locator(Selectors.START_EXTRACTION).click()

        # Wait for "complete" in the status area
        expect(unified_status).to_contain_text(Selectors.STATUS_SUCCESS_EXTRACTION, timeout=30000)
        
        # 2. Define Subject (Pre-Analysis)
        switch_to_tab(page, Labels.TAB_SUBJECT)
        
        # Click "Confirm Subject"
        pre_analyze_btn = page.locator(Selectors.START_PRE_ANALYSIS)
        expect(pre_analyze_btn).to_be_visible(timeout=10000)
        pre_analyze_btn.click()

        expect(unified_status).to_contain_text(Selectors.STATUS_SUCCESS_PRE_ANALYSIS, timeout=30000)

        # 3. Scene Selection & Propagation
        switch_to_tab(page, Labels.TAB_SCENES)

        prop_btn = page.locator(Selectors.PROPAGATE_MASKS)
        expect(prop_btn).to_be_visible(timeout=10000)
        expect(prop_btn).to_be_enabled(timeout=5000)
        prop_btn.click()

        expect(unified_status).to_contain_text(Selectors.STATUS_SUCCESS_PROPAGATION, timeout=30000)

        # 4. Analysis
        switch_to_tab(page, Labels.TAB_METRICS)

        ana_btn = page.locator(Selectors.START_ANALYSIS)
        expect(ana_btn).to_be_visible(timeout=10000)
        expect(ana_btn).to_be_enabled(timeout=5000)
        ana_btn.click()

        expect(unified_status).to_contain_text(Selectors.STATUS_SUCCESS_ANALYSIS, timeout=30000)

        # 5. Filtering & Export
        switch_to_tab(page, Labels.TAB_EXPORT)

        export_btn = page.locator(Selectors.EXPORT_BUTTON)
        expect(export_btn).to_be_visible(timeout=10000)
        expect(export_btn).to_be_enabled(timeout=5000)
        export_btn.click()
        
        expect(unified_status).to_contain_text(Selectors.STATUS_SUCCESS_EXPORT, timeout=30000)


class TestTabNavigation:
    """Tests for tab navigation and UI responsiveness."""

    def test_all_tabs_accessible(self, page: Page, app_server):
        """Verify all main tabs can be accessed and show expected content."""
        page.goto(BASE_URL)
        wait_for_app_ready(page)

        tabs = [Labels.TAB_SOURCE, Labels.TAB_SUBJECT, Labels.TAB_SCENES, Labels.TAB_METRICS, Labels.TAB_EXPORT]

        for tab_name in tabs:
            switch_to_tab(page, tab_name)
            # Verify tab button is active
            expect(page.get_by_role("tab", name=tab_name)).to_have_attribute("aria-selected", "true", timeout=5000)

    def test_tab_state_preserved(self, page: Page, app_server):
        """Verify tab state is preserved when switching tabs."""
        page.goto(BASE_URL)
        wait_for_app_ready(page)

        source_input = page.locator(Selectors.SOURCE_INPUT)
        source_input.fill("test_video.mp4")

        switch_to_tab(page, Labels.TAB_SUBJECT)
        switch_to_tab(page, Labels.TAB_SOURCE)

        expect(source_input).to_have_value("test_video.mp4")


class TestErrorHandling:
    """Tests for error display and recovery."""

    def test_empty_source_shows_message(self, page: Page, app_server):
        """Verify appropriate message when no source is provided."""
        page.goto(BASE_URL)
        wait_for_app_ready(page)

        page.locator(Selectors.START_EXTRACTION).click()

        # Check status for Error
        expect(page.locator(Selectors.UNIFIED_STATUS)).to_contain_text(Selectors.STATUS_ERROR_REGEX, timeout=10000)

        # Check logs
        open_accordion(page, Labels.SYSTEM_LOGS)
        expect(page.locator(Selectors.LOG_TEXTAREA)).to_contain_text("Error", timeout=10000)
