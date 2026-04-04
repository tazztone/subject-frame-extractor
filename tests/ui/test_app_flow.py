"""
Playwright E2E Tests for main application workflow.
Restructured for Gradio 5 and robust status tracking using centralized locators.
"""

import pytest
from playwright.sync_api import Page, expect

from .conftest import BASE_URL, Labels, Selectors, open_accordion, switch_to_tab, wait_for_app_ready


@pytest.mark.usefixtures("app_server")
class TestMainWorkflow:
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
        page.locator(Selectors.START_EXTRACTION).click(force=True)

        # Wait for success card (HTML)
        # We focus on the text content which is more robust than exact string matches
        expect(unified_status).to_contain_text("Extraction Complete", timeout=15000)

        # 2. Define Subject (Pre-Analysis)
        switch_to_tab(page, Labels.TAB_SUBJECT)

        # Click "Confirm Subject"
        pre_analyze_btn = page.locator(Selectors.START_PRE_ANALYSIS)
        expect(pre_analyze_btn).to_be_visible(timeout=10000)
        pre_analyze_btn.click(force=True)

        expect(unified_status).to_contain_text("Pre-Analysis Complete", timeout=15000)

        # 3. Scene Selection & Propagation
        switch_to_tab(page, Labels.TAB_SCENES)

        prop_btn = page.locator(Selectors.PROPAGATE_MASKS)
        expect(prop_btn).to_be_visible(timeout=10000)
        prop_btn.click(force=True)

        expect(unified_status).to_contain_text("Mask Propagation Complete", timeout=30000)

        # 4. Final Analysis
        switch_to_tab(page, Labels.TAB_METRICS)

        analyze_btn = page.locator(Selectors.START_ANALYSIS)
        expect(analyze_btn).to_be_visible(timeout=10000)
        analyze_btn.click(force=True)

        expect(unified_status).to_contain_text("Analysis Complete", timeout=30000)

        # 5. Export
        switch_to_tab(page, Labels.TAB_EXPORT)

        export_btn = page.locator(Selectors.EXPORT_BUTTON)
        expect(export_btn).to_be_visible(timeout=10000)
        export_btn.click(force=True)

        # We don't have a mock for Export Complete yet in mock_app.py
        # but the button should be clickable and not crash the UI.


@pytest.mark.usefixtures("app_server")
class TestTabNavigation:
    def test_all_tabs_accessible(self, page: Page, app_server):
        """Verify each major tab can be reached."""
        page.goto(BASE_URL)
        wait_for_app_ready(page)

        tabs = [Labels.TAB_SOURCE, Labels.TAB_SUBJECT, Labels.TAB_SCENES, Labels.TAB_METRICS, Labels.TAB_EXPORT]
        for tab_label in tabs:
            switch_to_tab(page, tab_label)
            # Check if a unique element for that tab is visible
            if tab_label == Labels.TAB_SOURCE:
                expect(page.locator(Selectors.START_EXTRACTION)).to_be_visible()
            elif tab_label == Labels.TAB_SUBJECT:
                expect(page.locator(Selectors.START_PRE_ANALYSIS)).to_be_visible()

    def test_tab_state_preserved(self, page: Page, app_server):
        """Verify input state is kept when switching tabs."""
        page.goto(BASE_URL)
        wait_for_app_ready(page)

        # Fill input on first tab
        test_val = "PERSISTENT_VALUE.mp4"
        page.locator(Selectors.SOURCE_INPUT).fill(test_val)

        # Switch away and back
        switch_to_tab(page, Labels.TAB_SUBJECT)
        switch_to_tab(page, Labels.TAB_SOURCE)

        expect(page.locator(Selectors.SOURCE_INPUT)).to_have_value(test_val)


@pytest.mark.usefixtures("app_server")
class TestErrorHandling:
    def test_empty_source_shows_message(self, page: Page, app_server):
        """Verify appropriate message when no source is provided."""
        page.goto(BASE_URL)
        wait_for_app_ready(page)

        unified_status = page.locator(Selectors.UNIFIED_STATUS)

        # Click extraction (will fail due to empty path)
        page.locator(Selectors.START_EXTRACTION).click(force=True)

        # Check status for Error
        expect(unified_status).to_contain_text(Selectors.STATUS_ERROR_REGEX, timeout=10000)

        # Open logs to ensure component is active/visible
        open_accordion(page, Labels.SYSTEM_LOGS)

        # Manual refresh to pull from queue immediately
        page.locator(Selectors.REFRESH_LOGS).click()
        # Settle time for Gradio update
        page.wait_for_timeout(500)

        # Check logs using the textarea inside the container (Gradio 5 standard)
        expect(page.locator(Selectors.LOG_TEXTAREA)).to_contain_text("Error", timeout=10000)
