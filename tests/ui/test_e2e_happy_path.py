import re

import pytest
from playwright.sync_api import Page, expect

from .conftest import BASE_URL, open_accordion, switch_to_tab, wait_for_app_ready
from .ui_locators import Labels, Selectors

pytestmark = pytest.mark.e2e


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
        # Use regex to allow for progress prefixes like "processing | 0.0s"
        expect(unified_status).to_contain_text(re.compile(r"Extraction Complete", re.IGNORECASE), timeout=15000)

        # 2. Define Subject (Pre-Analysis)
        switch_to_tab(page, Labels.TAB_SUBJECT)

        # Click "Confirm Subject"
        pre_analyze_btn = page.locator(Selectors.START_PRE_ANALYSIS)
        expect(pre_analyze_btn).to_be_visible(timeout=10000)
        pre_analyze_btn.click(force=True)

        expect(unified_status).to_contain_text(re.compile(r"Pre-Analysis Complete", re.IGNORECASE), timeout=15000)

        # 3. Scene Selection & Propagation
        switch_to_tab(page, Labels.TAB_SCENES)

        prop_btn = page.locator(Selectors.PROPAGATE_MASKS)
        expect(prop_btn).to_be_visible(timeout=10000)
        prop_btn.click(force=True)

        expect(unified_status).to_contain_text(re.compile(r"Mask Propagation Complete", re.IGNORECASE), timeout=30000)

        # 4. Final Analysis
        switch_to_tab(page, Labels.TAB_METRICS)

        analyze_btn = page.locator(Selectors.START_ANALYSIS)
        expect(analyze_btn).to_be_visible(timeout=10000)
        analyze_btn.click(force=True)

        expect(unified_status).to_contain_text(re.compile(r"Analysis Complete", re.IGNORECASE), timeout=30000)

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

        # Check status for Error - Use a broader regex to capture variations
        expect(unified_status).to_contain_text(re.compile(r"Error|Invalid|Failure", re.IGNORECASE), timeout=10000)

        # Open logs to ensure component is active/visible
        open_accordion(page, Labels.SYSTEM_LOGS)

        # Check logs using the textarea inside the container (Gradio 5 standard)
        # Note: LogViewer now has a seed message, and content is in 'value' not 'text'
        expect(page.locator(Selectors.LOG_TEXTAREA)).to_have_value(re.compile(r"ERROR", re.IGNORECASE), timeout=10000)


class TestAdvancedWorkflow:
    """
    Advanced E2E tests covering navigation restrictions and UI responsiveness using robust selectors.
    """

    def test_navigation_restrictions_error(self, page: Page, app_server):
        """
        Verify that attempting to move to Subject/Analysis stage without extraction results in a visible error.
        Tests: Pre-Analysis selection without data.
        """
        page.goto(BASE_URL)
        wait_for_app_ready(page)
        # 1. Switch to Subject tab
        switch_to_tab(page, Labels.TAB_SUBJECT)

        # 2. Click "Confirm Subject"
        pre_analyze_btn = page.locator(Selectors.START_PRE_ANALYSIS)
        expect(pre_analyze_btn).to_be_visible(timeout=10000)

        pre_analyze_btn.click(force=True)

        # 3. Verify Error in Status via visibility
        expect(page.get_by_text(re.compile(r"Error", re.IGNORECASE))).to_be_visible(timeout=10000)

        # 4. Open logs to confirm detail
        open_accordion(page, Labels.SYSTEM_LOGS)
        # Click refresh to force an immediate log render cycle
        page.locator(Selectors.REFRESH_LOGS).click()

        expect(page.locator(Selectors.LOG_TEXTAREA)).to_have_value(re.compile(r"Error", re.IGNORECASE), timeout=12000)

    def test_extraction_settings_persistence(self, page: Page, app_server):
        """
        Verify that changing extraction settings is reflected in the UI and extraction still starts.
        """
        page.goto(BASE_URL)
        wait_for_app_ready(page)

        # 1. Open Advanced Settings
        open_accordion(page, "Advanced Processing Settings")

        # 2. Change Megapixels - append numeric input selector for Gradio 5
        mp_input = page.locator(f"{Selectors.THUMB_MEGAPIXELS} input[type='number']")
        expect(mp_input).to_be_visible()
        mp_input.fill("1.2")
        mp_input.dispatch_event("input")

        # 3. Fill Source
        page.locator(Selectors.SOURCE_INPUT).fill("advanced_settings_test.mp4")

        # 4. Start Extraction
        page.locator(Selectors.START_EXTRACTION).click()

        # 5. Verify completion via visibility
        expect(page.get_by_text(re.compile(r"Extraction Complete", re.IGNORECASE))).to_be_visible(timeout=30000)

    def test_filtering_ui_responsiveness(self, page: Page, app_server):
        """
        Test the Metrics/Filtering tab UI controls responsiveness.
        """
        page.goto(BASE_URL)
        wait_for_app_ready(page)

        # Switch to Export tab
        switch_to_tab(page, Labels.TAB_EXPORT)

        smart_filter = page.get_by_label("Smart Filtering")
        expect(smart_filter).to_be_visible()

        # Toggle Smart Filtering
        smart_filter.check()
        expect(smart_filter).to_be_checked()

        # Verify that clicking it doesn't cause a global error box
        expect(page.locator(".toast-wrap")).not_to_be_visible()
