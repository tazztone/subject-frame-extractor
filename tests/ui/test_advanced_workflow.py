import pytest
from playwright.sync_api import Page, expect

from .conftest import BASE_URL, open_accordion, switch_to_tab, wait_for_app_ready
from .ui_locators import Labels, Selectors

# Mark as e2e test
pytestmark = pytest.mark.e2e


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
        expect(pre_analyze_btn).to_be_visible()
        pre_analyze_btn.click()

        # 3. Verify Error in Status
        expect(page.locator(Selectors.UNIFIED_STATUS)).to_contain_text(Selectors.STATUS_ERROR_REGEX, timeout=10000)

        # 4. Open logs to confirm detail
        open_accordion(page, Labels.SYSTEM_LOGS)
        expect(page.locator(Selectors.LOG_TEXTAREA)).to_contain_text("Error", timeout=10000)

    def test_extraction_settings_persistence(self, page: Page, app_server):
        """
        Verify that changing extraction settings is reflected in the UI and extraction still starts.
        """
        page.goto(BASE_URL)
        wait_for_app_ready(page)

        # 1. Open Advanced Settings
        open_accordion(page, "Advanced Processing Settings")

        # 2. Change Megapixels
        mp_input = page.locator(Selectors.THUMB_MEGAPIXELS)
        expect(mp_input).to_be_visible()
        mp_input.fill("1.0")

        # 3. Fill Source
        page.locator(Selectors.SOURCE_INPUT).fill("advanced_settings_test.mp4")

        # 4. Start Extraction
        page.locator(Selectors.START_EXTRACTION).click()

        # 5. Verify completion
        expect(page.locator(Selectors.UNIFIED_STATUS)).to_contain_text(
            Selectors.STATUS_SUCCESS_EXTRACTION, timeout=30000
        )

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
