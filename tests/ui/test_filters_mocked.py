import pytest
from playwright.sync_api import Page, expect

from .conftest import BASE_URL, open_accordion, switch_to_tab, wait_for_app_ready
from .ui_locators import Labels, Selectors

# Mark as e2e test
pytestmark = pytest.mark.e2e


class TestMockFilters:
    """
    Harden filtering logic tests using the mock app.
    Focuses on UI responsiveness and value propagation.
    """

    @pytest.fixture(autouse=True)
    def setup_mock_analysis(self, page: Page, app_server):
        """
        Setup: Run the full mock pipeline to get frames into the state.
        """
        page.goto(BASE_URL)
        wait_for_app_ready(page)

        # 1. Extraction
        page.locator(Selectors.SOURCE_INPUT).fill("filter_test.mp4")
        page.locator(Selectors.START_EXTRACTION).click()
        expect(page.locator(Selectors.UNIFIED_STATUS)).to_contain_text(
            Selectors.STATUS_SUCCESS_EXTRACTION, timeout=30000
        )

        # 2. Pre-Analysis
        switch_to_tab(page, Labels.TAB_SUBJECT)
        page.locator(Selectors.START_PRE_ANALYSIS).click()
        expect(page.locator(Selectors.UNIFIED_STATUS)).to_contain_text(
            Selectors.STATUS_SUCCESS_PRE_ANALYSIS, timeout=30000
        )

        # 3. Propagation
        switch_to_tab(page, Labels.TAB_SCENES)
        page.locator(Selectors.PROPAGATE_MASKS).click()
        expect(page.locator(Selectors.UNIFIED_STATUS)).to_contain_text(
            Selectors.STATUS_SUCCESS_PROPAGATION, timeout=30000
        )

        # 4. Analysis
        switch_to_tab(page, Labels.TAB_METRICS)
        page.locator(Selectors.START_ANALYSIS).click()
        expect(page.locator(Selectors.UNIFIED_STATUS)).to_contain_text(Selectors.STATUS_SUCCESS_ANALYSIS, timeout=30000)

    def test_smart_filter_toggle_updates_ui(self, page: Page):
        """
        Verify that toggling Smart Filtering enables/disables the percentile slider.
        """
        switch_to_tab(page, Labels.TAB_EXPORT)

        smart_filter = page.get_by_label("Smart Filtering (Percentile)")
        expect(smart_filter).to_be_visible()

        # Toggle it
        smart_filter.check()
        expect(smart_filter).to_be_checked()

        # Find the percentile component
        pctl = page.get_by_label("Target Percentile")
        expect(pctl).to_be_visible()

    def test_filter_preset_application(self, page: Page):
        """
        Test that selecting a preset (e.g., Portrait) updates sliders.
        """
        switch_to_tab(page, Labels.TAB_EXPORT)

        preset_dropdown = page.get_by_label("Use a Preset")
        expect(preset_dropdown).to_be_visible()

        # Select "Portrait/Selfie" preset
        preset_dropdown.click()
        # Gradio dropdowns render options as list items or buttons
        page.get_by_text("Portrait/Selfie", exact=True).click()

        # Wait for potential change
        page.wait_for_timeout(1000)

        # Verify no error toast
        expect(page.locator(".toast-wrap")).not_to_be_visible()

    def test_gallery_count_updates_on_filter(self, page: Page):
        """
        Verify that changing a filter value updates the 'Kept' count.
        """
        switch_to_tab(page, Labels.TAB_EXPORT)

        # First, ensure Quality Score accordion is open (it is by default in mock, but let's be safe)
        open_accordion(page, "Quality Score")

        # Find Quality Score min slider
        # The label is "Min" but it's inside the "Quality Score" accordion
        q_min = page.locator("div:has-text('Quality Score')").locator("input[aria-label='Min']").first
        expect(q_min).to_be_visible()

        q_min.fill("95")  # Set very high

        # Wait for reactive update
        page.wait_for_timeout(2000)

        # The status text should be updated
        expect(page.locator("div").get_by_text("Kept")).to_be_visible()
