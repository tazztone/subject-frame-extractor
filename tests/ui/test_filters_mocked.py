import pytest
from playwright.sync_api import expect

from .app_driver import AppDriver
from .ui_locators import Labels, Selectors

# Mark as e2e test
pytestmark = pytest.mark.e2e


class TestMockFilters:
    """
    Harden filtering logic tests using the mock app.
    Focuses on UI responsiveness and value propagation.
    """

    @pytest.fixture(autouse=True)
    def setup_mock_analysis(self, app_driver: AppDriver):
        """
        Setup: Run the full mock pipeline to get frames into the state.
        """
        # 1. Extraction
        app_driver.extract("filter_test.mp4").expect_status(Selectors.STATUS_SUCCESS_EXTRACTION, timeout=30000)
        # 2. Pre-Analysis
        app_driver.navigate(Labels.TAB_SUBJECT).pre_analyze().expect_status(
            Selectors.STATUS_SUCCESS_PRE_ANALYSIS, timeout=30000
        )
        # 3. Propagation
        app_driver.navigate(Labels.TAB_SCENES).propagate().expect_status(
            Selectors.STATUS_SUCCESS_PROPAGATION, timeout=30000
        )
        # 4. Analysis
        app_driver.navigate(Labels.TAB_METRICS).analyze().expect_status(
            Selectors.STATUS_SUCCESS_ANALYSIS, timeout=30000
        )

    def test_smart_filter_toggle_updates_ui(self, app_driver: AppDriver):
        """
        Verify that toggling Smart Filtering enables/disables the percentile slider.
        """
        app_driver.navigate(Labels.TAB_EXPORT)

        smart_filter = app_driver.page.get_by_label("Smart Filtering (Percentile)")
        expect(smart_filter).to_be_visible()

        smart_filter.check()
        expect(smart_filter).to_be_checked()

        pctl = app_driver.page.get_by_label("Target Percentile").first
        expect(pctl).to_be_visible()

    def test_filter_preset_application(self, app_driver: AppDriver):
        """
        Test that selecting a preset (e.g., Portrait) updates sliders.
        """
        page = app_driver.navigate(Labels.TAB_EXPORT).page

        preset_dropdown = page.get_by_label("Use a Preset")
        expect(preset_dropdown).to_be_visible()

        preset_dropdown.click()
        page.get_by_role("option", name="Portrait/Selfie").click()

        page.wait_for_timeout(1000)
        expect(page.locator(".toast-wrap")).not_to_be_visible()

    def test_gallery_count_updates_on_filter(self, app_driver: AppDriver):
        """
        Verify that changing a filter value updates the 'Kept' count.
        """
        app_driver.navigate(Labels.TAB_EXPORT)

        # Ensure Quality Score accordion is open
        app_driver.open_accordion("Quality Score")

        page = app_driver.page

        q_min = page.locator("div:has-text('Quality Score')").locator("input[aria-label='Min']").first
        expect(q_min).to_be_visible()
        q_min.fill("95")  # Set very high

        # Wait for reactive update
        page.wait_for_timeout(2000)
        expect(page.locator("div").get_by_text("Kept")).to_be_visible()
