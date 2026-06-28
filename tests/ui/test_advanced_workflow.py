import pytest
from playwright.sync_api import expect

from .app_driver import AppDriver
from .ui_locators import Labels, Selectors

# Mark as e2e test
pytestmark = pytest.mark.e2e


class TestAdvancedWorkflow:
    """
    Advanced E2E tests covering navigation restrictions and UI responsiveness.
    """

    def test_navigation_restrictions_error(self, app_driver: AppDriver):
        """
        Verify that attempting to move to Subject/Analysis stage without extraction
        results in a visible error.
        """
        # 1-2. Switch to Subject tab, click "Confirm Subject"
        app_driver.navigate(Labels.TAB_SUBJECT).pre_analyze()

        # 3. Verify Error in Status
        app_driver.expect_status(Selectors.STATUS_ERROR_REGEX, timeout=10000)

        # 4. Confirm detail in logs. ``expect_log`` polls (tolerates the 0.5s
        #    gr.Timer refresh) and auto-opens the logs accordion.
        app_driver.expect_log("Error", timeout=12000)

    def test_extraction_settings_persistence(self, app_driver: AppDriver):
        """
        Verify that changing extraction settings is reflected in the UI and
        extraction still starts.
        """
        # 1. Open Advanced Settings
        app_driver.open_accordion("Advanced Processing Settings")

        # 2. Change Megapixels (numeric input selector for Gradio strict mode)
        mp_input = app_driver.page.locator(f"{Selectors.THUMB_MEGAPIXELS} input[data-testid='number-input']")
        expect(mp_input).to_be_visible()
        mp_input.fill("1.0")

        # 3-4. Fill Source and start extraction
        app_driver.extract("advanced_settings_test.mp4")

        # 5. Verify completion
        app_driver.expect_status(Selectors.STATUS_SUCCESS_EXTRACTION, timeout=30000)

    def test_filtering_ui_responsiveness(self, app_driver: AppDriver):
        """Test the Metrics/Filtering tab UI controls responsiveness."""
        app_driver.navigate(Labels.TAB_EXPORT)

        smart_filter = app_driver.page.get_by_label("Smart Filtering")
        expect(smart_filter).to_be_visible()

        smart_filter.check()
        expect(smart_filter).to_be_checked()

        # Verify that clicking it doesn't cause a global error box
        app_driver.expect_no_error_toast()
