import pytest
from playwright.sync_api import Page, expect

from .app_driver import AppDriver
from .ui_locators import Labels, Selectors

# Mark as e2e test
pytestmark = pytest.mark.e2e


class TestExportFlow:
    """
    Comprehensive tests for the Export workflow.
    Uses the shared_analysis_session fixture to skip extraction/analysis setup.
    """

    def test_export_tab_elements_visibility(self, shared_analysis_session: Page):
        """Verify export tab is accessible and shows expected elements after analysis."""
        driver = AppDriver(shared_analysis_session)
        driver.navigate(Labels.TAB_EXPORT)

        driver.expect_visible(Selectors.FILTER_PRESET)
        driver.expect_visible(Selectors.EXPORT_BUTTON)
        driver.expect_visible(Selectors.DRY_RUN_BUTTON)

    def test_dry_run_summary_verification(self, shared_analysis_session: Page):
        """Test dry run export mode and verify the summary output in logs."""
        driver = AppDriver(shared_analysis_session)
        driver.navigate(Labels.TAB_EXPORT).dry_run()

        # Verify Success message in status (mock_app returns "🔍 Dry Run: 10 / 10 frames...")
        driver.expect_status("Dry Run", timeout=10000)
        # Verify detail in logs
        driver.expect_log("10 / 10 frames", timeout=5000)

    def test_filter_preset_interaction(self, shared_analysis_session: Page):
        """Verify that selecting a filter preset updates the UI state."""
        driver = AppDriver(shared_analysis_session)
        page = driver.navigate(Labels.TAB_EXPORT).page

        # Select "Portrait/Selfie" preset
        driver.select_preset("Portrait/Selfie")
        driver.expect_no_error_toast()

        # Verify a 'Kept' update surfaces (markdown/text in current mock UI)
        expect(page.locator(Selectors.SCENE_GALLERY_VIEW_TOGGLE).get_by_text("Kept")).to_be_visible(timeout=5000)

    def test_export_completion(self, shared_analysis_session: Page):
        """Verify that clicking Export completes and shows the success message."""
        driver = AppDriver(shared_analysis_session)
        driver.navigate(Labels.TAB_EXPORT).export()

        driver.expect_status(Selectors.STATUS_SUCCESS_EXPORT, timeout=15000)
