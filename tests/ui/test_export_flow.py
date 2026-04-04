import pytest
from playwright.sync_api import Page, expect

from .conftest import open_accordion, switch_to_tab
from .ui_locators import Selectors, Labels

# Mark as e2e test
pytestmark = pytest.mark.e2e


class TestExportFlow:
    """
    Comprehensive tests for the Export workflow.
    Uses the shared_analysis_session fixture to skip extraction/analysis setup.
    """

    def test_export_tab_elements_visibility(self, shared_analysis_session: Page):
        """Verify export tab is accessible and shows expected elements after analysis."""
        page = shared_analysis_session
        switch_to_tab(page, Labels.TAB_EXPORT)

        expect(page.locator(Selectors.FILTER_PRESET)).to_be_visible()
        expect(page.locator(Selectors.EXPORT_BUTTON)).to_be_visible()
        expect(page.locator(Selectors.DRY_RUN_BUTTON)).to_be_visible()

    def test_dry_run_summary_verification(self, shared_analysis_session: Page):
        """Test dry run export mode and verify the summary output in logs."""
        page = shared_analysis_session
        switch_to_tab(page, Labels.TAB_EXPORT)

        # 1. Click Dry Run
        dry_run_btn = page.locator(Selectors.DRY_RUN_BUTTON)
        expect(dry_run_btn).to_be_visible()
        dry_run_btn.click()

        # 2. Verify Success message in status
        # mock_app.py returns "🔍 Dry Run: 10 / 10 frames would be exported (MOCKED)."
        expect(page.locator(Selectors.UNIFIED_STATUS)).to_contain_text("Dry Run", timeout=10000)

        # 3. Verify detail in logs
        open_accordion(page, Labels.SYSTEM_LOGS)
        expect(page.locator(Selectors.LOG_TEXTAREA)).to_contain_text("10 / 10 frames", timeout=5000)

    def test_filter_preset_interaction(self, shared_analysis_session: Page):
        """Verify that selecting a filter preset updates the UI state."""
        page = shared_analysis_session
        switch_to_tab(page, Labels.TAB_EXPORT)

        # Select "Portrait/Selfie" preset
        preset_dropdown = page.locator(Selectors.FILTER_PRESET)
        preset_dropdown.click()
        # Gradio dropdowns in mock app have these keys
        page.get_by_text("Portrait/Selfie", exact=True).click()

        # Verify no error toast
        expect(page.locator(".toast-wrap")).not_to_be_visible()

        # Verify that clicking 'Apply' or just the preset change triggers a 'Kept' update
        # In current mock UI, it's a markdown or text
        expect(page.locator("div").get_by_text("Kept")).to_be_visible(timeout=5000)

    def test_export_completion(self, shared_analysis_session: Page):
        """Verify that clicking Export completes and shows the success message."""
        page = shared_analysis_session
        switch_to_tab(page, Labels.TAB_EXPORT)

        # 1. Click Export
        export_btn = page.locator(Selectors.EXPORT_BUTTON)
        expect(export_btn).to_be_visible()
        export_btn.click()

        # 2. Verify Success message
        expect(page.locator(Selectors.UNIFIED_STATUS)).to_contain_text(Selectors.STATUS_SUCCESS_EXPORT, timeout=15000)
