import pytest
from playwright.sync_api import Page, expect

from .conftest import BASE_URL, switch_to_tab, wait_for_app_ready
from .ui_locators import Labels, Selectors

# Mark as e2e test
pytestmark = pytest.mark.e2e


class TestCancellation:
    """
    Tests for the Cancel button during pipeline execution.
    Ensures that pipelines are interruptible and the UI stays responsive.
    """

    def test_cancel_extraction_midway(self, page: Page, app_server):
        """
        Start extraction → Wait for progress → Click Cancel → Verify Cancellation status.
        """
        page.goto(BASE_URL, timeout=60000)
        wait_for_app_ready(page)

        # 1. Fill source and start extraction
        page.locator(Selectors.SOURCE_INPUT).fill("cancel_test.mp4")
        page.locator(Selectors.START_EXTRACTION).click()

        # 2. In Gradio 5 with show_progress="hidden", button text doesn't change.
        # Instead, we wait for the UNIFIED_STATUS to reflect the mock action.
        expect(page.locator(Selectors.UNIFIED_STATUS)).to_contain_text("Mock Extraction", timeout=10000)

        # 3. Click Cancel
        cancel_btn = page.locator(Selectors.CANCEL_BUTTON)
        expect(cancel_btn).to_be_enabled(timeout=5000)
        cancel_btn.click()

        # 4. Verify status contains "Cancelled"
        # Since mock_app might return "Extraction Complete" if it happens too fast,
        # but normally cancellation sets state to "Cancelled" or similar.
        # Fixed mock_app handles cancel_event check.
        expect(page.locator(Selectors.UNIFIED_STATUS)).to_contain_text("Cancelled", timeout=10000)

        # 5. Reset and retry (Verify no stuck state)
        page.locator(Selectors.START_EXTRACTION).click()
        expect(page.locator(Selectors.UNIFIED_STATUS)).to_contain_text(
            Selectors.STATUS_SUCCESS_EXTRACTION, timeout=30000
        )

    def test_cancel_propagation_midway(self, page: Page, app_server):
        """
        Start propagation → Click Cancel → Verify Success of subsequent retry.
        """
        page.goto(BASE_URL)
        wait_for_app_ready(page)

        # Setup: Quick extraction
        page.locator(Selectors.SOURCE_INPUT).fill("prop_cancel.mp4")
        page.locator(Selectors.START_EXTRACTION).click()
        expect(page.locator(Selectors.UNIFIED_STATUS)).to_contain_text(
            Selectors.STATUS_SUCCESS_EXTRACTION, timeout=30000
        )

        # Setup: Quick pre-analysis
        switch_to_tab(page, Labels.TAB_SUBJECT)
        page.locator(Selectors.START_PRE_ANALYSIS).click()
        expect(page.locator(Selectors.UNIFIED_STATUS)).to_contain_text(
            Selectors.STATUS_SUCCESS_PRE_ANALYSIS, timeout=30000
        )

        # 1. Start Propagation
        switch_to_tab(page, Labels.TAB_SCENES)
        prop_btn = page.locator(Selectors.PROPAGATE_MASKS)
        prop_btn.click()
        
        # 2. Wait for busy state signal
        expect(prop_btn).to_contain_text("⏳ Processing", timeout=5000)

        # 3. Click Cancel quickly
        page.locator(Selectors.CANCEL_BUTTON).click()

        # 3. Verify status reflects cancellation or immediate recovery
        expect(page.locator(Selectors.UNIFIED_STATUS)).to_contain_text("Cancelled", timeout=10000)
