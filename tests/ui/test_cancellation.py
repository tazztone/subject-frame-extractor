import pytest
from playwright.sync_api import Page, expect

from .conftest import BASE_URL, switch_to_tab, wait_for_app_ready
from .ui_locators import Selectors, Labels

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
        page.goto(BASE_URL)
        wait_for_app_ready(page)

        # 1. Fill source and start extraction
        page.locator(Selectors.SOURCE_INPUT).fill("cancel_test.mp4")
        page.locator(Selectors.START_EXTRACTION).click()

        # 2. Wait for it to definitely be running
        # mock_app.py uses 'Mock Extraction' in tracker desc
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
        expect(page.locator(Selectors.UNIFIED_STATUS)).to_contain_text(Selectors.STATUS_SUCCESS_EXTRACTION, timeout=30000)

    def test_cancel_propagation_midway(self, page: Page, app_server):
        """
        Start propagation → Click Cancel → Verify Success of subsequent retry.
        """
        page.goto(BASE_URL)
        wait_for_app_ready(page)

        # Setup: Quick extraction
        page.locator(Selectors.SOURCE_INPUT).fill("prop_cancel.mp4")
        page.locator(Selectors.START_EXTRACTION).click()
        expect(page.locator(Selectors.UNIFIED_STATUS)).to_contain_text(Selectors.STATUS_SUCCESS_EXTRACTION, timeout=30000)

        # Setup: Quick pre-analysis
        switch_to_tab(page, Labels.TAB_SUBJECT)
        page.locator(Selectors.START_PRE_ANALYSIS).click()
        expect(page.locator(Selectors.UNIFIED_STATUS)).to_contain_text(Selectors.STATUS_SUCCESS_PRE_ANALYSIS, timeout=30000)

        # 1. Start Propagation
        switch_to_tab(page, Labels.TAB_SCENES)
        page.locator(Selectors.PROPAGATE_MASKS).click()

        # 2. Click Cancel quickly
        page.locator(Selectors.CANCEL_BUTTON).click()

        # 3. Verify status reflects cancellation or immediate recovery
        expect(page.locator(Selectors.UNIFIED_STATUS)).to_contain_text("Cancelled", timeout=10000)
