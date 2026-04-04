import pytest
from playwright.sync_api import Page, expect

from .conftest import BASE_URL, switch_to_tab, wait_for_app_ready, open_accordion
from .ui_locators import Selectors, Labels

# Mark as e2e test
pytestmark = pytest.mark.e2e


class TestErrorRecovery:
    """
    Tests for error paths and user recovery after pipeline/validation failures.
    Ensures the user can retry successfully without state corruption.
    """

    def test_extraction_with_invalid_path_to_success(self, page: Page, app_server):
        """
        Fill a nonsense path → Extract → Verify Error → Retry with valid path.
        """
        page.goto(BASE_URL)
        wait_for_app_ready(page)

        # 1. Fill invalid source
        page.locator(Selectors.SOURCE_INPUT).fill("nonsense_invalid_path.xyz")
        page.locator(Selectors.START_EXTRACTION).click()

        # 2. Verify Error in status
        expect(page.locator(Selectors.UNIFIED_STATUS)).to_contain_text(Selectors.STATUS_ERROR_REGEX, timeout=15000)

        # 3. Correct the path
        page.locator(Selectors.SOURCE_INPUT).fill("valid_retry_test.mp4")
        page.locator(Selectors.START_EXTRACTION).click()

        # 4. Verify Success
        expect(page.locator(Selectors.UNIFIED_STATUS)).to_contain_text(Selectors.STATUS_SUCCESS_EXTRACTION, timeout=30000)

    def test_tab_jump_restriction_recovery(self, page: Page, app_server):
        """
        Skip to Subject tab, click Confirm → Verify error message → Go back and extract → Ensure user is unblocked.
        """
        page.goto(BASE_URL)
        wait_for_app_ready(page)

        # 1. Skip to Subject
        switch_to_tab(page, Labels.TAB_SUBJECT)
        page.locator(Selectors.START_PRE_ANALYSIS).click()

        # 2. Verify error detail in status
        expect(page.locator(Selectors.UNIFIED_STATUS)).to_contain_text(Selectors.STATUS_ERROR_REGEX, timeout=10000)

        # 3. Go back and fix source
        switch_to_tab(page, Labels.TAB_SOURCE)
        page.locator(Selectors.SOURCE_INPUT).fill("navigation_recovery.mp4")
        page.locator(Selectors.START_EXTRACTION).click()
        expect(page.locator(Selectors.UNIFIED_STATUS)).to_contain_text(Selectors.STATUS_SUCCESS_EXTRACTION, timeout=30000)

        # 4. Now Subject tab action should succeed
        switch_to_tab(page, Labels.TAB_SUBJECT)
        page.locator(Selectors.START_PRE_ANALYSIS).click()
        expect(page.locator(Selectors.UNIFIED_STATUS)).to_contain_text(Selectors.STATUS_SUCCESS_PRE_ANALYSIS, timeout=30000)

    def test_export_precondition_failure(self, page: Page, app_server):
        """
        Go to Export tab, click Export with no data loaded → verify graceful error/warning.
        """
        page.goto(BASE_URL)
        wait_for_app_ready(page)

        # 1. Skip to Export
        switch_to_tab(page, Labels.TAB_EXPORT)

        # 2. Click Export
        page.locator(Selectors.EXPORT_BUTTON).click()

        # 3. Verify status reflects failure
        expect(page.locator(Selectors.UNIFIED_STATUS)).to_contain_text(Selectors.STATUS_ERROR_REGEX, timeout=15000)

        # 4. Verify logs
        open_accordion(page, Labels.SYSTEM_LOGS)
        expect(page.locator(Selectors.LOG_TEXTAREA)).to_contain_text("Error", timeout=10000)
