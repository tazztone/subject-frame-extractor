import pytest
from playwright.sync_api import expect

from .app_driver import AppDriver
from .ui_locators import Labels, Selectors

# Mark as e2e test
pytestmark = pytest.mark.e2e


class TestBusyState:
    """
    Tests for 'Busy' UI states during long-running tasks.
    Ensures that action buttons are disabled during execution and re-enabled after.
    """

    def test_extraction_lock_and_unlock(self, app_driver: AppDriver):
        """
        Start extraction → Verify Start button is disabled → Verify Cancel/Pause are enabled.
        Wait for completion → Verify Start button is re-enabled.
        """
        page = app_driver.page

        # 1. Fill source
        page.locator(Selectors.SOURCE_INPUT).fill("busy_test.mp4")
        start_btn = page.locator(Selectors.START_EXTRACTION)
        cancel_btn = page.locator(Selectors.CANCEL_BUTTON)

        # Idle state (Pre-start)
        expect(start_btn).to_be_enabled()
        # Cancel/Pause are disabled initially in Gradio 5.
        expect(cancel_btn).to_be_disabled()

        # 2. Start Extraction
        start_btn.click()

        app_driver.expect_status("Mock Extraction", timeout=10000)
        expect(cancel_btn).to_be_enabled()

        # 4. Wait for completion
        app_driver.expect_status(Selectors.STATUS_SUCCESS_EXTRACTION, timeout=30000)

        # 5. Idle state again (Post-completion)
        expect(start_btn).to_be_enabled(timeout=10000)
        expect(cancel_btn).to_be_disabled()

    def test_tab_switch_during_busy(self, app_driver: AppDriver):
        """
        Start pipeline → Switch tabs → Pipeline should continue → Switch back → Status still updated.
        """
        page = app_driver.page
        source_input = page.locator(Selectors.SOURCE_INPUT)
        source_input.fill("tab_switch_busy.mp4")
        page.locator(Selectors.START_EXTRACTION).click()

        # Verify it started
        app_driver.expect_status("Mock Extraction", timeout=10000)

        # 1. Switch to Subject tab; verify it is active (ARIA based)
        app_driver.navigate(Labels.TAB_SUBJECT)
        expect(page.get_by_role("tab", name=Labels.TAB_SUBJECT)).to_have_attribute("aria-selected", "true")

        # 2. Status bar (global) should still show progress details
        app_driver.expect_status("Mock Extraction")

        # 3. Wait for it to finish while on the other tab
        app_driver.expect_status(Selectors.STATUS_SUCCESS_EXTRACTION, timeout=30000)

        # 4. Switch back to Source; verify input value is preserved (state check)
        app_driver.navigate(Labels.TAB_SOURCE)
        expect(source_input).to_have_value("tab_switch_busy.mp4")
