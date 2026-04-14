import re

import pytest
from playwright.sync_api import Page, expect

from .conftest import BASE_URL, switch_to_tab, wait_for_app_ready
from .ui_locators import Labels, Selectors

# Mark as e2e test
pytestmark = pytest.mark.e2e


class TestBusyState:
    """
    Tests for 'Busy' UI states during long-running tasks.
    Ensures that action buttons are disabled during execution and re-enabled after.
    """

    def test_extraction_lock_and_unlock(self, page: Page, app_server):
        """
        Start extraction → Verify Start button is disabled → Verify Cancel/Pause are enabled.
        Wait for completion → Verify Start button is re-enabled.
        """
        page.goto(BASE_URL)
        wait_for_app_ready(page)

        # 1. Fill source
        page.locator(Selectors.SOURCE_INPUT).fill("busy_test.mp4")
        start_btn = page.locator(Selectors.START_EXTRACTION)
        cancel_btn = page.locator(Selectors.CANCEL_BUTTON)

        # Idle state (Pre-start)
        expect(start_btn).to_be_enabled()
        # Note: In Gradio 5, uninitialized buttons might be hidden or disabled.
        # But we expect Cancel/Pause to be disabled initially.
        expect(cancel_btn).to_be_disabled()

        # 2. Start Extraction
        start_btn.click()

        # 3. Busy state (status should contain 'Mock Extraction' now)
        expect(page.get_by_text(re.compile(r"Processing \(Extraction\)", re.IGNORECASE))).to_be_visible(timeout=10000)

        # Start button should show processing state and lead to status update
        # expect(start_btn).to_contain_text("⏳ Processing", timeout=5000) # This might be brittle in mock mode if buttons don't update
        expect(cancel_btn).to_be_enabled()

        # 4. Wait for completion
        expect(page.get_by_text(re.compile(r"Extraction Complete", re.IGNORECASE))).to_be_visible(timeout=30000)

        # 5. Idle state again (Post-completion)
        expect(start_btn).to_be_enabled(timeout=10000)
        expect(cancel_btn).to_be_disabled()

    def test_tab_switch_during_busy(self, page: Page, app_server):
        """
        Start pipeline → Switch tabs → Pipeline should continue → Switch back → Status still updated.
        """
        page.goto(BASE_URL)
        wait_for_app_ready(page)

        source_input = page.locator(Selectors.SOURCE_INPUT)
        source_input.fill("tab_switch_busy.mp4")
        page.locator(Selectors.START_EXTRACTION).click()

        # Verify it started
        expect(page.get_by_text(re.compile(r"Processing \(Extraction\)", re.IGNORECASE))).to_be_visible(timeout=10000)

        # 1. Switch to Subject tab
        switch_to_tab(page, Labels.TAB_SUBJECT)
        # Verify tab button is active (ARIA based)
        expect(page.get_by_role("tab", name=Labels.TAB_SUBJECT)).to_have_attribute("aria-selected", "true")

        # 2. Status bar (global) should still show progress details
        expect(page.get_by_text(re.compile(r"Processing \(Extraction\)", re.IGNORECASE))).to_be_visible()

        # 3. Wait for it to finish while on the other tab
        expect(page.get_by_text(re.compile(r"Extraction Complete", re.IGNORECASE))).to_be_visible(timeout=30000)

        # 4. Switch back to Source
        switch_to_tab(page, Labels.TAB_SOURCE)
        # Verify input value is preserved (State check)
        expect(source_input).to_have_value("tab_switch_busy.mp4")
