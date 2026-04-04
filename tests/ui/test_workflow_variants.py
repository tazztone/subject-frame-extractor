import pytest
from playwright.sync_api import Page, expect

from .conftest import BASE_URL, open_accordion, switch_to_tab, wait_for_app_ready
from .ui_locators import Labels, Selectors

# Mark as e2e test
pytestmark = pytest.mark.e2e


class TestWorkflowVariants:
    """
    E2E tests for different input variants (Image Folders vs Videos).
    """

    def test_image_folder_workflow_skips_propagation(self, page: Page, app_server):
        """Verify that selecting a folder of images skips propagation step."""
        page.goto(BASE_URL)
        wait_for_app_ready(page)

        # 1. Select 'All Frames' (Proxy for testing workflow variant)
        # 1. Select 'All Frames' (Proxy for testing workflow variant)
        page.locator(Selectors.EXTRACTION_METHOD).click()
        # Gradio 5 Dropdown selection fix
        page.get_by_role("listitem").filter(has_text="All Frames (Maximum Quality)").click(force=True)

        # 2. Fill a dummy path
        page.locator(Selectors.SOURCE_INPUT).fill("/home/user/pics")

        # 3. Start Extraction
        page.locator(Selectors.START_EXTRACTION).click()
        expect(page.locator(Selectors.UNIFIED_STATUS)).to_contain_text(
            Selectors.STATUS_SUCCESS_EXTRACTION, timeout=15000
        )

        # 4. Confirm Subject (Pre-Analysis)
        switch_to_tab(page, Labels.TAB_SUBJECT)
        page.locator(Selectors.START_PRE_ANALYSIS).click()
        expect(page.locator(Selectors.UNIFIED_STATUS)).to_contain_text(
            Selectors.STATUS_SUCCESS_PRE_ANALYSIS, timeout=15000
        )

        # 5. Verify Propagation button IS HIDDEN (Images don't need propagation)
        prop_button = page.locator(Selectors.PROPAGATE_MASKS)
        expect(prop_button).not_to_be_visible()

        # 6. Verify "Proceed to Metrics" hint in logs
        open_accordion(page, Labels.SYSTEM_LOGS)
        expect(page.locator(Selectors.LOG_TEXTAREA)).to_contain_text("Propagation is not needed for image folders")

    def test_video_workflow_shows_propagation(self, page: Page, app_server):
        """Verify that selecting a video shows propagation step."""
        page.goto(BASE_URL)
        wait_for_app_ready(page)

        # 1. Select 'Scene-based' (Default)
        # 1. Select 'Scene-based' (Default)
        page.locator(Selectors.EXTRACTION_METHOD).click()
        # Gradio 5 Dropdown selection fix
        page.get_by_role("listitem").filter(has_text="Scene-based").click(force=True)

        # 2. Fill a dummy video path
        page.locator(Selectors.SOURCE_INPUT).fill("mock_video.mp4")

        # 3. Start Extraction
        page.locator(Selectors.START_EXTRACTION).click()
        expect(page.locator(Selectors.UNIFIED_STATUS)).to_contain_text(
            Selectors.STATUS_SUCCESS_EXTRACTION, timeout=15000
        )

        # 4. Confirm Subject
        switch_to_tab(page, Labels.TAB_SUBJECT)
        page.locator(Selectors.START_PRE_ANALYSIS).click()
        expect(page.locator(Selectors.UNIFIED_STATUS)).to_contain_text(
            Selectors.STATUS_SUCCESS_PRE_ANALYSIS, timeout=15000
        )

        # 5. Verify Propagation button IS VISIBLE
        prop_button = page.locator(Selectors.PROPAGATE_MASKS)
        expect(prop_button).to_be_visible()
