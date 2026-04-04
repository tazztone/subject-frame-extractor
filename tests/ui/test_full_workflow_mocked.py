import pytest
from playwright.sync_api import Page, expect

from .conftest import BASE_URL, switch_to_tab, wait_for_app_ready
from .ui_locators import Labels, Selectors

# Mark as e2e test
pytestmark = pytest.mark.e2e


class TestFullWorkflowMocked:
    """
    Comprehensive E2E test simulating a full user journey using Playwright
    against the mock application (tests/mock_app.py).

    The mock app simulates backend processing without needing heavy models/GPU.
    """

    def test_full_user_journey(self, page: Page, app_server):
        """
        Simulates:
        1. Select Video Source (Extraction)
        2. Run Extraction
        3. Define Subject (Pre-analysis)
        4. Select a person/scene
        5. Run Propagation
        6. Filter Results
        7. Export
        """
        page.goto(BASE_URL)
        wait_for_app_ready(page)

        # 1. Source Tab
        # Enter video path
        source_input = page.locator(Selectors.SOURCE_INPUT)
        source_input.fill("test_journey.mp4")

        # Click Extract Frames
        extract_btn = page.locator(Selectors.START_EXTRACTION)
        expect(extract_btn).to_be_visible()
        extract_btn.click()

        # Wait for extraction to complete
        unified_status = page.locator(Selectors.UNIFIED_STATUS)
        expect(unified_status).to_contain_text(Selectors.STATUS_SUCCESS_EXTRACTION, timeout=30000)

        # 2. Subject Tab
        switch_to_tab(page, Labels.TAB_SUBJECT)

        # Click "Confirm Subject"
        pre_analyze_btn = page.locator(Selectors.START_PRE_ANALYSIS)
        expect(pre_analyze_btn).to_be_visible(timeout=10000)
        pre_analyze_btn.click()

        expect(unified_status).to_contain_text(Selectors.STATUS_SUCCESS_PRE_ANALYSIS, timeout=30000)

        # 3. Scenes Tab
        switch_to_tab(page, Labels.TAB_SCENES)

        # Click "Propagate Masks"
        propagate_btn = page.locator(Selectors.PROPAGATE_MASKS)
        expect(propagate_btn).to_be_visible(timeout=10000)
        propagate_btn.click()

        expect(unified_status).to_contain_text(Selectors.STATUS_SUCCESS_PROPAGATION, timeout=30000)

        # 4. Metrics Tab
        switch_to_tab(page, Labels.TAB_METRICS)

        # Click "Start Analysis"
        run_analysis_btn = page.locator(Selectors.START_ANALYSIS)
        expect(run_analysis_btn).to_be_visible(timeout=10000)
        run_analysis_btn.click()

        expect(unified_status).to_contain_text(Selectors.STATUS_SUCCESS_ANALYSIS, timeout=30000)

        # 5. Export Tab
        switch_to_tab(page, Labels.TAB_EXPORT)

        # Click "Start Export"
        export_btn = page.locator(Selectors.EXPORT_BUTTON)
        expect(export_btn).to_be_visible(timeout=10000)
        export_btn.click()

        expect(unified_status).to_contain_text(Selectors.STATUS_SUCCESS_EXPORT, timeout=30000)
