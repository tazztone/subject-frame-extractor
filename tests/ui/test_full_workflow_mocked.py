import pytest
from playwright.sync_api import Page, expect

# This test requires the app to be running (mock or real).
# The conftest.py in tests/ui/conftest.py handles starting the mock app server.
from .conftest import BASE_URL
from .ui_locators import Labels, Selectors


@pytest.fixture(scope="module")
def app_server_url(app_server):
    """
    Returns the URL of the running app server.
    app_server fixture from tests/ui/conftest.py starts the server if needed.
    """
    return BASE_URL


class TestFullWorkflowMocked:
    """
    Comprehensive E2E test simulating a full user journey using Playwright
    against the mock application (tests/mock_app.py).

    The mock app simulates backend processing without needing heavy models/GPU.
    """

    def switch_to_tab(self, page: Page, tab_name: str):
        """Robustly switch tabs in Gradio."""
        tab_btn = page.get_by_role("tab", name=tab_name)
        expect(tab_btn).to_be_visible()
        tab_btn.click(force=True)

        # Wait for the tab to be selected
        expect(tab_btn).to_have_attribute("aria-selected", "true")
        page.wait_for_timeout(1000)

    def test_full_user_journey(self, page: Page, app_server_url):
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
        page.goto(app_server_url)

        # --- 1. Source Tab ---
        # Wait for page load
        expect(page.get_by_text("Frame Extractor & Analyzer")).to_be_visible(timeout=30000)

        # Enter video path
        source_input = page.get_by_placeholder(Labels.SOURCE_PLACEHOLDER)
        if not source_input.is_visible():
            source_input = page.locator(Selectors.SOURCE_INPUT)

        source_input.fill("test.mp4")

        # Click Extract Frames
        extract_btn = page.locator(Selectors.START_EXTRACTION)
        expect(extract_btn).to_be_visible()
        extract_btn.click()

        # Wait for extraction to complete
        unified_status = page.locator(Selectors.UNIFIED_STATUS)
        # We wait for "Extraction Complete" to appear in the status area.
        # Gradio 5 might wrap this in a container with a timer.
        expect(unified_status).to_contain_text("Extraction Complete", timeout=30000)

        # --- 2. Subject Tab ---
        self.switch_to_tab(page, Labels.TAB_SUBJECT)

        # Click "Pre-Analyze Scenes"
        pre_analyze_btn = page.locator(Selectors.START_PRE_ANALYSIS)
        expect(pre_analyze_btn).to_be_visible(timeout=10000)
        pre_analyze_btn.click()

        expect(unified_status).to_contain_text("Pre-Analysis Complete", timeout=30000)

        # --- 3. Scenes Tab ---
        self.switch_to_tab(page, Labels.TAB_SCENES)

        # Click "Propagate Masks to All Frames"
        propagate_btn = page.locator(Selectors.PROPAGATE_MASKS)
        expect(propagate_btn).to_be_visible(timeout=10000)
        propagate_btn.click()

        expect(unified_status).to_contain_text("Propagation Complete", timeout=30000)

        # --- 4. Metrics Tab ---
        self.switch_to_tab(page, Labels.TAB_METRICS)

        # Click "Run Analysis"
        run_analysis_btn = page.locator(Selectors.START_ANALYSIS)
        expect(run_analysis_btn).to_be_visible(timeout=10000)
        run_analysis_btn.click()

        expect(unified_status).to_contain_text("Analysis Complete", timeout=30000)

        # --- 5. Export Tab ---
        self.switch_to_tab(page, Labels.TAB_EXPORT)

        # Click "Export Frames"
        export_btn = page.locator(Selectors.EXPORT_BUTTON)
        expect(export_btn).to_be_visible(timeout=10000)
        export_btn.click()

        expect(unified_status).to_contain_text("Export Complete", timeout=30000)
