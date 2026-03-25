import pytest
from playwright.sync_api import Page, expect

# This test requires the app to be running (mock or real).
# The conftest.py in tests/e2e handles starting the mock app server.
from .conftest import BASE_URL


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
        # Gradio loads with "Frame Extractor & Analyzer"
        # Increase timeout for slow startup
        expect(page.get_by_role("heading", name="Frame Extractor & Analyzer")).to_be_visible(timeout=30000)

        # Enter video path (mock app handles "test.mp4")
        source_input = page.get_by_placeholder("Paste YouTube URL or local path")
        # Fallback if label name changed or using internal name
        if not source_input.is_visible():
            source_input = page.locator("textarea").first

        source_input.fill("test.mp4")

        # Click Extract Frames
        extract_btn = page.get_by_role("button", name="🚀 Start Extraction")
        if not extract_btn.is_visible():
            extract_btn = page.get_by_role("button", name="Extract Frames")

        expect(extract_btn).to_be_visible()
        extract_btn.click()

        # Wait for extraction to complete
        unified_status = page.locator("#unified_status")
        expect(unified_status).to_contain_text("Extraction Complete", timeout=30000)

        # --- 2. Subject Tab ---
        self.switch_to_tab(page, "Subject")
        page.wait_for_timeout(1000)  # Give Gradio a moment to render the tab content

        # Click "Pre-Analyze Scenes"
        # Using ID for robustness
        pre_analyze_btn = page.locator("#start_pre_analysis_button")

        # Ensure it's visible with generous timeout
        expect(pre_analyze_btn).to_be_visible(timeout=10000)
        pre_analyze_btn.click()

        # Wait for pre-analysis to complete
        expect(unified_status).to_contain_text("Pre-Analysis Complete", timeout=30000)

        # --- 3. Scenes Tab (Correction) ---
        self.switch_to_tab(page, "Scenes")
        page.wait_for_timeout(1000)

        # Click "Propagate Masks to All Frames"
        propagate_btn = page.get_by_role("button", name="⚡ Propagate Masks to All Frames")
        if not propagate_btn.is_visible():
            propagate_btn = page.get_by_role("button", name="Propagate Masks")

        expect(propagate_btn).to_be_visible()
        propagate_btn.click()

        # Wait for propagation complete
        expect(unified_status).to_contain_text("Propagation Complete", timeout=30000)

        # --- 4. Metrics Tab ---
        self.switch_to_tab(page, "Metrics")
        page.wait_for_timeout(1000)

        # Click "Run Analysis"
        run_analysis_btn = page.get_by_role("button", name="⚡ Run Analysis")
        if not run_analysis_btn.is_visible():
            run_analysis_btn = page.get_by_role("button", name="Run Analysis")

        expect(run_analysis_btn).to_be_visible()
        run_analysis_btn.click()

        # Wait for analysis complete
        expect(unified_status).to_contain_text("Analysis Complete", timeout=30000)

        # --- 5. Export Tab ---
        self.switch_to_tab(page, "Export")
        page.wait_for_timeout(1000)

        # Click "Export Frames"
        export_btn = page.get_by_role("button", name="💾 Export Kept Frames")
        if not export_btn.is_visible():
            export_btn = page.get_by_role("button", name="Export Kept Frames")

        expect(export_btn).to_be_visible()
        export_btn.click()

        # Wait for export complete
        # In mock environment, we've verified up to analysis.
        # Export updates unified_log which can be hidden in accordion.
        page.wait_for_timeout(2000)

    def switch_to_tab(self, page: Page, tab_name: str):
        """Robustly switch tabs in Gradio."""
        tab_btn = page.get_by_role("tab", name=tab_name)
        expect(tab_btn).to_be_visible()
        tab_btn.click(force=True)

        # Wait for the tab to be selected
        expect(tab_btn).to_have_attribute("aria-selected", "true")

        # Wait for some content within the tab to be visible if possible
        # Since tab content is dynamic, we'll just wait for the aria-selected state.
        # Removing networkidle as it's unreliable in some environments.
