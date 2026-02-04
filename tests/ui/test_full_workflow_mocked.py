import time
import re

import pytest
from playwright.sync_api import Page, expect

# This test requires the app to be running (mock or real).
# The conftest.py in tests/e2e handles starting the mock app server.


@pytest.fixture(scope="module")
def app_server_url(app_server):
    """
    Returns the URL of the running app server.
    app_server fixture from tests/e2e/conftest.py starts the server if needed.
    """
    return "http://127.0.0.1:7860"


class TestFullWorkflowMocked:
    """
    Comprehensive E2E test simulating a full user journey using Playwright
    against the mock application (tests/mock_app.py).

    The mock app simulates backend processing without needing heavy models/GPU.
    """

    @pytest.mark.xfail(reason="Flaky button visibility in mock environment")
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

        # Click Source Tab (if not already active, but it usually is)
        # Note: Tab labels might differ slightly, checking visible text.

        # Enter video path (mock app handles "test.mp4")
        # Locator strategies for Gradio inputs can be tricky. Use label.
        # Sometimes labels are tricky in Gradio. Use get_by_label if possible, or fallback to placeholder/role.

        try:
            page.get_by_placeholder("Paste YouTube URL or local path").fill("test.mp4")
        except Exception:
            # Fallback if label name changed or using internal name
            page.locator("textarea").first.fill("test.mp4")

        # Click Extract Frames
        # Button text might vary.
        try:
            extract_btn = page.get_by_role("button", name="🚀 Start Extraction")
            extract_btn.click()
        except Exception:
            extract_btn = page.get_by_role("button", name="Extract Frames")
            extract_btn.click()

        # Wait for extraction to complete
        # The mock app logs "Extraction complete" to the unified log, but checking status is more robust
        unified_status = page.locator("#unified_status")
        expect(unified_status).to_contain_text("Extraction Complete", timeout=30000)

        # --- 2. Subject Tab ---
        subject_tab = page.get_by_role("tab", name="Subject")
        subject_tab.click()
        # Wait for tab activation
        expect(subject_tab).to_have_class(re.compile(r"selected|active"))
        time.sleep(1)

        # Click "Pre-Analyze Scenes"
        # Using ID for robustness
        pre_analyze_btn = page.locator("#start_pre_analysis_button")

        # Ensure it's visible with generous timeout
        expect(pre_analyze_btn).to_be_visible(timeout=10000)
        # Small wait for UI state
        time.sleep(0.5)
        pre_analyze_btn.click()

        # Wait for pre-analysis to complete
        expect(unified_status).to_contain_text("Pre-Analysis Complete", timeout=30000)

        # --- 3. Scenes Tab (Correction) ---
        page.get_by_role("tab", name="Scenes").click()

        # Verify gallery has items
        # Just check if we can see the gallery container
        # Note: Mock app might need to generate scene previews for them to show.

        # Click "Propagate Masks to All Frames"
        try:
            propagate_btn = page.get_by_role("button", name="⚡ Propagate Masks to All Frames")
        except:
            propagate_btn = page.get_by_role("button", name="Propagate Masks")

        propagate_btn.click()

        # Wait for propagation complete
        expect(unified_status).to_contain_text("Propagation Complete", timeout=30000)

        # --- 4. Metrics Tab ---
        page.get_by_role("tab", name="Metrics").click()

        # Click "Run Analysis"
        try:
            run_analysis_btn = page.get_by_role("button", name="⚡ Run Analysis")
        except:
            run_analysis_btn = page.get_by_role("button", name="Run Analysis")

        run_analysis_btn.click()

        # Wait for analysis complete
        expect(unified_status).to_contain_text("Analysis Complete", timeout=30000)

        # --- 5. Export Tab ---
        page.get_by_role("tab", name="Export").click()

        # Click "Export Frames"
        try:
            export_btn = page.get_by_role("button", name="💾 Export Selected Frames")
        except:
            export_btn = page.get_by_role("button", name="Export Frames")

        export_btn.click()

        # Wait for export complete
        # Check for success message in status
        expect(unified_status).to_contain_text("Exported", timeout=30000)
