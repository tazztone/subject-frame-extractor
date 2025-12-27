import time

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
            page.get_by_label("Video URL or Local Path").fill("test.mp4")
        except Exception:
            # Fallback if label name changed
            page.get_by_label("Video Path").fill("test.mp4")

        # Click Extract Frames
        # Button text might vary.
        try:
            extract_btn = page.get_by_role("button", name="🚀 Start Single Extraction")
            extract_btn.click()
        except Exception:
            extract_btn = page.get_by_role("button", name="Extract Frames")
            extract_btn.click()

        # Wait for extraction to complete
        # The mock app logs "Extraction complete" to the unified log
        unified_log = page.locator("#unified_log")
        expect(unified_log).to_contain_text("Extraction complete", timeout=30000)

        # --- 2. Subject Tab ---
        page.get_by_role("tab", name="Subject").click()

        # Click "Pre-Analyze Scenes"
        try:
            pre_analyze_btn = page.get_by_role("button", name="🌱 Find & Preview Best Frames")
        except:
            pre_analyze_btn = page.get_by_role("button", name="Pre-Analyze Scenes")

        # Ensure it's visible
        expect(pre_analyze_btn).to_be_visible()
        # Small wait for UI state
        time.sleep(1)
        pre_analyze_btn.click()

        # Wait for pre-analysis to complete
        expect(unified_log).to_contain_text("Pre-analysis complete", timeout=30000)

        # --- 3. Scenes Tab (Correction) ---
        page.get_by_role("tab", name="Scenes").click()

        # Verify gallery has items
        # Just check if we can see the gallery container
        # Note: Mock app might need to generate scene previews for them to show.

        # Click "Propagate Masks & Analyze"
        try:
            propagate_btn = page.get_by_role("button", name="⚡ Run Propagation & Analysis")
        except:
            propagate_btn = page.get_by_role("button", name="Propagate Masks & Analyze")

        propagate_btn.click()

        # Wait for analysis complete
        expect(unified_log).to_contain_text("Analysis complete", timeout=30000)

        # --- 4. Metrics Tab (Filtering) ---
        page.get_by_role("tab", name="Metrics").click()

        # Click "Apply Filters"
        apply_filters_btn = page.get_by_role("button", name="Apply Filters")
        apply_filters_btn.click()

        # --- 5. Export Tab ---
        page.get_by_role("tab", name="Export").click()

        # Click "Export Frames"
        try:
            export_btn = page.get_by_role("button", name="💾 Export Selected Frames")
        except:
            export_btn = page.get_by_role("button", name="Export Frames")

        export_btn.click()

        # Wait for export complete
        expect(unified_log).to_contain_text("Export complete", timeout=30000)
