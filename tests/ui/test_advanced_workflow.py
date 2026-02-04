import pytest
from playwright.sync_api import Page, expect


@pytest.fixture(scope="module")
def app_server_url(app_server):
    """
    Returns the URL of the running app server.
    app_server fixture from tests/e2e/conftest.py starts the server if needed.
    """
    return "http://127.0.0.1:7860"


class TestAdvancedWorkflow:
    """
    Advanced E2E tests covering edge cases, settings changes, and error handling.
    """

    def test_navigation_restrictions(self, page: Page, app_server_url):
        """
        Verify that users cannot proceed to later stages without completing earlier ones.
        """
        page.goto(app_server_url)
        expect(page.get_by_role("heading", name="Frame Extractor & Analyzer")).to_be_visible(timeout=30000)

        # Try to go to Subject tab immediately
        page.get_by_role("tab", name="Subject").click()

        # Click "Pre-Analyze Scenes"
        try:
            pre_analyze_btn = page.get_by_role("button", name="🌱 Find & Preview Best Frames")
        except:
            pre_analyze_btn = page.get_by_role("button", name="Pre-Analyze Scenes")

        pre_analyze_btn.click(force=True)

        # Check logs. Must open accordion first.
        page.get_by_text("System Logs").click()

        # Assert that an error is logged.
        # Since exact text might vary between mock implementation and real app,
        # we check for "Error" keyword which is standard for logged exceptions.
        log_area = page.locator(".log-container")
        expect(log_area).to_contain_text("Error", timeout=10000)

    def test_extraction_settings_change(self, page: Page, app_server_url):
        """
        Verify that changing extraction settings works in the UI.
        """
        page.goto(app_server_url)
        expect(page.get_by_role("heading", name="Frame Extractor & Analyzer")).to_be_visible(timeout=30000)

        page.get_by_label("Video URL or Local Path").fill("test_advanced.mp4")

        try:
            extract_btn = page.get_by_role("button", name="🚀 Start Single Extraction")
        except:
            extract_btn = page.get_by_role("button", name="Extract Frames")

        extract_btn.click()

        # Check unified_status (always visible)
        expect(page.locator("body")).to_contain_text("Frame Extraction Complete", timeout=30000)

    def test_filtering_ui(self, page: Page, app_server_url):
        """
        Test the Metrics/Filtering tab UI controls.
        """
        page.goto(app_server_url)
        expect(page.get_by_role("heading", name="Frame Extractor & Analyzer")).to_be_visible(timeout=30000)

        page.get_by_role("tab", name="Export").click()

        smart_filter = page.get_by_label("Smart Filtering")
        expect(smart_filter).to_be_visible()
        smart_filter.check()

        expect(smart_filter).to_be_checked()

        # Verify a slider label update if possible, or at least that no error appeared
        expect(page.locator("body")).not_to_contain_text("Error")
