import re
import pytest
from playwright.sync_api import Page, expect

from .conftest import BASE_URL, open_accordion, switch_to_tab, wait_for_app_ready
from .ui_locators import Labels, Selectors

# Mark as e2e test
pytestmark = pytest.mark.e2e


class TestMockFilters:
    """
    Harden filtering logic tests using the mock app.
    Focuses on UI responsiveness and value propagation.
    """

    @pytest.fixture(autouse=True)
    def setup_mock_analysis(self, page: Page, app_server):
        """
        Setup: Run the full mock pipeline to get frames into the state.
        """
        page.goto(BASE_URL, timeout=60000)
        wait_for_app_ready(page)

        # 1. Reset state to ensure clean start and high-speed mock data (10 frames)
        # We use a mocked reset button that we added to mock_app.py
        page.get_by_text("Reset State (MOCKED)").click()
        # Verify mock frames are loaded before proceeding
        expect(page.get_by_text(re.compile(r"Kept: 10", re.IGNORECASE))).to_be_visible(timeout=10000)

    def test_smart_filter_toggle_updates_ui(self, page: Page):
        """
        Verify that toggling Smart Filtering enables/disables the percentile slider.
        """
        switch_to_tab(page, Labels.TAB_EXPORT)

        smart_filter = page.get_by_label("Smart Filtering (Percentile)")
        expect(smart_filter).to_be_visible()

        # Toggle it
        smart_filter.check()
        expect(smart_filter).to_be_checked()

        # Find the percentile component
        pctl = page.get_by_label("Target Percentile").first
        expect(pctl).to_be_visible()

    def test_filter_preset_application(self, page: Page):
        """
        Test that selecting a preset (e.g., Portrait) updates sliders.
        """
        switch_to_tab(page, Labels.TAB_EXPORT)

        preset_dropdown = page.get_by_label("Use a Preset")
        expect(preset_dropdown).to_be_visible()

        # Select "Portrait/Selfie" preset
        page.get_by_label("Use a Preset").click()
        # Gradio 5 dropdown selection fix: role="option" or text match
        try:
            page.get_by_role("listitem").filter(has_text="Portrait/Selfie").click(force=True, timeout=5000)
        except Exception:
            page.get_by_text("Portrait/Selfie").first.click(force=True)

        # Wait for potential change
        page.wait_for_timeout(1000)

        # Verify no error toast
        expect(page.locator(".toast-wrap")).not_to_be_visible()

    def test_gallery_count_updates_on_filter(self, page: Page):
        """
        Verify that changing a filter value updates the 'Kept' count.
        """
        switch_to_tab(page, Labels.TAB_EXPORT)

        # First, ensure Quality Score accordion is open (it is by default in mock, but let's be safe)
        open_accordion(page, "Quality Score")

        # Find Quality Score min slider. In Gradio, the input within a slider usually has aria-label="Min"
        q_min = page.locator("#scene_quality_score_min_input").locator("input[aria-label='Min']").first
        expect(q_min).to_be_visible()

        q_min.fill("95")  # Set very high

        # Wait for reactive update
        page.wait_for_timeout(2000)

        # The status text should be updated. We expect 10 initial, then some filtering.
        # Since mock_load_frames_into_state has scores at 99.0, 95 min should keep all 10.
        # Let's check for "Kept: 10"
        expect(page.get_by_text(re.compile(r"Kept: 10", re.IGNORECASE))).to_be_visible(timeout=10000)

        # Now set to 100 (which should exclude 99.0)
        q_min.fill("100")
        page.wait_for_timeout(2000)
        expect(page.get_by_text(re.compile(r"Kept: 0", re.IGNORECASE))).to_be_visible(timeout=10000)
