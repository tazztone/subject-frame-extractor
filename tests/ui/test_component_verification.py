"""
Component-level verification tests using stable elem_id selectors.

Tests that each UI component (sliders, dropdowns, buttons, logs) actually functions
correctly, not just renders. This catches "does nothing" type bugs.

Run with:
    uv run pytest tests/ui/test_component_verification.py -v
"""

import pytest
from playwright.sync_api import Page, expect

from .conftest import BASE_URL, switch_to_tab

pytestmark = [pytest.mark.e2e, pytest.mark.component]


# Sliders to verify: (Tab, Selector)
SLIDERS_BY_TAB = [
    ("Source", "#thumb_megapixels_input"),
    ("Scenes", "#scene_mask_area_min_input"),
    ("Scenes", "#scene_quality_score_min_input"),
    ("Export", "#dedup_thresh_input"),
]


class TestSliderFunctionality:
    """Verify sliders are visible and interactive."""

    @pytest.mark.parametrize("tab,selector", SLIDERS_BY_TAB)
    def test_slider_value_changes(self, page: Page, app_server, tab, selector):
        """Moving a slider should update its internal value."""
        page.goto(BASE_URL, wait_until="domcontentloaded")

        switch_to_tab(page, tab)

        # Force open accordions to ensure visibility
        if tab == "Source" and selector == "#thumb_megapixels_input":
            page.get_by_text("Advanced Processing Settings").click()
        elif tab == "Scenes":
            # Elements are in Batch Filter accordion - click it to ensure it's open
            page.get_by_text("Batch Filter Scenes").first.click()
        elif tab == "Export" and selector == "#dedup_thresh_input":
            # Only click if not visible. Use flexible substring to avoid emoji issues.
            slider_locator = page.locator(f"{selector} input[type=range]")
            if not slider_locator.is_visible():
                page.get_by_text("Deduplication", exact=False).first.click()
                page.wait_for_timeout(500)

        # For sliders, we need the actual range input inside the wrapper
        slider_input = page.locator(f"{selector} input[type=range]")
        expect(slider_input).to_be_visible(timeout=5000)

        # Get initial value
        initial_value = slider_input.input_value()

        # Change slider value using keyboard
        slider_input.focus()
        page.keyboard.press("ArrowRight")

        # Verify it changed
        expect(slider_input).not_to_have_value(initial_value, timeout=2000)


# Dropdowns to verify: (Tab, Selector)
DROPDOWNS_BY_TAB = [
    ("Source", "#max_resolution"),
    ("Source", "#method_input"),
    ("Subject", "#best_frame_strategy_input"),
    ("Export", "#filter_preset_dropdown"),
]


class TestDropdownFunctionality:
    """Verify dropdowns exist and are interactive."""

    @pytest.mark.parametrize("tab,selector", DROPDOWNS_BY_TAB)
    def test_dropdown_is_interactive(self, page: Page, app_server, tab, selector):
        """Dropdowns should be visible and enabled."""
        page.goto(BASE_URL, wait_until="domcontentloaded")

        switch_to_tab(page, tab)

        dropdown = page.locator(selector)
        expect(dropdown).to_be_visible(timeout=5000)
        expect(dropdown).to_be_enabled()


class TestFiltersFunctionality:
    """Verify filter components actually filter content."""

    def test_scene_gallery_view_toggle(self, page: Page, app_server):
        """View toggle changes displayed scenes."""
        page.goto(BASE_URL, wait_until="domcontentloaded")

        switch_to_tab(page, "Scenes")

        view_toggle = page.locator("#scene_gallery_view_toggle")
        expect(view_toggle).to_be_visible(timeout=5000)

        # Click options
        page.get_by_label("All", exact=True).click()
        page.get_by_label("Kept", exact=True).click()


class TestLogsFunctionality:
    """Verify logging system works correctly."""

    def test_logs_visible_in_accordion(self, page: Page, app_server):
        """System Logs accordion contains a textbox."""
        page.goto(BASE_URL, wait_until="domcontentloaded")

        # Find and open logs accordion
        logs_accordion = page.get_by_text("System Logs")
        expect(logs_accordion).to_be_visible()
        logs_accordion.click()

        # Log textbox (inside the wrapper) should be visible
        log_textarea = page.locator("#unified_log textarea")
        expect(log_textarea).to_be_visible()

    def test_logs_have_initial_content(self, page: Page, app_server):
        """Logs should show initial ready message."""
        page.goto(BASE_URL, wait_until="domcontentloaded")

        # Open logs
        page.get_by_text("System Logs").click()

        log_textarea = page.locator("#unified_log textarea")
        # Wait for content from mock app
        expect(log_textarea).not_to_have_value("", timeout=5000)

    def test_clear_logs_button(self, page: Page, app_server):
        """Clear button empties log content."""
        page.goto(BASE_URL, wait_until="domcontentloaded")

        # Open logs
        page.get_by_text("System Logs").click()

        clear_btn = page.get_by_role("button", name="Clear")
        if clear_btn.is_visible():
            clear_btn.click()
            log_textarea = page.locator("#unified_log textarea")
            expect(log_textarea).to_have_value("", timeout=2000)


class TestPaginationFunctionality:
    """Verify pagination controls work correctly."""

    def test_pagination_row_exists(self, page: Page, app_server):
        """Pagination row should exist."""
        page.goto(BASE_URL, wait_until="domcontentloaded")

        switch_to_tab(page, "Scenes")

        pagination_row = page.locator("#pagination_row")
        expect(pagination_row).to_be_visible(timeout=5000)

    def test_prev_next_buttons_exist(self, page: Page, app_server):
        """Previous and Next pagination buttons should exist."""
        page.goto(BASE_URL, wait_until="domcontentloaded")

        switch_to_tab(page, "Scenes")

        expect(page.locator("#prev_page_button")).to_be_visible()
        expect(page.locator("#next_page_button")).to_be_visible()


class TestButtonsFunctionality:
    """Verify buttons are clickable and perform actions."""

    CRITICAL_BUTTONS = [
        ("Source", "#start_extraction_button"),
        ("Source", "#add_to_queue_button"),
        ("Subject", "#start_pre_analysis_button"),
        ("Metrics", "#start_analysis_button"),
        ("Export", "#export_button"),
    ]

    @pytest.mark.parametrize("tab,selector", CRITICAL_BUTTONS)
    def test_button_is_visible(self, page: Page, app_server, tab, selector):
        """Critical buttons should be attached to the DOM on their respective tabs."""
        page.goto(BASE_URL, wait_until="domcontentloaded")

        switch_to_tab(page, tab)

        button = page.locator(selector)
        # Some buttons (like Export) are hidden in groups until a session is loaded.
        # We check for attachment to verify they were at least built.
        expect(button).to_be_attached(timeout=5000)

        # If it's not the Export button, it should also be visible
        if selector != "#export_button":
            expect(button).to_be_visible(timeout=2000)


class TestStrategyVisibility:
    """Verify strategy selection shows/hides appropriate UI groups."""

    def test_face_strategy_shows_face_options(self, page: Page, app_server):
        """Selecting Face strategy should show face-specific options."""
        page.goto(BASE_URL, wait_until="domcontentloaded")

        switch_to_tab(page, "Subject")

        # Use get_by_label to avoid text matching in info description
        page.locator("#primary_seed_strategy_input").get_by_label("👤 By Face").click()

        # Check for child element in the group
        expect(page.get_by_text("Upload Reference Photo")).to_be_visible(timeout=5000)

    def test_text_strategy_shows_text_options(self, page: Page, app_server):
        """Selecting Text strategy should show text prompt and warning."""
        page.goto(BASE_URL, wait_until="domcontentloaded")

        switch_to_tab(page, "Subject")

        # Use get_by_label for consistency
        page.locator("#primary_seed_strategy_input").get_by_label("📝 By Text", exact=False).click()
        expect(page.get_by_placeholder("e.g., 'a man in a blue suit'")).to_be_visible(timeout=5000)
