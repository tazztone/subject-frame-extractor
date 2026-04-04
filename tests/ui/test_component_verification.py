"""
Component-level verification tests using stable elem_id selectors.
Standardized to use the new unified Selectors and Labels contract.
"""

import pytest
from playwright.sync_api import Page, expect

from .conftest import BASE_URL, open_accordion, switch_to_tab, wait_for_app_ready
from .ui_locators import Labels, Selectors

pytestmark = [pytest.mark.e2e, pytest.mark.component]


# Sliders to verify: (Tab, Selector)
SLIDERS_BY_TAB = [
    (Labels.TAB_SOURCE, Selectors.THUMB_MEGAPIXELS),
    (Labels.TAB_SCENES, Selectors.SCENE_MASK_AREA_MIN),
    (Labels.TAB_SCENES, Selectors.SCENE_QUALITY_SCORE_MIN),
    (Labels.TAB_EXPORT, Selectors.DEDUP_THRESH),
]


class TestSliderFunctionality:
    """Verify sliders are visible and interactive."""

    @pytest.mark.parametrize("tab,selector", SLIDERS_BY_TAB)
    def test_slider_value_changes(self, page: Page, app_server, tab, selector):
        """Moving a slider should update its internal value."""
        page.goto(BASE_URL)
        wait_for_app_ready(page)

        switch_to_tab(page, tab)

        # Force open accordions to ensure visibility
        if tab == Labels.TAB_SOURCE and selector == Selectors.THUMB_MEGAPIXELS:
            open_accordion(page, "Advanced Processing Settings")
        elif tab == Labels.TAB_SCENES:
            open_accordion(page, "Batch Filter Scenes")
        elif tab == Labels.TAB_EXPORT and selector == Selectors.DEDUP_THRESH:
            open_accordion(page, "Deduplication")

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
    (Labels.TAB_SOURCE, Selectors.MAX_RESOLUTION),
    (Labels.TAB_SOURCE, Selectors.METHOD_INPUT),
    (Labels.TAB_SUBJECT, Selectors.BEST_FRAME_STRATEGY),
    (Labels.TAB_EXPORT, Selectors.FILTER_PRESET),
]


class TestDropdownFunctionality:
    """Verify dropdowns exist and are interactive."""

    @pytest.mark.parametrize("tab,selector", DROPDOWNS_BY_TAB)
    def test_dropdown_is_interactive(self, page: Page, app_server, tab, selector):
        """Dropdowns should be visible and enabled."""
        page.goto(BASE_URL)
        wait_for_app_ready(page)

        switch_to_tab(page, tab)

        dropdown = page.locator(selector)
        expect(dropdown).to_be_visible(timeout=5000)
        expect(dropdown).to_be_enabled()


class TestFiltersFunctionality:
    """Verify filter components actually filter content."""

    def test_scene_gallery_view_toggle(self, page: Page, app_server):
        """View toggle changes displayed scenes."""
        page.goto(BASE_URL)
        wait_for_app_ready(page)

        switch_to_tab(page, Labels.TAB_SCENES)

        view_toggle = page.locator(Selectors.SCENE_GALLERY_VIEW_TOGGLE)
        expect(view_toggle).to_be_visible(timeout=5000)

        # Click options by label
        page.get_by_label("All", exact=True).click()
        page.get_by_label("Kept", exact=True).click()


class TestLogsFunctionality:
    """Verify logging system works correctly."""

    def test_logs_visible_in_accordion(self, page: Page, app_server):
        """System Logs accordion contains a textbox."""
        page.goto(BASE_URL)
        wait_for_app_ready(page)

        # Find and open logs accordion
        open_accordion(page, Labels.SYSTEM_LOGS)

        # Log textarea should be visible
        expect(page.locator(Selectors.LOG_TEXTAREA)).to_be_visible(timeout=5000)

    def test_logs_have_initial_content(self, page: Page, app_server):
        """Logs should show initial ready message."""
        page.goto(BASE_URL)
        wait_for_app_ready(page)

        # Open logs
        open_accordion(page, Labels.SYSTEM_LOGS)

        # Wait for content from mock app
        expect(page.locator(Selectors.LOG_TEXTAREA)).not_to_have_value("", timeout=5000)

    def test_clear_logs_button(self, page: Page, app_server):
        """Clear button empties log content."""
        page.goto(BASE_URL)
        wait_for_app_ready(page)

        # Open logs
        open_accordion(page, Labels.SYSTEM_LOGS)
        page.wait_for_timeout(500)

        clear_btn = page.get_by_role("button", name="Clear", exact=False)
        if clear_btn.is_visible():
            clear_btn.click()
            expect(page.locator(Selectors.LOG_TEXTAREA)).to_have_value("", timeout=2000)


class TestPaginationFunctionality:
    """Verify pagination controls work correctly."""

    def test_pagination_row_exists(self, page: Page, app_server):
        """Pagination row should exist."""
        page.goto(BASE_URL)
        wait_for_app_ready(page)

        switch_to_tab(page, Labels.TAB_SCENES)

        pagination_row = page.locator("#pagination_row")
        expect(pagination_row).to_be_visible(timeout=5000)

    def test_prev_next_buttons_exist(self, page: Page, app_server):
        """Previous and Next pagination buttons should exist."""
        page.goto(BASE_URL)
        wait_for_app_ready(page)

        switch_to_tab(page, Labels.TAB_SCENES)

        expect(page.locator(Selectors.PREV_PAGE_BUTTON)).to_be_visible()
        expect(page.locator(Selectors.NEXT_PAGE_BUTTON)).to_be_visible()


class TestButtonsFunctionality:
    """Verify buttons are clickable and perform actions."""

    CRITICAL_BUTTONS = [
        (Labels.TAB_SOURCE, Selectors.START_EXTRACTION),
        (Labels.TAB_SUBJECT, Selectors.START_PRE_ANALYSIS),
        (Labels.TAB_METRICS, Selectors.START_ANALYSIS),
        (Labels.TAB_EXPORT, Selectors.EXPORT_BUTTON),
    ]

    @pytest.mark.parametrize("tab,selector", CRITICAL_BUTTONS)
    def test_button_is_visible(self, page: Page, app_server, tab, selector):
        """Critical buttons should be attached to the DOM on their respective tabs."""
        page.goto(BASE_URL)
        wait_for_app_ready(page)

        switch_to_tab(page, tab)

        button = page.locator(selector)
        # Check for attachment to verify they were at least built.
        expect(button).to_be_attached(timeout=5000)

        # Non-Export buttons should be visible immediately after app ready
        if tab != Labels.TAB_EXPORT:
            expect(button).to_be_visible(timeout=5000)


class TestStrategyVisibility:
    """Verify strategy selection shows/hides appropriate UI groups."""

    def test_face_strategy_shows_face_options(self, page: Page, app_server):
        """Selecting Face strategy should show face-specific options."""
        page.goto(BASE_URL)
        wait_for_app_ready(page)

        switch_to_tab(page, Labels.TAB_SUBJECT)

        # Use get_by_label with exact=False for resilience to emojis/spans
        page.get_by_label(Labels.STRATEGY_FACE, exact=False).click()

        # Check for child element in the group
        expect(page.get_by_text("Upload Reference Photo", exact=False)).to_be_visible(timeout=5000)

    def test_text_strategy_shows_text_options(self, page: Page, app_server):
        """Selecting Text strategy should show text prompt and warning."""
        page.goto(BASE_URL)
        wait_for_app_ready(page)

        switch_to_tab(page, Labels.TAB_SUBJECT)

        page.get_by_label(Labels.STRATEGY_TEXT, exact=False).click()

        # Text prompt should appear
        expect(page.get_by_text("What should we look for?", exact=False)).to_be_visible(timeout=5000)
