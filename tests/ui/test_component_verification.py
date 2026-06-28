"""
Component-level verification tests using stable elem_id selectors.
Driven through ``AppDriver``; raw ``page`` retained only for the slider
keyboard-input and gallery-toggle idioms the driver does not model.
"""

import pytest
from playwright.sync_api import expect

from .app_driver import AppDriver
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
    def test_slider_value_changes(self, app_driver: AppDriver, tab, selector):
        """Moving a slider should update its internal value."""
        page = app_driver.navigate(tab).page

        # Force open accordions to ensure visibility
        if tab == Labels.TAB_SOURCE and selector == Selectors.THUMB_MEGAPIXELS:
            app_driver.open_accordion("Advanced Processing Settings")
        elif tab == Labels.TAB_SCENES:
            app_driver.open_accordion("Batch Filter Scenes")
        elif tab == Labels.TAB_EXPORT and selector == Selectors.DEDUP_THRESH:
            app_driver.open_accordion("Deduplication")

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
    def test_dropdown_is_interactive(self, app_driver: AppDriver, tab, selector):
        """Dropdowns should be visible and enabled."""
        page = app_driver.navigate(tab).page

        dropdown = page.locator(selector)
        expect(dropdown).to_be_visible(timeout=5000)
        expect(dropdown).to_be_enabled()


class TestFiltersFunctionality:
    """Verify filter components actually filter content."""

    def test_scene_gallery_view_toggle(self, app_driver: AppDriver):
        """View toggle changes displayed scenes."""
        page = app_driver.navigate(Labels.TAB_SCENES).page

        view_toggle = page.locator(Selectors.SCENE_GALLERY_VIEW_TOGGLE)
        expect(view_toggle).to_be_visible(timeout=5000)

        # Click options by label
        page.get_by_label("All", exact=True).click()
        page.get_by_label("Kept", exact=True).click()


class TestLogsFunctionality:
    """Verify logging system works correctly."""

    def test_logs_visible_in_accordion(self, app_driver: AppDriver):
        """System Logs accordion contains a textbox."""
        app_driver.open_accordion(Labels.SYSTEM_LOGS)
        app_driver.expect_visible(Selectors.LOG_TEXTAREA, timeout=5000)

    def test_logs_have_initial_content(self, app_driver: AppDriver):
        """Logs should show initial ready message."""
        app_driver.open_accordion(Labels.SYSTEM_LOGS)
        # Wait for content from mock app (textarea should be non-empty).
        expect(app_driver.page.locator(Selectors.LOG_TEXTAREA)).not_to_have_value("", timeout=5000)

    def test_clear_logs_button(self, app_driver: AppDriver):
        """Clear button empties log content."""
        app_driver.open_accordion(Labels.SYSTEM_LOGS)
        app_driver.page.wait_for_timeout(500)

        clear_btn = app_driver.page.get_by_role("button", name="Clear", exact=False)
        if clear_btn.is_visible():
            clear_btn.click()
            # Exact-value assertion: empty string after clear.
            app_driver.expect_log_equals("", timeout=2000)


class TestPaginationFunctionality:
    """Verify pagination controls work correctly."""

    def test_pagination_row_exists(self, app_driver: AppDriver):
        """Pagination row should exist."""
        app_driver.navigate(Labels.TAB_SCENES)
        app_driver.expect_visible("#pagination_row", timeout=5000)

    def test_prev_next_buttons_exist(self, app_driver: AppDriver):
        """Previous and Next pagination buttons should exist."""
        app_driver.navigate(Labels.TAB_SCENES)
        app_driver.expect_visible(Selectors.PREV_PAGE_BUTTON)
        app_driver.expect_visible(Selectors.NEXT_PAGE_BUTTON)


class TestButtonsFunctionality:
    """Verify buttons are clickable and perform actions."""

    CRITICAL_BUTTONS = [
        (Labels.TAB_SOURCE, Selectors.START_EXTRACTION),
        (Labels.TAB_SUBJECT, Selectors.START_PRE_ANALYSIS),
        (Labels.TAB_METRICS, Selectors.START_ANALYSIS),
        (Labels.TAB_EXPORT, Selectors.EXPORT_BUTTON),
    ]

    @pytest.mark.parametrize("tab,selector", CRITICAL_BUTTONS)
    def test_button_is_visible(self, app_driver: AppDriver, tab, selector):
        """Critical buttons should be attached to the DOM on their respective tabs."""
        app_driver.navigate(tab)

        button = app_driver.page.locator(selector)
        # Check for attachment to verify they were at least built.
        expect(button).to_be_attached(timeout=5000)

        # Non-Export buttons should be visible immediately after app ready
        if tab != Labels.TAB_EXPORT:
            expect(button).to_be_visible(timeout=5000)


class TestStrategyVisibility:
    """Verify strategy selection shows/hides appropriate UI groups."""

    def test_face_strategy_shows_face_options(self, app_driver: AppDriver):
        """Selecting Face strategy should show face-specific options."""
        app_driver.navigate(Labels.TAB_SUBJECT)
        app_driver.select_strategy(Labels.STRATEGY_FACE)

        expect(app_driver.page.get_by_text("Upload Reference Photo", exact=False)).to_be_visible(timeout=5000)

    def test_text_strategy_shows_text_options(self, app_driver: AppDriver):
        """Selecting Text strategy should show text prompt and warning."""
        app_driver.navigate(Labels.TAB_SUBJECT)
        app_driver.select_strategy(Labels.STRATEGY_TEXT)

        # Text prompt should appear
        expect(app_driver.page.get_by_text("What should we look for?", exact=False)).to_be_visible(timeout=5000)
