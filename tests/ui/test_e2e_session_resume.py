import pytest
from playwright.sync_api import Page, expect

from .conftest import BASE_URL, open_accordion, wait_for_app_ready
from .ui_locators import Labels, Selectors

# Mark as e2e test
pytestmark = pytest.mark.e2e


class TestSessionResume:
    """Verify that previous sessions can be resumed and UI state is restored."""

    def test_navigation_to_resume_accordion(self, page: Page, app_server):
        """Verify the transition to resume accordion exists."""
        page.goto(BASE_URL)
        wait_for_app_ready(page)

        # Check for Session Accordion
        accordion = page.get_by_role("button", name=Labels.SESSION_ACCORDION)
        expect(accordion).to_be_visible()

    def test_resume_elements_visibility(self, page: Page, app_server):
        """Verify UI elements within the resume accordion."""
        page.goto(BASE_URL)
        wait_for_app_ready(page)

        open_accordion(page, Labels.SESSION_ACCORDION)

        expect(page.locator(Selectors.SESSION_INPUT)).to_be_visible()
        expect(page.get_by_role("button", name="Load Session")).to_be_visible()
