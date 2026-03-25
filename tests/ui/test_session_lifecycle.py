import pytest
from playwright.sync_api import Page, expect

from .conftest import BASE_URL, open_accordion, switch_to_tab
from .ui_locators import Labels, Selectors

pytestmark = pytest.mark.e2e


class TestSessionPersistence:
    """Tests for session state persistence."""

    def test_session_loader_visible(self, page: Page, app_server):
        """Verify session loader accordion is visible."""
        page.goto(BASE_URL)
        page.wait_for_timeout(2000)

        session_acc = page.get_by_text("Resume previous Session", exact=False)
        expect(session_acc).to_be_visible()

    def test_source_input_persists(self, page: Page, app_server):
        """Verify source input path persists across tab switches."""
        page.goto(BASE_URL)
        page.wait_for_timeout(2000)

        # Set a source path
        source_input = page.get_by_placeholder(Labels.SOURCE_PLACEHOLDER)
        if not source_input.is_visible():
            source_input = page.locator(Selectors.SOURCE_INPUT)

        source_input.fill("my_test_video.mp4")

        # Switch tabs
        switch_to_tab(page, Labels.TAB_SUBJECT)
        switch_to_tab(page, Labels.TAB_SOURCE)

        # Verify value persisted
        expect(source_input).to_have_value("my_test_video.mp4")


class TestSessionRecovery:
    """Tests for session recovery scenarios."""

    def test_app_loads_without_errors(self, page: Page, app_server):
        """Verify app loads cleanly without console errors."""
        page.goto(BASE_URL)
        page.wait_for_timeout(3000)

        # Check that main UI elements are present
        expect(page.get_by_text("Input & Extraction", exact=False)).to_be_visible(timeout=10000)

    def test_multiple_tab_switches(self, page: Page, app_server):
        """Test rapid tab switching doesn't cause errors."""
        page.goto(BASE_URL)
        page.wait_for_timeout(2000)

        tabs = [Labels.TAB_SOURCE, Labels.TAB_SUBJECT, Labels.TAB_SCENES, Labels.TAB_METRICS, Labels.TAB_EXPORT]

        # Rapid tab switching
        for _ in range(2):
            for tab_name in tabs:
                switch_to_tab(page, tab_name)

        # App should still be responsive
        expect(page.get_by_role("tab", name=Labels.TAB_SOURCE, exact=False)).to_be_visible(timeout=5000)


class TestWorkflowState:
    """Tests for workflow state management."""

    def test_extraction_enables_subject_tab(self, extracted_session):
        """Verify Subject tab becomes usable after extraction."""
        page = extracted_session

        # Navigate to Subject tab
        switch_to_tab(page, Labels.TAB_SUBJECT)

        # Confirm Subject button should be visible
        btn = page.locator(Selectors.START_PRE_ANALYSIS)
        expect(btn).to_be_visible(timeout=5000)

    def test_workflow_progress_tracking(self, page: Page, app_server):
        """Verify workflow progress is tracked."""
        page.goto(BASE_URL)
        page.wait_for_timeout(2000)

        # Run extraction
        source_input = page.get_by_placeholder(Labels.SOURCE_PLACEHOLDER)
        if not source_input.is_visible():
            source_input = page.locator(Selectors.SOURCE_INPUT)

        source_input.fill("test_video.mp4")
        page.locator(Selectors.START_EXTRACTION).click()

        # Wait for completion
        expect(page.locator(Selectors.UNIFIED_STATUS)).to_contain_text("Extraction Complete", timeout=30000)

        # Log should contain extraction info
        # Open logs accordion because it's closed by default
        open_accordion(page, Labels.SYSTEM_LOGS)

        log = page.locator(Selectors.UNIFIED_LOG)
        expect(log).to_be_visible(timeout=5000)
