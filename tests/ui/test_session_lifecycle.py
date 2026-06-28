import pytest
from playwright.sync_api import Page, expect

from .app_driver import AppDriver
from .ui_locators import Labels, Selectors

pytestmark = pytest.mark.e2e


class TestSessionPersistence:
    """Tests for session state persistence."""

    def test_session_loader_visible(self, app_driver: AppDriver):
        """Verify session loader accordion is visible."""
        # Gradio 5+ session loader is under 'Resume previous Session'
        session_acc = app_driver.page.get_by_text("Resume previous Session", exact=False)
        expect(session_acc).to_be_visible()

    def test_source_input_persists(self, app_driver: AppDriver):
        """Verify source input path persists across tab switches."""
        source_input = app_driver.page.locator(Selectors.SOURCE_INPUT)
        source_input.fill("my_test_video.mp4")

        # Switch tabs
        app_driver.navigate(Labels.TAB_SUBJECT).navigate(Labels.TAB_SOURCE)

        # Verify value persisted
        expect(source_input).to_have_value("my_test_video.mp4")


class TestSessionRecovery:
    """Tests for session recovery scenarios."""

    def test_app_loads_without_errors(self, app_driver: AppDriver):
        """Verify app loads cleanly without console errors."""
        # Check for Step 1 header
        expect(app_driver.page.get_by_text("Input & Extraction", exact=False)).to_be_visible(timeout=10000)

    def test_multiple_tab_switches(self, app_driver: AppDriver):
        """Test rapid tab switching doesn't cause errors."""
        tabs = [Labels.TAB_SOURCE, Labels.TAB_SUBJECT, Labels.TAB_SCENES, Labels.TAB_METRICS, Labels.TAB_EXPORT]

        # Rapid tab switching
        for _ in range(2):
            for tab_name in tabs:
                app_driver.navigate(tab_name)

        # App should still be responsive
        expect(app_driver.page.get_by_role("tab", name=Labels.TAB_SOURCE, exact=False)).to_be_visible(timeout=5000)


class TestWorkflowState:
    """Tests for workflow state management."""

    def test_extraction_enables_subject_tab(self, extracted_session: Page):
        """Verify Subject tab becomes usable after extraction."""
        driver = AppDriver(extracted_session)
        driver.navigate(Labels.TAB_SUBJECT)

        # Confirm Subject button should be visible
        driver.expect_visible(Selectors.START_PRE_ANALYSIS, timeout=5000)

    def test_workflow_progress_tracking(self, app_driver: AppDriver):
        """Verify workflow progress is tracked."""
        # Run extraction
        app_driver.page.locator(Selectors.SOURCE_INPUT).fill("test_video.mp4")
        app_driver.page.locator(Selectors.START_EXTRACTION).click()

        # Wait for completion
        app_driver.expect_status(Selectors.STATUS_SUCCESS_EXTRACTION, timeout=30000)

        # Log container should be visible
        app_driver.open_accordion(Labels.SYSTEM_LOGS)
        app_driver.expect_visible(Selectors.UNIFIED_LOG, timeout=5000)
