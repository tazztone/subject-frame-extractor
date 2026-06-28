"""
Playwright E2E Tests for main application workflow.

Driven through ``AppDriver`` — the canonical UI interaction layer. Tests read
as intent + assertion; locator/synchronization knowledge lives in the driver.
"""

import pytest
from playwright.sync_api import expect

from .app_driver import AppDriver
from .ui_locators import Labels, Selectors


@pytest.mark.usefixtures("app_server")
class TestMainWorkflow:
    def test_full_user_flow(self, app_driver: AppDriver):
        """
        Tests the complete end-to-end workflow:
        Extraction -> Pre-Analysis -> Scene Selection -> Propagation -> Analysis -> Export
        """
        # 1. Frame Extraction
        app_driver.extract("dummy_video.mp4").expect_status("Extraction Complete", timeout=15000)

        # 2. Define Subject (Pre-Analysis)
        app_driver.navigate(Labels.TAB_SUBJECT).pre_analyze().expect_status("Pre-Analysis Complete", timeout=15000)

        # 3. Scene Selection & Propagation
        app_driver.navigate(Labels.TAB_SCENES).propagate().expect_status("Mask Propagation Complete", timeout=30000)

        # 4. Final Analysis
        app_driver.navigate(Labels.TAB_METRICS).analyze().expect_status("Analysis Complete", timeout=30000)

        # 5. Export
        app_driver.navigate(Labels.TAB_EXPORT).export()
        # We don't have a mock for Export Complete yet in mock_app.py
        # but the button should be clickable and not crash the UI.


@pytest.mark.usefixtures("app_server")
class TestTabNavigation:
    def test_all_tabs_accessible(self, app_driver: AppDriver):
        """Verify each major tab can be reached."""
        tabs = [Labels.TAB_SOURCE, Labels.TAB_SUBJECT, Labels.TAB_SCENES, Labels.TAB_METRICS, Labels.TAB_EXPORT]
        for tab_label in tabs:
            app_driver.navigate(tab_label)
            # Check if a unique element for that tab is visible
            if tab_label == Labels.TAB_SOURCE:
                app_driver.expect_visible(Selectors.START_EXTRACTION)
            elif tab_label == Labels.TAB_SUBJECT:
                app_driver.expect_visible(Selectors.START_PRE_ANALYSIS)

    def test_tab_state_preserved(self, app_driver: AppDriver):
        """Verify input state is kept when switching tabs."""
        # Fill input on first tab
        test_val = "PERSISTENT_VALUE.mp4"
        app_driver.page.locator(Selectors.SOURCE_INPUT).fill(test_val)

        # Switch away and back
        app_driver.navigate(Labels.TAB_SUBJECT).navigate(Labels.TAB_SOURCE)

        expect(app_driver.page.locator(Selectors.SOURCE_INPUT)).to_have_value(test_val)


@pytest.mark.usefixtures("app_server")
class TestErrorHandling:
    def test_empty_source_shows_message(self, app_driver: AppDriver):
        """Verify appropriate message when no source is provided."""
        # Click extraction (will fail due to empty path)
        app_driver.page.locator(Selectors.START_EXTRACTION).click(force=True)

        # Check status for Error, then confirm detail in the logs.
        app_driver.expect_status(Selectors.STATUS_ERROR_REGEX, timeout=10000)
        app_driver.expect_log("Error", timeout=10000)
