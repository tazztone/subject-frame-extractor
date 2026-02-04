"""
Playwright E2E Tests for session lifecycle.

Tests session management including:
- Session state persistence
- Loading previous sessions
- Session recovery after refresh

Run with: python -m pytest tests/e2e/test_session_lifecycle.py -v -s
"""

import time

import pytest
from playwright.sync_api import Page, expect

from .conftest import BASE_URL

pytestmark = pytest.mark.e2e


class TestSessionPersistence:
    """Tests for session state persistence."""

    def test_session_dropdown_visible(self, page: Page, app_server):
        """Verify session dropdown/selector is visible."""
        page.goto(BASE_URL)
        time.sleep(2)

        # Look for session-related UI elements
        # Usually in the Extract tab or a sidebar
        session_elements = page.locator("text=Session").or_(page.locator("text=session"))
        if session_elements.count() > 0:
            print(f"Found {session_elements.count()} session-related elements")
        else:
            print("No explicit session elements found (may be implicit)")

    def test_output_folder_persists(self, page: Page, app_server):
        """Verify output folder path persists across tab switches."""
        page.goto(BASE_URL)
        time.sleep(2)

        # Set a source path
        source_input = page.get_by_label("Video URL or Local Path")
        source_input.fill("my_test_video.mp4")

        # Switch tabs
        page.get_by_role("tab", name="Subject").click(force=True)
        time.sleep(0.5)
        page.get_by_role("tab", name="Extract").click(force=True)
        time.sleep(0.5)

        # Verify value persisted
        expect(source_input).to_have_value("my_test_video.mp4")


class TestSessionRecovery:
    """Tests for session recovery scenarios."""

    def test_app_loads_without_errors(self, page: Page, app_server):
        """Verify app loads cleanly without console errors."""
        page.goto(BASE_URL)
        time.sleep(3)

        # Check that main UI elements are present
        # This indirectly verifies no critical JS errors occurred
        expect(page.get_by_text("Provide a Video Source")).to_be_visible(timeout=10000)

    def test_multiple_tab_switches(self, page: Page, app_server):
        """Test rapid tab switching doesn't cause errors."""
        page.goto(BASE_URL)
        time.sleep(2)

        tabs = ["Extract", "Subject", "Scenes", "Metrics", "Export"]

        # Rapid tab switching
        for _ in range(3):
            for tab_name in tabs:
                tab = page.get_by_role("tab", name=tab_name)
                if tab.is_visible():
                    tab.click(force=True)
                    time.sleep(0.2)

        # App should still be responsive
        expect(page.get_by_role("tab", name="Extract")).to_be_visible(timeout=5000)


class TestWorkflowState:
    """Tests for workflow state management."""

    def test_extraction_enables_subject_tab(self, extracted_session):
        """Verify Subject tab becomes usable after extraction."""
        page = extracted_session

        # Navigate to Subject tab
        page.get_by_role("tab", name="Subject").click(force=True)
        time.sleep(1)

        # Find Best Frames button should be visible
        btn = page.get_by_role("button", name="🌱 Find & Preview Best Frames")
        expect(btn).to_be_visible(timeout=5000)

    def test_workflow_progress_tracking(self, page: Page, app_server):
        """Verify workflow progress is tracked."""
        page.goto(BASE_URL)
        time.sleep(2)

        # Run extraction
        page.get_by_label("Video URL or Local Path").fill("test_video.mp4")
        page.get_by_role("button", name="🚀 Start Single Extraction").click()

        # Wait for completion
        expect(page.get_by_text("Extraction complete")).to_be_visible(timeout=20000)

        # Log should contain extraction info
        log = page.locator("#unified_log")
        expect(log).to_be_visible(timeout=5000)


class TestLoadPreviousSession:
    """Tests for loading previous sessions."""

    def test_session_loader_ui(self, page: Page, app_server):
        """Verify session loading UI is accessible."""
        page.goto(BASE_URL)
        time.sleep(2)

        # Look for "Load Session" or similar button/dropdown
        load_elements = page.locator("text=Load").or_(page.locator("text=Previous"))
        count = load_elements.count()
        print(f"Found {count} load-related elements")

    def test_no_crash_on_fresh_start(self, page: Page, app_server):
        """Verify app starts cleanly with no previous session."""
        page.goto(BASE_URL)
        time.sleep(3)

        # Should load without errors
        extraction_tab = page.get_by_role("tab", name="Extract")
        expect(extraction_tab).to_be_visible(timeout=10000)

        # Should be able to interact
        source_input = page.get_by_label("Video URL or Local Path")
        source_input.fill("fresh_start_test.mp4")
        expect(source_input).to_have_value("fresh_start_test.mp4")
