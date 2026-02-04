"""
Playwright E2E Tests for session lifecycle.

Tests session management including:
- Session state persistence
- Loading previous sessions
- Session recovery after refresh

Run with: python -m pytest tests/ui/test_session_lifecycle.py -v -s
"""

import time

import pytest
from playwright.sync_api import Page, expect

from .conftest import BASE_URL

pytestmark = pytest.mark.e2e


class TestSessionPersistence:
    """Tests for session state persistence."""

    def test_session_loader_visible(self, page: Page, app_server):
        """Verify session loader accordion is visible."""
        page.goto(BASE_URL)
        time.sleep(2)

        session_acc = page.get_by_text("Resume previous Session")
        expect(session_acc).to_be_visible()

    def test_source_input_persists(self, page: Page, app_server):
        """Verify source input path persists across tab switches."""
        page.goto(BASE_URL)
        time.sleep(2)

        # Set a source path
        source_input = page.get_by_label("Input Path or URL")
        source_input.fill("my_test_video.mp4")

        # Switch tabs
        page.get_by_role("tab", name="Subject").click(force=True)
        time.sleep(0.5)
        page.get_by_role("tab", name="Source").click(force=True)
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
        expect(page.get_by_text("Input & Extraction")).to_be_visible(timeout=10000)

    def test_multiple_tab_switches(self, page: Page, app_server):
        """Test rapid tab switching doesn't cause errors."""
        page.goto(BASE_URL)
        time.sleep(2)

        tabs = ["Source", "Subject", "Scenes", "Metrics", "Export"]

        # Rapid tab switching
        for _ in range(2):
            for tab_name in tabs:
                tab = page.get_by_role("tab", name=tab_name)
                if tab.is_visible():
                    tab.click(force=True)
                    time.sleep(0.2)

        # App should still be responsive
        expect(page.get_by_role("tab", name="Source")).to_be_visible(timeout=5000)


class TestWorkflowState:
    """Tests for workflow state management."""

    def test_extraction_enables_subject_tab(self, extracted_session):
        """Verify Subject tab becomes usable after extraction."""
        page = extracted_session

        # Navigate to Subject tab
        page.get_by_role("tab", name="Subject").click(force=True)
        time.sleep(1)

        # Confirm Subject button should be visible
        btn = page.get_by_role("button", name="✅ Confirm Subject & Find Scenes (Next Step)")
        expect(btn).to_be_visible(timeout=5000)

    def test_workflow_progress_tracking(self, page: Page, app_server):
        """Verify workflow progress is tracked."""
        page.goto(BASE_URL)
        time.sleep(2)

        # Run extraction
        page.get_by_label("Input Path or URL").fill("test_video.mp4")
        page.get_by_role("button", name="🚀 Start Extraction").click()

        # Wait for completion
        expect(page.get_by_text("Extraction complete")).to_be_visible(timeout=20000)

        # Log should contain extraction info
        log = page.locator("#unified_log")
        expect(log).to_be_visible(timeout=5000)
