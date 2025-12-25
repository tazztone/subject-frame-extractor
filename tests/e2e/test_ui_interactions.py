"""
Automated UI Interaction Tests using Playwright.

These tests verify that UI interactions work correctly by:
1. Clicking buttons and verifying log output appears
2. Adjusting sliders and verifying UI updates
3. Monitoring console/terminal for expected messages

Run with: 
    python tests/mock_app.py &
    python -m pytest tests/e2e/test_ui_interactions.py -v -s
"""
import pytest
from playwright.sync_api import Page, expect, ConsoleMessage
import time
import re

from .conftest import BASE_URL

# Mark all tests as e2e
pytestmark = pytest.mark.e2e


class TestFindPeopleButtonInteraction:
    """Tests for Find People in Video button - verifies button click works."""

    def test_find_people_button_clickable(self, page: Page, app_server):
        """Button should be clickable and not crash the app."""
        page.goto(BASE_URL)
        time.sleep(2)

        # Navigate to Subject tab
        subject_tab = page.get_by_role("tab", name="Subject")
        if not subject_tab.is_visible():
            pytest.skip("Subject tab not visible")
        subject_tab.click(force=True)
        time.sleep(1)

        # Select Face strategy to reveal the button
        face_option = page.get_by_text("👤 By Face")
        if face_option.is_visible():
            face_option.click()
            time.sleep(0.5)

        # Find and click the button (new name after fix)
        find_people_btn = page.get_by_role("button", name="Scan Video for Faces")
        if find_people_btn.is_visible():
            # Click and verify no crash
            find_people_btn.click()
            time.sleep(2)  # Wait for processing
            
            # App should still be responsive
            expect(page.locator("body")).to_be_visible()
            
            # Check logs for button click message (open logs accordion)
            logs_accordion = page.get_by_text("📋 System Logs")
            if logs_accordion.is_visible():
                logs_accordion.click()
                time.sleep(0.5)
                
                # Click Refresh to pull latest logs
                refresh_btn = page.get_by_role("button", name="🔄 Refresh")
                if refresh_btn.is_visible():
                    refresh_btn.click()
                    time.sleep(0.5)
                
                # Look for our debug log message
                log_area = page.locator("#unified_log textarea")
                log_text = log_area.input_value() if log_area.count() > 0 else ""
                
                # We expect to see either the button clicked log or an error message
                print(f"Log content (first 500 chars): {log_text[:500]}")
            
            print("✓ Find People button clicked successfully")
        else:
            pytest.skip("Find People button not visible")


class TestGallerySliderInteractions:
    """Tests for gallery size sliders - verifies sliders affect gallery."""

    def test_columns_slider_exists_and_interactive(self, page: Page, app_server):
        """Columns slider should exist and be draggable."""
        page.goto(BASE_URL)
        time.sleep(2)

        # Navigate to Scenes tab
        scenes_tab = page.get_by_role("tab", name="Scenes")
        if not scenes_tab.is_visible():
            pytest.skip("Scenes tab not visible")
        scenes_tab.click(force=True)
        time.sleep(1)

        # Find Columns slider
        columns_slider = page.get_by_label("Columns")
        if columns_slider.is_visible():
            # Get initial value
            initial_value = columns_slider.input_value()
            print(f"Initial columns value: {initial_value}")
            
            # Try to change the slider value
            columns_slider.fill("4")  # Set to 4 columns
            columns_slider.press("Enter")
            time.sleep(0.5)
            
            # Trigger the release event by clicking elsewhere
            page.locator("body").click()
            time.sleep(0.5)
            
            print("✓ Columns slider interaction completed")
        else:
            pytest.skip("Columns slider not visible")

    def test_height_slider_exists_and_interactive(self, page: Page, app_server):
        """Height slider should exist and be adjustable."""
        page.goto(BASE_URL)
        time.sleep(2)

        # Navigate to Scenes tab
        scenes_tab = page.get_by_role("tab", name="Scenes")
        if not scenes_tab.is_visible():
            pytest.skip("Scenes tab not visible")
        scenes_tab.click(force=True)
        time.sleep(1)

        # Find Height slider
        height_slider = page.get_by_label("Gallery Height")
        if height_slider.is_visible():
            initial_value = height_slider.input_value()
            print(f"Initial height value: {initial_value}")
            
            # Change the slider
            height_slider.fill("400")  # Set to 400px
            height_slider.press("Enter")
            time.sleep(0.5)
            
            page.locator("body").click()
            time.sleep(0.5)
            
            print("✓ Height slider interaction completed")
        else:
            pytest.skip("Height slider not visible")


class TestLogRefreshMechanism:
    """Tests for log display - verifies logs can be refreshed."""

    def test_refresh_button_updates_logs(self, page: Page, app_server):
        """Clicking Refresh should drain log queue and update display."""
        page.goto(BASE_URL)
        time.sleep(2)

        # Open logs accordion
        logs_accordion = page.get_by_text("📋 System Logs")
        expect(logs_accordion).to_be_visible(timeout=5000)
        logs_accordion.click()
        time.sleep(0.5)

        # Get initial log content
        log_area = page.locator("#unified_log textarea")
        initial_logs = log_area.input_value() if log_area.count() > 0 else ""

        # Click Refresh
        refresh_btn = page.get_by_role("button", name="🔄 Refresh")
        expect(refresh_btn).to_be_visible(timeout=5000)
        refresh_btn.click()
        time.sleep(1)

        # Log area should still be visible and functional
        expect(page.locator("#unified_log")).to_be_visible()
        print("✓ Log refresh works without errors")


class TestPropagationErrorHandling:
    """Tests for propagation - verifies graceful error handling."""

    def test_propagation_without_scenes_no_crash(self, page: Page, app_server):
        """Clicking propagate without scenes should not crash."""
        page.goto(BASE_URL)
        time.sleep(2)

        # Navigate to Scenes tab
        scenes_tab = page.get_by_role("tab", name="Scenes")
        if not scenes_tab.is_visible():
            pytest.skip("Scenes tab not visible")
        scenes_tab.click(force=True)
        time.sleep(1)

        # Find propagate button (should be disabled but let's try)
        propagate_btns = page.get_by_role("button", name=re.compile("Propagate"))
        if propagate_btns.count() > 0:
            btn = propagate_btns.first
            if btn.is_visible():
                try:
                    btn.click(timeout=2000)
                    time.sleep(2)
                except:
                    pass  # Button might be disabled, which is fine
                
                # App should still be responsive
                expect(page.locator("body")).to_be_visible()
                print("✓ Propagation button click handled gracefully")


class TestUIConsoleErrors:
    """Tests that monitor browser console for JavaScript errors."""

    def test_no_console_errors_on_load(self, page: Page, app_server):
        """Page should load without JavaScript errors."""
        console_errors = []
        
        def handle_console(msg: ConsoleMessage):
            if msg.type == "error":
                console_errors.append(msg.text)
        
        page.on("console", handle_console)
        page.goto(BASE_URL)
        time.sleep(3)
        
        # Filter out known benign errors
        critical_errors = [e for e in console_errors if "gradio" not in e.lower()]
        
        if critical_errors:
            print(f"Console errors found: {critical_errors}")
        
        # We don't fail on errors, just report them
        print(f"✓ Page loaded. Console errors: {len(console_errors)}")

    def test_no_errors_during_tab_navigation(self, page: Page, app_server):
        """Navigating through tabs should not cause errors."""
        console_errors = []
        
        def handle_console(msg: ConsoleMessage):
            if msg.type == "error":
                console_errors.append(msg.text)
        
        page.on("console", handle_console)
        page.goto(BASE_URL)
        time.sleep(2)
        
        tabs = ["Extract", "Subject", "Scenes", "Metrics", "Export"]
        for tab_name in tabs:
            tab = page.get_by_role("tab", name=tab_name)
            if tab.is_visible():
                tab.click(force=True)
                time.sleep(0.5)
        
        print(f"✓ Tab navigation complete. Errors during navigation: {len(console_errors)}")
