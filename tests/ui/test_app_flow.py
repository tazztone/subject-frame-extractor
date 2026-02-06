"""
Playwright E2E Tests for main application workflow.

Restructured for Gradio 5 and robust status tracking.
"""

import time
import re

import pytest
from pathlib import Path
from playwright.sync_api import Page, expect

from .conftest import BASE_URL

# Mark all tests as e2e
pytestmark = pytest.mark.e2e


def switch_to_tab(page: Page, tab_name: str):
    """Robustly switch tabs in Gradio, handling potential race conditions."""
    tab_btn = page.get_by_role("tab", name=tab_name)
    # Wait for the tab button to exist and be visible
    tab_btn.wait_for(state="visible", timeout=15000)
    
    # Only click if not already selected
    if tab_btn.get_attribute("aria-selected") != "true":
        tab_btn.click(force=True)
        # Wait for tab activation
        expect(tab_btn).to_have_attribute("aria-selected", "true", timeout=10000)
    
    time.sleep(1.5)  # Allow time for tab content to render


def open_logs(page: Page):
    """Opens the system logs accordion if it's closed."""
    logs_acc = page.get_by_text(re.compile("System Logs"))
    if logs_acc.is_visible():
        logs_acc.click()
        time.sleep(1)


class TestMainWorkflow:
    """Complete end-to-end workflow tests."""

    def test_full_user_flow(self, page: Page, app_server):
        """
        Tests the complete end-to-end workflow:
        Extraction -> Pre-Analysis -> Scene Selection -> Propagation -> Analysis -> Export
        """
        page.goto(BASE_URL)
        expect(page.get_by_text("Frame Extractor & Analyzer v2.0")).to_be_visible(timeout=30000)

        # 1. Frame Extraction
        print("Step 1: Frame Extraction")
        page.get_by_placeholder("Paste YouTube URL or local path").fill("dummy_video.mp4")
        
        # Select Every Nth for speed in mock (though it doesn't matter for mock)
        page.get_by_role("button", name="🚀 Start Extraction").click()
        time.sleep(1) # Wait for event loop to register click

        # Wait for "complete" in the status area (always visible)
        expect(page.locator("#unified_status")).to_contain_text("Extraction Complete", timeout=30000)
        # Also verify the success card is present
        expect(page.locator(".success-card")).to_be_visible()
        print("  ✓ Extraction Complete")

        # 2. Define Subject (Pre-Analysis)
        print("Step 2: Define Subject")
        screenshot_dir = Path(__file__).parent.parent / "results" / "screenshots"
        screenshot_dir.mkdir(parents=True, exist_ok=True)
        
        page.screenshot(path=str(screenshot_dir / "before_tab_switch.png"))
        switch_to_tab(page, "Subject")
        page.screenshot(path=str(screenshot_dir / "after_tab_switch.png"))
        
        # Ensure tab content is rendered - Gradio tabs render content lazily
        # Note: Can't use reload as it loses Gradio state (extraction results, etc)
        # Use tab switch to force re-render instead
        print("  - Waiting for tab content...")
        subject_content_visible = False
        
        # Wait extra time for Gradio lazy rendering
        page.wait_for_timeout(2000)
        
        try:
            # First attempt: wait for the Subject tab header
            page.wait_for_selector("text=Step 2: Define Subject", state="visible", timeout=8000)
            subject_content_visible = True
        except Exception:
            print("  - Subject content not visible, trying tab switch...")
            page.screenshot(path=str(screenshot_dir / "before_tab_retry.png"))
            
        # Fallback: switch tabs to force Gradio to re-render content
        if not subject_content_visible:
            switch_to_tab(page, "Source")  # Switch away from Subject
            page.wait_for_timeout(1000)
            switch_to_tab(page, "Subject")  # Switch back
            page.wait_for_timeout(2000)  # Wait for content to render
            page.screenshot(path=str(screenshot_dir / "after_tab_retry.png"))
            
            try:
                page.wait_for_selector("text=Step 2: Define Subject", state="visible", timeout=10000)
                subject_content_visible = True
            except Exception as e:
                page.screenshot(path=str(screenshot_dir / "timeout_after_retry.png"))
                print(f"  ❌ Subject tab content NOT visible even after tab switch. Error: {e}")
                raise
        
        print("  ✓ Subject tab content loaded")
        
        # Confirm and find scenes
        # Use ID for more robust targeting in Gradio 5
        btn = page.locator("#start_pre_analysis_button")
        print(f"  - Waiting for button: {btn}")
        try:
            btn.wait_for(state="visible", timeout=10000)
        except Exception as e:
            page.screenshot(path=str(screenshot_dir / "timeout_confirm_subject_v2.png"))
            print(f"  ❌ Timeout waiting for Confirm Subject button. Screenshot saved. Error: {e}")
            raise
        btn.click()
        
        expect(page.locator("body")).to_contain_text("Pre-Analysis Complete", timeout=30000)
        print("  ✓ Pre-Analysis Complete")

        # 3. Scene Selection & Propagation
        print("Step 3: Scene Selection")
        switch_to_tab(page, "Scenes")
        
        # Wait for Scenes tab content to load
        # Note: Can't reload here as it loses Gradio state (scenes, button enabled, etc)
        # Use longer wait and force click approach instead
        page.wait_for_timeout(2000)  # Allow extra time for Gradio to render
        
        prop_btn = page.get_by_role("button", name=re.compile("Propagate Masks", re.IGNORECASE))
        try:
            expect(prop_btn).to_be_enabled(timeout=10000)
        except Exception as e:
            print(f"  - Propagate button disabled, waiting longer...")
            page.screenshot(path=str(screenshot_dir / "prop_btn_disabled.png"))
            # Try clicking tab again to trigger re-render
            switch_to_tab(page, "Subject")  # Switch away
            page.wait_for_timeout(500)
            switch_to_tab(page, "Scenes")  # Switch back
            page.wait_for_timeout(2000)
            expect(prop_btn).to_be_enabled(timeout=15000)
        
        prop_btn.click()
        
        expect(page.locator("body")).to_contain_text("Mask Propagation Complete", timeout=30000)
        print("  ✓ Propagation Complete")

        # 4. Analysis
        print("Step 4: Metrics & Analysis")
        switch_to_tab(page, "Metrics")
        
        # Wait for Metrics tab content to load
        page.wait_for_timeout(2000)
        
        ana_btn = page.get_by_role("button", name=re.compile("Run Analysis", re.IGNORECASE))
        try:
            expect(ana_btn).to_be_visible(timeout=10000)
            expect(ana_btn).to_be_enabled(timeout=5000)
        except Exception:
            print("  - Run Analysis button not ready, trying tab switch...")
            page.screenshot(path=str(screenshot_dir / "before_metrics_retry.png"))
             # Switch away and back to force render
            switch_to_tab(page, "Scenes")
            page.wait_for_timeout(500)
            switch_to_tab(page, "Metrics")
            page.wait_for_timeout(2000)
            
            expect(ana_btn).to_be_visible(timeout=15000)
            expect(ana_btn).to_be_enabled(timeout=10000)

        ana_btn.click()
        
        expect(page.locator("body")).to_contain_text("Analysis Complete", timeout=30000)
        print("  ✓ Analysis Complete")

        # 5. Filtering & Export
        print("Step 5: Export")
        switch_to_tab(page, "Export")
        
        # Wait for Export tab content to load
        page.wait_for_timeout(2000)
        
        export_btn = page.get_by_role("button", name=re.compile("Export Kept Frames", re.IGNORECASE))
        try:
            expect(export_btn).to_be_visible(timeout=10000)
            expect(export_btn).to_be_enabled(timeout=5000)
        except Exception:
            print("  - Export button not ready, trying tab switch...")
            page.screenshot(path=str(screenshot_dir / "before_export_retry.png"))
            switch_to_tab(page, "Metrics")
            page.wait_for_timeout(500)
            switch_to_tab(page, "Export")
            page.wait_for_timeout(2000)
            
            expect(export_btn).to_be_visible(timeout=15000)
            expect(export_btn).to_be_enabled(timeout=10000)

        export_btn.click()
        
        print("  ✓ Export Clicked")
        print("E2E Flow Passed (Simulated)")


class TestTabNavigation:
    """Tests for tab navigation and UI responsiveness."""

    def test_all_tabs_accessible(self, page: Page, app_server):
        """Verify all main tabs can be accessed and show expected content."""
        page.goto(BASE_URL)
        time.sleep(2)

        tabs = ["Source", "Subject", "Scenes", "Metrics", "Export"]

        for tab_name in tabs:
            print(f"Checking tab: {tab_name}")
            tab = page.get_by_role("tab", name=tab_name)
            expect(tab).to_be_visible(timeout=10000)
            tab.click(force=True)
            expect(tab).to_have_attribute("aria-selected", "true", timeout=5000)
            print(f"  ✓ {tab_name} tab accessible")

    def test_tab_state_preserved(self, page: Page, app_server):
        """Verify tab state is preserved when switching tabs."""
        page.goto(BASE_URL)
        time.sleep(2)

        source_input = page.get_by_placeholder("Paste YouTube URL or local path")
        source_input.fill("test_video.mp4")

        switch_to_tab(page, "Subject")
        switch_to_tab(page, "Source")

        expect(source_input).to_have_value("test_video.mp4")


class TestErrorHandling:
    """Tests for error display and recovery."""

    def test_empty_source_shows_message(self, page: Page, app_server):
        """Verify appropriate message when no source is provided."""
        page.goto(BASE_URL)
        time.sleep(2)

        page.get_by_role("button", name=re.compile("Start Extraction")).click()
        time.sleep(1)

        # Check logs for Error
        open_logs(page)
        # Give mock app a moment to yield the error
        expect(page.locator("#unified_log")).to_contain_text("Error", timeout=10000)
