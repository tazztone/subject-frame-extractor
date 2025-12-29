
import os
import re
import shutil
import time
from pathlib import Path
import pytest
from playwright.sync_api import expect, Page

# Use the shared app_server fixture from conftest.py
# Constants
PORT = 7860
BASE_URL = f"http://127.0.0.1:{PORT}"

def switch_to_tab(page: Page, tab_name: str):
    """Robustly switch tabs in Gradio."""
    # Click the tab button
    tab_btn = page.get_by_role("tab", name=tab_name)
    expect(tab_btn).to_be_visible()
    tab_btn.click(force=True)

    # Wait for the tab to be selected - checking class using regex
    expect(tab_btn).to_have_class(re.compile(r"selected|active"))

    # Wait for a brief moment for content animation
    time.sleep(1.0)

class TestWorkflowCompleteness:
    """
    Comprehensive E2E tests for the full workflow, hitting advanced paths
    to increase coverage of app_ui.py.
    """

    def test_full_advanced_workflow(self, page: Page, app_server):
        """
        Tests the complete workflow from extraction to export with advanced settings.
        This hits many conditional branches in app_ui.py.
        """
        page.goto(BASE_URL)
        expect(page).to_have_title(re.compile("Frame Extractor"))

        # --- Step 1: Extraction with Advanced Settings ---
        # Select "Time Interval" method
        # Use placeholder as label is hidden
        page.get_by_placeholder("Paste YouTube URL or local path").fill("dummy_video.mp4")

        # Toggle Advanced Settings
        page.get_by_text("🔧 Advanced Processing Settings").click()

        # Change Extraction Method to "Time Interval"
        page.get_by_label("Extraction Method").click()
        page.get_by_role("option", name="Time Interval").click()

        # Verify interval input appears
        interval_input = page.get_by_label("Interval (seconds)")
        expect(interval_input).to_be_visible()
        interval_input.fill("2.0")

        # Start Extraction
        page.get_by_role("button", name="🚀 Start Extraction").click()
        expect(page.get_by_text("Frame Extraction Complete")).to_be_visible(timeout=30000)

        # --- Step 2: Subject Definition (Advanced) ---
        switch_to_tab(page, "Subject")

        # The button name might be slightly different or contain markup/icons
        # Use regex to be more flexible
        find_btn = page.get_by_role("button", name=re.compile("Find & Preview Scenes"))
        expect(find_btn).to_be_visible()

        # Expand Advanced Model Configuration
        # Use a more specific selector if possible, or force click
        adv_config = page.get_by_text("🧠 Advanced Model Configuration")
        expect(adv_config).to_be_visible()
        adv_config.click()

        # Change Tracker Model
        page.get_by_label("Tracking Model").click()
        expect(page.get_by_role("option", name="sam3")).to_be_visible()
        page.get_by_role("option", name="sam3").click()

        # Run Pre-Analysis
        find_btn.click()
        expect(page.locator("#unified_status")).to_contain_text("Pre-Analysis Complete", timeout=30000)

        # --- Step 3: Scene Review (Bulk Actions) ---
        switch_to_tab(page, "Scenes")

        # Wait for Scenes content
        expect(page.get_by_text("Scene Overview")).to_be_visible()

        # Test "Batch Filter Scenes" accordion
        page.get_by_text("🔍 Batch Filter Scenes").click()
        expect(page.get_by_label("Min Subject Size (%)")).to_be_visible()

        # Trigger "Propagate Masks"
        page.get_by_role("button", name="⚡ Propagate Masks").click()
        expect(page.locator("#unified_status")).to_contain_text("Mask Propagation Complete", timeout=30000)

        # --- Step 4: Metrics (Toggle All) ---
        switch_to_tab(page, "Metrics")

        # Wait for Metrics content
        expect(page.get_by_text("Step 4: Analysis Metrics")).to_be_visible()

        # Expand Advanced Metrics
        page.get_by_text("🔧 Advanced / Legacy Metrics").click()

        # Toggle a few checkboxes
        page.get_by_label("Edge Strength").check()
        page.get_by_label("Contrast").check()

        # Run Analysis
        page.get_by_role("button", name="⚡ Run Analysis").click()
        expect(page.locator("#unified_status")).to_contain_text("Analysis Complete", timeout=30000)

        # --- Step 5: Export (Advanced) ---
        switch_to_tab(page, "Export")

        # Wait for Export content
        expect(page.get_by_text("Step 5: Filter & Export")).to_be_visible()

        # Open Deduplication Accordion
        expect(page.get_by_text("👯 Deduplication (Remove Duplicates)")).to_be_visible()

        # Change Deduplication Method
        page.get_by_label("Method").click()
        page.get_by_role("option", name="Off").click()

        # Open Advanced Export Options
        page.get_by_text("Advanced Export Options").click()
        page.get_by_label("✂️ Crop to Subject").check()

        # Dry Run
        page.get_by_role("button", name="Dry Run").click()

        # Check logs - open Accordion first
        page.get_by_text("📋 System Logs").click()
        # Wait for animation
        time.sleep(1.0)
        # Check text in the textarea inside the unified_log container
        # Since interactive=False, it might be a div or textarea.
        # We can check #unified_log text content directly if it's a container.
        expect(page.locator("#unified_log")).to_contain_text("Dry Run Export Complete", timeout=10000)

    def test_system_diagnostics(self, page: Page, app_server):
        """Tests the system diagnostics feature which runs a full dry-run pipeline."""
        page.goto(BASE_URL)

        # Open Help Accordion
        page.get_by_text("❓ Help / Troubleshooting").click()

        # Click Diagnostics
        page.get_by_role("button", name="Run System Diagnostics").click()

        # Open System Logs to see output
        page.get_by_text("📋 System Logs").click()
        # Wait for animation
        time.sleep(1.0)

        # Wait for completion and check logs
        # We target the textarea specifically to avoid "Textbox" label issues
        # But if it's empty, maybe it hasn't updated yet.
        # The container usually has the text.
        log_container = page.locator("#unified_log")
        expect(log_container).to_contain_text("System Diagnostics Report", timeout=60000)
        expect(log_container).to_contain_text("[SECTION 1: System & Environment]")
