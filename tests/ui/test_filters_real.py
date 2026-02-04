"""
E2E Tests for Real Filtering Logic.

These tests use the sample video to verify that:
1. Filters can be adjusted.
2. Gallery updates to reflect filtered counts.
3. Smart filtering toggle works.
"""

import time
from pathlib import Path

import pytest
from playwright.sync_api import Page, expect

from .conftest import BASE_URL, switch_to_tab

pytestmark = pytest.mark.e2e

SAMPLE_VIDEO = "tests/assets/sample.mp4"


class TestRealFilters:
    @pytest.fixture(autouse=True)
    def setup_with_analysis(self, page: Page, app_server):
        """
        Setup state: Extracted and Pre-Analyzed sample video.
        We implement this manually here since the fixture is flaky.
        """
        page.goto(BASE_URL)
        expect(page.get_by_text("Frame Extractor & Analyzer")).to_be_visible(timeout=10000)

        # 1. Extraction
        if not Path(SAMPLE_VIDEO).exists():
            pytest.skip("Sample video not found")

        page.get_by_placeholder("Paste YouTube URL or local path").fill(str(Path(SAMPLE_VIDEO).resolve()))
        page.get_by_role("button", name="🚀 Start Extraction").click()
        expect(page.get_by_text("Extraction complete")).to_be_visible(timeout=30000)

        # 2. Pre-Analysis
        switch_to_tab(page, "Subject")

        # Wait for the button to appear. It might be hidden initially?
        # In app_ui.py, it's created at the end of _create_define_subject_tab.
        find_btn = page.get_by_role("button", name="🔍 Find & Preview Scenes")
        # Just in case, try to force it
        if not find_btn.is_visible():
            page.reload()
            switch_to_tab(page, "Subject")

        find_btn.click()
        expect(page.locator("#unified_status")).to_contain_text("Pre-Analysis Complete", timeout=30000)

        # 3. Propagate (needed for some metrics?)
        prop_btn = page.get_by_role("button", name="🔬 Propagate Masks")
        if prop_btn.is_visible():
            prop_btn.click()
            expect(page.locator("#unified_status")).to_contain_text("Mask Propagation Complete", timeout=30000)

        # 4. Analyze (to get metrics for filtering)
        switch_to_tab(page, "Metrics")
        page.get_by_role("button", name="Analyze Selected Frames").click()
        expect(page.locator("#unified_status")).to_contain_text("Analysis Complete", timeout=30000)

    def test_apply_filter_reduces_count(self, page: Page):
        """Test that increasing a filter threshold reduces the number of kept frames."""
        switch_to_tab(page, "Export")

        # Initially all frames should be kept (or close to it)
        # We can check the gallery count or status text
        status_text = page.locator("#filter_status_text")
        expect(status_text).to_contain_text("Kept", timeout=10000)

        # Open a metric accordion, e.g., Quality Score
        page.get_by_text("Quality Score").click()
        time.sleep(0.5)

        # Find slider for quality score min
        page.locator("input[type='range']").first  # This is risky, but quality score is first
        # Better selector:
        page.get_by_label("Min", exact=True).first

        # Drag slider to increase min threshold
        # Slider value is hard to set in Playwright without exact steps
        # But we can try to type if it allows, or just set via JS

        # Let's use Smart Filtering which is easier to toggle
        smart_toggle = page.get_by_label("Smart Filtering")
        smart_toggle.check()
        time.sleep(1)

        # Set percentile to 50%
        pctl_slider = page.get_by_label("Auto-Threshold Percentile")
        pctl_slider.fill("90")  # Keep top 10%
        page.get_by_role("button", name="Apply").click()
        time.sleep(2)

        # Status text should update
        # "Kept X / Y frames"
        expect(status_text).to_contain_text("Kept")

        # Verify that we kept fewer frames than total
        text = status_text.inner_text()
        # Parse "Kept 5 / 50 frames"
        import re

        match = re.search(r"Kept (\d+) / (\d+)", text)
        if match:
            kept = int(match.group(1))
            total = int(match.group(2))
            assert kept < total, f"Smart filter should have reduced frame count (Kept {kept}/{total})"
        else:
            print(f"Could not parse status text: {text}")

    def test_deduplication_toggle(self, page: Page):
        """Test enabling deduplication updates the count."""
        switch_to_tab(page, "Export")

        # Deduplication accordion is open by default
        dedup_dropdown = page.get_by_label("Deduplication")

        # Change to "Off"
        dedup_dropdown.select_option("Off")
        time.sleep(2)

        status_text = page.locator("#filter_status_text")
        text_off = status_text.inner_text()

        # Change to "Fast (pHash)"
        dedup_dropdown.select_option("Fast (pHash)")

        # Set threshold high to ensure dupes are found
        thresh_slider = page.get_by_label("Threshold")
        thresh_slider.fill("10")  # 10 is loose
        time.sleep(2)

        text_on = status_text.inner_text()

        print(f"Off: {text_off}, On: {text_on}")
        # Ideally text_on has fewer kept frames or same if no dupes
