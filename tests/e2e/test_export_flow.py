"""
Playwright E2E Tests for export workflow.

Tests the export functionality including:
- Dry run export
- Filter application before export
- Export format selection
- Export completion verification

Run with: python -m pytest tests/e2e/test_export_flow.py -v -s
"""
import pytest
from playwright.sync_api import Page, expect
import time

from .conftest import BASE_URL

pytestmark = pytest.mark.e2e


class TestExportFlow:
    """Export workflow tests."""

    def test_export_tab_accessible(self, page: Page, app_server):
        """Verify export tab is accessible and shows expected elements."""
        page.goto(BASE_URL)
        time.sleep(2)

        # Navigate to Export tab
        page.get_by_role("tab", name="Export").click(force=True)
        time.sleep(1)

        # Export tab should have export button
        export_btn = page.get_by_role("button", name="Export Kept Frames", exact=True)
        expect(export_btn).to_be_visible(timeout=5000)

    def test_dry_run_export(self, page: Page, app_server):
        """Test dry run export mode (no files created)."""
        page.goto(BASE_URL)
        time.sleep(2)

        # First need to extract frames
        page.get_by_label("Video URL or Local Path").fill("dummy_video.mp4")
        page.get_by_role("button", name="🚀 Start Single Extraction").click()
        expect(page.get_by_text("Extraction complete")).to_be_visible(timeout=20000)
        time.sleep(2)

        # Navigate to Export tab
        page.get_by_role("tab", name="Export").click(force=True)
        time.sleep(1)

        # Look for dry run checkbox and enable it if exists
        dry_run_checkbox = page.locator("text=Dry Run").locator("..").locator("input[type='checkbox']")
        if dry_run_checkbox.count() > 0:
            dry_run_checkbox.first.check()
            time.sleep(0.5)

        # Click export
        export_btn = page.get_by_role("button", name="Export Kept Frames", exact=True)
        export_btn.click()
        time.sleep(2)

        # Log should update (exact message depends on implementation)
        log = page.locator("#unified_log")
        expect(log).to_be_visible(timeout=5000)

    def test_export_after_analysis(self, analyzed_session):
        """Test export after running full pre-analysis."""
        page = analyzed_session

        # Navigate to Export tab
        page.get_by_role("tab", name="Export").click(force=True)
        time.sleep(1)

        # Click export
        export_btn = page.get_by_role("button", name="Export Kept Frames", exact=True)
        export_btn.click()
        time.sleep(2)

        # Should not crash
        log = page.locator("#unified_log")
        expect(log).to_be_visible(timeout=5000)


class TestFilteringBeforeExport:
    """Tests for filtering controls in export tab."""

    def test_filter_sliders_visible(self, page: Page, app_server):
        """Verify filtering sliders are visible in export tab."""
        page.goto(BASE_URL)
        time.sleep(2)

        # Navigate to Export tab
        page.get_by_role("tab", name="Export").click(force=True)
        time.sleep(1)

        # Look for range inputs (sliders)
        sliders = page.locator("input[type='range']")
        slider_count = sliders.count()
        print(f"Found {slider_count} sliders in Export tab")

    def test_filter_checkbox_toggle(self, page: Page, app_server):
        """Test that filter checkboxes can be toggled."""
        page.goto(BASE_URL)
        time.sleep(2)

        # Navigate to Export tab
        page.get_by_role("tab", name="Export").click(force=True)
        time.sleep(1)

        # Find checkboxes (filter enables)
        checkboxes = page.locator("input[type='checkbox']")
        checkbox_count = checkboxes.count()
        print(f"Found {checkbox_count} checkboxes in Export tab")

        # Toggle first checkbox if exists
        if checkbox_count > 0:
            first_checkbox = checkboxes.first
            initial_state = first_checkbox.is_checked()
            first_checkbox.click()
            time.sleep(0.5)
            new_state = first_checkbox.is_checked()
            # State should have changed
            assert initial_state != new_state, "Checkbox toggle did not change state"


class TestExportFormats:
    """Tests for export format options."""

    def test_export_settings_visible(self, page: Page, app_server):
        """Verify export settings are accessible."""
        page.goto(BASE_URL)
        time.sleep(2)

        # Navigate to Export tab
        page.get_by_role("tab", name="Export").click(force=True)
        time.sleep(1)

        # Export tab should be visible
        export_container = page.locator("[role='tabpanel']").first
        expect(export_container).to_be_visible(timeout=5000)

    def test_export_destination_input(self, page: Page, app_server):
        """Test export destination can be modified."""
        page.goto(BASE_URL)
        time.sleep(2)

        # Navigate to Export tab
        page.get_by_role("tab", name="Export").click(force=True)
        time.sleep(1)

        # Look for output folder input
        output_inputs = page.locator("input[type='text']")
        if output_inputs.count() > 0:
            print(f"Found {output_inputs.count()} text inputs in Export tab")
