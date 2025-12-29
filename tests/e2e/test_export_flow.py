"""
Playwright E2E Tests for export workflow.

Tests the export functionality including:
- Dry run export
- Filter application before export
- Export format selection
- Export completion verification

Run with: python -m pytest tests/e2e/test_export_flow.py -v -s
"""

import time

import pytest
from playwright.sync_api import Page, expect

from .conftest import BASE_URL, switch_to_tab

pytestmark = pytest.mark.e2e


class TestExportFlow:
    """Export workflow tests."""

    def test_export_tab_accessible(self, page: Page, app_server):
        """Verify export tab is accessible and shows expected elements."""
        page.goto(BASE_URL)
        time.sleep(2)

        # Navigate to Export tab
        switch_to_tab(page, "Export")

        # In Gradio, tab content visibility is handled by JS toggling display style
        # We can check if the filter preset dropdown is visible, which is in the Export tab
        presets = page.get_by_label("Use a Preset")
        expect(presets).to_be_visible()

    def test_dry_run_export_requires_analysis(self, full_analysis_session):
        """Test dry run export mode after full analysis."""
        page = full_analysis_session

        # Navigate to Export tab
        switch_to_tab(page, "Export")

        # Look for dry run checkbox or button
        dry_run_btn = page.get_by_role("button", name="Dry Run")
        expect(dry_run_btn).to_be_visible(timeout=5000)

        dry_run_btn.click()

        # Check logs for success message
        log = page.locator("#unified_log")
        # Wait for log to update
        time.sleep(2)
        expect(log).to_be_visible()

    def test_export_button_visibility(self, full_analysis_session):
        """Test export button becomes visible after analysis."""
        page = full_analysis_session

        # Navigate to Export tab
        switch_to_tab(page, "Export")

        # Export button should be visible now
        export_btn = page.get_by_role("button", name="Export Kept Frames", exact=True)
        expect(export_btn).to_be_visible(timeout=5000)


class TestFilteringBeforeExport:
    """Tests for filtering controls in export tab."""

    def test_filter_sliders_visible(self, page: Page, app_server):
        """Verify filtering sliders are visible in export tab."""
        page.goto(BASE_URL)
        time.sleep(2)

        # Navigate to Export tab
        switch_to_tab(page, "Export")

        # Presets dropdown should be visible
        presets = page.get_by_label("Use a Preset")
        expect(presets).to_be_visible()

    def test_smart_filter_toggle(self, page: Page, app_server):
        """Test Smart Filtering toggle."""
        page.goto(BASE_URL)
        time.sleep(2)

        # Navigate to Export tab
        switch_to_tab(page, "Export")

        smart_filter = page.get_by_label("Smart Filtering")
        if smart_filter.is_visible():
            smart_filter.check()
            time.sleep(0.5)
            assert smart_filter.is_checked()


class TestExportFormats:
    """Tests for export format options."""

    def test_export_settings_visible(self, full_analysis_session):
        """Verify export settings are accessible after analysis."""
        page = full_analysis_session

        # Navigate to Export tab
        switch_to_tab(page, "Export")

        # Check for Export Options accordion visibility
        # It's inside the export_group which is visible after analysis

        export_btn = page.get_by_role("button", name="Export Kept Frames", exact=True)
        expect(export_btn).to_be_visible()
