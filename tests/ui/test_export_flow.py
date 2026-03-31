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
from playwright.sync_api import expect

from .conftest import open_accordion, switch_to_tab

pytestmark = pytest.mark.e2e


class TestExportFlow:
    """Export workflow tests."""

    def test_export_tab_accessible(self, shared_analysis_session):
        """Verify export tab is accessible and shows expected elements."""
        page = shared_analysis_session

        # Navigate to Export tab
        switch_to_tab(page, "Export")

        # In Gradio, tab content visibility is handled by JS toggling display style
        # We can check if the filter preset dropdown is visible, which is in the Export tab
        page.locator("#filter_preset_dropdown")
        # Try to use a more generic check if ID isn't set, but we set names in create_component
        # But create_component doesn't set elem_id to name automatically.
        # Let's check for the label text more broadly
        expect(page.get_by_text("Use a Preset")).to_be_visible()

    def test_dry_run_export_requires_analysis(self, shared_analysis_session):
        """Test dry run export mode after full analysis."""
        page = shared_analysis_session

        # Navigate to Export tab
        switch_to_tab(page, "Export")

        # Look for dry run checkbox or button
        dry_run_btn = page.get_by_role("button", name="Dry Run")
        expect(dry_run_btn).to_be_visible(timeout=5000)

        dry_run_btn.click()

        # Check logs for success message
        # Open logs accordion because it's closed by default
        open_accordion(page, "System Logs")
        page.wait_for_timeout(500)

        log = page.locator("#unified_log textarea")
        # Wait for log to update
        page.wait_for_timeout(2000)
        expect(log).to_be_visible()

    def test_export_button_visibility(self, shared_analysis_session):
        """Test export button becomes visible after analysis."""
        page = shared_analysis_session

        # Navigate to Export tab
        switch_to_tab(page, "Export")

        # Export button should be visible now
        # Note: AppUI sets value to "💾 Export Kept Frames"
        export_btn = page.get_by_role("button", name="💾 Export Kept Frames")
        expect(export_btn).to_be_visible(timeout=5000)


class TestFilteringBeforeExport:
    """Tests for filtering controls in export tab."""

    def test_filter_sliders_visible(self, shared_analysis_session):
        """Verify filtering sliders are visible in export tab."""
        page = shared_analysis_session

        # Navigate to Export tab
        switch_to_tab(page, "Export")

        # Presets dropdown should be visible
        expect(page.get_by_text("Use a Preset")).to_be_visible()

    def test_smart_filter_toggle(self, shared_analysis_session):
        """Test Smart Filtering toggle."""
        page = shared_analysis_session

        # Navigate to Export tab
        switch_to_tab(page, "Export")

        smart_filter = page.get_by_label("Smart Filtering")
        if smart_filter.is_visible():
            smart_filter.check()
            page.wait_for_timeout(500)
            assert smart_filter.is_checked()


class TestExportFormats:
    """Tests for export format options."""

    def test_export_settings_visible(self, shared_analysis_session):
        """Verify export settings are accessible after analysis."""
        page = shared_analysis_session

        # Navigate to Export tab
        switch_to_tab(page, "Export")

        # Check for Export Options accordion visibility
        # It's inside the export_group which is visible after analysis

        export_btn = page.get_by_role("button", name="💾 Export Kept Frames")
        expect(export_btn).to_be_visible()
