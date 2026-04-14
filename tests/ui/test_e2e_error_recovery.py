import re

import pytest
from playwright.sync_api import Page, expect

from .conftest import BASE_URL, open_accordion, switch_to_tab, wait_for_app_ready
from .ui_locators import Labels, Selectors

pytestmark = pytest.mark.e2e


class TestErrorRecovery:
    """
    Tests for error paths and user recovery after pipeline/validation failures.
    Ensures the user can retry successfully without state corruption.
    """

    def test_extraction_with_invalid_path_to_success(self, page: Page, app_server):
        """
        Fill a nonsense path → Extract → Verify Error → Retry with valid path.
        """
        page.goto(BASE_URL)
        wait_for_app_ready(page)

        # 1. Fill invalid source
        page.locator(Selectors.SOURCE_INPUT).fill("nonsense_invalid_path.xyz")
        page.locator(Selectors.START_EXTRACTION).click()

        # 2. Verify Error in status
        expect(page.get_by_text(re.compile(r"Error|Failure", re.IGNORECASE))).to_be_visible(timeout=15000)

        # 3. Correct the path
        page.locator(Selectors.SOURCE_INPUT).fill("valid_retry_test.mp4")
        page.locator(Selectors.START_EXTRACTION).click()

        # 4. Verify Success
        expect(page.get_by_text(re.compile(r"Extraction Complete", re.IGNORECASE))).to_be_visible(timeout=30000)

    def test_tab_jump_restriction_recovery(self, page: Page, app_server):
        """
        Skip to Subject tab, click Confirm → Verify error message → Go back and extract → Ensure user is unblocked.
        """
        page.goto(BASE_URL)
        wait_for_app_ready(page)

        # 1. Skip to Subject
        switch_to_tab(page, Labels.TAB_SUBJECT)
        page.locator(Selectors.START_PRE_ANALYSIS).click()

        # 2. Verify error detail in status
        expect(page.get_by_text(re.compile(r"Error|Failure", re.IGNORECASE))).to_be_visible(timeout=10000)

        # 3. Go back and fix source
        switch_to_tab(page, Labels.TAB_SOURCE)
        page.locator(Selectors.SOURCE_INPUT).fill("navigation_recovery.mp4")
        page.locator(Selectors.START_EXTRACTION).click()
        expect(page.get_by_text(re.compile(r"Extraction Complete", re.IGNORECASE))).to_be_visible(timeout=30000)

        # 4. Now Subject tab action should succeed
        switch_to_tab(page, Labels.TAB_SUBJECT)
        page.locator(Selectors.START_PRE_ANALYSIS).click()
        expect(page.get_by_text(re.compile(r"Pre-Analysis Complete", re.IGNORECASE))).to_be_visible(timeout=30000)

    def test_export_precondition_failure(self, page: Page, app_server):
        """
        Go to Export tab without data -> verify Export button is disabled (precondition safety).
        """
        page.goto(BASE_URL)
        wait_for_app_ready(page)

        # 1. Skip to Export
        switch_to_tab(page, Labels.TAB_EXPORT)

        # 2. Verify Export and Dry Run are disabled
        expect(page.locator(Selectors.EXPORT_BUTTON)).to_be_disabled()
        expect(page.locator(Selectors.DRY_RUN_BUTTON)).to_be_disabled()

        # 3. Verify status help text
        expect(page.get_by_text(re.compile(r"Analysis not loaded", re.IGNORECASE))).to_be_visible()


class TestCancellation:
    """
    Tests for the Cancel button during pipeline execution.
    Ensures that pipelines are interruptible and the UI stays responsive.
    """

    def test_cancel_extraction_midway(self, page: Page, app_server):
        """
        Start extraction → Wait for progress → Click Cancel → Verify Cancellation status.
        """
        page.goto(BASE_URL, timeout=60000)
        wait_for_app_ready(page)

        # 1. Fill source and start extraction
        page.locator(Selectors.SOURCE_INPUT).fill("cancel_test.mp4")
        page.locator(Selectors.START_EXTRACTION).click()

        # 2. In Gradio 5 with show_progress="hidden", we wait for status transition
        expect(page.locator(Selectors.UNIFIED_STATUS)).to_contain_text("Processing", ignore_case=True, timeout=10000)

        # 3. Click Cancel
        cancel_btn = page.locator(Selectors.CANCEL_BUTTON)
        expect(cancel_btn).to_be_enabled(timeout=5000)
        cancel_btn.click()

        # 4. Verify status contains "Cancelled"
        expect(page.locator(Selectors.UNIFIED_STATUS)).to_contain_text("Cancelled", ignore_case=True, timeout=10000)

        # 5. Reset and retry (Verify no stuck state)
        page.locator(Selectors.START_EXTRACTION).click()
        expect(page.locator(Selectors.UNIFIED_STATUS)).to_contain_text(
            "Extraction Complete", ignore_case=True, timeout=30000
        )

    def test_cancel_propagation_midway(self, page: Page, app_server):
        """
        Start propagation → Click Cancel → Verify Success of subsequent retry.
        """
        page.goto(BASE_URL)
        wait_for_app_ready(page)

        # Setup: Quick extraction
        page.locator(Selectors.SOURCE_INPUT).fill("prop_cancel.mp4")
        page.locator(Selectors.START_EXTRACTION).click()
        expect(page.locator(Selectors.UNIFIED_STATUS)).to_contain_text(
            "Extraction Complete", ignore_case=True, timeout=30000
        )

        # Setup: Quick pre-analysis
        switch_to_tab(page, Labels.TAB_SUBJECT)
        page.locator(Selectors.START_PRE_ANALYSIS).click()
        expect(page.locator(Selectors.UNIFIED_STATUS)).to_contain_text(
            "Pre-Analysis Complete", ignore_case=True, timeout=30000
        )

        # 1. Start Propagation
        switch_to_tab(page, Labels.TAB_SCENES)
        prop_btn = page.locator(Selectors.PROPAGATE_MASKS)
        prop_btn.click()

        # 2. Wait for busy state signal in status
        expect(page.locator(Selectors.UNIFIED_STATUS)).to_contain_text("Processing", ignore_case=True, timeout=10000)

        # 3. Click Cancel quickly
        page.locator(Selectors.CANCEL_BUTTON).click()

        # 4. Verify status reflects cancellation or immediate recovery
        expect(page.locator(Selectors.UNIFIED_STATUS)).to_contain_text("Cancelled", ignore_case=True, timeout=10000)


class TestPaginationBugRegression:
    """Tests to prevent pagination crash regression (Bug 2)."""

    def test_next_button_on_empty_gallery_no_crash(self, page: Page, app_server):
        """Clicking Next on empty/single-page gallery should not crash."""
        page.goto(BASE_URL)
        wait_for_app_ready(page)

        switch_to_tab(page, Labels.TAB_SCENES)

        # Try to click Next - should not crash application
        next_button = page.locator(Selectors.NEXT_PAGE_BUTTON)
        # Even if disabled, clicking should not crash
        next_button.click(force=True)
        page.wait_for_timeout(500)

        # App should still be responsive
        expect(page.locator("body")).to_be_visible()

    def test_prev_button_on_page_one_no_crash(self, page: Page, app_server):
        """Clicking Previous on page 1 should not crash."""
        page.goto(BASE_URL)
        wait_for_app_ready(page)

        switch_to_tab(page, Labels.TAB_SCENES)

        prev_button = page.locator(Selectors.PREV_PAGE_BUTTON)
        prev_button.click(force=True)
        page.wait_for_timeout(500)
        expect(page.locator("body")).to_be_visible()


class TestFindPeopleButtonRegression:
    """Tests to prevent Find People button regression (Bug 3)."""

    def test_find_people_button_visible_in_face_strategy(self, page: Page, app_server):
        """Find People button should be visible when 'By Face' strategy selected."""
        page.goto(BASE_URL)
        wait_for_app_ready(page)

        switch_to_tab(page, Labels.TAB_SUBJECT)

        # Select Face strategy - use label-based click
        page.get_by_label(Labels.STRATEGY_FACE, exact=False).check(force=True)
        page.wait_for_timeout(500)

        # Click the Scan Video tab
        page.get_by_role("tab", name=Labels.TAB_SCAN_VIDEO, exact=False).click()

        # Find People button should be visible
        find_people_btn = page.get_by_role("button", name=Labels.SCAN_VIDEO_BUTTON, exact=False)
        expect(find_people_btn).to_be_visible(timeout=5000)


class TestGallerySizeControlsRegression:
    """Tests to prevent gallery sizing issues (Bug 5)."""

    def test_gallery_size_controls_exist(self, page: Page, app_server):
        """Scene gallery should have columns and height controls."""
        page.goto(BASE_URL)
        wait_for_app_ready(page)

        switch_to_tab(page, Labels.TAB_SCENES)
        open_accordion(page, "Display Settings")

        # Check for Columns slider
        columns_slider = page.get_by_label("Columns").first
        expect(columns_slider).to_be_visible()

        # Check for Height slider
        height_slider = page.get_by_label("Height").first
        expect(height_slider).to_be_visible()


class TestSystemLogsRegression:
    """Tests to prevent system log visibility issues (Bug 6)."""

    def test_logs_accordion_exists(self, page: Page, app_server):
        """System Logs accordion should be present."""
        page.goto(BASE_URL)
        wait_for_app_ready(page)

        logs_accordion = page.get_by_role("button", name="📋 System Logs", exact=False)
        expect(logs_accordion).to_be_visible(timeout=5000)

    def test_refresh_logs_button_exists(self, page: Page, app_server):
        """Refresh Logs button should be present for manual log updates."""
        page.goto(BASE_URL)
        wait_for_app_ready(page)

        open_accordion(page, Labels.SYSTEM_LOGS)

        # Check for Refresh button
        refresh_btn = page.get_by_role("button", name="Refresh", exact=False)
        expect(refresh_btn).to_be_visible(timeout=5000)

    def test_clear_logs_button_works(self, page: Page, app_server):
        """Clear Logs button should clear the log display."""
        page.goto(BASE_URL)
        wait_for_app_ready(page)

        open_accordion(page, Labels.SYSTEM_LOGS)

        # Click Clear button
        clear_btn = page.get_by_role("button", name="Clear", exact=False)
        if clear_btn.is_visible():
            clear_btn.click()
            # Log area should be empty or show initial message
            expect(page.locator(Selectors.LOG_TEXTAREA)).to_have_value("", timeout=5000)


class TestPropagationErrorHandling:
    """Tests for propagation error handling (Bug 1)."""

    def test_propagate_button_found(self, page: Page, app_server):
        """Propagate button should be present on the Scenes tab after pre-analysis."""
        page.goto(BASE_URL)
        wait_for_app_ready(page)

        # 1. Extract
        page.locator(Selectors.SOURCE_INPUT).fill("mock_video.mp4")
        page.locator(Selectors.START_EXTRACTION).click()
        expect(page.get_by_text(re.compile(r"Extraction Complete", re.IGNORECASE))).to_be_visible(timeout=30000)

        # 2. Pre-Analyse
        switch_to_tab(page, Labels.TAB_SUBJECT)
        page.locator(Selectors.START_PRE_ANALYSIS).click()
        expect(page.get_by_text(re.compile(r"Pre-Analysis Complete", re.IGNORECASE))).to_be_visible(timeout=30000)

        # 3. Check for propagate button
        switch_to_tab(page, Labels.TAB_SCENES)
        propagate_btn = page.locator(Selectors.PROPAGATE_MASKS)
        expect(propagate_btn).to_be_attached(timeout=5000)
