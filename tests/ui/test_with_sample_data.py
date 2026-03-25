"""
E2E tests using sample data to verify the full app workflow.
"""

from pathlib import Path

import pytest
from playwright.sync_api import Page, expect

from .conftest import BASE_URL, open_accordion, switch_to_tab, wait_for_app_ready
from .ui_locators import Labels, Selectors

# Mark all tests as e2e
pytestmark = pytest.mark.e2e

# Sample file paths (calculated relative to this file)
SAMPLE_VIDEO = str(Path(__file__).parent.parent / "assets" / "sample.mp4")
SAMPLE_IMAGE = str(Path(__file__).parent.parent / "assets" / "sample.jpg")


@pytest.fixture
def extracted_video_session(page: Page, app_server):
    """Fixture that brings the app to an 'extracted' state."""
    page.goto(BASE_URL)
    wait_for_app_ready(page)

    source_input = page.get_by_placeholder(Labels.SOURCE_PLACEHOLDER)
    if not source_input.is_visible():
        source_input = page.locator(Selectors.SOURCE_INPUT)

    source_input.fill(SAMPLE_VIDEO)
    page.locator(Selectors.START_EXTRACTION).click()

    expect(page.locator(Selectors.UNIFIED_STATUS)).to_contain_text("Extraction Complete", timeout=30000)
    return page


class TestSampleDataWorkflow:
    """Verifies the complete workflow using real sample data (mocked)."""

    def test_full_workflow_with_sample_video(self, page: Page, app_server):
        """Test extraction -> seeding -> analysis -> export with a mock sample."""
        page.goto(BASE_URL)
        wait_for_app_ready(page)

        # 1. Extraction (Source Tab)
        source_input = page.get_by_placeholder(Labels.SOURCE_PLACEHOLDER)
        if not source_input.is_visible():
            source_input = page.locator(Selectors.SOURCE_INPUT)

        source_input.fill(SAMPLE_VIDEO)

        # Start and wait for completion
        page.locator(Selectors.START_EXTRACTION).click()
        expect(page.locator(Selectors.UNIFIED_STATUS)).to_contain_text("Extraction Complete", timeout=30000)

        # 2. Seeding (Subject Tab)
        switch_to_tab(page, Labels.TAB_SUBJECT)

        # Choose Text strategy
        page.get_by_text(Labels.STRATEGY_TEXT, exact=False).click()
        page.get_by_placeholder("e.g., 'a man in a blue suit'").fill("protagonist")

        # Start pre-analysis
        page.locator(Selectors.START_PRE_ANALYSIS).click()
        expect(page.locator(Selectors.UNIFIED_STATUS)).to_contain_text("Pre-Analysis Complete", timeout=30000)

        # 3. Verification (Scenes Tab)
        switch_to_tab(page, Labels.TAB_SCENES)

        # Gallery should be visible and have items
        gallery = page.locator(Selectors.SCENE_GALLERY)
        expect(gallery).to_be_visible(timeout=10000)

        # Check for thumbnails
        expect(gallery.locator("button.thumbnail-item")).to_have_count(4, timeout=10000)

        # 4. Analysis (Metrics Tab)
        switch_to_tab(page, Labels.TAB_METRICS)

        # Run analysis
        page.locator(Selectors.START_ANALYSIS).click()
        expect(page.locator(Selectors.UNIFIED_STATUS)).to_contain_text("Analysis Complete", timeout=30000)

        # 5. Export (Export Tab)
        switch_to_tab(page, Labels.TAB_EXPORT)

        # Click Export
        page.locator(Selectors.EXPORT_BUTTON).click()
        expect(page.locator(Selectors.UNIFIED_STATUS)).to_contain_text("Export Complete", timeout=20000)

        # 6. Verify Log Trails
        open_accordion(page, Labels.SYSTEM_LOGS)
        log_content = page.locator(Selectors.LOG_TEXTAREA).input_value()

        assert "Extraction" in log_content
        assert "Pre-Analysis" in log_content
        assert "Analysis" in log_content
        assert "Export" in log_content


class TestSeedingOptions:
    """Tests different seeding options with sample data."""

    def test_face_upload_flow(self, page: Page, app_server):
        """Test uploading a reference face image."""
        page.goto(BASE_URL)
        wait_for_app_ready(page)

        switch_to_tab(page, Labels.TAB_SUBJECT)

        # Face strategy is default or click it
        page.get_by_text(Labels.STRATEGY_FACE, exact=False).click()

        # Find upload component by elem_id
        file_input = page.locator("#face_ref_img_upload_input input[type=file]")
        expect(file_input).to_be_attached(timeout=5000)
        file_input.set_input_files(SAMPLE_IMAGE)

        # Should show a preview or success
        expect(page.get_by_text("Upload Reference Photo", exact=False)).to_be_visible()

    def test_text_prompt_flow(self, page: Page, app_server):
        """Test entering a text prompt."""
        page.goto(BASE_URL)
        wait_for_app_ready(page)

        switch_to_tab(page, Labels.TAB_SUBJECT)

        # Select Text
        page.get_by_text(Labels.STRATEGY_TEXT, exact=False).click()

        # Fill prompt
        prompt = page.get_by_placeholder("e.g., 'a man in a blue suit'")
        prompt.fill("subject in red hat")

        expect(prompt).to_have_value("subject in red hat")
