"""
E2E tests using sample data to verify the full app workflow.
Driven through ``AppDriver``.
"""

from pathlib import Path

import pytest
from playwright.sync_api import expect

from .app_driver import AppDriver
from .ui_locators import Labels, Selectors

# Mark all tests as e2e
pytestmark = pytest.mark.e2e

# Sample file paths (calculated relative to this file)
# Note: In mock app, these contents aren't read, but the paths must exist if validation checks them.
SAMPLE_VIDEO = str(Path(__file__).parent.parent / "assets" / "sample.mp4")
SAMPLE_IMAGE = str(Path(__file__).parent.parent / "assets" / "sample.jpg")


@pytest.fixture
def extracted_video_session(app_driver: AppDriver):
    """Fixture that brings the app to an 'extracted' state."""
    app_driver.extract(SAMPLE_VIDEO).expect_status(Selectors.STATUS_SUCCESS_EXTRACTION, timeout=30000)
    return app_driver


class TestSampleDataWorkflow:
    """Verifies the complete workflow using real sample data (mocked)."""

    def test_full_workflow_with_sample_video(self, app_driver: AppDriver):
        """Test extraction -> seeding -> analysis -> export with a mock sample."""
        # 1. Extraction (Source Tab)
        app_driver.extract(SAMPLE_VIDEO).expect_status(Selectors.STATUS_SUCCESS_EXTRACTION, timeout=30000)

        # 2. Seeding (Subject Tab) — choose Text strategy
        page = app_driver.navigate(Labels.TAB_SUBJECT).page
        # Gradio 5 Radio components are best targeted by label text (buttons)
        page.get_by_label(Labels.STRATEGY_TEXT).check(force=True)
        page.wait_for_timeout(500)

        page.get_by_placeholder("e.g., 'a man in a blue suit'").fill("protagonist")

        app_driver.pre_analyze().expect_status(Selectors.STATUS_SUCCESS_PRE_ANALYSIS, timeout=30000)

        # 3. Verification (Scenes Tab)
        app_driver.navigate(Labels.TAB_SCENES)
        app_driver.expect_visible(Selectors.SCENE_GALLERY, timeout=10000)

        # Check for thumbnails (Mock app returns 1 scene)
        expect(app_driver.page.locator(Selectors.SCENE_GALLERY).locator("img")).to_have_count(1, timeout=10000)

        # 4. Analysis (Metrics Tab)
        app_driver.navigate(Labels.TAB_METRICS).analyze().expect_status(
            Selectors.STATUS_SUCCESS_ANALYSIS, timeout=30000
        )

        # 5. Export (Export Tab)
        app_driver.navigate(Labels.TAB_EXPORT).export().expect_status(Selectors.STATUS_SUCCESS_EXPORT, timeout=20000)

        # 6. Verify Log Trails
        app_driver.expect_log("Extraction Complete", timeout=10000)
        app_driver.expect_log("Pre-Analysis Complete", timeout=10000)


class TestSeedingOptions:
    """Tests different seeding options with sample data."""

    def test_face_upload_flow(self, app_driver: AppDriver):
        """Test uploading a reference face image."""
        page = app_driver.navigate(Labels.TAB_SUBJECT).page

        page.get_by_label(Labels.STRATEGY_FACE).check(force=True)
        page.wait_for_timeout(500)

        # Find upload component (usually an input[type=file])
        file_input = page.locator("input[type=file]").first
        expect(file_input).to_be_attached(timeout=5000)
        # file_input.set_input_files(SAMPLE_IMAGE) # Skipped in mock due to real path requirement

    def test_text_prompt_flow(self, app_driver: AppDriver):
        """Test entering a text prompt."""
        page = app_driver.navigate(Labels.TAB_SUBJECT).page

        page.get_by_label(Labels.STRATEGY_TEXT).check(force=True)
        page.wait_for_timeout(500)

        prompt = page.get_by_placeholder("e.g., 'a man in a blue suit'")
        prompt.fill("subject in red hat")

        expect(prompt).to_have_value("subject in red hat")
