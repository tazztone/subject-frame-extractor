import pytest
from playwright.sync_api import expect

from .app_driver import AppDriver
from .ui_locators import Labels, Selectors

# Mark as e2e test
pytestmark = pytest.mark.e2e


class TestWorkflowVariants:
    """
    E2E tests for different input variants (Image Folders vs Videos).
    """

    def test_image_folder_workflow_skips_propagation(self, app_driver: AppDriver):
        """Verify that selecting a folder of images skips propagation step."""
        # 1-3. Select 'All Frames' method, fill folder path, start extraction
        app_driver.extract("/home/user/pics", method="All Frames (Maximum Quality)").expect_status(
            Selectors.STATUS_SUCCESS_EXTRACTION, timeout=15000
        )

        # 4. Confirm Subject (Pre-Analysis)
        app_driver.navigate(Labels.TAB_SUBJECT).pre_analyze().expect_status(
            Selectors.STATUS_SUCCESS_PRE_ANALYSIS, timeout=15000
        )

        # 5. Verify Propagation button IS HIDDEN (Images don't need propagation)
        expect(app_driver.page.locator(Selectors.PROPAGATE_MASKS)).not_to_be_visible()

        # 6. Verify "Proceed to Metrics" hint in logs
        app_driver.expect_log("Propagation is not needed for image folders")

    def test_video_workflow_shows_propagation(self, app_driver: AppDriver):
        """Verify that selecting a video shows propagation step."""
        # 1-3. Select 'Scene-based', fill video path, start extraction
        app_driver.extract("mock_video.mp4", method="Scene-based").expect_status(
            Selectors.STATUS_SUCCESS_EXTRACTION, timeout=15000
        )

        # 4. Confirm Subject
        app_driver.navigate(Labels.TAB_SUBJECT).pre_analyze().expect_status(
            Selectors.STATUS_SUCCESS_PRE_ANALYSIS, timeout=15000
        )

        # 5. Verify Propagation button IS VISIBLE
        app_driver.navigate(Labels.TAB_SCENES)
        app_driver.expect_visible(Selectors.PROPAGATE_MASKS)
