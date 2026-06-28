import pytest

from .app_driver import AppDriver
from .ui_locators import Labels, Selectors

# Mark as e2e test
pytestmark = pytest.mark.e2e


class TestCancellation:
    """
    Tests for the Cancel button during pipeline execution.
    Ensures that pipelines are interruptible and the UI stays responsive.
    """

    def test_cancel_extraction_midway(self, app_driver: AppDriver):
        """
        Start extraction → Wait for progress → Click Cancel → Verify Cancellation status.
        """
        # 1. Fill source and start extraction
        app_driver.extract("cancel_test.mp4")

        # 2. Wait for it to definitely be running (mock_app uses 'Mock Extraction')
        app_driver.expect_status("Mock Extraction", timeout=10000)

        # 3. Click Cancel
        app_driver.click_cancel()

        # 4. Verify status reflects cancellation.
        #    If the mock completes first it may show Complete; Cancelled is the
        #    expected mid-run outcome handled by mock_app's cancel_event check.
        app_driver.expect_status("Cancelled", timeout=10000)

        # 5. Reset and retry (verify no stuck state)
        app_driver.page.locator(Selectors.START_EXTRACTION).click()
        app_driver.expect_status(Selectors.STATUS_SUCCESS_EXTRACTION, timeout=30000)

    def test_cancel_propagation_midway(self, app_driver: AppDriver):
        """
        Start propagation → Click Cancel → Verify cancellation or immediate recovery.
        """
        # Setup: quick extraction
        app_driver.extract("prop_cancel.mp4").expect_status(Selectors.STATUS_SUCCESS_EXTRACTION, timeout=30000)

        # Setup: quick pre-analysis
        app_driver.navigate(Labels.TAB_SUBJECT).pre_analyze().expect_status(
            Selectors.STATUS_SUCCESS_PRE_ANALYSIS, timeout=30000
        )

        # 1. Start Propagation
        app_driver.navigate(Labels.TAB_SCENES).propagate()

        # 2. Wait for it to definitely be running
        app_driver.expect_status("Mock Propagation", timeout=10000)

        # 3. Click Cancel quickly
        app_driver.click_cancel()

        # 4. Verify status reflects cancellation
        app_driver.expect_status("Cancelled", timeout=10000)
