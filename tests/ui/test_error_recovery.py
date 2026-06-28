import pytest

from .app_driver import AppDriver
from .ui_locators import Labels, Selectors

# Mark as e2e test
pytestmark = pytest.mark.e2e


class TestErrorRecovery:
    """
    Tests for error paths and user recovery after pipeline/validation failures.
    Ensures the user can retry successfully without state corruption.
    """

    def test_extraction_with_invalid_path_to_success(self, app_driver: AppDriver):
        """
        Fill a nonsense path → Extract → Verify Error → Retry with valid path.
        """
        # 1. Fill invalid source
        app_driver.extract("nonsense_invalid_path.xyz")
        # 2. Verify Error in status
        app_driver.expect_status(Selectors.STATUS_ERROR_REGEX, timeout=15000)

        # 3. Correct the path and retry
        app_driver.extract("valid_retry_test.mp4")
        # 4. Verify Success
        app_driver.expect_status(Selectors.STATUS_SUCCESS_EXTRACTION, timeout=30000)

    def test_tab_jump_restriction_recovery(self, app_driver: AppDriver):
        """
        Skip to Subject tab, click Confirm → Verify error message → Go back and extract → Ensure user is unblocked.
        """
        # 1. Skip to Subject
        app_driver.navigate(Labels.TAB_SUBJECT).pre_analyze()

        # 2. Verify error detail in status
        app_driver.expect_status(Selectors.STATUS_ERROR_REGEX, timeout=10000)

        # 3. Go back and fix source
        app_driver.navigate(Labels.TAB_SOURCE).extract("navigation_recovery.mp4")
        app_driver.expect_status(Selectors.STATUS_SUCCESS_EXTRACTION, timeout=30000)

        # 4. Now Subject tab action should succeed
        app_driver.navigate(Labels.TAB_SUBJECT).pre_analyze()
        app_driver.expect_status(Selectors.STATUS_SUCCESS_PRE_ANALYSIS, timeout=30000)

    def test_export_precondition_failure(self, app_driver: AppDriver):
        """
        Go to Export tab, verify Export button is disabled when no data is loaded.
        """
        # Skip to Export
        app_driver.navigate(Labels.TAB_EXPORT)

        # Verify Export button is disabled
        app_driver.expect_disabled(Selectors.EXPORT_BUTTON, timeout=10000)
