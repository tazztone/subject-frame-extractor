import pytest

from .app_driver import AppDriver
from .ui_locators import Labels, Selectors

# Mark as e2e test
pytestmark = pytest.mark.e2e


class TestFullWorkflowMocked:
    """
    Comprehensive E2E test simulating a full user journey using Playwright
    against the mock application (tests/mock_app.py).

    The mock app simulates backend processing without needing heavy models/GPU.
    """

    def test_full_user_journey(self, app_driver: AppDriver):
        """
        Simulates:
        1. Select Video Source (Extraction)
        2. Run Extraction
        3. Define Subject (Pre-analysis)
        4. Select a person/scene
        5. Run Propagation
        6. Filter Results
        7. Export
        """
        # 1. Source Tab — enter video path and extract
        app_driver.extract("test_journey.mp4").expect_status(Selectors.STATUS_SUCCESS_EXTRACTION, timeout=30000)

        # 2. Subject Tab — Confirm Subject
        app_driver.navigate(Labels.TAB_SUBJECT).pre_analyze().expect_status(
            Selectors.STATUS_SUCCESS_PRE_ANALYSIS, timeout=30000
        )

        # 3. Scenes Tab — Propagate Masks
        app_driver.navigate(Labels.TAB_SCENES).propagate().expect_status(
            Selectors.STATUS_SUCCESS_PROPAGATION, timeout=30000
        )

        # 4. Metrics Tab — Start Analysis
        app_driver.navigate(Labels.TAB_METRICS).analyze().expect_status(
            Selectors.STATUS_SUCCESS_ANALYSIS, timeout=30000
        )

        # 5. Export Tab — Start Export
        app_driver.navigate(Labels.TAB_EXPORT).export().expect_status(Selectors.STATUS_SUCCESS_EXPORT, timeout=30000)
