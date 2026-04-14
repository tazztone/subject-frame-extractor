from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from core.application_state import ApplicationState
from core.pipeline_results import PreAnalysisResult
from tests.mock_app import mock_extraction_wrapper, mock_pre_analysis_wrapper


class MockApp:
    def __init__(self):
        self.components = {
            "application_state": "state_id",
            "unified_status": "status_id",
            "unified_log": "log_id",
            "progress_details": "progress_id",
        }
        self.progress_queue = MagicMock()
        from threading import Event

        self.cancel_event = Event()
        self.logger = MagicMock()


def test_extraction_blocks_on_empty_source():
    """Tier 1: Verify extraction error handling for empty source path."""
    handler = MagicMock()
    handler.app = MockApp()
    state = ApplicationState()

    # Run the generator with empty args
    results = list(mock_extraction_wrapper(handler, state, None, None))

    assert len(results) == 1
    yielded = results[0]

    status_id = handler.app.components["unified_status"]

    assert "Error/Failure" in str(yielded[status_id])
    # Now check logger instead of yield dict
    handler.logger.error.assert_called_with("Error/Failure: Invalid Source Path")


def test_extraction_success_yields_correct_status():
    """Tier 1: Verify extraction success state."""
    handler = MagicMock()
    handler.app = MockApp()
    state = ApplicationState()

    # source_path is args[0]
    results = list(mock_extraction_wrapper(handler, state, "valid_video.mp4"))

    assert len(results) >= 1
    status_id = handler.app.components["unified_status"]
    assert any("Extraction Complete" in str(u.get(status_id, "")) for u in results)


def test_pre_analysis_blocks_on_missing_extracted_video():
    """Tier 1: Verify pre-analysis error handling for missing video."""
    handler = MagicMock()
    handler.app = MockApp()
    state = ApplicationState()  # extracted_video_path is ""

    results = list(mock_pre_analysis_wrapper(handler, state))

    assert len(results) == 1
    yielded = results[0]

    status_id = handler.app.components["unified_status"]
    assert "⚠️ Error: No extracted video found" in str(yielded[status_id])


def test_pre_analysis_result_validation():
    """Tier 1: Verify PreAnalysisResult validation logic."""
    valid_data = {"unified_log": "Success", "scenes": [], "output_dir": "/tmp", "video_path": "test.mp4", "done": True}

    # Assert valid data passes
    result = PreAnalysisResult(**valid_data)
    assert result.unified_log == "Success"

    # Assert missing optional fields (like scenes) uses default
    invalid_data = valid_data.copy()
    del invalid_data["scenes"]
    result2 = PreAnalysisResult(**invalid_data)
    assert result2.scenes == []

    # Assert invalid type raises ValidationError
    invalid_data = valid_data.copy()
    invalid_data["scenes"] = "not a list"
    with pytest.raises(ValidationError):
        PreAnalysisResult(**invalid_data)
