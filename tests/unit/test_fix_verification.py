from unittest.mock import MagicMock, patch

import numpy as np

from core.enums import SeedStrategy
from core.events import PreAnalysisEvent
from core.models import AnalysisParameters
from core.scene_utils.seed_selector import SeedSelector


def test_tracker_model_name_defaults_to_sam2():
    """Verify that AnalysisParameters and PreAnalysisEvent default to 'sam2'."""
    params = AnalysisParameters()
    assert params.tracker_model_name == "sam2"

    event = PreAnalysisEvent(output_folder="/tmp", video_path="test.mp4", method="scene")
    assert event.tracker_model_name == "sam2"


def test_face_fallback_when_no_people_detected():
    """Verify that SeedSelector falls back to face detection when person detection fails."""
    # Mock dependencies
    logger = MagicMock()
    config = MagicMock()
    params = AnalysisParameters()

    selector = SeedSelector(logger, config, params)

    # Mock face analyzer
    mock_face = MagicMock()
    mock_face.bbox = np.array([10, 10, 50, 50])
    mock_face.det_score = 0.9

    selector.face_analyzer = MagicMock()
    selector.face_analyzer.get.return_value = [mock_face]

    # Mock tracker to return no boxes
    selector.tracker = MagicMock()
    selector.tracker.detect_objects.return_value = []

    # Mock _expand_face_to_body and _get_person_boxes (to return empty)
    with (
        patch.object(selector, "_get_person_boxes", return_value=[]),
        patch.object(selector, "_expand_face_to_body", return_value=[10, 10, 100, 200]),
    ):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        box, details = selector._choose_person_by_strategy(frame, params)

        assert details["type"] == "face_fallback_expanded"
        assert box == [10, 10, 100, 200]
        assert details["face_conf"] == 0.9


def test_strategy_routing_with_enum():
    """Verify that SeedSelector correctly routes strategies using the SeedStrategy enum."""
    logger = MagicMock()
    config = MagicMock()
    params = AnalysisParameters()
    selector = SeedSelector(logger, config, params)

    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    with (
        patch.object(selector, "_identity_first_seed") as mock_id,
        patch.object(selector, "_object_first_seed") as mock_obj,
        patch.object(selector, "_choose_person_by_strategy") as mock_auto,
    ):
        # Test Face Reference
        params.primary_seed_strategy = SeedStrategy.FACE_REFERENCE.value
        selector.reference_embedding = np.array([0.1])
        params.enable_face_filter = True
        selector.face_analyzer = MagicMock()
        selector.select_seed(frame, params)
        mock_id.assert_called_once()

        # Test Text Description
        params.primary_seed_strategy = SeedStrategy.TEXT_DESCRIPTION.value
        selector.select_seed(frame, params)
        mock_obj.assert_called_once()

        # Test Automatic
        params.primary_seed_strategy = SeedStrategy.AUTOMATIC.value
        selector.select_seed(frame, params)
        mock_auto.assert_called_once()
