from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from core.enums import SeedStrategy
from core.models import AnalysisParameters
from core.scene_utils.seed_selector import SeedSelector


@pytest.fixture
def mock_config():
    config = MagicMock()
    config.seeding_face_similarity_threshold = 0.5
    config.seeding_iou_threshold = 0.5
    config.seeding_face_contain_score = 10.0
    config.seeding_confidence_score_multiplier = 1.0
    config.seeding_iou_bonus = 5.0
    config.seeding_face_to_body_expansion_factors = [1.5, 4.0, 0.2]
    config.seeding_final_fallback_box = [0.25, 0.1, 0.5, 0.8]
    config.seeding_balanced_score_weights = {"area": 0.4, "confidence": 0.4, "edge": 0.2}
    return config


@pytest.fixture
def mock_logger():
    return MagicMock()


@pytest.fixture
def selector(mock_config, mock_logger):
    params = AnalysisParameters(
        source_path="test.mp4",
        output_folder="/tmp",
    )
    return SeedSelector(params=params, config=mock_config, logger=mock_logger)


def test_select_seed_identity_first_no_ref(selector):
    """Test identity-first strategy fallback when reference embedding is missing."""
    selector.params.primary_seed_strategy = SeedStrategy.FACE_REFERENCE.value
    selector.params.compute_face_sim = True
    selector.face_analyzer = MagicMock()
    selector.reference_embedding = None

    with patch.object(selector, "_object_first_seed", return_value=([0, 0, 10, 10], {})) as mock_obj:
        selector.select_seed(np.zeros((100, 100, 3), dtype=np.uint8))
        mock_obj.assert_called_once()


def test_find_target_face_error_paths(selector):
    """Test error paths in _find_target_face."""
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    # 1. No analyzer
    selector.face_analyzer = None
    _, details = selector._find_target_face(frame)
    assert details["error"] == "face_analyzer_not_initialized"

    # 2. Analyzer exception
    selector.face_analyzer = MagicMock()
    selector.face_analyzer.get.side_effect = Exception("InsightFace Fail")
    _, details = selector._find_target_face(frame)
    assert details["error"] == "InsightFace Fail"

    # 3. No faces found
    selector.face_analyzer.get.side_effect = None
    selector.face_analyzer.get.return_value = []
    _, details = selector._find_target_face(frame)
    assert details["error"] == "no_faces_detected"

    # 4. No reference embedding
    selector.face_analyzer.get.return_value = [MagicMock()]
    selector.reference_embedding = None
    _, details = selector._find_target_face(frame)
    assert details["error"] == "reference_embedding_not_initialized"


def test_get_subject_boxes_from_scene(selector):
    """Test getting subject boxes from scene metadata."""
    scene = MagicMock()
    scene.person_detections = [{"bbox": [0, 0, 10, 10], "conf": 1.0}]
    res = selector._get_subject_boxes(np.zeros((10, 10, 3)), scene)
    assert res == scene.person_detections

    # From selected_bbox
    scene.person_detections = None
    scene.selected_bbox = [0, 0, 5, 5]
    res = selector._get_subject_boxes(np.zeros((10, 10, 3)), scene)
    assert res[0]["bbox"] == [0, 0, 5, 5]


def test_get_subject_boxes_no_tracker(selector):
    """Test _get_subject_boxes when tracker is None."""
    selector.tracker = None
    assert selector._get_subject_boxes(np.zeros((10, 10, 3))) == []


def test_get_text_prompt_boxes_error_paths(selector):
    """Test error paths in _get_text_prompt_boxes."""
    frame = np.zeros((10, 10, 3))

    # 1. No tracker or no prompt
    selector.tracker = None
    res, details = selector._get_text_prompt_boxes(frame, {"text_prompt": ""})
    assert res == []

    # 2. Tracker exception
    selector.tracker = MagicMock()
    selector.tracker.detect_objects.side_effect = Exception("SAM3 Fail")
    res, details = selector._get_text_prompt_boxes(frame, {"text_prompt": "person"})
    assert "SAM3 Fail" in details["error"]


def test_choose_subject_by_strategy_no_analyzer_fallback(selector):
    """Test choose_subject fallback when face_analyzer is None."""
    selector.tracker = MagicMock()
    selector.tracker.detect_objects.return_value = []
    selector.face_analyzer = None

    res_box, details = selector._choose_subject_by_strategy(np.zeros((100, 100, 3), dtype=np.uint8), {})
    assert details["type"] == "no_subjects_fallback"


def test_choose_subject_by_strategy_post_selection_failure(selector):
    """Test face analysis failure during post-selection."""
    selector.tracker = MagicMock()
    selector.tracker.detect_objects.return_value = [{"bbox": [0, 0, 10, 10], "conf": 0.9, "type": "person"}]
    selector.face_analyzer = MagicMock()
    selector.face_analyzer.get.side_effect = Exception("Post-selection Fail")
    selector.reference_embedding = np.random.rand(512)

    # Should still succeed but with no seed_face_sim
    res_box, details = selector._choose_subject_by_strategy(np.zeros((100, 100, 3), dtype=np.uint8), {})
    assert details["seed_face_sim"] is None


def test_get_mask_for_bbox_error_paths(selector):
    """Test error paths in _get_mask_for_bbox."""
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    bbox = [0, 0, 5, 5]

    # 1. No tracker
    selector.tracker = None
    assert selector._get_mask_for_bbox(frame, bbox) is None

    # 2. GPU OOM
    selector.tracker = MagicMock()
    with patch("core.scene_utils.seed_selector.torch.cuda.OutOfMemoryError", RuntimeError):
        selector.tracker.init_video.side_effect = RuntimeError("out of memory")
        with patch("core.scene_utils.seed_selector.torch.cuda.is_available", return_value=True, create=True):
            with patch("core.scene_utils.seed_selector.torch.cuda.empty_cache") as mock_empty:
                assert selector._get_mask_for_bbox(frame, bbox) is None
                mock_empty.assert_called_once()

    # 3. Generic Exception
    selector.tracker.init_video.side_effect = Exception("Generic Fail")
    assert selector._get_mask_for_bbox(frame, bbox) is None
