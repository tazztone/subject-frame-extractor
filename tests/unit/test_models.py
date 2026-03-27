from unittest.mock import MagicMock

import pytest

from core.enums import SceneStatus
from core.models import AnalysisParameters, Frame, Scene, SceneState, _coerce, _sanitize_face_ref


def test_coerce_none():
    assert _coerce(None, str) is None


def test_analysis_parameters_from_ui_empty_string():
    logger = MagicMock()
    config = MagicMock()
    config.model_dump.return_value = {}
    params = AnalysisParameters.from_ui(logger, config, method="  ")
    assert params.method == ""  # Default


def test_coerce():
    assert _coerce("true", bool) is True
    assert _coerce("yes", bool) is True
    assert _coerce("1", bool) is True
    assert _coerce("off", bool) is False
    assert _coerce(123, int) == 123
    assert _coerce("45.6", float) == 45.6
    with pytest.raises(ValueError):
        _coerce("not a number", int)


def test_frame_model():
    import numpy as np

    frame = Frame(image_data=np.zeros((10, 10, 3), dtype=np.uint8), frame_number=10)
    assert frame.frame_number == 10


def test_sanitize_face_ref(tmp_path):
    logger = MagicMock()
    # Path exists
    ref = tmp_path / "ref.jpg"
    ref.touch()
    path, enabled = _sanitize_face_ref({"face_ref_img_path": str(ref)}, logger)
    assert enabled is True
    assert path == str(ref)

    # Path does not exist
    path, enabled = _sanitize_face_ref({"face_ref_img_path": "missing.jpg"}, logger)
    assert enabled is False
    assert logger.warning.called

    # Same as video path
    path, enabled = _sanitize_face_ref({"face_ref_img_path": str(ref), "video_path": str(ref)}, logger)
    assert enabled is False


def test_scene_state():
    scene = Scene(shot_id=1, start_frame=0, end_frame=10)
    state = SceneState(scene)

    # Update seed
    state.update_seed_result([0, 0, 10, 10], {"conf": 0.9})
    assert state.scene.initial_bbox == [0, 0, 10, 10]
    assert state.scene.selected_bbox == [0, 0, 10, 10]

    # Set manual bbox
    state.set_manual_bbox([5, 5, 15, 15], "user")
    assert state.scene.is_overridden is True
    assert state.scene.status == SceneStatus.INCLUDED

    # Reset
    state.reset()
    assert state.scene.is_overridden is False
    assert state.scene.selected_bbox == [0, 0, 10, 10]

    # Include/Exclude
    state.include()
    assert state.scene.status == SceneStatus.INCLUDED
    state.exclude()
    assert state.scene.status == SceneStatus.EXCLUDED


def test_analysis_parameters_from_ui():
    logger = MagicMock()
    config = MagicMock()
    config.model_dump.return_value = {"default_tracker_model_name": "sam2", "filter_default_quality_score": True}
    config.default_thumb_megapixels = 0.5

    # Test basic UI creation
    params = AnalysisParameters.from_ui(
        logger, config, tracker_model_name="sam2", thumb_megapixels="0.8", compute_niqe="true"
    )
    # The default from config is used initially, then overridden by kwargs
    assert params.tracker_model_name == "sam2"
    assert params.thumb_megapixels == 0.8
    assert params.compute_niqe is True
    assert params.compute_quality_score is True  # From config filter_default

    # Test invalid thumb_mp
    params2 = AnalysisParameters.from_ui(logger, config, thumb_megapixels="-1")
    assert params2.thumb_megapixels == 0.5  # default

    # Test invalid pre_sample_nth
    params3 = AnalysisParameters.from_ui(logger, config, pre_sample_nth="0")
    assert params3.pre_sample_nth == 1


def test_scene_state_dict_and_initial_bbox():
    # Test initialization from dict (Line 126)
    data = {"shot_id": 1, "start_frame": 0, "end_frame": 10, "seed_result": {"bbox": [0, 0, 10, 10]}}
    state = SceneState(data)  # Line 126
    # Line 132-133: initial_bbox should be set from seed_result if missing
    assert state.scene.initial_bbox == [0, 0, 10, 10]
    assert state.scene.selected_bbox == [0, 0, 10, 10]

    # Line 138: data property
    assert state.data["shot_id"] == 1

    # Line 151: reset is_overridden if bbox match
    state.set_manual_bbox([0, 0, 10, 10], "reset")
    assert state.scene.is_overridden is False


def test_analysis_parameters_optional_coercion():
    # Line 280-282: Handle Optional types in from_ui
    logger = MagicMock()
    config = MagicMock()
    config.model_dump.return_value = {}
    # thumb_megapixels is float
    params = AnalysisParameters.from_ui(logger, config, thumb_megapixels=1.2)
    assert params.thumb_megapixels == 1.2


def test_coerce_more():
    assert _coerce("1", bool) is True
    assert _coerce("yes", bool) is True
    assert _coerce("on", bool) is True
    assert _coerce("0", bool) is False
    assert _coerce(True, bool) is True


def test_frame_metrics_defaults():
    from core.models import FrameMetrics

    metrics = FrameMetrics()
    assert metrics.quality_score == 0.0
