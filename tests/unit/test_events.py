from unittest.mock import patch

import pytest

from core.events import ExtractionEvent, PreAnalysisEvent, UIEvent


def test_extraction_event_validation():
    """Test ExtractionEvent validation logic."""
    # Valid event
    event = ExtractionEvent(
        source_path="video.mp4",
        method="all",
        interval=5.0,
        nth_frame=5,
        max_resolution="1080p",
        thumb_megapixels=0.5,
        scene_detect=True,
        output_folder="out",
    )
    assert event.source_path == "video.mp4"

    # Missing both source_path and upload_video
    with pytest.raises(ValueError, match="Please provide a Source Path or Upload a Video"):
        ExtractionEvent(
            source_path="",
            upload_video=None,
            method="all",
            interval=5.0,
            nth_frame=5,
            max_resolution="1080p",
            thumb_megapixels=0.5,
            scene_detect=True,
            output_folder="out",
        )


def test_pre_analysis_event_validation():
    """Test PreAnalysisEvent validation logic."""
    common_args = {
        "output_folder": "out",
        "video_path": "video.mp4",
        "face_model_name": "buffalo_l",
        "tracker_model_name": "sam3",
        "best_frame_strategy": "Largest Person",
        "min_mask_area_pct": 1.0,
        "sharpness_base_scale": 2500.0,
        "edge_strength_base_scale": 100.0,
        "primary_seed_strategy": "Find Prominent Person",
        "pre_sample_nth": 1,
    }

    # Valid event
    event = PreAnalysisEvent(**common_args)
    assert event.output_folder == "out"

    # Test face_ref_img_path validation (invalid path)
    with patch("core.events.Path.is_file", return_value=False):
        event_with_face = PreAnalysisEvent(**common_args, enable_face_filter=True, face_ref_img_path="nonexistent.jpg")
        assert event_with_face.enable_face_filter is False
        assert event_with_face.face_ref_img_path == ""

    # Test face_ref_img_path validation (valid path)
    with patch("core.events.Path.is_file", return_value=True):
        event_with_face = PreAnalysisEvent(**common_args, enable_face_filter=True, face_ref_img_path="existent.jpg")
        assert event_with_face.face_ref_img_path == "existent.jpg"

    # Test face_ref_img_path same as video_path
    event_same = PreAnalysisEvent(**common_args, face_ref_img_path="video.mp4")
    assert event_same.face_ref_img_path == ""

    # Test face_ref_img_path invalid extension
    with patch("core.events.Path.is_file", return_value=True):
        event_ext = PreAnalysisEvent(
            **common_args,
            face_ref_img_path="video.mp4",  # already handled by video_path check, let's use another
        )
        # Re-testing with another invalid ext
        event_ext = PreAnalysisEvent(**common_args, face_ref_img_path="file.txt")
        assert event_ext.face_ref_img_path == ""


def test_ui_event_extra_ignore():
    """Test that UIEvent ignores extra fields as configured."""
    event = UIEvent(extra_field="ignore me")
    # Should not raise error and should not have extra_field
    assert not hasattr(event, "extra_field")
