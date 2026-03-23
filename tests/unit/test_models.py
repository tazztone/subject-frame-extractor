from core.events import ExtractionEvent, PreAnalysisEvent
from core.models import AnalysisParameters, Frame, Scene
from core.operators import OperatorResult


def test_operator_result():
    # Success case
    res = OperatorResult(data={"val": 1})
    assert res.success is True
    assert res.data["val"] == 1


def test_analysis_parameters():
    params = AnalysisParameters(output_folder="out")
    assert params.output_folder == "out"
    assert params.scene_detect is False  # Default


def test_extraction_event():
    event = ExtractionEvent(
        source_path="video.mp4",
        method="every_nth_frame",
        interval=1.0,
        nth_frame=3,
        max_resolution="1080",
        thumb_megapixels=0.5,
        scene_detect=True,
    )
    assert event.source_path == "video.mp4"
    assert event.nth_frame == 3


def test_pre_analysis_event():
    event = PreAnalysisEvent(
        output_folder="out",
        video_path="v.mp4",
        face_model_name="f",
        tracker_model_name="t",
        best_frame_strategy="s",
        min_mask_area_pct=1.0,
        sharpness_base_scale=2500.0,
        edge_strength_base_scale=100.0,
        primary_seed_strategy="a",
    )
    assert event.output_folder == "out"


def test_frame_model():
    import numpy as np

    frame = Frame(image_data=np.zeros((10, 10, 3), dtype=np.uint8), frame_number=10)
    assert frame.frame_number == 10


def test_scene_model():
    scene = Scene(shot_id=1, start_frame=0, end_frame=100)
    assert scene.shot_id == 1
    assert scene.end_frame == 100
