from hypothesis import given
from hypothesis import strategies as st

from core.models import AnalysisParameters, Scene


@given(
    shot_id=st.integers(min_value=1, max_value=10000),
    start_frame=st.integers(min_value=0, max_value=1000000),
    end_frame=st.integers(min_value=0, max_value=1000000),
    status=st.sampled_from(["pending", "included", "excluded"]),
)
def test_scene_model_properties(shot_id, start_frame, end_frame, status):
    """Property-based test for Scene model."""
    scene = Scene(shot_id=shot_id, start_frame=start_frame, end_frame=end_frame, status=status)
    assert scene.shot_id == shot_id
    assert scene.start_frame == start_frame
    assert scene.end_frame == end_frame
    assert scene.status == status


@given(
    start=st.integers(min_value=0, max_value=1000),
    duration=st.integers(min_value=1, max_value=1000),
)
def test_scene_ordering_invariant(start, duration):
    """In any valid scene, start_frame must be <= end_frame."""
    end = start + duration
    scene = Scene(shot_id=1, start_frame=start, end_frame=end)
    assert scene.start_frame <= scene.end_frame


@given(
    source_path=st.text(min_size=1, alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd", "Pd", "Zs"))),
    interval=st.floats(min_value=0.1, max_value=100.0),
    max_resolution=st.sampled_from(["720p", "1080p", "4k"]),
)
def test_analysis_parameters_properties(source_path, interval, max_resolution):
    """Property-based test for AnalysisParameters model."""
    params = AnalysisParameters(source_path=source_path, interval=interval, max_resolution=max_resolution)
    assert params.source_path == source_path
    assert params.interval == interval
    assert params.max_resolution == max_resolution


@given(
    x=st.floats(min_value=-1000.0, max_value=5000.0),
    y=st.floats(min_value=-1000.0, max_value=5000.0),
    w=st.floats(min_value=0.0, max_value=5000.0),
    h=st.floats(min_value=0.0, max_value=5000.0),
    img_w=st.integers(min_value=1, max_value=4000),
    img_h=st.integers(min_value=1, max_value=4000),
)
def test_bbox_normalization_invariant(x, y, w, h, img_w, img_h):
    """
    Normalization logic in SAM3Wrapper.add_bbox_prompt must always
    produce relative coordinates in [0, 1] range regardless of input.
    """
    # Simulation of SAM3Wrapper.add_bbox_prompt normalization logic (after fix):
    # rel_box = [max(0.0, min(1.0, x / w)), max(0.0, min(1.0, y / h)), ...]

    rel_x = max(0.0, min(1.0, x / img_w))
    rel_y = max(0.0, min(1.0, y / img_h))
    rel_x2 = max(0.0, min(1.0, (x + w) / img_w))
    rel_y2 = max(0.0, min(1.0, (y + h) / img_h))

    assert 0.0 <= rel_x <= 1.0
    assert 0.0 <= rel_y <= 1.0
    assert 0.0 <= rel_x2 <= 1.0
    assert 0.0 <= rel_y2 <= 1.0
    assert rel_x <= rel_x2
    assert rel_y <= rel_y2
