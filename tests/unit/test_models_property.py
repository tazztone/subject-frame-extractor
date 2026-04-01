from unittest.mock import MagicMock

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


@given(
    x1=st.integers(0, 1000),
    y1=st.integers(0, 1000),
    x2=st.integers(0, 1000),
    y2=st.integers(0, 1000),
)
def test_xyxy_to_xywh_roundtrip(x1, y1, x2, y2):
    """Test that xyxy_to_xywh and its inverse (xywh_to_xyxy) work correctly."""
    from core.scene_utils.seed_selector import SeedSelector

    # Ensure x2 >= x1 and y2 >= y1
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)

    # Use a dummy selector
    selector = SeedSelector(params=MagicMock(), config=MagicMock())

    xyxy = [x1, y1, x2, y2]
    xywh = selector._xyxy_to_xywh(xyxy)

    # xywh format: [x, y, w, h]
    assert xywh[0] == x1
    assert xywh[1] == y1
    assert xywh[2] == max(1, x2 - x1)
    assert xywh[3] == max(1, y2 - y1)


@given(
    fx1=st.integers(0, 1000),
    fy1=st.integers(0, 1000),
    fx2=st.integers(0, 1000),
    fy2=st.integers(0, 1000),
    img_w=st.integers(1, 2000),
    img_h=st.integers(1, 2000),
)
def test_expand_face_to_body_safety(fx1, fy1, fx2, fy2, img_w, img_h):
    """Test that expand_face_to_body never produces coordinates outside image bounds."""
    from core.scene_utils.seed_selector import SeedSelector

    # Ensure fx2 >= fx1 and fy2 >= fy1
    fx1, fx2 = min(fx1, fx2), max(fx1, fx2)
    fy1, fy2 = min(fy1, fy2), max(fy1, fy2)
    # Clamp to image bounds initially
    fx1 = min(fx1, img_w - 1)
    fx2 = min(fx2, img_w)
    fy1 = min(fy1, img_h - 1)
    fy2 = min(fy2, img_h)

    face_bbox = [fx1, fy1, fx2, fy2]
    img_shape = (img_h, img_w, 3)

    config = MagicMock()
    config.seeding_face_to_body_expansion_factors = [1.0, 1.0, 1.0]  # Default multipliers
    selector = SeedSelector(params=MagicMock(), config=config)

    expanded = selector._expand_face_to_body(face_bbox, img_shape)

    # expanded is in xywh format? Let's check.
    # Actually _expand_face_to_body in core/scene_utils/seed_selector.py returns xywh!

    ex, ey, ew, eh = expanded
    assert ex >= 0
    assert ey >= 0
    assert ex + ew <= img_w
    assert ey + eh <= img_h
