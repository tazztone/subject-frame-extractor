from hypothesis import given
from hypothesis import strategies as st

from core.models import AnalysisParameters, Scene


@given(
    shot_id=st.integers(min_value=-1000, max_value=10000),
    start_frame=st.integers(min_value=-1000, max_value=1000000),
    end_frame=st.integers(min_value=-1000, max_value=1000000),
    status=st.text(min_size=0, max_size=50),
)
def test_scene_model_properties(shot_id, start_frame, end_frame, status):
    """Property-based test for Scene model."""
    # We expect Pydantic to handle basic validation if we add it,
    # but for now we just check if it instantiates without crashing
    # or if it correctly rejects invalid values if we had validators.
    try:
        scene = Scene(shot_id=shot_id, start_frame=start_frame, end_frame=end_frame, status=status)
        assert scene.shot_id == shot_id
    except (ValueError, TypeError):
        # If we had strict validation, this would be expected for some inputs
        pass


@given(source_path=st.text(min_size=1), interval=st.floats(min_value=-1.0, max_value=100.0), max_resolution=st.text())
def test_analysis_parameters_properties(source_path, interval, max_resolution):
    """Property-based test for AnalysisParameters model."""
    try:
        params = AnalysisParameters(source_path=source_path, interval=interval, max_resolution=max_resolution)
        assert params.source_path == source_path
    except (ValueError, TypeError):
        pass
