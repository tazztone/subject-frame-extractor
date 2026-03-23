from unittest.mock import patch

from core.operators.viz import histogram_svg


def test_histogram_svg_basic():
    """Test standard SVG generation."""
    # counts, bins
    hist_data = ([1, 5, 10, 2], [0, 10, 20, 30, 40])
    svg = histogram_svg(hist_data, title="Test Plot")
    assert "<svg" in svg

    import core.operators.viz

    if core.operators.viz.plt:
        assert "Test Plot" in svg
        # The warning should NOT be present
        assert "matplotlib missing" not in svg.lower()
    else:
        assert "Matplotlib missing" in svg


def test_histogram_svg_empty():
    """Test with empty data."""
    assert histogram_svg(None) == ""
    assert histogram_svg(([], [])) == ""


def test_histogram_svg_missing_matplotlib():
    """Test behavior when matplotlib is not installed."""
    hist_data = ([1], [0, 1])
    import core.operators.viz

    old_plt = core.operators.viz.plt
    core.operators.viz.plt = None
    try:
        svg = histogram_svg(hist_data)
        assert "Matplotlib missing" in svg
    finally:
        core.operators.viz.plt = old_plt


def test_histogram_svg_error():
    """Test error handling during plotting."""
    with patch("matplotlib.pyplot.subplots", side_effect=Exception("Plot Error")):
        svg = histogram_svg(([1], [0, 1]))
        assert "Plotting failed" in svg
