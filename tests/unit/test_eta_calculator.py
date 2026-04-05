"""
Unit tests for ETACalculator.
"""

from core.eta_calculator import ETACalculator


def test_calculate_eta_none():
    """Test calculate_eta with None EMA."""
    assert ETACalculator.calculate_eta(100, 10, None) is None


def test_calculate_eta_zero():
    """Test calculate_eta with 0 remaining."""
    assert ETACalculator.calculate_eta(100, 100, 1.5) == 0


def test_calculate_eta_normal():
    """Test calculate_eta with normal values."""
    # 100 total, 20 done, 80 remaining. 80 * 2.0 = 160.0
    assert ETACalculator.calculate_eta(100, 20, 2.0) == 160.0


def test_format_eta_none():
    """Test format_eta with None."""
    assert ETACalculator.format_eta(None) == "—"


def test_format_eta_seconds():
    """Test format_eta with seconds only."""
    assert ETACalculator.format_eta(45.6) == "45s"


def test_format_eta_minutes():
    """Test format_eta with minutes and seconds."""
    assert ETACalculator.format_eta(125) == "2m 5s"


def test_format_eta_hours():
    """Test format_eta with hours and minutes."""
    # 3600 + 120 = 3720
    assert ETACalculator.format_eta(3725) == "1h 2m"
