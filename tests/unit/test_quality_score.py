from unittest.mock import MagicMock

import pytest

from core.operators.base import OperatorContext
from core.operators.quality_score import QualityScoreOperator


def test_quality_score_missing_config():
    """Test behavior when config is missing."""
    op = QualityScoreOperator()
    ctx = OperatorContext(image_rgb=MagicMock())
    result = op.execute(ctx)
    assert not result.success
    assert "Missing application config" in result.error


def test_quality_score_calculation():
    """Test that quality score is calculated correctly from weights and metrics."""
    op = QualityScoreOperator()

    # Use a plain class to ensure getattr(config, "attr", default) works as expected
    class MockConfig:
        quality_weights_sharpness = 50
        quality_weights_niqe = 30
        quality_weights_contrast = 20

    mock_config = MockConfig()

    # Mock shared data with normalized metrics (0-1 range)
    shared_data = {
        "normalized_metrics": {
            "sharpness": 1.0,  # 50 * 1.0 = 50.0
            "niqe": 0.5,  # 30 * 0.5 = 15.0
            "contrast": 0.0,  # 20 * 0.0 = 0.0
        }
    }
    # Total quality = 50.0 + 15.0 + 0.0 = 65.0

    ctx = OperatorContext(image_rgb=MagicMock(), config=mock_config, shared_data=shared_data)
    result = op.execute(ctx)

    assert result.success
    assert result.metrics["quality_score"] == pytest.approx(65.0)


def test_quality_score_all_zeros():
    """Test with all zeros."""
    op = QualityScoreOperator()

    class MockConfig:
        pass

    mock_config = MockConfig()

    ctx = OperatorContext(image_rgb=MagicMock(), config=mock_config, shared_data={})
    result = op.execute(ctx)
    assert result.success
    assert result.metrics["quality_score"] == 0.0


def test_quality_score_empty_shared_data():
    """Test with weights but no metrics."""
    op = QualityScoreOperator()

    class MockConfig:
        quality_weights_sharpness = 100

    mock_config = MockConfig()

    ctx = OperatorContext(image_rgb=MagicMock(), config=mock_config, shared_data={})
    result = op.execute(ctx)
    assert result.success
    assert result.metrics["quality_score"] == 0.0
