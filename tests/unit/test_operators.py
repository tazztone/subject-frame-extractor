import cv2
import numpy as np
import pytest

from core.operators import OperatorContext
from core.operators.entropy import EntropyOperator
from core.operators.mask_operators import SubjectMaskAreaOperator
from core.operators.quality_score import QualityScoreOperator
from core.operators.sharpness import SharpnessOperator


def test_entropy_operator(mock_config):
    op = EntropyOperator()
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    ctx = OperatorContext(image_rgb=img, config=mock_config)
    res = op.execute(ctx)
    assert res.success
    assert res.metrics["entropy"] < 0.1


def test_quality_score_operator(mock_config):
    op = QualityScoreOperator()
    shared = {"normalized_metrics": {"sharpness": 0.8, "entropy": 0.9, "niqe": 0.7}}
    ctx = OperatorContext(image_rgb=np.zeros((10, 10, 3)), config=mock_config, shared_data=shared)
    res = op.execute(ctx)
    assert res.success
    assert "quality_score" in res.metrics
    assert 0 <= res.metrics["quality_score"] <= 100


def test_sharpness_operator(mock_config):
    op = SharpnessOperator()
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.rectangle(img, (10, 10), (90, 90), (255, 255, 255), -1)
    ctx = OperatorContext(image_rgb=img, config=mock_config)
    res = op.execute(ctx)
    assert res.success
    assert "sharpness" in res.metrics


def test_mask_area_operator(mock_config):
    op = SubjectMaskAreaOperator()
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[10:20, 10:20] = 255  # 100 pixels = 1.0%

    ctx = OperatorContext(image_rgb=np.zeros((100, 100, 3)), mask=mask, config=mock_config)
    res = op.execute(ctx)
    assert res.success
    assert res.metrics["mask_area_pct"] == pytest.approx(1.0)
