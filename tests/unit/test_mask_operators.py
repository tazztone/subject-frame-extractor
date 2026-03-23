import numpy as np

from core.operators.base import OperatorContext
from core.operators.mask_operators import PhashOperator, SubjectMaskAreaOperator


def test_phash_operator():
    """Test phash generation."""
    op = PhashOperator()
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    ctx = OperatorContext(image_rgb=img)
    result = op.execute(ctx)
    assert result.success
    assert "phash" in result.data


def test_subject_mask_area_full():
    """Test 100% mask area."""
    op = SubjectMaskAreaOperator()
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    mask = np.full((100, 100), 255, dtype=np.uint8)
    ctx = OperatorContext(image_rgb=img, mask=mask)
    result = op.execute(ctx)
    assert result.success
    assert result.metrics["mask_area_pct"] == 100.0


def test_subject_mask_area_half():
    """Test 50% mask area."""
    op = SubjectMaskAreaOperator()
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[:50, :] = 255
    ctx = OperatorContext(image_rgb=img, mask=mask)
    result = op.execute(ctx)
    assert result.success
    assert result.metrics["mask_area_pct"] == 50.0


def test_subject_mask_area_missing():
    """Test with missing mask."""
    op = SubjectMaskAreaOperator()
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    ctx = OperatorContext(image_rgb=img)
    result = op.execute(ctx)
    assert result.success
    assert result.metrics["mask_area_pct"] == 0.0
    assert "Missing mask" in result.warnings
