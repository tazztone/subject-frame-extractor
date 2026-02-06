import numpy as np
import pytest
from examples.operators.pixel_count import PixelCountOperator
from core.operators import OperatorContext

def test_pixel_count_all_black():
    op = PixelCountOperator()
    ctx = OperatorContext(image_rgb=np.zeros((10, 10, 3), dtype=np.uint8))
    result = op.execute(ctx)
    assert result.metrics["pixel_count"] == 0.0

def test_pixel_count_all_white():
    op = PixelCountOperator()
    ctx = OperatorContext(image_rgb=np.full((10, 10, 3), 255, dtype=np.uint8))
    result = op.execute(ctx)
    assert result.metrics["pixel_count"] == 100.0  # 10x10 pixels
