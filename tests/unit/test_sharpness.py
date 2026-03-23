import numpy as np

from core.operators.base import OperatorContext
from core.operators.sharpness import SharpnessOperator


def test_sharpness_blurry():
    """Blurry image should have low sharpness."""
    op = SharpnessOperator()
    # 100x100 solid image (zero variance)
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    ctx = OperatorContext(image_rgb=img)
    result = op.execute(ctx)
    assert result.success
    assert result.metrics["sharpness_score"] == 0.0


def test_sharpness_focused():
    """Focused image (checkerboard) should have higher sharpness."""
    op = SharpnessOperator()
    # 100x100 checkerboard
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    for i in range(0, 100, 10):
        for j in range(0, 100, 10):
            if (i + j) % 20 == 0:
                img[i : i + 10, j : j + 10] = 255

    ctx = OperatorContext(image_rgb=img)
    result = op.execute(ctx)
    assert result.success
    assert result.metrics["sharpness_score"] > 20.0


def test_sharpness_with_mask():
    """Test sharpness focusing on a masked region."""
    op = SharpnessOperator()
    # Top half sharp, bottom half blurry
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    # Checkerboard in top half
    for i in range(0, 50, 10):
        for j in range(0, 100, 10):
            if (i + j) % 20 == 0:
                img[i : i + 10, j : j + 10] = 255

    # Bottom half is solid (zero variance)

    # Mask covers only the sharp top half
    mask_top = np.zeros((100, 100), dtype=np.uint8)
    mask_top[:40, :] = 255  # Stay away from the boundary at 50
    ctx_top = OperatorContext(image_rgb=img, mask=mask_top)
    result_top = op.execute(ctx_top)
    assert result_top.metrics["sharpness_score"] > 20.0

    # Mask covers only the blurry bottom half
    mask_bottom = np.zeros((100, 100), dtype=np.uint8)
    mask_bottom[60:, :] = 255  # Stay away from the boundary at 50
    ctx_bottom = OperatorContext(image_rgb=img, mask=mask_bottom)
    result_bottom = op.execute(ctx_bottom)
    assert result_bottom.metrics["sharpness_score"] == 0.0


def test_sharpness_custom_scale():
    """Test sharpness with a custom scale in config."""
    op = SharpnessOperator()
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    # Just a few white pixels to create some variance but not saturate
    img[50, 50] = 255
    img[51, 51] = 255

    # Default scale 2500
    ctx_default = OperatorContext(image_rgb=img)
    result_default = op.execute(ctx_default)

    # Custom scale 10000 (should yield LOWER normalized score)
    class MockConfig:
        sharpness_base_scale = 10000.0

    mock_config = MockConfig()
    ctx_custom = OperatorContext(image_rgb=img, config=mock_config)
    result_custom = op.execute(ctx_custom)

    assert result_custom.metrics["sharpness"] < result_default.metrics["sharpness"]


def test_sharpness_empty_region():
    """Test sharpness with a mask that has too few pixels."""
    op = SharpnessOperator()
    img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    # Tiny mask
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[0, 0:5] = 255

    ctx = OperatorContext(image_rgb=img, mask=mask)
    result = op.execute(ctx)
    # Should revert to full image or report empty
    # implementation says if np.sum(active_mask) >= 100 it uses it, otherwise uses original laplacian
    assert result.success
    assert result.metrics["sharpness_score"] > 0
