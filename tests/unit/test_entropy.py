import numpy as np

from core.operators.base import OperatorContext
from core.operators.entropy import EntropyOperator


def test_entropy_solid_color():
    """Entropy of a solid color image should be 0."""
    op = EntropyOperator()
    # 100x100 solid black image
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    ctx = OperatorContext(image_rgb=img)

    result = op.execute(ctx)
    assert result.success
    assert result.metrics["entropy"] == 0.0
    assert result.metrics["entropy_score"] == 0.0


def test_entropy_random_noise():
    """Entropy of random noise should be high."""
    op = EntropyOperator()
    # 100x100 random noise (uniform distribution)
    # Max entropy for 256 bins is 8.0 bits.
    np.random.seed(42)
    img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    ctx = OperatorContext(image_rgb=img)

    result = op.execute(ctx)
    assert result.success
    # Shannon entropy for uniform 0-255 is approx 8.0.
    # Our mapping is entropy / 8.0 * 100.0.
    assert result.metrics["entropy_score"] > 90.0
    assert result.metrics["entropy"] > 0.9


def test_entropy_with_mask():
    """Test entropy with a mask that isolates a specific region."""
    op = EntropyOperator()
    # Left half black (0 entropy), right half random noise (high entropy)
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    np.random.seed(42)
    img[:, 50:, :] = np.random.randint(0, 256, (100, 50, 3), dtype=np.uint8)

    # Mask covers only the black left half
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[:, :50] = 255

    ctx = OperatorContext(image_rgb=img, mask=mask)
    result = op.execute(ctx)
    assert result.success
    assert result.metrics["entropy"] < 0.1  # Should be near 0 because it's mostly solid

    # Mask covers only the noisy right half
    mask_noisy = np.zeros((100, 100), dtype=np.uint8)
    mask_noisy[:, 50:] = 255
    ctx_noisy = OperatorContext(image_rgb=img, mask=mask_noisy)
    result_noisy = op.execute(ctx_noisy)
    assert result_noisy.success
    assert result_noisy.metrics["entropy"] > 0.9


def test_entropy_empty_mask():
    """Test entropy when the mask excludes all pixels."""
    op = EntropyOperator()
    img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    # Mask with only a few pixels (less than 100 threshold in implementation)
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[0, 0:10] = 255

    ctx = OperatorContext(image_rgb=img, mask=mask)
    result = op.execute(ctx)
    # The implementation uses the full image if mask < 100 pixels
    assert result.success
    assert result.metrics["entropy_score"] > 0


def test_entropy_error_handling():
    """Test error handling with invalid input."""
    op = EntropyOperator()
    # Invalid image type (None) - should fail because cvtColor will raise
    ctx = OperatorContext(image_rgb=None)
    result = op.execute(ctx)
    assert not result.success  # This uses the @property
    assert result.error is not None
