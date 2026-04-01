from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

from core.image_utils import compute_entropy, draw_bbox, postprocess_mask, render_mask_overlay, rgb_to_pil


def test_postprocess_mask_basic(mock_config):
    # Simple binary mask with two blobs
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[10:20, 10:20] = 255
    mask[50:60, 50:70] = 255  # Larger blob

    processed = postprocess_mask(mask, mock_config)
    assert np.any(processed)
    # With keep_largest_only, only the larger blob (50:60, 50:70) should remain
    assert processed[15, 15] == 0
    assert processed[55, 55] == 255


def test_postprocess_mask_empty(mock_config):
    assert postprocess_mask(None, mock_config) is None
    empty = np.array([])
    assert postprocess_mask(empty, mock_config).size == 0


def test_render_mask_overlay_basic():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[10:20, 10:20] = 255
    logger = MagicMock()

    overlay = render_mask_overlay(image, mask, alpha=0.5, logger=logger)
    assert overlay.shape == (100, 100, 3)
    assert np.any(overlay[15, 15, 0] > 0)


def test_render_mask_overlay_resize():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    mask = np.zeros((50, 50), dtype=np.uint8)  # Different size
    mask[5:10, 5:10] = 255
    logger = MagicMock()

    overlay = render_mask_overlay(image, mask, alpha=0.5, logger=logger)
    assert overlay.shape == (100, 100, 3)
    # The mask should have been resized to 100x100
    assert np.any(overlay[15, 15, 0] > 0)


def test_render_mask_overlay_invalid_shape():
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    mask = np.zeros((10, 10, 3), dtype=np.uint8)  # Should be 2D or 3D with 1 channel
    logger = MagicMock()

    overlay = render_mask_overlay(image, mask, alpha=0.5, logger=logger)
    assert np.array_equal(overlay, image)
    assert logger.warning.called


def test_render_mask_overlay_none():
    logger = MagicMock()
    assert render_mask_overlay(None, None, 0.5, logger).size == 0
    img = np.zeros((10, 10, 3))
    assert np.array_equal(render_mask_overlay(img, None, 0.5, logger), img)


def test_rgb_to_pil():
    img_rgb = np.zeros((10, 10, 3), dtype=np.uint8)
    img_rgb[0, 0] = [255, 128, 64]

    pil_img = rgb_to_pil(img_rgb)
    assert isinstance(pil_img, Image.Image)
    assert pil_img.getpixel((0, 0)) == (255, 128, 64)


def test_draw_bbox(mock_config):
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    bbox = [10, 20, 30, 40]

    res = draw_bbox(image, bbox, mock_config, label="Object")
    assert res.shape == (100, 100, 3)
    assert np.any(res > 0)
    # Check that original image is not modified
    assert np.all(image == 0)


def test_compute_entropy():
    # Solid
    hist = np.zeros(256)
    hist[128] = 100
    assert compute_entropy(hist, 8.0) == pytest.approx(0.0, abs=1e-7)

    # Simple half-half split for deterministic entropy
    hist = np.zeros(256)
    hist[10] = 50
    hist[20] = 50
    # Expected entropy: - (0.5 * log2(0.5) + 0.5 * log2(0.5)) = 1.0
    # Normalized: 1.0 / 8.0 = 0.125
    assert compute_entropy(hist, 8.0) == pytest.approx(0.125)
