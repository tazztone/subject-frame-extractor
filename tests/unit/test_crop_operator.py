import numpy as np
import pytest

from core.operators.crop import calculate_best_crop, crop_image_with_subject


def test_calculate_best_crop_extreme_ratios():
    """Test crop calculation with very wide or very tall aspect ratios."""
    img_w, img_h = 1920, 1080
    subject_bbox = [900, 500, 100, 100]  # x, y, w, h

    # 1. Very wide (e.g., 21:9)
    aspect_ratios = [("wide", 21 / 9)]
    crop = calculate_best_crop(img_w, img_h, subject_bbox, aspect_ratios, padding_factor=1.0)
    assert crop is not None
    assert crop["w"] / crop["h"] == pytest.approx(21 / 9, rel=1e-2)

    # 2. Very tall (e.g., 9:21)
    aspect_ratios = [("tall", 9 / 21)]
    crop = calculate_best_crop(img_w, img_h, subject_bbox, aspect_ratios, padding_factor=1.0)
    assert crop is not None
    assert crop["w"] / crop["h"] == pytest.approx(9 / 21, rel=1e-2)


def test_calculate_best_crop_at_boundaries():
    """Test crop calculation when subject is near the image edge."""
    img_w, img_h = 100, 100

    # Top-left corner
    subject_bbox = [0, 0, 10, 10]
    aspect_ratios = [("square", 1.0)]
    crop = calculate_best_crop(img_w, img_h, subject_bbox, aspect_ratios, padding_factor=2.0)
    assert crop["x"] == 0
    assert crop["y"] == 0
    assert crop["w"] >= 20

    # Bottom-right corner
    subject_bbox = [90, 90, 10, 10]
    crop = calculate_best_crop(img_w, img_h, subject_bbox, aspect_ratios, padding_factor=2.0)
    assert crop["x"] + crop["w"] <= 100
    assert crop["y"] + crop["h"] <= 100


def test_crop_image_with_subject_no_subject():
    """Test crop function when no subject bbox is found in mask."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    mask = np.zeros((100, 100), dtype=np.uint8)  # Empty mask

    cropped, ar_name = crop_image_with_subject(img, mask, [("1:1", 1.0)], padding_factor=1.0)
    assert cropped is None
    assert "No subject" in ar_name
