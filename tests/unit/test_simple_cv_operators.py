from unittest.mock import patch

# Helper to get real cv2 if needed for logic tests
import numpy as np
import pytest

from core.operators import OperatorContext
from core.operators.entropy import EntropyOperator
from core.operators.simple_cv import BrightnessOperator, ContrastOperator, EdgeStrengthOperator

# ============================================================================
# EdgeStrengthOperator
# ============================================================================


class TestEdgeStrengthOperator:
    @pytest.fixture
    def operator(self):
        return EdgeStrengthOperator()

    def test_config(self, operator):
        """Config values are correct."""
        assert operator.config.name == "edge_strength"
        assert operator.config.requires_mask is True

    def test_solid_color_is_zero(self, operator):
        """Solid color image has zero edge strength."""
        img = np.full((100, 100, 3), 128, dtype=np.uint8)
        ctx = OperatorContext(image_rgb=img)
        with (
            patch("core.operators.simple_cv.cv2.Sobel", side_effect=lambda x, d, dx, dy, **k: np.zeros_like(x)),
            patch("core.operators.simple_cv.cv2.cvtColor", side_effect=lambda x, c: x.mean(axis=2).astype(np.uint8)),
        ):
            result = operator.execute(ctx)
        assert result.success, result.error
        assert result.metrics["edge_strength_score"] == 0.0

    def test_edges_are_detected(self, operator):
        """Image with strong edges has >0 edge strength."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:, 50:] = 255
        ctx = OperatorContext(image_rgb=img)
        with (
            patch("core.operators.simple_cv.cv2.Sobel", side_effect=lambda x, d, dx, dy, **k: x),
            patch("core.operators.simple_cv.cv2.cvtColor", side_effect=lambda x, c: x.mean(axis=2).astype(np.uint8)),
        ):
            result = operator.execute(ctx)
        assert result.success, result.error
        assert result.metrics["edge_strength_score"] > 0.0

    def test_scaling(self, operator, mock_config):
        """Config scale affects score."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:, 50:] = 255
        ctx_default = OperatorContext(image_rgb=img)

        # Lower scale = higher score
        mock_config.edge_strength_base_scale = 10.0
        ctx_scaled = OperatorContext(image_rgb=img, config=mock_config)

        with (
            patch("core.operators.simple_cv.cv2.Sobel", side_effect=lambda x, d, dx, dy, **k: x),
            patch("core.operators.simple_cv.cv2.cvtColor", side_effect=lambda x, c: x.mean(axis=2).astype(np.uint8)),
        ):
            res_def = operator.execute(ctx_default)
            res_scale = operator.execute(ctx_scaled)

        assert res_scale.metrics["edge_strength_score"] >= res_def.metrics["edge_strength_score"]


# ============================================================================
# ContrastOperator
# ============================================================================


class TestContrastOperator:
    @pytest.fixture
    def operator(self):
        return ContrastOperator()

    def test_config(self, operator):
        assert operator.config.name == "contrast"

    def test_uniform_image_zero_contrast(self, operator):
        """Uniform image has zero contrast."""
        img = np.full((100, 100, 3), 128, dtype=np.uint8)
        ctx = OperatorContext(image_rgb=img)
        with (
            patch("core.operators.simple_cv.cv2.cvtColor", side_effect=lambda x, c: x.mean(axis=2)),
            patch("core.operators.simple_cv.cv2.Laplacian", side_effect=lambda x, d: x),
        ):
            result = operator.execute(ctx)
        assert result.metrics["contrast_score"] == 0.0

    def test_high_contrast_pattern(self, operator):
        """Black and white image has high contrast."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:50, :] = 255
        ctx = OperatorContext(image_rgb=img)
        with (
            patch("core.operators.simple_cv.cv2.cvtColor", side_effect=lambda x, c: x.mean(axis=2)),
            patch("core.operators.simple_cv.cv2.Laplacian", side_effect=lambda x, d: x),
        ):
            result = operator.execute(ctx)
        assert result.metrics["contrast_score"] > 1.0

    def test_masking_affects_contrast(self, operator):
        """Masking changes calculation to include only masked pixels."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:, 50:] = 255
        mask_left = np.zeros((100, 100), dtype=np.uint8)
        mask_left[:, :50] = 255
        ctx = OperatorContext(image_rgb=img, mask=mask_left)
        with (
            patch("core.operators.simple_cv.cv2.cvtColor", side_effect=lambda x, c: x.mean(axis=2)),
            patch("core.operators.simple_cv.cv2.Laplacian", side_effect=lambda x, d: x),
        ):
            result = operator.execute(ctx)
        assert result.metrics["contrast_score"] < 1.0


# ============================================================================
# BrightnessOperator
# ============================================================================


class TestBrightnessOperator:
    @pytest.fixture
    def operator(self):
        return BrightnessOperator()

    def test_black_image(self, operator):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        ctx = OperatorContext(image_rgb=img)
        with patch("core.operators.simple_cv.cv2.cvtColor", side_effect=lambda x, c: x.mean(axis=2).astype(x.dtype)):
            result = operator.execute(ctx)
        assert result.metrics["brightness_score"] == 0.0

    def test_white_image(self, operator):
        img = np.full((100, 100, 3), 255, dtype=np.uint8)
        ctx = OperatorContext(image_rgb=img)
        with patch("core.operators.simple_cv.cv2.cvtColor", side_effect=lambda x, c: x.mean(axis=2).astype(x.dtype)):
            result = operator.execute(ctx)
        assert result.metrics["brightness_score"] == pytest.approx(100.0, abs=0.1)

    def test_gray_image_with_mask(self, operator):
        img = np.full((100, 100, 3), 128, dtype=np.uint8)
        # Add a white patch
        img[0:10, 0:10] = 255
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[0:10, 0:10] = 255
        ctx = OperatorContext(image_rgb=img, mask=mask)
        with patch("core.operators.simple_cv.cv2.cvtColor", side_effect=lambda x, c: x.mean(axis=2).astype(x.dtype)):
            result = operator.execute(ctx)
        assert result.metrics["brightness_score"] == pytest.approx(100.0)


# ============================================================================
# EntropyOperator
# ============================================================================


class TestEntropyOperator:
    @pytest.fixture
    def operator(self):
        return EntropyOperator()

    def test_config(self, operator):
        assert operator.config.name == "entropy"

    def test_uniform_image_zero_entropy(self, operator):
        """Uniform image has 0 entropy."""
        img = np.full((100, 100, 3), 128, dtype=np.uint8)
        ctx = OperatorContext(image_rgb=img)
        with patch("core.operators.entropy.cv2.cvtColor", side_effect=lambda x, c: x.mean(axis=2).astype(np.uint8)):
            result = operator.execute(ctx)
        assert result.metrics["entropy_score"] == 0.0

    def test_random_noise_high_entropy(self, operator, sample_image):
        """Random noise has high entropy."""
        # sample_image is 100x100 = 10000 pixels
        ctx = OperatorContext(image_rgb=sample_image)
        with (
            patch("core.operators.entropy.cv2.cvtColor", side_effect=lambda x, c: x.mean(axis=2).astype(np.uint8)),
            patch("core.operators.entropy.cv2.calcHist", return_value=np.full((256, 1), 10000 / 256.0)),
        ):
            result = operator.execute(ctx)
        # Should be close to max (~100) or at least high
        assert result.metrics["entropy_score"] > 80.0
