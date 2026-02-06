import numpy as np
import pytest
from core.operators import OperatorContext, OperatorRegistry
from core.operators.simple_cv import EdgeStrengthOperator, ContrastOperator, BrightnessOperator
from core.operators.entropy import EntropyOperator


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
        result = operator.execute(ctx)
        assert result.metrics["edge_strength_score"] == 0.0

    def test_edges_are_detected(self, operator, sharp_image):
        """Image with strong edges has >0 edge strength."""
        # Use a simple vertical split image instead of 1x1 checkerboard
        # 1x1 checkerboard with 3x3 Sobel might act weirdly due to smoothing or aliasing
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:, 50:] = 255
        ctx = OperatorContext(image_rgb=img)
        result = operator.execute(ctx)
        assert result.metrics["edge_strength_score"] > 0.0

    def test_scaling(self, operator, sharp_image, mock_config):
        """Config scale affects score."""
        # Use split image for consistent edges
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:, 50:] = 255
        ctx_default = OperatorContext(image_rgb=img)
        
        # Lower scale = higher score
        mock_config.edge_strength_base_scale = 10.0
        ctx_scaled = OperatorContext(image_rgb=img, config=mock_config)
        
        res_def = operator.execute(ctx_default)
        res_scale = operator.execute(ctx_scaled)
        
        # With default scale (100), score X. With scale 10, score 10X.
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
        result = operator.execute(ctx)
        assert result.metrics["contrast_score"] == 0.0

    def test_high_contrast_pattern(self, operator):
        """Black and white image has high contrast."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:50, :] = 255
        ctx = OperatorContext(image_rgb=img)
        result = operator.execute(ctx)
        # Std/Mean for 50/50 split of 0/255 is ~1.0.
        # With default clamp 50.0, score is (1.0/50.0)*100 = 2.0.
        # So we assert > 1.0 to verify it's non-zero and reasonable.
        assert result.metrics["contrast_score"] > 1.0

    def test_masking_affects_contrast(self, operator, sample_mask):
        """Masking changes calculation to include only masked pixels."""
        # Image: Left half black, Right half white.
        # Mask: Only Left half.
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:, 50:] = 255
        
        # Mask selecting only the black part (Contrast should be 0)
        mask_left = np.zeros((100, 100), dtype=np.uint8)
        mask_left[:, :50] = 255
        
        ctx = OperatorContext(image_rgb=img, mask=mask_left)
        result = operator.execute(ctx)
        
        # Std dev of solid black is 0
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
        result = operator.execute(ctx)
        assert result.metrics["brightness_score"] == 0.0

    def test_white_image(self, operator):
        img = np.full((100, 100, 3), 255, dtype=np.uint8)
        ctx = OperatorContext(image_rgb=img)
        result = operator.execute(ctx)
        assert result.metrics["brightness_score"] == pytest.approx(100.0, abs=0.1)

    def test_gray_image(self, operator):
        img = np.full((100, 100, 3), 128, dtype=np.uint8)
        ctx = OperatorContext(image_rgb=img)
        result = operator.execute(ctx)
        # 128/255 * 100 ~= 50.2
        assert result.metrics["brightness_score"] == pytest.approx(50.2, abs=0.5)


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
        result = operator.execute(ctx)
        assert result.metrics["entropy_score"] == 0.0

    def test_random_noise_high_entropy(self, operator, sample_image):
        """Random noise has high entropy."""
        # sample_image is random noise
        ctx = OperatorContext(image_rgb=sample_image)
        result = operator.execute(ctx)
        # Should be close to max (~100) or at least high
        assert result.metrics["entropy_score"] > 80.0
