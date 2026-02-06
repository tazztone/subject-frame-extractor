"""
Unit tests for the Operator framework.

Tests cover:
- OperatorConfig default values and UI metadata
- OperatorContext creation and fields
- OperatorResult success/error handling
- OperatorRegistry lifecycle management
- run_operators bridge function
- Protocol compliance
"""

from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

from core.operators import (
    Operator,
    OperatorConfig,
    OperatorContext,
    OperatorResult,
    OperatorRegistry,
    register_operator,
    run_operators,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_image():
    """100x100 RGB image with random noise."""
    return np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)


@pytest.fixture
def sample_mask():
    """100x100 grayscale mask (center region active)."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[25:75, 25:75] = 255
    return mask


@pytest.fixture
def mock_config():
    """Mock Config object with sharpness_base_scale."""
    config = MagicMock()
    config.sharpness_base_scale = 2500.0
    return config


@pytest.fixture
def sharp_image():
    """High-frequency checkerboard pattern (sharp)."""
    pattern = np.indices((100, 100)).sum(axis=0) % 2
    img = (pattern * 255).astype(np.uint8)
    return np.stack([img, img, img], axis=-1)


@pytest.fixture
def blurry_image():
    """Gaussian blurred uniform gray (blurry)."""
    gray = np.full((100, 100, 3), 128, dtype=np.uint8)
    return cv2.GaussianBlur(gray, (21, 21), 0)


@pytest.fixture(autouse=True)
def clear_registry():
    """Clear operator registry before each test."""
    OperatorRegistry.clear()
    yield
    OperatorRegistry.clear()


# ============================================================================
# TestOperatorConfig
# ============================================================================


class TestOperatorConfig:
    """Tests for OperatorConfig dataclass."""

    def test_required_fields(self):
        """Config requires name and display_name."""
        config = OperatorConfig(name="test", display_name="Test Metric")
        assert config.name == "test"
        assert config.display_name == "Test Metric"

    def test_default_values(self):
        """Config has sensible defaults."""
        config = OperatorConfig(name="test", display_name="Test")
        assert config.category == "quality"
        assert config.default_enabled is True
        assert config.requires_mask is False
        assert config.requires_face is False
        assert config.min_value == 0.0
        assert config.max_value == 100.0
        assert config.description == ""

    def test_ui_metadata(self):
        """Config supports UI metadata fields."""
        config = OperatorConfig(
            name="test",
            display_name="Test Metric",
            min_value=0.0,
            max_value=50.0,
            description="A test metric for testing",
        )
        assert config.min_value == 0.0
        assert config.max_value == 50.0
        assert config.description == "A test metric for testing"


# ============================================================================
# TestOperatorContext
# ============================================================================


class TestOperatorContext:
    """Tests for OperatorContext dataclass."""

    def test_minimal_creation(self, sample_image):
        """Context can be created with just image_rgb."""
        ctx = OperatorContext(image_rgb=sample_image)
        assert ctx.image_rgb is sample_image
        assert ctx.mask is None
        assert ctx.config is None
        assert ctx.params == {}

    def test_all_fields(self, sample_image, sample_mask, mock_config):
        """Context supports all optional fields."""
        ctx = OperatorContext(
            image_rgb=sample_image,
            mask=sample_mask,
            config=mock_config,
            params={"key": "value"},
        )
        assert ctx.mask is sample_mask
        assert ctx.config is mock_config
        assert ctx.params == {"key": "value"}


# ============================================================================
# TestOperatorResult
# ============================================================================


class TestOperatorResult:
    """Tests for OperatorResult dataclass."""

    def test_success_case(self):
        """Result with metrics only indicates success."""
        result = OperatorResult(metrics={"score": 85.0})
        assert result.metrics == {"score": 85.0}
        assert result.error is None
        assert result.success is True

    def test_error_case(self):
        """Result with error indicates failure."""
        result = OperatorResult(metrics={}, error="Something went wrong")
        assert result.error == "Something went wrong"
        assert result.success is False

    def test_warnings(self):
        """Result can include non-fatal warnings."""
        result = OperatorResult(
            metrics={"score": 75.0},
            warnings=["Low confidence", "Unusual input"],
        )
        assert len(result.warnings) == 2
        assert result.success is True  # Warnings don't indicate failure


# ============================================================================
# TestOperatorRegistry
# ============================================================================


class TestOperatorRegistry:
    """Tests for OperatorRegistry."""

    def test_register_and_get(self, sample_image):
        """Can register and retrieve an operator."""

        class MockOperator:
            @property
            def config(self):
                return OperatorConfig(name="mock", display_name="Mock")

            def execute(self, ctx):
                return OperatorResult(metrics={"mock_score": 50.0})

        op = MockOperator()
        OperatorRegistry.register(op)

        retrieved = OperatorRegistry.get("mock")
        assert retrieved is op

    def test_get_unknown_returns_none(self):
        """Getting unknown operator returns None."""
        assert OperatorRegistry.get("nonexistent") is None

    def test_list_all(self):
        """list_all returns all operator configs."""

        class Op1:
            @property
            def config(self):
                return OperatorConfig(name="op1", display_name="Op 1")

            def execute(self, ctx):
                return OperatorResult()

        class Op2:
            @property
            def config(self):
                return OperatorConfig(name="op2", display_name="Op 2")

            def execute(self, ctx):
                return OperatorResult()

        OperatorRegistry.register(Op1())
        OperatorRegistry.register(Op2())

        configs = OperatorRegistry.list_all()
        names = [c.name for c in configs]
        assert "op1" in names
        assert "op2" in names

    def test_initialize_all(self, mock_config):
        """initialize_all calls initialize on operators."""
        initialized = []

        class StatefulOp:
            @property
            def config(self):
                return OperatorConfig(name="stateful", display_name="Stateful")

            def execute(self, ctx):
                return OperatorResult()

            def initialize(self, config):
                initialized.append(config)

        OperatorRegistry.register(StatefulOp())
        OperatorRegistry.initialize_all(mock_config)

        assert len(initialized) == 1
        assert initialized[0] is mock_config

    def test_cleanup_all(self):
        """cleanup_all calls cleanup on initialized operators."""
        cleaned = []

        class StatefulOp:
            @property
            def config(self):
                return OperatorConfig(name="cleanup_test", display_name="Cleanup")

            def execute(self, ctx):
                return OperatorResult()

            def initialize(self, config):
                pass

            def cleanup(self):
                cleaned.append(True)

        OperatorRegistry.register(StatefulOp())
        OperatorRegistry.initialize_all(None)
        OperatorRegistry.cleanup_all()

        assert len(cleaned) == 1


# ============================================================================
# TestRegisterDecorator
# ============================================================================


class TestRegisterDecorator:
    """Tests for @register_operator decorator."""

    def test_decorator_registers(self):
        """Decorator registers the operator."""

        @register_operator
        class DecoratedOp:
            @property
            def config(self):
                return OperatorConfig(name="decorated", display_name="Decorated")

            def execute(self, ctx):
                return OperatorResult()

        assert OperatorRegistry.get("decorated") is not None


# ============================================================================
# TestRunOperators
# ============================================================================


class TestRunOperators:
    """Tests for run_operators bridge function."""

    def test_runs_all_operators(self, sample_image):
        """run_operators executes all registered operators."""

        @register_operator
        class Op1:
            @property
            def config(self):
                return OperatorConfig(name="bridge_op1", display_name="Op1")

            def execute(self, ctx):
                return OperatorResult(metrics={"op1_score": 10.0})

        @register_operator
        class Op2:
            @property
            def config(self):
                return OperatorConfig(name="bridge_op2", display_name="Op2")

            def execute(self, ctx):
                return OperatorResult(metrics={"op2_score": 20.0})

        results = run_operators(sample_image)

        assert "bridge_op1" in results
        assert "bridge_op2" in results
        assert results["bridge_op1"].metrics["op1_score"] == 10.0

    def test_runs_specific_operators(self, sample_image):
        """run_operators can run specific operators only."""

        @register_operator
        class OpA:
            @property
            def config(self):
                return OperatorConfig(name="specific_a", display_name="A")

            def execute(self, ctx):
                return OperatorResult(metrics={"a": 1.0})

        @register_operator
        class OpB:
            @property
            def config(self):
                return OperatorConfig(name="specific_b", display_name="B")

            def execute(self, ctx):
                return OperatorResult(metrics={"b": 2.0})

        results = run_operators(sample_image, operators=["specific_a"])

        assert "specific_a" in results
        assert "specific_b" not in results

    def test_error_isolation(self, sample_image):
        """One operator failing doesn't break others."""

        @register_operator
        class GoodOp:
            @property
            def config(self):
                return OperatorConfig(name="iso_good", display_name="Good")

            def execute(self, ctx):
                return OperatorResult(metrics={"good": 100.0})

        @register_operator
        class BadOp:
            @property
            def config(self):
                return OperatorConfig(name="iso_bad", display_name="Bad")

            def execute(self, ctx):
                raise ValueError("Intentional error")

        results = run_operators(sample_image)

        assert results["iso_good"].success is True
        assert results["iso_bad"].success is False
        assert "Intentional error" in results["iso_bad"].error

    def test_unknown_operator_error(self, sample_image):
        """Requesting unknown operator returns error result."""
        results = run_operators(sample_image, operators=["nonexistent"])

        assert "nonexistent" in results
        assert results["nonexistent"].success is False
        assert "not found" in results["nonexistent"].error


# ============================================================================
# TestOperatorProtocol
# ============================================================================


class TestOperatorProtocol:
    """Tests for Operator Protocol compliance."""

    def test_protocol_is_runtime_checkable(self):
        """Operator Protocol is runtime checkable."""

        class CompliantOp:
            @property
            def config(self):
                return OperatorConfig(name="compliant", display_name="Compliant")

            def execute(self, ctx):
                return OperatorResult()

            def initialize(self, config):
                pass

            def cleanup(self):
                pass

        assert isinstance(CompliantOp(), Operator)

    def test_minimal_implementation(self):
        """Operator works with just config and execute."""

        class MinimalOp:
            @property
            def config(self):
                return OperatorConfig(name="minimal", display_name="Minimal")

            def execute(self, ctx):
                return OperatorResult(metrics={"value": 42.0})

        op = MinimalOp()
        result = op.execute(OperatorContext(image_rgb=np.zeros((10, 10, 3))))
        assert result.metrics["value"] == 42.0


# ============================================================================
# TestSharpnessOperator
# ============================================================================


class TestSharpnessOperator:
    """Tests for the SharpnessOperator implementation."""

    @pytest.fixture
    def sharpness_operator(self):
        """Fresh SharpnessOperator instance."""
        # Import here to avoid registration conflicts with autouse fixture
        from core.operators.sharpness import SharpnessOperator
        return SharpnessOperator()

    def test_config_values(self, sharpness_operator):
        """Config has expected values."""
        cfg = sharpness_operator.config
        assert cfg.name == "sharpness"
        assert cfg.display_name == "Sharpness Score"
        assert cfg.category == "quality"
        assert cfg.requires_mask is True
        assert cfg.min_value == 0.0
        assert cfg.max_value == 100.0

    def test_execute_returns_operator_result(self, sharpness_operator, sample_image):
        """Execute returns OperatorResult type."""
        ctx = OperatorContext(image_rgb=sample_image)
        result = sharpness_operator.execute(ctx)
        assert isinstance(result, OperatorResult)

    def test_execute_has_sharpness_score(self, sharpness_operator, sample_image):
        """Result contains sharpness_score metric."""
        ctx = OperatorContext(image_rgb=sample_image)
        result = sharpness_operator.execute(ctx)
        assert "sharpness_score" in result.metrics

    def test_score_in_valid_range(self, sharpness_operator, sample_image):
        """Score is between 0 and 100."""
        ctx = OperatorContext(image_rgb=sample_image)
        result = sharpness_operator.execute(ctx)
        score = result.metrics["sharpness_score"]
        assert 0.0 <= score <= 100.0

    def test_sharp_higher_than_blurry(self, sharpness_operator, sharp_image, blurry_image):
        """Sharp image scores higher than blurry image."""
        sharp_ctx = OperatorContext(image_rgb=sharp_image)
        blurry_ctx = OperatorContext(image_rgb=blurry_image)
        
        sharp_result = sharpness_operator.execute(sharp_ctx)
        blurry_result = sharpness_operator.execute(blurry_ctx)
        
        assert sharp_result.metrics["sharpness_score"] > blurry_result.metrics["sharpness_score"]

    def test_with_mask_changes_score(self, sharpness_operator, sample_mask):
        """Score differs with mask vs without mask."""
        # Use a gradient image where center (masked area) differs from edges
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        # High frequency in corners (outside mask)
        img[:25, :25] = np.indices((25, 25)).sum(axis=0)[:, :, np.newaxis] % 2 * 255
        img[75:, 75:] = np.indices((25, 25)).sum(axis=0)[:, :, np.newaxis] % 2 * 255
        # Low frequency in center (inside mask)
        img[25:75, 25:75] = 128
        
        no_mask_ctx = OperatorContext(image_rgb=img)
        with_mask_ctx = OperatorContext(image_rgb=img, mask=sample_mask)
        
        no_mask_result = sharpness_operator.execute(no_mask_ctx)
        with_mask_result = sharpness_operator.execute(with_mask_ctx)
        
        # Masked version should be lower (blurry center only)
        assert no_mask_result.metrics["sharpness_score"] > with_mask_result.metrics["sharpness_score"]

    def test_error_handling(self, sharpness_operator):
        """Invalid input returns OperatorResult with error."""
        # Pass invalid image data
        ctx = OperatorContext(image_rgb=np.array([]))  # Empty array
        result = sharpness_operator.execute(ctx)
        
        assert result.success is False
        assert result.error is not None

    def test_uses_config_scale(self, sharpness_operator, sample_image):
        """Operator uses sharpness_base_scale from config."""
        mock_cfg = MagicMock()
        mock_cfg.sharpness_base_scale = 500.0  # Lower scale = higher scores
        
        ctx_default = OperatorContext(image_rgb=sample_image)
        ctx_custom = OperatorContext(image_rgb=sample_image, config=mock_cfg)
        
        result_default = sharpness_operator.execute(ctx_default)
        result_custom = sharpness_operator.execute(ctx_custom)
        
        # Lower scale should produce higher score for same image
        assert result_custom.metrics["sharpness_score"] >= result_default.metrics["sharpness_score"]
