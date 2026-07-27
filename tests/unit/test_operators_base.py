import numpy as np
import pytest

from core.operators.base import (
    FilterDefinition,
    Operator,
    OperatorConfig,
    OperatorContext,
    OperatorResult,
)


def test_operator_config_defaults():
    config = OperatorConfig(name="test_op", display_name="Test Operator")
    assert config.name == "test_op"
    assert config.display_name == "Test Operator"
    assert config.category == "quality"
    assert config.default_enabled is True
    assert config.requires_mask is False
    assert config.requires_face is False
    assert config.requires_tensor is False
    assert config.min_value == 0.0
    assert config.max_value == 100.0
    assert config.description == ""


def test_operator_config_custom():
    config = OperatorConfig(
        name="custom_op",
        display_name="Custom Operator",
        category="face",
        default_enabled=False,
        requires_mask=True,
        requires_face=True,
        requires_tensor=True,
        min_value=-10.0,
        max_value=10.0,
        description="A custom operator",
    )
    assert config.name == "custom_op"
    assert config.category == "face"
    assert config.default_enabled is False
    assert config.requires_mask is True
    assert config.requires_face is True
    assert config.requires_tensor is True
    assert config.min_value == -10.0
    assert config.max_value == 10.0
    assert config.description == "A custom operator"


def test_operator_context_defaults():
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    ctx = OperatorContext(image_rgb=img)

    np.testing.assert_array_equal(ctx.image_rgb, img)
    assert ctx.image_tensor is None
    assert ctx.mask is None
    assert ctx.mask_tensor is None
    assert ctx.config is None
    assert ctx.model_registry is None
    assert ctx.logger is None
    assert ctx.shared_data == {}
    assert ctx.params == {}


def test_operator_context_custom():
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    mask = np.zeros((10, 10), dtype=np.uint8)
    shared = {"key": "value"}
    params = {"param1": 42}

    ctx = OperatorContext(
        image_rgb=img,
        image_tensor="tensor_mock",
        mask=mask,
        mask_tensor="mask_tensor_mock",
        config="config_mock",
        model_registry="registry_mock",
        logger="logger_mock",
        shared_data=shared,
        params=params,
    )

    assert ctx.image_tensor == "tensor_mock"
    np.testing.assert_array_equal(ctx.mask, mask)
    assert ctx.mask_tensor == "mask_tensor_mock"
    assert ctx.config == "config_mock"
    assert ctx.model_registry == "registry_mock"
    assert ctx.logger == "logger_mock"
    assert ctx.shared_data == shared
    assert ctx.params == params


def test_operator_result_defaults():
    result = OperatorResult()
    assert result.metrics == {}
    assert result.data == {}
    assert result.error is None
    assert result.warnings == []
    assert result.success is True


def test_operator_result_with_error():
    result = OperatorResult(error="Something went wrong")
    assert result.error == "Something went wrong"
    assert result.success is False


def test_operator_result_custom():
    result = OperatorResult(
        metrics={"score": 0.95},
        data={"label": "good"},
        warnings=["A minor issue occurred"]
    )
    assert result.metrics == {"score": 0.95}
    assert result.data == {"label": "good"}
    assert result.error is None
    assert result.warnings == ["A minor issue occurred"]
    assert result.success is True


def test_filter_definition_defaults():
    filter_def = FilterDefinition(
        key="test_filter",
        filter_type="range",
        metadata_path=("metrics", "test_filter")
    )
    assert filter_def.key == "test_filter"
    assert filter_def.filter_type == "range"
    assert filter_def.metadata_path == ("metrics", "test_filter")
    assert filter_def.default_min == float("-inf")
    assert filter_def.default_max == float("inf")
    assert filter_def.enabled_key is None
    assert filter_def.reason_low is None
    assert filter_def.reason_high is None
    assert filter_def.reason_range is None
    assert filter_def.reason_missing is None
    assert filter_def.histogram_range == (0.0, 100.0)


def test_filter_definition_custom():
    filter_def = FilterDefinition(
        key="test_filter",
        filter_type="min",
        metadata_path=("test",),
        default_min=0.0,
        default_max=50.0,
        enabled_key="enable_test",
        reason_low="Too low",
        reason_high="Too high",
        reason_range="Out of range",
        reason_missing="Missing",
        histogram_range=(-10.0, 10.0)
    )
    assert filter_def.default_min == 0.0
    assert filter_def.default_max == 50.0
    assert filter_def.enabled_key == "enable_test"
    assert filter_def.reason_low == "Too low"
    assert filter_def.reason_high == "Too high"
    assert filter_def.reason_range == "Out of range"
    assert filter_def.reason_missing == "Missing"
    assert filter_def.histogram_range == (-10.0, 10.0)


def test_operator_protocol():
    class ValidOperator:
        @property
        def config(self) -> OperatorConfig:
            return OperatorConfig(name="valid", display_name="Valid")

        @property
        def filter_definitions(self) -> list[FilterDefinition]:
            return []

        def execute(self, ctx: OperatorContext) -> OperatorResult:
            return OperatorResult()

    class InvalidOperator:
        def execute(self, ctx: OperatorContext) -> OperatorResult:
            return OperatorResult()

    # Test valid operator
    valid_op = ValidOperator()
    assert isinstance(valid_op, Operator)

    # Test invalid operator (missing properties required by protocol)
    invalid_op = InvalidOperator()
    assert not isinstance(invalid_op, Operator)
