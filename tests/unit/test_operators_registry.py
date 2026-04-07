from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from core.operators.base import Operator, OperatorConfig, OperatorResult
from core.operators.registry import OperatorRegistry, discover_operators, register_operator, run_operators


class MockOperator(Operator):
    def __init__(self, name="mock_op", requires_tensor=False):
        self._config = OperatorConfig(
            name=name,
            display_name=name.capitalize(),
            description="Test",
            requires_tensor=requires_tensor,
        )
        self.initialized = False
        self.cleaned = False

    @property
    def config(self) -> OperatorConfig:
        return self._config

    def initialize(self, config):
        self.initialized = True

    def execute(self, ctx):
        return OperatorResult(metrics={"val": 0.5})

    def cleanup(self):
        self.cleaned = True


@pytest.fixture(autouse=True)
def clear_registry():
    OperatorRegistry.clear()
    yield
    OperatorRegistry.clear()


def test_registry_register_get():
    op = MockOperator("test")
    OperatorRegistry.register(op)
    assert OperatorRegistry.get("test") == op
    assert "test" in OperatorRegistry.list_names()
    assert OperatorRegistry.list_all()[0].name == "test"


def test_registry_duplicate_registration_warns():
    op1 = MockOperator("dup")
    op2 = MockOperator("dup")
    with patch("core.operators.registry.logger") as mock_logger:
        OperatorRegistry.register(op1)
        OperatorRegistry.register(op2)
        assert mock_logger.warning.called
        assert "already registered" in mock_logger.warning.call_args[0][0]
    assert OperatorRegistry.get("dup") == op2


def test_registry_initialize_cleanup():
    op = MockOperator("test")
    OperatorRegistry.register(op)

    OperatorRegistry.initialize_all(None)
    assert op.initialized is True
    assert "test" in OperatorRegistry._initialized

    OperatorRegistry.cleanup_all()
    assert op.cleaned is True
    assert "test" not in OperatorRegistry._initialized


def test_register_operator_decorator():
    @register_operator
    class DecoratedOperator(MockOperator):
        def __init__(self):
            super().__init__("decorated")

    assert "decorated" in OperatorRegistry.list_names()
    assert isinstance(OperatorRegistry.get("decorated"), DecoratedOperator)


@patch("importlib.import_module")
@patch("pkgutil.iter_modules")
def test_discover_operators(mock_iter, mock_import):
    mock_iter.return_value = [(None, "some_op", False)]
    # Mocking successful import
    mock_import.return_value = MagicMock(__path__=["some/path"])

    names = discover_operators("core.operators")
    assert isinstance(names, list)


def test_run_operators_basic():
    op1 = MockOperator("op1")
    op2 = MockOperator("op2")
    OperatorRegistry.register(op1)
    OperatorRegistry.register(op2)

    img = np.zeros((10, 10, 3), dtype=np.uint8)
    results = run_operators(img, operators=["op1"])

    assert "op1" in results
    assert "op2" not in results
    assert results["op1"].metrics["val"] == 0.5


def test_run_operators_not_found():
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    results = run_operators(img, operators=["missing"])
    assert "missing" in results
    assert results["missing"].success is False
    assert "not found" in results["missing"].error


def test_run_operators_retry_logic():
    class FailingOperator(MockOperator):
        def __init__(self):
            super().__init__("failing")
            self.attempts = 0

        def execute(self, ctx):
            self.attempts += 1
            if self.attempts < 2:
                raise RuntimeError("Transient error")
            return OperatorResult(metrics={"val": 0.8})

    op = FailingOperator()
    OperatorRegistry.register(op)

    logger = MagicMock()
    img = np.zeros((10, 10, 3), dtype=np.uint8)

    # We mock time.sleep to speed up tests
    with patch("time.sleep"):
        results = run_operators(img, operators=["failing"], logger=logger)

    assert results["failing"].success is True
    assert results["failing"].metrics["val"] == 0.8
    assert op.attempts == 2
    assert logger.warning.called


def test_run_operators_with_tensor():
    op = MockOperator("tensor_op", requires_tensor=True)
    OperatorRegistry.register(op)

    img = np.zeros((10, 10, 3), dtype=np.uint8)
    mask = np.zeros((10, 10), dtype=np.uint8)

    with (
        patch("core.operators.registry.is_cuda_available", return_value=False),
        patch("core.operators.registry.torch.from_numpy") as mock_from_numpy,
    ):
        mock_tensor = MagicMock()
        mock_from_numpy.return_value = mock_tensor
        mock_tensor.float.return_value = mock_tensor
        mock_tensor.__truediv__.return_value = mock_tensor
        mock_tensor.permute.return_value = mock_tensor
        mock_tensor.unsqueeze.return_value = mock_tensor
        mock_tensor.to.return_value = mock_tensor

        run_operators(image_rgb=img, mask=mask, operators=["tensor_op"])
        assert mock_from_numpy.called


def test_run_operators_quality_score_last():
    op_q = MockOperator("quality_score")
    op_a = MockOperator("a")
    OperatorRegistry.register(op_q)
    OperatorRegistry.register(op_a)

    img = np.zeros((10, 10, 3), dtype=np.uint8)
    with patch.object(OperatorRegistry, "get", wraps=OperatorRegistry.get) as mock_get:
        run_operators(img, operators=["quality_score", "a"])
        # Calls: get(quality_score) for tensor check, get(a) for tensor check,
        # then get(a) for execution, then get(quality_score) for execution.
        calls = [c.args[0] for c in mock_get.call_args_list]
        exec_calls = [c for c in calls if c in ["a", "quality_score"]]
        assert exec_calls[-2:] == ["a", "quality_score"]
