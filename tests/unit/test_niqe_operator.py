from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from core.operators import OperatorContext
from core.operators.niqe import NiqeOperator
from tests.conftest import createmocktensor


class TestNiqeOperator:
    @pytest.fixture
    def operator(self):
        return NiqeOperator()

    def test_initialize_success(self, operator):
        mock_model = MagicMock()
        mock_pyiqa = MagicMock()
        mock_pyiqa.create_metric.return_value = mock_model

        with patch.dict("sys.modules", {"pyiqa": mock_pyiqa}):
            operator.initialize(MagicMock())
            assert operator.model == mock_model

    def test_initialize_error_paths(self, operator):
        # Case 1: ImportError
        with patch.dict("sys.modules", {"pyiqa": None}):
            operator.initialize(MagicMock())
            assert operator.model is None

        # Case 2: create_metric raises Exception
        mock_pyiqa = MagicMock()
        mock_pyiqa.create_metric.side_effect = Exception("Load fail")
        with patch.dict("sys.modules", {"pyiqa": mock_pyiqa}):
            operator.initialize(MagicMock())
            assert operator.model is None

    def test_execute_flow(self, operator):
        # Setup mock model
        mock_model = MagicMock()
        # Create a mock tensor that behaves like torch.tensor([5.0])
        mock_res = MagicMock()
        mock_res.item.return_value = 5.0
        mock_model.return_value = mock_res
        mock_model.device = "cpu"

        mock_pyiqa = MagicMock()
        mock_pyiqa.create_metric.return_value = mock_model

        from core.config import Config

        mock_config = Config(quality_niqe_offset=15.0, quality_niqe_scale_factor=10.0)

        with patch.dict("sys.modules", {"pyiqa": mock_pyiqa}):
            operator.initialize(mock_config)

            img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            img_tensor = createmocktensor("image_tensor", (1, 3, 100, 100))
            ctx = OperatorContext(image_rgb=img, image_tensor=img_tensor, config=mock_config)

            result = operator.execute(ctx)

            assert result.success is True
            # result = max(0, min(100, (offset - val) * scale))
            # (15.0 - 5.0) * 10.0 = 100.0
            assert result.metrics["niqe_score"] == 100.0
            assert result.metrics["niqe"] == 1.0

    def test_execute_no_model(self, operator):
        ctx = OperatorContext(image_rgb=np.zeros((10, 10, 3)), image_tensor=None, config=MagicMock())
        result = operator.execute(ctx)
        assert result.success is False
        assert "NIQE not initialized" in result.error

    def test_execute_with_config_overrides(self, operator):
        mock_model = MagicMock()
        operator.model = mock_model

        img = np.zeros((10, 10, 3), dtype=np.uint8)
        img_tensor = createmocktensor("img", (1, 3, 10, 10))

        mock_res = MagicMock()
        mock_res.item.return_value = 10.0
        mock_model.return_value = mock_res
        mock_model.device = "cpu"

        # Test with custom offset and scale provided in ctx.config
        from core.config import Config

        custom_config = Config(quality_niqe_offset=100.0, quality_niqe_scale_factor=2.0)
        ctx = OperatorContext(image_rgb=img, image_tensor=img_tensor, config=custom_config)

        result = operator.execute(ctx)
        # (100.0 - 10.0) * 2.0 = 180.0 -> clamped to 100.0
        assert result.metrics["niqe_score"] == 100.0
        assert result.metrics["niqe"] == 1.0

    def test_execute_with_various_config_attributes(self, operator):
        """Test with different combinations of config attributes."""
        mock_model = MagicMock()
        mock_model.device = "cpu"
        operator.model = mock_model

        img_tensor = createmocktensor("img", (1, 3, 10, 10))
        ctx = OperatorContext(image_rgb=np.zeros((10, 10, 3)), image_tensor=img_tensor, config=MagicMock())

        # Mock result
        mock_res = MagicMock()
        mock_res.item.return_value = 5.0
        mock_model.return_value = mock_res

        # Case 1: Config has both offset and scale
        ctx.config.quality_niqe_offset = 20.0
        ctx.config.quality_niqe_scale_factor = 2.0
        res1 = operator.execute(ctx)
        assert res1.metrics["niqe_score"] == 30.0  # (20 - 5)*2

        # Case 2: Config missing offset (should use default 20.0)
        delattr(ctx.config, "quality_niqe_offset")
        # Add niqe_offset as fallback
        ctx.config.niqe_offset = 10.0
        res2 = operator.execute(ctx)
        assert res2.metrics["niqe_score"] == 10.0  # (10 - 5)*2

        # Case 3: Config missing scale (should use default 5.0)
        delattr(ctx.config, "quality_niqe_scale_factor")
        res3 = operator.execute(ctx)
        assert res3.metrics["niqe_score"] == 25.0  # (10 - 5)*5

    def test_execute_with_mask_tensor(self, operator):
        """Test execution when a mask tensor is provided."""
        mock_model = MagicMock()
        mock_model.device = "cpu"
        operator.model = mock_model

        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img_tensor = createmocktensor("img", (1, 3, 100, 100))
        mask_tensor = createmocktensor("mask", (1, 1, 100, 100))

        # Mocking the metric model return
        mock_res = MagicMock()
        mock_res.item.return_value = 5.0
        mock_model.return_value = mock_res

        mock_config = MagicMock()
        mock_config.quality_niqe_offset = 0.0
        mock_config.quality_niqe_scale_factor = 1.0

        ctx = OperatorContext(image_rgb=img, image_tensor=img_tensor, mask_tensor=mask_tensor, config=mock_config)

        result = operator.execute(ctx)
        assert result.success is True

        # Verification: The mask tensor should have been used.
        # Inside NiqeOperator.execute, if mask_tensor exists:
        # called_tensor = image_tensor * mask_tensor.bool().float()
        # model(called_tensor)

        called_args = mock_model.call_args[0][0]
        # Since everything is mocked, called_args will be the result of a calc_result.
        assert "calc_result" in str(called_args)

    def test_execute_exception_handling(self, operator):
        """Test that exceptions during model call are caught."""
        mock_model = MagicMock()
        mock_model.device = "cpu"
        mock_model.side_effect = Exception("Runtime error")
        operator.model = mock_model

        img_tensor = createmocktensor("img", (1, 3, 10, 10))
        ctx = OperatorContext(image_rgb=np.zeros((10, 10, 3)), image_tensor=img_tensor, config=MagicMock())

        result = operator.execute(ctx)
        assert result.success is False
        assert "Runtime error" in result.error
