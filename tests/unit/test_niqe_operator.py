from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from core.operators import OperatorContext
from core.operators.niqe import NiqeOperator


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
        mock_model.device = torch.device("cpu")

        mock_pyiqa = MagicMock()
        mock_pyiqa.create_metric.return_value = mock_model

        from core.config import Config

        mock_config = Config(quality_niqe_offset=15.0, quality_niqe_scale_factor=10.0)

        with patch.dict("sys.modules", {"pyiqa": mock_pyiqa}):
            operator.initialize(mock_config)

            img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            img_tensor = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0)
            ctx = OperatorContext(image_rgb=img, image_tensor=img_tensor, config=mock_config)

            result = operator.execute(ctx)

            assert result.success is True
            # (15.0 - 5.0) * 10.0 = 100.0 (capped at 100)
            assert result.metrics["niqe_score"] == 100.0

    def test_execute_with_config_overrides(self, operator):
        mock_model = MagicMock()
        mock_res = MagicMock()
        mock_res.item.return_value = 10.0
        mock_model.return_value = mock_res
        mock_model.device = torch.device("cpu")

        mock_pyiqa = MagicMock()
        mock_pyiqa.create_metric.return_value = mock_model

        # Test custom offset/scale
        from core.config import Config

        mock_config = Config(quality_niqe_offset=12.0, quality_niqe_scale_factor=5.0)

        with patch.dict("sys.modules", {"pyiqa": mock_pyiqa}):
            operator.initialize(mock_config)

            img = np.zeros((100, 100, 3), dtype=np.uint8)
            img_tensor = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0)
            ctx = OperatorContext(image_rgb=img, image_tensor=img_tensor, config=mock_config)

            result = operator.execute(ctx)

            # (12.0 - 10.0) * 5.0 = 10.0
            assert result.metrics["niqe_score"] == 10.0

    def test_execute_with_various_config_attributes(self, operator):
        """Test fallback to 'niqe_offset' if 'quality_niqe_offset' is missing."""
        mock_model = MagicMock()
        mock_res = MagicMock()
        mock_res.item.return_value = 10.0
        mock_model.return_value = mock_res
        mock_model.device = torch.device("cpu")

        mock_pyiqa = MagicMock()
        mock_pyiqa.create_metric.return_value = mock_model

        class SimpleConfig:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        mock_config = SimpleConfig(niqe_offset=20.0, quality_niqe_scale_factor=2.0)

        with patch.dict("sys.modules", {"pyiqa": mock_pyiqa}):
            operator.initialize(mock_config)

            img = np.zeros((10, 10, 3), dtype=np.uint8)
            img_tensor = torch.zeros((1, 3, 10, 10))
            ctx = OperatorContext(image_rgb=img, image_tensor=img_tensor, config=mock_config)

            result = operator.execute(ctx)
            # (20.0 - 10.0) * 2.0 = 20.0
            assert pytest.approx(result.metrics["niqe_score"]) == 20.0

    def test_execute_with_mask_tensor(self, operator):
        """Test that mask_tensor is used in execute."""
        mock_model = MagicMock()
        mock_res = MagicMock()
        mock_res.item.return_value = 5.0
        mock_model.return_value = mock_res
        mock_model.device = torch.device("cpu")

        # Manually set the model to avoid pyiqa import/init issues
        operator.model = mock_model
        operator.device = "cpu"

        img_tensor = torch.ones((1, 3, 10, 10))
        mask_tensor = torch.zeros((1, 1, 10, 10))
        mask_tensor[:, :, :5, :] = 1.0  # Top half only

        ctx = OperatorContext(image_rgb=np.zeros((10, 10, 3)), image_tensor=img_tensor, mask_tensor=mask_tensor)

        operator.execute(ctx)

        # Verify the model was called
        assert mock_model.called, "mock_model was not called"
        # Verify the model was called with the masked tensor
        called_tensor = mock_model.call_args[0][0]
        assert torch.all(called_tensor == mask_tensor)

    def test_execute_error_paths(self, operator):
        # Case 1: Model returns NaN or error (simulated via exception)
        mock_model = MagicMock()
        mock_model.side_effect = Exception("NIQE failure")
        mock_model.device = torch.device("cpu")

        mock_pyiqa = MagicMock()
        mock_pyiqa.create_metric.return_value = mock_model

        with patch.dict("sys.modules", {"pyiqa": mock_pyiqa}):
            operator.initialize(MagicMock())

            ctx = OperatorContext(image_rgb=np.zeros((10, 10, 3)), image_tensor=torch.zeros((1, 3, 10, 10)))
            result = operator.execute(ctx)
            assert result.success is False
            assert "NIQE failure" in result.error

    def test_cleanup_clears_model(self, operator):
        operator.model = MagicMock()
        operator.cleanup()
        assert operator.model is None
