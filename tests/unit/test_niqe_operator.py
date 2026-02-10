from unittest.mock import MagicMock, patch
import numpy as np
import pytest
import torch

from core.operators.niqe import NiqeOperator
from core.operators import OperatorContext

class TestNiqeOperator:
    @pytest.fixture
    def operator(self):
        return NiqeOperator()

    def test_config(self, operator):
        cfg = operator.config
        assert cfg.name == "niqe"
        assert cfg.display_name == "NIQE Score"
        assert cfg.category == "quality"

    def test_uninitialized_execution_fails(self, operator):
        ctx = OperatorContext(image_rgb=np.zeros((100, 100, 3), dtype=np.uint8))
        result = operator.execute(ctx)
        assert result.success is False
        assert "not initialized" in result.error

    @patch("pyiqa.create_metric")
    def test_initialize_loads_model(self, mock_create, operator):
        mock_config = MagicMock()
        operator.initialize(mock_config)
        from unittest.mock import ANY
        mock_create.assert_called_once_with("niqe", device=ANY)
        assert operator.model is not None

    @patch("pyiqa.create_metric")
    def test_execute_flow(self, mock_create, operator):
        # Setup mock model
        mock_model = MagicMock()
        mock_model.return_value = torch.tensor([5.0]) # Lower NIQE is better
        mock_model.device = torch.device("cpu")
        mock_create.return_value = mock_model
        
        mock_config = MagicMock()
        mock_config.quality_niqe_offset = 15.0
        mock_config.quality_niqe_scale_factor = 10.0
        
        operator.initialize(mock_config)
        
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        ctx = OperatorContext(image_rgb=img, config=mock_config)
        
        result = operator.execute(ctx)
        
        assert result.success is True
        # (15.0 - 5.0) * 10.0 = 100.0 (capped at 100)
        assert result.metrics["niqe_score"] == 100.0

    @patch("pyiqa.create_metric")
    def test_execute_with_config_overrides(self, mock_create, operator):
        mock_model = MagicMock()
        mock_model.return_value = torch.tensor([10.0])
        mock_model.device = torch.device("cpu")
        mock_create.return_value = mock_model
        
        # Test custom offset/scale
        mock_config = MagicMock()
        mock_config.quality_niqe_offset = 12.0
        mock_config.quality_niqe_scale_factor = 5.0
        
        operator.initialize(mock_config)
        
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        ctx = OperatorContext(image_rgb=img, config=mock_config)
        
        result = operator.execute(ctx)
        
        # (12.0 - 10.0) * 5.0 = 10.0
        assert result.metrics["niqe_score"] == 10.0

        def test_cleanup_clears_model(self, operator):

            operator.model = MagicMock()

            operator.cleanup()

            assert operator.model is None

    