import pytest
from unittest.mock import MagicMock, patch
import numpy as np
import torch
import sys
from core.operators import OperatorContext, OperatorResult
from core.operators.niqe import NiqeOperator

class TestNiqeOperator:
    @pytest.fixture
    def operator(self):
        return NiqeOperator()

    @pytest.fixture
    def sample_image(self):
        return np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

    def test_config(self, operator):
        assert operator.config.name == "niqe"
        
    def test_uninitialized_execution_fails(self, operator, sample_image):
        """Execute fails if initialize() not called."""
        ctx = OperatorContext(image_rgb=sample_image)
        result = operator.execute(ctx)
        assert result.success is False
        assert "not initialized" in result.error

    @patch("core.operators.niqe.torch")
    def test_initialize_loads_model(self, mock_torch, operator):
        """Initialize loads model via pyiqa."""
        mock_torch.cuda.is_available.return_value = False
        mock_config = MagicMock()
        
        # Mock pyiqa in sys.modules
        mock_pyiqa = MagicMock()
        with patch.dict(sys.modules, {"pyiqa": mock_pyiqa}):
            operator.initialize(mock_config)
            
            mock_pyiqa.create_metric.assert_called_once()
            assert operator.model is not None
            assert operator.device == "cpu"

    @patch("core.operators.niqe.torch") 
    def test_execute_flow(self, mock_torch, operator, sample_image):
        """Execute runs model and normalizes score."""
        # Setup mocks
        mock_model = MagicMock()
        mock_model.return_value = 10.0 # Raw NIQE score
        
        # Configure torch mocks for the tensor chain
        mock_tensor = MagicMock()
        mock_torch.from_numpy.return_value.float.return_value.div.return_value.permute.return_value.unsqueeze.return_value = mock_tensor
        # Configure default device
        mock_torch.cuda.is_available.return_value = False
        
        # initialize uses pyiqa
        mock_pyiqa = MagicMock()
        mock_pyiqa.create_metric.return_value = mock_model
        
        with patch.dict(sys.modules, {"pyiqa": mock_pyiqa}):
            operator.initialize(None)
            
            # Execute
            ctx = OperatorContext(image_rgb=sample_image)
            result = operator.execute(ctx)
            
            assert result.success is True
            # Score calculation: 100 - (10.0 * 2) = 80.0
            assert result.metrics["niqe_score"] == 80.0

    @patch("core.operators.niqe.torch")
    def test_execute_with_config_overrides(self, mock_torch, operator, sample_image):
        """Config overrides affect normalization."""
        mock_model = MagicMock()
        mock_model.return_value = 10.0
        
        # Force device valid
        mock_torch.cuda.is_available.return_value = False
        mock_torch.from_numpy.return_value.float.return_value.div.return_value.permute.return_value.unsqueeze.return_value = MagicMock()

        mock_pyiqa = MagicMock()
        mock_pyiqa.create_metric.return_value = mock_model
        
        with patch.dict(sys.modules, {"pyiqa": mock_pyiqa}):
            operator.initialize(None)
            
            mock_cfg = MagicMock()
            mock_cfg.quality_niqe_offset = 50.0
            mock_cfg.quality_niqe_scale_factor = 1.0
            
            ctx = OperatorContext(image_rgb=sample_image, config=mock_cfg)
            result = operator.execute(ctx)
            
            # Score: 50 - (10 * 1) = 40.0
            assert result.metrics["niqe_score"] == 40.0

    def test_cleanup_clears_model(self, operator):
        operator.model = "something"
        operator.cleanup()
        assert operator.model is None
