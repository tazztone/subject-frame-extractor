"""
Unit tests for ModelRegistry tracker loading.
"""

from unittest.mock import MagicMock, patch

import pytest

from core.managers.registry import ModelRegistry


@pytest.fixture
def mock_config():
    config = MagicMock()
    config.sam2_checkpoint_url = "http://example.com/sam2.pt"
    config.sam3_checkpoint_url = "http://example.com/sam3.safetensors"  # Test safetensors replacement
    config.models_dir = "models"
    config.retry_max_attempts = 3
    config.retry_backoff_seconds = [1, 2]
    config.user_agent = "test-agent"
    config.huggingface_token = "test-token"
    config.default_tracker_model_name = "sam2"
    return config


@pytest.fixture
def registry():
    # ModelRegistry uses standard logging by default, but we can pass a mock logger
    mock_logger = MagicMock()
    # Ensure all logger methods exist
    mock_logger.success = MagicMock()
    mock_logger.info = MagicMock()
    mock_logger.warning = MagicMock()
    mock_logger.error = MagicMock()
    return ModelRegistry(logger=mock_logger)


@patch("core.io_utils.download_model")
@patch("core.managers.tracker_factory.build_tracker")
@patch("core.managers.registry.Path.exists", return_value=True)
def test_get_tracker_sam2(mock_exists, mock_build, mock_download, registry, mock_config):
    """Test loading SAM2 tracker and cache hit."""
    mock_build.return_value = MagicMock(name="SAM2Instance")

    # First load
    tracker1 = registry.get_tracker(
        model_name="sam2", models_path="models", user_agent="test-agent", retry_params=(3, (1, 2)), config=mock_config
    )

    assert tracker1 == mock_build.return_value
    assert mock_build.call_count == 1

    # Second load (cache hit)
    tracker2 = registry.get_tracker(
        model_name="sam2", models_path="models", user_agent="test-agent", retry_params=(3, (1, 2)), config=mock_config
    )

    assert tracker2 == tracker1
    assert mock_build.call_count == 1  # Still 1


@patch("core.io_utils.download_model")
@patch("core.managers.tracker_factory.build_tracker")
@patch("core.managers.registry.Path.exists", return_value=True)
def test_get_tracker_sam3_safetensors(mock_exists, mock_build, mock_download, registry, mock_config):
    """Test loading SAM3 tracker with safetensors replacement."""
    mock_build.return_value = MagicMock(name="SAM3Instance")

    registry.get_tracker(
        model_name="sam3", models_path="models", user_agent="test-agent", retry_params=(3, (1, 2)), config=mock_config
    )

    # Verify .safetensors was replaced by .pt in URL if it was in the config
    # In my config fixture I set it to .safetensors
    mock_download.assert_not_called()  # Because exists=True

    # To test replacement we need exists=False
    with patch("core.managers.registry.Path.exists", return_value=False), patch("core.managers.registry.Path.mkdir"):
        registry.get_tracker(
            model_name="sam3_new",  # different key to avoid cache
            models_path="models",
            user_agent="test-agent",
            retry_params=(3, (1, 2)),
            config=mock_config,
        )
        assert mock_download.called
        assert ".pt" in mock_download.call_args[1]["url"]
        assert ".safetensors" not in mock_download.call_args[1]["url"]


def test_clear_registry(registry):
    """Test clearing the registry and shutting down models."""
    mock_model = MagicMock()
    mock_model.shutdown = MagicMock()
    registry._models["test"] = mock_model

    registry.clear()

    assert "test" not in registry._models
    assert mock_model.shutdown.called


def test_get_tracker_failure_path(registry, mock_config):
    """Test error handling when loader fails."""
    with patch.object(registry, "_load_tracker_impl", side_effect=Exception("Failed")):
        tracker = registry.get_tracker(
            model_name="sam2",
            models_path="models",
            user_agent="test-agent",
            retry_params=(3, (1, 2)),
            config=mock_config,
        )
        assert tracker is None
        assert registry.logger.error.called


def test_get_tracker_oom_fallback(registry, mock_config):
    """Test CPU fallback on CUDA OOM."""
    # We need to ensure the logger mock has 'success' because get_or_load calls it
    with (
        patch.object(registry, "_load_tracker_impl") as mock_load,
        patch("core.managers.registry.torch.cuda.is_available", return_value=True),
        patch("core.managers.registry.torch.cuda.empty_cache") as mock_empty,
    ):
        # First call raises OOM
        mock_load.side_effect = [RuntimeError("out of memory"), MagicMock(name="CPUTracker")]

        tracker = registry.get_tracker(
            model_name="sam2_oom",
            models_path="models",
            user_agent="test-agent",
            retry_params=(3, (1, 2)),
            config=mock_config,
        )

        assert tracker is not None
        assert mock_load.call_count == 2
        # Check that first call was cuda and second was cpu
        assert mock_load.call_args_list[0][0][4] == "cuda"
        assert mock_load.call_args_list[1][0][4] == "cpu"
        assert mock_empty.called
