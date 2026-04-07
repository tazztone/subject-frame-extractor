import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from core.managers.registry import ModelRegistry


class TestModelRegistry:
    """
    Tests for core/managers/registry.py
    """

    @pytest.fixture
    def registry(self, mock_logger):
        return ModelRegistry(logger=mock_logger)

    def test_get_or_load_happy_path(self, registry):
        loader = MagicMock(return_value="model_instance")
        val = registry.get_or_load("test_key", loader)
        assert val == "model_instance"
        assert loader.call_count == 1

        # Subsequent call should return cached
        val2 = registry.get_or_load("test_key", loader)
        assert val2 == "model_instance"
        assert loader.call_count == 1

    def test_get_or_load_oom_retry(self, registry):
        """Test that OOM is bubbled up (retry logic moved to specialized loaders)."""
        loader = MagicMock()
        loader.side_effect = [RuntimeError("out of memory"), "success"]
        with pytest.raises(RuntimeError):
            registry.get_or_load("oom_key", loader)

    def test_sticky_failure_logic(self, registry):
        """Test that failures are cached (sticky failure)."""
        loader = MagicMock(side_effect=ValueError("Permanent failure"))

        with pytest.raises(ValueError, match="Permanent failure"):
            registry.get_or_load("fail_key", loader)

        assert "fail_key" in registry._failed_models

        # Subsequent call should return None immediately without calling loader
        loader.reset_mock()
        val = registry.get_or_load("fail_key", loader)
        assert val is None
        loader.assert_not_called()

    def test_clear_and_reload(self, registry):
        """Test that clear() removes models and allows reload."""
        mock_model = MagicMock()
        loader = MagicMock(return_value=mock_model)

        registry.get_or_load("key", loader)
        registry.clear()

        assert "key" not in registry._models
        mock_model.shutdown.assert_called_once()  # or close()

        # Should reload
        registry.get_or_load("key", loader)
        assert loader.call_count == 2

    def test_concurrent_loading(self, registry):
        """Test that concurrent loads only call the factory once."""
        call_count = [0]

        def slow_loader():
            call_count[0] += 1
            time.sleep(0.1)
            return f"model_{call_count[0]}"

        def worker():
            registry.get_or_load("concurrent_key", slow_loader)

        t1 = threading.Thread(target=worker)
        t2 = threading.Thread(target=worker)

        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert call_count[0] == 1
        assert registry._models["concurrent_key"] == "model_1"

    def test_get_tracker_oom_fallback(self, registry):
        """Test tracker init fallback to CPU on OOM."""
        registry.runtime_device_override = None

        with patch.object(registry, "_load_tracker_impl") as mock_load:
            # First call OOMs on CUDA, second succeeds on CPU
            mock_load.side_effect = [RuntimeError("out of memory"), "cpu_tracker"]

            # Force 'cuda' so OOM logic triggers
            with patch("core.managers.registry.get_device", return_value="cuda"):
                tracker = registry.get_tracker("sam3", models_path="/tmp")
                assert tracker == "cpu_tracker"
                assert registry.runtime_device_override == "cpu"
                assert mock_load.call_count == 2
                # Positional args of _load_tracker_impl:
                # 0: model_name, 1: models_path, 2: user_agent, 3: retry_params, 4: device
                assert mock_load.call_args_list[0][0][4] == "cuda"
                assert mock_load.call_args_list[1][0][4] == "cpu"
