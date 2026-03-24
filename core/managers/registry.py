import gc
import logging
import threading
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

import torch

if TYPE_CHECKING:
    from core.config import Config
    from core.logger import AppLogger


class ModelRegistry:
    """Thread-safe registry for lazy loading and caching of heavy ML models."""

    def __init__(self, logger: Optional["AppLogger"] = None):
        self._models: Dict[str, Any] = {}
        self._locks: Dict[str, threading.Lock] = defaultdict(threading.Lock)
        self._registry_lock = threading.RLock()
        self.logger = logger or logging.getLogger(__name__)
        self.runtime_device_override: Optional[str] = None

    def get_or_load(self, key: str, loader_fn: Callable[[], Any]) -> Any:
        """Retrieves a model by key, loading it via loader_fn if not present."""
        with self._registry_lock:
            if key in self._models:
                return self._models[key]

        with self._locks[key]:
            with self._registry_lock:
                if key in self._models:
                    return self._models[key]

            if self.logger:
                self.logger.info(f"Loading model '{key}'...")

            try:
                val = loader_fn()
            except torch.cuda.OutOfMemoryError:
                if self.logger:
                    self.logger.warning(f"CUDA OOM loading '{key}'. Clearing models and retrying...")
                self.clear()
                val = loader_fn()
            except Exception as e:
                if "out of memory" in str(e).lower():
                    if self.logger:
                        self.logger.warning(f"Potential OOM loading '{key}'. Clearing models and retrying...")
                    self.clear()
                    val = loader_fn()
                else:
                    raise e

            with self._registry_lock:
                self._models[key] = val

            if self.logger:
                self.logger.success(f"Model '{key}' loaded successfully.")
        return val

    def clear(self):
        """Clears all models and triggers memory cleanup."""
        if self.logger:
            self.logger.info("Clearing models from registry.")
        with self._registry_lock:
            models_to_clear = list(self._models.items())
            self._models.clear()

        for key, model in models_to_clear:
            try:
                if hasattr(model, "shutdown") and callable(model.shutdown):
                    model.shutdown()
                elif hasattr(model, "close") and callable(model.close):
                    model.close()
            except Exception as e:
                self.logger.warning(f"Error shutting down model {key}: {e}")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_tracker(
        self, model_name: str, models_path: str, user_agent: str, retry_params: tuple, config: "Config"
    ) -> Optional[Any]:
        """Loads subject tracker with CPU fallback on OOM."""
        # Imports moved to inner functions or re-exports in __init__.py

        key = f"tracker_{model_name}"

        def _loader():
            device = self.runtime_device_override or ("cuda" if torch.cuda.is_available() else "cpu")
            try:
                return self._load_tracker_impl(model_name, models_path, user_agent, retry_params, device, config)
            except RuntimeError as e:
                if "out of memory" in str(e) and device == "cuda":
                    self.logger.warning("CUDA OOM during tracker init. Switching to CPU.")
                    torch.cuda.empty_cache()
                    self.runtime_device_override = "cpu"
                    return self._load_tracker_impl(model_name, models_path, user_agent, retry_params, "cpu", config)
                raise e

        try:
            return self.get_or_load(key, _loader)
        except Exception as e:
            self.logger.error(f"Failed to initialize tracker: {e}", exc_info=True)
            return None

    def _load_tracker_impl(
        self, model_name: str, models_path: str, user_agent: str, retry_params: tuple, device: str, config: "Config"
    ):
        from core.error_handling import ErrorHandler
        from core.io_utils import download_model

        from .tracker_factory import build_tracker

        if model_name == "sam2":
            checkpoint_filename = "sam2.1_hiera_tiny.pt"
            url = config.sam2_checkpoint_url
            description = "SAM2.1 Model"
        else:
            # Fallback to sam3 for any other name or "sam3"
            checkpoint_filename = "sam3.pt"
            url = config.sam3_checkpoint_url
            description = "SAM3 Model"

        checkpoint_path = Path(models_path) / checkpoint_filename
        if not checkpoint_path.exists():
            if ".safetensors" in url:
                url = url.replace(".safetensors", ".pt")
            download_model(
                url=url,
                dest_path=checkpoint_path,
                description=description,
                logger=self.logger,
                error_handler=ErrorHandler(self.logger, *retry_params),
                user_agent=user_agent,
                token=config.huggingface_token,
            )
        return build_tracker(model_name, str(checkpoint_path), device=device)
