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

    from .sam3 import SAM3Wrapper


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

            val = loader_fn()

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
    ) -> Optional["SAM3Wrapper"]:
        """Loads SAM3 tracker with CPU fallback on OOM."""
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
        from core.utils import download_model

        from .sam3 import SAM3Wrapper

        checkpoint_path = Path(models_path) / "sam3.pt"
        if not checkpoint_path.exists():
            url = config.sam3_checkpoint_url
            if ".safetensors" in url:
                url = url.replace(".safetensors", ".pt")
            download_model(
                url=url,
                dest_path=checkpoint_path,
                description="SAM3 Model",
                logger=self.logger,
                error_handler=ErrorHandler(self.logger, *retry_params),
                user_agent=user_agent,
                token=config.huggingface_token,
            )
        return SAM3Wrapper(str(checkpoint_path), device=device)
