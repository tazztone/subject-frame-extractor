from __future__ import annotations

import gc
import logging
import threading
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Set

from core.utils.device import empty_cache, get_device, is_cuda_available

if TYPE_CHECKING:
    from core.config import Config
    from core.logger import LoggerLike


class ModelRegistry:
    """Thread-safe registry for lazy loading and caching of heavy ML models."""

    def __init__(self, logger: Optional["LoggerLike"] = None):
        self._models: Dict[str, Any] = {}
        self._failed_models: Set[str] = set()
        self._locks: Dict[str, threading.Lock] = defaultdict(threading.Lock)
        self._registry_lock = threading.RLock()
        self.logger: "LoggerLike" = logger or logging.getLogger(__name__)
        self.runtime_device_override: Optional[str] = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.clear()

    def get_or_load(self, key: str, loader_fn: Callable[[], Any]) -> Any:
        """Retrieves a model by key, loading it via loader_fn if not present."""
        with self._registry_lock:
            if key in self._models:
                return self._models[key]

            if key in self._failed_models:
                return None

        with self._locks[key]:
            with self._registry_lock:
                if key in self._models:
                    return self._models[key]

            if self.logger:
                self.logger.info(f"Loading model '{key}'...")

            try:
                val = loader_fn()
            except Exception as e:
                with self._registry_lock:
                    self._failed_models.add(key)
                raise e

            with self._registry_lock:
                self._models[key] = val

            if self.logger:
                if hasattr(self.logger, "success"):
                    self.logger.success(f"Model '{key}' loaded successfully.")  # type: ignore
                else:
                    self.logger.info(f"Model '{key}' loaded successfully.")
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
        if is_cuda_available():
            empty_cache()

    def get_tracker(
        self,
        model_name: str,
        models_path: Optional[str] = None,
        user_agent: Optional[str] = None,
        retry_params: Optional[tuple] = None,
        config: Optional["Config"] = None,
    ) -> Optional[Any]:
        """Loads subject tracker with CPU fallback on OOM."""
        # Imports moved to inner functions or re-exports in __init__.py

        # Fallback to self.logger.config if not provided
        _config = config
        if _config is None and hasattr(self, "default_config"):
            _config = getattr(self, "default_config")
        elif _config is None and hasattr(self.logger, "config") and not isinstance(getattr(self.logger, "config"), str):
            _config = self.logger.config  # type: ignore

        _models_path = models_path or (str(_config.models_dir) if _config else None)
        _user_agent = user_agent or (_config.user_agent if _config else "SubjectFrameExtractor")
        _retry_params = retry_params or (1, [1])

        if not _models_path:
            self.logger.error("get_tracker: models_path is None and no config provided — cannot locate checkpoint.")
            return None

        key = f"tracker_{model_name}"

        def _loader():
            device = self.runtime_device_override or get_device()
            try:
                return self._load_tracker_impl(model_name, _models_path, _user_agent, _retry_params, device, _config)
            except Exception as e:
                import torch

                _oom_type = getattr(torch.cuda, "OutOfMemoryError", None)
                is_oom = "out of memory" in str(e).lower() or (isinstance(_oom_type, type) and isinstance(e, _oom_type))
                if is_oom and device == "cuda":
                    self.logger.warning("CUDA OOM during tracker init. Switching to CPU.")
                    self.runtime_device_override = "cpu"
                    return self._load_tracker_impl(model_name, _models_path, _user_agent, _retry_params, "cpu", _config)
                raise e

        return self.get_or_load(key, _loader)

    def get_subject_detector(
        self, model_name: str, model_path: str, logger: "LoggerLike", device: str
    ) -> Optional[Any]:
        """Retrieves or loads a subject detector (YOLO family) with CPU fallback on OOM."""
        key = f"detector_{model_name}"

        def _loader():
            # Lazy import to avoid top-level onnxruntime dependency
            from .subject_detector import SubjectDetector

            current_device = self.runtime_device_override or device
            try:
                return SubjectDetector(model_path, logger, device=current_device)
            except Exception as e:
                import torch

                _oom_type = getattr(torch.cuda, "OutOfMemoryError", None)
                is_oom = "out of memory" in str(e).lower() or (isinstance(_oom_type, type) and isinstance(e, _oom_type))
                if is_oom and current_device == "cuda":
                    self.logger.warning(f"CUDA OOM during detector '{model_name}' init. Switching to CPU.")
                    self.runtime_device_override = "cpu"
                    return SubjectDetector(model_path, logger, device="cpu")
                raise e

        try:
            return self.get_or_load(key, _loader)
        except Exception as e:
            self.logger.error(f"Failed to initialize subject detector {model_name}: {e}", exc_info=True)
            return None

    def _load_tracker_impl(
        self,
        model_name: str,
        models_path: str,
        user_agent: str,
        retry_params: tuple,
        device: str,
        config: Optional["Config"] = None,
    ):
        from core.error_handling import ErrorHandler
        from core.io_utils import download_model

        from .tracker_factory import TrackerBackend, build_tracker

        if model_name == "sam2":
            checkpoint_filename = "sam2.1_hiera_tiny.pt"
            url = config.sam2_checkpoint_url if config else ""
            description = "SAM2.1 Model"
            backend: TrackerBackend = "sam2"
        elif model_name == "sam3":
            checkpoint_filename = "sam3.1_multiplex.pt"
            url = config.sam3_checkpoint_url if config else ""
            description = "SAM3.1 Model"
            backend: TrackerBackend = "sam3"
        else:
            raise ValueError(f"Unknown tracker model '{model_name}'. Must be 'sam2' or 'sam3'.")

        checkpoint_path = Path(models_path) / checkpoint_filename
        if model_name == "sam3":
            # Fallback logic: check for legacy sam3.pt before deciding to download
            if not checkpoint_path.exists():
                legacy_path = Path(models_path) / "sam3.pt"
                if legacy_path.exists():
                    self.logger.info(f"SAM 3.1 checkpoint not found; falling back to legacy {legacy_path.name}")
                    checkpoint_path = legacy_path

        if not checkpoint_path.exists():
            if url and ".safetensors" in url:
                url = url.replace(".safetensors", ".pt")

            if not url:
                raise ValueError(
                    f"No checkpoint URL provided for {model_name} and local file {checkpoint_path} not found."
                )

            download_model(
                url=url,
                dest_path=checkpoint_path,
                description=description,
                logger=self.logger,
                error_handler=ErrorHandler(self.logger, *retry_params),
                user_agent=user_agent,
                token=config.huggingface_token if config else None,
            )
        return build_tracker(backend, str(checkpoint_path), device=device, config=config)
