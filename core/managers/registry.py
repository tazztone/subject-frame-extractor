from __future__ import annotations

import gc
import logging
import threading
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Set

import torch

if TYPE_CHECKING:
    from core.config import Config
    from core.logger import LoggerLike


class ModelRegistry:
    """Thread-safe registry for lazy loading and caching of heavy ML models."""

    def __init__(self, logger: Optional["LoggerLike"] = None, config: Optional["Config"] = None):
        self._models: Dict[str, Any] = {}
        self._failed_models: Set[str] = set()
        self._locks: Dict[str, threading.Lock] = defaultdict(threading.Lock)
        self._registry_lock = threading.RLock()
        self.logger: "LoggerLike" = logger or logging.getLogger(__name__)
        self.config: Optional["Config"] = config
        self.runtime_device_override: Optional[str] = None
        self._thread_local = threading.local()
        self._landmarkers: list[Any] = []  # Tracks per-thread FaceLandmarker instances

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
            landmarkers_to_clear = list(self._landmarkers)
            self._landmarkers.clear()
            self._thread_local = threading.local()

        for key, model in models_to_clear:
            try:
                if hasattr(model, "shutdown") and callable(model.shutdown):
                    model.shutdown()
                elif hasattr(model, "close") and callable(model.close):
                    model.close()
            except Exception as e:
                self.logger.warning(f"Error shutting down model {key}: {e}")

        for landmarker in landmarkers_to_clear:
            try:
                if hasattr(landmarker, "close") and callable(landmarker.close):
                    landmarker.close()
            except Exception as e:
                self.logger.warning(f"Error closing face landmarker: {e}")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_face_analyzer(
        self,
        model_name: str,
        det_size_tuple: tuple,
        device: str = "cpu",
    ) -> Any:
        """Gets or loads the InsightFace FaceAnalysis app, with OOM handling."""
        from insightface.app import FaceAnalysis

        _config = self.config
        if _config is None and hasattr(self.logger, "config"):
            _config = self.logger.config  # type: ignore
        models_path = str(_config.models_dir) if _config else "/tmp"

        model_key = f"face_analyzer_{model_name}_{device}_{det_size_tuple}"

        def _loader():
            self.logger.info(f"Loading face model: {model_name} on device: {device}")
            try:
                is_cuda = device == "cuda"
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if is_cuda else ["CPUExecutionProvider"]
                analyzer = FaceAnalysis(name=model_name, root=models_path, providers=providers)
                analyzer.prepare(ctx_id=0 if is_cuda else -1, det_size=det_size_tuple)
                if hasattr(self.logger, "success"):
                    self.logger.success(f"Face model loaded with {'CUDA' if is_cuda else 'CPU'}.")  # type: ignore
                else:
                    self.logger.info(f"Face model loaded with {'CUDA' if is_cuda else 'CPU'}.")
                return analyzer
            except Exception as e:
                if "out of memory" in str(e) and device == "cuda":
                    self.logger.warning("CUDA OOM, retrying with CPU...")
                    try:
                        analyzer = FaceAnalysis(name=model_name, root=models_path, providers=["CPUExecutionProvider"])
                        analyzer.prepare(ctx_id=-1, det_size=det_size_tuple)
                        return analyzer
                    except Exception as cpu_e:
                        self.logger.error(f"CPU fallback also failed: {cpu_e}")
                        raise RuntimeError(
                            f"Could not initialize face analysis model. CPU fallback also failed: {cpu_e}"
                        ) from cpu_e
                raise RuntimeError(f"Could not initialize face analysis model. Error: {e}") from e

        return self.get_or_load(model_key, _loader)

    def get_face_landmarker(self, model_path: Optional[str] = None, logger: Optional["LoggerLike"] = None) -> Any:
        """Returns a thread-local MediaPipe FaceLandmarker, tracked for cleanup and downloaded if needed."""
        _logger = logger or self.logger
        _config = self.config
        if _config is None and hasattr(_logger, "config"):
            _config = _logger.config  # type: ignore

        if not model_path:
            if not _config:
                raise ValueError("No model_path or config provided for FaceLandmarker.")
            model_url = _config.face_landmarker_url
            model_path = str(Path(_config.models_dir) / Path(model_url).name)

            if not Path(model_path).exists():
                from core.error_handling import ErrorHandler
                from core.io_utils import download_model

                download_model(
                    model_url,
                    Path(model_path),
                    "MediaPipe Face Landmarker",
                    _logger,
                    ErrorHandler(_logger, _config.retry_max_attempts, _config.retry_backoff_seconds),
                    _config.user_agent,
                    expected_sha256=_config.face_landmarker_sha256,
                )

        if hasattr(self._thread_local, "face_landmarker"):
            return self._thread_local.face_landmarker

        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        _logger.info("Initializing MediaPipe FaceLandmarker for thread.", component="face_landmarker")
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1,
            min_face_detection_confidence=0.3,
            min_face_presence_confidence=0.3,
        )
        detector = vision.FaceLandmarker.create_from_options(options)
        self._thread_local.face_landmarker = detector
        with self._registry_lock:
            self._landmarkers.append(detector)
        return detector

    def get_tracker(
        self,
        model_name: str,
        models_path: Optional[str] = None,
        user_agent: Optional[str] = None,
        retry_params: Optional[tuple] = None,
        config: Optional["Config"] = None,
    ) -> Optional[Any]:
        """Loads subject tracker with CPU fallback on OOM."""
        _config = config or self.config
        if _config is None and hasattr(self.logger, "config"):
            _config = self.logger.config  # type: ignore

        _models_path = models_path or (str(_config.models_dir) if _config else None)
        _user_agent = user_agent or (_config.user_agent if _config else "SubjectFrameExtractor")
        _retry_params = retry_params or (1, [1])

        if not _models_path:
            self.logger.error("get_tracker: models_path is None and no config provided — cannot locate checkpoint.")
            return None

        key = f"tracker_{model_name}"

        def _loader():
            device = self.runtime_device_override or ("cuda" if torch.cuda.is_available() else "cpu")
            try:
                return self._load_tracker_impl(model_name, _models_path, _user_agent, _retry_params, device, _config)
            except RuntimeError as e:
                if "out of memory" in str(e) and device == "cuda":
                    self.logger.warning("CUDA OOM during tracker init. Switching to CPU.")
                    self.runtime_device_override = "cpu"
                    return self._load_tracker_impl(model_name, _models_path, _user_agent, _retry_params, "cpu", _config)
                raise e

        try:
            return self.get_or_load(key, _loader)
        except Exception as e:
            self.logger.error(f"Failed to initialize tracker: {e}", exc_info=True)
            return None

    def get_subject_detector(
        self,
        model_name: str,
        model_path: Optional[str] = None,
        logger: Optional["LoggerLike"] = None,
        device: Optional[str] = None,
    ) -> Optional[Any]:
        """Retrieves or loads a subject detector (YOLO family) with CPU fallback on OOM, downloading if needed."""
        _logger = logger or self.logger
        _config = self.config
        if _config is None and hasattr(_logger, "config"):
            _config = _logger.config  # type: ignore
        _device = device or self.runtime_device_override or ("cuda" if torch.cuda.is_available() else "cpu")

        if not model_path:
            if not _config:
                raise ValueError("No model_path or config provided for subject detector.")

            url_map = {
                "YOLO12l-Seg": _config.yolo12l_seg_url,
                "YOLO26n": _config.yolo26n_url,
                "YOLO26s": _config.yolo26s_url,
                "YOLO26m": _config.yolo26m_url,
                "YOLO26l": _config.yolo26l_url,
                "YOLO26x": _config.yolo26x_url,
            }
            model_url = url_map.get(model_name)
            if not model_url:
                _logger.error(f"No URL configured for detector model: {model_name}")
                return None

            model_path = str(Path(_config.models_dir) / Path(model_url).name)

            if not Path(model_path).exists():
                from core.error_handling import ErrorHandler
                from core.io_utils import download_model

                download_model(
                    model_url,
                    Path(model_path),
                    f"Person Detector ({model_name})",
                    _logger,
                    ErrorHandler(_logger, _config.retry_max_attempts, _config.retry_backoff_seconds),
                    _config.user_agent,
                )

        key = f"detector_{model_name}"

        def _loader():
            # Lazy import to avoid top-level onnxruntime dependency
            from .subject_detector import SubjectDetector

            current_device = self.runtime_device_override or _device
            try:
                return SubjectDetector(model_path, _logger, device=current_device)
            except RuntimeError as e:
                if "out of memory" in str(e).lower() and current_device == "cuda":
                    _logger.warning(f"CUDA OOM during detector '{model_name}' init. Switching to CPU.")
                    self.runtime_device_override = "cpu"
                    return SubjectDetector(model_path, _logger, device="cpu")
                raise e

        try:
            return self.get_or_load(key, _loader)
        except Exception as e:
            _logger.error(f"Failed to initialize subject detector {model_name}: {e}", exc_info=True)
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
            raise ValueError("SAM2.1 model has been retired. Please use 'sam3' (SAM3.1 Multiplex) instead.")
        elif model_name == "sam3":
            checkpoint_filename = "sam3.1_multiplex_fp16.safetensors"
            url = config.sam3_checkpoint_url if config else ""
            description = "SAM3.1 Multiplex Model"
            backend: TrackerBackend = "sam3"
        else:
            raise ValueError(f"Unknown tracker model '{model_name}'. Must be 'sam3'.")

        checkpoint_path = Path(models_path) / checkpoint_filename

        if not checkpoint_path.exists():
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
