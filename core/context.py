from __future__ import annotations

import threading
from queue import Queue
from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:
    from core.config import Config
    from core.logger import AppLogger
    from core.managers import ModelRegistry, ThumbnailManager


class AnalysisContext:
    """Bundles pipeline execution parameters to deepen interfaces and manage model lifecycles."""

    def __init__(
        self,
        config: Config,
        logger: AppLogger,
        progress_queue: Queue,
        cancel_event: threading.Event,
        thumbnail_manager: ThumbnailManager,
        model_registry: ModelRegistry,
        cuda_available: bool,
        progress: Optional[Callable] = None,
    ):
        self.config = config
        self.logger = logger
        self.progress_queue = progress_queue
        self.cancel_event = cancel_event
        self.thumbnail_manager = thumbnail_manager
        self.model_registry = model_registry
        self.cuda_available = cuda_available
        self.progress = progress
        self.loaded_models: dict = {}
