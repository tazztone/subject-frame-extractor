import gc
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from core.config import Config
    from core.logger import AppLogger


class ThumbnailManager:
    """Manages an in-memory LRU cache for image thumbnails."""

    def __init__(self, logger: "AppLogger", config: "Config"):
        """Initializes the manager with a configurable cache size."""
        self.logger = logger
        self.config = config
        self.cache = OrderedDict()
        self.max_size = self.config.cache_size
        self.logger.info(f"ThumbnailManager initialized with cache size {self.max_size}")

    def get(self, thumb_path: Path) -> Optional[np.ndarray]:
        """Retrieves a thumbnail from cache or loads it from disk."""
        if not isinstance(thumb_path, Path):
            thumb_path = Path(thumb_path)
        if thumb_path in self.cache:
            self.cache.move_to_end(thumb_path)
            return self.cache[thumb_path]
        if not thumb_path.exists():
            return None
        if len(self.cache) > self.max_size * self.config.cache_cleanup_threshold:
            self._cleanup_old_entries()
        try:
            with Image.open(thumb_path) as pil_thumb:
                thumb_img = np.array(pil_thumb.convert("RGB"))
            self.cache[thumb_path] = thumb_img
            while len(self.cache) > self.max_size:
                self.cache.popitem(last=False)
            return thumb_img
        except Exception as e:
            self.logger.warning("Failed to load thumbnail with Pillow", extra={"path": str(thumb_path), "error": e})
            return None

    def clear_cache(self):
        """Standard cache clearing and GC triggering."""
        self.cache.clear()
        gc.collect()

    def _cleanup_old_entries(self):
        num_to_remove = int(self.max_size * self.config.cache_eviction_factor)
        for _ in range(num_to_remove):
            if not self.cache:
                break
            self.cache.popitem(last=False)
