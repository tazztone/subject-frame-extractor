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
    """Manages an in-memory LRU cache for image thumbnails with size limits."""

    def __init__(self, logger: "AppLogger", config: "Config"):
        """Initializes the manager with a configurable cache size and memory limit."""
        self.logger = logger
        self.config = config
        self.cache = OrderedDict()
        self.max_count = self.config.cache_size
        self.max_bytes = self.config.thumbnail_cache_max_mb * 1024 * 1024
        self.current_bytes = 0
        self.logger.info(
            f"ThumbnailManager initialized: max_count={self.max_count}, max_memory={self.max_bytes / (1024 * 1024):.1f}MB"
        )

    def get(self, thumb_path: Path) -> Optional[np.ndarray]:
        """Retrieves a thumbnail from cache or loads it from disk."""
        if not isinstance(thumb_path, Path):
            thumb_path = Path(thumb_path)
        if thumb_path in self.cache:
            self.cache.move_to_end(thumb_path)
            return self.cache[thumb_path]
        if not thumb_path.exists():
            return None

        # Proactive cleanup if too many items
        if len(self.cache) > self.max_count * self.config.cache_cleanup_threshold:
            self._cleanup_old_entries()

        try:
            with Image.open(thumb_path) as pil_thumb:
                thumb_img = np.array(pil_thumb.convert("RGB"))

            size_bytes = thumb_img.nbytes

            # Ensure we have space for the new item
            while (
                self.current_bytes + size_bytes > self.max_bytes or len(self.cache) >= self.max_count
            ) and self.cache:
                _, old_img = self.cache.popitem(last=False)
                self.current_bytes -= old_img.nbytes

            self.cache[thumb_path] = thumb_img
            self.current_bytes += size_bytes

            return thumb_img
        except Exception as e:
            self.logger.warning("Failed to load thumbnail with Pillow", extra={"path": str(thumb_path), "error": e})
            return None

    def clear_cache(self):
        """Standard cache clearing and GC triggering."""
        self.cache.clear()
        self.current_bytes = 0
        gc.collect()

    def _cleanup_old_entries(self):
        num_to_remove = int(self.max_count * self.config.cache_eviction_factor)
        for _ in range(num_to_remove):
            if not self.cache:
                break
            _, old_img = self.cache.popitem(last=False)
            self.current_bytes -= old_img.nbytes
