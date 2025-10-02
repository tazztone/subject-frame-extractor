"""Thumbnail caching system with LRU policy."""

import numpy as np
from collections import OrderedDict
from pathlib import Path
from PIL import Image


class ThumbnailManager:
    """Manages loading and caching of thumbnails with an LRU policy."""
    def __init__(self, max_size=200):
        # Import logger locally to avoid circular imports
        from app.core.logging import UnifiedLogger
        logger = UnifiedLogger()

        self.cache = OrderedDict()
        self.max_size = max_size
        init_msg = f"ThumbnailManager initialized with cache size {max_size}"
        logger.info(init_msg)

    def get(self, thumb_path: Path):
        """Get thumbnail from cache or load from disk as RGB numpy array."""
        # Import logger locally to avoid circular imports
        from app.core.logging import UnifiedLogger
        logger = UnifiedLogger()

        if not isinstance(thumb_path, Path):
            thumb_path = Path(thumb_path)

        if thumb_path in self.cache:
            self.cache.move_to_end(thumb_path)
            return self.cache[thumb_path]

        if not thumb_path.exists():
            return None

        try:
            with Image.open(thumb_path) as pil_thumb:
                thumb_rgb_pil = pil_thumb.convert("RGB")
                thumb_img = np.array(thumb_rgb_pil)

            self.cache[thumb_path] = thumb_img
            if len(self.cache) > self.max_size:
                self.cache.popitem(last=False)

            return thumb_img
        except Exception as e:
            error_extra = {'path': str(thumb_path), 'error': e}
            logger.warning("Failed to load thumbnail", extra=error_extra)
            return None
