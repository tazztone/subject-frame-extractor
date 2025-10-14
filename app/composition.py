"""Composition root for dependency injection and application setup."""

import threading
from queue import Queue

from app.config import Config
from app.logging import UnifiedLogger
from app.thumb_cache import ThumbnailManager
from app.performance_optimizer import AdaptiveResourceManager


class CompositionRoot:
    """Composition root that wires up all dependencies."""
    
    def __init__(self):
        """Initialize the composition root with all dependencies."""
        # Core dependencies
        self.config = Config()
        self.logger = self.config.setup_directories_and_logger()
        self.thumbnail_manager = ThumbnailManager(
            max_size=self.config.thumbnail_cache_size
        )
        self.resource_manager = AdaptiveResourceManager(
            logger=self.logger,
            config=self.config
        )
        self.resource_manager.start_monitoring()
        
        # Shared resources
        self.progress_queue = Queue()
        self.cancel_event = threading.Event()
        
        # Set up logger with progress queue
        self.logger.set_progress_queue(self.progress_queue)
        
        # UI will be lazy-loaded to avoid import issues
        self._app_ui = None
    
    def get_app_ui(self):
        """Get the configured AppUI instance (lazy-loaded)."""
        if self._app_ui is None:
            from app.app_ui import EnhancedAppUI
            self._app_ui = EnhancedAppUI(
                config=self.config,
                logger=self.logger,
                progress_queue=self.progress_queue,
                cancel_event=self.cancel_event
            )
        return self._app_ui
    
    def get_config(self):
        """Get the configuration instance."""
        return self.config
    
    def get_logger(self):
        """Get the logger instance."""
        return self.logger
    
    def get_thumbnail_manager(self):
        """Get the thumbnail manager instance."""
        return self.thumbnail_manager
    
    def get_resource_manager(self):
        """Get the resource manager instance."""
        return self.resource_manager

    def cleanup(self):
        """Clean up resources."""
        if hasattr(self.thumbnail_manager, 'cleanup'):
            self.thumbnail_manager.cleanup()
        self.resource_manager.stop_monitoring()
        self.cancel_event.set()