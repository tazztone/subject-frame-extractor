"""Composition root for dependency injection and application setup."""

import threading
from queue import Queue

from app.config import Config
from app.logging_enhanced import EnhancedLogger
from app.thumb_cache import ThumbnailManager
from app.performance_optimizer import AdaptiveResourceManager
from app.progress_enhanced import AdvancedProgressTracker


class CompositionRoot:
    """Composition root that wires up all dependencies."""
    
    def __init__(self):
        """Initialize the composition root with all dependencies."""
        # Core dependencies
        self.config = Config()
        self.logger = EnhancedLogger(
            log_dir=self.config.DIRS['logs'],
            enable_performance_monitoring=True
        )
        self.thumbnail_manager = ThumbnailManager(
            logger=self.logger,
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

        # Create the tracker here as the single source of truth
        self.progress_tracker = AdvancedProgressTracker(self.progress_queue, self.logger)
        
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
                cancel_event=self.cancel_event,
                # Inject all shared dependencies into the UI
                thumbnail_manager=self.get_thumbnail_manager(),
                resource_manager=self.get_resource_manager(),
                progress_tracker=self.progress_tracker
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