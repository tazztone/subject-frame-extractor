"""Base pipeline class for video processing operations."""

import threading
from queue import Queue

from app.logging import UnifiedLogger


class Pipeline:
    """Base class for all processing pipelines."""
    
    def __init__(self, params, progress_queue: Queue, cancel_event: threading.Event):
        
        self.params = params
        self.progress_queue = progress_queue
        self.cancel_event = cancel_event
        self.logger = UnifiedLogger()