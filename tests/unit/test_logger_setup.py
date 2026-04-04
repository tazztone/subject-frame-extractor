import logging
from queue import Queue

import pytest

from core.config import Config
from core.logger import GradioQueueHandler, setup_logging


def test_gradio_queue_handler_formatter_default():
    """Verify GradioQueueHandler hardcodes its simple formatter."""
    queue = Queue()
    handler = GradioQueueHandler(queue)

    # Create a dummy record
    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname="test.py", lineno=1, msg="Test Message", args=(), exc_info=None
    )

    formatted = handler.format(record)
    assert formatted == "[INFO] Test Message"


def test_root_logger_isolation():
    """Verify setup_logging does NOT attach GradioQueueHandler to the root logger."""
    config = Config()
    progress_queue = Queue()

    # Get root logger
    root_logger = logging.getLogger("")

    setup_logging(config, progress_queue=progress_queue)

    # Check root logger handlers after setup
    current_handlers = root_logger.handlers

    # Verify no GradioQueueHandler was added to root
    for handler in current_handlers:
        if isinstance(handler, GradioQueueHandler):
            pytest.fail("GradioQueueHandler was incorrectly added to the root logger.")
