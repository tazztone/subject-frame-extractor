"""
Logging Infrastructure for Frame Extractor & Analyzer
"""

import json
import logging
import logging.config
import os
import traceback
from datetime import datetime
from pathlib import Path
from queue import Queue
from typing import TYPE_CHECKING, Any, Dict, Optional

from pydantic import BaseModel

if TYPE_CHECKING:
    from core.config import Config

# --- CONSTANTS ---

SUCCESS_LEVEL_NUM = 25
logging.addLevelName(SUCCESS_LEVEL_NUM, "SUCCESS")


# --- MODELS ---


class LogEvent(BaseModel):
    """Represents a structured log entry."""

    timestamp: str
    level: str
    message: str
    component: str
    operation: Optional[str] = None
    duration_ms: Optional[float] = None
    error_type: Optional[str] = None
    stack_trace: Optional[str] = None
    user_context: Optional[Dict[str, Any]] = None
    performance_metrics: Optional[Dict[str, Any]] = None
    custom_fields: Optional[Dict[str, Any]] = None


# --- HANDLERS ---


class GradioQueueHandler(logging.Handler):
    """Logging handler that redirects logs to a Gradio progress queue."""

    def __init__(self, queue: Queue):
        super().__init__()
        self.queue = queue

    def emit(self, record: logging.LogRecord):
        try:
            msg = self.format(record)
            self.queue.put({"log": msg, "unified_log": msg})
        except Exception:
            self.handleError(record)


# --- FORMATTERS ---


class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to log levels."""

    COLORS = {
        "DEBUG": "\033[36m",
        "INFO": "\033[37m",
        "WARNING": "\033[33m",
        "ERROR": "\033[31m",
        "CRITICAL": "\033[35m",
        "SUCCESS": "\033[32m",
        "RESET": "\033[0m",
    }

    def format(self, record: logging.LogRecord) -> str:
        """Formats the log record with color codes."""
        original_levelname = record.levelname
        try:
            color = self.COLORS.get(original_levelname, self.COLORS["RESET"])
            record.levelname = f"{color}{original_levelname}{self.COLORS['RESET']}"
            return super().format(record)
        finally:
            record.levelname = original_levelname


# --- LOGGER ---


def setup_logging(
    config: "Config", 
    log_dir: Optional[Path] = None, 
    log_to_console: bool = True,
    progress_queue: Optional[Queue] = None
):
    """
    Sets up the global logging configuration using dictConfig.
    """
    # 1. Framework Silencing
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["ABSL_LOGGING_LEVEL"] = "error"
    
    # 2. Path Setup
    log_dir = log_dir or Path(config.logs_dir)
    log_dir.mkdir(exist_ok=True, parents=True)
    
    # Use a consistent 'run.log' name if log_dir is a session folder, 
    # otherwise use a timestamped name.
    if "e2e_output" in str(log_dir) or "test_logging_output" in str(log_dir):
        session_log_file = log_dir / "run.log"
    else:
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_log_file = log_dir / f"session_{session_id}.log"

    # 3. dictConfig Definition
    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": config.log_format,
            },
            "simple": {
                "format": "[%(levelname)s] %(message)s",
            },
            "colored": {
                "()": ColoredFormatter,
                "format": config.log_format,
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "colored" if config.log_colored else "standard",
                "level": config.log_level,
            },
            "file": {
                "class": "logging.FileHandler",
                "filename": str(session_log_file),
                "encoding": "utf-8",
                "formatter": "standard",
                "level": "DEBUG",
            },
        },
        "loggers": {
            "": {  # Root logger
                "handlers": ["console", "file"],
                "level": "DEBUG",
                "propagate": True,
            },
            "app_logger": {
                "handlers": ["console", "file"],
                "level": "DEBUG",
                "propagate": False,
            },
            # Silence and Standardize noisy libraries
            "matplotlib": {"level": "WARNING"},
            "PIL": {"level": "WARNING"},
            "absl": {"level": "ERROR"},
            "tensorflow": {"level": "ERROR"},
            "insightface": {"level": "WARNING"},
            "pyscenedetect": {"level": "WARNING"},
            # SAM3 specific fix: Force it to use our root handlers
            "sam3": {
                "level": "INFO",
                "propagate": True,
            },
        },
    }

    # Remove console if not requested (e.g. background tasks)
    if not log_to_console:
        logging_config["handlers"]["console"]["level"] = "CRITICAL"
        
    # Add Gradio Queue Handler if provided
    if progress_queue:
        logging_config["handlers"]["gradio"] = {
            "()": GradioQueueHandler,
            "queue": progress_queue,
            "formatter": "simple",
            "level": "INFO",
        }
        logging_config["loggers"]["app_logger"]["handlers"].append("gradio")
        logging_config["loggers"][""]["handlers"].append("gradio")

    logging.config.dictConfig(logging_config)
    
    # 4. Post-Config Cleanup for recalcitrant loggers (like SAM3)
    sam3_logger = logging.getLogger("sam3")
    sam3_logger.handlers.clear()
    sam3_logger.propagate = True
    
    return session_log_file


class AppLogger:
    """
    A streamlined interface for the application's logging.
    Now acts as a proxy to standard logging calls.
    """

    def __init__(self, config: "Config", **kwargs):
        """
        Initializes the AppLogger. setup_logging() MUST be called once before this.
        """
        self.config = config
        self.logger = logging.getLogger("app_logger")

    def _log(self, level: str, message: str, component: str, **kwargs):
        """Helper to create a structured log and pass to standard logger."""
        # Extract exc_info if it exists to avoid overwriting it in 'extra'
        exc_info = kwargs.pop("exc_info", None)
        
        extra = {"component": component, **kwargs}
        log_level = getattr(logging, level.upper(), logging.INFO)
        if level.upper() == "SUCCESS":
            log_level = SUCCESS_LEVEL_NUM
            
        self.logger.log(log_level, f"{message} [{component}]", extra=extra, exc_info=exc_info)

    def debug(self, message: str, component: str = "system", **kwargs):
        self._log("DEBUG", message, component, **kwargs)

    def info(self, message: str, component: str = "system", **kwargs):
        self._log("INFO", message, component, **kwargs)

    def warning(self, message: str, component: str = "system", **kwargs):
        self._log("WARNING", message, component, **kwargs)

    def error(self, message: str, component: str = "system", **kwargs):
        self._log("ERROR", message, component, **kwargs)

    def success(self, message: str, component: str = "system", **kwargs):
        self._log("SUCCESS", message, component, **kwargs)

    def critical(self, message: str, component: str = "system", **kwargs):
        self._log("CRITICAL", message, component, **kwargs)

    def set_progress_queue(self, queue: Queue):
        """No longer used. UI should call setup_logging with progress_queue."""
        pass

    def copy_log_to_output(self, output_dir: Path):
        """No longer needed."""
        pass

