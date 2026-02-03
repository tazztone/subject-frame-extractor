"""
Logging Infrastructure for Frame Extractor & Analyzer
"""

import json
import logging
import traceback
import gzip
import shutil
import os
from logging.handlers import RotatingFileHandler
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


class JsonFormatter(logging.Formatter):
    """Formatter that outputs logs as JSON strings."""

    def format(self, record: logging.LogRecord) -> str:
        """Formats the log record as a JSON string."""
        log_event_obj = getattr(record, "log_event", None)
        if isinstance(log_event_obj, LogEvent):
            log_dict = log_event_obj.model_dump(exclude_none=True)
        else:
            log_dict = {
                "timestamp": self.formatTime(record, self.datefmt),
                "level": record.levelname,
                "message": record.getMessage(),
                "component": record.name,
            }
            if record.exc_info:
                log_dict["stack_trace"] = self.formatException(record.exc_info)
        return json.dumps(log_dict, default=str, ensure_ascii=False)


# --- LOGGER ---


# TODO: Keep logging simple for local use
class AppLogger:
    """A streamlined logger for the application, consolidating output into a single run log."""

    def __init__(
        self, config: "Config", log_dir: Optional[Path] = None, log_to_file: bool = True, log_to_console: bool = True
    ):
        """
        Initializes the AppLogger.
        """
        self.config = config
        self.log_dir = log_dir or Path(self.config.logs_dir)
        self.log_dir.mkdir(exist_ok=True, parents=True)
        self.progress_queue = None
        
        # Use a consistent 'run.log' name if log_dir is a session folder, 
        # otherwise use a timestamped name.
        if "e2e_output" in str(self.log_dir):
            self.session_log_file = self.log_dir / "run.log"
        else:
            self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.session_log_file = self.log_dir / f"session_{self.session_id}.log"

        self.logger = logging.getLogger(f"app_logger_{id(self)}")
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False
        self.logger.handlers.clear()

        if log_to_console:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(ColoredFormatter(self.config.log_format))
            console_handler.setLevel(self.config.log_level)
            self.logger.addHandler(console_handler)

        if log_to_file:
            file_handler = logging.FileHandler(self.session_log_file, encoding="utf-8")
            file_handler.setFormatter(logging.Formatter(self.config.log_format))
            file_handler.setLevel(logging.DEBUG)
            self.logger.addHandler(file_handler)

    def set_progress_queue(self, queue: Queue):
        """Sets the queue used for sending logs to the UI."""
        self.progress_queue = queue

    def _create_log_event(self, level: str, message: str, component: str, **kwargs) -> LogEvent:
        """Helper to create a structured LogEvent object."""
        exc_info = kwargs.pop("exc_info", None)
        extra = kwargs.pop("extra", None)
        if exc_info:
            kwargs["stack_trace"] = traceback.format_exc()
        if extra:
            kwargs["custom_fields"] = extra
        return LogEvent(
            timestamp=datetime.now().isoformat(), level=level, message=message, component=component, **kwargs
        )

    def _log_event(self, event: LogEvent):
        """Dispatches the LogEvent to standard logging and the UI queue."""
        log_level_num = getattr(logging, event.level.upper(), logging.INFO)
        if event.level.upper() == "SUCCESS":
            log_level_num = SUCCESS_LEVEL_NUM

        extra_info = f" [{event.component}]"
        log_message = f"{event.message}{extra_info}"
        if event.stack_trace:
            log_message += f"\n{event.stack_trace}"

        self.logger.log(log_level_num, log_message)

        if self.progress_queue:
            ui_message = f"[{event.level}] {event.message}"
            self.progress_queue.put({"log": ui_message})

    def debug(self, message: str, component: str = "system", **kwargs):
        self._log_event(self._create_log_event("DEBUG", message, component, **kwargs))

    def info(self, message: str, component: str = "system", **kwargs):
        self._log_event(self._create_log_event("INFO", message, component, **kwargs))

    def warning(self, message: str, component: str = "system", **kwargs):
        self._log_event(self._create_log_event("WARNING", message, component, **kwargs))

    def error(self, message: str, component: str = "system", **kwargs):
        self._log_event(self._create_log_event("ERROR", message, component, **kwargs))

    def success(self, message: str, component: str = "system", **kwargs):
        self._log_event(self._create_log_event("SUCCESS", message, component, **kwargs))

    def critical(self, message: str, component: str = "system", **kwargs):
        self._log_event(self._create_log_event("CRITICAL", message, component, **kwargs))

    def copy_log_to_output(self, output_dir: Path):
        """No longer needed with consolidated logging."""
        pass

