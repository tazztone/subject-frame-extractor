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
    """A comprehensive logger for the application."""

    def __init__(
        self, config: "Config", log_dir: Optional[Path] = None, log_to_file: bool = True, log_to_console: bool = True
    ):
        """
        Initializes the AppLogger.

        Args:
            config: Application configuration.
            log_dir: Directory to store log files.
            log_to_file: Whether to write logs to files.
            log_to_console: Whether to print logs to the console.
        """
        self.config = config
        self.log_dir = log_dir or Path(self.config.logs_dir)
        self.log_dir.mkdir(exist_ok=True, parents=True)
        self.progress_queue = None
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_log_file = self.log_dir / f"session_{self.session_id}.log"
        self.structured_log_file = self.log_dir / self.config.log_structured_path
        self.logger = logging.getLogger(f"enhanced_logger_{self.session_id}")
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False
        self.logger.handlers.clear()

        # Clean up old logs (compress them)
        if log_to_file:
            self._cleanup_old_logs()

        if log_to_console and self.config.log_colored:
            self._setup_console_handler()
        if log_to_file:
            self._setup_file_handlers()

    def _setup_console_handler(self):
        """Configures the console logging handler."""
        console_handler = logging.StreamHandler()
        console_formatter = ColoredFormatter(self.config.log_format)
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(self.config.log_level)
        self.logger.addHandler(console_handler)

    def _setup_file_handlers(self):
        """Configures file logging handlers (plain text and JSONL)."""

        def compress_rotator(source, dest):
            """Helper to compress rotated log files."""
            with open(source, "rb") as f_in:
                with gzip.open(dest, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
            os.remove(source)

        def compress_namer(name):
            """Helper to name compressed log files."""
            return name + ".gz"

        # Session Log Handler (Rotating + Compression)
        file_handler = RotatingFileHandler(
            self.session_log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.rotator = compress_rotator
        file_handler.namer = compress_namer

        file_formatter = logging.Formatter(self.config.log_format)
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.DEBUG)
        self.logger.addHandler(file_handler)

        # Structured Log Handler (Rotating + Compression)
        structured_handler = RotatingFileHandler(
            self.structured_log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding="utf-8",
        )
        structured_handler.rotator = compress_rotator
        structured_handler.namer = compress_namer

        structured_formatter = JsonFormatter()
        structured_handler.setFormatter(structured_formatter)
        structured_handler.setLevel(logging.DEBUG)
        self.logger.addHandler(structured_handler)

    def _cleanup_old_logs(self):
        """Compresses old log files to save disk space."""
        try:
            # Compress old session logs
            for log_file in self.log_dir.glob("session_*.log"):
                if log_file.resolve() != self.session_log_file.resolve():
                    self._compress_file(log_file)

            # Compress rotated structured logs (if they were left uncompressed by previous versions)
            for log_file in self.log_dir.glob(f"{self.config.log_structured_path}.*"):
                # RotatingFileHandler names backups as .1, .2 etc.
                # Check if it ends in digit (meaning it's a rotated part) and not .gz
                if log_file.suffix[1:].isdigit():
                    self._compress_file(log_file)

        except Exception as e:
            # We don't want to crash application startup if cleanup fails
            print(f"Warning: Failed to cleanup old logs: {e}")

    def _compress_file(self, file_path: Path):
        """Compresses a single file and removes the original."""
        try:
            gz_path = file_path.with_suffix(file_path.suffix + ".gz")
            if gz_path.exists():
                return  # Already compressed

            with open(file_path, "rb") as f_in:
                with gzip.open(gz_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
            file_path.unlink()
        except Exception as e:
            print(f"Warning: Failed to compress {file_path}: {e}")

    def set_progress_queue(self, queue: Queue):
        """Sets the queue used for sending logs to the UI."""
        self.progress_queue = queue

    def _create_log_event(self, level: str, message: str, component: str, **kwargs) -> LogEvent:
        """Helper to create a structured LogEvent object."""
        exc_info = kwargs.pop("exc_info", None)
        extra = kwargs.pop("extra", None)
        if "stacktrace" in kwargs:
            kwargs["stack_trace"] = kwargs.pop("stacktrace")
        if exc_info:
            kwargs["stack_trace"] = traceback.format_exc()
        if extra:
            kwargs["custom_fields"] = kwargs.get("custom_fields", {})
            kwargs["custom_fields"].update(extra)
        return LogEvent(
            timestamp=datetime.now().isoformat(), level=level, message=message, component=component, **kwargs
        )

    def _log_event(self, event: LogEvent):
        """Dispatches the LogEvent to standard logging and the UI queue."""
        log_level_name = event.level.upper()
        log_level = getattr(logging, log_level_name, logging.INFO)
        if log_level_name == "SUCCESS":
            log_level = SUCCESS_LEVEL_NUM

        extra_info = f" [{event.component}]"
        if event.operation:
            extra_info += f" [{event.operation}]"
        if event.duration_ms:
            extra_info += f" ({event.duration_ms:.1f}ms)"

        log_message = f"{event.message}{extra_info}"
        if event.stack_trace:
            log_message += f"\n{event.stack_trace}"

        self.logger.log(log_level, log_message, extra={"log_event": event})

        if self.progress_queue:
            ui_message = f"[{event.level}] {event.message}"
            if event.operation:
                ui_message = f"[{event.operation}] {ui_message}"
            self.progress_queue.put({"log": ui_message})

    def debug(self, message: str, component: str = "system", **kwargs):
        """Logs a debug message."""
        self._log_event(self._create_log_event("DEBUG", message, component, **kwargs))

    def info(self, message: str, component: str = "system", **kwargs):
        """Logs an info message."""
        self._log_event(self._create_log_event("INFO", message, component, **kwargs))

    def warning(self, message: str, component: str = "system", **kwargs):
        """Logs a warning message."""
        self._log_event(self._create_log_event("WARNING", message, component, **kwargs))

    def error(self, message: str, component: str = "system", **kwargs):
        """Logs an error message."""
        self._log_event(self._create_log_event("ERROR", message, component, **kwargs))

    def success(self, message: str, component: str = "system", **kwargs):
        """Logs a success message."""
        self._log_event(self._create_log_event("SUCCESS", message, component, **kwargs))

    def critical(self, message: str, component: str = "system", **kwargs):
        """Logs a critical error message."""
        self._log_event(self._create_log_event("CRITICAL", message, component, **kwargs))
