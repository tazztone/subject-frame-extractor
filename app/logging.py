"""Unified logging system with custom SUCCESS level and structured formatting."""

import logging


# Add custom SUCCESS log level for more semantic logging
SUCCESS_LEVEL = 25
logging.addLevelName(SUCCESS_LEVEL, "SUCCESS")


def success_log_method(self, message, *args, **kwargs):
    if self.isEnabledFor(SUCCESS_LEVEL):
        self._log(SUCCESS_LEVEL, message, args, **kwargs)


logging.Logger.success = success_log_method


class StructuredFormatter(logging.Formatter):
    """Custom formatter to include extra context in log messages."""
    def format(self, record):
        # Start with the default formatted message
        s = super().format(record)
        # Find and append extra context
        extra_items = {
            k: v for k, v in record.__dict__.items()
            if k not in logging.LogRecord.__dict__ and k != 'args'
        }
        if extra_items:
            items_str = ', '.join(f'{k}={v}' for k, v in extra_items.items())
            s += f" [{items_str}]"
        return s


class UnifiedLogger:
    def __init__(self):
        self.progress_queue = None
        self.logger = logging.getLogger('unified_logger')
        # Prevent adding handlers multiple times on hot-reload
        if not self.logger.handlers:
            self.logger.setLevel(logging.INFO)
            self.logger.propagate = False

            # Use a simple formatter for the console to keep it clean
            console_formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            )
            ch = logging.StreamHandler()
            ch.setFormatter(console_formatter)
            self.logger.addHandler(ch)

    def set_progress_queue(self, queue):
        """Dynamically set the queue for UI log updates."""
        self.progress_queue = queue

    def add_handler(self, handler):
        """Add a new handler, e.g., for a per-run log file."""
        self.logger.addHandler(handler)
        return handler

    def remove_handler(self, handler):
        """Remove a handler and ensure it's closed properly."""
        if handler:
            self.logger.removeHandler(handler)
            handler.close()

    def _log(self, level_name, message, exc_info=False, extra=None):
        level = logging.getLevelName(level_name.upper())
        self.logger.log(level, message, exc_info=exc_info, extra=extra)
        if self.progress_queue:
            extra_str = ""
            if extra:
                items = ', '.join(f'{k}={v}' for k, v in extra.items())
                extra_str = f" [{items}]"
            log_msg = f"[{level_name.upper()}] {message}{extra_str}"
            self.progress_queue.put({"log": log_msg})

    def info(self, message, extra=None, **kwargs):
        self._log('INFO', message, extra=extra or kwargs)

    def warning(self, message, extra=None, **kwargs):
        self._log('WARNING', message, extra=extra or kwargs)

    def error(self, message, exc_info=False, extra=None, **kwargs):
        self._log('ERROR', message, exc_info=exc_info, extra=extra or kwargs)

    def critical(self, message, exc_info=False, extra=None, **kwargs):
        self._log('CRITICAL', message, exc_info=exc_info,
                  extra=extra or kwargs)

    def success(self, message, extra=None, **kwargs):
        self._log('SUCCESS', message, extra=extra or kwargs)

    def pipeline_error(self, operation, e, extra=None, **kwargs):
        full_context = {'error': str(e), **(extra or kwargs)}
        self.error(f"{operation} failed", exc_info=True, extra=full_context)
        return {"error": str(e)}
