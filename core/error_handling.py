"""
Error Handling Infrastructure for Frame Extractor & Analyzer
"""
import functools
import time
import traceback
from enum import Enum
from typing import Any, Callable, Optional


class ErrorSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RecoveryStrategy(Enum):
    RETRY = "retry"
    FALLBACK = "fallback"
    SKIP = "skip"
    ABORT = "abort"


class ErrorHandler:
    def __init__(self, logger: 'AppLogger', max_attempts: int, backoff_seconds: list):
        """
        Initializes the ErrorHandler.

        Args:
            logger: Application logger.
            max_attempts: Default maximum retry attempts.
            backoff_seconds: List of backoff delays in seconds.
        """
        self.logger = logger
        self.max_attempts = max_attempts
        self.backoff_seconds = backoff_seconds

    def with_retry(self, max_attempts: Optional[int] = None, backoff_seconds: Optional[list] = None, recoverable_exceptions: tuple = (Exception,)):
        """
        Decorator that retries the function call upon failure.

        Args:
            max_attempts: Maximum number of attempts.
            backoff_seconds: List of backoff times between retries.
            recoverable_exceptions: Tuple of exceptions to catch and retry.

        Returns:
            Decorated function.
        """
        max_attempts = max_attempts or self.max_attempts
        backoff_seconds = backoff_seconds or self.backoff_seconds
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs) -> Any:
                last_exception = None
                for attempt in range(max_attempts):
                    try:
                        return func(*args, **kwargs)
                    except recoverable_exceptions as e:
                        last_exception = e
                        if attempt < max_attempts - 1:
                            sleep_time = backoff_seconds[min(attempt, len(backoff_seconds) - 1)]
                            self.logger.warning(f"Attempt {attempt + 1} failed, retrying in {sleep_time}s: {str(e)}", component="error_handler")
                            time.sleep(sleep_time)
                        else:
                            self.logger.error(f"All retry attempts failed for {func.__name__}: {str(e)}", component="error_handler", stack_trace=traceback.format_exc())
                raise last_exception
            return wrapper
        return decorator

    def with_fallback(self, fallback_func: Callable):
        """
        Decorator that executes a fallback function if the primary function fails.

        Args:
            fallback_func: Function to call on failure.

        Returns:
            Decorated function.
        """
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs) -> Any:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    self.logger.warning(f"Primary function {func.__name__} failed, using fallback: {str(e)}", component="error_handler")
                    try:
                        return fallback_func(*args, **kwargs)
                    except Exception as fallback_error:
                        self.logger.error(f"Both primary and fallback functions failed for {func.__name__}", component="error_handler", stack_trace=traceback.format_exc())
                        raise fallback_error
            return wrapper
        return decorator
