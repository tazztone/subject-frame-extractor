import functools
import time
from typing import Callable, Any, Optional, List, Dict
from enum import Enum
import traceback

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
    """Comprehensive error handling with automatic recovery strategies."""

    def __init__(self, logger, config):
        self.logger = logger
        self.config = config
        self.error_count = 0
        self.recovery_attempts = {}

    def with_retry(self,
                  max_attempts: int = 3,
                  backoff_seconds: List[float] = [1, 5, 15],
                  recoverable_exceptions: tuple = (Exception,)):
        """Decorator for automatic retry with exponential backoff."""

        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs) -> Any:
                last_exception = None

                for attempt in range(max_attempts):
                    try:
                        return func(*args, **kwargs)
                    except recoverable_exceptions as e:
                        last_exception = e

                        if attempt < max_attempts - 1:  # Not the last attempt
                            sleep_time = backoff_seconds[min(attempt, len(backoff_seconds) - 1)]
                            self.logger.warning(
                                f"Attempt {attempt + 1} failed, retrying in {sleep_time}s: {str(e)}",
                                component="error_handler",
                                custom_fields={
                                    'function': func.__name__,
                                    'attempt': attempt + 1,
                                    'max_attempts': max_attempts,
                                    'retry_delay': sleep_time
                                }
                            )
                            time.sleep(sleep_time)
                        else:
                            self.logger.error(
                                f"All retry attempts failed for {func.__name__}: {str(e)}",
                                component="error_handler",
                                error_type=type(e).__name__,
                                stack_trace=traceback.format_exc(),
                                custom_fields={
                                    'function': func.__name__,
                                    'total_attempts': max_attempts
                                }
                            )

                raise last_exception

            return wrapper
        return decorator

    def with_fallback(self, fallback_func: Callable):
        """Decorator for automatic fallback to alternative implementation."""

        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs) -> Any:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    self.logger.warning(
                        f"Primary function {func.__name__} failed, using fallback: {str(e)}",
                        component="error_handler",
                        custom_fields={
                            'primary_function': func.__name__,
                            'fallback_function': fallback_func.__name__,
                            'error': str(e)
                        }
                    )

                    try:
                        return fallback_func(*args, **kwargs)
                    except Exception as fallback_error:
                        self.logger.error(
                            f"Both primary and fallback functions failed",
                            component="error_handler",
                            error_type=type(fallback_error).__name__,
                            stack_trace=traceback.format_exc(),
                            custom_fields={
                                'primary_function': func.__name__,
                                'fallback_function': fallback_func.__name__,
                                'primary_error': str(e),
                                'fallback_error': str(fallback_error)
                            }
                        )
                        raise fallback_error

            return wrapper
        return decorator
