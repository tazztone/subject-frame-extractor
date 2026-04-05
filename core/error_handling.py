"""
Error Handling Infrastructure for Frame Extractor & Analyzer
"""

import functools
import time
import traceback
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Generator, Optional

from core.pipeline_results import PipelineResult

if TYPE_CHECKING:
    from core.logger import LoggerLike


class ErrorSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# TODO: Consider adding NOTIFY strategy for non-blocking alerts
class RecoveryStrategy(Enum):
    RETRY = "retry"
    FALLBACK = "fallback"
    SKIP = "skip"
    ABORT = "abort"


class ErrorHandler:
    def __init__(self, logger: "LoggerLike", max_attempts: int, backoff_seconds: list):
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

    def _log_with_component(self, level: str, message: str, component: str, **kwargs):
        """Helper to call logger with or without component support."""
        log_fn = getattr(self.logger, level.lower())
        # Check if the logger method supports 'component' directly (like AppLogger)
        # We can't easily check the signature at runtime for every call, but we know our AppLogger.
        # A more robust way is to check if it's an instance of AppLogger or just check if it's a proxy.
        from core.logger import AppLogger

        if isinstance(self.logger, AppLogger):
            log_fn(message, component=component, **kwargs)
        else:
            # For standard logger, put component in 'extra'
            extra = kwargs.pop("extra", {})
            extra["component"] = component
            log_fn(message, extra=extra, **kwargs)

    def with_retry(
        self,
        max_attempts: Optional[int] = None,
        backoff_seconds: Optional[list] = None,
        recoverable_exceptions: tuple = (Exception,),
    ):
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
                            self._log_with_component(
                                "warning",
                                f"Attempt {attempt + 1} failed, retrying in {sleep_time}s: {str(e)}",
                                component="error_handler",
                            )
                            time.sleep(sleep_time)
                        else:
                            self._log_with_component(
                                "error",
                                f"All retry attempts failed for {func.__name__}: {str(e)}",
                                component="error_handler",
                                stack_trace=traceback.format_exc(),
                            )
                if last_exception:
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
                    self._log_with_component(
                        "warning",
                        f"Primary function {func.__name__} failed, using fallback: {str(e)}",
                        component="error_handler",
                    )
                    try:
                        return fallback_func(*args, **kwargs)
                    except Exception as fallback_error:
                        self._log_with_component(
                            "error",
                            f"Both primary and fallback functions failed for {func.__name__}",
                            component="error_handler",
                            stack_trace=traceback.format_exc(),
                        )
                        raise fallback_error

            return wrapper

        return decorator


def handle_common_errors(func: Callable) -> Callable:
    """
    Standalone decorator that wraps a generator func, yielding an error dict/result on failure.
    Finds the logger in the arguments to provide contextual logging.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Generator[Any, None, None]:
        # Try to find logger in args or kwargs
        logger = kwargs.get("logger")
        if not logger:
            for arg in args:
                if hasattr(arg, "critical") and hasattr(arg, "error"):
                    # Use AppLogger or standard logger
                    logger = arg
                    break

        try:
            yield from func(*args, **kwargs)
        except Exception as e:
            # Detect CUDA OOM if torch is available
            is_oom = False
            try:
                import torch

                # If torch is a mock, OutOfMemoryError might be a MagicMock instance, not a type.
                # Use getattr and isinstance(oom_type, type) to safely check.
                cuda_module = getattr(torch, "cuda", None)
                oom_type = getattr(cuda_module, "OutOfMemoryError", None)
                if oom_type and isinstance(oom_type, type) and isinstance(e, oom_type):
                    is_oom = True
                # Fallback for mock environments where identity might diverge
                elif type(e).__name__ == "OutOfMemoryError" or "out of memory" in str(e).lower():
                    is_oom = True
            except (ImportError, AttributeError):
                pass

            if is_oom:
                msg = "GPU memory error: Out of memory. Try reducing batch size or resolution."
            elif isinstance(e, FileNotFoundError):
                msg = f"File not found: {e}"
            elif isinstance(e, (ValueError, TypeError)):
                msg = f"Invalid argument or configuration: {e}"
            elif isinstance(e, RuntimeError):
                msg = f"Runtime error during pipeline: {e}"
            else:
                msg = f"Pipeline operation '{func.__name__}' failed: {e}"

            if logger:
                # Use logger.error if available
                logger.error(msg, exc_info=True)
            else:
                print(f"ERROR in {func.__name__}: {msg}")
                traceback.print_exc()

            yield PipelineResult(
                success=False,
                error=str(e),
                unified_log=f"❌ **Error:** {msg}",
                status_message=msg,
                error_message=msg,
                done=False,
            ).model_dump()

    return wrapper
