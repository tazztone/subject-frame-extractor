from __future__ import annotations

import functools
import traceback
from inspect import isgeneratorfunction
from typing import Any, Callable


def safe_ui_callback(context: str):
    """
    Decorator to wrap UI callbacks with error handling, supporting both returns and generators.
    Expected to be used on methods of a class that has a 'logger' and 'components' (or an 'app' instance).
    """

    def decorator(func: Callable):
        if isgeneratorfunction(func):

            @functools.wraps(func)
            def generator_wrapper(instance: Any, *args, **kwargs):
                # Try to find app/logger
                app = getattr(instance, "app", instance)
                try:
                    yield from func(instance, *args, **kwargs)
                except Exception as e:
                    yield _handle_ui_exception(app, e, context)

            return generator_wrapper
        else:

            @functools.wraps(func)
            def sync_wrapper(instance: Any, *args, **kwargs):
                app = getattr(instance, "app", instance)
                try:
                    return func(instance, *args, **kwargs)
                except Exception as e:
                    return _handle_ui_exception(app, e, context)

            return sync_wrapper

    return decorator


def _handle_ui_exception(app: Any, e: Exception, context: str = "Operation") -> dict:
    """Standardized exception handling for UI callbacks."""
    error_msg = f"[ERROR] {context} failed: {e}\n{traceback.format_exc()}"
    if hasattr(app, "logger"):
        app.logger.error(error_msg)

    # Try to return dict with component updates
    updates = {}
    if hasattr(app, "components"):
        if "unified_log" in app.components:
            updates[app.components["unified_log"]] = error_msg
        if "unified_status" in app.components:
            updates[app.components["unified_status"]] = f"❌ **{context} Failed.** Check logs."

    return updates
