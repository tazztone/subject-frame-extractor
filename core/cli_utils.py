"""
Internal CLI utilities for runtime setup and pipeline execution.
"""

import threading
from pathlib import Path
from queue import Queue

import click

from core.config import Config
from core.logger import AppLogger, setup_logging
from core.managers import ModelRegistry, ThumbnailManager


def _setup_runtime(output_dir: Path, verbose: bool = False):
    """Initialize shared runtime components."""
    config = Config()
    config.monitoring_memory_watchdog_enabled = False  # Disable watchdog for CLI
    config.log_level = "DEBUG" if verbose else "INFO"

    progress_queue = Queue()
    setup_logging(config, log_dir=output_dir, log_to_console=True, progress_queue=None)
    logger = AppLogger(config)

    cancel_event = threading.Event()
    model_registry = ModelRegistry(logger)
    thumbnail_manager = ThumbnailManager(logger, config)

    return config, logger, progress_queue, cancel_event, model_registry, thumbnail_manager


def _run_pipeline(generator, stage_name: str) -> dict:
    """Consume a pipeline generator and return the final result."""
    result = None
    for update in generator:
        if isinstance(update, dict):
            result = update
            # Print progress if available
            if "unified_log" in update:
                click.echo(f"  {update['unified_log']}")

    if not result or not result.get("done"):
        if result:
            raise click.ClickException(f"{stage_name} failed: {result.get('unified_log', 'Unknown error')}")
        else:
            raise click.ClickException(f"{stage_name} failed unexpectedly.")

    click.secho(f"✓ {stage_name} complete.", fg="green")
    return result
