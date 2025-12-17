"""
Frame Extractor & Analyzer v2.0
"""
import sys
from pathlib import Path

# Ensure project root and SAM3_repo are in path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'SAM3_repo'))

import threading
from queue import Queue
import torch
import gc

from core.config import Config
from core.logger import AppLogger
from core.managers import ModelRegistry, ThumbnailManager
from ui.app_ui import AppUI

def cleanup_models(model_registry):
    """
    Clears the model registry and performs garbage collection.

    Args:
        model_registry: The ModelRegistry instance to clear.
    """
    if model_registry:
        model_registry.clear()
    torch.cuda.empty_cache()
    gc.collect()

def main():
    """
    Main entry point for the application.

    Initializes configuration, logging, managers, and the Gradio UI.
    """
    model_registry = None
    try:
        config = Config()
        logger = AppLogger(config=config)
        model_registry = ModelRegistry(logger=logger)
        thumbnail_manager = ThumbnailManager(logger, config)
        progress_queue = Queue()
        cancel_event = threading.Event()
        logger.set_progress_queue(progress_queue)

        app_ui = AppUI(config, logger, progress_queue, cancel_event, thumbnail_manager, model_registry)
        demo = app_ui.build_ui()
        logger.info("Frame Extractor & Analyzer v2.0\nStarting application...")
        demo.launch()
    except KeyboardInterrupt:
        if 'logger' in locals():
            logger.info("\nApplication stopped by user")
    except Exception as e:
        if 'logger' in locals():
            logger.error(f"Error starting application: {e}", exc_info=True)
        else:
            print(f"Error starting application: {e}")
        sys.exit(1)
    finally:
        cleanup_models(model_registry)

if __name__ == "__main__":
    main()
