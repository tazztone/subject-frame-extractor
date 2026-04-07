"""
Frame Extractor & Analyzer v4.0.0
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is in path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    import onnxruntime as ort

    if hasattr(ort, "preload_dlls"):
        ort.preload_dlls()
except ImportError:
    pass

import gc
import threading
from queue import Queue

from core.config import Config
from core.database import Database
from core.logger import AppLogger, setup_logging
from core.managers import ModelRegistry, ThumbnailManager
from core.utils.device import empty_cache
from ui.app_ui import AppUI


def cleanup_models(model_registry):
    """
    Clears the model registry and performs garbage collection.

    Args:
        model_registry: The ModelRegistry instance to clear.
    """
    if model_registry:
        model_registry.clear()
    empty_cache()
    gc.collect()


def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Frame Extractor & Analyzer v4.0.0")

    # Server Configuration
    parser.add_argument("--server-name", type=str, help="Server host (e.g., 0.0.0.0)")
    parser.add_argument("--server-port", type=int, help="Server port (e.g., 7860)")
    parser.add_argument("--share", action="store_true", default=None, help="Create a public link")
    parser.add_argument("--auth", type=str, help="Authentication credentials (username:password)")

    # SSL Configuration
    parser.add_argument("--ssl-keyfile", type=str, help="Path to SSL key file")
    parser.add_argument("--ssl-certfile", type=str, help="Path to SSL certificate file")
    parser.add_argument("--ssl-verify", action="store_true", default=None, help="Enable SSL verification")
    parser.add_argument("--no-ssl-verify", action="store_false", dest="ssl_verify", help="Disable SSL verification")
    parser.add_argument(
        "--debug", action="store_true", default=False, help="Enable debug mode (shows Export tab groups at startup)"
    )

    return parser.parse_args()


def main():
    """
    Main entry point for the application.

    Initializes configuration, logging, managers, and the Gradio UI.
    """
    args = parse_args()
    model_registry = None
    try:
        # Create config with CLI overrides
        config_overrides = {k: v for k, v in vars(args).items() if v is not None}
        config = Config(**config_overrides)

        progress_queue = Queue()
        session_log_file = setup_logging(config, progress_queue=progress_queue)
        logger = AppLogger(config=config, session_log_file=session_log_file)

        # 2. Initialize Core components
        config.debug = args.debug  # Sync CLI flag to config
        model_registry = ModelRegistry(logger=logger)
        thumbnail_manager = ThumbnailManager(logger, config)
        database = Database(logger=logger)
        cancel_event = threading.Event()

        app_ui = AppUI(
            config,
            logger,
            progress_queue,
            cancel_event,
            thumbnail_manager,
            model_registry,
            database,
            debug_mode=args.debug,
        )

        demo = app_ui.build_ui()
        logger.info("Frame Extractor & Analyzer v4.0.0\nStarting application...")

        # Prepare launch parameters
        launch_kwargs = {
            "server_name": config.server_name,
            "server_port": config.server_port,
            "share": config.share,
            "ssl_keyfile": config.ssl_keyfile,
            "ssl_certfile": config.ssl_certfile,
            "ssl_verify": config.ssl_verify,
            "allowed_paths": [str(project_root), str(Path(config.downloads_dir).resolve()), *config.allowed_paths],
        }

        if config.auth:
            auth_parts = config.auth.split(":")
            if len(auth_parts) == 2:
                launch_kwargs["auth"] = (auth_parts[0], auth_parts[1])
            else:
                logger.warning(f"Invalid auth format '{config.auth}'. Expected 'username:password'. Auth disabled.")

        logger.info(f"Launching on {config.server_name}:{config.server_port} (Share: {config.share})")
        demo.launch(css=app_ui.css, **launch_kwargs)
    except KeyboardInterrupt:
        if "logger" in locals():
            logger.info("\nApplication stopped by user")
    except Exception as e:
        if "logger" in locals():
            logger.error(f"Error starting application: {e}", exc_info=True)
        else:
            print(f"Error starting application: {e}")
        sys.exit(1)
    finally:
        cleanup_models(model_registry)


if __name__ == "__main__":
    main()
