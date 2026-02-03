import os
import sys
from unittest.mock import MagicMock
import threading
from queue import Queue

# Mocking parts of the app to just test UI initialization
sys.path.append(os.getcwd())

from core.config import Config
from core.logger import AppLogger
from core.managers import ModelRegistry, ThumbnailManager
from ui.app_ui import AppUI

def test_ui_init():
    config = Config()
    logger = AppLogger(config)
    progress_queue = Queue()
    cancel_event = threading.Event()
    thumbnail_manager = MagicMock(spec=ThumbnailManager)
    model_registry = MagicMock(spec=ModelRegistry)
    
    app = AppUI(config, logger, progress_queue, cancel_event, thumbnail_manager, model_registry)
    try:
        app.build_ui()
        print("UI Initialization Successful")
    except KeyError as e:
        print(f"UI Initialization FAILED: KeyError {e}")
        sys.exit(1)
    except Exception as e:
        print(f"UI Initialization FAILED: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_ui_init()
