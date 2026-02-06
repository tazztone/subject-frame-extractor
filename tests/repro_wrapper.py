
import sys
import os
sys.path.append(os.getcwd())

import gradio as gr
from unittest.mock import MagicMock
from ui.app_ui import AppUI, ApplicationState
from core.events import ExtractionEvent

def test_wrapper():
    # Mock dependencies
    config = MagicMock()
    config.default_primary_seed_strategy = "Auto"
    
    # Mock AppUI with just enough to run the wrapper
    app = MagicMock(spec=AppUI)
    app.config = config
    app.ext_ui_map_keys = ["source_path", "upload_video"]
    app.app_logger = MagicMock()
    
    # Mock components for return dict keys
    app.components = {
        "unified_log": "log_component",
        "unified_status": "status_component" 
    }
    
    # Bind the method from the real class to the mock instance
    app.run_extraction_wrapper = AppUI.run_extraction_wrapper.__get__(app, AppUI)
    
    # Create dummy state
    state = ApplicationState()
    
    # Run with empty args (simulating missing source_path)
    # Args order: current_state, source_path, upload_video
    # But wrapper takes *args and zips with self.ext_ui_map_keys
    
    # Case 1: Empty String for source_path
    print("--- Running wrapper with Empty args ---")
    gen = app.run_extraction_wrapper(state, "", None)
    
    try:
        result = next(gen)
        print("Yielded Result:", result)
        
        # Verify content
        if app.components["unified_log"] in result:
             log_msg = result[app.components["unified_log"]]
             print(f"Log Message: {log_msg}")
             if "[ERROR] Validation failed" in log_msg:
                 print("SUCCESS: Validation error caught and logged.")
             else:
                 print("FAILURE: Validation error NOT in log.")
        else:
             print("FAILURE: unified_log not updated.")
            
    except StopIteration:
        print("FAILURE: Generator finished without yielding.")
    except Exception as e:
        print(f"FAILURE: Generator raised exception: {e}")

if __name__ == "__main__":
    test_wrapper()
