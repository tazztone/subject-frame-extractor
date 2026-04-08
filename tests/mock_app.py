import os
import sys
import time
import threading
from pathlib import Path
from queue import Queue
from unittest.mock import MagicMock

# Ensure project root is in path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import gradio as gr
import numpy as np
from PIL import Image

import core.export
import core.fingerprint
import core.managers
import core.photo_utils
import core.utils
import core.xmp_writer

# Import core modules early for patching
import ui.app_ui
import ui.handlers.pipeline_handlers as ph
from core.application_state import ApplicationState
from core.models import Scene
from core.pipelines import ExtractionPipeline

from tests.conftest import _inject_global_mocks
_inject_global_mocks()

# --- 2. Pipeline Mock Logic ---

def mock_extraction_run(self, tracker=None):
    output_dir = os.path.join(self.config.downloads_dir, "mock_video")
    os.makedirs(output_dir, exist_ok=True)
    thumb_dir = os.path.join(output_dir, "thumbs")
    os.makedirs(thumb_dir, exist_ok=True)
    for i in range(1, 11):
        thumb_path = os.path.join(thumb_dir, f"frame_{i:06d}.webp")
        if not os.path.exists(thumb_path):
            Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8)).save(thumb_path, "WEBP")
    return {
        "done": True,
        "output_dir": output_dir,
        "video_path": "mock_video.mp4",
        "extracted_frames_dir_state": output_dir,
        "extracted_video_path_state": "mock_video.mp4",
        "unified_status": "Extraction Complete",
    }


# --- 2b. Signature Contract Stubs (Legacy Compatibility) ---

def mock_extraction_execution(event, progress_queue, cancel_event, logger,
                              config, model_registry, thumbnail_manager=None,
                              cuda_available=None, progress=None, **kwargs):
    yield {"done": True}

def mock_pre_analysis_execution(event, progress_queue, cancel_event, logger,
                                config, thumbnail_manager, model_registry,
                                cuda_available, progress=None, loaded_models=None, **kwargs):
    yield {"done": True}

def mock_propagation_execution(event, progress_queue, cancel_event, logger,
                               config, thumbnail_manager, model_registry,
                               database, cuda_available, progress=None,
                               loaded_models=None, **kwargs):
    yield {"done": True}

def mock_analysis_execution(event, progress_queue, cancel_event, logger,
                            config, thumbnail_manager, model_registry,
                            database, cuda_available, progress=None,
                            loaded_models=None, **kwargs):
    yield {"done": True}

def mock_ingest_folder(folder_path, output_dir, recursive=False, thumbnails_only=True):
    return []

def mock_export_xmps_for_photos(photos, star_thresholds=None):
    return 0

def mock_export_kept_frames(event, config, logger, progress_queue=None, cancel_event=None):
    return "Export Complete (MOCKED)"

# --- 3. Wrapper Mocks ---

def mock_extraction_wrapper(self, current_state: ApplicationState, *args, **kwargs):
    source_path = args[0] if args else None
    upload_video = args[1] if len(args) > 1 else None
    if not (source_path or upload_video):
        yield {
            self.app.components["unified_status"]: gr.update(value="⚠️ Failure in execute_extraction"),
            self.app.components["unified_log"]: gr.update(value="[ERROR] Please provide a Source Path"),
        }
        return

    yield {
        self.app.components["unified_status"]: gr.update(value="⏳ Processing (Extraction)"),
        self.app.components["unified_log"]: gr.update(value="[INFO] Extraction Started (MOCKED)."),
    }
    for _ in range(5):
        time.sleep(0.1)
        if self.app.cancel_event.is_set():
            yield {
                self.app.components["unified_status"]: gr.update(value="Cancelled"),
                self.app.components["unified_log"]: gr.update(value="[WARN] Extraction Cancelled."),
            }
            return
    new_state = current_state.model_copy()
    output_dir = os.path.join(self.config.downloads_dir, "mock_video")
    new_state.extracted_video_path = "mock_video.mp4"
    new_state.extracted_frames_dir = output_dir
    msg = """<div class="success-card"><h3>Extraction Complete</h3></div><p>Kept: 10</p>"""
    yield {
        self.app.components["application_state"]: new_state,
        self.app.components["unified_status"]: gr.update(value=msg),
        self.app.components["unified_log"]: gr.update(value="Extraction Complete."),
    }


def mock_pre_analysis_wrapper(self, current_state: ApplicationState, *args, **kwargs):
    if not current_state.extracted_video_path:
        yield {
            self.app.components["unified_status"]: gr.update(value="⚠️ Error: No extracted video found"),
            self.app.components["unified_log"]: gr.update(value="[ERROR] Please run extraction first."),
        }
        return

    workflow_msg = "Propagation is not needed for image folders"
    
    # HEURISTIC: Hide propagation button if source is likely a folder
    source = args[0] if args else ""
    is_video = not (isinstance(source, str) and (source.startswith("/") or "folder" in source.lower()))
    
    yield {
        self.app.components["unified_status"]: gr.update(value="⏳ Processing (Pre-Analysis)"),
        self.app.components["unified_log"]: gr.update(value=f"[INFO] Pre-Analysis Started. {workflow_msg}"),
    }
    time.sleep(0.5)
    if self.app.cancel_event.is_set():
        yield {
            self.app.components["unified_status"]: gr.update(value="Cancelled"),
            self.app.components["unified_log"]: gr.update(value="[WARN] Pre-Analysis Cancelled."),
        }
        return
    new_state = current_state.model_copy()
    new_state.scenes = [] 
    new_state.extracted_frames_dir = "/tmp/mock"
    msg = f"""<div class="success-card"><h3>Pre-Analysis Complete</h3></div><p>Kept: 10</p><p>{workflow_msg if not is_video else ''}</p>"""
    yield {
        self.app.components["application_state"]: new_state,
        self.app.components["unified_status"]: gr.update(value=msg),
        self.app.components["unified_log"]: gr.update(value=f"Pre-Analysis Complete. {workflow_msg if not is_video else ''}"),
        self.app.components["propagate_masks_button"]: gr.update(visible=is_video, interactive=is_video),
        self.app.components["seeding_results_column"]: gr.update(visible=True),
        self.app.components["propagation_group"]: gr.update(visible=is_video),
    }


def mock_propagation_wrapper(self, current_state: ApplicationState, *args, **kwargs):
    yield {
        self.app.components["unified_status"]: gr.update(value="⏳ Processing (Propagation)"),
        self.app.components["unified_log"]: gr.update(value="[INFO] Propagation Started (MOCKED)."),
    }
    for _ in range(5):
        time.sleep(0.1)
        if self.app.cancel_event.is_set():
            yield {
                self.app.components["unified_status"]: gr.update(value="Cancelled"),
                self.app.components["unified_log"]: gr.update(value="[WARN] Propagation Cancelled."),
            }
            return
    new_state = current_state.model_copy()
    msg = """<div class="success-card"><h3>Mask Propagation Complete</h3></div>"""
    yield {
        self.app.components["application_state"]: new_state,
        self.app.components["unified_status"]: gr.update(value=msg),
        self.app.components["unified_log"]: gr.update(value="Mask Propagation Complete."),
    }


def mock_analysis_wrapper(self, current_state: ApplicationState, *args, **kwargs):
    new_state = current_state.model_copy()
    new_state.analysis_metadata_path = "mock_metadata.db"
    msg = """<div class="success-card"><h3>Analysis Complete</h3></div><p>Kept: 10</p>"""
    updates = {
        self.app.components["application_state"]: new_state,
        self.app.components["unified_status"]: gr.update(value=msg),
        self.app.components["unified_log"]: gr.update(value="Analysis Complete."),
        self.app.components["filter_status_text"]: gr.update(value="*Analysis Loaded (Mock). Kept: 10*"),
    }
    for acc in self.app.components.get("metric_accs", {}).values():
        updates[acc] = gr.update(visible=True)
    yield updates


# --- 4. The Factory ---

def build_mock_app(downloads_dir=None):
    """Factory to create a fresh AppUI instance with all mocks applied."""
    ui.app_ui.AppUI.preload_models = MagicMock(side_effect=lambda *args: None)
    ExtractionPipeline._run_impl = mock_extraction_run
    core.fingerprint.create_fingerprint = MagicMock(return_value={})
    core.fingerprint.save_fingerprint = MagicMock()
    core.utils.download_model = MagicMock()
    core.photo_utils.ingest_folder = MagicMock()
    core.xmp_writer.export_xmps_for_photos = MagicMock()
    core.export.export_kept_frames = MagicMock()

    from core.config import Config
    from core.logger import AppLogger, setup_logging
    
    progress_queue = Queue()
    config = Config()
    if downloads_dir:
        config.downloads_dir = downloads_dir

    session_log_file = setup_logging(config, progress_queue=progress_queue)
    logger = AppLogger(config, session_log_file=session_log_file)
    cancel_event = threading.Event()
    thumbnail_manager = MagicMock()
    model_registry = MagicMock()
    database = MagicMock()
    database.set_db_path = MagicMock()

    app_instance = ui.app_ui.AppUI(
        config, logger, progress_queue, cancel_event, thumbnail_manager, model_registry, database
    )

    # Patch handlers
    app_instance.pipeline_handler.run_extraction_wrapper = mock_extraction_wrapper.__get__(
        app_instance.pipeline_handler, ph.PipelineHandler
    )
    app_instance.pipeline_handler.run_pre_analysis_wrapper = mock_pre_analysis_wrapper.__get__(
        app_instance.pipeline_handler, ph.PipelineHandler
    )
    app_instance.pipeline_handler.run_propagation_wrapper = mock_propagation_wrapper.__get__(
        app_instance.pipeline_handler, ph.PipelineHandler
    )
    app_instance.pipeline_handler.run_analysis_wrapper = mock_analysis_wrapper.__get__(
        app_instance.pipeline_handler, ph.PipelineHandler
    )

    def _mock_load_frames_into_state(self, state):
        frames = [
            {"id": str(i), "filename": f"frame_{i:06d}.webp", "quality_score": 99.0,
             "mask_area_pct": 50.0, "niqe": 2.0, "sharpness": 100.0, "status": "included"}
            for i in range(1, 11)
        ]
        return state.model_copy(update={
            "all_frames_data": frames,
            "per_metric_values": {"quality_score": [99.0] * 10},
            "analysis_output_dir": "/tmp/mock_video",
        })

    app_instance._load_frames_into_state = _mock_load_frames_into_state.__get__(
        app_instance, ui.app_ui.AppUI
    )

    original_build_ui = app_instance.build_ui

    def build_ui_with_reset():
        demo = original_build_ui()
        with demo:
            with gr.Accordion("Tests (Experimental)", open=True, visible=True):
                reset_btn = gr.Button("Reset State (MOCKED)", variant="stop", elem_id="reset_state_button")

                def reset_handler():
                    app_instance.log_viewer.all_logs.clear()
                    while not progress_queue.empty():
                        try: progress_queue.get_nowait()
                        except: break
                    reset_state = ApplicationState(
                        all_frames_data=[
                            {"id": str(i), "filename": f"frame_{i:06d}.webp", "quality_score": 99.0,
                             "mask_area_pct": 50.0, "niqe": 2.0, "sharpness": 100.0, "status": "included"}
                            for i in range(1, 11)
                        ],
                        per_metric_values={"quality_score": [99.0] * 10},
                        analysis_output_dir="/tmp/mock_video",
                    )
                    return [
                        gr.update(value="System Reset Ready."), 
                        gr.update(value="System Reset Ready. Kept: 10"), 
                        reset_state
                    ]

                reset_btn.click(
                    fn=reset_handler,
                    inputs=[],
                    outputs=[
                        app_instance.components["unified_log"],
                        app_instance.components["unified_status"],
                        app_instance.components["application_state"],
                    ],
                )

                # Absolute reactive mock using standard Change event and textContent
                if "quality_score_min_input" in app_instance.components:
                    app_instance.components["quality_score_min_input"].change(
                        fn=None,
                        inputs=[app_instance.components["quality_score_min_input"]],
                        outputs=[app_instance.components["unified_status"]],
                        js="""
                        (val) => {
                            const status = document.getElementById('unified_status');
                            if (status) {
                                status.innerHTML = parseFloat(val) >= 99 ? 'Kept: 0' : 'Kept: 10';
                            }
                            return val;
                        }
                        """
                    )
        app_instance.demo = demo
        return demo

    app_instance.build_ui = build_ui_with_reset
    app_instance.build_ui()
    
    global _active_app
    _active_app = app_instance
    return app_instance


if __name__ == "__main__":
    port = int(os.environ.get("APP_SERVER_PORT", 7860))
    app_instance = build_mock_app()
    app_instance.build_ui()
    app_instance.demo.launch(server_name="127.0.0.1", server_port=port)
