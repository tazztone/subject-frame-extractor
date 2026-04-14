import os
import sys
import threading
import time
from pathlib import Path
from queue import Queue
from unittest.mock import MagicMock

# Ensure project root is in path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from tests.helpers.sys_mock_modules import inject_mocks_into_sys

inject_mocks_into_sys()

import gradio as gr
import numpy as np
from PIL import Image

# Global instance for fixture access
_active_app = None


def get_active_app():
    global _active_app
    return _active_app


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


# --- 3. Wrapper Mocks ---


def mock_extraction_wrapper(self, current_state, *args, **kwargs):
    # Extract source arg correctly
    source = args[0] if args else ""
    upload = None  # Not mocked fully
    method = args[2] if len(args) > 2 else "all_frames"

    if not (source or upload) or "invalid" in str(source).lower() or "nonsense" in str(source).lower():
        yield {
            self.app.components["unified_status"]: gr.update(value="⚠️ Error/Failure: Failure in execute_extraction"),
            self.app.components["unified_log"]: gr.update(value="[ERROR] Please provide a Source Path"),
            self.app.components["application_state"]: current_state,
        }
        # Required for unit test: tests/unit/test_pipeline_logic.py:42
        self.logger.error("Error/Failure: Invalid Source Path")
        return

    self.app.cancel_event.clear()
    yield {
        self.app.components["unified_status"]: gr.update(value="⏳ Processing (Extraction)"),
        self.app.components["application_state"]: current_state,
        **(self.app._set_busy_state(True) if hasattr(self.app, "_set_busy_state") else {}),
    }

    time.sleep(0.5)
    if self.app.cancel_event.is_set():
        yield {
            self.app.components["unified_status"]: gr.update(value="Cancelled"),
            self.app.components["application_state"]: current_state,
            **(self.app._set_busy_state(False) if hasattr(self.app, "_set_busy_state") else {}),
        }
        return

    new_state = current_state.model_copy()
    is_video = True
    if method == "all_frames":
        is_video = False
    elif isinstance(source, str):
        path_lower = source.lower()
        if "pics" in path_lower or "folder" in path_lower:
            is_video = False
        if path_lower.endswith((".mp4", ".mov", ".avi")):
            is_video = True

    output_dir = os.path.join(self.config.downloads_dir, "extracted_frames")
    os.makedirs(output_dir, exist_ok=True)

    if is_video:
        new_state.extracted_video_path = os.path.join(output_dir, "mock_video.mp4")
        with open(new_state.extracted_video_path, "w") as f:
            f.write("dummy")
    else:
        new_state.extracted_video_path = source

    new_state.extracted_frames_dir = output_dir
    msg = "Extraction Complete. Kept: 10"
    yield {
        self.app.components["application_state"]: new_state,
        self.app.components["unified_status"]: gr.update(value=msg),
        self.app.components["unified_log"]: "Extraction complete: 10 frames saved.",
        **(self.app._set_busy_state(False) if hasattr(self.app, "_set_busy_state") else {}),
    }


def mock_pre_analysis_wrapper(self, current_state, *args, **kwargs):
    # Guard: fail fast if extraction hasn't run yet
    if not current_state.extracted_video_path:
        updates = self.app._set_busy_state(False) if hasattr(self.app, "_set_busy_state") else {}
        updates.update(
            {
                self.app.components["unified_status"]: gr.update(value="⚠️ Error: No extracted video found"),
                self.app.components["unified_log"]: gr.update(value="[ERROR] Please run extraction first."),
                self.app.components["application_state"]: current_state,
            }
        )
        yield updates
        return

    yield {
        self.app.components["unified_status"]: gr.update(value="⏳ Processing (Pre-Analysis)"),
        self.app.components["application_state"]: current_state,
        **(self.app._set_busy_state(True) if hasattr(self.app, "_set_busy_state") else {}),
    }
    time.sleep(0.5)
    new_state = current_state.model_copy()
    new_state.scenes = []

    msg = "Pre-Analysis Complete. Kept: 10"
    log_msg = "Pre-Analysis complete."
    is_image_folder = False
    if "folder" in str(new_state.extracted_video_path).lower() or "pics" in str(new_state.extracted_video_path).lower():
        msg += ". Propagation skipped (source is image folder)."
        log_msg += " Propagation is not needed for image folders."
        is_image_folder = True

    base_updates = self.app._set_busy_state(False) if hasattr(self.app, "_set_busy_state") else {}
    prop_btn = self.app.components["propagate_masks_button"]
    prop_btn_val = base_updates.get(prop_btn)

    # Merge properties
    prop_update = gr.update(visible=not is_image_folder, interactive=True)
    if prop_btn_val:
        # If _set_busy_state provided an update, try to preserve its value/interactivity
        if hasattr(prop_btn_val, "value"):
            prop_update = gr.update(visible=not is_image_folder, interactive=True, value=prop_btn_val.value)

    base_updates.update(
        {
            self.app.components["application_state"]: new_state,
            self.app.components["unified_status"]: gr.update(value=msg),
            self.app.components["unified_log"]: log_msg,
            self.app.components["seeding_results_column"]: gr.update(visible=True),
            self.app.components["propagation_group"]: gr.update(visible=True),
            self.app.components["main_tabs"]: gr.update(selected=2),
            prop_btn: prop_update,
        }
    )

    yield base_updates


def mock_propagation_wrapper(self, current_state, *args, **kwargs):
    yield {
        self.app.components["unified_status"]: gr.update(value="⏳ Processing (Propagation)"),
        self.app.components["application_state"]: current_state,
        **(self.app._set_busy_state(True) if hasattr(self.app, "_set_busy_state") else {}),
    }

    # Poll cancel_event every 0.1s — gives the test a ~1s window to cancel
    for _ in range(10):
        time.sleep(0.1)
        if self.app.cancel_event.is_set():
            updates = self.app._set_busy_state(False) if hasattr(self.app, "_set_busy_state") else {}
            updates.update(
                {
                    self.app.components["unified_status"]: gr.update(value="Cancelled"),
                    self.app.components["application_state"]: current_state,
                }
            )
            yield updates
            return

    yield {
        self.app.components["unified_status"]: gr.update(
            value="""<div class="success-card"><h3>Mask Propagation Complete</h3></div>"""
        ),
        self.app.components["application_state"]: current_state,
        **(self.app._set_busy_state(False) if hasattr(self.app, "_set_busy_state") else {}),
    }


def mock_analysis_wrapper(self, current_state, *args, **kwargs):
    yield {
        self.app.components["unified_status"]: gr.update(value="⏳ Processing (Analysis)"),
        self.app.components["application_state"]: current_state,
        **(self.app._set_busy_state(True) if hasattr(self.app, "_set_busy_state") else {}),
    }
    time.sleep(0.5)
    new_state = current_state.model_copy()
    new_state.analysis_metadata_path = "/tmp/mock.db"
    yield {
        self.app.components["application_state"]: new_state,
        self.app.components["unified_status"]: gr.update(value="Analysis Complete. Kept: 10"),
        self.app.components["filtering_tab"]: gr.update(interactive=True),
        **(self.app._set_busy_state(False) if hasattr(self.app, "_set_busy_state") else {}),
    }


# --- 4. Signature Match Stubs for test_signatures.py ---


def mock_extraction_execution(
    event,
    progress_queue,
    cancel_event,
    logger,
    config,
    model_registry,
    thumbnail_manager=None,
    cuda_available=None,
    progress=None,
):
    yield {"done": True, "output_dir": "/tmp/mock", "video_path": "mock.mp4"}


def mock_pre_analysis_execution(
    event,
    progress_queue,
    cancel_event,
    logger,
    config,
    thumbnail_manager,
    model_registry,
    cuda_available,
    progress=None,
    loaded_models=None,
):
    yield {"done": True, "scenes": [], "output_dir": "/tmp/mock"}


def mock_propagation_execution(
    event,
    progress_queue,
    cancel_event,
    logger,
    config,
    thumbnail_manager,
    model_registry,
    database,
    cuda_available,
    progress=None,
    loaded_models=None,
):
    yield {"done": True, "output_dir": "/tmp/mock"}


def mock_analysis_execution(
    event,
    progress_queue,
    cancel_event,
    logger,
    config,
    thumbnail_manager,
    model_registry,
    database,
    cuda_available,
    progress=None,
    loaded_models=None,
):
    yield {"done": True, "output_dir": "/tmp/mock"}


def mock_ingest_folder(folder_path, output_dir, recursive=False, thumbnails_only=True):
    return []


def mock_export_xmps_for_photos(photos, star_thresholds=None):
    return 0


def mock_export_kept_frames(event, config, logger, progress_queue=None, cancel_event=None):
    return "Export Mock Success"


def mock_session_load_wrapper(self, session_path, *args, **kwargs):
    if "invalid" in str(session_path).lower() or "/non/existent" in str(session_path):
        yield {self.app.components["unified_status"]: gr.update(value="⚠️ Error: Session directory does not exist")}
        return

    from core.application_state import ApplicationState

    reset_state = ApplicationState(
        extracted_video_path="test_video.mp4",
        thumb_megapixels=1.2,
        all_frames_data=[{"id": str(i), "status": "included"} for i in range(1, 3)],
        per_metric_values={"quality_score": [99.0, 99.0]},
    )
    yield {
        self.app.components["application_state"]: reset_state,
        self.app.components["unified_status"]: gr.update(value="Session Loaded. Kept: 2"),
        self.app.components["source_input"]: gr.update(value="input.mp4"),
        self.app.components["thumb_megapixels_input"]: gr.update(value=1.2),
        self.app.components["filtering_tab"]: gr.update(interactive=True),
    }


def build_mock_app(downloads_dir=None, build=True):
    # Delayed imports to avoid circular dependency hell
    import ui.app_ui
    import ui.handlers.pipeline_handlers as ph
    from core.config import Config
    from core.logger import AppLogger, setup_logging
    from core.pipelines import ExtractionPipeline

    global _active_app
    ui.app_ui.AppUI.preload_models = MagicMock(side_effect=lambda *args: None)
    ExtractionPipeline._run_impl = mock_extraction_run

    config = Config()
    if downloads_dir:
        config.downloads_dir = downloads_dir

    # Re-enable progress queue and session log file
    progress_queue = Queue()
    session_log_file = setup_logging(config, progress_queue=progress_queue)
    logger = AppLogger(config, session_log_file=session_log_file)
    app_instance = ui.app_ui.AppUI(
        config, logger, progress_queue, threading.Event(), MagicMock(), MagicMock(), MagicMock()
    )

    ph_obj = app_instance.pipeline_handler
    ph_obj.run_extraction_wrapper = mock_extraction_wrapper.__get__(ph_obj, ph.PipelineHandler)
    ph_obj.run_pre_analysis_wrapper = mock_pre_analysis_wrapper.__get__(ph_obj, ph.PipelineHandler)
    ph_obj.run_propagation_wrapper = mock_propagation_wrapper.__get__(ph_obj, ph.PipelineHandler)
    ph_obj.run_analysis_wrapper = mock_analysis_wrapper.__get__(ph_obj, ph.PipelineHandler)
    ph_obj.run_session_load_wrapper = mock_session_load_wrapper.__get__(ph_obj, ph.PipelineHandler)

    orig_build = app_instance.build_ui

    def build_ui_mocked():
        from core.application_state import ApplicationState

        demo = orig_build()
        with demo:
            with gr.Accordion("Tests", open=True):
                reset_btn = gr.Button("Reset State (MOCKED)", elem_id="reset_state_button")

                def reset_h():
                    s = ApplicationState()
                    s.all_frames_data = [{"id": str(i), "status": "included"} for i in range(1, 11)]
                    s.per_metric_values = {"quality_score": [99.0] * 10}
                    return ["System Reset Ready.", "System Reset Ready. Kept: 10", s]

                reset_btn.click(
                    reset_h,
                    None,
                    [
                        app_instance.components["unified_log"],
                        app_instance.components["unified_status"],
                        app_instance.components["application_state"],
                    ],
                )

                if "quality_score_min_input" in app_instance.components:
                    app_instance.components["quality_score_min_input"].change(
                        lambda v: f"Kept: {0 if float(v) >= 99 else 10}",
                        app_instance.components["quality_score_min_input"],
                        app_instance.components["unified_status"],
                    )
                if "filter_preset_dropdown" in app_instance.components:
                    app_instance.components["filter_preset_dropdown"].change(
                        lambda v: f"Kept: {5 if 'Aggressive' in v else 10}",
                        app_instance.components["filter_preset_dropdown"],
                        app_instance.components["unified_status"],
                    )
        return demo

    app_instance.build_ui = build_ui_mocked
    _active_app = app_instance
    if build:
        app_instance.build_ui()
    return app_instance


if __name__ == "__main__":
    app = build_mock_app(build=False)
    demo = app.build_ui()
    demo.launch(server_port=int(os.environ.get("APP_SERVER_PORT", 7860)))
