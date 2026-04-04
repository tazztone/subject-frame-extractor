import os
import sys
import time
import types
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

# --- 1. Mock Heavy Dependencies ---


def create_mock_module(name, attributes=None):
    mock_mod = types.ModuleType(name)
    if attributes:
        for attr, val in attributes.items():
            setattr(mock_mod, attr, val)
    return mock_mod


# Mock Torch
mock_torch = MagicMock(name="torch")
mock_torch.cuda.is_available.return_value = False
mock_torch.__version__ = "2.0.0"
mock_torch.nn.Module = MagicMock
mock_torch.Tensor = MagicMock


class TransparentContext:
    def __enter__(self, *args, **kwargs):
        return self

    def __exit__(self, *args, **kwargs):
        pass

    def __call__(self, func=None):
        return func if func else self


mock_torch.no_grad = TransparentContext
mock_torch.inference_mode = TransparentContext

# Setup sys.modules mocks before any other imports
modules_to_mock = [
    "torch",
    "torch.cuda",
    "torch.nn",
    "torchvision",
    "torchvision.ops",
    "torchvision.transforms",
    "insightface",
    "insightface.app",
    "sam3",
    "sam3.model_builder",
    "mediapipe",
    "pyiqa",
    "scenedetect",
    "yt_dlp",
    "numba",
    "lpips",
    "matplotlib",
    "matplotlib.pyplot",
    "skimage",
    "skimage.metrics",
    "safetensors",
    "onnxruntime",
    "sam3.model",
    "sam3.utils",
    "sam3.model.sam3_video_predictor",
    "sam3.model.sam3_video_inference",
    "sam3.model.sam3_base_predictor",
    "sam3.model.sam3_multiplex_video_predictor",
    "sam3.model.sam3_multiplex_tracking",
    "sam3.model.sam3_multiplex_base",
]

for mod_name in modules_to_mock:
    sys.modules[mod_name] = MagicMock()

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

    import json

    frame_map = {i: f"frame_{i:06d}.webp" for i in range(1, 11)}
    with open(os.path.join(output_dir, "frame_map.json"), "w") as f:
        json.dump(list(frame_map.keys()), f)

    return {
        "done": True,
        "output_dir": output_dir,
        "video_path": "mock_video.mp4",
        "extracted_frames_dir_state": output_dir,
        "extracted_video_path_state": "mock_video.mp4",
        "unified_status": "Extraction Complete",
    }


def mock_pre_analysis_execution(
    event,
    progress_queue,
    cancel_event,
    logger,
    config,
    thumbnail_manager,
    cuda_available,
    progress=None,
    model_registry=None,
    loaded_models=None,
    **kwargs,
):
    scenes = [
        Scene(
            shot_id=1,
            start_frame=0,
            end_frame=25,
            status="included",
            filename="frame_000001.webp",
            seed_result={"bbox": [100, 100, 200, 200], "conf": 0.9},
            seed_metrics={"score": 0.9},
        ).model_dump()
    ]
    yield {
        "unified_log": "Pre-Analysis Complete.",
        "scenes": scenes,
        "done": True,
        "unified_status": "Pre-Analysis Complete",
        "output_dir": event.output_folder,
        "video_path": event.video_path,
    }


def mock_propagation_execution(
    event,
    progress_queue,
    cancel_event,
    logger,
    config,
    thumbnail_manager,
    cuda_available,
    progress=None,
    model_registry=None,
    loaded_models=None,
    **kwargs,
):
    yield {
        "unified_log": "Propagation complete.",
        "done": True,
        "scenes": event.scenes,
        "unified_status": "Propagation Complete",
    }


def mock_analysis_execution(
    event,
    progress_queue,
    cancel_event,
    logger,
    config,
    thumbnail_manager,
    cuda_available,
    progress=None,
    model_registry=None,
    loaded_models=None,
    **kwargs,
):
    yield {"unified_log": "Analysis complete.", "done": True, "unified_status": "Analysis Complete"}


def mock_ingest_folder(folder_path, output_dir, recursive=False, thumbnails_only=True):
    return [{"id": "mock", "source": Path("test.jpg"), "preview": Path("test.jpg"), "type": "jpeg"}]


def mock_export_xmps_for_photos(photos, star_thresholds=None):
    return len(photos)


def mock_export_kept_frames(event, config, logger, progress_queue=None, cancel_event=None):
    return "Export Complete (MOCKED)"


# --- 3. Wrapper Mocks ---


def mock_extraction_wrapper(self, current_state: ApplicationState, *args, **kwargs):
    source_path = args[0] if args else None
    upload_video = args[1] if len(args) > 1 else None
    effective_source = source_path or upload_video
    if not effective_source:
        log_msg = "[ERROR] Please provide a Source Path"
        self.app.progress_queue.put({"log": log_msg})
        # Direct yield for immediate feedback in synchronous handlers,
        # though LogViewer will eventually refresh from queue.
        yield {
            self.app.components["unified_status"]: "⚠️ Failure in execute_extraction",
            self.app.components["unified_log"]: log_msg,
        }
        return

    # Small delay to ensure UI transition is detectable
    time.sleep(0.5)

    # Simulate extraction results that update the state
    new_state = current_state.model_copy()
    output_dir = os.path.join(self.config.downloads_dir, "mock_video")
    new_state.extracted_video_path = "mock_video.mp4"
    new_state.extracted_frames_dir = output_dir

    msg = """<div class="success-card"><h3>Extraction Complete</h3></div>"""

    yield {
        self.app.components["application_state"]: new_state,
        self.app.components["unified_status"]: msg,
        self.app.components["unified_log"]: "Extraction Complete.",
    }


def mock_pre_analysis_wrapper(self, current_state: ApplicationState, *args, **kwargs):
    if not current_state.extracted_video_path:
        log_msg = "[ERROR] No extracted video found."
        self.app.progress_queue.put({"log": log_msg})
        yield {
            self.app.components["unified_status"]: "⚠️ Error: No extracted video found.",
            self.app.components["unified_log"]: log_msg,
        }
        return

    # Ensure progress log is visible
    self.app.progress_queue.put({"log": "[INFO] Pre-Analysis Started (MOCKED)."})

    # Simulate success
    new_state = current_state.model_copy()
    new_state.scenes = []  # Mocked
    new_state.extracted_frames_dir = "/tmp/mock"

    msg = """<div class="success-card"><h3>Pre-Analysis Complete</h3></div>"""

    yield {
        self.app.components["application_state"]: new_state,
        self.app.components["unified_status"]: msg,
        self.app.components["unified_log"]: "Pre-Analysis Complete.",
        self.app.components["propagate_masks_button"]: gr.update(visible=True, interactive=True),
        self.app.components["seeding_results_column"]: gr.update(visible=True),
        self.app.components["propagation_group"]: gr.update(visible=True),
    }


def mock_propagation_wrapper(self, current_state: ApplicationState, *args, **kwargs):
    print("[Mock] Running Propagation Wrapper")
    new_state = current_state.model_copy()
    msg = """<div class="success-card"><h3>Mask Propagation Complete</h3></div>"""
    yield {
        self.app.components["application_state"]: new_state,
        self.app.components["unified_status"]: msg,
        self.app.components["unified_log"]: "Propagation Complete.",
    }


def mock_analysis_wrapper(self, current_state: ApplicationState, *args, **kwargs):
    print("[Mock] Running Analysis Wrapper")
    new_state = current_state.model_copy()
    new_state.analysis_metadata_path = "mock_metadata.db"
    msg = """<div class="success-card"><h3>Analysis Complete</h3></div>"""
    yield {
        self.app.components["application_state"]: new_state,
        self.app.components["unified_status"]: msg,
        self.app.components["unified_log"]: "Analysis Complete.",
    }


# --- 4. The Factory ---

_active_app = None


def get_active_app():
    return _active_app


def build_mock_app(downloads_dir=None):
    """Factory to create a fresh AppUI instance with all mocks applied."""
    global _active_app

    # 1. Class-level patches for methods called during __init__
    ui.app_ui.AppUI.preload_models = MagicMock(side_effect=lambda *args: None)

    ExtractionPipeline._run_impl = mock_extraction_run
    core.fingerprint.create_fingerprint = MagicMock(return_value={})
    core.fingerprint.save_fingerprint = MagicMock()
    core.utils.download_model = MagicMock()

    # Missing attributes for signature test
    core.photo_utils.ingest_folder = mock_ingest_folder
    core.xmp_writer.export_xmps_for_photos = mock_export_xmps_for_photos
    core.export.export_kept_frames = mock_export_kept_frames

    # 2. Instantiate
    from core.config import Config
    from core.logger import AppLogger, setup_logging

    progress_queue = Queue()
    config = Config()
    if downloads_dir:
        config.downloads_dir = downloads_dir

    session_log_file = setup_logging(config, progress_queue=progress_queue)
    logger = AppLogger(config, session_log_file=session_log_file)
    cancel_event = MagicMock()
    thumbnail_manager = MagicMock()
    model_registry = MagicMock()

    app_instance = ui.app_ui.AppUI(config, logger, progress_queue, cancel_event, thumbnail_manager, model_registry)

    # 3. Instance-level patches for handlers (CRITICAL for component alignment)
    # We use __get__ to bind the function as a method to the instance
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

    # Wrap build_ui to add the reset button
    original_build_ui = app_instance.build_ui

    def build_ui_with_reset(self):
        demo = original_build_ui()
        with demo:
            with gr.Accordion("Tests (Experimental)", open=True, visible=True):
                reset_btn = gr.Button("Reset State (MOCKED)", variant="stop", elem_id="reset_state_button")

                def reset_handler():
                    app_instance.log_viewer.all_logs.clear()
                    while not progress_queue.empty():
                        try:
                            progress_queue.get_nowait()
                        except:
                            break
                    return [
                        "System Reset Ready.",  # unified_log
                        "System Reset Ready.",  # unified_status
                        ApplicationState(),  # application_state
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
        return demo

    app_instance.build_ui = types.MethodType(build_ui_with_reset, app_instance)
    app_instance.build_ui()  # Populate components
    _active_app = app_instance
    return app_instance


if __name__ == "__main__":
    port = int(os.environ.get("APP_SERVER_PORT", 7860))
    app_ui = build_mock_app()
    demo = app_ui.build_ui()
    demo.launch(server_name="127.0.0.1", server_port=port)
