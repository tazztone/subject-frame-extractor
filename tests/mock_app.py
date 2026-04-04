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

# Import core modules early for patching
import app
import core
import core.export
import core.fingerprint
import core.managers
import core.photo_utils
import core.pipelines
import core.utils
import core.xmp_writer
import ui.app_ui
import ui.handlers.pipeline_handlers
from core.application_state import ApplicationState
from core.models import Scene
from core.pipelines import ExtractionPipeline

# Global queue for mock log updates
global_progress_queue = Queue()

# Registry for the active AppUI instance (Fixes "Dual Instance" bug)
_active_app = None


def get_active_app():
    """Returns the most recently initialized AppUI instance."""
    return _active_app


# --- 1. Mock Heavy Dependencies ---


def create_mock_module(name, attributes=None):
    """Creates a proper ModuleType instance populated with mocks/attributes."""
    mock_mod = types.ModuleType(name)
    if attributes:
        for attr, val in attributes.items():
            setattr(mock_mod, attr, val)
    return mock_mod


# Initialize mock objects
mock_torch = MagicMock(name="torch")
mock_torch.cuda.is_available.return_value = False
mock_torch.__version__ = "2.0.0"
mock_torch.nn.Module = MagicMock
mock_torch.Tensor = MagicMock

mock_sam3 = MagicMock(name="sam3")
mock_sam3.model_builder = MagicMock()
mock_sam3.model_builder.build_sam3_predictor = MagicMock()
mock_sam3.model_builder.build_sam3_multiplex_video_predictor = MagicMock()
# Keep legacy name for potential backward compatibility in some tests if needed
mock_sam3.model_builder.build_sam3_video_predictor = MagicMock()


# Create Stable exception classes
class OutOfMemoryError(RuntimeError):
    pass


class VideoOpenFailure(RuntimeError):
    pass


class TransparentContext:
    """Empty context manager that does nothing but allows 'with' blocks."""

    def __enter__(self, *args, **kwargs):
        return self

    def __exit__(self, *args, **kwargs):
        pass

    def __call__(self, func=None):
        if func is not None:
            return func
        return self


def _create_mock_tensor(name="tensor", shape=None, value=None, **kwargs):
    class MockTensor(MagicMock):
        def __len__(self):
            if hasattr(self, "_mock_shape") and self._mock_shape is not None:
                return self._mock_shape[0] if len(self._mock_shape) > 0 else 0
            return 0

        def __getitem__(self, idx):
            new_shape = None
            if hasattr(self, "_mock_shape") and self._mock_shape is not None:
                if len(self._mock_shape) > 0:
                    new_shape = self._mock_shape[1:]
            return _create_mock_tensor(f"{self._mock_name}[{idx}]", shape=new_shape)

        def __gt__(self, other):
            return _create_mock_tensor(f"{self._mock_name} > {other}", shape=getattr(self, "_mock_shape", None))

        def __bool__(self):
            return True

        def cpu(self):
            return self

        def numpy(self):
            s = getattr(self, "_mock_shape", None)
            if s is None:
                s = (100, 100)
            if isinstance(s, int):
                s = (s,)
            return np.zeros(s, dtype=np.float32)

    mock_t = MockTensor(name=name)
    mock_t._mock_shape = shape
    if shape is not None:
        mock_t.shape = shape
    mock_t.device = mock_torch.device("cpu")
    mock_t.dtype = mock_torch.float32
    mock_t.size.side_effect = lambda dim=None: shape if dim is None else shape[dim]
    mock_t.__mul__ = MagicMock(return_value=mock_t)
    mock_t.__add__ = MagicMock(return_value=mock_t)
    mock_t.__sub__ = MagicMock(return_value=mock_t)
    mock_t.__truediv__ = MagicMock(return_value=mock_t)
    mock_t.item.return_value = value if value is not None else 1.0
    return mock_t


mock_torch.cuda.OutOfMemoryError = OutOfMemoryError
mock_torch.cuda.get_device_name = MagicMock(return_value="Mock GPU")
mock_torch.float32 = MagicMock(name="torch.float32")
mock_torch.uint8 = MagicMock(name="torch.uint8")
mock_torch.device = MagicMock()
mock_torch.from_numpy = MagicMock(side_effect=lambda np_arr: _create_mock_tensor("from_numpy", np_arr.shape))
mock_torch.zeros = MagicMock(side_effect=lambda shape, **kwargs: _create_mock_tensor("zeros", shape))
mock_torch.ones = MagicMock(side_effect=lambda shape, **kwargs: _create_mock_tensor("ones", shape))
mock_torch.no_grad = TransparentContext
mock_torch.inference_mode = TransparentContext

modules_map = {
    "torch": create_mock_module(
        "torch",
        {
            "cuda": mock_torch.cuda,
            "nn": mock_torch.nn,
            "Tensor": mock_torch.Tensor,
            "device": mock_torch.device,
            "float32": mock_torch.float32,
            "uint8": mock_torch.uint8,
            "from_numpy": mock_torch.from_numpy,
            "zeros": mock_torch.zeros,
            "ones": mock_torch.ones,
            "no_grad": mock_torch.no_grad,
            "inference_mode": mock_torch.inference_mode,
        },
    ),
    "torch.cuda": mock_torch.cuda,
    "torch.nn": mock_torch.nn,
    "torchvision": create_mock_module("torchvision", {"ops": MagicMock(), "transforms": MagicMock()}),
    "torchvision.ops": MagicMock(),
    "torchvision.transforms": MagicMock(),
    "insightface": create_mock_module(
        "insightface", {"app": create_mock_module("insightface.app", {"FaceAnalysis": MagicMock()})}
    ),
    "insightface.app": MagicMock(),
    "sam3": create_mock_module("sam3", {"model_builder": mock_sam3.model_builder}),
    "sam3.model_builder": mock_sam3.model_builder,
    "mediapipe": create_mock_module("mediapipe", {"tasks": MagicMock()}),
    "pyiqa": MagicMock(),
    "scenedetect": create_mock_module(
        "scenedetect", {"detect": MagicMock(), "VideoOpenFailure": VideoOpenFailure, "ContentDetector": MagicMock()}
    ),
    "yt_dlp": create_mock_module(
        "yt_dlp",
        {"utils": create_mock_module("yt_dlp.utils", {"DownloadError": type("DownloadError", (Exception,), {})})},
    ),
    "numba": create_mock_module("numba", {"njit": lambda f: f, "jit": lambda f: f, "cuda": MagicMock()}),
    "lpips": MagicMock(),
    "matplotlib": create_mock_module(
        "matplotlib", {"pyplot": MagicMock(), "use": MagicMock(), "get_backend": MagicMock(return_value="agg")}
    ),
    "matplotlib.pyplot": MagicMock(),
    "skimage": create_mock_module("skimage", {"metrics": MagicMock()}),
    "skimage.metrics": MagicMock(),
    "safetensors": create_mock_module("safetensors", {"torch": MagicMock()}),
    "onnxruntime": MagicMock(),
}

modules_map["sam3.model"] = create_mock_module("sam3.model")
modules_map["sam3.utils"] = create_mock_module("sam3.utils")
modules_map["sam3.model.sam3_video_predictor"] = create_mock_module("sam3.model.sam3_video_predictor")
modules_map["sam3.model.sam3_video_predictor"].SAM3VideoPredictor = MagicMock()
modules_map["sam3.model.sam3_video_inference"] = create_mock_module("sam3.model.sam3_video_inference")
modules_map["sam3.model.sam3_video_inference"].SAM3VideoInference = MagicMock()

# SAM 3.1 Multiplex Mocks
modules_map["sam3.model.sam3_base_predictor"] = create_mock_module("sam3.model.sam3_base_predictor")
modules_map["sam3.model.sam3_multiplex_video_predictor"] = create_mock_module(
    "sam3.model.sam3_multiplex_video_predictor"
)
modules_map["sam3.model.sam3_multiplex_tracking"] = create_mock_module("sam3.model.sam3_multiplex_tracking")
modules_map["sam3.model.sam3_multiplex_base"] = create_mock_module("sam3.model.sam3_multiplex_base")

for mod_name, mod_obj in modules_map.items():
    if not isinstance(mod_obj, types.ModuleType):
        mock_val = mod_obj
        mod_obj = types.ModuleType(mod_name)
        if isinstance(mock_val, MagicMock):
            for attr in dir(mock_val):
                if not attr.startswith("__"):
                    setattr(mod_obj, attr, getattr(mock_val, attr))
    sys.modules[mod_name] = mod_obj

# --- 2. Patch Pipeline Logic for E2E Speed ---


def mock_extraction_run(self, tracker=None):
    """Mocks the extraction process."""
    print("[Mock] Running Extraction...")
    if tracker:
        tracker.start(10, desc="Mock Extraction")
        for _ in range(10):
            time.sleep(0.01)
            tracker.step(1)
        tracker.set_stage("Extraction Complete")
        tracker.done_stage()

    output_dir = os.path.join(self.config.downloads_dir, "mock_video")
    os.makedirs(output_dir, exist_ok=True)
    thumb_dir = os.path.join(output_dir, "thumbs")
    os.makedirs(thumb_dir, exist_ok=True)

    for i in range(1, 11):
        thumb_path = os.path.join(thumb_dir, f"frame_{i:06d}.webp")
        if not os.path.exists(thumb_path):
            img_data = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            Image.fromarray(img_data).save(thumb_path, "WEBP")

    import json

    # Frame map needs at least 5 frames for some tests
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


def mock_pre_analysis_execution(event, *args, **kwargs):
    """Mocks execute_pre_analysis generator."""
    print("[Mock] Running Pre-Analysis...")
    config = kwargs.get("config") or getattr(args[0] if args else None, "config", None)
    if not config:
        output_dir = "mock_video"
    else:
        output_dir = os.path.join(config.downloads_dir, "mock_video")

    # Return a scene with a seed_result so propagate button enables
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
        "output_dir": output_dir,
        "done": True,
        "unified_status": "Pre-Analysis Complete",
    }


def mock_propagation_execution(event, *args, **kwargs):
    print("[Mock] Running Propagation...")
    yield {
        "unified_log": "Propagation complete.",
        "output_dir": event.output_folder,
        "done": True,
        "scenes": event.scenes,
        "unified_status": "Propagation Complete",
    }


def mock_analysis_execution(event, *args, **kwargs):
    print("[Mock] Running Analysis...")
    output_dir = event.output_folder
    metadata_path = os.path.join(output_dir, "metadata.db")
    yield {
        "unified_log": "Analysis complete.",
        "output_dir": output_dir,
        "metadata_path": metadata_path,
        "done": True,
        "unified_status": "Analysis Complete",
    }


def mock_analysis_orchestrator(event, *args, **kwargs):
    print("[Mock] Running Analysis Orchestrator...")
    yield from mock_pre_analysis_execution(event, *args, **kwargs)
    yield from mock_propagation_execution(event, *args, **kwargs)
    yield from mock_analysis_execution(event, *args, **kwargs)


def mock_ingest_folder(folder_path: str, *args, **kwargs):
    """Mocks ingesting a folder of photos."""
    print(f"[Mock] Ingesting Folder: {folder_path}")
    return [{"filename": "sample.jpg", "path": os.path.join(folder_path, "sample.jpg")}]


def mock_export_xmps_for_photos(photos: list, star_thresholds=None):
    """Mocks XMP export."""
    print(f"[Mock] Exporting {len(photos)} XMPs...")
    return len(photos)


def mock_export_kept_frames(*args, **kwargs):
    print("[Mock] Running Export...")
    return "Export Complete"


def mock_dry_run_export(*args, **kwargs):
    return "🔍 Dry Run: 10 / 10 frames would be exported (MOCKED)."


def reset_app_state():
    global global_state
    from core.application_state import ApplicationState

    global_state = ApplicationState()
    return global_state


def mock_session_load_wrapper(self, session_path: str, current_state: ApplicationState):
    """Mocks PipelineHandler.run_session_load_wrapper."""
    print(f"[Mock] Loading Session: {session_path}")
    if "/non/existent" in session_path:
        # Simulate error for invalid path test
        yield {
            self.app.components["unified_status"]: "Error Loading Session",
            self.app.components["unified_log"]: f"[ERROR] Session directory does not exist: {session_path}",
        }
        return

    new_state = current_state.model_copy()
    new_state.analysis_output_dir = session_path
    new_state.extracted_video_path = "mock_video.mp4"

    # Minimal UI updates to satisfy test assertions
    yield {
        self.app.components["application_state"]: new_state,
        self.app.components["unified_status"]: "Session Loaded.",
        self.app.components["unified_log"]: f"Successfully loaded session from: {session_path}",
        self.app.components["thumb_megapixels_input"]: gr.update(value=0.8),
        self.app.components["filtering_tab"]: gr.update(interactive=True),
        self.app.components["method_input"]: gr.update(value="scene"),
        self.app.components["source_input"]: gr.update(value="mock_video.mp4"),
        self.app.components["scene_filter_status"]: gr.update(value="Found 2 unique people"),
    }


def mock_extraction_wrapper(self, current_state: ApplicationState, *args, **kwargs):
    """Mocks PipelineHandler.run_extraction_wrapper."""
    print(f"[Mock] Running Extraction Wrapper with source: {current_state.source_path}")

    if not current_state.source_path:
        log_msg = "❌ **Error:** Please provide a Source Path or Upload a Video."
        self.app.progress_queue.put({"log": f"[ERROR] {log_msg}"})
        # Match real PipelineHandler error behavior
        yield {
            self.app.components["unified_status"]: "⚠️ Failure in execute_extraction",
            self.app.components["unified_log"]: log_msg,
        }
        return

    self.app.progress_queue.put({"log": "[INFO] Extraction Complete (MOCKED)."})
    yield {
        self.app.components["unified_status"]: "Extraction Complete",
        self.app.components["unified_log"]: "Extraction Complete.",
    }


def mock_pre_analysis_wrapper(self, current_state: ApplicationState, *args, **kwargs):
    """Mocks PipelineHandler.run_pre_analysis_wrapper handles prerequisite checks."""
    app = self.app
    print(f"[Mock] Running Pre-Analysis Wrapper (App Instance: {id(app)})")
    if not current_state.extracted_video_path:
        log_msg = "❌ **Error:** No extracted video found. Please run Extraction first."
        log_entry = f"[ERROR] {log_msg}"
        app.progress_queue.put({"log": log_entry})
        yield {
            app.components["unified_status"]: "⚠️ Error: No extracted video found.",
            app.components["unified_log"]: log_msg,
        }
        return
    log_entry = "[INFO] Pre-Analysis Started (MOCKED)."
    app.progress_queue.put({"log": log_entry})
    from unittest.mock import MagicMock

    mock_event = MagicMock()
    mock_event.video_path = current_state.extracted_video_path
    mock_event.output_folder = self.config.downloads_dir
    yield from mock_pre_analysis_execution(
        mock_event, MagicMock(), MagicMock(), self.logger, self.config, self.thumbnail_manager, False
    )


def mock_propagation_wrapper(self, current_state: ApplicationState, *args, **kwargs):
    """Mocks PipelineHandler.run_propagation_wrapper."""
    print("[Mock] Running Propagation Wrapper")
    yield from mock_propagation_execution(MagicMock(output_folder="mock_video", scenes=[]))


def mock_analysis_wrapper(self, current_state: ApplicationState, *args, **kwargs):
    """Mocks PipelineHandler.run_analysis_wrapper."""
    print("[Mock] Running Analysis Wrapper")
    yield from mock_analysis_execution(MagicMock(output_folder="mock_video"))


# --- 3. Apply Patches ---

ui.app_ui.AppUI.preload_models = MagicMock(side_effect=lambda *args: None)

# Patch PipelineHandler methods at the CLASS level before AppUI instantiates it
import ui.handlers.pipeline_handlers as ph

ph.PipelineHandler.run_session_load_wrapper = mock_session_load_wrapper
ph.PipelineHandler.run_extraction_wrapper = mock_extraction_wrapper
ph.PipelineHandler.run_pre_analysis_wrapper = mock_pre_analysis_wrapper
ph.PipelineHandler.run_propagation_wrapper = mock_propagation_wrapper
ph.PipelineHandler.run_analysis_wrapper = mock_analysis_wrapper

original_main = ui.app_ui.AppUI.build_ui


# Capture the active instance during initialization
original_app_ui_init = ui.app_ui.AppUI.__init__


def mock_app_ui_init(self, *args, **kwargs):
    global _active_app
    original_app_ui_init(self, *args, **kwargs)
    _active_app = self
    print(f"[Mock] Captured Active App Instance: {id(_active_app)}")


ui.app_ui.AppUI.__init__ = mock_app_ui_init


def mock_build_ui(self, *args, **kwargs):
    def reset_app_state_handler():
        """Handles the Reset State button click, returning updates for the UI."""
        new_state = reset_app_state()

        # Drain the progress_queue so stale messages from the previous test
        # don't surface via the LogViewer timer after reset.
        while not global_progress_queue.empty():
            try:
                global_progress_queue.get_nowait()
            except Exception:
                break

        # Clear LogViewer internal state to avoid "Ghost Logs"
        if hasattr(self, "log_viewer"):
            self.log_viewer.all_logs.clear()
            self.log_viewer._last_rendered_log = ""

        c = self.components
        return {
            c["application_state"]: new_state,
            c["unified_status"]: "System Reset Ready.",
            c["unified_log"]: "System Reset Ready.",
        }

    demo = original_main(self, *args, **kwargs)
    with demo:
        with gr.Accordion("Tests (Experimental)", open=True, visible=True):
            reset_btn = gr.Button("Reset State (MOCKED)", variant="stop", elem_id="reset_state_button")
            reset_btn.click(
                fn=reset_app_state_handler,
                inputs=[],
                outputs=[
                    self.components["unified_log"],
                    self.components["unified_status"],
                    self.components["application_state"],
                ],
            )
    return demo


ui.app_ui.AppUI.build_ui = mock_build_ui

for target in [core.pipelines, ui.app_ui, ui.handlers.pipeline_handlers]:
    if hasattr(target, "execute_pre_analysis"):
        target.execute_pre_analysis = mock_pre_analysis_execution
    if hasattr(target, "execute_propagation"):
        target.execute_propagation = mock_propagation_execution
    if hasattr(target, "execute_analysis"):
        target.execute_analysis = mock_analysis_execution
    if hasattr(target, "execute_analysis_orchestrator"):
        target.execute_analysis_orchestrator = mock_analysis_orchestrator

core.fingerprint.create_fingerprint = MagicMock(return_value={})
core.fingerprint.save_fingerprint = MagicMock()
ui.app_ui.export_kept_frames = mock_export_kept_frames
ui.app_ui.dry_run_export = mock_dry_run_export
core.export.export_kept_frames = mock_export_kept_frames
core.utils.download_model = MagicMock()
core.managers.download_model = MagicMock()

ExtractionPipeline._run_impl = mock_extraction_run
core.photo_utils.ingest_folder = mock_ingest_folder
core.xmp_writer.export_xmps_for_photos = mock_export_xmps_for_photos

if __name__ == "__main__":
    os.environ["APP_DEBUG_MODE"] = "true"
    app.Queue = lambda: global_progress_queue
    app.main()
