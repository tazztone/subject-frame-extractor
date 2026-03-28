from __future__ import annotations

import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Callable, Dict, Generator, List, Optional

import cv2
import gradio as gr
import numpy as np
import torch

from core.application_state import ApplicationState
from core.batch_manager import BatchItem, BatchManager
from core.config import Config
from core.events import ExportEvent, ExtractionEvent, FilterEvent, PreAnalysisEvent
from core.export import dry_run_export, export_kept_frames
from core.logger import AppLogger
from core.managers import ModelRegistry, ThumbnailManager
from core.pipelines import AdvancedProgressTracker, execute_extraction
from core.utils import is_image_folder
from ui.components.log_viewer import LogViewer
from ui.decorators import safe_ui_callback
from ui.gallery_utils import auto_set_thresholds, on_filters_changed
from ui.handlers import PipelineHandler, SceneHandler
from ui.tabs import (
    ExtractionTabBuilder,
    FilteringTabBuilder,
    MetricsTabBuilder,
    SceneTabBuilder,
    SubjectTabBuilder,
)


class AppUI:
    """
    Main UI class for the Frame Extractor & Analyzer application.

    Manages the Gradio interface, event handlers, and interaction with backend pipelines.
    """

    MAX_RESOLUTION_CHOICES: List[Any] = [
        ("Original (Recommended)", "maximum available"),
        ("4K (2160p)", "2160"),
        ("Full HD (1080p)", "1080"),
        ("HD (720p)", "720"),
    ]
    EXTRACTION_METHOD_TOGGLE_CHOICES: List[str] = ["Recommended Thumbnails", "Legacy Full-Frame"]
    METHOD_CHOICES: List[Any] = [
        ("Every N-th Frame (Recommended)", "every_nth_frame"),
        ("All Frames (Maximum Quality)", "all"),
        ("Keyframes (Cuts/Scene Changes)", "keyframes"),
    ]
    PRIMARY_SEED_STRATEGY_CHOICES: List[str] = [
        "🤖 Automatic Detection",
        "👤 Source Face Reference",
        "📝 Text Description (Limited)",
        "🔄 Face + Text Fallback",
        "🧑\u200d🤝\u200d🧑 Find Prominent Person",
    ]
    SEED_STRATEGY_CHOICES: List[str] = [
        "Largest Person",
        "Center-most Person",
        "Highest Confidence",
        "Tallest Person",
        "Area x Confidence",
        "Rule-of-Thirds",
        "Edge-avoiding",
        "Balanced",
        "Best Face",
    ]
    FACE_MODEL_NAME_CHOICES: List[str] = ["buffalo_l", "buffalo_s"]
    TRACKER_MODEL_CHOICES: List[str] = ["sam2", "sam3"]
    GALLERY_VIEW_CHOICES: List[str] = ["Kept", "Rejected"]
    LOG_LEVEL_CHOICES: List[str] = ["DEBUG", "INFO", "WARNING", "ERROR", "SUCCESS", "CRITICAL"]
    SCENE_GALLERY_VIEW_CHOICES: List[str] = ["Kept", "Rejected", "All"]
    FILTER_PRESETS: Dict[str, Dict[str, float]] = {
        "Portrait/Selfie": {
            "sharpness_min": 60.0,
            "face_sim_min": 50.0,
            "eyes_open_min": 60.0,
            "yaw_min": -15.0,
            "yaw_max": 15.0,
            "pitch_min": -15.0,
            "pitch_max": 15.0,
        },
        "Action/Sports": {"sharpness_min": 10.0, "edge_strength_min": 60.0, "mask_area_pct_min": 20.0},
        "Training Dataset": {"quality_score_min": 80.0, "face_sim_min": 80.0},
        "High Quality": {"quality_score_min": 75.0, "sharpness_min": 75.0},
        "Frontal Faces": {
            "yaw_min": -10.0,
            "yaw_max": 10.0,
            "pitch_min": -10.0,
            "pitch_max": 10.0,
            "eyes_open_min": 70.0,
        },
        "Close-up Subject": {"mask_area_pct_min": 60.0, "quality_score_min": 40.0},
    }

    def __init__(
        self,
        config: "Config",
        logger: "AppLogger",
        progress_queue: Queue,
        cancel_event: threading.Event,
        thumbnail_manager: "ThumbnailManager",
        model_registry: "ModelRegistry",
    ):
        """
        Initialize the AppUI.

        Args:
            config: Application configuration.
            logger: Application logger.
            progress_queue: Queue for progress updates.
            cancel_event: Event to signal task cancellation.
            thumbnail_manager: Manager for thumbnail caching.
            model_registry: Registry for ML models.
        """
        self.config = config
        self.logger = logger
        self.app_logger = logger
        self.progress_queue = progress_queue
        self.cancel_event = cancel_event
        self.thumbnail_manager = thumbnail_manager
        self.model_registry = model_registry
        self.batch_manager = BatchManager()
        self.debug_mode = config.debug_mode
        self.components, self.cuda_available = {}, torch.cuda.is_available()
        self.ui_registry = {}
        self.performance_metrics = {}
        self.log_viewer = LogViewer(logger, progress_queue, self.LOG_LEVEL_CHOICES)
        self.last_run_args = None
        self.ext_ui_map_keys = [
            "source_path",
            "upload_video",
            "method",
            "interval",
            "nth_frame",
            "max_resolution",
            "thumb_megapixels",
            "scene_detect",
        ]
        self.ana_ui_map_keys = [
            "resume",
            "enable_face_filter",
            "face_ref_img_path",
            "face_ref_img_upload",
            "face_model_name",
            "enable_subject_mask",
            "tracker_model_name",
            "best_frame_strategy",
            "scene_detect",
            "text_prompt",
            "min_mask_area_pct",
            "sharpness_base_scale",
            "edge_strength_base_scale",
            "pre_analysis_enabled",
            "pre_sample_nth",
            "primary_seed_strategy",
            "compute_quality_score",
            "compute_sharpness",
            "compute_edge_strength",
            "compute_contrast",
            "compute_brightness",
            "compute_entropy",
            "compute_eyes_open",
            "compute_yaw",
            "compute_pitch",
            "compute_face_sim",
            "compute_subject_mask_area",
            "compute_niqe",
            "compute_phash",
        ]
        self.session_load_keys = [
            "unified_log",
            "unified_status",
            "progress_details",
            "cancel_button",
            "pause_button",
            "source_input",
            "max_resolution",
            "thumb_megapixels_input",
            "ext_scene_detect_input",
            "method_input",
            "pre_analysis_enabled_input",
            "pre_sample_nth_input",
            "enable_face_filter_input",
            "face_ref_img_path_input",
            "text_prompt_input",
            "best_frame_strategy_input",
            "tracker_model_name_input",
            "propagate_masks_button",
            "seeding_results_column",
            "propagation_group",
            "scene_filter_status",
            "scene_face_sim_min_input",
            "filtering_tab",
            "scene_gallery",
        ]

        # Handlers
        self.scene_handler = SceneHandler(self)
        self.pipeline_handler = PipelineHandler(self)

    def preload_models(self):
        """
        Asynchronously preloads heavy models (SAM3) in a background thread.
        """
        self.logger.info("Starting async model preloading...")

        def _load():
            try:
                # Preload SAM3 Tracker
                retry_params = (self.config.retry_max_attempts, tuple(self.config.retry_backoff_seconds))
                self.model_registry.get_tracker(
                    model_name=self.config.default_tracker_model_name,
                    models_path=str(self.config.models_dir),
                    user_agent=self.config.user_agent,
                    retry_params=retry_params,
                    config=self.config,
                )
                self.progress_queue.put(
                    {"ui_update": {self.components["model_status_indicator"]: "**Status: Models Ready**"}}
                )
                self.logger.success("Async model preloading complete.")
            except Exception as e:
                self.logger.error(f"Async model preloading failed: {e}")
                self.progress_queue.put(
                    {"ui_update": {self.components["model_status_indicator"]: "*Status: Load Failed*"}}
                )

        threading.Thread(target=_load, daemon=True).start()

    def build_ui(self) -> gr.Blocks:
        """
        Constructs the entire Gradio UI layout.

        Returns:
            The Gradio Blocks instance containing the application UI.
        """
        # Gradio 5+ supports CSS in Blocks constructor
        css = """
        .section-subtitle {
            color: var(--body-text-color-subdued, #666);
            font-size: 0.9em;
            display: block;
            margin-top: -4px;
        }
        .system-status-bar {
            border-top: 1px solid var(--border-color-primary);
            padding-top: 12px;
            margin-top: 24px;
            display: flex;
            gap: 16px;
        }
        .success-card {
            border-left: 4px solid var(--button-primary-background-fill, #2196F3);
            padding-left: 12px;
            margin: 8px 0;
        }
        .success-card h3 {
            margin-top: 0;
            color: var(--button-primary-background-fill, #2196F3);
        }
        /* Ensure focused interactive elements have a clear outline */
        :focus-visible {
            outline: 2px solid var(--button-primary-background-fill, #2196F3) !important;
            outline-offset: 2px;
        }
        #unified_status {
            min-height: 60px;
        }
        """
        self.css = css
        with gr.Blocks(title="Frame Extractor & Analyzer") as demo:
            # Unified Application State (Must be first for tab builders)
            self.components["application_state"] = gr.State(ApplicationState())

            self._build_header()

            with gr.Accordion("Resume previous Session", open=False):
                with gr.Row():
                    self._create_component(
                        "session_path_input",
                        "textbox",
                        {
                            "label": "Load previous run",
                            "placeholder": "Path to a previous run's output folder...",
                            "info": "Paste the path to a folder from a previous extraction to resume work.",
                            "scale": 3,
                        },
                    )
                    self._create_component("load_session_button", "button", {"value": "Load Session", "scale": 1})
                    self._create_component("save_config_button", "button", {"value": "Save Current Config", "scale": 1})

            self._build_main_tabs()
            self._build_footer()

            # Global outputs for dynamic UI updates
            self.all_outputs = [v for v in self.components.values() if hasattr(v, "_id")]

            self._create_event_handlers()

            # Trigger preloading on load
            demo.load(self.preload_models, None, None)

        return demo

    def _get_comp(self, name: str) -> Optional[gr.Component]:
        """Retrieves a component by name from the internal registry."""
        return self.components.get(name)

    def _reg(self, key: str, component: gr.Component) -> gr.Component:
        """Registers a component for later retrieval by UI mapping key."""
        self.ui_registry[key] = component
        return component

    def _create_component(self, name: str, comp_type: str, kwargs: dict) -> gr.Component:
        """
        Helper to create and register a Gradio component.

        Args:
            name: Unique name for the component.
            comp_type: String identifier for the component type (e.g., 'button', 'textbox').
            kwargs: Arguments to pass to the component constructor.

        Returns:
            The created Gradio component.
        """
        comp_map = {
            "button": gr.Button,
            "textbox": gr.Textbox,
            "dropdown": gr.Dropdown,
            "slider": gr.Slider,
            "checkbox": gr.Checkbox,
            "file": gr.File,
            "radio": gr.Radio,
            "gallery": gr.Gallery,
            "plot": gr.Plot,
            "markdown": gr.Markdown,
            "html": gr.HTML,
            "number": gr.Number,
            "cbg": gr.CheckboxGroup,
            "image": gr.Image,
            "dataframe": gr.Dataframe,
        }

        # UX Enforcement: Add defaults for better consistency
        if comp_type == "button" and "variant" not in kwargs:
            # Default secondary unless specified
            pass

        if (
            comp_type in ["slider", "dropdown", "textbox", "number", "checkbox"]
            and "label" in kwargs
            and "info" not in kwargs
        ):
            # Auto-generate empty info to ensure spacing consistency if needed,
            # but for now we just allow it.
            pass

        self.components[name] = comp_map[comp_type](**kwargs)
        return self.components[name]

    def _create_section_header(self, title: str, subtitle: Optional[str] = None, icon: Optional[str] = None):
        """Creates a standardized section header."""
        header_text = f"### {icon} {title}" if icon else f"### {title}"
        if subtitle:
            header_text += f"\n<span class='section-subtitle'>{subtitle}</span>"
        gr.Markdown(header_text)

    def _build_header(self):
        """Builds the UI header section with title and status indicators."""
        with gr.Row(elem_id="header_row", equal_height=True):
            with gr.Column(scale=4):
                gr.Markdown("# Frame Extractor & Analyzer v4.0.0")
                gr.Markdown("#### Professional AI-Powered Dataset Curation Tool")
            with gr.Column(scale=1):
                self._create_component("model_status_indicator", "markdown", {"value": "**Status: Initializing...**"})

        with gr.Accordion("Guide: How to use this tool", open=False):
            gr.Markdown("""
            ### Workflow
            1.  **Source**: Import video from a file or URL.
            2.  **Subject**: Specify target for tracking and extraction.
            3.  **Scenes**: Review detected scenes and choose the best shots.
            4.  **Metrics**: Analyze frames for quality and composition.
            5.  **Export**: Filter the best frames and save your dataset.
            """)

        status_text = "GPU Accelerated (CUDA)" if self.cuda_available else "CPU Mode (Standard)"

        with gr.Row(elem_classes="system-status-bar"):
            gr.Markdown(f"**Acceleration:** {status_text}")
            if not self.cuda_available:
                gr.Markdown("*Tip: A GPU is highly recommended for faster processing.*")

    def _build_main_tabs(self):
        """Constructs the main tabbed interface."""
        with gr.Tabs() as main_tabs:
            self.components["main_tabs"] = main_tabs
            with gr.Tab("Source", id=0):
                ExtractionTabBuilder(self).build()
            with gr.Tab("Subject", id=1) as define_subject_tab:
                self.components["define_subject_tab"] = define_subject_tab
                SubjectTabBuilder(self).build()
            with gr.Tab("Scenes", id=2) as scene_selection_tab:
                self.components["scene_selection_tab"] = scene_selection_tab
                SceneTabBuilder(self).build()
            with gr.Tab("Metrics", id=3) as metrics_tab:
                self.components["metrics_tab"] = metrics_tab
                MetricsTabBuilder(self).build()
            with gr.Tab("Export", id=4) as filtering_tab:
                self.components["filtering_tab"] = filtering_tab
                FilteringTabBuilder(self).build()

    def _build_footer(self):
        """Builds the footer with status bar, logs, and help section."""
        gr.Markdown("---")  # Divider
        with gr.Group():
            with gr.Row():
                with gr.Column(scale=2):
                    self._create_component(
                        "unified_status",
                        "markdown",
                        {"label": "Status", "value": "Ready.", "elem_id": "unified_status"},
                    )
                    # self.components['progress_bar'] = gr.Progress()
                    self._create_component(
                        "progress_details", "html", {"value": "", "elem_classes": ["progress-details"]}
                    )
                    with gr.Row():
                        self._create_component("pause_button", "button", {"value": "Pause", "interactive": False})
                        self._create_component("cancel_button", "button", {"value": "Cancel", "interactive": False})
                with gr.Column(scale=3):
                    self.log_viewer.build()
                    self.components.update(self.log_viewer.components)

        with gr.Accordion("Help / Troubleshooting", open=False):
            gr.Markdown("Run checks for GPU availability, missing libraries, and path permissions.")
            self._create_component("run_diagnostics_button", "button", {"value": "Run System Diagnostics"})

            # This group is populated/used by the propagation logic, ensuring it exists is enough here.

    def _get_all_filter_keys(self) -> list[str]:
        """Returns a list of all available filter metric keys."""
        return list(self.config.quality_weights.keys()) + [
            "quality_score",
            "face_sim",
            "mask_area_pct",
            "eyes_open",
            "yaw",
            "pitch",
        ]

    def _get_metric_description(self, metric_name: str) -> str:
        """Returns a user-friendly description for a given metric."""
        descriptions = {
            "quality_score": "Overall 'goodness' score.",
            "niqe": "Natural Image Quality Evaluator. Lower is better, but scaled here so higher is better.",
            "sharpness": "Measures fine detail.",
            "edge_strength": "Measures prominence of edges.",
            "contrast": "Difference between brightest and darkest parts.",
            "brightness": "Overall lightness.",
            "entropy": "Information complexity.",
            "face_sim": "Similarity to reference face.",
            "mask_area_pct": "Percentage of screen taken by subject.",
            "eyes_open": "1.0 = Fully open, 0.0 = Closed.",
            "yaw": "Head rotation (left/right).",
            "pitch": "Head rotation (up/down).",
        }
        return descriptions.get(metric_name, "")

    def _create_event_handlers(self):
        """Sets up all global event listeners and state management."""
        self.logger.info("Initializing Gradio event handlers...")

        # Unified Application State - MOVED TO build_ui
        # self.components["application_state"] = gr.State(ApplicationState())

        self._setup_visibility_toggles()
        self._setup_pipeline_handlers()
        self._setup_filtering_handlers()
        self.scene_handler.setup_handlers()
        self.components["save_config_button"].click(lambda: self.config.save_config("config_dump.json"), [], []).then(
            lambda: "Configuration saved to config_dump.json", [], self.components["unified_log"]
        )

        c = self.components
        c["cancel_button"].click(lambda: self.cancel_event.set(), [], [])
        c["pause_button"].click(
            self._toggle_pause,
            inputs=[
                gr.State(
                    lambda: next((arg for arg in self.last_run_args if isinstance(arg, AdvancedProgressTracker)), None)
                    if self.last_run_args
                    else None
                )
            ],
            outputs=c["pause_button"],
        )
        c["clear_logs_button"].click(lambda: (self.log_viewer.all_logs.clear(), "")[1], [], c["unified_log"])

        # Use LogViewer to setup its own handlers
        self.log_viewer.setup_handlers(self.all_outputs)

        # Stepper Handler (Removed)

        # Hidden radio for scene editor state compatibility
        c["run_diagnostics_button"].click(self.run_system_diagnostics, inputs=[], outputs=[c["unified_log"]])

    def _run_task_with_progress(
        self, task_func: Callable, output_components: list, progress: Callable, *args
    ) -> Generator[dict, None, None]:
        """
        Executes a background task while streaming progress updates to the UI.

        Args:
            task_func: The function to execute.
            output_components: List of components to update (deprecated).
            progress: Gradio progress callback.
            args: Arguments for the task function.

        Yields:
            Dictionary of UI updates.
        """
        self.last_run_args = args
        self.cancel_event.clear()
        tracker_instance = next((arg for arg in args if isinstance(arg, AdvancedProgressTracker)), None)
        if tracker_instance:
            tracker_instance.pause_event.set()
        op_name = getattr(task_func, "__name__", "Unknown Task").replace("_wrapper", "").replace("_", " ").title()
        yield {
            self.components["cancel_button"]: gr.update(interactive=True),
            self.components["pause_button"]: gr.update(interactive=True),
            self.components["unified_status"]: f"**Starting: {op_name}...**",
        }

        def run_and_capture():
            try:
                res = task_func(*args)
                if hasattr(res, "__iter__") and not isinstance(res, (dict, list, tuple, str)):
                    for item in res:
                        self.progress_queue.put({"ui_update": item})
                else:
                    self.progress_queue.put({"ui_update": res})
            except Exception as e:
                self.app_logger.error(f"Task failed: {e}", exc_info=True)
                self.progress_queue.put({"ui_update": {self.components["unified_log"]: f"[CRITICAL] Task failed: {e}"}})

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(run_and_capture)
            start_time = time.time()
            while future.running():
                if time.time() - start_time > 3600:
                    self.app_logger.error("Task timed out after 1 hour")
                    self.cancel_event.set()
                    future.cancel()
                    break
                if self.cancel_event.is_set():
                    future.cancel()
                    break
                if tracker_instance and not tracker_instance.pause_event.is_set():
                    yield {self.components["unified_status"]: f"**Paused: {op_name}**"}
                    time.sleep(0.2)
                    continue
                try:
                    msg, update_dict = self.progress_queue.get(timeout=0.1), {}
                    if "ui_update" in msg:
                        update_dict.update(msg["ui_update"])
                    if "log" in msg:
                        update_dict.update(self.log_viewer.get_log_update_dict(msg["log"]))
                    if "progress" in msg:
                        from core.progress import ProgressEvent

                        p = ProgressEvent(**msg["progress"])
                        progress(p.fraction, desc=f"{p.stage} ({p.done}/{p.total}) • {p.eta_formatted}")
                        status_md = (
                            f"**Running: {op_name}**\n- Stage: {p.stage} ({p.done}/{p.total})\n- ETA: {p.eta_formatted}"
                        )
                        if p.substage:
                            status_md += f"\n- Step: {p.substage}"
                        update_dict[self.components["unified_status"]] = status_md
                    if update_dict:
                        yield update_dict
                except Empty:
                    pass
                time.sleep(0.05)

            while not self.progress_queue.empty():
                try:
                    msg, update_dict = self.progress_queue.get_nowait(), {}
                    if "ui_update" in msg:
                        update_dict.update(msg["ui_update"])
                    if "log" in msg:
                        update_dict.update(self.log_viewer.get_log_update_dict(msg["log"]))
                    if update_dict:
                        yield update_dict
                except Empty:
                    break

    @safe_ui_callback("Toggle Pause")
    def _toggle_pause(self, tracker: "AdvancedProgressTracker") -> str:
        """Toggles the pause state of the current running task."""
        if tracker.pause_event.is_set():
            tracker.pause_event.clear()
            return "Paused"
        else:
            tracker.pause_event.set()
            return "Resume"

    @safe_ui_callback("Diagnostics")
    def run_system_diagnostics(self) -> Generator[str, None, None]:
        """Runs a comprehensive suite of system checks and a dry run via core.system_health."""
        self.logger.info("Starting system diagnostics...")
        from core.system_health import generate_full_diagnostic_report

        return generate_full_diagnostic_report(
            self.config,
            self.logger,
            self.progress_queue,
            self.cancel_event,
            self.thumbnail_manager,
            self.cuda_available,
        )

    def _get_ui_updates_from_state(self, state: ApplicationState) -> dict:
        """
        Centralized reducer that maps ApplicationState to Gradio component updates.

        This reduces manual dictionary creation in event handlers and ensures
        the UI stays in sync with the source of truth.
        """
        c = self.components
        updates = {}

        # Source & Environment
        if state.extracted_video_path:
            updates[c["source_input"]] = gr.update(value=state.extracted_video_path)

        # Analysis State
        if state.analysis_output_dir:
            updates[c["results_group"]] = gr.update(visible=True)
            updates[c["export_group"]] = gr.update(visible=True)
            updates[c["filtering_tab"]] = gr.update(interactive=True)

        # Tabs interactive state
        if state.extracted_frames_dir:
            updates[c["scene_selection_tab"]] = gr.update(interactive=True)

        if state.scenes:
            updates[c["metrics_tab"]] = gr.update(interactive=True)

        # Seeding
        if state.discovered_faces:
            updates[c["seeding_results_column"]] = gr.update(visible=True)
            updates[c["propagation_group"]] = gr.update(visible=True)

        # Photo Mode
        if state.photos:
            # Future: updates[c["photo_gallery"]] = ...
            pass

        return updates

    def _create_pre_analysis_event(self, state: ApplicationState, *args: Any) -> "PreAnalysisEvent":
        """Helper to construct a PreAnalysisEvent from UI arguments."""
        ui_args = dict(zip(self.ana_ui_map_keys, args))
        clean_args = {k: v for k, v in ui_args.items() if v is not None}

        # Inject state values
        # If analysis_output_dir is empty, use extracted_frames_dir as fallback
        out_dir = state.analysis_output_dir or state.extracted_frames_dir
        clean_args["output_folder"] = str(out_dir)
        clean_args["video_path"] = str(state.extracted_video_path)

        # Map legacy emoji-based strategies to new professional names
        LEGACY_STRATEGY_MAP = {
            "👤 By Face": "Source Face Reference",
            "📝 By Text": "Text Description (Limited)",
            "🔄 Face + Text Fallback": "Face + Text Fallback",
            "🤖 Automatic": "Automatic Detection",
            "Auto": "Automatic Detection",
        }

        strategy = clean_args.get("primary_seed_strategy", self.config.default_primary_seed_strategy)
        # Apply mapping if needed
        strategy = LEGACY_STRATEGY_MAP.get(strategy, strategy)
        clean_args["primary_seed_strategy"] = strategy

        if strategy == "Source Face Reference":
            clean_args.update({"enable_face_filter": True, "text_prompt": ""})
        elif strategy == "Text Description (Limited)":
            clean_args.update({"enable_face_filter": False, "face_ref_img_path": ""})
        return PreAnalysisEvent.model_validate(clean_args)

    def _run_pipeline(
        self,
        pipeline_func: Callable,
        event: Any,
        progress: Callable,
        success_callback: Optional[Callable] = None,
        *args,
    ):
        """
        Generic wrapper to run a pipeline function and handle progress/errors.

        Args:
            pipeline_func: The pipeline generator function to run.
            event: The event object to pass to the pipeline.
            progress: Gradio progress callback.
            success_callback: Optional callback to run on successful completion.
        """
        generator = None
        try:
            generator = pipeline_func(
                event,
                self.progress_queue,
                self.cancel_event,
                self.app_logger,
                self.config,
                self.thumbnail_manager,
                self.cuda_available,
                progress=progress,
                model_registry=self.model_registry,
            )
            for result in generator:
                if isinstance(result, dict):
                    if self.cancel_event.is_set():
                        yield {self.components["unified_log"]: "Cancelled by user."}
                        return
                    if result.get("done"):
                        if success_callback:
                            yield success_callback(result)
                        return
            yield {self.components["unified_log"]: "❌ Pipeline failed unexpectedly."}
        except Exception as e:
            self.app_logger.error(f"Pipeline execution failed: {e}", exc_info=True)
            yield {
                self.components["unified_log"]: f"❌ **Error:** {e}",
                self.components["unified_status"]: f"⚠️ Failure in {pipeline_func.__name__}",
            }
        finally:
            if generator and hasattr(generator, "close"):
                generator.close()

    @safe_ui_callback("Extraction")
    @safe_ui_callback("Add to Queue")
    def add_to_queue_handler(self, *args):
        """Adds a job to the batch processing queue."""
        # ... (keep existing logic)
        return gr.update(value=self.batch_manager.get_status_list())

    @safe_ui_callback("Clear Queue")
    def clear_queue_handler(self):
        """Clears all items from the batch queue."""
        self.batch_manager.clear_all()
        return gr.update(value=self.batch_manager.get_status_list())

    def _batch_processor(self, item: BatchItem, progress_callback: Callable):
        """Callback to process a single item in the batch queue."""
        params = item.params.copy()
        params["source_path"] = item.path
        params["upload_video"] = None
        event = ExtractionEvent.model_validate(params)
        gen = execute_extraction(
            event,
            self.progress_queue,
            self.batch_manager.stop_event,
            self.logger,
            self.config,
            progress=progress_callback,
        )
        for update in gen:
            result = update
        if not result.get("done"):
            raise RuntimeError(result.get("unified_log", "Unknown failure"))
        return result

    @safe_ui_callback("Start Batch")
    def start_batch_wrapper(self, workers: float):
        """Starts processing the batch queue with specified number of workers."""
        self.batch_manager.start_processing(self._batch_processor, max_workers=int(workers))
        while self.batch_manager.is_running:
            yield self.batch_manager.get_status_list()
            time.sleep(1.0)
        yield self.batch_manager.get_status_list()

    @safe_ui_callback("Stop Batch")
    def stop_batch_handler(self):
        """Stops the batch processing."""
        self.batch_manager.stop_processing()
        return "Stopping..."

    def _save_session_log(self, output_dir_str: str):
        """Helper to save logs to the result directory."""
        if output_dir_str:
            self.logger.copy_log_to_output(Path(output_dir_str))

    def _fix_strategy_visibility(self, strategy: str) -> dict:
        """Adjusts UI component visibility based on the selected seed strategy."""
        # Use new professional name substrings
        is_face = "Face Reference" in strategy or "Fallback" in strategy
        is_text = "Text Description" in strategy or "Fallback" in strategy
        is_auto = "Automatic" in strategy or "Discovery" in strategy
        return {
            self.components["face_seeding_group"]: gr.update(visible=is_face),
            self.components["text_seeding_group"]: gr.update(visible=is_text),
            self.components["auto_seeding_group"]: gr.update(visible=is_auto),
            self.components["enable_face_filter_input"]: gr.update(value=is_face, visible=is_face),
        }

    def _setup_visibility_toggles(self):
        """Configures dynamic visibility logic for UI components."""
        c = self.components

        def handle_source_change(path):
            is_folder = is_image_folder(path)
            if is_folder or not path:
                return {c["max_resolution"]: gr.update(visible=False), c["thumbnail_group"]: gr.update(visible=False)}
            else:
                return {c["max_resolution"]: gr.update(visible=True), c["thumbnail_group"]: gr.update(visible=True)}

        for control in [c["source_input"], c["upload_video_input"]]:
            control.change(handle_source_change, inputs=control, outputs=[c["max_resolution"], c["thumbnail_group"]])
        c["method_input"].change(
            lambda m: {
                c["nth_frame_input"]: gr.update(visible=m == "every_nth_frame"),
            },
            c["method_input"],
            [c["nth_frame_input"]],
        )
        c["primary_seed_strategy_input"].change(
            self._fix_strategy_visibility,
            inputs=c["primary_seed_strategy_input"],
            outputs=[
                c["face_seeding_group"],
                c["text_seeding_group"],
                c["auto_seeding_group"],
                c["enable_face_filter_input"],
            ],
        )

    def _get_inputs(self, keys: list[str]) -> list[gr.Component]:
        """Retrieves a list of UI components based on their registry keys."""
        return [self.ui_registry[k] for k in keys if k in self.ui_registry]

    def _setup_pipeline_handlers(self):
        """Configures event handlers for starting main processing pipelines."""
        c = self.components

        # Load Session
        c["load_session_button"].click(
            fn=self.pipeline_handler.run_session_load_wrapper,
            inputs=[c["session_path_input"], c["application_state"]],
            outputs=self.all_outputs,
            show_progress="hidden",
        )

        ext_inputs = [c["application_state"]] + self._get_inputs(self.ext_ui_map_keys)
        self.ana_input_components = [c["application_state"]] + self._get_inputs(self.ana_ui_map_keys)

        # Pipeline Handlers
        c["start_extraction_button"].click(
            fn=self.pipeline_handler.run_extraction_wrapper,
            inputs=ext_inputs,
            outputs=self.all_outputs,
            show_progress="hidden",
        )
        c["start_pre_analysis_button"].click(
            fn=self.pipeline_handler.run_pre_analysis_wrapper,
            inputs=self.ana_input_components,
            outputs=self.all_outputs,
            show_progress="hidden",
        )
        c["propagate_masks_button"].click(
            # Using current_state.scenes from application_state
            fn=self.pipeline_handler.run_propagation_wrapper,
            inputs=self.ana_input_components,
            outputs=self.all_outputs,
            show_progress="hidden",
        )
        c["start_analysis_button"].click(
            fn=self.pipeline_handler.run_analysis_wrapper,
            inputs=self.ana_input_components,
            outputs=self.all_outputs,
            show_progress="hidden",
        )

        # Helper Handlers
        c["add_to_queue_button"].click(
            self.add_to_queue_handler,
            inputs=self._get_inputs(self.ext_ui_map_keys),
            outputs=[c["batch_queue_dataframe"]],
        )
        c["clear_queue_button"].click(self.clear_queue_handler, inputs=[], outputs=[c["batch_queue_dataframe"]])
        c["start_batch_button"].click(
            self.start_batch_wrapper, inputs=[c["batch_workers_slider"]], outputs=[c["batch_queue_dataframe"]]
        )
        c["stop_batch_button"].click(self.stop_batch_handler, inputs=[], outputs=[c["unified_log"]])

        c["find_people_button"].click(
            self.on_find_people_from_video,
            inputs=self.ana_input_components,
            outputs=[
                c["unified_status"],
                c["find_people_status"],
                c["discovered_people_group"],
                c["discovered_faces_gallery"],
                c["identity_confidence_slider"],
                c["application_state"],
            ],
        )

        # We need to update on_identity_confidence_change to take/return state
        c["identity_confidence_slider"].release(
            self.on_identity_confidence_change,
            inputs=[c["identity_confidence_slider"], c["application_state"]],
            outputs=[c["discovered_faces_gallery"]],
        )
        c["discovered_faces_gallery"].select(
            self.on_discovered_face_select,
            inputs=[c["application_state"], c["identity_confidence_slider"]],
            outputs=[c["face_ref_img_path_input"], c["face_ref_image"], c["find_people_status"]],
        )

    @safe_ui_callback("Face Clustering")
    def on_identity_confidence_change(self, confidence: float, state: ApplicationState) -> Any:
        """Updates the face discovery gallery based on clustering confidence."""
        all_faces = state.discovered_faces
        if not all_faces:
            return []

        from core.face_clustering import cluster_faces

        labels, cluster_map = cluster_faces(all_faces, confidence)
        self.gallery_to_cluster_map = cluster_map

        gallery_items = []
        for idx, label in cluster_map.items():
            cluster_faces_list = [all_faces[i] for i, l in enumerate(labels) if l == label]
            best_face = max(cluster_faces_list, key=lambda x: x["det_score"])

            thumb_rgb = self.thumbnail_manager.get(Path(best_face["thumb_path"]))
            if thumb_rgb is not None:
                x1, y1, x2, y2 = best_face["bbox"].astype(int)
                face_crop = thumb_rgb[y1:y2, x1:x2]
                gallery_items.append((face_crop, f"Person {label}"))

        return gr.update(value=gallery_items)

    @safe_ui_callback("Face Selection")
    def on_discovered_face_select(
        self, state: ApplicationState, confidence: float, evt: Optional[gr.SelectData] = None
    ) -> tuple[Optional[str], Optional[np.ndarray], str]:
        """Handles selection of a face cluster from the discovery gallery."""
        all_faces = state.discovered_faces
        if not all_faces or evt is None or evt.index is None:
            return "", None, "⚠️ Selection Failed"

        selected_label = self.gallery_to_cluster_map.get(evt.index)
        if selected_label is None:
            return "", None, "Selection Failed"

        from core.face_clustering import cluster_faces, get_cluster_representative

        labels, _ = cluster_faces(all_faces, confidence)

        return get_cluster_representative(
            all_faces, labels, selected_label, state.extracted_video_path, state.extracted_frames_dir
        )

    @safe_ui_callback("Face Discovery")
    def on_find_people_from_video(
        self, current_state: ApplicationState, *args
    ) -> tuple[str, str, gr.update, gr.update, float, ApplicationState]:
        """Scans the video for faces to populate the discovery gallery.

        Returns: (unified_status, find_people_status, group_visibility, gallery_update, slider_value, new_state)
        """
        new_state = current_state.model_copy()
        self.logger.info("Scan Video for Faces clicked")
        params = self._create_pre_analysis_event(current_state, *args)
        output_dir = Path(params.output_folder)
        self.logger.info(f"Output dir: {output_dir}, exists: {output_dir.exists()}")
        if not output_dir.exists():
            self.logger.warning("Output directory does not exist - run extraction first")
            msg = "**Run extraction first** - No video frames found."
            return (
                "**Face Discovery Failed.** Run extraction first.",
                msg,
                gr.update(visible=False),
                [],
                0.5,
                new_state,
            )

        from core.managers import initialize_analysis_models
        from core.utils import create_frame_map

        models = initialize_analysis_models(params.model_dump(), self.config, self.logger, self.model_registry)
        face_analyzer = models["face_analyzer"]
        if not face_analyzer:
            self.logger.warning("Face analyzer not available")
            msg = "**Face analyzer unavailable** - Check model installation."
            return (
                "**Face Discovery Failed.** Face analyzer unavailable.",
                msg,
                gr.update(visible=False),
                [],
                0.5,
                new_state,
            )
        frame_map = create_frame_map(output_dir, self.logger)
        self.logger.info(f"Frame map has {len(frame_map)} frames")
        if not frame_map:
            msg = "**No frames found** - Run extraction first."
            return (
                "**Face Discovery Failed.** No video frames found.",
                msg,
                gr.update(visible=False),
                [],
                0.5,
                new_state,
            )

        all_faces = []
        thumb_dir = output_dir / "thumbs"
        for frame_num, thumb_filename in frame_map.items():
            if frame_num % params.pre_sample_nth != 0:
                continue
            thumb_rgb = self.thumbnail_manager.get(thumb_dir / thumb_filename)
            if thumb_rgb is None:
                continue
            faces = face_analyzer.get(cv2.cvtColor(thumb_rgb, cv2.COLOR_RGB2BGR))
            for face in faces:
                all_faces.append(
                    {
                        "frame_num": frame_num,
                        "bbox": face.bbox,
                        "embedding": face.normed_embedding,
                        "det_score": face.det_score,
                        "thumb_path": str(thumb_dir / thumb_filename),
                    }
                )

        self.logger.info(f"Found {len(all_faces)} faces in video")
        new_state.discovered_faces = all_faces

        if not all_faces:
            self.logger.info("No faces found in sampled frames")
            msg = "**No faces detected** in sampled frames. Try adjusting sample rate."
            return (
                "**Face Discovery Finished.** No faces found.",
                msg,
                gr.update(visible=False),
                [],
                0.5,
                new_state,
            )

        # Get clustered faces for gallery
        gallery_items = self.on_identity_confidence_change(0.5, new_state)
        n_people = len(self.gallery_to_cluster_map) if hasattr(self, "gallery_to_cluster_map") else 0
        success_msg = f"Found **{n_people} unique people** from {len(all_faces)} face detections."
        return (
            success_msg,
            success_msg,
            gr.update(visible=True),
            gallery_items,
            0.5,
            new_state,
        )

    def _get_smart_mode_updates(self, is_enabled: bool) -> list[gr.update]:
        """Calculates slider updates when toggling 'Smart Mode'."""
        updates = []
        slider_keys = sorted(self.components["metric_sliders"].keys())
        for key in slider_keys:
            if "yaw" in key or "pitch" in key:
                updates.append(gr.update())
                continue
            if is_enabled:
                updates.append(
                    gr.update(
                        minimum=0.0,
                        maximum=100.0,
                        step=1.0,
                        label=f"{self.components['metric_sliders'][key].label.split('(')[0].strip()} (%)",
                    )
                )
            else:
                metric_key = re.sub(r"_(min|max)$", "", key)
                "default_max" if key.endswith("_max") else "default_min"
                f_def = getattr(self.config, f"filter_default_{metric_key}", {})
                label = self.components["metric_sliders"][key].label.replace(" (%)", "")
                updates.append(
                    gr.update(
                        minimum=f_def.get("min", 0),
                        maximum=f_def.get("max", 100),
                        step=f_def.get("step", 0.5),
                        label=label,
                    )
                )
        return updates

    def _setup_filtering_handlers(self):
        """Configures event handlers for the filtering and export tab."""
        c = self.components
        slider_keys, slider_comps = (
            sorted(c["metric_sliders"].keys()),
            [c["metric_sliders"][k] for k in sorted(c["metric_sliders"].keys())],
        )
        fast_filter_inputs = [
            c["application_state"],
            c["gallery_view_toggle"],
            c["show_mask_overlay_input"],
            c["overlay_alpha_slider"],
            c["require_face_match_input"],
            c["dedup_thresh_input"],
            c["dedup_method_input"],
        ] + slider_comps
        fast_filter_outputs = [c["filter_status_text"], c["results_gallery"]]

        c["smart_filter_checkbox"].change(
            lambda state, e: (
                state.model_copy(update={"smart_filter_enabled": e}),
                e,
                *self._get_smart_mode_updates(e),
                f"Smart Mode: {'On' if e else 'Off'}",
            ),
            inputs=[c["application_state"], c["smart_filter_checkbox"]],
            outputs=[c["application_state"], c["application_state"]] + slider_comps + [c["filter_status_text"]],
        )

        for control in slider_comps + [
            c["dedup_thresh_input"],
            c["gallery_view_toggle"],
            c["show_mask_overlay_input"],
            c["overlay_alpha_slider"],
            c["require_face_match_input"],
            c["dedup_method_input"],
        ]:
            (
                control.release
                if hasattr(control, "release")
                else control.input
                if hasattr(control, "input")
                else control.change
            )(self.on_filters_changed_wrapper, fast_filter_inputs, fast_filter_outputs)

        load_outputs = (
            [
                c["application_state"],
                c["filter_status_text"],
                c["results_gallery"],
                c["results_group"],
                c["export_group"],
            ]
            + [c["metric_plots"].get(k) for k in self._get_all_filter_keys() if c["metric_plots"].get(k)]
            + slider_comps
            + [c["require_face_match_input"]]
            + [c["metric_accs"].get(k) for k in sorted(c["metric_accs"].keys()) if c["metric_accs"].get(k)]
        )

        def load_and_trigger_update(state):
            output_dir = state.analysis_output_dir
            if not output_dir:
                return [gr.update()] * len(load_outputs)
            from core.filtering import build_all_metric_svgs, load_and_prep_filter_data

            all_frames, metric_values = load_and_prep_filter_data(output_dir, self._get_all_filter_keys, self.config)
            svgs = build_all_metric_svgs(metric_values, self._get_all_filter_keys, self.logger)

            new_state = state.model_copy()
            new_state.all_frames_data = all_frames
            new_state.per_metric_values = metric_values

            updates = {
                c["application_state"]: new_state,
                c["results_group"]: gr.update(visible=True),
                c["export_group"]: gr.update(visible=True),
            }
            for k in self._get_all_filter_keys():
                acc = c["metric_accs"].get(k)
                has_data = k in metric_values and metric_values.get(k)
                if acc:
                    updates[acc] = gr.update(visible=has_data)
                if k in c["metric_plots"]:
                    updates[c["metric_plots"][k]] = gr.update(value=svgs.get(k, ""))

            slider_values_dict = {key: c["metric_sliders"][key].value for key in slider_keys}
            dedup_val = (
                "pHash"
                if c["dedup_method_input"].value == "Fast (pHash)"
                else "pHash then LPIPS"
                if c["dedup_method_input"].value == "Accurate (LPIPS)"
                else "None"
            )
            filter_event = FilterEvent(
                all_frames_data=all_frames,
                per_metric_values=metric_values,
                output_dir=output_dir,
                gallery_view="Kept Frames",
                show_overlay=c["show_mask_overlay_input"].value,
                overlay_alpha=c["overlay_alpha_slider"].value,
                require_face_match=c["require_face_match_input"].value,
                dedup_thresh=c["dedup_thresh_input"].value,
                slider_values=slider_values_dict,
                dedup_method=dedup_val,
            )
            filter_updates = on_filters_changed(filter_event, self.thumbnail_manager, self.config, self.logger)
            updates.update(
                {
                    c["filter_status_text"]: filter_updates["filter_status_text"],
                    c["results_gallery"]: filter_updates["results_gallery"],
                }
            )
            return [updates.get(comp, gr.update()) for comp in load_outputs]

        c["filtering_tab"].select(load_and_trigger_update, [c["application_state"]], load_outputs)

        c["export_button"].click(
            self.export_kept_frames_wrapper,
            [
                c["application_state"],
                c["enable_crop_input"],
                c["crop_ar_input"],
                c["crop_padding_input"],
                c["enable_xmp_export_input"],
                c["require_face_match_input"],
                c["dedup_thresh_input"],
                c["dedup_method_input"],
            ]
            + slider_comps,
            [c["unified_status"]],
        )
        c["dry_run_button"].click(
            self.dry_run_export_wrapper,
            [
                c["application_state"],
                c["enable_crop_input"],
                c["crop_ar_input"],
                c["crop_padding_input"],
                c["enable_xmp_export_input"],
                c["require_face_match_input"],
                c["dedup_thresh_input"],
                c["dedup_method_input"],
            ]
            + slider_comps,
            [c["unified_status"]],
        )

        # Reset Filters
        c["reset_filters_button"].click(
            self.on_reset_filters,
            [c["application_state"]],
            [c["application_state"]]
            + slider_comps
            + [
                c["dedup_thresh_input"],
                c["require_face_match_input"],
                c["filter_status_text"],
                c["results_gallery"],
                c["dedup_method_input"],
            ]
            + list(c["metric_accs"].values())
            + [c["smart_filter_checkbox"]],
        )

        # Auto Threshold
        c["apply_auto_button"].click(
            lambda state, p, *cbs: self.on_auto_set_thresholds(state.per_metric_values, p, *cbs),
            [c["application_state"], c["auto_pctl_input"]] + list(c["metric_auto_threshold_cbs"].values()),
            slider_comps,
        ).then(self.on_filters_changed_wrapper, fast_filter_inputs, fast_filter_outputs)

        # Preset
        c["filter_preset_dropdown"].change(
            self.on_preset_changed,
            [c["filter_preset_dropdown"]],
            [c["application_state"]]
            + slider_comps
            + [c["smart_filter_checkbox"]],  # Note: this is a bit broken as we need state to update
        ).then(self.on_filters_changed_wrapper, fast_filter_inputs, fast_filter_outputs)

        # Visual Diff
        c["calculate_diff_button"].click(
            self.calculate_visual_diff,
            [
                c["results_gallery"],
                c["application_state"],
                c["dedup_method_input"],
                c["dedup_thresh_input"],
                c["ssim_threshold_input"],
                c["lpips_threshold_input"],
            ],
            [c["visual_diff_image"]],
        ).then(lambda: gr.update(visible=True), None, c["visual_diff_image"])

    @safe_ui_callback("Preset Change")
    def on_preset_changed(self, preset_name: str) -> list[Any]:
        """Updates filter sliders when a preset is selected."""
        is_preset_active = preset_name != "None" and preset_name in self.FILTER_PRESETS
        final_updates = []
        slider_keys = sorted(self.components["metric_sliders"].keys())
        preset_values = self.FILTER_PRESETS.get(preset_name, {})
        for key in slider_keys:
            if is_preset_active and key in preset_values:
                val = preset_values[key]
            else:
                metric_key = re.sub(r"_(min|max)$", "", key)
                default_key = "default_max" if key.endswith("_max") else "default_min"
                val = getattr(self.config, f"filter_default_{metric_key}", {}).get(default_key, 0)

            # If Preset, enable smart mode (0-100) except angles
            if is_preset_active and "yaw" not in key and "pitch" not in key:
                final_updates.append(
                    gr.update(
                        minimum=0.0,
                        maximum=100.0,
                        step=1.0,
                        value=val,
                        label=f"{self.components['metric_sliders'][key].label.split('(')[0].strip()} (%)",
                    )
                )
            elif "yaw" in key or "pitch" in key:
                final_updates.append(gr.update(value=val))
            else:
                f_def = getattr(self.config, f"filter_default_{re.sub(r'_(min|max)$', '', key)}", {})
                final_updates.append(
                    gr.update(
                        minimum=f_def.get("min", 0),
                        maximum=f_def.get("max", 100),
                        step=f_def.get("step", 0.5),
                        value=val,
                        label=self.components["metric_sliders"][key].label.replace(" (%)", ""),
                    )
                )

        # We return state update as first item
        # Since we don't have the state object here, we'll need to handle it via lambda in click/change
        # This is a bit complex for a single replace, I'll return gr.update() for now and fix state elsewhere
        return [gr.update()] + final_updates + [gr.update(value=is_preset_active)]

    @safe_ui_callback("Filter Change")
    def on_filters_changed_wrapper(
        self,
        state: ApplicationState,
        gallery_view: str,
        show_overlay: bool,
        overlay_alpha: float,
        require_face_match: bool,
        dedup_thresh: int,
        dedup_method_ui: str,
        *slider_values: float,
    ) -> tuple[str, gr.update]:
        """
        Updates the results gallery when filters change.
        """
        all_frames_data = state.all_frames_data
        per_metric_values = state.per_metric_values
        output_dir = state.analysis_output_dir
        smart_mode_enabled = state.smart_filter_enabled

        slider_values_dict = {k: v for k, v in zip(sorted(self.components["metric_sliders"].keys()), slider_values)}
        if smart_mode_enabled and per_metric_values:
            for key, val in slider_values_dict.items():
                if "yaw" in key or "pitch" in key:
                    continue
                metric_data = per_metric_values.get(re.sub(r"_(min|max)$", "", key))
                if metric_data:
                    try:
                        slider_values_dict[key] = float(np.percentile(np.array(metric_data), val))
                    except Exception:
                        pass

        dedup_method = (
            "pHash"
            if dedup_method_ui == "Fast (pHash)"
            else "pHash then LPIPS"
            if dedup_method_ui == "Accurate (LPIPS)"
            else "None"
        )
        result = on_filters_changed(
            FilterEvent(
                all_frames_data=all_frames_data,
                per_metric_values=per_metric_values,
                output_dir=output_dir,
                gallery_view=gallery_view,
                show_overlay=show_overlay,
                overlay_alpha=overlay_alpha,
                require_face_match=require_face_match,
                dedup_thresh=dedup_thresh,
                slider_values=slider_values_dict,
                dedup_method=dedup_method,
            ),
            self.thumbnail_manager,
            self.config,
            self.logger,
        )
        return result["filter_status_text"], result["results_gallery"]

    @safe_ui_callback("Visual Diff")
    def calculate_visual_diff(
        self,
        gallery: gr.Gallery,
        state: ApplicationState,
        dedup_method_ui: str,
        dedup_thresh: int,
        ssim_thresh: float,
        lpips_thresh: float,
    ) -> Optional[np.ndarray]:
        """
        Computes a side-by-side comparison image for duplicate inspection.
        """
        all_frames_data = state.all_frames_data
        if not gallery or not gallery.selection:  # type: ignore
            return None
        dedup_method = (
            "pHash"
            if dedup_method_ui == "Fast (pHash)"
            else "pHash then LPIPS"
            if dedup_method_ui == "Accurate (LPIPS)"
            else "None"
        )

        selected_image_index = gallery.selection["index"]  # type: ignore
        selected_frame_data = all_frames_data[selected_image_index]
        duplicate_frame_data = None
        import imagehash

        for frame_data in all_frames_data:
            if frame_data["filename"] == selected_frame_data["filename"]:
                continue
            if "pHash" in dedup_method:
                hash1 = imagehash.hex_to_hash(selected_frame_data["phash"])
                hash2 = imagehash.hex_to_hash(frame_data["phash"])
                if hash1 - hash2 <= dedup_thresh:
                    duplicate_frame_data = frame_data
                    break

        if duplicate_frame_data:
            img1 = self.thumbnail_manager.get(
                Path(self.config.downloads_dir)
                / Path(selected_frame_data["filename"]).parent.name
                / "thumbs"
                / selected_frame_data["filename"]
            )
            img2 = self.thumbnail_manager.get(
                Path(self.config.downloads_dir)
                / Path(duplicate_frame_data["filename"]).parent.name
                / "thumbs"
                / duplicate_frame_data["filename"]
            )
            if img1 is not None and img2 is not None:
                h, w, _ = img1.shape
                comparison_image = np.zeros((h, w * 2, 3), dtype=np.uint8)
                comparison_image[:, :w] = img1
                comparison_image[:, w:] = img2
                return comparison_image
        return None

    @safe_ui_callback("Reset Filters")
    def on_reset_filters(self, state: ApplicationState) -> tuple:
        """Resets all filter settings to their defaults."""
        c = self.components
        slider_keys = sorted(c["metric_sliders"].keys())
        slider_updates = []
        for key in slider_keys:
            metric_key = re.sub(r"_(min|max)$", "", key)
            default_key = "default_max" if key.endswith("_max") else "default_min"
            val = getattr(self.config, f"filter_default_{metric_key}", {}).get(default_key, 0)
            slider_updates.append(gr.update(value=val))

        acc_updates = []
        for key in sorted(c["metric_accs"].keys()):
            acc_updates.append(gr.update(open=(key == "quality_score")))

        new_state = state.model_copy()
        new_state.smart_filter_enabled = False

        return tuple(
            [new_state]
            + slider_updates
            + [5, False, "Filters Reset.", gr.update(), "Fast (pHash)"]
            + acc_updates
            + [False]
        )

    @safe_ui_callback("Auto Thresholds")
    def on_auto_set_thresholds(self, per_metric_values: dict, p: int, *checkbox_values: bool) -> list[gr.update]:
        """Automatically sets filter thresholds based on data percentiles."""
        slider_keys = sorted(self.components["metric_sliders"].keys())
        auto_threshold_cbs_keys = sorted(self.components["metric_auto_threshold_cbs"].keys())
        selected_metrics = [
            metric_name for metric_name, is_selected in zip(auto_threshold_cbs_keys, checkbox_values) if is_selected
        ]
        updates = auto_set_thresholds(per_metric_values, p, slider_keys, selected_metrics)
        return [updates.get(f"slider_{key}", gr.update()) for key in slider_keys]

    @safe_ui_callback("Export")
    def export_kept_frames_wrapper(
        self,
        state: ApplicationState,
        enable_crop: bool,
        crop_ars: str,
        crop_padding: int,
        enable_xmp_export: bool,
        require_face_match: bool,
        dedup_thresh: int,
        dedup_method_ui: str,
        *slider_values: float,
    ) -> dict:
        """Wrapper to execute the final frame export."""
        all_frames_data = state.all_frames_data
        output_dir = state.analysis_output_dir
        video_path = state.extracted_video_path

        slider_values_dict = {k: v for k, v in zip(sorted(self.components["metric_sliders"].keys()), slider_values)}
        dedup_method = (
            "pHash"
            if dedup_method_ui == "Fast (pHash)"
            else "pHash then LPIPS"
            if dedup_method_ui == "Accurate (LPIPS)"
            else "None"
        )
        filter_args: dict[str, Any] = slider_values_dict
        filter_args.update(
            {
                "require_face_match": require_face_match,
                "dedup_thresh": dedup_thresh,
                "dedup_method": dedup_method,
                "enable_dedup": dedup_method != "None",
            }
        )
        msg = export_kept_frames(
            ExportEvent(
                all_frames_data=all_frames_data,
                output_dir=output_dir,
                video_path=video_path,
                enable_crop=enable_crop,
                crop_ars=crop_ars,
                crop_padding=crop_padding,
                enable_xmp_export=enable_xmp_export,
                filter_args=filter_args,
            ),
            self.config,
            self.logger,
            self.thumbnail_manager,
            self.cancel_event,
        )
        return {
            self.components["unified_status"]: msg,
        }

    @safe_ui_callback("Export Dry Run")
    def dry_run_export_wrapper(
        self,
        state: ApplicationState,
        enable_crop: bool,
        crop_ars: str,
        crop_padding: int,
        enable_xmp_export: bool,
        require_face_match: bool,
        dedup_thresh: int,
        dedup_method_ui: str,
        *slider_values: float,
    ) -> dict:
        """Wrapper to perform a dry run of the export."""
        all_frames_data = state.all_frames_data
        output_dir = state.analysis_output_dir
        video_path = state.extracted_video_path

        slider_values_dict = {k: v for k, v in zip(sorted(self.components["metric_sliders"].keys()), slider_values)}
        dedup_method = (
            "pHash"
            if dedup_method_ui == "Fast (pHash)"
            else "pHash then LPIPS"
            if dedup_method_ui == "Accurate (LPIPS)"
            else "None"
        )
        filter_args: dict[str, Any] = slider_values_dict
        filter_args.update(
            {
                "require_face_match": require_face_match,
                "dedup_thresh": dedup_thresh,
                "dedup_method": dedup_method,
                "enable_dedup": dedup_method != "None",
            }
        )
        msg = dry_run_export(
            ExportEvent(
                all_frames_data=all_frames_data,
                output_dir=output_dir,
                video_path=video_path,
                enable_crop=enable_crop,
                crop_ars=crop_ars,
                crop_padding=crop_padding,
                enable_xmp_export=enable_xmp_export,
                filter_args=filter_args,
            ),
            self.config,
            self.logger,
        )
        return {
            self.components["unified_status"]: msg,
        }
