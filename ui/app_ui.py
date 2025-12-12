from __future__ import annotations
import threading
import time
import sys
import re
from pathlib import Path
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, List, Dict, Any, Callable, Deque, Generator
from collections import deque
import gradio as gr
import torch
import numpy as np
import cv2

from config import Config
from logger import AppLogger
from core.managers import ThumbnailManager, ModelRegistry
from core.models import Scene, SceneState, AnalysisParameters
from core.utils import is_image_folder
from core.scene_utils import (
    toggle_scene_status, save_scene_seeds, _recompute_single_preview,
    _create_analysis_context, _wire_recompute_handler, get_scene_status_text
)
from core.pipelines import (
    execute_extraction, execute_pre_analysis, execute_propagation,
    execute_analysis, execute_session_load, AdvancedProgressTracker
)
from core.export import export_kept_frames, dry_run_export
from ui.gallery_utils import (
    build_scene_gallery_items, on_filters_changed, auto_set_thresholds,
    _update_gallery, scene_caption, create_scene_thumbnail_with_badge
)
from events import ExtractionEvent, PreAnalysisEvent, PropagationEvent, SessionLoadEvent, FilterEvent, ExportEvent
from core.batch_manager import BatchManager, BatchStatus, BatchItem
import uuid

class AppUI:
    MAX_RESOLUTION_CHOICES: List[str] = ["maximum available", "2160", "1080", "720"]
    EXTRACTION_METHOD_TOGGLE_CHOICES: List[str] = ["Recommended Thumbnails", "Legacy Full-Frame"]
    METHOD_CHOICES: List[str] = ["keyframes", "interval", "every_nth_frame", "nth_plus_keyframes", "all"]
    PRIMARY_SEED_STRATEGY_CHOICES: List[str] = ["🤖 Automatic", "👤 By Face", "📝 By Text", "🔄 Face + Text Fallback", "🧑‍🤝‍🧑 Find Prominent Person"]
    SEED_STRATEGY_CHOICES: List[str] = ["Largest Person", "Center-most Person", "Highest Confidence", "Tallest Person", "Area x Confidence", "Rule-of-Thirds", "Edge-avoiding", "Balanced", "Best Face"]
    PERSON_DETECTOR_MODEL_CHOICES: List[str] = ['yolo11x.pt', 'yolo11s.pt']
    FACE_MODEL_NAME_CHOICES: List[str] = ["buffalo_l", "buffalo_s"]
    TRACKER_MODEL_CHOICES: List[str] = ["sam3"]  # SAM3 model
    GALLERY_VIEW_CHOICES: List[str] = ["Kept", "Rejected"]
    LOG_LEVEL_CHOICES: List[str] = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'SUCCESS', 'CRITICAL']
    SCENE_GALLERY_VIEW_CHOICES: List[str] = ["Kept", "Rejected", "All"]
    FILTER_PRESETS: Dict[str, Dict[str, float]] = {
        "Sharp Portraits": {"sharpness_min": 60.0, "sharpness_max": 100.0, "edge_strength_min": 50.0, "edge_strength_max": 100.0, "face_sim_min": 0.5, "mask_area_pct_min": 10.0, "eyes_open_min": 0.8, "yaw_min": -15.0, "yaw_max": 15.0, "pitch_min": -15.0, "pitch_max": 15.0},
        "Close-up Subject": {"mask_area_pct_min": 25.0, "mask_area_pct_max": 100.0, "quality_score_min": 50.0},
        "High Naturalness": {"niqe_min": 0.0, "niqe_max": 40.0, "contrast_min": 20.0, "contrast_max": 80.0, "brightness_min": 30.0, "brightness_max": 70.0}
    }

    def __init__(self, config: 'Config', logger: 'AppLogger', progress_queue: Queue, cancel_event: threading.Event, thumbnail_manager: 'ThumbnailManager', model_registry: 'ModelRegistry'):
        self.config = config
        self.logger = logger
        self.app_logger = logger
        self.progress_queue = progress_queue
        self.cancel_event = cancel_event
        self.thumbnail_manager = thumbnail_manager
        self.model_registry = model_registry
        self.batch_manager = BatchManager()
        self.components, self.cuda_available = {}, torch.cuda.is_available()
        self.ui_registry = {}
        self.performance_metrics, self.log_filter_level, self.all_logs = {}, "INFO", []
        self.last_run_args = None
        self.ext_ui_map_keys = ['source_path', 'upload_video', 'method', 'interval', 'nth_frame', 'max_resolution', 'thumb_megapixels', 'scene_detect']
        self.ana_ui_map_keys = ['output_folder', 'video_path', 'resume', 'enable_face_filter', 'face_ref_img_path', 'face_ref_img_upload', 'face_model_name', 'enable_subject_mask', 'tracker_model_name', 'best_frame_strategy', 'scene_detect', 'text_prompt', 'min_mask_area_pct', 'sharpness_base_scale', 'edge_strength_base_scale', 'pre_analysis_enabled', 'pre_sample_nth', 'primary_seed_strategy', 'compute_quality_score', 'compute_sharpness', 'compute_edge_strength', 'compute_contrast', 'compute_brightness', 'compute_entropy', 'compute_eyes_open', 'compute_yaw', 'compute_pitch', 'compute_face_sim', 'compute_subject_mask_area', 'compute_niqe', 'compute_phash']
        self.session_load_keys = ['unified_log', 'unified_status', 'progress_details', 'cancel_button', 'pause_button', 'source_input', 'max_resolution', 'thumb_megapixels_input', 'ext_scene_detect_input', 'method_input', 'pre_analysis_enabled_input', 'pre_sample_nth_input', 'enable_face_filter_input', 'face_ref_img_path_input', 'text_prompt_input', 'best_frame_strategy_input', 'tracker_model_name_input', 'extracted_video_path_state', 'extracted_frames_dir_state', 'analysis_output_dir_state', 'analysis_metadata_path_state', 'scenes_state', 'propagate_masks_button', 'seeding_results_column', 'propagation_group', 'scene_filter_status', 'scene_face_sim_min_input', 'filtering_tab', 'scene_gallery', 'scene_gallery_index_map_state']

        # Undo/Redo History
        self.history_depth = 10

    def preload_models(self):
        """Asynchronously preloads heavy models."""
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
                    config=self.config
                )
                self.progress_queue.put({"ui_update": {self.components['model_status_indicator']: "🟢 All Models Ready"}})
                self.logger.success("Async model preloading complete.")
            except Exception as e:
                self.logger.error(f"Async model preloading failed: {e}")
                self.progress_queue.put({"ui_update": {self.components['model_status_indicator']: "🔴 Model Load Failed"}})

        threading.Thread(target=_load, daemon=True).start()

    def build_ui(self) -> gr.Blocks:
        # css argument is deprecated in Gradio 5+
        css = """.gradio-gallery { overflow-y: hidden !important; } .gradio-gallery img { width: 100%; height: 100%; object-fit: scale-down; object-position: top left; } .plot-and-slider-column { max-width: 560px !important; margin: auto; } .scene-editor { border: 1px solid #444; padding: 10px; border-radius: 5px; } .log-container > .gr-utils-error { display: none !important; } .progress-details { font-size: 1rem !important; color: #333 !important; font-weight: 500; padding: 8px 0; } .gr-progress .progress { height: 28px !important; }"""
        with gr.Blocks() as demo:
            self._build_header()
            with gr.Accordion("🔄 resume previous Session", open=False):
                with gr.Row():
                    self._create_component('session_path_input', 'textbox', {'label': "Load previous run", 'placeholder': "Path to a previous run's output folder..."})
                    self._create_component('load_session_button', 'button', {'value': "📂 Load Session"})
                    self._create_component('save_config_button', 'button', {'value': "💾 Save Current Config"})
            with gr.Accordion("⚙️ System Diagnostics", open=False):
                self._create_component('run_diagnostics_button', 'button', {'value': "Run System Diagnostics"})
            self._build_main_tabs()
            self._build_footer()
            self._create_event_handlers()

            # Trigger preloading on load
            demo.load(self.preload_models, None, None)

        return demo

    def _get_comp(self, name: str) -> Optional[gr.components.Component]: return self.components.get(name)
    def _reg(self, key: str, component: gr.components.Component) -> gr.components.Component: self.ui_registry[key] = component; return component
    def _create_component(self, name: str, comp_type: str, kwargs: dict) -> gr.components.Component:
        comp_map = {'button': gr.Button, 'textbox': gr.Textbox, 'dropdown': gr.Dropdown, 'slider': gr.Slider, 'checkbox': gr.Checkbox, 'file': gr.File, 'radio': gr.Radio, 'gallery': gr.Gallery, 'plot': gr.Plot, 'markdown': gr.Markdown, 'html': gr.HTML, 'number': gr.Number, 'cbg': gr.CheckboxGroup, 'image': gr.Image, 'dataframe': gr.Dataframe}
        self.components[name] = comp_map[comp_type](**kwargs)
        return self.components[name]

    def _build_header(self):
        with gr.Row():
            with gr.Column(scale=3):
                gr.Markdown("# 🎬 Frame Extractor & Analyzer v2.0")
            with gr.Column(scale=1):
                self._create_component('model_status_indicator', 'markdown', {'value': "🟡 Loading Models..."})
        status_color = "🟢" if self.cuda_available else "🟡"
        status_text = "GPU Accelerated" if self.cuda_available else "CPU Mode (Slower)"
        gr.Markdown(f"{status_color} **{status_text}**")
        if not self.cuda_available: gr.Markdown("⚠️ **CPU Mode** — GPU-dependent features are disabled or will be slow.")

    def _build_main_tabs(self):
        with gr.Tabs() as main_tabs:
            self.components['main_tabs'] = main_tabs
            with gr.Tab("📹 1. Frame Extraction", id=0): self._create_extraction_tab()
            with gr.Tab("👩🏼‍🦰 2. Define Subject", id=1) as define_subject_tab: self.components['define_subject_tab'] = define_subject_tab; self._create_define_subject_tab()
            with gr.Tab("🎞️ 3. Scene Selection", id=2) as scene_selection_tab: self.components['scene_selection_tab'] = scene_selection_tab; self._create_scene_selection_tab()
            with gr.Tab("📝 4. Metrics", id=3) as metrics_tab: self.components['metrics_tab'] = metrics_tab; self._create_metrics_tab()
            with gr.Tab("📊 5. Filtering & Export", id=4) as filtering_tab: self.components['filtering_tab'] = filtering_tab; self._create_filtering_tab()

    def _build_footer(self):
        with gr.Row():
            with gr.Column(scale=2):
                self._create_component('unified_status', 'markdown', {'label': "📊 Status & Messages", 'value': "Welcome! Ready to start."})
                self.components['progress_bar'] = gr.Progress()
                self._create_component('progress_details', 'html', {'value': '', 'elem_classes': ['progress-details']})
                with gr.Row():
                    self._create_component('pause_button', 'button', {'value': '⏸️ Pause', 'interactive': False})
                    self._create_component('cancel_button', 'button', {'value': '⏹️ Cancel', 'interactive': False})
            with gr.Column(scale=3):
                with gr.Accordion("📋 Verbose Processing Log (for debugging)", open=False):
                    self._create_component('unified_log', 'textbox', {'lines': 15, 'interactive': False, 'autoscroll': True, 'elem_classes': ['log-container'], 'elem_id': 'unified_log'})
                    with gr.Row():
                        self._create_component('log_level_filter', 'dropdown', {'choices': self.LOG_LEVEL_CHOICES, 'value': 'INFO', 'label': 'Log Level', 'scale': 1})
                        self._create_component('clear_logs_button', 'button', {'value': '🗑️ Clear', 'scale': 1})
                        self._create_component('export_logs_button', 'button', {'value': '📥 Export', 'scale': 1})

    def _create_extraction_tab(self):
        gr.Markdown("### Step 1: Provide a Video Source")
        with gr.Row():
            with gr.Column(scale=2): self._reg('source_path', self._create_component('source_input', 'textbox', {'label': "Video URL or Local Path", 'placeholder': "Enter YouTube URL or local video file path (or folder of videos)", 'info': "The application can download videos directly from YouTube or use a video file you have on your computer."}))
            with gr.Column(scale=1): self._reg('max_resolution', self._create_component('max_resolution', 'dropdown', {'choices': self.MAX_RESOLUTION_CHOICES, 'value': self.config.default_max_resolution, 'label': "Max Download Resolution", 'info': "For YouTube videos, select the maximum resolution to download. 'Maximum available' will get the best quality possible."}))
        self._reg('upload_video', self._create_component('upload_video_input', 'file', {'label': "Or Upload Video File(s)", 'file_count': "multiple", 'file_types': ["video"], 'type': "filepath"}))
        gr.Markdown("---"); gr.Markdown("### Step 2: Configure Extraction Method")
        with gr.Group(visible=True) as thumbnail_group:
            self.components['thumbnail_group'] = thumbnail_group
            gr.Markdown("**Thumbnail Extraction:** This is the fastest and most efficient way to process your video. It quickly extracts low-resolution, lightweight thumbnails for every frame. This allows you to perform scene analysis, find the best shots, and select your desired frames *before* extracting the final, full-resolution images. This workflow saves significant time and disk space.")
            with gr.Accordion("Advanced Settings", open=False):
                self._reg('thumb_megapixels', self._create_component('thumb_megapixels_input', 'slider', {'label': "Thumbnail Size (MP)", 'minimum': 0.1, 'maximum': 2.0, 'step': 0.1, 'value': self.config.default_thumb_megapixels, 'info': "Controls the resolution of the extracted thumbnails. Higher values create larger, more detailed thumbnails but increase extraction time and disk usage. 0.5 MP is a good balance for most videos."}))
                self._reg('scene_detect', self._create_component('ext_scene_detect_input', 'checkbox', {'label': "Use Scene Detection", 'value': self.config.default_scene_detect, 'info': "Automatically detects scene changes in the video. This is highly recommended as it groups frames into logical shots, making it much easier to find the best content in the next step."}))
                self._reg('method', self._create_component('method_input', 'dropdown', {'choices': self.METHOD_CHOICES, 'value': self.config.default_method, 'label': "Frame Selection Method", 'info': "- **Keyframes:** Extracts only the keyframes (I-frames). Good for a quick summary.\n- **Interval:** Extracts one frame every X seconds.\n- **Every Nth Frame:** Extracts one frame every N decoded frames.\n- **Nth + Keyframes:** Keeps keyframes plus frames at a regular cadence.\n- **All:** Extracts every single frame. (Warning: massive disk usage and time)."}))
                self._reg('interval', self._create_component('interval_input', 'number', {'label': "Interval (seconds)", 'value': self.config.default_interval, 'minimum': 0.1, 'step': 0.1, 'visible': self.config.default_method == 'interval'}))
                self._reg('nth_frame', self._create_component('nth_frame_input', 'number', {'label': "N-th Frame Value", 'value': self.config.default_nth_frame, 'minimum': 1, 'step': 1, 'visible': self.config.default_method in ['every_nth_frame', 'nth_plus_keyframes']}))
        gr.Markdown("---"); gr.Markdown("### Step 3: Start Processing")
        with gr.Row():
             self.components['start_extraction_button'] = gr.Button("🚀 Start Single Extraction", variant="secondary")
             self._create_component('add_to_queue_button', 'button', {'value': "➕ Add to Batch Queue", 'variant': 'primary'})

        with gr.Accordion("📚 Batch Processing Queue", open=True) as batch_accordion:
             self.components['batch_accordion'] = batch_accordion
             self._create_component('batch_queue_dataframe', 'dataframe', {'headers': ["Path", "Status", "Progress", "Message"], 'datatype': ["str", "str", "number", "str"], 'interactive': False, 'value': []})
             with gr.Row():
                 self._create_component('start_batch_button', 'button', {'value': "▶️ Start Batch Processing", 'variant': "primary"})
                 self._create_component('stop_batch_button', 'button', {'value': "⏹️ Stop Batch", 'variant': "stop"})
                 self._create_component('clear_queue_button', 'button', {'value': "🗑️ Clear Queue"})
             self._create_component('batch_workers_slider', 'slider', {'label': "Max Parallel Workers", 'minimum': 1, 'maximum': 4, 'value': 1, 'step': 1, 'info': "Increase only if you have sufficient memory (CPU/RAM). GPU is usually single-threaded."})

    def _create_define_subject_tab(self):
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 🎯 Step 1: Choose Your Seeding Strategy")
                gr.Markdown("""This step analyzes each scene to find the best frame and automatically detects people using YOLO. The system will: 1. Find the highest quality frame in each scene 2. Detect all people in that frame 3. Select the best subject based on your chosen strategy 4. Generate a preview with the subject highlighted""")
                self._reg('primary_seed_strategy', self._create_component('primary_seed_strategy_input', 'radio', {'choices': self.PRIMARY_SEED_STRATEGY_CHOICES, 'value': self.config.default_primary_seed_strategy, 'label': "Primary Best-Frame Selection Strategy", 'info': "Select the main method for identifying the subject in each scene. This initial identification is called the 'best-frame selection'."}))
                with gr.Group(visible="By Face" in self.config.default_primary_seed_strategy or "Fallback" in self.config.default_primary_seed_strategy) as face_seeding_group:
                    self.components['face_seeding_group'] = face_seeding_group
                    gr.Markdown("#### 👤 Configure Face Selection")
                    gr.Markdown("This strategy prioritizes finding a specific person. Upload a clear, frontal photo of the person you want to track. The system will analyze each scene to find the frame where this person is most clearly visible and use it as the starting point (the 'best frame').")
                    with gr.Row():
                        self._reg('face_ref_img_upload', self._create_component('face_ref_img_upload_input', 'file', {'label': "Upload Face Reference Image", 'type': "filepath"}))
                        self._create_component('face_ref_image', 'image', {'label': "Reference Image", 'interactive': False})
                        with gr.Column():
                            self._reg('face_ref_img_path', self._create_component('face_ref_img_path_input', 'textbox', {'label': "Or provide a local file path"}))
                            self._reg('enable_face_filter', self._create_component('enable_face_filter_input', 'checkbox', {'label': "Enable Face Similarity (must be checked for face selection)", 'value': self.config.default_enable_face_filter, 'interactive': True, 'visible': "By Face" in self.config.default_primary_seed_strategy or "Fallback" in self.config.default_primary_seed_strategy}))
                    self._create_component('find_people_button', 'button', {'value': "Find People From Video"})
                    with gr.Group(visible=False) as discovered_people_group:
                        self.components['discovered_people_group'] = discovered_people_group
                        self._create_component('discovered_faces_gallery', 'gallery', {'label': "Discovered People", 'columns': 8, 'height': 'auto'})
                        self._create_component('identity_confidence_slider', 'slider', {'label': "Identity Confidence", 'minimum': 0.0, 'maximum': 1.0, 'step': 0.05, 'value': 0.5})
                with gr.Group(visible="By Text" in self.config.default_primary_seed_strategy or "Fallback" in self.config.default_primary_seed_strategy) as text_seeding_group:
                    self.components['text_seeding_group'] = text_seeding_group
                    gr.Markdown("#### 📝 Configure Text Selection")
                    gr.Markdown("This strategy uses a text description to find the subject. It's useful for identifying objects, or people described by their clothing or appearance when a reference photo isn't available.")
                    with gr.Accordion("Text Prompt Settings", open=True):
                        gr.Markdown("Use SAM3 for text-based object detection with custom prompts.")
                        self._reg('text_prompt', self._create_component('text_prompt_input', 'textbox', {'label': "Text Prompt", 'placeholder': "e.g., 'a woman in a red dress'", 'value': self.config.default_text_prompt, 'info': "Describe the main subject to find the best frame (e.g., 'player wearing number 10', 'person in the green shirt')."}))
                with gr.Group(visible="Prominent Person" in self.config.default_primary_seed_strategy) as auto_seeding_group:
                    self.components['auto_seeding_group'] = auto_seeding_group
                    gr.Markdown("#### 🧑‍🤝‍🧑 Configure Prominent Person Selection")
                    gr.Markdown("This is a simple, fully automatic mode. It uses SAM3 (with prompt 'person') to find all people in the scene and then selects one based on a simple rule, like who is largest or most central.")
                    self._reg('best_frame_strategy', self._create_component('best_frame_strategy_input', 'dropdown', {'choices': self.SEED_STRATEGY_CHOICES, 'value': "Largest Person", 'label': "Selection Method", 'info': "'Largest' picks the person with the biggest bounding box. 'Center-most' picks the person closest to the center. 'Highest Confidence' selects the person with the highest detection confidence. 'Tallest Person' prefers subjects that are standing. 'Area x Confidence' balances size and confidence. 'Rule-of-Thirds' prefers subjects near the thirds lines. 'Edge-avoiding' avoids subjects near the frame's edge. 'Balanced' provides a good mix of area, confidence, and edge-avoidance. 'Best Face' selects the person with the highest quality face detection."}))
                self._create_component('person_radio', 'radio', {'label': "Select Person", 'choices': [], 'visible': False})
                with gr.Accordion("Advanced Settings", open=False):
                    gr.Markdown("These settings control the underlying models and analysis parameters. Adjust them only if you understand their effect.")
                    self._reg('pre_analysis_enabled', self._create_component('pre_analysis_enabled_input', 'checkbox', {'label': 'Enable Pre-Analysis to find best frame', 'value': self.config.default_pre_analysis_enabled, 'info': "Analyzes a subset of frames in each scene to automatically find the highest quality frame to use as the 'best frame' for masking. Highly recommended."}))
                    self._reg('pre_sample_nth', self._create_component('pre_sample_nth_input', 'number', {'label': 'Sample every Nth thumbnail for pre-analysis', 'value': self.config.default_pre_sample_nth, 'interactive': True, 'info': "For faster pre-analysis, check every Nth frame in a scene instead of all of them. A value of 5 is a good starting point."}))
                    self._reg('face_model_name', self._create_component('face_model_name_input', 'dropdown', {'choices': self.FACE_MODEL_NAME_CHOICES, 'value': self.config.default_face_model_name, 'label': "Face Recognition Model", 'info': "InsightFace model for face matching. 'l' (large) is more accurate; 's' (small) is faster and uses less memory."}))
                    self._reg('tracker_model_name', self._create_component('tracker_model_name_input', 'dropdown', {'choices': self.TRACKER_MODEL_CHOICES, 'value': self.config.default_tracker_model_name, 'label': "Mask Tracking Model", 'info': "The SAM3 model used for tracking the subject mask across frames."}))
                    self._reg('resume', self._create_component('resume_input', 'checkbox', {'label': 'Resume', 'value': self.config.default_resume, 'interactive': True, 'visible': False}))
                    self._reg('enable_subject_mask', self._create_component('enable_subject_mask_input', 'checkbox', {'label': 'Enable Subject Mask', 'value': self.config.default_enable_subject_mask, 'interactive': True, 'visible': False}))
                    self._reg('min_mask_area_pct', self._create_component('min_mask_area_pct_input', 'slider', {'label': 'Min Mask Area Pct', 'value': self.config.default_min_mask_area_pct, 'interactive': True, 'visible': False}))
                    self._reg('sharpness_base_scale', self._create_component('sharpness_base_scale_input', 'slider', {'label': 'Sharpness Base Scale', 'value': self.config.default_sharpness_base_scale, 'minimum': 1, 'maximum': 10000, 'interactive': True, 'visible': False}))
                    self._reg('edge_strength_base_scale', self._create_component('edge_strength_base_scale_input', 'slider', {'label': 'Edge Strength Base Scale', 'value': self.config.default_edge_strength_base_scale, 'minimum': 1, 'maximum': 10000, 'interactive': True, 'visible': False}))
                self._create_component('start_pre_analysis_button', 'button', {'value': '🌱 Find & Preview Best Frames', 'variant': 'primary'})
                with gr.Group(visible=False) as propagation_group: self.components['propagation_group'] = propagation_group

    def _create_scene_selection_tab(self):
        with gr.Column(scale=2, visible=False) as seeding_results_column:
            self.components['seeding_results_column'] = seeding_results_column
            gr.Markdown("""### 🎭 Step 2: Review & Refine Scene Selection\nReview the automatically detected subjects and refine the selection if needed. Each scene shows the best frame with the selected subject highlighted.""")
            with gr.Accordion("Scene Filtering", open=True):
                self._create_component('scene_filter_status', 'markdown', {'value': 'No scenes loaded.'})
                with gr.Row():
                    self._create_component('scene_mask_area_min_input', 'slider', {'label': "Min Best Frame Mask Area %", 'minimum': 0.0, 'maximum': 100.0, 'value': self.config.default_min_mask_area_pct, 'step': 0.1})
                    self._create_component('scene_face_sim_min_input', 'slider', {'label': "Min Best Frame Face Sim", 'minimum': 0.0, 'maximum': 1.0, 'value': 0.0, 'step': 0.05, 'visible': False})
                    self._create_component('scene_confidence_min_input', 'slider', {'label': "Min Best Frame Confidence", 'minimum': 0.0, 'maximum': 1.0, 'value': 0.0, 'step': 0.05})
            with gr.Accordion("Scene Gallery", open=True):
                self._create_component('scene_gallery_view_toggle', 'radio', {'label': "Show", 'choices': ["Kept", "Rejected", "All"], 'value': "Kept"})
                with gr.Row(elem_id="pagination_row"):
                    self._create_component('prev_page_button', 'button', {'value': '⬅️ Previous'})
                    self._create_component('page_number_input', 'number', {'label': 'Page', 'value': 1, 'precision': 0})
                    self._create_component('total_pages_label', 'markdown', {'value': '/ 1 pages'})
                    self._create_component('next_page_button', 'button', {'value': 'Next ➡️'})
                self.components['scene_gallery'] = gr.Gallery(label="Scenes", columns=10, rows=2, height=560, show_label=True, allow_preview=True, container=True)
            with gr.Accordion("Scene Editor", open=False, elem_classes="scene-editor") as sceneeditoraccordion:
                self.components["sceneeditoraccordion"] = sceneeditoraccordion
                self._create_component("sceneeditorstatusmd", "markdown", {"value": "Select a scene to edit."})
                with gr.Group() as yolo_seed_group:
                    self.components['yolo_seed_group'] = yolo_seed_group
                    self._create_component('scene_editor_yolo_subject_id', 'radio', {'label': "Detected Subjects", 'info': "Select the auto-detected subject to use for seeding.", 'interactive': True, 'choices': [], 'visible': False})
                with gr.Accordion("Advanced Seeding (optional)", open=False):
                    gr.Markdown("Use a text prompt for seeding. This will override the automatic detection above.")
                    self._create_component("sceneeditorpromptinput", "textbox", {"label": "Text Prompt", "info": "e.g., 'person in a red shirt'"})
                with gr.Row():
                    self._create_component("scenerecomputebutton", "button", {"value": "▶️ Recompute Preview"})
                    self._create_component("sceneincludebutton", "button", {"value": "✅ Keep Scene"})
                    self._create_component("sceneexcludebutton", "button", {"value": "❌ Reject Scene"})
                    self._create_component("sceneresetbutton", "button", {"value": "🔄 Reset Scene"})
                    self._create_component("sceneundobutton", "button", {"value": "↩️ Undo"}) # Undo button added
            gr.Markdown("---"); gr.Markdown("### 🔬 Step 3: Propagate Masks"); gr.Markdown("Once you are satisfied with the seeds, propagate the masks to the rest of the frames in the selected scenes.")
            self._create_component('propagate_masks_button', 'button', {'value': '🔬 Propagate Masks on Kept Scenes', 'variant': 'primary', 'interactive': False})

    def _create_metrics_tab(self):
        gr.Markdown("### Step 4: Select Metrics to Compute")
        gr.Markdown("Choose which metrics to calculate during the analysis phase. More metrics provide more filtering options but may increase processing time.")
        with gr.Row():
            with gr.Column():
                self._reg('compute_quality_score', self._create_component('compute_quality_score', 'checkbox', {'label': "Quality Score", 'value': True}))
                self._reg('compute_sharpness', self._create_component('compute_sharpness', 'checkbox', {'label': "Sharpness", 'value': True}))
                self._reg('compute_edge_strength', self._create_component('compute_edge_strength', 'checkbox', {'label': "Edge Strength", 'value': True}))
                self._reg('compute_contrast', self._create_component('compute_contrast', 'checkbox', {'label': "Contrast", 'value': True}))
                self._reg('compute_brightness', self._create_component('compute_brightness', 'checkbox', {'label': "Brightness", 'value': True}))
                self._reg('compute_entropy', self._create_component('compute_entropy', 'checkbox', {'label': "Entropy", 'value': True}))
            with gr.Column():
                self._reg('compute_eyes_open', self._create_component('compute_eyes_open', 'checkbox', {'label': "Eyes Open", 'value': True}))
                self._reg('compute_yaw', self._create_component('compute_yaw', 'checkbox', {'label': "Yaw", 'value': True}))
                self._reg('compute_pitch', self._create_component('compute_pitch', 'checkbox', {'label': "Pitch", 'value': True}))
                self._reg('compute_face_sim', self._create_component('compute_face_sim', 'checkbox', {'label': "Face Similarity", 'value': True}))
                self._reg('compute_subject_mask_area', self._create_component('compute_subject_mask_area', 'checkbox', {'label': "Subject Mask Area", 'value': True}))

                import pyiqa
                niqe_avail = pyiqa is not None
                self._reg('compute_niqe', self._create_component('compute_niqe', 'checkbox', {'label': "NIQE", 'value': niqe_avail, 'interactive': niqe_avail, 'info': "Requires 'pyiqa' to be installed."}))
        with gr.Accordion("Deduplication Settings", open=True):
            self._reg('compute_phash', self._create_component('compute_phash', 'checkbox', {'label': "Compute p-hash for Deduplication", 'value': True}))
        self.components['start_analysis_button'] = gr.Button("Analyze Selected Frames", variant="primary")

    def _create_filtering_tab(self):
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 🎛️ Filter Controls")
                gr.Markdown("Use these controls to refine your selection of frames. You can set minimum and maximum thresholds for various quality metrics.")
                self._create_component('filter_preset_dropdown', 'dropdown', {'label': "Filter Presets", 'choices': ["None"] + list(self.FILTER_PRESETS.keys())})
                self._create_component('auto_pctl_input', 'slider', {'label': 'Auto-Threshold Percentile', 'minimum': 1, 'maximum': 99, 'value': self.config.gradio_auto_pctl_input, 'step': 1, 'info': "Quickly set all 'Min' sliders to a certain percentile of the data. For example, setting this to 75 and clicking 'Apply' will automatically reject the bottom 75% of frames for each metric."})
                with gr.Row():
                    self._create_component('apply_auto_button', 'button', {'value': 'Apply Percentile to Mins'})
                    self._create_component('reset_filters_button', 'button', {'value': "Reset Filters"})
                with gr.Row():
                    self._create_component('expand_all_metrics_button', 'button', {'value': 'Expand All'})
                    self._create_component('collapse_all_metrics_button', 'button', {'value': 'Collapse All'})
                self._create_component('filter_status_text', 'markdown', {'value': "Load an analysis to begin."})
                self.components['metric_plots'], self.components['metric_sliders'], self.components['metric_accs'], self.components['metric_auto_threshold_cbs'] = {}, {}, {}, {}
                with gr.Accordion("Deduplication", open=True, visible=True) as dedup_acc:
                    self.components['metric_accs']['dedup'] = dedup_acc
                    f_def = self.config.filter_default_dedup_thresh
                    self._create_component('dedup_method_input', 'dropdown', {'label': "Deduplication Method", 'choices': ["None", "pHash", "SSIM", "LPIPS", "pHash then LPIPS"], 'value': "pHash"})
                    self._create_component('dedup_thresh_input', 'slider', {'label': "pHash Threshold", 'minimum': f_def['min'], 'maximum': f_def['max'], 'value': f_def['default'], 'step': f_def['step'], 'info': "Filters out visually similar frames. A lower value is stricter (more filtering). A value of 0 means only identical images will be removed. Set to -1 to disable."})
                    self._create_component('ssim_threshold_input', 'slider', {'label': "SSIM Threshold", 'minimum': 0.0, 'maximum': 1.0, 'value': 0.95, 'step': 0.01, 'visible': False})
                    self._create_component('lpips_threshold_input', 'slider', {'label': "LPIPS Threshold", 'minimum': 0.0, 'maximum': 1.0, 'value': 0.1, 'step': 0.01, 'visible': False})
                    self._create_component('dedup_visual_diff_input', 'checkbox', {'label': "Enable Visual Diff", 'value': False})
                    self._create_component('visual_diff_image', 'image', {'label': "Visual Diff", 'visible': False})
                    self._create_component('calculate_diff_button', 'button', {'value': "Calculate Diff", 'visible': False})
                metric_configs = {'quality_score': {'open': True}, 'niqe': {'open': False}, 'sharpness': {'open': True}, 'edge_strength': {'open': True}, 'contrast': {'open': True}, 'brightness': {'open': False}, 'entropy': {'open': False}, 'face_sim': {'open': False}, 'mask_area_pct': {'open': False}, 'eyes_open': {'open': True}, 'yaw': {'open': True}, 'pitch': {'open': True}}
                for metric_name, metric_config in metric_configs.items():
                    if not hasattr(self.config, f"filter_default_{metric_name}"): continue
                    f_def = getattr(self.config, f"filter_default_{metric_name}")
                    with gr.Accordion(metric_name.replace('_', ' ').title(), open=metric_config['open'], visible=False) as acc:
                        self.components['metric_accs'][metric_name] = acc
                        gr.Markdown(self.get_metric_description(metric_name), elem_classes="metric-description")
                        with gr.Column(elem_classes="plot-and-slider-column"):
                            self.components['metric_plots'][metric_name] = self._create_component(f'plot_{metric_name}', 'html', {'visible': True})
                            self.components['metric_sliders'][f"{metric_name}_min"] = self._create_component(f'slider_{metric_name}_min', 'slider', {'label': "Min", 'minimum': f_def['min'], 'maximum': f_def['max'], 'value': f_def.get('default_min', f_def['min']), 'step': f_def['step'], 'interactive': True, 'visible': True})
                            if 'default_max' in f_def: self.components['metric_sliders'][f"{metric_name}_max"] = self._create_component(f'slider_{metric_name}_max', 'slider', {'label': "Max", 'minimum': f_def['min'], 'maximum': f_def['max'], 'value': f_def['default_max'], 'step': f_def['step'], 'interactive': True, 'visible': True})
                            self.components['metric_auto_threshold_cbs'][metric_name] = self._create_component(f'auto_threshold_{metric_name}', 'checkbox', {'label': "Auto-Threshold this metric", 'value': False, 'interactive': True, 'visible': True})
                            if metric_name == "face_sim": self._create_component('require_face_match_input', 'checkbox', {'label': "Reject if no face", 'value': self.config.default_require_face_match, 'visible': True, 'info': "If checked, any frame without a detected face that meets the similarity threshold will be rejected."})
            with gr.Column(scale=2):
                with gr.Group(visible=False) as results_group:
                    self.components['results_group'] = results_group
                    gr.Markdown("### 🖼️ Step 2: Review Results")
                    with gr.Row():
                        self._create_component('gallery_view_toggle', 'radio', {'choices': self.GALLERY_VIEW_CHOICES, 'value': "Kept", 'label': "Show in Gallery"})
                        self._create_component('show_mask_overlay_input', 'checkbox', {'label': "Show Mask Overlay", 'value': self.config.gradio_show_mask_overlay})
                        self._create_component('overlay_alpha_slider', 'slider', {'label': "Overlay Alpha", 'minimum': 0.0, 'maximum': 1.0, 'value': self.config.gradio_overlay_alpha, 'step': 0.1})
                    self._create_component('results_gallery', 'gallery', {'columns': [4, 6, 8], 'rows': 2, 'height': 'auto', 'preview': True, 'allow_preview': True, 'object_fit': 'contain'})
                with gr.Group(visible=False) as export_group:
                    self.components['export_group'] = export_group
                    gr.Markdown("### 📤 Step 3: Export")
                    with gr.Row():
                        self._create_component('export_button', 'button', {'value': "Export Kept Frames", 'variant': "primary"})
                        self._create_component('dry_run_button', 'button', {'value': "Dry Run Export"})
                    with gr.Accordion("Export Options", open=True):
                        with gr.Row():
                            self._create_component('enable_crop_input', 'checkbox', {'label': "✂️ Crop to Subject", 'value': self.config.export_enable_crop})
                            self._create_component('crop_padding_input', 'slider', {'label': "Padding %", 'value': self.config.export_crop_padding})
                        self._create_component('crop_ar_input', 'textbox', {'label': "Crop ARs", 'value': self.config.export_crop_ars, 'info': "Comma-separated list (e.g., 16:9, 1:1). The best-fitting AR for each subject's mask will be chosen automatically."})

    def get_all_filter_keys(self) -> list[str]: return list(self.config.quality_weights.keys()) + ["quality_score", "face_sim", "mask_area_pct", "eyes_open", "yaw", "pitch"]

    def get_metric_description(self, metric_name: str) -> str:
        descriptions = {
            "quality_score": "A weighted average of all other quality metrics, providing an overall 'goodness' score for the frame.",
            "niqe": "Natural Image Quality Evaluator. A no-reference, opinion-unaware quality score. Lower is generally better, but it's scaled here so higher is better (like other metrics). Tends to favor clean, natural-looking images.",
            "sharpness": "Measures the amount of fine detail and edge clarity. Higher values indicate a sharper, more in-focus image.",
            "edge_strength": "Specifically measures the prominence of edges in the image. It's related to sharpness but focuses more on strong outlines.",
            "contrast": "The difference between the brightest and darkest parts of the image. Very high or very low contrast can be undesirable.",
            "brightness": "The overall lightness or darkness of the image.",
            "entropy": "Measures the amount of 'information' or complexity in the image. A very blurry or plain image will have low entropy.",
            "face_sim": "Face Similarity. How closely the best-detected face in the frame matches the reference face image. Only appears if a reference face is used.",
            "mask_area_pct": "Mask Area Percentage. The percentage of the screen taken up by the subject's mask. Useful for filtering out frames where the subject is too small or distant.",
            "eyes_open": "A score from 0.0 to 1.0 indicating how open the eyes are. A value of 1.0 means the eyes are fully open, and 0.0 means they are fully closed.",
            "yaw": "The rotation of the head around the vertical axis (turning left or right).",
            "pitch": "The rotation of the head around the side-to-side axis (looking up or down)."
        }
        return descriptions.get(metric_name, "No description available.")

    def _create_event_handlers(self):
        self.logger.info("Initializing Gradio event handlers...")
        self.components.update({'extracted_video_path_state': gr.State(""), 'extracted_frames_dir_state': gr.State(""), 'analysis_output_dir_state': gr.State(""), 'analysis_metadata_path_state': gr.State(""), 'all_frames_data_state': gr.State([]), 'per_metric_values_state': gr.State({}), 'scenes_state': gr.State([]), 'selected_scene_id_state': gr.State(None), 'scene_gallery_index_map_state': gr.State([]), 'gallery_image_state': gr.State(None), 'gallery_shape_state': gr.State(None), 'yolo_results_state': gr.State({}), 'discovered_faces_state': gr.State([]), 'resume_state': gr.State(False), 'enable_subject_mask_state': gr.State(True), 'min_mask_area_pct_state': gr.State(1.0), 'sharpness_base_scale_state': gr.State(2500.0), 'edge_strength_base_scale_state': gr.State(100.0)})

        # Undo/Redo State
        self.components['scene_history_state'] = gr.State(deque(maxlen=self.history_depth))

        self._setup_visibility_toggles(); self._setup_pipeline_handlers(); self._setup_filtering_handlers(); self._setup_bulk_scene_handlers()
        self.components['save_config_button'].click(lambda: self.config.save_config('config_dump.json'), [], []).then(lambda: "Configuration saved to config_dump.json", [], self.components['unified_log'])

        c = self.components
        c['cancel_button'].click(lambda: self.cancel_event.set(), [], [])
        c['pause_button'].click(
            self._toggle_pause,
            inputs=[gr.State(lambda: next((arg for arg in self.last_run_args if isinstance(arg, AdvancedProgressTracker)), None) if self.last_run_args else None)],
            outputs=c['pause_button']
        )
        c['clear_logs_button'].click(lambda: (self.all_logs.clear(), "")[1], [], c['unified_log'])
        c['log_level_filter'].change(lambda level: (setattr(self, 'log_filter_level', level), "\n".join([l for l in self.all_logs if self.log_filter_level.upper() == "DEBUG" or f"[{level.upper()}]" in l][-1000:]))[1], c['log_level_filter'], c['unified_log'])
        c['scene_editor_yolo_subject_id'].change(
            self.on_select_yolo_subject_wrapper,
            inputs=[c['scene_editor_yolo_subject_id'], c['scenes_state'], c['selected_scene_id_state'], c['extracted_frames_dir_state'], c['scene_gallery_view_toggle'], c['scene_history_state']] + self.ana_input_components,
            outputs=[c['scenes_state'], c['scene_gallery'], c['scene_gallery_index_map_state'], c['sceneeditorstatusmd'], c['scene_history_state']]
        )
        c['run_diagnostics_button'].click(self.run_system_diagnostics, inputs=[], outputs=[c['unified_log']])

    def _push_history(self, scenes: List[Dict], history: Deque) -> Deque:
        # Deep copy scenes to history
        import copy
        history.append(copy.deepcopy(scenes))
        return history

    def _undo_last_action(self, scenes: List[Dict], history: Deque, output_dir: str, view: str) -> tuple:
        if not history:
            return scenes, gr.update(), gr.update(), "Nothing to undo.", history

        prev_scenes = history.pop()

        # Save restored state to disk
        save_scene_seeds([Scene(**s) for s in prev_scenes], output_dir, self.logger)

        gallery_items, index_map, _ = build_scene_gallery_items(prev_scenes, view, output_dir)
        status_text, button_update = get_scene_status_text([Scene(**s) for s in prev_scenes])

        return prev_scenes, gr.update(value=gallery_items), gr.update(value=index_map), "Undid last action.", history

    def _run_task_with_progress(self, task_func: Callable, output_components: list, progress: Callable, *args) -> Generator[dict, None, None]:
        self.last_run_args = args
        self.cancel_event.clear()
        tracker_instance = next((arg for arg in args if isinstance(arg, AdvancedProgressTracker)), None)
        if tracker_instance: tracker_instance.pause_event.set()
        op_name = getattr(task_func, '__name__', 'Unknown Task').replace('_wrapper', '').replace('_', ' ').title()
        yield {self.components['cancel_button']: gr.update(interactive=True), self.components['pause_button']: gr.update(interactive=True), self.components['unified_status']: f"🚀 **Starting: {op_name}...**"}

        def run_and_capture():
            try:
                # Inject model_registry into args if it's expected
                # However, task_func here is wrapper functions like run_pre_analysis_wrapper
                # which call _run_pipeline.
                # Wait, _run_task_with_progress calls wrapper. wrapper calls _run_pipeline.
                # So I need to modify _run_pipeline, not _run_task_with_progress.
                # But I am editing AppUI class.
                res = task_func(*args)
                if hasattr(res, '__iter__') and not isinstance(res, (dict, list, tuple, str)):
                    for item in res:
                        self.progress_queue.put({"ui_update": item})
                else:
                    self.progress_queue.put({"ui_update": res})
            except Exception as e:
                self.app_logger.error(f"Task failed: {e}", exc_info=True)
                self.progress_queue.put({"ui_update": {"unified_log": f"[CRITICAL] Task failed: {e}"}})

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(run_and_capture)
            start_time = time.time()
            while future.running():
                if time.time() - start_time > 3600: self.app_logger.error("Task timed out after 1 hour"); self.cancel_event.set(); future.cancel(); break
                if self.cancel_event.is_set(): future.cancel(); break
                if tracker_instance and not tracker_instance.pause_event.is_set(): yield {self.components['unified_status']: f"⏸️ **Paused: {op_name}**"}; time.sleep(0.2); continue
                try:
                    msg, update_dict = self.progress_queue.get(timeout=0.1), {}
                    if "ui_update" in msg:
                        update_dict.update(msg["ui_update"])
                    if "log" in msg:
                        self.all_logs.append(msg['log'])
                        log_level_map = {level: i for i, level in enumerate(self.LOG_LEVEL_CHOICES)}
                        current_filter_level = log_level_map.get(self.log_filter_level.upper(), 1)
                        filtered_logs = [l for l in self.all_logs if any(f"[{level}]" in l for level in self.LOG_LEVEL_CHOICES[current_filter_level:])]
                        update_dict[self.components['unified_log']] = "\n".join(filtered_logs[-1000:])
                    if "progress" in msg:
                        from progress import ProgressEvent # Local import to avoid circular dependency
                        p = ProgressEvent(**msg["progress"])
                        progress(p.fraction, desc=f"{p.stage} ({p.done}/{p.total}) • {p.eta_formatted}")
                        status_md = f"**Running: {op_name}**\n- Stage: {p.stage} ({p.done}/{p.total})\n- ETA: {p.eta_formatted}"
                        if p.substage: status_md += f"\n- Step: {p.substage}"
                        update_dict[self.components['unified_status']] = status_md
                    if update_dict: yield update_dict
                except Empty: pass
                time.sleep(0.05)

            # Drain remaining items from queue
            while not self.progress_queue.empty():
                try:
                    msg, update_dict = self.progress_queue.get_nowait(), {}
                    if "ui_update" in msg:
                        update_dict.update(msg["ui_update"])
                    if "log" in msg:
                        self.all_logs.append(msg['log'])
                        log_level_map = {level: i for i, level in enumerate(self.LOG_LEVEL_CHOICES)}
                        current_filter_level = log_level_map.get(self.log_filter_level.upper(), 1)
                        filtered_logs = [l for l in self.all_logs if any(f"[{level}]" in l for level in self.LOG_LEVEL_CHOICES[current_filter_level:])]
                        update_dict[self.components['unified_log']] = "\n".join(filtered_logs[-1000:])
                    if update_dict: yield update_dict
                except Empty: break

    def on_select_yolo_subject_wrapper(self, subject_id: str, scenes: list, shot_id: int, outdir: str, view: str, history: Deque, *ana_args) -> tuple:
        """Wrapper for handling subject selection from the YOLO radio buttons."""
        try:
            if not subject_id: return scenes, gr.update(), gr.update(), "Please select a Subject ID.", history

            # Push history before modification
            history = self._push_history(scenes, history)

            subject_idx = int(subject_id) - 1
            scene = next((s for s in scenes if s['shot_id'] == shot_id), None)
            if not scene: return scenes, gr.update(), gr.update(), "Scene not found.", history
            yolo_boxes = scene.get('yolo_detections', [])
            if not (0 <= subject_idx < len(yolo_boxes)): return scenes, gr.update(), gr.update(), f"Invalid Subject ID. Please enter a number between 1 and {len(yolo_boxes)}.", history

            masker = _create_analysis_context(self.config, self.logger, self.thumbnail_manager, self.cuda_available, self.ana_ui_map_keys, list(ana_args), self.model_registry)
            selected_box = yolo_boxes[subject_idx]
            # Access seed_selector from masker instance
            selected_xywh = masker.seed_selector._xyxy_to_xywh(selected_box['bbox'])
            overrides = {"manual_bbox_xywh": selected_xywh, "seedtype": "yolo_manual"}
            scene_idx = scenes.index(scene)
            if 'initial_bbox' not in scenes[scene_idx] or scenes[scene_idx]['initial_bbox'] is None:
                scenes[scene_idx]['initial_bbox'] = selected_xywh
            scenes[scene_idx]['selected_bbox'] = selected_xywh
            initial_bbox = scenes[scene_idx].get('initial_bbox')
            scenes[scene_idx]['is_overridden'] = initial_bbox is not None and selected_xywh != initial_bbox

            scene_state = SceneState(scenes[scene_idx])
            _recompute_single_preview(scene_state, masker, overrides, self.thumbnail_manager, self.logger)
            scenes[scene_idx] = scene_state.data # Update dict

            save_scene_seeds([Scene(**s) for s in scenes], outdir, self.logger)
            gallery_items, index_map, _ = build_scene_gallery_items(scenes, view, outdir)
            return scenes, gr.update(value=gallery_items), gr.update(value=index_map), f"Subject {subject_id} selected and preview recomputed.", history
        except (ValueError, TypeError):
            gallery_items, index_map, _ = build_scene_gallery_items(scenes, view, outdir)
            return scenes, gr.update(value=gallery_items), gr.update(value=index_map), "Invalid Subject ID. Please enter a number.", history
        except Exception as e:
            self.logger.error("Failed to select YOLO subject", exc_info=True)
            gallery_items, index_map, _ = build_scene_gallery_items(scenes, view, outdir)
            return scenes, gr.update(value=gallery_items), gr.update(value=index_map), f"Error: {e}", history

    def _setup_bulk_scene_handlers(self):
        """Sets up Gradio event handlers for the scene selection and editing tab."""
        c = self.components

        def on_page_change(scenes, view, output_dir, page_num):
            page_num = int(page_num)
            items, index_map, total_pages = build_scene_gallery_items(scenes, view, output_dir, page_num=page_num)
            return gr.update(value=items), index_map, f"/ {total_pages} pages", page_num

        def _refresh_scene_gallery(scenes, view, output_dir):
            items, index_map, total_pages = build_scene_gallery_items(scenes, view, output_dir, page_num=1)
            return gr.update(value=items), index_map, f"/ {total_pages} pages", 1

        c['scene_gallery_view_toggle'].change(_refresh_scene_gallery, [c['scenes_state'], c['scene_gallery_view_toggle'], c['extracted_frames_dir_state']], [c['scene_gallery'], c['scene_gallery_index_map_state'], c['total_pages_label'], c['page_number_input']])
        c['next_page_button'].click(lambda scenes, view, output_dir, page_num: on_page_change(scenes, view, output_dir, page_num + 1), [c['scenes_state'], c['scene_gallery_view_toggle'], c['extracted_frames_dir_state'], c['page_number_input']], [c['scene_gallery'], c['scene_gallery_index_map_state'], c['total_pages_label'], c['page_number_input']])
        c['prev_page_button'].click(lambda scenes, view, output_dir, page_num: on_page_change(scenes, view, output_dir, page_num - 1), [c['scenes_state'], c['scene_gallery_view_toggle'], c['extracted_frames_dir_state'], c['page_number_input']], [c['scene_gallery'], c['scene_gallery_index_map_state'], c['total_pages_label'], c['page_number_input']])
        c['page_number_input'].submit(on_page_change, [c['scenes_state'], c['scene_gallery_view_toggle'], c['extracted_frames_dir_state'], c['page_number_input']], [c['scene_gallery'], c['scene_gallery_index_map_state'], c['total_pages_label'], c['page_number_input']])

        c['scene_gallery'].select(self.on_select_for_edit, inputs=[c['scenes_state'], c['scene_gallery_view_toggle'], c['scene_gallery_index_map_state'], c['extracted_frames_dir_state'], c['yolo_results_state']], outputs=[c['scenes_state'], c['scene_filter_status'], c['scene_gallery'], c['scene_gallery_index_map_state'], c['selected_scene_id_state'], c['sceneeditorstatusmd'], c['sceneeditorpromptinput'], c['sceneeditoraccordion'], c['gallery_image_state'], c['gallery_shape_state'], c['scene_editor_yolo_subject_id'], c['propagate_masks_button'], c['yolo_results_state']])

        c['scenerecomputebutton'].click(fn=lambda scenes, shot_id, outdir, view, txt, subject_id, history, *ana_args: _wire_recompute_handler(self.config, self.app_logger, self.thumbnail_manager, [Scene(**s) for s in scenes], shot_id, outdir, txt, view, self.ana_ui_map_keys, list(ana_args), self.cuda_available, self.model_registry) if (txt and txt.strip()) else self.on_select_yolo_subject_wrapper(subject_id, scenes, shot_id, outdir, view, history, *ana_args), inputs=[c['scenes_state'], c['selected_scene_id_state'], c['analysis_output_dir_state'], c['scene_gallery_view_toggle'], c['sceneeditorpromptinput'], c['scene_editor_yolo_subject_id'], c['scene_history_state'], *self.ana_input_components], outputs=[c['scenes_state'], c['scene_gallery'], c['scene_gallery_index_map_state'], c['sceneeditorstatusmd'], c['scene_history_state']]) # Updated outputs for history

        c['sceneresetbutton'].click(self.on_reset_scene_wrapper, inputs=[c['scenes_state'], c['selected_scene_id_state'], c['analysis_output_dir_state'], c['scene_gallery_view_toggle'], c['scene_history_state']] + self.ana_input_components, outputs=[c['scenes_state'], c['scene_gallery'], c['scene_gallery_index_map_state'], c['sceneeditorstatusmd'], c['scene_history_state']])

        c['sceneincludebutton'].click(lambda s, sid, out, v, h: self.on_editor_toggle(s, sid, out, v, "included", h), inputs=[c['scenes_state'], c['selected_scene_id_state'], c['extracted_frames_dir_state'], c['scene_gallery_view_toggle'], c['scene_history_state']], outputs=[c['scenes_state'], c['scene_filter_status'], c['scene_gallery'], c['scene_gallery_index_map_state'], c['propagate_masks_button'], c['scene_history_state']])
        c['sceneexcludebutton'].click(lambda s, sid, out, v, h: self.on_editor_toggle(s, sid, out, v, "excluded", h), inputs=[c['scenes_state'], c['selected_scene_id_state'], c['extracted_frames_dir_state'], c['scene_gallery_view_toggle'], c['scene_history_state']], outputs=[c['scenes_state'], c['scene_filter_status'], c['scene_gallery'], c['scene_gallery_index_map_state'], c['propagate_masks_button'], c['scene_history_state']])

        c['sceneundobutton'].click(self._undo_last_action, inputs=[c['scenes_state'], c['scene_history_state'], c['extracted_frames_dir_state'], c['scene_gallery_view_toggle']], outputs=[c['scenes_state'], c['scene_gallery'], c['scene_gallery_index_map_state'], c['sceneeditorstatusmd'], c['scene_history_state']])

        def init_scene_gallery(scenes, view, outdir):
            if not scenes: return gr.update(value=[]), []
            gallery_items, index_map, _ = build_scene_gallery_items(scenes, view, outdir)
            return gr.update(value=gallery_items), index_map

        c['scenes_state'].change(init_scene_gallery, [c['scenes_state'], c['scene_gallery_view_toggle'], c['extracted_frames_dir_state']], [c['scene_gallery'], c['scene_gallery_index_map_state']])

        bulk_action_outputs = [c['scenes_state'], c['scene_filter_status'], c['scene_gallery'], c['scene_gallery_index_map_state'], c['propagate_masks_button']]
        bulk_filter_inputs = [c['scenes_state'], c['scene_mask_area_min_input'], c['scene_face_sim_min_input'], c['scene_confidence_min_input'], c['enable_face_filter_input'], c['extracted_frames_dir_state'], c['scene_gallery_view_toggle']]
        for comp in [c['scene_mask_area_min_input'], c['scene_face_sim_min_input'], c['scene_confidence_min_input']]:
            comp.release(self.on_apply_bulk_scene_filters_extended, bulk_filter_inputs, bulk_action_outputs)

    def on_reset_scene_wrapper(self, scenes, shot_id, outdir, view, history, *ana_args):
        try:
            # Push history
            history = self._push_history(scenes, history)

            scene_idx = next((i for i, s in enumerate(scenes) if s['shot_id'] == shot_id), None)
            if scene_idx is None: return scenes, gr.update(), gr.update(), "Scene not found.", history
            scene = scenes[scene_idx]
            scene.update({'seed_config': {}, 'seed_result': {}, 'seed_metrics': {}, 'manual_status_change': False, 'status': 'included', 'is_overridden': False, 'selected_bbox': scene.get('initial_bbox')})
            masker = _create_analysis_context(self.config, self.logger, self.thumbnail_manager, self.cuda_available, self.ana_ui_map_keys, list(ana_args), self.model_registry)

            scene_state = SceneState(scenes[scene_idx])
            _recompute_single_preview(scene_state, masker, {}, self.thumbnail_manager, self.logger)
            scenes[scene_idx] = scene_state.data

            save_scene_seeds([Scene(**s) for s in scenes], outdir, self.logger)
            gallery_items, index_map, _ = build_scene_gallery_items(scenes, view, outdir)
            return scenes, gr.update(value=gallery_items), gr.update(value=index_map), f"Scene {shot_id} has been reset to its original state.", history
        except Exception as e:
            self.logger.error(f"Failed to reset scene {shot_id}", exc_info=True)
            gallery_items, index_map, _ = build_scene_gallery_items(scenes, view, outdir)
            return scenes, gr.update(value=gallery_items), gr.update(value=index_map), f"Error resetting scene: {e}", history

    def _empty_selection_response(self, scenes, indexmap):
        status_text, button_update = get_scene_status_text([Scene(**s) for s in scenes])
        return (scenes, status_text, gr.update(), indexmap, None, "Select a scene from the gallery to edit its properties.", "", gr.update(open=False), None, None, gr.update(visible=False, choices=[], value=None), button_update, {})

    def on_select_for_edit(self, scenes, view, indexmap, outputdir, yoloresultsstate, event: Optional[gr.EventData] = None, request: Optional[gr.Request] = None):
        sel_idx = getattr(event, "index", None) if event else None
        if sel_idx is None: return self._empty_selection_response(scenes, indexmap)
        if not scenes or not indexmap or not (0 <= sel_idx < len(indexmap)) or not (0 <= (scene_idx_in_state := indexmap[sel_idx]) < len(scenes)):
            self.logger.error(f"Invalid gallery or scene index on selection: gallery_idx={sel_idx}, scene_idx={scene_idx_in_state}")
            return self._empty_selection_response(scenes, indexmap)
        scene = scenes[scene_idx_in_state]
        cfg = scene.get("seed_config") or {}
        shotid = scene.get("shot_id")

        # Scene thumb logic
        previews_dir = Path(outputdir) / "previews"
        thumb_path = previews_dir / f"scene_{shotid:05d}.jpg"
        thumb_path_str = str(thumb_path) if thumb_path.exists() else None

        gallery_image = self.thumbnail_manager.get(Path(thumb_path_str)) if thumb_path_str else None
        gallery_shape = gallery_image.shape[:2] if gallery_image is not None else None
        status_md = f"**Editing Scene {shotid}** (Frames {scene.get('start_frame', '?')}-{scene.get('end_frame', '?')})"
        prompt = cfg.get("text_prompt", "")
        subject_choices = [f"{i+1}" for i in range(len(scene.get('yolo_detections', [])))]
        subject_id_update = gr.update(choices=subject_choices, value=None, visible=bool(subject_choices))
        _, button_update = get_scene_status_text([Scene(**s) for s in scenes])
        return (scenes, get_scene_status_text([Scene(**s) for s in scenes])[0], gr.update(), indexmap, shotid, gr.update(value=status_md), gr.update(value=prompt), gr.update(open=True), gallery_image, gallery_shape, subject_id_update, button_update, yoloresultsstate)

    def on_editor_toggle(self, scenes, selected_shotid, outputfolder, view, new_status, history):
        """Toggles the status of a scene from the scene editor."""
        # Push history
        history = self._push_history(scenes, history)

        scenes_objs = [Scene(**s) for s in scenes]
        scenes_objs, status_text, _, button_update = toggle_scene_status(scenes_objs, selected_shotid, new_status, outputfolder, self.logger)
        scenes = [s.model_dump() for s in scenes_objs]

        items, index_map, _ = build_scene_gallery_items(scenes, view, outputfolder)
        return scenes, status_text, gr.update(value=items), gr.update(value=index_map), button_update, history

    def _toggle_pause(self, tracker: 'AdvancedProgressTracker') -> str:
        """Toggles the pause state of a running task."""
        if tracker.pause_event.is_set(): tracker.pause_event.clear(); return "⏸️ Paused"
        else: tracker.pause_event.set(); return "▶️ Resume"

    def run_system_diagnostics(self) -> Generator[str, None, None]:
        """Runs a comprehensive suite of system checks and a dry run."""
        self.logger.info("Starting system diagnostics...")
        report = ["\n\n--- System Diagnostics Report ---", "\n[SECTION 1: System & Environment]"]
        try: report.append(f"  - Python Version: OK ({sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro})")
        except Exception as e: report.append(f"  - Python Version: FAILED ({e})")
        try:
            report.append(f"  - PyTorch Version: OK ({torch.__version__})")
            if torch.cuda.is_available(): report.append(f"  - CUDA: OK (Version: {torch.version.cuda}, GPU: {torch.cuda.get_device_name(0)})")
            else: report.append("  - CUDA: NOT AVAILABLE (Running in CPU mode)")
        except Exception as e: report.append(f"  - PyTorch/CUDA Check: FAILED ({e})")
        report.append("\n[SECTION 2: Core Dependencies]")
        for dep in ["cv2", "gradio", "imagehash", "mediapipe", "sam3"]:
            try: __import__(dep.split('.')[0]); report.append(f"  - {dep}: OK")
            except ImportError: report.append(f"  - {dep}: FAILED (Not Installed)")
        report.append("\n[SECTION 3: Paths & Assets]")
        for name, path in {"Models Directory": Path(self.config.models_dir), "Dry Run Assets": Path("dry-run-assets"), "Sample Video": Path("dry-run-assets/sample.mp4"), "Sample Image": Path("dry-run-assets/sample.jpg")}.items():
            report.append(f"  - {name}: {'OK' if path.exists() else 'FAILED'} (Path: {path})")
        report.append("\n[SECTION 4: Model Loading Simulation]")
        report.append("  - Skipping Model Loading Simulation (Models loaded on demand)")
        report.append("\n[SECTION 5: E2E Pipeline Simulation]")
        temp_output_dir = Path(self.config.downloads_dir) / "dry_run_output"
        shutil.rmtree(temp_output_dir, ignore_errors=True); temp_output_dir.mkdir(parents=True, exist_ok=True)
        try:
            report.append("  - Stage 1: Frame Extraction...")
            ext_event = ExtractionEvent(source_path="dry-run-assets/sample.mp4", method='interval', interval='1.0', max_resolution="720", thumbnails_only=True, thumb_megapixels=0.2, scene_detect=True)
            ext_result = deque(execute_extraction(ext_event, self.progress_queue, self.cancel_event, self.logger, self.config), maxlen=1)[0]
            if not ext_result.get("done"): raise RuntimeError("Extraction failed")
            report[-1] += " OK"
            report.append("  - Stage 2: Pre-analysis...")
            pre_ana_event = PreAnalysisEvent(output_folder=ext_result['extracted_frames_dir_state'], video_path=ext_result['extracted_video_path_state'], scene_detect=True, pre_analysis_enabled=True, pre_sample_nth=1, primary_seed_strategy="🧑‍🤝‍🧑 Find Prominent Person", face_model_name="buffalo_l", tracker_model_name="sam3", min_mask_area_pct=1.0, sharpness_base_scale=2500.0, edge_strength_base_scale=100.0)
            pre_ana_result = deque(execute_pre_analysis(pre_ana_event, self.progress_queue, self.cancel_event, self.logger, self.config, self.thumbnail_manager, self.cuda_available), maxlen=1)[0]
            if not pre_ana_result.get("done"): raise RuntimeError(f"Pre-analysis failed: {pre_ana_result}")
            report[-1] += " OK"
            scenes = pre_ana_result['scenes']
            report.append("  - Stage 3: Mask Propagation...")
            prop_event = PropagationEvent(output_folder=pre_ana_result['output_dir'], video_path=ext_result['extracted_video_path_state'], scenes=scenes, analysis_params=pre_ana_event)
            prop_result = deque(execute_propagation(prop_event, self.progress_queue, self.cancel_event, self.logger, self.config, self.thumbnail_manager, self.cuda_available), maxlen=1)[0]
            if not prop_result.get("done"): raise RuntimeError("Propagation failed")
            report[-1] += " OK"
            report.append("  - Stage 4: Frame Analysis...")
            ana_result = deque(execute_analysis(prop_event, self.progress_queue, self.cancel_event, self.logger, self.config, self.thumbnail_manager, self.cuda_available), maxlen=1)[0]
            if not ana_result.get("done"): raise RuntimeError("Analysis failed")
            report[-1] += " OK"
            output_dir = ana_result['output_dir']
            from core.filtering import load_and_prep_filter_data, apply_all_filters_vectorized
            all_frames, _ = load_and_prep_filter_data(output_dir, self.get_all_filter_keys, self.config)
            report.append("  - Stage 5: Filtering...")
            kept, _, _, _ = apply_all_filters_vectorized(all_frames, {'require_face_match': False, 'dedup_thresh': -1}, self.config, output_dir=ana_result['output_dir'])
            report[-1] += f" OK (kept {len(kept)} frames)"
            report.append("  - Stage 6: Export...")
            export_event = ExportEvent(all_frames_data=all_frames, output_dir=ana_result['output_dir'], video_path=ext_result['extracted_video_path_state'], enable_crop=False, crop_ars="", crop_padding=0, filter_args={'require_face_match': False, 'dedup_thresh': -1})
            export_msg = export_kept_frames(export_event, self.config, self.logger, self.thumbnail_manager, self.cancel_event)
            if "Error" in export_msg: raise RuntimeError(f"Export failed: {export_msg}")
            report[-1] += " OK"
        except Exception as e:
            error_message = f"FAILED ({e})"
            if "..." in report[-1]: report[-1] += error_message
            else: report.append(f"  - Pipeline Simulation: {error_message}")
            self.logger.error("Dry run pipeline failed", exc_info=True)
        final_report = "\n".join(report)
        self.logger.info(final_report)
        yield final_report

    def _create_pre_analysis_event(self, *args: Any) -> 'PreAnalysisEvent':
        """Creates a `PreAnalysisEvent` from the raw Gradio UI component values."""
        ui_args = dict(zip(self.ana_ui_map_keys, args))
        clean_args = {k: v for k, v in ui_args.items() if v is not None}
        strategy = clean_args.get('primary_seed_strategy', self.config.default_primary_seed_strategy)
        if strategy == "👤 By Face": clean_args.update({'enable_face_filter': True, 'text_prompt': ""})
        elif strategy == "📝 By Text": clean_args.update({'enable_face_filter': False, 'face_ref_img_path': ""})
        return PreAnalysisEvent.model_validate(clean_args)

    def _run_pipeline(self, pipeline_func: Callable, event: Any, progress: Callable, success_callback: Optional[Callable] = None, *args):
        """A generic wrapper for executing a pipeline function."""
        try:
            # Pass model_registry to pipeline functions
            for result in pipeline_func(event, self.progress_queue, self.cancel_event, self.app_logger, self.config, self.thumbnail_manager, self.cuda_available, progress=progress, model_registry=self.model_registry):
                if isinstance(result, dict):
                    if self.cancel_event.is_set(): yield {"unified_log": f"{pipeline_func.__name__} cancelled."}; return
                    if result.get("done"):
                        if success_callback: yield success_callback(result)
                        return
            yield {"unified_log": f"❌ {pipeline_func.__name__} did not complete successfully."}
        except Exception as e:
            self.app_logger.error(f"{pipeline_func.__name__} execution failed", exc_info=True)
            yield {"unified_log": f"[ERROR] An unexpected error occurred in {pipeline_func.__name__}: {e}"}

    def run_extraction_wrapper(self, *args):
        """Wrapper for the extraction pipeline."""
        ui_args = dict(zip(self.ext_ui_map_keys, args))
        if isinstance(ui_args.get('upload_video'), list):
             if ui_args['upload_video']: ui_args['upload_video'] = ui_args['upload_video'][0]
             else: ui_args['upload_video'] = None
        clean_args = {k: v for k, v in ui_args.items() if v is not None}
        event = ExtractionEvent.model_validate(clean_args)
        yield from self._run_pipeline(execute_extraction, event, gr.Progress(), self._on_extraction_success)

    def add_to_queue_handler(self, *args):
        ui_args = dict(zip(self.ext_ui_map_keys, args))
        source_path = ui_args.get('source_path')
        upload_video = ui_args.get('upload_video')
        paths = []
        if upload_video:
             if isinstance(upload_video, str): upload_video = [upload_video]
             if isinstance(upload_video, list): paths.extend(upload_video)
        if source_path:
             path = Path(source_path)
             if path.is_dir():
                 video_exts = {'.mp4', '.mkv', '.avi', '.mov', '.webm', '.flv', '.wmv', '.m4v'}
                 for p in path.iterdir():
                     if p.is_file() and p.suffix.lower() in video_exts: paths.append(str(p))
             else: paths.append(str(path))
        if not paths: return gr.update(value=self.batch_manager.get_status_list())
        clean_args = {k: v for k, v in ui_args.items() if v is not None}
        clean_args.pop('source_path', None); clean_args.pop('upload_video', None)
        with self.batch_manager.lock:
             for p in paths:
                 item = BatchItem(id=str(uuid.uuid4()), path=str(p), params=clean_args.copy())
                 self.batch_manager.queue.append(item)
        return gr.update(value=self.batch_manager.get_status_list())

    def clear_queue_handler(self):
        self.batch_manager.clear_all()
        return gr.update(value=self.batch_manager.get_status_list())

    def _batch_processor(self, item: BatchItem, progress_callback: Callable):
        params = item.params.copy(); params['source_path'] = item.path; params['upload_video'] = None
        event = ExtractionEvent.model_validate(params)
        gen = execute_extraction(event, self.progress_queue, self.batch_manager.stop_event, self.logger, self.config, progress=progress_callback)
        result = {}
        for update in gen: result = update
        if not result.get('done'): raise RuntimeError(result.get('unified_log', 'Unknown failure'))
        return result

    def start_batch_wrapper(self, workers: float):
        if not self.batch_manager.queue: yield self.batch_manager.get_status_list(); return
        self.batch_manager.start_processing(self._batch_processor, max_workers=int(workers))
        while self.batch_manager.is_running:
             yield self.batch_manager.get_status_list()
             time.sleep(1.0)
        yield self.batch_manager.get_status_list()

    def stop_batch_handler(self):
        self.batch_manager.stop_processing()
        return "Stopping..."

    def _on_extraction_success(self, result: dict) -> dict:
        """Callback for successful extraction."""
        return {
            self.components['extracted_video_path_state']: result['extracted_video_path_state'],
            self.components['extracted_frames_dir_state']: result['extracted_frames_dir_state'],
            self.components['unified_log']: f"Extraction complete. Frames saved to {result['extracted_frames_dir_state']}"
        }

    def _on_pre_analysis_success(self, result: dict) -> dict:
        """Callback for successful pre-analysis."""
        scenes_objs = [Scene(**s) for s in result['scenes']]
        status_text, button_update = get_scene_status_text(scenes_objs)
        return {
            self.components['scenes_state']: result['scenes'],
            self.components['analysis_output_dir_state']: result['output_dir'],
            self.components['unified_log']: f"Pre-analysis complete. Found {len(result['scenes'])} scenes.",
            self.components['seeding_results_column']: result.get('seeding_results_column', gr.update(visible=True)),
            self.components['propagation_group']: result.get('propagation_group', gr.update(visible=True)),
            self.components['propagate_masks_button']: button_update,
            self.components['scene_filter_status']: status_text
        }

    def run_pre_analysis_wrapper(self, *args):
        """Wrapper for the pre-analysis pipeline."""
        event = self._create_pre_analysis_event(*args)
        yield from self._run_pipeline(execute_pre_analysis, event, gr.Progress(), self._on_pre_analysis_success)

    def run_propagation_wrapper(self, scenes, *args):
        """Wrapper for the mask propagation pipeline."""
        if not scenes: yield {"unified_log": "No scenes to propagate. Run Pre-Analysis first."}; return
        params = self._create_pre_analysis_event(*args)
        event = PropagationEvent(output_folder=params.output_folder, video_path=params.video_path, scenes=scenes, analysis_params=params)
        yield from self._run_pipeline(execute_propagation, event, gr.Progress(), self._on_propagation_success)

    def _on_propagation_success(self, result: dict) -> dict:
        """Callback for successful propagation."""
        return {
            self.components['scenes_state']: result['scenes'],
            self.components['unified_log']: "Propagation complete."
        }

    def run_analysis_wrapper(self, scenes, *args):
        """Wrapper for the analysis pipeline."""
        if not scenes: yield {"unified_log": "No scenes to analyze. Run Pre-Analysis first."}; return
        params = self._create_pre_analysis_event(*args)
        event = PropagationEvent(output_folder=params.output_folder, video_path=params.video_path, scenes=scenes, analysis_params=params)
        yield from self._run_pipeline(execute_analysis, event, gr.Progress(), self._on_analysis_success)

    def _on_analysis_success(self, result: dict) -> dict:
        """Callback for successful analysis."""
        return {
            self.components['analysis_metadata_path_state']: result['metadata_path'],
            self.components['unified_log']: f"Analysis complete. Metadata saved to {result['metadata_path']}"
        }

    def run_session_load_wrapper(self, session_path: str):
        """Wrapper for loading a saved session."""
        event = SessionLoadEvent(session_path=session_path)
        yield from self._run_pipeline(execute_session_load, event, gr.Progress(), lambda res: {
            self.components['extracted_video_path_state']: res['extracted_video_path_state'],
            self.components['extracted_frames_dir_state']: res['extracted_frames_dir_state'],
            self.components['analysis_output_dir_state']: res['analysis_output_dir_state'],
            self.components['analysis_metadata_path_state']: res['analysis_metadata_path_state'],
            self.components['scenes_state']: res['scenes'],
            self.components['unified_log']: f"Session loaded from {session_path}"
        })

    def _fix_strategy_visibility(self, strategy: str) -> dict:
        """Updates UI visibility based on the selected seeding strategy."""
        is_face = "By Face" in strategy or "Fallback" in strategy
        is_text = "By Text" in strategy or "Fallback" in strategy
        is_auto = "Prominent Person" in strategy
        return {
            self.components['face_seeding_group']: gr.update(visible=is_face),
            self.components['text_seeding_group']: gr.update(visible=is_text),
            self.components['auto_seeding_group']: gr.update(visible=is_auto),
            self.components['enable_face_filter_input']: gr.update(value=is_face, visible=is_face),
        }

    def _setup_visibility_toggles(self):
        """Sets up Gradio event handlers for dynamically showing/hiding UI components."""
        c = self.components
        def handle_source_change(path):
            is_folder = is_image_folder(path)
            if is_folder or not path: return {c['max_resolution']: gr.update(visible=False), c['thumbnail_group']: gr.update(visible=False)}
            else: return {c['max_resolution']: gr.update(visible=True), c['thumbnail_group']: gr.update(visible=True)}
        source_controls = [c['source_input'], c['upload_video_input']]
        video_specific_outputs = [c['max_resolution'], c['thumbnail_group']]
        for control in source_controls: control.change(handle_source_change, inputs=control, outputs=video_specific_outputs)
        c['method_input'].change(lambda m: {c['interval_input']: gr.update(visible=m == 'interval'), c['nth_frame_input']: gr.update(visible=m in ['every_nth_frame', 'nth_plus_keyframes'])}, c['method_input'], [c['interval_input'], c['nth_frame_input']])
        c['primary_seed_strategy_input'].change(self._fix_strategy_visibility, inputs=c['primary_seed_strategy_input'], outputs=[c['face_seeding_group'], c['text_seeding_group'], c['auto_seeding_group'], c['enable_face_filter_input']])

    def get_inputs(self, keys: list[str]) -> list[gr.components.Component]:
        return [self.ui_registry[k] for k in keys if k in self.ui_registry]

    def _setup_pipeline_handlers(self):
        """Sets up the Gradio event handlers for the main processing pipelines."""
        c = self.components
        all_outputs = [v for v in c.values() if hasattr(v, "_id")]
        def session_load_handler(session_path, progress=gr.Progress()):
            session_load_keys_filtered = [k for k in self.session_load_keys if k != 'progress_bar']
            session_load_outputs = [c[key] for key in session_load_keys_filtered if key in c and hasattr(c[key], "_id")]
            yield from self._run_task_with_progress(self.run_session_load_wrapper, session_load_outputs, progress, session_path)
        def extraction_handler(*args, progress=gr.Progress()): yield from self._run_task_with_progress(self.run_extraction_wrapper, all_outputs, progress, *args)
        def pre_analysis_handler(*args, progress=gr.Progress()): yield from self._run_task_with_progress(self.run_pre_analysis_wrapper, all_outputs, progress, *args)
        def propagation_handler(scenes, *args, progress=gr.Progress()): yield from self._run_task_with_progress(self.run_propagation_wrapper, all_outputs, progress, scenes, *args)
        def analysis_handler(scenes, *args, progress=gr.Progress()): yield from self._run_task_with_progress(self.run_analysis_wrapper, all_outputs, progress, scenes, *args)

        c['load_session_button'].click(fn=session_load_handler, inputs=[c['session_path_input']], outputs=all_outputs, show_progress="hidden")
        ext_inputs = self.get_inputs(self.ext_ui_map_keys)
        self.ana_input_components = [c['extracted_frames_dir_state'], c['extracted_video_path_state']]
        self.ana_input_components.extend(self.get_inputs(self.ana_ui_map_keys))
        prop_inputs = [c['scenes_state']] + self.ana_input_components
        c['start_extraction_button'].click(fn=extraction_handler, inputs=ext_inputs, outputs=all_outputs, show_progress="hidden").then(lambda d: gr.update(selected=1) if d else gr.update(), c['extracted_frames_dir_state'], c['main_tabs'])

        c['add_to_queue_button'].click(self.add_to_queue_handler, inputs=ext_inputs, outputs=[c['batch_queue_dataframe']])
        c['clear_queue_button'].click(self.clear_queue_handler, inputs=[], outputs=[c['batch_queue_dataframe']])
        c['start_batch_button'].click(self.start_batch_wrapper, inputs=[c['batch_workers_slider']], outputs=[c['batch_queue_dataframe']])
        c['stop_batch_button'].click(self.stop_batch_handler, inputs=[], outputs=[])

        c['start_pre_analysis_button'].click(fn=pre_analysis_handler, inputs=self.ana_input_components, outputs=all_outputs, show_progress="hidden")
        c['propagate_masks_button'].click(fn=propagation_handler, inputs=prop_inputs, outputs=all_outputs, show_progress="hidden").then(lambda p: gr.update(selected=3) if p else gr.update(), c['analysis_output_dir_state'], c['main_tabs'])
        analysis_inputs = [c['scenes_state']] + self.ana_input_components
        c['start_analysis_button'].click(fn=analysis_handler, inputs=analysis_inputs, outputs=all_outputs, show_progress="hidden").then(lambda p: gr.update(selected=4) if p else gr.update(), c['analysis_metadata_path_state'], c['main_tabs'])
        c['find_people_button'].click(self.on_find_people_from_video, inputs=self.ana_input_components, outputs=[c['discovered_people_group'], c['discovered_faces_gallery'], c['identity_confidence_slider'], c['discovered_faces_state']])
        c['identity_confidence_slider'].release(self.on_identity_confidence_change, inputs=[c['identity_confidence_slider'], c['discovered_faces_state']], outputs=[c['discovered_faces_gallery']])
        c['discovered_faces_gallery'].select(self.on_discovered_face_select, inputs=[c['discovered_faces_state'], c['identity_confidence_slider']] + self.ana_input_components, outputs=[c['face_ref_img_path_input'], c['face_ref_image']])

    def on_identity_confidence_change(self, confidence: float, all_faces: list) -> gr.update:
        """Handler for when the identity confidence slider is changed."""
        if not all_faces: return []
        from sklearn.cluster import DBSCAN
        embeddings = np.array([face['embedding'] for face in all_faces])
        eps = 1.0 - confidence
        clustering = DBSCAN(eps=eps, min_samples=2, metric="cosine").fit(embeddings)
        labels = clustering.labels_
        unique_labels = sorted(list(set(labels)))
        gallery_items = []
        self.gallery_to_cluster_map = {}
        gallery_idx = 0
        for label in unique_labels:
            if label == -1: continue
            self.gallery_to_cluster_map[gallery_idx] = label
            gallery_idx += 1
            cluster_faces = [all_faces[i] for i, l in enumerate(labels) if l == label]
            best_face = max(cluster_faces, key=lambda x: x['det_score'])
            thumb_rgb = self.thumbnail_manager.get(Path(best_face['thumb_path']))
            x1, y1, x2, y2 = best_face['bbox'].astype(int)
            face_crop = thumb_rgb[y1:y2, x1:x2]
            gallery_items.append((face_crop, f"Person {label}"))
        return gr.update(value=gallery_items)

    def on_discovered_face_select(self, all_faces: list, confidence: float, *args, evt: gr.EventData = None) -> tuple[str, Optional[np.ndarray]]:
        """Handler for when a face is selected from the discovered faces gallery."""
        if not all_faces or evt is None or evt.index is None: return "", None
        selected_person_label = self.gallery_to_cluster_map.get(evt.index)
        if selected_person_label is None: self.logger.error(f"Could not find cluster label for gallery index {evt.index}"); return "", None
        params = self._create_pre_analysis_event(*args)
        video_path = params.video_path
        from sklearn.cluster import DBSCAN
        embeddings = np.array([face['embedding'] for face in all_faces])
        eps = 1.0 - confidence
        clustering = DBSCAN(eps=eps, min_samples=2, metric="cosine").fit(embeddings)
        labels = clustering.labels_
        cluster_faces = [all_faces[i] for i, l in enumerate(labels) if l == selected_person_label]
        if not cluster_faces: return "", None
        best_face = max(cluster_faces, key=lambda x: x['det_score'])
        best_frame_num = best_face['frame_num']
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, best_frame_num)
        ret, frame = cap.read()
        cap.release()
        if not ret: return "", None
        x1, y1, x2, y2 = best_face['bbox'].astype(int)
        thumb_rgb = self.thumbnail_manager.get(Path(best_face['thumb_path']))
        h, w, _ = thumb_rgb.shape
        fh, fw, _ = frame.shape
        x1, y1, x2, y2 = int(x1 * fw/w), int(y1 * fh/h), int(x2 * fw/w), int(y2 * fh/h)
        face_crop = frame[y1:y2, x1:x2]
        face_crop_path = Path(params.output_folder) / "reference_face.png"
        cv2.imwrite(str(face_crop_path), face_crop)
        return str(face_crop_path), face_crop

    def on_find_people_from_video(self, *args) -> tuple[gr.update, list, float, list]:
        """Handler for the 'Find People From Video' button."""
        try:
            params = self._create_pre_analysis_event(*args)
            output_dir = Path(params.output_folder)
            if not output_dir.exists(): return gr.update(visible=False), [], 0.5, []
            from core.managers import initialize_analysis_models
            models = initialize_analysis_models(params, self.config, self.logger, self.model_registry)
            face_analyzer = models['face_analyzer']
            if not face_analyzer: self.logger.error("Face analyzer not available."); return gr.update(visible=False), [], 0.5, []
            from core.utils import create_frame_map
            frame_map = create_frame_map(output_dir, self.logger)
            if not frame_map: self.logger.error("Frame map not found."); return gr.update(visible=False), [], 0.5, []
            all_faces = []
            thumb_dir = output_dir / "thumbs"
            # Person detection using Face Analysis directly (InsightFace detects faces, which implies people)
            # The original code used `person_detector` (YOLO) then `face_analyzer`.
            # If we don't have YOLO, we rely on face_analyzer finding faces.
            # But wait, original code iterated `people = person_detector.detect_boxes(thumb_rgb)`.
            # Since we removed person_detector, we just run face_analyzer on the whole image.

            for frame_num, thumb_filename in frame_map.items():
                if frame_num % params.pre_sample_nth != 0: continue
                thumb_path = thumb_dir / thumb_filename
                thumb_rgb = self.thumbnail_manager.get(thumb_path)
                if thumb_rgb is None: continue

                thumb_bgr = cv2.cvtColor(thumb_rgb, cv2.COLOR_RGB2BGR)
                faces = face_analyzer.get(thumb_bgr)
                for face in faces:
                    all_faces.append({'frame_num': frame_num, 'bbox': face.bbox, 'embedding': face.normed_embedding, 'det_score': face.det_score, 'thumb_path': str(thumb_path)})
            if not all_faces: self.logger.warning("No faces found in the video."); return gr.update(visible=True), [], 0.5, []
            from sklearn.cluster import DBSCAN
            embeddings = np.array([face['embedding'] for face in all_faces])
            clustering = DBSCAN(eps=0.5, min_samples=2, metric="cosine").fit(embeddings)
            labels = clustering.labels_
            unique_labels = sorted(list(set(labels)))
            gallery_items = []
            self.gallery_to_cluster_map = {}
            gallery_idx = 0
            for label in unique_labels:
                if label == -1: continue
                self.gallery_to_cluster_map[gallery_idx] = label
                gallery_idx += 1
                cluster_faces = [all_faces[i] for i, l in enumerate(labels) if l == label]
                best_face = max(cluster_faces, key=lambda x: x['det_score'])
                thumb_rgb = self.thumbnail_manager.get(Path(best_face['thumb_path']))
                x1, y1, x2, y2 = best_face['bbox'].astype(int)
                face_crop = thumb_rgb[y1:y2, x1:x2]
                gallery_items.append((face_crop, f"Person {label}"))
            return gr.update(visible=True), gallery_items, 0.5, all_faces
        except Exception as e:
            self.logger.error(f"Error in on_find_people_from_video: {e}", exc_info=True)
            return gr.update(visible=False), [], 0.5, []

    def on_apply_bulk_scene_filters_extended(self, scenes: list, min_mask_area: float, min_face_sim: float, min_confidence: float, enable_face_filter: bool, output_folder: str, view: str) -> tuple:
        """Handler for applying bulk filters to the scene gallery."""
        if not scenes:
            status_text, button_update = get_scene_status_text([Scene(**s) for s in scenes] if scenes else [])
            return [], status_text, gr.update(), [], button_update, "/ 1 pages", 1
        self.logger.info("Applying bulk scene filters", extra={"min_mask_area": min_mask_area, "min_face_sim": min_face_sim, "min_confidence": min_confidence, "enable_face_filter": enable_face_filter})
        scenes_objs = [Scene(**s) for s in scenes]
        for scene in scenes_objs:
            if scene.manual_status_change: continue
            rejection_reasons = []
            seed_result = scene.seed_result or {}
            details = seed_result.get('details', {})
            seed_metrics = scene.seed_metrics or {}
            if details.get('mask_area_pct', 101.0) < min_mask_area: rejection_reasons.append("Min Seed Mask Area")
            if enable_face_filter and seed_metrics.get('best_face_sim', 1.01) < min_face_sim: rejection_reasons.append("Min Seed Face Sim")
            if seed_metrics.get('score', 101.0) < min_confidence: rejection_reasons.append("Min Seed Confidence")
            scene.rejection_reasons = rejection_reasons
            if rejection_reasons: scene.status = 'excluded'
            else: scene.status = 'included'

        save_scene_seeds(scenes_objs, output_folder, self.logger)
        scenes_dicts = [s.model_dump() for s in scenes_objs]
        gallery_items, new_index_map, total_pages = build_scene_gallery_items(scenes_dicts, view, output_folder, page_num=1)
        status_text, button_update = get_scene_status_text(scenes_objs)
        return scenes_dicts, status_text, gr.update(value=gallery_items), new_index_map, button_update, f"/ {total_pages} pages", 1

    def _setup_filtering_handlers(self):
        c = self.components
        slider_keys, slider_comps = sorted(c['metric_sliders'].keys()), [c['metric_sliders'][k] for k in sorted(c['metric_sliders'].keys())]
        fast_filter_inputs = [c['all_frames_data_state'], c['per_metric_values_state'], c['analysis_output_dir_state'], c['gallery_view_toggle'], c['show_mask_overlay_input'], c['overlay_alpha_slider'], c['require_face_match_input'], c['dedup_thresh_input'], c['dedup_method_input']] + slider_comps
        fast_filter_outputs = [c['filter_status_text'], c['results_gallery']]
        for control in (slider_comps + [c['dedup_thresh_input'], c['gallery_view_toggle'], c['show_mask_overlay_input'], c['overlay_alpha_slider'], c['require_face_match_input'], c['dedup_method_input']]):
            (control.release if hasattr(control, 'release') else control.input if hasattr(control, 'input') else control.change)(self.on_filters_changed_wrapper, fast_filter_inputs, fast_filter_outputs)

        load_outputs = ([c['all_frames_data_state'], c['per_metric_values_state'], c['filter_status_text'], c['results_gallery'], c['results_group'], c['export_group']] + [c['metric_plots'].get(k) for k in self.get_all_filter_keys() if c['metric_plots'].get(k)] + slider_comps + [c['require_face_match_input']] + [c['metric_accs'].get(k) for k in sorted(c['metric_accs'].keys()) if c['metric_accs'].get(k)])

        def load_and_trigger_update(output_dir):
            if not output_dir: return [gr.update()] * len(load_outputs)
            from core.filtering import load_and_prep_filter_data, build_all_metric_svgs
            all_frames, metric_values = load_and_prep_filter_data(output_dir, self.get_all_filter_keys, self.config)
            svgs = build_all_metric_svgs(metric_values, self.get_all_filter_keys, self.logger)
            updates = {c['all_frames_data_state']: all_frames, c['per_metric_values_state']: metric_values, c['results_group']: gr.update(visible=True), c['export_group']: gr.update(visible=True)}
            for k in self.get_all_filter_keys():
                acc = c['metric_accs'].get(k)
                plot_comp = c['metric_plots'].get(k)
                has_data = k in metric_values and metric_values.get(k)
                if acc: updates[acc] = gr.update(visible=has_data)
                if plot_comp and has_data: updates[plot_comp] = gr.update(value=svgs.get(k, ""))
            slider_values_dict = {key: c['metric_sliders'][key].value for key in slider_keys}
            slider_values_dict['enable_dedup'] = (c['dedup_method_input'].value != "None")
            filter_event = FilterEvent(all_frames_data=all_frames, per_metric_values=metric_values, output_dir=output_dir, gallery_view="Kept Frames", show_overlay=self.config.gradio_show_mask_overlay, overlay_alpha=self.config.gradio_overlay_alpha, require_face_match=c['require_face_match_input'].value, dedup_thresh=c['dedup_thresh_input'].value, slider_values=slider_values_dict, dedup_method=c['dedup_method_input'].value)
            filter_updates = on_filters_changed(filter_event, self.thumbnail_manager, self.config, self.logger)
            updates.update({c['filter_status_text']: filter_updates['filter_status_text'], c['results_gallery']: filter_updates['results_gallery']})
            final_updates_list = [updates.get(comp, gr.update()) for comp in load_outputs]
            return final_updates_list

        c['filtering_tab'].select(load_and_trigger_update, [c['analysis_output_dir_state']], load_outputs)
        export_inputs = [c['all_frames_data_state'], c['analysis_output_dir_state'], c['extracted_video_path_state'], c['enable_crop_input'], c['crop_ar_input'], c['crop_padding_input'], c['require_face_match_input'], c['dedup_thresh_input'], c['dedup_method_input']] + slider_comps
        c['export_button'].click(self.export_kept_frames_wrapper, export_inputs, c['unified_log'])
        c['dry_run_button'].click(self.dry_run_export_wrapper, export_inputs, c['unified_log'])

        reset_outputs_comps = (slider_comps + [c['dedup_thresh_input'], c['require_face_match_input'], c['filter_status_text'], c['results_gallery']] + [c['metric_accs'][k] for k in sorted(c['metric_accs'].keys())] + [c['dedup_method_input']])
        c['reset_filters_button'].click(self.on_reset_filters, [c['all_frames_data_state'], c['per_metric_values_state'], c['analysis_output_dir_state']], reset_outputs_comps)

        auto_threshold_checkboxes = [c['metric_auto_threshold_cbs'][k] for k in sorted(c['metric_auto_threshold_cbs'].keys())]
        auto_set_inputs = [c['per_metric_values_state'], c['auto_pctl_input']] + auto_threshold_checkboxes
        c['apply_auto_button'].click(self.on_auto_set_thresholds, auto_set_inputs, [c['metric_sliders'][k] for k in slider_keys]).then(self.on_filters_changed_wrapper, fast_filter_inputs, fast_filter_outputs)

        all_accordions = list(c['metric_accs'].values())
        c['expand_all_metrics_button'].click(lambda: {acc: gr.update(open=True) for acc in all_accordions}, [], all_accordions)
        c['collapse_all_metrics_button'].click(lambda: {acc: gr.update(open=False) for acc in all_accordions}, [], all_accordions)

        c['dedup_method_input'].change(lambda method: {c['dedup_thresh_input']: gr.update(visible=method == 'pHash', label=f"{method} Threshold"), c['ssim_threshold_input']: gr.update(visible=method == 'SSIM'), c['lpips_threshold_input']: gr.update(visible=method == 'LPIPS')}, c['dedup_method_input'], [c['dedup_thresh_input'], c['ssim_threshold_input'], c['lpips_threshold_input']])
        c['dedup_visual_diff_input'].change(lambda x: {c['visual_diff_image']: gr.update(visible=x), c['calculate_diff_button']: gr.update(visible=x)}, c['dedup_visual_diff_input'], [c['visual_diff_image'], c['calculate_diff_button']])
        c['calculate_diff_button'].click(self.calculate_visual_diff, [c['results_gallery'], c['all_frames_data_state'], c['dedup_method_input'], c['dedup_thresh_input'], c['ssim_threshold_input'], c['lpips_threshold_input']], [c['visual_diff_image']])
        c['filter_preset_dropdown'].change(self.on_preset_changed, [c['filter_preset_dropdown']], list(c['metric_sliders'].values())).then(self.on_filters_changed_wrapper, fast_filter_inputs, fast_filter_outputs)

    def on_preset_changed(self, preset_name: str) -> dict:
        """Applies a filter preset by updating the values of the metric sliders."""
        updates = {}
        slider_keys = sorted(self.components['metric_sliders'].keys())
        if preset_name == "None" or preset_name not in self.FILTER_PRESETS:
            for key in slider_keys:
                metric_key = re.sub(r'_(min|max)$', '', key)
                default_key = 'default_max' if key.endswith('_max') else 'default_min'
                f_def = getattr(self.config, f"filter_default_{metric_key}", {})
                default_val = f_def.get(default_key, 0)
                updates[self.components['metric_sliders'][key]] = gr.update(value=default_val)
            return updates
        preset = self.FILTER_PRESETS[preset_name]
        for key in slider_keys:
            if key in preset: updates[self.components['metric_sliders'][key]] = gr.update(value=preset[key])
            else:
                metric_key = re.sub(r'_(min|max)$', '', key)
                default_key = 'default_max' if key.endswith('_max') else 'default_min'
                f_def = getattr(self.config, f"filter_default_{metric_key}", {})
                default_val = f_def.get(default_key, 0)
                updates[self.components['metric_sliders'][key]] = gr.update(value=default_val)
        return updates

    def on_filters_changed_wrapper(self, all_frames_data: list, per_metric_values: dict, output_dir: str, gallery_view: str, show_overlay: bool, overlay_alpha: float, require_face_match: bool, dedup_thresh: int, dedup_method: str, *slider_values: float) -> tuple[str, gr.update]:
        """Wrapper for the `on_filters_changed` event handler."""
        slider_values_dict = {k: v for k, v in zip(sorted(self.components['metric_sliders'].keys()), slider_values)}
        enable_dedup = dedup_method != "None"
        event_filters = slider_values_dict
        event_filters['enable_dedup'] = enable_dedup
        event_filters['dedup_method'] = dedup_method
        result = on_filters_changed(FilterEvent(all_frames_data=all_frames_data, per_metric_values=per_metric_values, output_dir=output_dir, gallery_view=gallery_view, show_overlay=show_overlay, overlay_alpha=overlay_alpha, require_face_match=require_face_match, dedup_thresh=dedup_thresh, slider_values=slider_values_dict, dedup_method=dedup_method), self.thumbnail_manager, self.config, self.logger)
        return result['filter_status_text'], result['results_gallery']

    def calculate_visual_diff(self, gallery: gr.Gallery, all_frames_data: list, dedup_method: str, dedup_thresh: int, ssim_thresh: float, lpips_thresh: float) -> Optional[np.ndarray]:
        """Calculates and displays a visual diff for a selected duplicate frame."""
        if not gallery or not gallery.selection: return None
        selected_image_index = gallery.selection['index']
        selected_frame_data = all_frames_data[selected_image_index]
        duplicate_frame_data = None
        import imagehash
        from skimage.metrics import structural_similarity as ssim
        from torchvision import transforms
        import lpips

        for frame_data in all_frames_data:
            if frame_data['filename'] == selected_frame_data['filename']: continue
            if dedup_method == "pHash":
                hash1 = imagehash.hex_to_hash(selected_frame_data['phash'])
                hash2 = imagehash.hex_to_hash(frame_data['phash'])
                if hash1 - hash2 <= dedup_thresh: duplicate_frame_data = frame_data; break
            elif dedup_method == "SSIM":
                img1 = self.thumbnail_manager.get(Path(self.config.paths.downloads) / Path(selected_frame_data['filename']).parent.name / "thumbs" / selected_frame_data['filename'])
                img2 = self.thumbnail_manager.get(Path(self.config.paths.downloads) / Path(frame_data['filename']).parent.name / "thumbs" / frame_data['filename'])
                if img1 is not None and img2 is not None:
                    img1_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
                    img2_gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
                    similarity = ssim(img1_gray, img2_gray)
                    if similarity >= ssim_thresh: duplicate_frame_data = frame_data; break
            elif dedup_method == "LPIPS":
                loss_fn = lpips.LPIPS(net='alex')
                transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
                img1 = self.thumbnail_manager.get(Path(self.config.paths.downloads) / Path(selected_frame_data['filename']).parent.name / "thumbs" / selected_frame_data['filename'])
                img2 = self.thumbnail_manager.get(Path(self.config.paths.downloads) / Path(frame_data['filename']).parent.name / "thumbs" / frame_data['filename'])
                if img1 is not None and img2 is not None:
                    img1_t = transform(img1).unsqueeze(0)
                    img2_t = transform(img2).unsqueeze(0)
                    distance = loss_fn.forward(img1_t, img2_t).item()
                    if distance <= lpips_thresh: duplicate_frame_data = frame_data; break
        if duplicate_frame_data:
            img1 = self.thumbnail_manager.get(Path(self.config.paths.downloads) / Path(selected_frame_data['filename']).parent.name / "thumbs" / selected_frame_data['filename'])
            img2 = self.thumbnail_manager.get(Path(self.config.paths.downloads) / Path(duplicate_frame_data['filename']).parent.name / "thumbs" / duplicate_frame_data['filename'])
            if img1 is not None and img2 is not None:
                h, w, _ = img1.shape
                comparison_image = np.zeros((h, w * 2, 3), dtype=np.uint8)
                comparison_image[:, :w] = img1
                comparison_image[:, w:] = img2
                return comparison_image
        return None

    def on_reset_filters(self, all_frames_data: list, per_metric_values: dict, output_dir: str) -> tuple:
        """Handler for the 'Reset Filters' button."""
        c = self.components
        slider_keys = sorted(c['metric_sliders'].keys())
        acc_keys = sorted(c['metric_accs'].keys())
        slider_default_values = []
        slider_updates = []
        for key in slider_keys:
            metric_key = re.sub(r'_(min|max)$', '', key)
            default_key = 'default_max' if key.endswith('_max') else 'default_min'
            f_def = getattr(self.config, f"filter_default_{metric_key}", {})
            default_val = f_def.get(default_key, 0)
            slider_updates.append(gr.update(value=default_val))
            slider_default_values.append(default_val)
        face_match_default = self.config.default_require_face_match
        dedup_default = self.config.filter_default_dedup_thresh['default']
        dedup_update = gr.update(value=dedup_default)
        face_match_update = gr.update(value=face_match_default)
        if all_frames_data:
            slider_defaults_dict = {key: val for key, val in zip(slider_keys, slider_default_values)}
            filter_event = FilterEvent(all_frames_data=all_frames_data, per_metric_values=per_metric_values, output_dir=output_dir, gallery_view="Kept", show_overlay=self.config.gradio_show_mask_overlay, overlay_alpha=self.config.gradio_overlay_alpha, require_face_match=face_match_default, dedup_thresh=dedup_default, dedup_method="pHash", slider_values=slider_defaults_dict)
            from ui.gallery_utils import on_filters_changed
            filter_updates = on_filters_changed(filter_event, self.thumbnail_manager, self.config, self.logger)
            status_update = filter_updates['filter_status_text']
            gallery_update = filter_updates['results_gallery']
        else:
            status_update = "Load an analysis to begin."
            gallery_update = gr.update(value=[])
        acc_updates = []
        for key in acc_keys:
            if all_frames_data:
                visible = False
                if key == 'dedup': visible = any('phash' in f for f in all_frames_data)
                elif key == 'face_sim': visible = 'face_sim' in per_metric_values and any(per_metric_values['face_sim'])
                else: visible = key in per_metric_values
                preferred_open = next((candidate for candidate in ['quality_score', 'sharpness'] if candidate in per_metric_values), None)
                acc_updates.append(gr.update(visible=visible, open=(key == preferred_open)))
            else: acc_updates.append(gr.update(visible=False))
        dedup_method_update = gr.update(value="pHash")
        return tuple(slider_updates + [dedup_update, face_match_update, status_update, gallery_update] + acc_updates + [dedup_method_update])

    def on_auto_set_thresholds(self, per_metric_values: dict, p: int, *checkbox_values: bool) -> list[gr.update]:
        """Handler for the 'Apply Percentile to Mins' button."""
        slider_keys = sorted(self.components['metric_sliders'].keys())
        auto_threshold_cbs_keys = sorted(self.components['metric_auto_threshold_cbs'].keys())
        selected_metrics = [metric_name for metric_name, is_selected in zip(auto_threshold_cbs_keys, checkbox_values) if is_selected]
        updates = auto_set_thresholds(per_metric_values, p, slider_keys, selected_metrics)
        return [updates.get(f'slider_{key}', gr.update()) for key in slider_keys]

    def export_kept_frames_wrapper(self, all_frames_data: list, output_dir: str, video_path: str, enable_crop: bool, crop_ars: str, crop_padding: int, require_face_match: bool, dedup_thresh: int, dedup_method: str, *slider_values: float) -> str:
        """Wrapper for the export function."""
        filter_args = {k: v for k, v in zip(sorted(self.components['metric_sliders'].keys()), slider_values)}
        enable_dedup = dedup_method != "None"
        filter_args.update({"require_face_match": require_face_match, "dedup_thresh": dedup_thresh, "dedup_method": dedup_method, "enable_dedup": enable_dedup})
        return export_kept_frames(ExportEvent(all_frames_data=all_frames_data, output_dir=output_dir, video_path=video_path, enable_crop=enable_crop, crop_ars=crop_ars, crop_padding=crop_padding, filter_args=filter_args), self.config, self.logger, self.thumbnail_manager, self.cancel_event)

    def dry_run_export_wrapper(self, all_frames_data: list, output_dir: str, video_path: str, enable_crop: bool, crop_ars: str, crop_padding: int, require_face_match: bool, dedup_thresh: int, dedup_method: str, *slider_values: float) -> str:
        """Wrapper for the dry run export function."""
        filter_args = {k: v for k, v in zip(sorted(self.components['metric_sliders'].keys()), slider_values)}
        filter_args.update({"require_face_match": require_face_match, "dedup_thresh": dedup_thresh, "enable_dedup": dedup_method != "None"})
        return dry_run_export(ExportEvent(all_frames_data=all_frames_data, output_dir=output_dir, video_path=video_path, enable_crop=enable_crop, crop_ars=crop_ars, crop_padding=crop_padding, filter_args=filter_args), self.config)
