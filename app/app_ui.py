import gradio as gr
import torch
import threading
import time
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor
from app.config import Config
from app.logging import UnifiedLogger
from app.thumb_cache import ThumbnailManager
from app.events import (ExtractionEvent, PreAnalysisEvent, PropagationEvent,
                              FilterEvent, ExportEvent, SessionLoadEvent)
from app.pipeline_logic import (
    execute_extraction,
    execute_pre_analysis,
    execute_propagation,
    execute_session_load
)
from app.filter_logic import (load_and_prep_filter_data, build_all_metric_svgs,
                                    on_filters_changed, reset_filters,
                                    auto_set_thresholds, apply_all_filters_vectorized)
from app.scene_logic import (toggle_scene_status, apply_bulk_scene_filters,
                                   apply_scene_overrides, get_scene_status_text,
                                   save_scene_seeds)


class AppUI:
    """Main Gradio UI class for the frame extractor application."""
    
    def __init__(self, config=None, logger=None, progress_queue=None, 
                 cancel_event=None):
        self.config = config or Config()
        self.logger = logger or UnifiedLogger()
        self.progress_queue = progress_queue or Queue()
        self.cancel_event = cancel_event or threading.Event()
        
        self.components = {}
        self.cuda_available = torch.cuda.is_available()
        self.thumbnail_manager = ThumbnailManager(
            max_size=self.config.thumbnail_cache_size
        )
        
        self.ext_ui_map_keys = [
            'source_path', 'upload_video', 'method', 'interval', 'nth_frame',
            'fast_scene', 'max_resolution', 'use_png', 'thumbnails_only',
            'thumb_megapixels', 'scene_detect'
        ]
        self.ana_ui_map_keys = [
            'output_folder', 'video_path', 'resume', 'enable_face_filter',
            'face_ref_img_path', 'face_ref_img_upload', 'face_model_name',
            'enable_subject_mask', 'dam4sam_model_name', 'person_detector_model',
            'seed_strategy', 'scene_detect', 'enable_dedup', 'text_prompt',
            'box_threshold', 'text_threshold', 'min_mask_area_pct',
            'sharpness_base_scale', 'edge_strength_base_scale',
            'gdino_config_path', 'gdino_checkpoint_path',
            'pre_analysis_enabled', 'pre_sample_nth', 'primary_seed_strategy'
        ]

        self.session_load_keys = [
            'unified_log', 'unified_status', 'progress_bar',
            # Extraction Tab
            'source_input', 'max_resolution', 'thumbnails_only_input',
            'thumb_megapixels_input', 'ext_scene_detect_input', 'method_input',
            'use_png_input',
            # Analysis Tab
            'pre_analysis_enabled_input', 'pre_sample_nth_input',
            'enable_face_filter_input', 'face_model_name_input',
            'face_ref_img_path_input', 'text_prompt_input',
            'seed_strategy_input', 'person_detector_model_input',
            'dam4sam_model_name_input', 'enable_dedup_input',
            # States
            'extracted_video_path_state', 'extracted_frames_dir_state',
            'analysis_output_dir_state', 'analysis_metadata_path_state',
            'scenes_state',
            # Other UI elements
            'propagate_masks_button', 'filtering_tab',
            'scene_face_sim_min_input'
        ]

    def build_ui(self):
        """Build the complete Gradio interface."""
        css = """.plot-and-slider-column { max-width: 560px !important; margin: auto; } .scene-editor { border: 1px solid #444; padding: 10px; border-radius: 5px; }"""
        
        with gr.Blocks(theme=gr.themes.Default(), css=css) as demo:
            gr.Markdown("# 🎬 Frame Extractor & Analyzer v2.0")
            if not self.cuda_available:
                gr.Markdown("⚠️ **CPU Mode** — GPU-dependent features are "
                          "disabled or will be slow.")

            with gr.Accordion("🔄 resume previous Session", open=False):
                with gr.Row():
                    self._create_component('session_path_input', 'textbox', {
                        'label': "Load previous run",
                        'placeholder': "Path to a previous run's output folder..."
                    })
                    self._create_component('load_session_button', 'button', {
                        'value': "📂 Load Session"
                    })

            with gr.Tabs() as main_tabs:
                self.components['main_tabs'] = main_tabs
                with gr.Tab("📹 1. Frame Extraction"):
                    self._create_extraction_tab()
                with gr.Tab("🎯 2. Seeding & Scene Selection", id=1) as analysis_tab:
                    self.components['analysis_tab'] = analysis_tab
                    self._create_analysis_tab()
                with gr.Tab("📊 3. Filtering & Export", id=2) as filtering_tab:
                    self.components['filtering_tab'] = filtering_tab
                    self._create_filtering_tab()

            with gr.Row():
                with gr.Column(scale=2):
                    self._create_component('unified_log', 'textbox', {
                        'label': "📋 Processing Log", 'lines': 10, 
                        'interactive': False, 'autoscroll': True
                    })
                with gr.Column(scale=1):
                    self._create_component('unified_status', 'textbox', {
                        'label': "📊 Status Summary", 'lines': 2,
                        'interactive': False
                    })
                    with gr.Row():
                        self.components['progress_bar'] = gr.Progress()
            self._create_event_handlers()
        return demo

    def _create_component(self, name, comp_type, kwargs):
        """Create and register a Gradio component."""
        comp_map = {
            'button': gr.Button, 'textbox': gr.Textbox, 
            'dropdown': gr.Dropdown, 'slider': gr.Slider,
            'checkbox': gr.Checkbox, 'file': gr.File, 'radio': gr.Radio, 
            'gallery': gr.Gallery, 'plot': gr.Plot, 'markdown': gr.Markdown, 
            'html': gr.HTML, 'number': gr.Number
        }
        self.components[name] = comp_map[comp_type](**kwargs)
        return self.components[name]

    def _create_extraction_tab(self):
        """Create the frame extraction tab."""
        gr.Markdown("### Step 1: Provide a Video Source")
        with gr.Row():
            with gr.Column(scale=2):
                self._create_component('source_input', 'textbox', {
                    'label': "Video URL or Local Path",
                    'placeholder': "Enter YouTube URL or local video file path"
                })
            with gr.Column(scale=1):
                self._create_component('max_resolution', 'dropdown', {
                    'choices': ["maximum available", "2160", "1080", "720"],
                    'value': self.config.ui_defaults['max_resolution'],
                    'label': "Download Resolution"
                })
        self._create_component('upload_video_input', 'file', {
            'label': "Or Upload a Video File", 'file_types': ["video"], 'type': "filepath"
        })

        gr.Markdown("---")
        gr.Markdown("### Step 2: Configure Extraction Method")
        self._create_component('thumbnails_only_input', 'checkbox', {
            'label': "Use Recommended Thumbnail Extraction (Faster, For Pre-Analysis)",
            'value': self.config.ui_defaults['thumbnails_only']
        })

        with gr.Accordion("Thumbnail Settings", open=True) as recommended_accordion:
            self.components['recommended_accordion'] = recommended_accordion
            gr.Markdown("This is the fastest and most efficient method. It extracts lightweight thumbnails for scene analysis, allowing you to quickly find the best frames *before* extracting full-resolution images.")
            self._create_component('thumb_megapixels_input', 'slider', {
                'label': "Thumbnail Size (MP)", 'minimum': 0.1, 'maximum': 2.0, 'step': 0.1,
                'value': self.config.ui_defaults['thumb_megapixels']
            })
            self._create_component('ext_scene_detect_input', 'checkbox', {
                'label': "Use Scene Detection (Recommended)",
                'value': self.config.ui_defaults['scene_detect']
            })

        with gr.Accordion("Advanced: Legacy Full-Frame Extraction", open=False) as legacy_accordion:
            self.components['legacy_accordion'] = legacy_accordion
            gr.Markdown("This method extracts full-resolution frames directly, which can be slow and generate a large number of files. Use this only if you have specific needs and understand the performance implications.")
            method_choices = ["keyframes", "interval", "every_nth_frame", "all", "scene"]
            self._create_component('method_input', 'dropdown', {
                'choices': method_choices, 'value': self.config.ui_defaults['method'], 'label': "Extraction Method"
            })
            self._create_component('interval_input', 'textbox', {
                'label': "Interval (seconds)", 'value': self.config.ui_defaults['interval'], 'visible': False
            })
            self._create_component('nth_frame_input', 'textbox', {
                'label': "N-th Frame Value", 'value': self.config.ui_defaults["nth_frame"], 'visible': False
            })
            self._create_component('fast_scene_input', 'checkbox', {
                'label': "Fast Scene Detect (for 'scene' method)", 'visible': False
            })
            self._create_component('use_png_input', 'checkbox', {
                'label': "Save as PNG (slower, larger files)", 'value': self.config.ui_defaults['use_png']
            })

        gr.Markdown("---")
        gr.Markdown("### Step 3: Start Extraction")
        start_btn = gr.Button("🚀 Start Extraction", variant="primary")
        self.components.update({'start_extraction_button': start_btn})

    def _create_analysis_tab(self):
        """Create the analysis and seeding tab."""
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 🎯 Step 1: Choose Your Seeding Strategy")
                self._create_component('primary_seed_strategy_input', 'radio', {
                    'choices': ["👤 By Face", "📝 By Text", "🔄 Face + Text Fallback", "🤖 Automatic"],
                    'value': "🔄 Face + Text Fallback",  # Make fallback the default
                    'label': "Primary Seeding Strategy"
                })

                # --- Face Seeding Group ---
                with gr.Group(visible=False) as face_seeding_group:
                    self.components['face_seeding_group'] = face_seeding_group
                    gr.Markdown("#### 👤 Configure Face Seeding")
                    gr.Markdown("Upload a clear image of the person you want to find. The system will search for this person in the video.")
                    self._create_component('face_ref_img_upload_input', 'file', {
                        'label': "Upload Face Reference Image", 'type': "filepath"
                    })
                    self._create_component('face_ref_img_path_input', 'textbox', {
                        'label': "Or provide a local file path"
                    })
                    self._create_component('enable_face_filter_input', 'checkbox', {
                        'label': "Enable Face Similarity (must be checked for face seeding)",
                        'value': False, 'interactive': False
                    })

                # --- Text Seeding Group ---
                with gr.Group(visible=False) as text_seeding_group:
                    self.components['text_seeding_group'] = text_seeding_group
                    gr.Markdown("#### 📝 Configure Text Seeding")
                    gr.Markdown("Describe the subject or object you want to find. Be as specific as possible for better results.")
                    self._create_component('text_prompt_input', 'textbox', {
                        'label': "Text Prompt",
                        'placeholder': "e.g., 'a woman in a red dress'",
                        'value': self.config.ui_defaults['text_prompt']
                    })

                # --- Automatic Seeding Group ---
                with gr.Group(visible=True) as auto_seeding_group:
                    self.components['auto_seeding_group'] = auto_seeding_group
                    gr.Markdown("#### 🤖 Configure Automatic Seeding")
                    gr.Markdown("The system will automatically identify the most prominent person in each scene. This is a good general-purpose starting point.")
                    self._create_component('seed_strategy_input', 'dropdown', {
                        'choices': ["Largest Person", "Center-most Person"],
                        'value': "Largest Person",
                        'label': "Automatic Seeding Method"
                    })

                with gr.Accordion("Advanced Settings", open=False):
                    gr.Markdown("These settings control the underlying models and analysis parameters. Adjust them only if you understand their effect.")
                    self._create_component('pre_analysis_enabled_input', 'checkbox', {
                        'label': 'Enable Pre-Analysis to find best seed frame',
                        'value': self.config.ui_defaults['pre_analysis_enabled']
                    })
                    self._create_component('pre_sample_nth_input', 'number', {
                        'label': 'Sample every Nth thumbnail for pre-analysis',
                        'value': self.config.ui_defaults['pre_sample_nth'],
                        'interactive': True
                    })
                    self._create_component('person_detector_model_input', 'dropdown', {
                        'choices': ['yolo11x.pt', 'yolo11s.pt'],
                        'value': self.config.ui_defaults['person_detector_model'],
                        'label': "Person Detector (for Automatic)"
                    })
                    self._create_component('face_model_name_input', 'dropdown', {
                        'choices': ["buffalo_l", "buffalo_s"],
                        'value': self.config.ui_defaults['face_model_name'],
                        'label': "Face Model (for Face Seeding)"
                    })
                    self._create_component('dam4sam_model_name_input', 'dropdown', {
                        'choices': ["sam21pp-T", "sam21pp-S", "sam21pp-B+", "sam21pp-L"],
                        'value': self.config.ui_defaults['dam4sam_model_name'],
                        'label': "SAM Tracker Model"
                    })
                    self._create_component('enable_dedup_input', 'checkbox', {
                        'label': "Enable Deduplication (pHash)",
                        'value': self.config.ui_defaults.get('enable_dedup', False)
                    })

                self._create_component('start_pre_analysis_button', 'button', {
                    'value': '🌱 Find & Preview Scene Seeds', 'variant': 'primary'
                })

                with gr.Group(visible=False) as propagation_group:
                    self.components['propagation_group'] = propagation_group
                    gr.Markdown("---")
                    gr.Markdown("### 🔬 Step 3: Propagate Masks")
                    gr.Markdown("Once you are satisfied with the seeds, propagate the masks to the rest of the frames in the selected scenes.")

                    self._create_component('propagate_masks_button', 'button', {
                        'value': '🔬 Propagate Masks on Kept Scenes',
                        'variant': 'primary',
                        'interactive': False
                    })

            with gr.Column(scale=2, visible=False) as seeding_results_column:
                self.components['seeding_results_column'] = seeding_results_column
                gr.Markdown("### 🎭 Step 2: Review & Refine Seeds")
                self._create_component('seeding_preview_gallery', 'gallery', {
                    'label': 'Scene Seed Previews',
                    'columns': [4, 6, 8],
                    'rows': 2,
                    'height': 'auto',
                    'preview': True,
                    'allow_preview': True,
                    'object_fit': 'contain'
                })
                
                with gr.Accordion("Scene Editor", open=False, 
                                elem_classes="scene-editor") as scene_editor_accordion:
                    self.components['scene_editor_accordion'] = scene_editor_accordion
                    self._create_component('scene_editor_status_md', 'markdown', {
                        'value': "Select a scene to edit."
                    })
                    with gr.Row():
                        self._create_component('scene_editor_prompt_input', 
                                             'textbox', {
                            'label': 'Per-Scene Text Prompt'
                        })
                    with gr.Row():
                        self._create_component('scene_editor_box_thresh_input', 
                                             'slider', {
                            'label': "Box Thresh", 'minimum': 0.0, 'maximum': 1.0,
                            'step': 0.05,
                            'value': self.config.grounding_dino_params['box_threshold']
                        })
                        self._create_component('scene_editor_text_thresh_input',
                                             'slider', {
                            'label': "Text Thresh", 'minimum': 0.0, 'maximum': 1.0,
                            'step': 0.05,
                            'value': self.config.grounding_dino_params['text_threshold']
                        })
                    with gr.Row():
                        self._create_component('scene_recompute_button', 'button', {
                            'value': '🔄 Recompute Preview'
                        })
                        self._create_component('scene_include_button', 'button', {
                            'value': '👍 Include'
                        })
                        self._create_component('scene_exclude_button', 'button', {
                            'value': '👎 Exclude'
                        })
                        
                with gr.Accordion("Bulk Scene Actions & Filters", open=True):
                    self._create_component('scene_filter_status', 'markdown', {
                        'value': 'No scenes loaded.'
                    })
                    self._create_component('scene_mask_area_min_input', 'slider', {
                        'label': "Min Seed Mask Area %", 'minimum': 0.0,
                        'maximum': 100.0, 'value': self.config.min_mask_area_pct,
                        'step': 0.1
                    })
                    self._create_component('scene_face_sim_min_input', 'slider', {
                        'label': "Min Seed Face Sim", 'minimum': 0.0,
                        'maximum': 1.0, 'value': 0.5, 'step': 0.05,
                        'visible': False
                    })
                    self._create_component('scene_confidence_min_input', 'slider', {
                        'label': "Min Seed Confidence", 'minimum': 0.0,
                        'maximum': 1.0, 'value': 0.0, 'step': 0.05
                    })
                    with gr.Row():
                        self._create_component('bulk_include_all_button', 'button', {
                            'value': 'Include All'
                        })
                        self._create_component('bulk_exclude_all_button', 'button', {
                            'value': 'Exclude All'
                        })

    def _create_filtering_tab(self):
        """Create the filtering and export tab."""
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 🎛️ Filter Controls")
                self._create_component('auto_pctl_input', 'slider', {
                    'label': 'Auto-Threshold Percentile', 'minimum': 1,
                    'maximum': 99, 'value': 75, 'step': 1
                })
                with gr.Row():
                    self._create_component('apply_auto_button', 'button', {
                        'value': 'Apply Percentile to Mins'
                    })
                    self._create_component('reset_filters_button', 'button', {
                        'value': "Reset Filters"
                    })
                self._create_component('filter_status_text', 'markdown', {
                    'value': "Load an analysis to begin."
                })

                self.components['metric_plots'] = {}
                self.components['metric_sliders'] = {}
                
                with gr.Accordion("Deduplication", open=True, visible=True):
                    f_def = self.config.filter_defaults['dedup_thresh']
                    self._create_component('dedup_thresh_input', 'slider', {
                        'label': "Similarity Threshold", 'minimum': f_def['min'],
                        'maximum': f_def['max'], 'value': f_def['default'],
                        'step': f_def['step']
                    })

                filter_display_order = [
                    ('niqe', True), ('sharpness', True), ('edge_strength', True),
                    ('contrast', True), ('brightness', False), ('entropy', False),
                    ('face_sim', False), ('mask_area_pct', False)
                ]

                for metric_name, open_default in filter_display_order:
                    if metric_name not in self.config.filter_defaults:
                        continue
                    f_def = self.config.filter_defaults[metric_name]
                    with gr.Accordion(metric_name.replace('_', ' ').title(), 
                                    open=open_default):
                        with gr.Column(elem_classes="plot-and-slider-column"):
                            plot_name = f'plot_{metric_name}'
                            self.components['metric_plots'][metric_name] = (
                                self._create_component(plot_name, 'html', {
                                    'visible': False
                                })
                            )
                            
                            min_slider_name = f'slider_{metric_name}_min'
                            self.components['metric_sliders'][f"{metric_name}_min"] = (
                                self._create_component(min_slider_name, 'slider', {
                                    'label': "Min", 'minimum': f_def['min'],
                                    'maximum': f_def['max'],
                                    'value': f_def['default_min'],
                                    'step': f_def['step'], 'interactive': True,
                                    'visible': False
                                })
                            )
                            
                            if 'default_max' in f_def:
                                max_slider_name = f'slider_{metric_name}_max'
                                self.components['metric_sliders'][f"{metric_name}_max"] = (
                                    self._create_component(max_slider_name, 'slider', {
                                        'label': "Max", 'minimum': f_def['min'],
                                        'maximum': f_def['max'],
                                        'value': f_def['default_max'],
                                        'step': f_def['step'], 'interactive': True,
                                        'visible': False
                                    })
                                )
                                
                            if metric_name == "face_sim":
                                self._create_component('require_face_match_input', 
                                                     'checkbox', {
                                    'label': "Reject if no face",
                                    'value': self.config.ui_defaults['require_face_match'],
                                    'visible': False
                                })

            with gr.Column(scale=2):
                with gr.Group(visible=False) as results_group:
                    self.components['results_group'] = results_group
                    gr.Markdown("### 🖼️ Step 2: Review Results")
                    with gr.Row():
                        self._create_component('gallery_view_toggle', 'radio', {
                            'choices': ["Kept Frames", "Rejected Frames"],
                            'value': "Kept Frames",
                            'label': "Show in Gallery"
                        })
                        self._create_component('show_mask_overlay_input', 'checkbox', {
                            'label': "Show Mask Overlay",
                            'value': True
                        })
                        self._create_component('overlay_alpha_slider', 'slider', {
                            'label': "Overlay Alpha", 'minimum': 0.0, 'maximum': 1.0,
                            'value': 0.6, 'step': 0.1
                        })
                    self._create_component('results_gallery', 'gallery', {
                        'columns': [4, 6, 8], 'rows': 2, 'height': 'auto',
                        'preview': True, 'allow_preview': True, 'object_fit': 'contain'
                    })

                with gr.Group(visible=False) as export_group:
                    self.components['export_group'] = export_group
                    gr.Markdown("### 📤 Step 3: Export")
                    self._create_component('export_button', 'button', {
                        'value': "Export Kept Frames",
                        'variant': "primary"
                    })
                    with gr.Accordion("Export Options", open=True):
                        with gr.Row():
                            self._create_component('enable_crop_input', 'checkbox', {
                                'label': "✂️ Crop to Subject",
                                'value': True
                            })
                            self._create_component('crop_padding_input', 'slider', {
                                'label': "Padding %",
                                'value': 1
                            })
                        self._create_component('crop_ar_input', 'textbox', {
                            'label': "Crop ARs",
                            'value': "16:9,1:1,9:16",
                            'info': "Comma-separated list (e.g., 16:9, 1:1). The best-fitting AR for each subject's mask will be chosen automatically."
                        })

    def get_all_filter_keys(self):
        """Get all available filter keys."""
        return self.config.QUALITY_METRICS + ["face_sim", "mask_area_pct"]

    def _create_event_handlers(self):
        """Set up all UI event handlers."""
        self.components.update({
            'extracted_video_path_state': gr.State(""),
            'extracted_frames_dir_state': gr.State(""),
            'analysis_output_dir_state': gr.State(""),
            'analysis_metadata_path_state': gr.State(""),
            'all_frames_data_state': gr.State([]),
            'per_metric_values_state': gr.State({}),
            'scenes_state': gr.State([]),
            'selected_scene_id_state': gr.State(None)
        })
        self._setup_visibility_toggles()
        self._setup_pipeline_handlers()
        self._setup_filtering_handlers()
        self._setup_scene_editor_handlers()
        self._setup_bulk_scene_handlers()

    def _run_task_with_progress(self, task_func, output_keys, progress, *args):
        """
        Runs a long-running task in a background thread and updates the UI
        with progress via the provided Gradio progress object.
        """
        self.cancel_event.clear()
        log_buffer = []
        processed, total, stage = 0, 1, "Initializing"
        start_time = time.time()

        initial_updates = {'unified_log': "Starting task..."}
        yield tuple(initial_updates.get(k, gr.update()) for k in output_keys)
        progress(0, "Initializing...")

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(task_func, *args)

            while not future.done():
                if self.cancel_event.is_set():
                    future.cancel()
                    break

                try:
                    msg = self.progress_queue.get(timeout=0.1)
                    if "log" in msg:
                        log_buffer.append(f"[{time.strftime('%H:%M:%S')}] {msg['log']}")
                    if "stage" in msg:
                        stage, processed = msg["stage"], 0
                        start_time = time.time()
                    if "total" in msg: total = msg["total"] or 1
                    if "progress" in msg: processed += msg["progress"]
                except Empty:
                    pass

                elapsed = time.time() - start_time
                rate = processed / elapsed if elapsed > 0 else 0
                eta = (total - processed) / rate if rate > 0 else float('inf')
                eta_str = f"{int(eta//60):02d}:{int(eta%60):02d}" if eta != float('inf') else "∞"
                progress_pct = processed / total if total > 0 else 0

                progress(progress_pct, f"{stage} ({processed}/{total}) - ETA: {eta_str}")

                # Yield other UI updates without throttling
                progress_updates = {'unified_log': "\n".join(log_buffer)}
                yield tuple(progress_updates.get(k, gr.update()) for k in output_keys)
                time.sleep(0.1)

        final_updates = {}
        try:
            result_dict = future.result()
            if self.cancel_event.is_set():
                final_msg, final_label = "⏹️ Task cancelled by user", "Cancelled"
            else:
                final_msg = result_dict.get("log", "✅ Task completed successfully")
                final_label = "Complete"
                final_updates = result_dict
        except Exception as e:
            self.logger.error("Task execution failed", exc_info=True)
            final_msg, final_label = f"❌ Task failed: {e}", "Failed"

        log_buffer.append(final_msg)
        final_updates['unified_log'] = "\n".join(log_buffer)
        progress(1.0, final_label)

        yield tuple(final_updates.get(k, gr.update()) for k in output_keys)

    def run_extraction_wrapper(self, *args):
        """Wrapper for extraction pipeline. Returns a dict of final updates."""
        ui_args = dict(zip(self.ext_ui_map_keys, args))
        event = ExtractionEvent(**ui_args)

        final_result = {}
        for result in execute_extraction(event, self.progress_queue,
                                         self.cancel_event, self.logger, self.config):
            if isinstance(result, dict):
                final_result.update(result)

        if final_result.get("done"):
            video_path = final_result.get("extracted_video_path_state", "") or final_result.get("video_path", "")
            frames_dir = final_result.get("extracted_frames_dir_state", "") or final_result.get("output_dir", "")
            return {
                "log": final_result.get("log", "✅ Extraction completed successfully."),
                "extracted_video_path_state": video_path,
                "extracted_frames_dir_state": frames_dir,
                "main_tabs": gr.update(selected=1)
            }
        return {"log": final_result.get("log", "❌ Extraction failed or was cancelled.")}

    def run_pre_analysis_wrapper(self, *args):
        """Wrapper for pre-analysis pipeline. Returns a dict of final updates."""
        ui_args = dict(zip(self.ana_ui_map_keys, args))
        strategy = ui_args.pop('primary_seed_strategy', '🤖 Automatic')

        # Configure args based on strategy
        if strategy == "👤 By Face": ui_args.update({'enable_face_filter': True, 'text_prompt': ""})
        elif strategy == "📝 By Text": ui_args.update({'enable_face_filter': False, 'face_ref_img_path': "", 'face_ref_img_upload': None})
        elif strategy == "🔄 Face + Text Fallback": ui_args['enable_face_filter'] = True
        elif strategy == "🤖 Automatic": ui_args.update({'enable_face_filter': False, 'text_prompt': "", 'face_ref_img_path': "", 'face_ref_img_upload': None})

        event = PreAnalysisEvent(**ui_args)
        final_result = {}
        for result in execute_pre_analysis(event, self.progress_queue,
                                           self.cancel_event, self.logger,
                                           self.config, self.thumbnail_manager,
                                           self.cuda_available):
            if isinstance(result, dict):
                final_result.update(result)

        if final_result.get("done"):
            scenes = final_result.get('scenes', [])
            if scenes:
                save_scene_seeds(scenes, final_result['output_dir'], self.logger)
            status_text = get_scene_status_text(scenes)
            has_face_sim = any(s.get('seed_metrics', {}).get('best_face_sim') is not None for s in scenes)

            return {
                "log": final_result.get("log", "✅ Pre-analysis completed successfully."),
                "seeding_preview_gallery": gr.update(value=final_result.get('previews')),
                "scenes_state": scenes,
                "propagate_masks_button": gr.update(interactive=True),
                "scene_filter_status": status_text,
                "scene_face_sim_min_input": gr.update(visible=has_face_sim),
                "seeding_results_column": gr.update(visible=True),
                "propagation_group": gr.update(visible=True)
            }
        return {"log": final_result.get("log", "❌ Pre-analysis failed or was cancelled.")}

    def run_propagation_wrapper(self, scenes, *args):
        """Wrapper for propagation pipeline. Returns a dict of final updates."""
        ui_args = dict(zip(self.ana_ui_map_keys, args))
        strategy = ui_args.pop('primary_seed_strategy', '🤖 Automatic')

        # Configure args based on strategy
        if strategy == "👤 By Face": ui_args.update({'enable_face_filter': True, 'text_prompt': ""})
        elif strategy == "📝 By Text": ui_args.update({'enable_face_filter': False, 'face_ref_img_path': "", 'face_ref_img_upload': None})
        elif strategy == "🔄 Face + Text Fallback": ui_args['enable_face_filter'] = True
        elif strategy == "🤖 Automatic": ui_args.update({'enable_face_filter': False, 'text_prompt': "", 'face_ref_img_path': "", 'face_ref_img_upload': None})

        analysis_params = PreAnalysisEvent(**ui_args)
        event = PropagationEvent(
            output_folder=ui_args['output_folder'], video_path=ui_args['video_path'],
            scenes=scenes, analysis_params=analysis_params
        )

        final_result = {}
        for result in execute_propagation(event, self.progress_queue,
                                          self.cancel_event, self.logger,
                                          self.config, self.thumbnail_manager,
                                          self.cuda_available):
            if isinstance(result, dict):
                final_result.update(result)

        if final_result.get("done"):
            return {
                "log": final_result.get("log", "✅ Propagation completed successfully."),
                "analysis_output_dir_state": final_result.get('output_dir', ""),
                "analysis_metadata_path_state": final_result.get('metadata_path', ""),
                "filtering_tab": gr.update(interactive=True),
                "main_tabs": gr.update(selected=2)
            }
        return {"log": final_result.get("log", "❌ Propagation failed or was cancelled.")}

    def run_session_load_wrapper(self, session_path):
        """Wrapper for session loading. Returns a dict of final updates."""
        event = SessionLoadEvent(session_path=session_path)
        final_result = {}
        # Assuming run_pipeline_logic is a generator that yields updates
        for result in run_pipeline_logic(event, self.progress_queue, self.cancel_event,
                                         self.logger, self.config,
                                         self.thumbnail_manager, self.cuda_available):
            if isinstance(result, dict):
                final_result.update(result)
        return final_result

    def _setup_visibility_toggles(self):
        """Set up UI visibility toggles."""
        c = self.components
        c['method_input'].change(
            lambda m: (gr.update(visible=m == 'interval'),
                      gr.update(visible=m == 'scene'),
                      gr.update(visible=m == 'every_nth_frame')),
            c['method_input'],
            [c['interval_input'], c['fast_scene_input'], c['nth_frame_input']]
        )
        def toggle_extraction_accordions(is_recommended):
            return {
                c['recommended_accordion']: gr.update(visible=is_recommended),
                c['legacy_accordion']: gr.update(visible=not is_recommended)
            }

        c['thumbnails_only_input'].change(
            toggle_extraction_accordions,
            [c['thumbnails_only_input']],
            [c['recommended_accordion'], c['legacy_accordion']]
        )

        def on_strategy_change(strategy):
            is_face = strategy == "👤 By Face"
            is_text = strategy == "📝 By Text"
            is_fallback = strategy == "🔄 Face + Text Fallback"
            is_auto = strategy == "🤖 Automatic"
            
            return {
                c['face_seeding_group']: gr.update(visible=is_face or is_fallback),
                c['text_seeding_group']: gr.update(visible=is_text or is_fallback),
                c['auto_seeding_group']: gr.update(visible=is_auto),
                c['enable_face_filter_input']: gr.update(value=is_face or is_fallback)
            }

        c['primary_seed_strategy_input'].change(
            on_strategy_change,
            [c['primary_seed_strategy_input']],
            [c['face_seeding_group'], c['text_seeding_group'],
             c['auto_seeding_group'], c['enable_face_filter_input']]
        )

    def _setup_pipeline_handlers(self):
        """Set up pipeline execution handlers using the correct progress pattern."""
        c = self.components

        # --- Handler Definitions ---
        def session_load_handler(session_path, progress=gr.Progress()):
            output_keys = [k for k in self.session_load_keys if k != 'progress_bar']
            yield from self._run_task_with_progress(
                self.run_session_load_wrapper, output_keys, progress, session_path
            )

        def extraction_handler(*args, progress=gr.Progress()):
            output_keys = [
                'unified_log', 'extracted_video_path_state',
                'extracted_frames_dir_state', 'main_tabs'
            ]
            yield from self._run_task_with_progress(
                self.run_extraction_wrapper, output_keys, progress, *args
            )

        def pre_analysis_handler(*args, progress=gr.Progress()):
            output_keys = [
                'unified_log', 'seeding_preview_gallery', 'scenes_state',
                'propagate_masks_button', 'scene_filter_status',
                'scene_face_sim_min_input', 'seeding_results_column',
                'propagation_group'
            ]
            yield from self._run_task_with_progress(
                self.run_pre_analysis_wrapper, output_keys, progress, *args
            )

        def propagation_handler(scenes, *args, progress=gr.Progress()):
            output_keys = [
                'unified_log', 'analysis_output_dir_state',
                'analysis_metadata_path_state', 'filtering_tab', 'main_tabs'
            ]
            yield from self._run_task_with_progress(
                self.run_propagation_wrapper, output_keys, progress, scenes, *args
            )

        # --- Component to Input Mapping ---
        ext_comp_map = {
            'source_path': 'source_input', 'upload_video': 'upload_video_input',
            'max_resolution': 'max_resolution', 'scene_detect': 'ext_scene_detect_input',
            **{k: f"{k}_input" for k in self.ext_ui_map_keys if k not in ['source_path', 'upload_video', 'max_resolution', 'scene_detect']}
        }
        ext_inputs = [c[ext_comp_map[k]] for k in self.ext_ui_map_keys]

        ana_comp_map = {
            'output_folder': 'extracted_frames_dir_state', 'video_path': 'extracted_video_path_state',
            'resume': gr.State(False), 'enable_face_filter': 'enable_face_filter_input',
            'face_ref_img_path': 'face_ref_img_path_input', 'face_ref_img_upload': 'face_ref_img_upload_input',
            'face_model_name': 'face_model_name_input', 'enable_subject_mask': gr.State(True),
            'dam4sam_model_name': 'dam4sam_model_name_input', 'person_detector_model': 'person_detector_model_input',
            'seed_strategy': 'seed_strategy_input', 'scene_detect': 'ext_scene_detect_input',
            'enable_dedup': 'enable_dedup_input', 'text_prompt': 'text_prompt_input',
            'box_threshold': 'scene_editor_box_thresh_input', 'text_threshold': 'scene_editor_text_thresh_input',
            'min_mask_area_pct': gr.State(self.config.min_mask_area_pct),
            'sharpness_base_scale': gr.State(self.config.sharpness_base_scale),
            'edge_strength_base_scale': gr.State(self.config.edge_strength_base_scale),
            'gdino_config_path': gr.State(str(self.config.GROUNDING_DINO_CONFIG)),
            'gdino_checkpoint_path': gr.State(str(self.config.GROUNDING_DINO_CKPT)),
            'pre_analysis_enabled': 'pre_analysis_enabled_input', 'pre_sample_nth': 'pre_sample_nth_input',
            'primary_seed_strategy': 'primary_seed_strategy_input'
        }
        self.ana_input_components = [c.get(ana_comp_map[k], ana_comp_map[k]) for k in self.ana_ui_map_keys]
        prop_inputs = [c['scenes_state']] + self.ana_input_components

        # --- Click Handler Assignments ---
        session_output_keys = [k for k in self.session_load_keys if k != 'progress_bar']
        c['load_session_button'].click(
            session_load_handler,
            inputs=[c['session_path_input']],
            outputs=[c.get(k) for k in session_output_keys]
        )

        extraction_output_keys = [
            'unified_log', 'extracted_video_path_state',
            'extracted_frames_dir_state', 'main_tabs'
        ]
        c['start_extraction_button'].click(
            extraction_handler,
            inputs=ext_inputs,
            outputs=[c.get(k) for k in extraction_output_keys]
        )

        pre_analysis_output_keys = [
            'unified_log', 'seeding_preview_gallery', 'scenes_state',
            'propagate_masks_button', 'scene_filter_status',
            'scene_face_sim_min_input', 'seeding_results_column',
            'propagation_group'
        ]
        c['start_pre_analysis_button'].click(
            pre_analysis_handler,
            inputs=self.ana_input_components,
            outputs=[c.get(k) for k in pre_analysis_output_keys]
        )

        propagation_output_keys = [
            'unified_log', 'analysis_output_dir_state',
            'analysis_metadata_path_state', 'filtering_tab', 'main_tabs'
        ]
        c['propagate_masks_button'].click(
            propagation_handler,
            inputs=prop_inputs,
            outputs=[c.get(k) for k in propagation_output_keys]
        )

    def _setup_scene_editor_handlers(self):
        """Set up scene editor event handlers."""
        c = self.components

        def on_select_scene(scenes, evt: gr.SelectData):
            if not scenes or evt.index is None:
                return (gr.update(open=False), None, "",
                       self.config.grounding_dino_params['box_threshold'],
                       self.config.grounding_dino_params['text_threshold'])
            scene = scenes[evt.index]
            cfg = scene.get('seed_config', {})
            status_md = (f"**Editing Scene {scene['shot_id']}** "
                        f"(Frames {scene['start_frame']}-{scene['end_frame']})")
            prompt = cfg.get('text_prompt', '') if cfg else ''

            return (gr.update(open=True, value=status_md), scene['shot_id'],
                   prompt,
                   cfg.get('box_threshold', self.config.grounding_dino_params['box_threshold']),
                   cfg.get('text_threshold', self.config.grounding_dino_params['text_threshold']))

        c['seeding_preview_gallery'].select(
            on_select_scene, [c['scenes_state']],
            [c['scene_editor_accordion'], c['selected_scene_id_state'],
             c['scene_editor_prompt_input'],
             c['scene_editor_box_thresh_input'], 
             c['scene_editor_text_thresh_input']]
        )

        recompute_inputs = [
            c['scenes_state'], c['selected_scene_id_state'],
            c['scene_editor_prompt_input'],
            c['scene_editor_box_thresh_input'], 
            c['scene_editor_text_thresh_input'],
            c['extracted_frames_dir_state']
        ] + self.ana_input_components
        
        c['scene_recompute_button'].click(
            self.on_apply_scene_overrides,
            inputs=recompute_inputs,
            outputs=[c['seeding_preview_gallery'], c['scenes_state'],
                    c['unified_status']]
        )

        include_exclude_inputs = [c['scenes_state'], c['selected_scene_id_state'],
                                c['extracted_frames_dir_state']]
        include_exclude_outputs = [c['scenes_state'], c['scene_filter_status'],
                                 c['unified_status']]
        c['scene_include_button'].click(
            self.on_toggle_scene_status,
            include_exclude_inputs + [gr.State('included')], include_exclude_outputs
        )
        c['scene_exclude_button'].click(
            self.on_toggle_scene_status,
            include_exclude_inputs + [gr.State('excluded')], include_exclude_outputs
        )

    def _setup_bulk_scene_handlers(self):
        """Set up bulk scene operation handlers."""
        c = self.components

        def bulk_toggle(scenes, new_status, output_folder):
            if not scenes:
                return [], "No scenes to update."
            for s in scenes:
                s['status'] = new_status
                s['manual_status_change'] = True
            save_scene_seeds(scenes, output_folder, self.logger)
            status_text = get_scene_status_text(scenes)
            return scenes, status_text

        c['bulk_include_all_button'].click(
            lambda s, folder: bulk_toggle(s, 'included', folder),
            [c['scenes_state'], c['extracted_frames_dir_state']],
            [c['scenes_state'], c['scene_filter_status']]
        )
        c['bulk_exclude_all_button'].click(
            lambda s, folder: bulk_toggle(s, 'excluded', folder),
            [c['scenes_state'], c['extracted_frames_dir_state']],
            [c['scenes_state'], c['scene_filter_status']]
        )

        bulk_filter_inputs = [
            c['scenes_state'], c['scene_mask_area_min_input'],
            c['scene_face_sim_min_input'], c['scene_confidence_min_input'],
            c['enable_face_filter_input'], c['extracted_frames_dir_state']
        ]
        bulk_filter_outputs = [c['scenes_state'], c['scene_filter_status']]

        c['scene_mask_area_min_input'].release(
            self.on_apply_bulk_scene_filters,
            bulk_filter_inputs,
            bulk_filter_outputs
        )
        for comp in [c['scene_face_sim_min_input'], c['scene_confidence_min_input']]:
            comp.release(
                self.on_apply_bulk_scene_filters,
                bulk_filter_inputs,
                bulk_filter_outputs
            )

    def _setup_filtering_handlers(self):
        """Set up filtering and export handlers."""
        c = self.components
        slider_keys = sorted(c['metric_sliders'].keys())
        slider_comps = [c['metric_sliders'][k] for k in slider_keys]
        
        fast_filter_inputs = [
            c['all_frames_data_state'], c['per_metric_values_state'],
            c['analysis_output_dir_state'], c['gallery_view_toggle'],
            c['show_mask_overlay_input'], c['overlay_alpha_slider'],
            c['require_face_match_input'], c['dedup_thresh_input']
        ] + slider_comps
        fast_filter_outputs = [c['filter_status_text'], c['results_gallery']]

        for control in (slider_comps + [c['dedup_thresh_input'], 
                                       c['gallery_view_toggle'],
                                       c['show_mask_overlay_input'],
                                       c['overlay_alpha_slider'],
                                       c['require_face_match_input']]):
            handler = (control.release if hasattr(control, 'release') 
                      else control.input if hasattr(control, 'input') 
                      else control.change)
            handler(self.on_filters_changed_wrapper, fast_filter_inputs,
                   fast_filter_outputs)

        def load_and_trigger_update(metadata_path, output_dir):
            if not metadata_path or not output_dir:
                return [gr.update()] * len(load_outputs)

            all_frames, metric_values = load_and_prep_filter_data(
                metadata_path, self.get_all_filter_keys)
            svgs = build_all_metric_svgs(metric_values, self.get_all_filter_keys, self.logger)

            updates = {
                c['all_frames_data_state']: all_frames,
                c['per_metric_values_state']: metric_values,
                c['results_group']: gr.update(visible=True),
                c['export_group']: gr.update(visible=True)
            }

            for k in self.get_all_filter_keys():
                has_data = k in metric_values and len(metric_values.get(k, [])) > 0
                updates[c['metric_plots'][k]] = gr.update(visible=has_data,
                                                        value=svgs.get(k, ""))
                if f"{k}_min" in c['metric_sliders']:
                    updates[c['metric_sliders'][f"{k}_min"]] = gr.update(
                        visible=has_data)
                if f"{k}_max" in c['metric_sliders']:
                    updates[c['metric_sliders'][f"{k}_max"]] = gr.update(
                        visible=has_data)
                if k == "face_sim":
                    updates[c['require_face_match_input']] = gr.update(
                        visible=has_data)

            slider_values = {key: c['metric_sliders'][key].value for key in slider_keys}
            filter_event = FilterEvent(
                all_frames_data=all_frames,
                per_metric_values=metric_values,
                output_dir=output_dir,
                gallery_view="Kept Frames",
                show_overlay=True,
                overlay_alpha=0.6,
                require_face_match=c['require_face_match_input'].value,
                dedup_thresh=c['dedup_thresh_input'].value,
                slider_values=slider_values
            )

            filter_updates = on_filters_changed(filter_event, self.thumbnail_manager)
            updates.update({
                c['filter_status_text']: filter_updates['filter_status_text'],
                c['results_gallery']: filter_updates['results_gallery']
            })

            return [updates.get(comp, gr.update()) for comp in load_outputs]

        load_outputs = ([c['all_frames_data_state'], c['per_metric_values_state'],
                        c['filter_status_text'], c['results_gallery'],
                        c['results_group'], c['export_group']] +
                       [c['metric_plots'][k] for k in self.get_all_filter_keys()] +
                       slider_comps + [c['require_face_match_input']])
        
        c['filtering_tab'].select(load_and_trigger_update,
                                [c['analysis_metadata_path_state'],
                                 c['analysis_output_dir_state']], load_outputs)
        export_inputs = [
            c['all_frames_data_state'], c['analysis_output_dir_state'],
            c['extracted_video_path_state'], c['enable_crop_input'],
            c['crop_ar_input'], c['crop_padding_input'],
            c['require_face_match_input'], c['dedup_thresh_input']
        ] + slider_comps
        c['export_button'].click(self.export_kept_frames_wrapper, export_inputs,
                               c['unified_log'])

        # Setup for reset button
        reset_outputs_comps = (
            slider_comps +
            [c['dedup_thresh_input'],
             c['require_face_match_input'],
             c['filter_status_text'],
             c['results_gallery']]
        )
        c['reset_filters_button'].click(
            self.on_reset_filters,
            inputs=[c['all_frames_data_state'], c['per_metric_values_state'],
                    c['analysis_output_dir_state']],
            outputs=reset_outputs_comps
        )

        auto_set_outputs = [c['metric_sliders'][k] for k in slider_keys]
        c['apply_auto_button'].click(
            self.on_auto_set_thresholds,
            [c['per_metric_values_state'], c['auto_pctl_input']],
            auto_set_outputs
        ).then(
            self.on_filters_changed_wrapper, fast_filter_inputs, fast_filter_outputs
        )

    def on_filters_changed_wrapper(self, all_frames_data, per_metric_values, output_dir,
                                 gallery_view, show_overlay, overlay_alpha,
                                 require_face_match, dedup_thresh, *slider_values):
        """Wrapper for on_filters_changed logic."""
        slider_keys = sorted(self.components['metric_sliders'].keys())
        slider_values_dict = {key: val for key, val in zip(slider_keys, slider_values)}

        event = FilterEvent(
            all_frames_data=all_frames_data,
            per_metric_values=per_metric_values,
            output_dir=output_dir,
            gallery_view=gallery_view,
            show_overlay=show_overlay,
            overlay_alpha=overlay_alpha,
            require_face_match=require_face_match,
            dedup_thresh=dedup_thresh,
            slider_values=slider_values_dict
        )

        result = on_filters_changed(event, self.thumbnail_manager)
        return result['filter_status_text'], result['results_gallery']

    def on_reset_filters(self, all_frames_data, per_metric_values, output_dir):
        """Wrapper for reset_filters logic that returns updates in a fixed order."""
        slider_keys = sorted(self.components['metric_sliders'].keys())
        result = reset_filters(all_frames_data, per_metric_values, output_dir,
                               self.config, slider_keys, self.thumbnail_manager)

        # `result` is a dictionary: {'component_name': gr.update(...), ...}
        # The order of returned updates must match the `outputs` of the click handler.

        updates = []
        # 1. Metric sliders (in sorted key order)
        for key in slider_keys:
            comp_name = f"slider_{key}"
            updates.append(result.get(comp_name, gr.update()))

        # 2. Dedup slider
        updates.append(result.get('dedup_thresh_input', gr.update()))

        # 3. Require face match checkbox
        updates.append(result.get('require_face_match_input', gr.update()))

        # 4. Filter status text
        updates.append(result.get('filter_status_text', gr.update()))

        # 5. Results gallery
        updates.append(result.get('results_gallery', gr.update()))

        return tuple(updates)

    def on_auto_set_thresholds(self, per_metric_values, p):
        """Wrapper for auto_set_thresholds logic."""
        slider_keys = sorted(self.components['metric_sliders'].keys())
        updates = auto_set_thresholds(per_metric_values, p, slider_keys)

        # The logic function returns a dict of updates. We need to return a list
        # of values in the correct order for the Gradio `outputs`.
        return [updates.get(f'slider_{key}', gr.update()) for key in slider_keys]

    def on_toggle_scene_status(self, scenes_list, selected_shot_id, output_folder, new_status):
        """Wrapper for toggle_scene_status logic."""
        scenes, status_text, unified_status = toggle_scene_status(
            scenes_list, selected_shot_id, new_status, output_folder, self.logger)
        return scenes, status_text, unified_status

    def on_apply_bulk_scene_filters(self, scenes, min_mask_area, min_face_sim,
                                    min_confidence, enable_face_filter,
                                    output_folder):
        """Wrapper for apply_bulk_scene_filters logic."""
        scenes, status_text = apply_bulk_scene_filters(
            scenes, min_mask_area, min_face_sim, min_confidence,
            enable_face_filter, output_folder, self.logger
        )
        return scenes, status_text

    def on_apply_scene_overrides(self, scenes_list, selected_shot_id, prompt,
                                box_th, text_th, output_folder, *ana_args):
        """Wrapper for apply_scene_overrides logic."""
        gallery, scenes, status = apply_scene_overrides(
            scenes_list, selected_shot_id, prompt, box_th, text_th,
            output_folder, self.ana_ui_map_keys, ana_args,
            self.cuda_available, self.thumbnail_manager, self.logger
        )
        return gallery, scenes, status

    def export_kept_frames_wrapper(self, all_frames_data, output_dir, video_path,
                                 enable_crop, crop_ars, crop_padding,
                                 require_face_match, dedup_thresh, *slider_values):
        """Wrapper for exporting frames."""
        slider_keys = sorted(self.components['metric_sliders'].keys())
        slider_values_dict = {key: val for key, val in zip(slider_keys, slider_values)}

        filter_args = slider_values_dict
        filter_args.update({
            "require_face_match": require_face_match,
            "dedup_thresh": dedup_thresh
        })

        event = ExportEvent(
            all_frames_data=all_frames_data,
            output_dir=output_dir,
            video_path=video_path,
            enable_crop=enable_crop,
            crop_ars=crop_ars,
            crop_padding=crop_padding,
            filter_args=filter_args
        )

        # This logic remains here as it's tightly coupled to ffmpeg execution
        # and file system operations based on UI state.
        # It could be moved, but it's a smaller piece of logic.
        return self.export_kept_frames(event)

    def export_kept_frames(self, event: ExportEvent):
        """Export filtered frames to a new directory."""
        import subprocess
        from datetime import datetime
        from pathlib import Path
        import json
        import cv2
        import numpy as np

        if not event.all_frames_data:
            return "No metadata to export."
        if not event.video_path or not Path(event.video_path).exists():
            return "[ERROR] Original video path is required for export."

        try:
            filters = event.filter_args.copy()
            filters.update({
                "face_sim_enabled": any("face_sim" in f for f in event.all_frames_data),
                "mask_area_enabled": any("mask_area_pct" in f for f in event.all_frames_data),
                "enable_dedup": any('phash' in f for f in event.all_frames_data)
            })

            kept, _, _, _ = apply_all_filters_vectorized(event.all_frames_data, filters)
            if not kept:
                return "No frames kept after filtering. Nothing to export."

            out_root = Path(event.output_dir)
            frame_map_path = out_root / "frame_map.json"
            if not frame_map_path.exists():
                return "[ERROR] frame_map.json not found. Cannot export."

            with frame_map_path.open('r', encoding='utf-8') as f:
                frame_map_list = json.load(f)

            fn_to_orig_map = {
                f"frame_{i+1:06d}.png": orig
                for i, orig in enumerate(sorted(frame_map_list))
            }

            frames_to_extract = sorted([
                fn_to_orig_map[f['filename']] for f in kept
                if f['filename'] in fn_to_orig_map
            ])

            if not frames_to_extract:
                return "No frames to extract."

            select_exprs = [f"eq(n,{fn})" for fn in frames_to_extract]
            select_filter = f"select='{'+'.join(select_exprs)}'"

            export_dir = (out_root.parent /
                         f"{out_root.name}_exported_"
                         f"{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            export_dir.mkdir(exist_ok=True, parents=True)

            cmd = ['ffmpeg', '-y', '-i', str(event.video_path), '-vf', select_filter,
                  '-vsync', 'vfr', str(export_dir / "frame_%06d.png")]

            self.logger.info("Starting final export extraction...",
                           extra={'command': ' '.join(cmd)})
            subprocess.run(cmd, check=True, capture_output=True, text=True)

            if event.enable_crop:
                self.logger.info("Starting crop export...")
                crop_dir = export_dir / "cropped"
                crop_dir.mkdir(exist_ok=True)

                try:
                    aspect_ratios = [
                        tuple(map(int, ar.strip().split(':')))
                        for ar in event.crop_ars.split(',') if ar.strip()
                    ]
                except Exception:
                    return "Invalid aspect ratio format. Use 'width:height,width:height' e.g. '16:9,1:1'."

                masks_root = out_root / "masks"
                num_cropped = 0

                for frame_meta in kept:
                    if self.cancel_event.is_set(): break

                    try:
                        full_frame_path = export_dir / frame_meta['filename']
                        if not full_frame_path.exists(): continue

                        mask_path = masks_root / frame_meta.get('mask_path', '')
                        if not mask_path.exists(): continue

                        frame_img = cv2.imread(str(full_frame_path))
                        mask_img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

                        if frame_img is None or mask_img is None: continue

                        contours, _ = cv2.findContours(mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        if not contours: continue

                        # Combine all contours to get the overall bounding box of the subject
                        all_points = np.concatenate(contours)
                        x, y, w, h = cv2.boundingRect(all_points)

                        # Get dimensions for scaling
                        frame_h, frame_w = frame_img.shape[:2]
                        mask_h, mask_w = mask_img.shape[:2]

                        if mask_h != frame_h or mask_w != frame_w:
                            self.logger.info(f"Scaling bounding box for {frame_meta['filename']} from {mask_w}x{mask_h} to {frame_w}x{frame_h}")
                            scale_x = frame_w / mask_w
                            scale_y = frame_h / mask_h
                            x, y, w, h = int(x * scale_x), int(y * scale_y), int(w * scale_x), int(h * scale_y)

                        # Calculate padding in pixels
                        padding_px_w = int(w * (event.crop_padding / 100.0))
                        padding_px_h = int(h * (event.crop_padding / 100.0))

                        # Apply padding and clamp to frame boundaries
                        x1 = max(0, x - padding_px_w)
                        y1 = max(0, y - padding_px_h)
                        x2 = min(frame_w, x + w + padding_px_w)
                        y2 = min(frame_h, y + h + padding_px_h)

                        # The padded bounding box
                        x_pad, y_pad = x1, y1
                        w_pad, h_pad = x2 - x1, y2 - y1

                        if w_pad <= 0 or h_pad <= 0: continue

                        center_x, center_y = x_pad + w_pad // 2, y_pad + h_pad // 2

                        # Find best aspect ratio that fits the mask area
                        if not aspect_ratios or h_pad == 0:
                            continue

                        padded_ar = w_pad / h_pad
                        best_ar_dim = None
                        min_ar_diff = float('inf')

                        for ar_w, ar_h in aspect_ratios:
                            target_ar = ar_w / ar_h
                            diff = abs(target_ar - padded_ar)
                            if diff < min_ar_diff:
                                min_ar_diff = diff
                                best_ar_dim = (ar_w, ar_h)

                        if not best_ar_dim:
                            continue

                        ar_w, ar_h = best_ar_dim
                        target_ar = ar_w / ar_h

                        if w_pad / h_pad > target_ar:
                            new_h = w_pad / target_ar
                            new_w = w_pad
                        else:
                            new_w = h_pad * target_ar
                            new_h = h_pad

                        new_x = center_x - new_w / 2
                        new_y = center_y - new_h / 2

                        new_x, new_y = int(max(0, new_x)), int(max(0, new_y))
                        new_w, new_h = int(new_w), int(new_h)

                        if new_x + new_w > frame_img.shape[1]:
                            new_w = frame_img.shape[1] - new_x
                        if new_y + new_h > frame_img.shape[0]:
                            new_h = frame_img.shape[0] - new_y

                        cropped_img = frame_img[new_y:new_y+new_h, new_x:new_x+new_w]

                        if cropped_img.size > 0:
                            base_name = Path(frame_meta['filename']).stem
                            crop_filename = f"{base_name}_crop_{ar_w}x{ar_h}.png"
                            cv2.imwrite(str(crop_dir / crop_filename), cropped_img)
                            num_cropped += 1

                    except Exception as e:
                        self.logger.error(f"Failed to crop frame {frame_meta['filename']}", exc_info=True)

                self.logger.info(f"Cropping complete. Saved {num_cropped} cropped images.")

            return f"Exported {len(frames_to_extract)} frames to {export_dir.name}."

        except subprocess.CalledProcessError as e:
            self.logger.error("FFmpeg export failed", exc_info=True,
                            extra={'stderr': e.stderr})
            return "Error during export: FFmpeg failed. Check logs."
        except Exception as e:
            self.logger.error("Error during export process", exc_info=True)
            return f"Error during export: {e}"