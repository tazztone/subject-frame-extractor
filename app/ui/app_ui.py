import gradio as gr
import torch
import threading
from queue import Queue

from app.core.config import Config
from app.core.logging import UnifiedLogger
from app.core.thumb_cache import ThumbnailManager
from app.logic.events import (ExtractionEvent, PreAnalysisEvent, PropagationEvent,
                              FilterEvent, ExportEvent)
from app.logic.pipeline_logic import run_pipeline_logic
from app.logic.filter_logic import (load_and_prep_filter_data, build_all_metric_svgs,
                                    on_filters_changed, reset_filters,
                                    auto_set_thresholds, apply_all_filters_vectorized)
from app.logic.scene_logic import (toggle_scene_status, apply_bulk_scene_filters,
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
            'pre_analysis_enabled', 'pre_sample_nth'
        ]

    def build_ui(self):
        """Build the complete Gradio interface."""
        css = """.plot-and-slider-column { max-width: 560px !important; margin: auto; } .scene-editor { border: 1px solid #444; padding: 10px; border-radius: 5px; }"""
        
        with gr.Blocks(theme=gr.themes.Default(), css=css) as demo:
            gr.Markdown("# 🎬 Frame Extractor & Analyzer v2.0")
            if not self.cuda_available:
                gr.Markdown("⚠️ **CPU Mode** — GPU-dependent features are "
                          "disabled or will be slow.")

            with gr.Tabs():
                with gr.Tab("📹 1. Frame Extraction"):
                    self._create_extraction_tab()
                with gr.Tab("🎯 2. Seeding & Scene Selection") as analysis_tab:
                    self.components['analysis_tab'] = analysis_tab
                    self._create_analysis_tab()
                with gr.Tab("📊 3. Filtering & Export") as filtering_tab:
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
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 📹 Video Source")
                self._create_component('source_input', 'textbox', {
                    'label': "Video URL or Local Path",
                    'placeholder': "Enter YouTube URL or local video file path"
                })
                self._create_component('max_resolution', 'dropdown', {
                    'choices': ["maximum available", "2160", "1080", "720"],
                    'value': self.config.ui_defaults['max_resolution'],
                    'label': "DL Res"
                })
                self._create_component('upload_video_input', 'file', {
                    'label': "Or Upload Video",
                    'file_types': ["video"],
                    'type': "filepath"
                })
            with gr.Column():
                gr.Markdown("### ⚙️ Extraction Settings")
                with gr.Accordion("Thumbnail Extraction (Recommended)", 
                                open=True):
                    self._create_component('thumbnails_only_input', 'checkbox', {
                        'label': "Extract Thumbnails Only",
                        'value': self.config.ui_defaults['thumbnails_only']
                    })
                    self._create_component('thumb_megapixels_input', 'slider', {
                        'label': "Thumbnail Size (MP)", 'minimum': 0.1,
                        'maximum': 2.0, 'step': 0.1,
                        'value': self.config.ui_defaults['thumb_megapixels']
                    })
                    self._create_component('ext_scene_detect_input', 'checkbox', {
                        'label': "Use Scene Detection",
                        'value': self.config.ui_defaults['scene_detect']
                    })
                with gr.Accordion("Legacy Full-Frame Extraction", open=False):
                    method_choices = ["keyframes", "interval", "every_nth_frame", 
                                    "all", "scene"]
                    self._create_component('method_input', 'dropdown', {
                        'choices': method_choices,
                        'value': self.config.ui_defaults['method'],
                        'label': "Method"
                    })
                    self._create_component('interval_input', 'textbox', {
                        'label': "Interval (s)",
                        'value': self.config.ui_defaults['interval'],
                        'visible': False
                    })
                    self._create_component('nth_frame_input', 'textbox', {
                        'label': "N-th Frame Value",
                        'value': self.config.ui_defaults["nth_frame"],
                        'visible': False
                    })
                    self._create_component('fast_scene_input', 'checkbox', {
                        'label': "Fast Scene Detect",
                        'visible': False
                    })
                    self._create_component('use_png_input', 'checkbox', {
                        'label': "Save as PNG",
                        'value': self.config.ui_defaults['use_png']
                    })

        start_btn = gr.Button("🚀 Start Extraction", variant="primary")
        self.components.update({'start_extraction_button': start_btn})

    def _create_analysis_tab(self):
        """Create the analysis and seeding tab."""
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📁 Input & Pre-Analysis")
                self._create_component('frames_folder_input', 'textbox', {
                    'label': "📂 Extracted Frames Folder"
                })
                self._create_component('analysis_video_path_input', 'textbox', {
                    'label': "🎥 Original Video Path (for Export)"
                })
                
                with gr.Group():
                    self._create_component('pre_analysis_enabled_input', 
                                         'checkbox', {
                        'label': 'Enable Pre-Analysis to find best seed frame',
                        'value': self.config.ui_defaults['pre_analysis_enabled']
                    })
                    self._create_component('pre_sample_nth_input', 'number', {
                        'label': 'Sample every Nth thumbnail for pre-analysis',
                        'value': self.config.ui_defaults['pre_sample_nth'],
                        'interactive': True
                    })
                    
                with gr.Accordion("Global Seeding Settings", open=True):
                    self._create_component('enable_face_filter_input', 
                                         'checkbox', {
                        'label': "Enable Face Similarity",
                        'value': self.config.ui_defaults['enable_face_filter']
                    })
                    self._create_component('face_model_name_input', 'dropdown', {
                        'choices': ["buffalo_l", "buffalo_s"],
                        'value': self.config.ui_defaults['face_model_name'],
                        'label': "Face Model"
                    })
                    self._create_component('face_ref_img_path_input', 'textbox', {
                        'label': "📸 Face Image Path"
                    })
                    self._create_component('face_ref_img_upload_input', 'file', {
                        'label': "📤 Or Upload Face",
                        'type': "filepath"
                    })
                    self._create_component('text_prompt_input', 'textbox', {
                        'label': "Ground with text",
                        'placeholder': "e.g., 'a woman in a red dress'",
                        'value': self.config.ui_defaults['text_prompt']
                    })
                    self._create_component('seed_strategy_input', 'dropdown', {
                        'choices': ["Largest Person", "Center-most Person"],
                        'value': "Largest Person",
                        'label': "Fallback Seed Strategy (if no face/prompt)"
                    })
                    self._create_component('person_detector_model_input', 
                                         'dropdown', {
                        'choices': ['yolo11x.pt', 'yolo11s.pt'],
                        'value': self.config.ui_defaults['person_detector_model'],
                        'label': "Person Detector"
                    })
                    self._create_component('dam4sam_model_name_input', 'dropdown', {
                        'choices': ["sam21pp-T", "sam21pp-S", "sam21pp-B+", 
                                  "sam21pp-L"],
                        'value': self.config.ui_defaults['dam4sam_model_name'],
                        'label': "SAM Tracker Model"
                    })
                    
                with gr.Accordion("Advanced Analysis Settings", open=True):
                    self._create_component('enable_dedup_input', 'checkbox', {
                        'label': "Enable Deduplication (pHash)",
                        'value': self.config.ui_defaults.get('enable_dedup', False)
                    })

                self._create_component('start_pre_analysis_button', 'button', {
                    'value': '🌱 Start Pre-Analysis & Seeding Preview',
                    'variant': 'primary'
                })
                self._create_component('propagate_masks_button', 'button', {
                    'value': '🔬 Propagate Masks on Kept Scenes',
                    'variant': 'primary',
                    'interactive': False
                })

            with gr.Column(scale=2):
                gr.Markdown("### 🎭 Seeding Preview & Scene Filtering")
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
                        'maximum': 1.0, 'value': 0.5, 'step': 0.05
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
                self._create_component('load_analysis_for_filtering_button', 
                                     'button', {
                    'value': "🔄 Load/Refresh Analysis Results"
                })
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
                gr.Markdown("### 🖼️ Results Gallery")
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
                self._create_component('export_button', 'button', {
                    'value': "📤 Export Kept Frames",
                    'variant': "primary"
                })
                with gr.Row():
                    self._create_component('enable_crop_input', 'checkbox', {
                        'label': "✂️ Crop to Subject",
                        'value': True
                    })
                    self._create_component('crop_ar_input', 'textbox', {
                        'label': "ARs",
                        'value': "16:9,1:1,9:16"
                    })
                    self._create_component('crop_padding_input', 'slider', {
                        'label': "Padding %",
                        'value': 1
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

    def run_extraction_wrapper(self, *args):
        """Wrapper for extraction pipeline."""
        ui_args = dict(zip(self.ext_ui_map_keys, args))
        event = ExtractionEvent(**ui_args)
        yield from run_pipeline_logic(event, self.progress_queue, self.cancel_event,
                                      self.logger, self.config,
                                      self.thumbnail_manager, self.cuda_available)

    def run_pre_analysis_wrapper(self, *args):
        """Wrapper for pre-analysis pipeline."""
        ui_args = dict(zip(self.ana_ui_map_keys, args))
        event = PreAnalysisEvent(**ui_args)

        for result in run_pipeline_logic(event, self.progress_queue, self.cancel_event,
                                         self.logger, self.config,
                                         self.thumbnail_manager, self.cuda_available):
            if 'scenes_state' in result:
                scenes = result['scenes_state']
                save_scene_seeds(scenes, event.output_folder, self.logger)
                result['scene_filter_status'] = get_scene_status_text(scenes)
            yield result

    def run_propagation_wrapper(self, output_folder, video_path, scenes, *args):
        """Wrapper for propagation pipeline."""
        ui_args = dict(zip(self.ana_ui_map_keys, args))
        analysis_params = PreAnalysisEvent(**ui_args)
        event = PropagationEvent(
            output_folder=output_folder,
            video_path=video_path,
            scenes=scenes,
            analysis_params=analysis_params
        )
        yield from run_pipeline_logic(event, self.progress_queue, self.cancel_event,
                                      self.logger, self.config,
                                      self.thumbnail_manager)

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
        c['thumbnails_only_input'].change(
            lambda x: (gr.update(interactive=x), gr.update(interactive=x),
                      gr.update(interactive=not x)),
            c['thumbnails_only_input'],
            [c['thumb_megapixels_input'], c['ext_scene_detect_input'],
             c['method_input']]
        )

    def _setup_pipeline_handlers(self):
        """Set up pipeline execution handlers."""
        c = self.components
        
        ext_comp_map = {k: f"{k}_input" for k in self.ext_ui_map_keys}
        ext_comp_map.update({
            'source_path': 'source_input',
            'upload_video': 'upload_video_input',
            'max_resolution': 'max_resolution',
            'scene_detect': 'ext_scene_detect_input'
        })
        ext_inputs = [c[ext_comp_map[k]] for k in self.ext_ui_map_keys]
        ext_outputs = [
            c['unified_log'], c['unified_status'],
            c['extracted_video_path_state'], c['extracted_frames_dir_state'],
            c['frames_folder_input'], c['analysis_video_path_input']
        ]
        c['start_extraction_button'].click(self.run_extraction_wrapper,
                                          ext_inputs, ext_outputs)

        ana_comp_map = {
            'output_folder': 'frames_folder_input',
            'video_path': 'analysis_video_path_input',
            'resume': gr.State(False),
            'enable_face_filter': 'enable_face_filter_input',
            'face_ref_img_path': 'face_ref_img_path_input',
            'face_ref_img_upload': 'face_ref_img_upload_input',
            'face_model_name': 'face_model_name_input',
            'enable_subject_mask': gr.State(True),
            'dam4sam_model_name': 'dam4sam_model_name_input',
            'person_detector_model': 'person_detector_model_input',
            'seed_strategy': 'seed_strategy_input',
            'scene_detect': 'ext_scene_detect_input',
            'enable_dedup': 'enable_dedup_input',
            'text_prompt': 'text_prompt_input',
            'box_threshold': 'scene_editor_box_thresh_input',
            'text_threshold': 'scene_editor_text_thresh_input',
            'min_mask_area_pct': gr.State(self.config.min_mask_area_pct),
            'sharpness_base_scale': gr.State(self.config.sharpness_base_scale),
            'edge_strength_base_scale': gr.State(self.config.edge_strength_base_scale),
            'gdino_config_path': gr.State(str(self.config.GROUNDING_DINO_CONFIG)),
            'gdino_checkpoint_path': gr.State(str(self.config.GROUNDING_DINO_CKPT)),
            'pre_analysis_enabled': 'pre_analysis_enabled_input',
            'pre_sample_nth': 'pre_sample_nth_input'
        }
        self.ana_input_components = [
            c.get(ana_comp_map[k], ana_comp_map[k]) for k in self.ana_ui_map_keys
        ]

        pre_ana_outputs = [
            c['unified_log'], c['unified_status'], c['seeding_preview_gallery'],
            c['scenes_state'], c['propagate_masks_button'], c['scene_filter_status']
        ]
        c['start_pre_analysis_button'].click(self.run_pre_analysis_wrapper,
                                           self.ana_input_components,
                                           pre_ana_outputs)

        prop_inputs = ([c['frames_folder_input'], c['analysis_video_path_input'],
                       c['scenes_state']] + self.ana_input_components)
        prop_outputs = [
            c['unified_log'], c['unified_status'], c['analysis_output_dir_state'],
            c['analysis_metadata_path_state'], c['filtering_tab']
        ]
        c['propagate_masks_button'].click(self.run_propagation_wrapper,
                                        prop_inputs, prop_outputs)

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
            c['frames_folder_input']
        ] + self.ana_input_components
        
        c['scene_recompute_button'].click(
            self.on_apply_scene_overrides,
            inputs=recompute_inputs,
            outputs=[c['seeding_preview_gallery'], c['scenes_state'], 
                    c['unified_status']]
        )

        include_exclude_inputs = [c['scenes_state'], c['selected_scene_id_state'],
                                c['frames_folder_input']]
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
            [c['scenes_state'], c['frames_folder_input']],
            [c['scenes_state'], c['scene_filter_status']]
        )
        c['bulk_exclude_all_button'].click(
            lambda s, folder: bulk_toggle(s, 'excluded', folder),
            [c['scenes_state'], c['frames_folder_input']],
            [c['scenes_state'], c['scene_filter_status']]
        )

        bulk_filter_inputs = [
            c['scenes_state'], c['scene_mask_area_min_input'],
            c['scene_face_sim_min_input'], c['enable_face_filter_input'],
            c['frames_folder_input']
        ]
        bulk_filter_outputs = [c['scenes_state'], c['scene_filter_status']]

        c['scene_mask_area_min_input'].release(
            self.on_apply_bulk_scene_filters,
            bulk_filter_inputs,
            bulk_filter_outputs
        )
        c['scene_face_sim_min_input'].release(
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
                c['per_metric_values_state']: metric_values
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
                        c['filter_status_text'], c['results_gallery']] +
                       [c['metric_plots'][k] for k in self.get_all_filter_keys()] +
                       slider_comps + [c['require_face_match_input']])
        
        c['filtering_tab'].select(load_and_trigger_update,
                                [c['analysis_metadata_path_state'],
                                 c['analysis_output_dir_state']], load_outputs)
        c['load_analysis_for_filtering_button'].click(
            load_and_trigger_update,
            [c['analysis_metadata_path_state'], c['analysis_output_dir_state']],
            load_outputs
        )

        export_inputs = [
            c['all_frames_data_state'], c['analysis_output_dir_state'],
            c['extracted_video_path_state'], c['enable_crop_input'],
            c['crop_ar_input'], c['crop_padding_input'],
            c['require_face_match_input'], c['dedup_thresh_input']
        ] + slider_comps
        c['export_button'].click(self.export_kept_frames_wrapper, export_inputs,
                               c['unified_log'])

        reset_outputs = {
            c[k]: v for k, v in
            reset_filters(None, None, None, self.config, slider_keys, self.thumbnail_manager).items()
        }

        c['reset_filters_button'].click(
            self.on_reset_filters,
            [c['all_frames_data_state'], c['per_metric_values_state'],
             c['analysis_output_dir_state']],
            list(reset_outputs.keys())
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
        """Wrapper for reset_filters logic."""
        slider_keys = sorted(self.components['metric_sliders'].keys())
        result = reset_filters(all_frames_data, per_metric_values, output_dir,
                               self.config, slider_keys, self.thumbnail_manager)
        
        # The logic function returns a dict of updates. We need to return a list
        # of values in the correct order for the Gradio `outputs`.
        # The order is determined by the `reset_outputs` dict in _setup_filtering_handlers
        c = self.components
        reset_outputs_order = (
            [c['metric_sliders'][k] for k in slider_keys] +
            [c['require_face_match_input'], c['dedup_thresh_input'],
             c['filter_status_text'], c['results_gallery']]
        )
        
        # Map component objects to their names to look up results
        comp_to_name_map = {v: k for k, v in c.items()}
        
        final_result = []
        for comp in reset_outputs_order:
            comp_name = comp_to_name_map.get(comp)
            if comp_name:
                # Logic function returns keys like 'slider_niqe_min', but component name is 'slider_niqe_min'
                # So we need to match them.
                update_key = next((k for k in result.keys() if k.endswith(comp_name)), None)
                if update_key:
                     final_result.append(result[update_key])
                else: # for filter_status_text and results_gallery
                     final_result.append(result.get(comp_name, gr.update()))
            else:
                final_result.append(gr.update())

        return final_result

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
                                    enable_face_filter, output_folder):
        """Wrapper for apply_bulk_scene_filters logic."""
        scenes, status_text = apply_bulk_scene_filters(
            scenes, min_mask_area, min_face_sim, enable_face_filter,
            output_folder, self.logger
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

            return f"Exported {len(frames_to_extract)} frames to {export_dir.name}."
            
        except subprocess.CalledProcessError as e:
            self.logger.error("FFmpeg export failed", exc_info=True,
                            extra={'stderr': e.stderr})
            return "Error during export: FFmpeg failed. Check logs."
        except Exception as e:
            self.logger.error("Error during export process", exc_info=True)
            return f"Error during export: {e}"