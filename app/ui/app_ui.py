"""Gradio UI for the frame extractor application."""

import json
import re
import shutil
import subprocess
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from queue import Queue, Empty
from dataclasses import asdict

import cv2
import gradio as gr
import numpy as np
import torch

from app.core.config import Config
from app.core.logging import UnifiedLogger
from app.core.thumb_cache import ThumbnailManager
from app.core.utils import _to_json_safe
from app.domain.models import Scene, AnalysisParameters
from app.io.frames import rgb_to_pil, render_mask_overlay
# ML imports are lazy-loaded to avoid dependency issues
from app.masking.subject_masker import SubjectMasker
from app.pipelines.extract import ExtractionPipeline
from app.pipelines.analyze import AnalysisPipeline


class AppUI:
    """Main Gradio UI class for the frame extractor application."""
    
    def __init__(self, config=None, logger=None, progress_queue=None, 
                 cancel_event=None):
        # Accept injected dependencies or create defaults
        self.config = config or Config()
        self.logger = logger or UnifiedLogger()
        self.progress_queue = progress_queue
        self.cancel_event = cancel_event or threading.Event()
        
        self.components = {}
        self.last_task_result = {}
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
                        'label': "📸 Reference Image Path"
                    })
                    self._create_component('face_ref_img_upload_input', 'file', {
                        'label': "📤 Or Upload",
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
        yield from self._run_pipeline("extraction", ui_args)

    def run_pre_analysis_wrapper(self, *args):
        """Wrapper for pre-analysis pipeline."""
        ui_args = dict(zip(self.ana_ui_map_keys, args))
        yield from self._run_pipeline("pre_analysis", ui_args)

    def run_propagation_wrapper(self, output_folder, video_path, scenes, *args):
        """Wrapper for propagation pipeline."""
        ui_args = dict(zip(self.ana_ui_map_keys, args))
        ui_args['output_folder'] = output_folder
        ui_args['video_path'] = video_path
        yield from self._run_pipeline("propagation", ui_args, scenes=scenes)

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
        
        # Extraction pipeline
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

        # Analysis pipeline
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
                       self.config.GROUNDING_BOX_THRESHOLD,
                       self.config.GROUNDING_TEXT_THRESHOLD)
            scene = scenes[evt.index]
            cfg = scene.get('seed_config', {})
            status_md = (f"**Editing Scene {scene['shot_id']}** "
                        f"(Frames {scene['start_frame']}-{scene['end_frame']})")
            prompt = cfg.get('text_prompt', '') if cfg else ''

            return (gr.update(open=True, value=status_md), scene['shot_id'],
                   prompt,
                   cfg.get('box_threshold', self.config.GROUNDING_BOX_THRESHOLD),
                   cfg.get('text_threshold', self.config.GROUNDING_TEXT_THRESHOLD))

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
            self.apply_scene_overrides,
            inputs=recompute_inputs,
            outputs=[c['seeding_preview_gallery'], c['scenes_state'], 
                    c['unified_status']]
        )

        include_exclude_inputs = [c['scenes_state'], c['selected_scene_id_state'],
                                c['frames_folder_input']]
        include_exclude_outputs = [c['scenes_state'], c['scene_filter_status'],
                                 c['unified_status']]
        c['scene_include_button'].click(
            lambda s, sid, folder: self._toggle_scene_status(s, sid, 'included', 
                                                           folder),
            include_exclude_inputs, include_exclude_outputs
        )
        c['scene_exclude_button'].click(
            lambda s, sid, folder: self._toggle_scene_status(s, sid, 'excluded',
                                                           folder),
            include_exclude_inputs, include_exclude_outputs
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
            self._save_scene_seeds(scenes, output_folder)
            status_text = self._get_scene_status_text(scenes)
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
            self.apply_bulk_scene_filters,
            bulk_filter_inputs,
            bulk_filter_outputs
        )
        c['scene_face_sim_min_input'].release(
            self.apply_bulk_scene_filters,
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
            handler(self.on_filters_changed, fast_filter_inputs, 
                   fast_filter_outputs)

        def load_and_trigger_update(metadata_path, output_dir):
            if not metadata_path or not output_dir:
                return [gr.update()] * len(load_outputs)
            all_frames, metric_values = self.load_and_prep_filter_data(
                metadata_path)
            svgs = self.build_all_metric_svgs(metric_values)
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

            default_filters = [c['metric_sliders'][k].value for k in slider_keys]
            status, gallery = self.on_filters_changed(
                all_frames, metric_values, output_dir, "Kept Frames", True, 0.6,
                c['require_face_match_input'].value, 
                c['dedup_thresh_input'].value,
                *default_filters
            )
            updates[c['filter_status_text']] = status
            updates[c['results_gallery']] = gallery
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
        c['export_button'].click(self.export_kept_frames, export_inputs,
                               c['unified_log'])

        reset_outputs = (slider_comps + [c['require_face_match_input'],
                                        c['dedup_thresh_input'],
                                        c['filter_status_text'],
                                        c['results_gallery']])
        c['reset_filters_button'].click(
            self.reset_filters,
            [c['all_frames_data_state'], c['per_metric_values_state'],
             c['analysis_output_dir_state']],
            reset_outputs
        )
        c['apply_auto_button'].click(
            self.auto_set_thresholds,
            [c['per_metric_values_state'], c['auto_pctl_input']],
            slider_comps
        ).then(
            self.on_filters_changed, fast_filter_inputs, fast_filter_outputs
        )

    def _run_pipeline(self, pipeline_type, ui_args, scenes=None):
        """Execute a pipeline with progress tracking."""
        self.cancel_event.clear()
        q = Queue()
        if self.progress_queue:
            q = self.progress_queue
        else:
            self.logger.set_progress_queue(q)

        try:
            if 'upload_video' in ui_args and ui_args['upload_video']:
                source = ui_args.pop('upload_video')
                dest = str(self.config.DIRS['downloads'] / Path(source).name)
                shutil.copy2(source, dest)
                ui_args['source_path'] = dest
            if 'face_ref_img_upload' in ui_args and ui_args['face_ref_img_upload']:
                ref_upload = ui_args.pop('face_ref_img_upload')
                dest = self.config.DIRS['downloads'] / Path(ref_upload).name
                shutil.copy2(ref_upload, dest)
                ui_args['face_ref_img_path'] = str(dest)

            params = AnalysisParameters.from_ui(**ui_args)

            if pipeline_type == "extraction":
                yield from self.execute_extraction(params, q)
            elif pipeline_type == "pre_analysis":
                yield from self.execute_pre_analysis(params, q)
            elif pipeline_type == "propagation":
                yield from self.execute_propagation(params, q, scenes)
        except Exception as e:
            self.logger.error(f"{pipeline_type} setup failed", exc_info=True)
            yield {
                self.components['unified_log']: str(e),
                self.components['unified_status']: f"[ERROR] {e}"
            }

    def execute_extraction(self, params, q):
        """Execute extraction pipeline."""
        yield {
            self.components['unified_log']: "",
            self.components['unified_status']: "Starting extraction..."
        }
        pipeline = ExtractionPipeline(params, q, self.cancel_event)
        yield from self._run_task(pipeline.run, q)
        result = self.last_task_result
        if result.get("done"):
            yield {
                self.components['unified_log']: "Extraction complete.",
                self.components['unified_status']: f"Output: {result['output_dir']}",
                self.components['extracted_video_path_state']: result.get("video_path", ""),
                self.components['extracted_frames_dir_state']: result["output_dir"],
                self.components['frames_folder_input']: result["output_dir"],
                self.components['analysis_video_path_input']: result.get("video_path", "")
            }

    def _save_scene_seeds(self, scenes_list, output_dir_str):
        """Save scene seeds to JSON file."""
        if not scenes_list or not output_dir_str:
            return
        output_dir = Path(output_dir_str)
        scene_seeds = {
            str(s['shot_id']): {
                'seed_frame_idx': s.get('best_seed_frame'),
                'seed_type': s.get('seed_result', {}).get('details', {}).get('type'),
                'seed_config': s.get('seed_config', {}),
                'status': s.get('status', 'pending'),
                'seed_metrics': s.get('seed_metrics', {})
            }
            for s in scenes_list
        }
        try:
            (output_dir / "scene_seeds.json").write_text(
                json.dumps(_to_json_safe(scene_seeds), indent=2),
                encoding='utf-8'
            )
            self.logger.info("Saved scene_seeds.json")
        except Exception as e:
            self.logger.error("Failed to save scene_seeds.json", exc_info=True)

    def execute_pre_analysis(self, params, q):
        """Execute pre-analysis pipeline."""
        import pyiqa
        
        yield {
            self.components['unified_log']: "",
            self.components['unified_status']: "Starting Pre-Analysis..."
        }

        output_dir = Path(params.output_folder)
        scenes_path = output_dir / "scenes.json"
        if not scenes_path.exists():
            yield {
                self.components['unified_log']: "[ERROR] scenes.json not found. "
                                              "Run extraction with scene detection."
            }
            return

        with scenes_path.open('r', encoding='utf-8') as f:
            shots = json.load(f)
        scenes = [Scene(shot_id=i, start_frame=s, end_frame=e)
                 for i, (s, e) in enumerate(shots)]

        scene_seeds_path = output_dir / "scene_seeds.json"
        if scene_seeds_path.exists() and params.resume:
            self.logger.info("Loading existing scene_seeds.json")
            with scene_seeds_path.open('r', encoding='utf-8') as f:
                loaded_seeds = json.load(f)
            for scene in scenes:
                seed_data = loaded_seeds.get(str(scene.shot_id))
                if seed_data:
                    scene.best_seed_frame = seed_data.get('seed_frame_idx')
                    scene.seed_config = seed_data.get('seed_config', {})
                    scene.status = seed_data.get('status', 'pending')
                    scene.seed_metrics = seed_data.get('seed_metrics', {})

        niqe_metric, face_analyzer, ref_emb, person_detector = None, None, None, None
        device = "cuda" if self.cuda_available else "cpu"
        
        if params.pre_analysis_enabled:
            niqe_metric = pyiqa.create_metric('niqe', device=device)
        if params.enable_face_filter:
            from app.ml.face import get_face_analyzer
            face_analyzer = get_face_analyzer(params.face_model_name)
            if params.face_ref_img_path:
                ref_img = cv2.imread(params.face_ref_img_path)
                if ref_img is not None:
                    faces = face_analyzer.get(ref_img)
                    if faces:
                        ref_emb = max(faces, key=lambda x: x.det_score).normed_embedding
        from app.ml.person import get_person_detector
        person_detector = get_person_detector(params.person_detector_model, device)

        masker = SubjectMasker(params, q, self.cancel_event,
                             face_analyzer=face_analyzer,
                             reference_embedding=ref_emb,
                             person_detector=person_detector,
                             niqe_metric=niqe_metric,
                             thumbnail_manager=self.thumbnail_manager)
        masker.frame_map = masker._create_frame_map(str(output_dir))

        previews = []
        for i, scene in enumerate(scenes):
            q.put({
                "stage": f"Pre-analyzing scene {i+1}/{len(scenes)}",
                "total": len(scenes),
                "progress": i
            })

            if not scene.best_seed_frame:
                masker._select_best_seed_frame_in_scene(scene, str(output_dir))

            fname = masker.frame_map.get(scene.best_seed_frame)
            if not fname:
                self.logger.warning(
                    f"Could not find best_seed_frame {scene.best_seed_frame} "
                    f"in frame_map for scene {scene.shot_id}"
                )
                continue

            thumb_path = output_dir / "thumbs" / f"{Path(fname).stem}.webp"
            thumb_rgb = self.thumbnail_manager.get(thumb_path)

            if thumb_rgb is None:
                self.logger.warning(
                    f"Could not load thumbnail for best_seed_frame "
                    f"{scene.best_seed_frame} at path {thumb_path}"
                )
                continue

            bbox, details = masker.get_seed_for_frame(
                thumb_rgb, seed_config=scene.seed_config or params
            )
            scene.seed_result = {'bbox': bbox, 'details': details}

            mask = masker.get_mask_for_bbox(thumb_rgb, bbox) if bbox else None
            if mask is not None:
                h, w = mask.shape[:2]
                area_pct = ((np.sum(mask > 0) / (h * w)) * 100 
                           if (h * w) > 0 else 0.0)
                scene.seed_result['details']['mask_area_pct'] = area_pct

            overlay_rgb = (render_mask_overlay(thumb_rgb, mask) 
                          if mask is not None 
                          else masker.draw_bbox(thumb_rgb, bbox))

            caption = (f"Scene {scene.shot_id} (Seed: {scene.best_seed_frame}) "
                      f"| {details.get('type', 'N/A')}")
            previews.append((overlay_rgb, caption))
            scene.preview_path = "dummy"
            if scene.status == 'pending':
                scene.status = 'included'

        scenes_as_dict = [asdict(s) for s in scenes]
        self._save_scene_seeds(scenes_as_dict, str(output_dir))
        q.put({"stage": "Pre-analysis complete", "progress": len(scenes)})

        yield {
            self.components['unified_log']: "Pre-analysis complete.",
            self.components['unified_status']: f"{len(scenes)} scenes found.",
            self.components['seeding_preview_gallery']: gr.update(value=previews),
            self.components['scenes_state']: scenes_as_dict,
            self.components['propagate_masks_button']: gr.update(interactive=True),
            self.components['scene_filter_status']: self._get_scene_status_text(scenes_as_dict)
        }

    def execute_propagation(self, params, q, scenes_dict):
        """Execute propagation pipeline."""
        scenes_to_process = [Scene(**s) for s in scenes_dict 
                           if s['status'] == 'included']
        if not scenes_to_process:
            yield {
                self.components['unified_log']: "No scenes were included for propagation.",
                self.components['unified_status']: "Propagation skipped."
            }
            return

        yield {
            self.components['unified_log']: "",
            self.components['unified_status']: f"Starting propagation on {len(scenes_to_process)} scenes..."
        }

        pipeline = AnalysisPipeline(params, q, self.cancel_event,
                                  thumbnail_manager=self.thumbnail_manager)
        yield from self._run_task(
            lambda: pipeline.run_full_analysis(scenes_to_process), q
        )

        result = self.last_task_result
        if result.get("done"):
            yield {
                self.components['unified_log']: "Propagation and analysis complete.",
                self.components['unified_status']: f"Metadata saved to {result['metadata_path']}",
                self.components['analysis_output_dir_state']: result['output_dir'],
                self.components['analysis_metadata_path_state']: result['metadata_path'],
                self.components['filtering_tab']: gr.update(interactive=True)
            }

    def _run_task(self, task_func, progress_queue):
        """Run a task with progress tracking."""
        log_buffer, processed, total, stage = [], 0, 1, "Initializing"
        start_time, last_yield = time.time(), 0.0

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(task_func)
            while future.running():
                if self.cancel_event.is_set():
                    break
                try:
                    msg = progress_queue.get(timeout=0.1)
                    if "log" in msg:
                        log_buffer.append(msg["log"])
                    if "stage" in msg:
                        stage, processed, start_time = msg["stage"], 0, time.time()
                    if "total" in msg:
                        total = msg["total"] or 1
                    if "progress" in msg:
                        processed += msg["progress"]

                    if time.time() - last_yield > 0.25:
                        elapsed = time.time() - start_time
                        rate = processed / elapsed if elapsed > 0 else 0
                        eta = (total - processed) / rate if rate > 0 else 0
                        status = (f"**{stage}:** {processed}/{total} "
                                f"({processed/total:.1%}) | {rate:.1f} items/s | "
                                f"ETA: {int(eta//60):02d}:{int(eta%60):02d}")
                        yield {
                            self.components['unified_log']: "\n".join(log_buffer),
                            self.components['unified_status']: status
                        }
                        last_yield = time.time()
                except Empty:
                    pass

        self.last_task_result = future.result() or {}
        if "log" in self.last_task_result:
            log_buffer.append(self.last_task_result["log"])
        if "error" in self.last_task_result:
            log_buffer.append(f"[ERROR] {self.last_task_result['error']}")
            
        if self.cancel_event.is_set():
            status_text = "⏹️ Cancelled."
        elif 'error' in self.last_task_result:
            status_text = f"❌ Error: {self.last_task_result.get('error')}"
        else:
            status_text = "✅ Complete."
            
        yield {
            self.components['unified_log']: "\n".join(log_buffer),
            self.components['unified_status']: status_text
        }

    def histogram_svg(self, hist_data, title=""):
        """Generate SVG histogram for metrics."""
        if not hist_data:
            return ""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.ticker as mticker
            import io
            
            counts, bins = hist_data
            if (not isinstance(counts, list) or not isinstance(bins, list) or
                len(bins) != len(counts) + 1):
                return ""
                
            with plt.style.context("dark_background"):
                fig, ax = plt.subplots(figsize=(4.6, 2.2), dpi=120)
                ax.bar(bins[:-1], counts, width=np.diff(bins), 
                      color="#7aa2ff", alpha=0.85, align="edge")
                ax.grid(axis="y", alpha=0.2)
                ax.margins(x=0)
                ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
                for side in ("top", "right"):
                    ax.spines[side].set_visible(False)
                ax.tick_params(labelsize=8)
                ax.set_title(title)
                buf = io.StringIO()
                fig.savefig(buf, format="svg", bbox_inches="tight")
                plt.close(fig)
            return buf.getvalue()
        except Exception as e:
            self.logger.error("Failed to generate histogram SVG.", exc_info=True)
            return ""

    def build_all_metric_svgs(self, per_metric_values):
        """Build SVG histograms for all metrics."""
        svgs = {}
        for k in self.get_all_filter_keys():
            if (h := per_metric_values.get(f"{k}_hist")):
                svgs[k] = self.histogram_svg(h, title="")
        return svgs

    @staticmethod
    def _apply_all_filters_vectorized(all_frames_data, filters):
        """Apply all filters to frame data using vectorized operations."""
        from app.core.config import Config
        import imagehash
        
        config = Config()
        
        if not all_frames_data:
            return [], [], Counter(), {}
            
        num_frames = len(all_frames_data)
        filenames = [f['filename'] for f in all_frames_data]
        metric_arrays = {}
        
        for k in config.QUALITY_METRICS:
            metric_arrays[k] = np.array([
                f.get("metrics", {}).get(f"{k}_score", np.nan)
                for f in all_frames_data
            ], dtype=np.float32)
            
        metric_arrays["face_sim"] = np.array([
            f.get("face_sim", np.nan) for f in all_frames_data
        ], dtype=np.float32)
        metric_arrays["mask_area_pct"] = np.array([
            f.get("mask_area_pct", np.nan) for f in all_frames_data
        ], dtype=np.float32)
        
        kept_mask = np.ones(num_frames, dtype=bool)
        reasons = defaultdict(list)

        # --- 1. Deduplication (run first on all frames) ---
        dedup_thresh_val = filters.get("dedup_thresh", 5)
        if filters.get("enable_dedup") and dedup_thresh_val != -1:
            all_indices = list(range(num_frames))
            sorted_indices = sorted(all_indices, key=lambda i: filenames[i])
            hashes = {
                i: imagehash.hex_to_hash(all_frames_data[i]['phash'])
                for i in sorted_indices if 'phash' in all_frames_data[i]
            }

            for i in range(1, len(sorted_indices)):
                current_idx = sorted_indices[i]
                prev_idx = sorted_indices[i-1]
                if prev_idx in hashes and current_idx in hashes:
                    if (hashes[prev_idx] - hashes[current_idx]) <= dedup_thresh_val:
                        kept_mask[current_idx] = False
                        reasons[filenames[current_idx]].append('duplicate')

        # --- 2. Quality & Metric Filters (run on remaining frames) ---
        for k in config.QUALITY_METRICS:
            min_val, max_val = filters.get(f"{k}_min", 0), filters.get(f"{k}_max", 100)
            current_kept_indices = np.where(kept_mask)[0]
            values_to_check = metric_arrays[k][current_kept_indices]

            low_mask_rel = values_to_check < min_val
            high_mask_rel = values_to_check > max_val

            low_indices_abs = current_kept_indices[low_mask_rel]
            high_indices_abs = current_kept_indices[high_mask_rel]

            for i in low_indices_abs:
                reasons[filenames[i]].append(f"{k}_low")
            for i in high_indices_abs:
                reasons[filenames[i]].append(f"{k}_high")

            kept_mask[low_indices_abs] = False
            kept_mask[high_indices_abs] = False

        if filters.get("face_sim_enabled"):
            current_kept_indices = np.where(kept_mask)[0]
            face_sim_values = metric_arrays["face_sim"][current_kept_indices]

            valid = ~np.isnan(face_sim_values)
            low_mask_rel = valid & (face_sim_values < filters.get("face_sim_min", 0.5))
            low_indices_abs = current_kept_indices[low_mask_rel]
            for i in low_indices_abs:
                reasons[filenames[i]].append("face_sim_low")
            kept_mask[low_indices_abs] = False

            if filters.get("require_face_match"):
                missing_mask_rel = ~valid
                missing_indices_abs = current_kept_indices[missing_mask_rel]
                for i in missing_indices_abs:
                    reasons[filenames[i]].append("face_missing")
                kept_mask[missing_indices_abs] = False

        if filters.get("mask_area_enabled"):
            current_kept_indices = np.where(kept_mask)[0]
            mask_area_values = metric_arrays["mask_area_pct"][current_kept_indices]

            small_mask_rel = mask_area_values < filters.get("mask_area_pct_min", 1.0)
            small_indices_abs = current_kept_indices[small_mask_rel]
            for i in small_indices_abs:
                reasons[filenames[i]].append("mask_too_small")
            kept_mask[small_indices_abs] = False

        kept = [all_frames_data[i] for i in np.where(kept_mask)[0]]
        rejected = [all_frames_data[i] for i in np.where(~kept_mask)[0]]
        counts = Counter(r for r_list in reasons.values() for r in r_list)
        return kept, rejected, counts, reasons

    def load_and_prep_filter_data(self, metadata_path):
        """Load and prepare frame data for filtering."""
        if not metadata_path or not Path(metadata_path).exists():
            return [], {}
            
        with Path(metadata_path).open('r', encoding='utf-8') as f:
            try:
                next(f)  # skip header
            except StopIteration:
                return [], {}
            all_frames = [json.loads(line) for line in f if line.strip()]

        metric_values = {}
        for k in self.get_all_filter_keys():
            is_face_sim = k == 'face_sim'
            values = np.asarray([
                f.get(k, f.get("metrics", {}).get(f"{k}_score"))
                for f in all_frames
                if (f.get(k) is not None or 
                    f.get("metrics", {}).get(f"{k}_score") is not None)
            ], dtype=float)
            
            if values.size > 0:
                hist_range = (0, 1) if is_face_sim else (0, 100)
                counts, bins = np.histogram(values, bins=50, range=hist_range)
                metric_values[k] = values.tolist()
                metric_values[f"{k}_hist"] = (counts.tolist(), bins.tolist())
        return all_frames, metric_values

    def _update_gallery(self, all_frames_data, filters, output_dir,
                       gallery_view, show_overlay, overlay_alpha):
        """Update the results gallery based on current filters."""
        kept, rejected, counts, per_frame_reasons = self._apply_all_filters_vectorized(
            all_frames_data, filters or {}
        )
        status_parts = [f"**Kept:** {len(kept)}/{len(all_frames_data)}"]
        if counts:
            rejections = ', '.join([f'{k}: {v}' for k, v in counts.most_common(3)])
            status_parts.append(f"**Rejections:** {rejections}")
        status_text = " | ".join(status_parts)
        
        frames_to_show = rejected if gallery_view == "Rejected Frames" else kept
        preview_images = []
        
        if output_dir:
            output_path = Path(output_dir)
            thumb_dir = output_path / "thumbs"
            masks_dir = output_path / "masks"
            
            for f_meta in frames_to_show[:100]:
                thumb_path = thumb_dir / f"{Path(f_meta['filename']).stem}.webp"
                caption = ""
                if gallery_view == "Rejected Frames":
                    reasons_list = per_frame_reasons.get(f_meta['filename'], [])
                    caption = f"Reasons: {', '.join(reasons_list)}"

                thumb_rgb_np = self.thumbnail_manager.get(thumb_path)
                if thumb_rgb_np is None:
                    continue

                if (show_overlay and not f_meta.get("mask_empty", True) and
                    (mask_name := f_meta.get("mask_path"))):
                    mask_path = masks_dir / mask_name
                    if mask_path.exists():
                        mask_gray = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                        thumb_overlay_rgb = render_mask_overlay(
                            thumb_rgb_np, mask_gray, float(overlay_alpha)
                        )
                        preview_images.append((thumb_overlay_rgb, caption))
                    else:
                        preview_images.append((thumb_rgb_np, caption))
                else:
                    preview_images.append((thumb_rgb_np, caption))

        gallery_rows = 1 if gallery_view == "Rejected Frames" else 2
        return status_text, gr.update(value=preview_images, rows=gallery_rows)

    def on_filters_changed(self, all_frames_data, per_metric_values, output_dir,
                          gallery_view, show_overlay, overlay_alpha,
                          require_face_match, dedup_thresh, *slider_values):
        """Handle filter changes and update gallery."""
        if not all_frames_data:
            return "Run analysis to see results.", []
            
        slider_keys = sorted(self.components['metric_sliders'].keys())
        filters = {key: val for key, val in zip(slider_keys, slider_values)}
        filters.update({
            "require_face_match": require_face_match,
            "dedup_thresh": dedup_thresh,
            "face_sim_enabled": bool(per_metric_values.get("face_sim")),
            "mask_area_enabled": bool(per_metric_values.get("mask_area_pct")),
            "enable_dedup": (any('phash' in f for f in all_frames_data)
                           if all_frames_data else False)
        })
        return self._update_gallery(all_frames_data, filters, output_dir,
                                  gallery_view, show_overlay, overlay_alpha)

    def export_kept_frames(self, all_frames_data, output_dir, video_path,
                          enable_crop, crop_ars, crop_padding, *filter_args):
        """Export filtered frames to a new directory."""
        if not all_frames_data:
            return "No metadata to export."
        if not video_path or not Path(video_path).exists():
            return "[ERROR] Original video path is required for export."
            
        try:
            slider_keys = sorted(self.components['metric_sliders'].keys())
            require_face_match, dedup_thresh, *slider_values = filter_args
            filters = {key: val for key, val in zip(slider_keys, slider_values)}
            filters.update({
                "require_face_match": require_face_match,
                "dedup_thresh": dedup_thresh,
                "face_sim_enabled": any("face_sim" in f for f in all_frames_data),
                "mask_area_enabled": any("mask_area_pct" in f for f in all_frames_data),
                "enable_dedup": any('phash' in f for f in all_frames_data)
            })

            kept, _, _, _ = self._apply_all_filters_vectorized(all_frames_data, filters)
            if not kept:
                return "No frames kept after filtering. Nothing to export."

            out_root = Path(output_dir)
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

            cmd = ['ffmpeg', '-y', '-i', str(video_path), '-vf', select_filter,
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

    def reset_filters(self, all_frames_data, per_metric_values, output_dir):
        """Reset all filters to default values."""
        output_values = []
        slider_default_values = []
        slider_keys = sorted(self.components['metric_sliders'].keys())
        
        for key in slider_keys:
            metric_key = re.sub(r'_(min|max)$', '', key)
            default_key = 'default_max' if key.endswith('_max') else 'default_min'
            default_val = self.config.filter_defaults[metric_key][default_key]
            output_values.append(gr.update(value=default_val))
            slider_default_values.append(default_val)

        face_match_default = self.config.ui_defaults['require_face_match']
        dedup_default = self.config.filter_defaults['dedup_thresh']['default']
        output_values.append(gr.update(value=face_match_default))
        output_values.append(gr.update(value=dedup_default))

        if all_frames_data:
            status_text, gallery_update = self.on_filters_changed(
                all_frames_data, per_metric_values, output_dir, "Kept Frames",
                True, 0.6, face_match_default, dedup_default,
                *slider_default_values
            )
            output_values.extend([status_text, gallery_update])
        else:
            output_values.extend(["Load an analysis to begin.", []])
        return output_values

    def auto_set_thresholds(self, per_metric_values, p=75):
        """Auto-set filter thresholds based on percentiles."""
        slider_keys = sorted(self.components['metric_sliders'].keys())
        updates = [gr.update() for _ in slider_keys]
        if not per_metric_values:
            return updates
            
        pmap = {
            k: float(np.percentile(np.asarray(vals, dtype=np.float32), p))
            for k, vals in per_metric_values.items()
            if not k.endswith('_hist') and vals
        }
        
        for i, key in enumerate(slider_keys):
            if key.endswith('_min'):
                metric = key[:-4]
                if metric in pmap:
                    updates[i] = gr.update(value=round(pmap[metric], 2))
        return updates

    def _get_scene_status_text(self, scenes_list):
        """Get status text for scenes."""
        if not scenes_list:
            return "No scenes loaded."
        num_included = sum(1 for s in scenes_list if s['status'] == 'included')
        return f"{num_included}/{len(scenes_list)} scenes included for propagation."

    def _toggle_scene_status(self, scenes_list, selected_shot_id, new_status,
                           output_folder):
        """Toggle the status of a specific scene."""
        if selected_shot_id is None or not scenes_list:
            return (scenes_list, self._get_scene_status_text(scenes_list),
                   "No scene selected.")

        scene_found = False
        for s in scenes_list:
            if s['shot_id'] == selected_shot_id:
                s['status'] = new_status
                s['manual_status_change'] = True
                scene_found = True
                break

        if scene_found:
            self._save_scene_seeds(scenes_list, output_folder)
            return (scenes_list, self._get_scene_status_text(scenes_list),
                   f"Scene {selected_shot_id} status set to {new_status}.")
        else:
            return (scenes_list, self._get_scene_status_text(scenes_list),
                   f"Could not find scene {selected_shot_id}.")

    def apply_bulk_scene_filters(self, scenes, min_mask_area, min_face_sim,
                                enable_face_filter, output_folder):
        """Apply bulk filters to scenes."""
        if not scenes:
            return [], "No scenes to filter."

        for scene in scenes:
            if scene.get('manual_status_change'):
                continue

            is_excluded = False
            seed_result = scene.get('seed_result', {})
            details = seed_result.get('details', {})

            mask_area = details.get('mask_area_pct', 101)
            if mask_area < min_mask_area:
                is_excluded = True

            if enable_face_filter and not is_excluded:
                face_sim = details.get('seed_face_sim', 1.01)
                if face_sim < min_face_sim:
                    is_excluded = True

            scene['status'] = 'excluded' if is_excluded else 'included'

        self._save_scene_seeds(scenes, output_folder)
        return scenes, self._get_scene_status_text(scenes)

    def apply_scene_overrides(self, scenes_list, selected_shot_id, prompt,
                            box_th, text_th, output_folder, *ana_args):
        """Apply overrides to a specific scene."""
        if selected_shot_id is None or not scenes_list:
            return (gr.update(), scenes_list,
                   "No scene selected to apply overrides.")

        scene_idx, scene_dict = next(
            ((i, s) for i, s in enumerate(scenes_list)
             if s['shot_id'] == selected_shot_id), (None, None)
        )
        if scene_dict is None:
            return (gr.update(), scenes_list,
                   "Error: Selected scene not found in state.")

        try:
            scene_dict['seed_config'] = {
                'text_prompt': prompt,
                'box_threshold': box_th,
                'text_threshold': text_th,
            }

            ui_args = dict(zip(self.ana_ui_map_keys, ana_args))
            ui_args['output_folder'] = output_folder
            params = AnalysisParameters.from_ui(**ui_args)

            face_analyzer, ref_emb, person_detector = None, None, None
            device = "cuda" if self.cuda_available else "cpu"
            if params.enable_face_filter:
                from app.ml.face import get_face_analyzer
                face_analyzer = get_face_analyzer(params.face_model_name)
                if params.face_ref_img_path:
                    ref_img = cv2.imread(params.face_ref_img_path)
                    if ref_img is not None:
                        faces = face_analyzer.get(ref_img)
                        if faces:
                            ref_emb = max(faces, key=lambda x: x.det_score).normed_embedding
            from app.ml.person import get_person_detector
            person_detector = get_person_detector(params.person_detector_model, device)

            masker = SubjectMasker(params, Queue(), threading.Event(),
                                 face_analyzer=face_analyzer,
                                 reference_embedding=ref_emb,
                                 person_detector=person_detector,
                                 thumbnail_manager=self.thumbnail_manager)
            masker.frame_map = masker._create_frame_map(output_folder)

            fname = masker.frame_map.get(scene_dict['best_seed_frame'])
            if not fname:
                raise ValueError("Framemap lookup failed for re-seeding.")

            thumb_path = (Path(output_folder) / "thumbs" /
                         f"{Path(fname).stem}.webp")
            thumb_rgb = self.thumbnail_manager.get(thumb_path)

            bbox, details = masker.get_seed_for_frame(thumb_rgb,
                                                    scene_dict['seed_config'])
            scene_dict['seed_result'] = {'bbox': bbox, 'details': details}

            self._save_scene_seeds(scenes_list, output_folder)

            updated_gallery_previews = self._regenerate_all_previews(
                scenes_list, output_folder, masker
            )

            return (updated_gallery_previews, scenes_list,
                   f"Scene {selected_shot_id} updated and saved.")

        except Exception as e:
            self.logger.error("Failed to apply scene overrides", exc_info=True)
            return gr.update(), scenes_list, f"[ERROR] {e}"

    def _regenerate_all_previews(self, scenes_list, output_folder, masker):
        """Regenerate all scene previews."""
        previews = []
        output_dir = Path(output_folder)
        
        for scene_dict in scenes_list:
            fname = masker.frame_map.get(scene_dict['best_seed_frame'])
            if not fname:
                continue

            thumb_path = output_dir / "thumbs" / f"{Path(fname).stem}.webp"
            thumb_rgb = self.thumbnail_manager.get(thumb_path)
            if thumb_rgb is None:
                continue

            bbox = scene_dict.get('seed_result', {}).get('bbox')
            details = scene_dict.get('seed_result', {}).get('details', {})

            mask = masker.get_mask_for_bbox(thumb_rgb, bbox) if bbox else None
            overlay_rgb = (render_mask_overlay(thumb_rgb, mask)
                          if mask is not None
                          else masker.draw_bbox(thumb_rgb, bbox))

            caption = (f"Scene {scene_dict['shot_id']} "
                      f"(Seed: {scene_dict['best_seed_frame']}) | "
                      f"{details.get('type', 'N/A')}")
            previews.append((overlay_rgb, caption))
        return previews

    def _parse_ar(self, s: str) -> tuple[int, int]:
        """Parse aspect ratio string."""
        try:
            if isinstance(s, str) and ":" in s:
                w_str, h_str = s.split(":", 1)
                return max(int(w_str), 1), max(int(h_str), 1)
        except Exception:
            pass
        return 1, 1

    def _crop_frame(self, img: np.ndarray, mask: np.ndarray, crop_ars: str,
                   padding: int) -> np.ndarray:
        """Crop frame to subject with aspect ratio constraints."""
        h, w = img.shape[:2]
        if mask is None:
            return img
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask = (mask > 128).astype(np.uint8)
        ys, xs = np.where(mask > 0)
        if ys.size == 0:
            return img
            
        x1, x2, y1, y2 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
        bw, bh = x2 - x1, y2 - y1
        pad_x = int(round(bw * padding/100.0))
        pad_y = int(round(bh * padding/100.0))
        x1, y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
        x2, y2 = min(w, x2 + pad_x), min(h, y2 + pad_y)
        bw, bh = x2 - x1, y2 - y1
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        ars = [self._parse_ar(s.strip()) for s in str(crop_ars).split(',')
               if s.strip()]
        if not ars:
            return img[y1:y2, x1:x2]

        def expand_to_ar(r):
            if bw / (bh + 1e-9) < r:
                new_w, new_h = int(np.ceil(bh * r)), bh
            else:
                new_w, new_h = bw, int(np.ceil(bw / r))
            if new_w > w or new_h > h:
                return None
            x1n, y1n = int(round(cx - new_w / 2)), int(round(cy - new_h / 2))
            if x1n < 0:
                x1n = 0
            if y1n < 0:
                y1n = 0
            if x1n + new_w > w:
                x1n = w - new_w
            if y1n + new_h > h:
                y1n = h - new_h
            x2n, y2n = x1n + new_w, y1n + new_h
            if x1n > x1 or y1n > y1 or x2n < x2 or y2n < y2:
                return None
            return (x1n, y1n, x2n, y2n, (new_w * new_h) / max(1, bw * bh))

        candidates = []
        for ar in ars:
            r_w, r_h = (ar if isinstance(ar, (tuple, list)) and len(ar) == 2
                       else (1, 1))
            if r_h > 0:
                res = expand_to_ar(r_w / r_h)
                if res:
                    candidates.append(res)
        if candidates:
            x1n, y1n, x2n, y2n, _ = sorted(candidates, key=lambda t: t[4])[0]
            return img[y1n:y2n, x1n:x2n]
        return img[y1:y2, x1:x2]
