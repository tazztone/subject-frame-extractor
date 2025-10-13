import threading
from queue import Queue

import gradio as gr
import shutil
from pathlib import Path

from app.config import Config
from app.events import (ExtractionEvent, PreAnalysisEvent, PropagationEvent,
                              FilterEvent, ExportEvent, SessionLoadEvent)
from app.logging import UnifiedLogger
from app.pipeline_logic import (execute_extraction, execute_pre_analysis,
                                      execute_propagation, execute_session_load)
from app.filter_logic import (load_and_prep_filter_data, build_all_metric_svgs,
                                    on_filters_changed, reset_filters,
                                    auto_set_thresholds, apply_all_filters_vectorized)
from app.scene_logic import (toggle_scene_status, apply_bulk_scene_filters,
                                   apply_scene_overrides, get_scene_status_text,
                                   save_scene_seeds)
from app.thumb_cache import ThumbnailManager


class Backend:
    """Backend class for the application."""

    def __init__(self, config: Config, logger: UnifiedLogger,
                 progress_queue: Queue, cancel_event: threading.Event,
                 thumbnail_manager: ThumbnailManager, cuda_available: bool):
        self.config = config
        self.logger = logger
        self.progress_queue = progress_queue
        self.cancel_event = cancel_event
        self.thumbnail_manager = thumbnail_manager
        self.cuda_available = cuda_available

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
            'unified_log', 'unified_status',
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
    def _yield_gradio_updates(self, logic_generator, output_keys):
        """Helper to convert dicts from logic to tuples for Gradio."""
        for result_dict in logic_generator:
            yield tuple(result_dict.get(k, gr.update()) for k in output_keys)

    def run_extraction_wrapper(self, *args):
        """Wrapper for extraction pipeline."""
        ui_args = dict(zip(self.ext_ui_map_keys, args))

        if ui_args.get('upload_video'):
            source = ui_args.pop('upload_video')
            dest = str(self.config.DIRS['downloads'] / Path(source).name)
            shutil.copy2(source, dest)
            ui_args['source_path'] = dest

        event = ExtractionEvent(**ui_args)

        # Execute the pipeline and collect results
        final_result = {}
        for result in execute_extraction(event, self.progress_queue,
                                         self.cancel_event, self.logger, self.config):
            # Only update final_result if result is a dictionary
            if isinstance(result, dict):
                final_result.update(result)  # Merge instead of overwrite

        # Safe access with fallbacks
        if final_result.get("done"):
            video_path = final_result.get("extracted_video_path_state", "") or final_result.get("video_path", "")
            frames_dir = final_result.get("extracted_frames_dir_state", "") or final_result.get("output_dir", "")

            return {
                "unified_log": final_result.get("log", "✅ Extraction completed successfully."),
                "extracted_video_path_state": video_path,
                "extracted_frames_dir_state": frames_dir,
                "main_tabs": gr.update(selected=1),
            }

        return {
            "unified_log": final_result.get("log", "❌ Extraction failed or was cancelled."),
        }

    def run_pre_analysis_wrapper(self, *args):
        """Wrapper for pre-analysis pipeline."""
        ui_args = dict(zip(self.ana_ui_map_keys, args))

        # Adapt ui_args based on the primary seeding strategy
        strategy = ui_args.pop('primary_seed_strategy', '🤖 Automatic')
        if strategy == "👤 By Face":
            ui_args['enable_face_filter'] = True
            ui_args['text_prompt'] = ""
        elif strategy == "📝 By Text":
            ui_args['enable_face_filter'] = False
            ui_args['face_ref_img_path'] = ""
            ui_args['face_ref_img_upload'] = None
        elif strategy == "🔄 Face + Text Fallback":
            ui_args['enable_face_filter'] = True
            # Keep both face reference and text prompt
            # Don't clear either - both are needed for fallback
        elif strategy == "🤖 Automatic":
            ui_args['enable_face_filter'] = False
            ui_args['text_prompt'] = ""
            ui_args['face_ref_img_path'] = ""
            ui_args['face_ref_img_upload'] = None

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
            has_face_sim = any(
                s.get('seed_metrics', {}).get('best_face_sim') is not None
                for s in scenes
            )
            return {
                "unified_log": final_result.get("log", "✅ Pre-analysis completed successfully."),
                "seeding_preview_gallery": gr.update(value=final_result.get('previews')),
                "scenes_state": scenes,
                "propagate_masks_button": gr.update(interactive=True),
                "scene_filter_status": status_text,
                "scene_face_sim_min_input": gr.update(visible=has_face_sim),
                "seeding_results_column": gr.update(visible=True),
                "propagation_group": gr.update(visible=True),
            }
        else:
            return {
                "unified_log": final_result.get("log", "❌ Pre-analysis failed or was cancelled."),
            }

    def run_propagation_wrapper(self, scenes, *args):
        """Wrapper for propagation pipeline."""
        ui_args = dict(zip(self.ana_ui_map_keys, args))

        # Adapt ui_args based on the primary seeding strategy
        strategy = ui_args.pop('primary_seed_strategy', '🤖 Automatic')
        if strategy == "👤 By Face":
            ui_args['enable_face_filter'] = True
            ui_args['text_prompt'] = ""
        elif strategy == "📝 By Text":
            ui_args['enable_face_filter'] = False
            ui_args['face_ref_img_path'] = ""
            ui_args['face_ref_img_upload'] = None
        elif strategy == "🔄 Face + Text Fallback":
            ui_args['enable_face_filter'] = True
            # Keep both face reference and text prompt
            # Don't clear either - both are needed for fallback
        elif strategy == "🤖 Automatic":
            ui_args['enable_face_filter'] = False
            ui_args['text_prompt'] = ""
            ui_args['face_ref_img_path'] = ""
            ui_args['face_ref_img_upload'] = None

        analysis_params = PreAnalysisEvent(**ui_args)
        event = PropagationEvent(
            output_folder=ui_args['output_folder'],
            video_path=ui_args['video_path'],
            scenes=scenes,
            analysis_params=analysis_params
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
                "unified_log": final_result.get("log", "✅ Propagation completed successfully."),
                "analysis_output_dir_state": final_result.get('output_dir', ""),
                "analysis_metadata_path_state": final_result.get('metadata_path', ""),
                "filtering_tab": gr.update(interactive=True),
                "main_tabs": gr.update(selected=2),
            }
        else:
            return {
                "unified_log": final_result.get("log", "❌ Propagation failed or was cancelled."),
            }

    def run_session_load_wrapper(self, session_path):
        """Wrapper for session loading."""
        event = SessionLoadEvent(session_path=session_path)

        logic_gen = execute_session_load(event, self.progress_queue,
                                         self.cancel_event, self.logger, self.config,
                                         self.thumbnail_manager)

        yield from self._yield_gradio_updates(logic_gen, self.session_load_keys)


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