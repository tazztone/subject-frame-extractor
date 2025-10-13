import threading
from queue import Queue

import gradio as gr

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

            return (
                final_result.get("log", "✅ Extraction completed successfully."),
                video_path,
                frames_dir,
                gr.update(selected=1)
            )

        return (
            final_result.get("log", "❌ Extraction failed or was cancelled."),
            "",
            "",
            gr.update()
        )

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
            return (
                final_result.get("log", "✅ Pre-analysis completed successfully."),
                gr.update(value=final_result.get('previews')),
                scenes,
                gr.update(interactive=True),
                status_text,
                gr.update(visible=has_face_sim),
                gr.update(visible=True),
                gr.update(visible=True)
            )
        else:
            return (
                final_result.get("log", "❌ Pre-analysis failed or was cancelled."),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update()
            )

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
            return (
                final_result.get("log", "✅ Propagation completed successfully."),
                final_result.get('output_dir', ""),
                final_result.get('metadata_path', ""),
                gr.update(interactive=True),
                gr.update(selected=2)
            )
        else:
            return (
                final_result.get("log", "❌ Propagation failed or was cancelled."),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update()
            )

    def run_session_load_wrapper(self, session_path):
        """Wrapper for session loading."""
        event = SessionLoadEvent(session_path=session_path)

        logic_gen = execute_session_load(event, self.logger, self.config,
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

                        all_points = np.concatenate(contours)
                        x, y, w, h = cv2.boundingRect(all_points)

                        frame_h, frame_w = frame_img.shape[:2]
                        mask_h, mask_w = mask_img.shape[:2]

                        if mask_h != frame_h or mask_w != frame_w:
                            self.logger.info(f"Scaling bounding box for {frame_meta['filename']} from {mask_w}x{mask_h} to {frame_w}x{frame_h}")
                            scale_x = frame_w / mask_w
                            scale_y = frame_h / mask_h
                            x, y, w, h = int(x * scale_x), int(y * scale_y), int(w * scale_x), int(h * scale_y)

                        padding_px_w = int(w * (event.crop_padding / 100.0))
                        padding_px_h = int(h * (event.crop_padding / 100.0))

                        x1 = max(0, x - padding_px_w)
                        y1 = max(0, y - padding_px_h)
                        x2 = min(frame_w, x + w + padding_px_w)
                        y2 = min(frame_h, y + h + padding_px_h)

                        x_pad, y_pad = x1, y1
                        w_pad, h_pad = x2 - x1, y2 - y1

                        if w_pad <= 0 or h_pad <= 0: continue

                        center_x, center_y = x_pad + w_pad // 2, y_pad + h_pad // 2

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