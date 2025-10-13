import json
import shutil
import threading
from abc import ABC, abstractmethod
from dataclasses import asdict
from pathlib import Path
from queue import Queue

import cv2
import gradio as gr
import numpy as np

from app.config import Config
from app.events import (ExtractionEvent, PreAnalysisEvent, PropagationEvent,
                              SessionLoadEvent)
from app.extract import ExtractionPipeline as ExtractionTool
from app.frames import create_frame_map, render_mask_overlay
from app.logging import UnifiedLogger
from app.models import AnalysisParameters, Scene
from app.scene_logic import get_scene_status_text


class Pipeline(ABC):
    """Base class for all pipelines."""

    def __init__(self, params: AnalysisParameters, progress_queue: Queue,
                 cancel_event: threading.Event, logger: UnifiedLogger,
                 config: Config, thumbnail_manager):
        self.params = params
        self.progress_queue = progress_queue
        self.cancel_event = cancel_event
        self.logger = logger
        self.config = config
        self.thumbnail_manager = thumbnail_manager

    @abstractmethod
    def run(self):
        """Run the pipeline."""
        pass


class ExtractionPipeline(Pipeline):
    """Pipeline for extracting frames from a video."""

    def run(self):
        """Run the extraction pipeline."""
        self.progress_queue.put({"stage": "Starting extraction...", "total": 1, "progress": 0})

        pipeline = ExtractionTool(self.params, self.progress_queue, self.cancel_event)
        result = pipeline.run()

        if result.get("done"):
            return {
                "log": "Extraction complete.",
                "status": f"Output: {result['output_dir']}",
                "extracted_video_path_state": result.get("video_path", ""),
                "extracted_frames_dir_state": result["output_dir"],
                "done": True
            }
        return {"done": False}


class PreAnalysisPipeline(Pipeline):
    """Pipeline for pre-analyzing scenes."""

    def run(self):
        """Run the pre-analysis pipeline."""
        import pyiqa
        from app.subject_masker import SubjectMasker

        yield {"unified_log": "", "unified_status": "Starting Pre-Analysis...", "progress_bar": gr.update(visible=True, value=0)}

        if self.params.face_ref_img_upload:
            ref_upload = self.params.face_ref_img_upload
            dest = self.config.DIRS['downloads'] / Path(ref_upload).name
            shutil.copy2(ref_upload, dest)
            self.params.face_ref_img_path = str(dest)

        output_dir = Path(self.params.output_folder)

        # Save the run configuration
        run_config_path = output_dir / "run_config.json"
        try:
            config_to_save = asdict(self.params)
            config_to_save.pop('face_ref_img_upload', None)
            with run_config_path.open('w', encoding='utf-8') as f:
                json.dump(config_to_save, f, indent=4)
        except Exception as e:
            self.logger.error(f"Failed to save run configuration: {e}", exc_info=True)

        scenes_path = output_dir / "scenes.json"
        if not scenes_path.exists():
            yield {"log": "[ERROR] scenes.json not found. Run extraction with scene detection."}
            return

        with scenes_path.open('r', encoding='utf-8') as f:
            shots = json.load(f)
        scenes = [Scene(shot_id=i, start_frame=s, end_frame=e) for i, (s, e) in enumerate(shots)]

        scene_seeds_path = output_dir / "scene_seeds.json"
        if scene_seeds_path.exists() and self.params.resume:
            with scene_seeds_path.open('r', encoding='utf-8') as f:
                loaded_seeds = json.load(f)
            for scene in scenes:
                if str(scene.shot_id) in loaded_seeds:
                    seed_data = loaded_seeds[str(scene.shot_id)]
                    scene.best_seed_frame = seed_data.get('seed_frame_idx')
                    scene.seed_config = seed_data.get('seed_config', {})
                    scene.status = seed_data.get('status', 'pending')
                    scene.seed_metrics = seed_data.get('seed_metrics', {})

        device = "cuda" if self.params.cuda_available else "cpu"
        niqe_metric = pyiqa.create_metric('niqe', device=device) if self.params.pre_analysis_enabled else None
        face_analyzer, ref_emb = None, None
        if self.params.enable_face_filter:
            from app.face import get_face_analyzer
            face_analyzer = get_face_analyzer(self.params.face_model_name)
            if self.params.face_ref_img_path and Path(self.params.face_ref_img_path).exists():
                ref_img = cv2.imread(self.params.face_ref_img_path)
                faces = face_analyzer.get(ref_img)
                if faces:
                    ref_emb = max(faces, key=lambda x: x.det_score).normed_embedding

        from app.person import get_person_detector
        person_detector = get_person_detector(self.params.person_detector_model, device)

        masker = SubjectMasker(self.params, self.progress_queue, self.cancel_event, face_analyzer=face_analyzer,
                             reference_embedding=ref_emb, person_detector=person_detector,
                             niqe_metric=niqe_metric, thumbnail_manager=self.thumbnail_manager)
        masker.frame_map = masker._create_frame_map(str(output_dir))

        self.progress_queue.put({"stage": "Pre-analyzing scenes", "total": len(scenes)})
        previews = []
        for scene in scenes:
            if self.cancel_event.is_set(): break
            if not scene.best_seed_frame:
                masker._select_best_seed_frame_in_scene(scene, str(output_dir))

            fname = masker.frame_map.get(scene.best_seed_frame)
            if not fname: continue

            thumb_path = output_dir / "thumbs" / f"{Path(fname).stem}.webp"
            thumb_rgb = self.thumbnail_manager.get(thumb_path)
            if thumb_rgb is None: continue

            bbox, details = masker.get_seed_for_frame(thumb_rgb, seed_config=scene.seed_config or self.params)
            scene.seed_result = {'bbox': bbox, 'details': details}
            mask = masker.get_mask_for_bbox(thumb_rgb, bbox) if bbox else None
            if mask is not None:
                h, w = mask.shape[:2]
                area_pct = (np.sum(mask > 0) / (h * w)) * 100 if h * w > 0 else 0.0
                scene.seed_result['details']['mask_area_pct'] = area_pct

            overlay_rgb = (render_mask_overlay(thumb_rgb, mask) if mask is not None else masker.draw_bbox(thumb_rgb, bbox))
            caption = f"Scene {scene.shot_id} (Seed: {scene.best_seed_frame}) | {details.get('type', 'N/A')}"
            previews.append((overlay_rgb, caption))
            if scene.status == 'pending': scene.status = 'included'
            self.progress_queue.put({"progress": 1})

        yield {
            "log": "Pre-analysis complete.",
            "status": f"{len(scenes)} scenes found.",
            "previews": previews,
            "scenes": [asdict(s) for s in scenes],
            "output_dir": str(output_dir),
            "done": True
        }


class PropagationPipeline(Pipeline):
    """Pipeline for propagating masks."""

    def run(self):
        """Run the propagation pipeline."""
        from app.analyze import AnalysisPipeline as AnalysisTool

        scenes_to_process = [Scene(**s) for s in self.params.scenes if s['status'] == 'included']
        if not scenes_to_process:
            yield {"log": "No scenes were included for propagation.", "status": "Propagation skipped."}
            return

        self.progress_queue.put({"stage": f"Starting propagation on {len(scenes_to_process)} scenes...", "total": 1, "progress": 0})

        pipeline = AnalysisTool(self.params.analysis_params, self.progress_queue, self.cancel_event, thumbnail_manager=self.thumbnail_manager)
        result = pipeline.run_full_analysis(scenes_to_process)

        if result.get("done"):
            yield {
                "log": "Propagation and analysis complete.",
                "status": f"Metadata saved to {result['metadata_path']}",
                "output_dir": result['output_dir'],
                "metadata_path": result['metadata_path'],
                "done": True
            }


class SessionLoadPipeline(Pipeline):
    """Pipeline for loading a session."""

    def run(self):
        """Run the session load pipeline."""
        session_path = Path(self.params.session_path)
        config_path = session_path / "run_config.json"

        if not config_path.exists():
            self.logger.error(f"Session load failed: run_config.json not found in {session_path}")
            yield {
                "log": f"[ERROR] Could not find 'run_config.json' in the specified directory: {session_path}",
                "status": "Session load failed."
            }
            return

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                run_config = json.load(f)
            self.logger.info(f"Loaded run configuration from {config_path}")

            updates = {
                'source_input': gr.update(value=run_config.get('source_path', '')),
                'max_resolution': gr.update(value=run_config.get('max_resolution', '1080')),
                'thumbnails_only_input': gr.update(value=run_config.get('thumbnails_only', True)),
                'thumb_megapixels_input': gr.update(value=run_config.get('thumb_megapixels', 0.5)),
                'ext_scene_detect_input': gr.update(value=run_config.get('scene_detect', True)),
                'method_input': gr.update(value=run_config.get('method', 'scene')),
                'use_png_input': gr.update(value=run_config.get('use_png', False)),
                'pre_analysis_enabled_input': gr.update(value=run_config.get('pre_analysis_enabled', True)),
                'pre_sample_nth_input': gr.update(value=run_config.get('pre_sample_nth', 1)),
                'enable_face_filter_input': gr.update(value=run_config.get('enable_face_filter', False)),
                'face_model_name_input': gr.update(value=run_config.get('face_model_name', 'buffalo_l')),
                'face_ref_img_path_input': gr.update(value=run_config.get('face_ref_img_path', '')),
                'text_prompt_input': gr.update(value=run_config.get('text_prompt', '')),
                'seed_strategy_input': gr.update(value=run_config.get('seed_strategy', 'Largest Person')),
                'person_detector_model_input': gr.update(value=run_config.get('person_detector_model', 'yolo11x.pt')),
                'dam4sam_model_name_input': gr.update(value=run_config.get('dam4sam_model_name', 'sam21pp-T')),
                'enable_dedup_input': gr.update(value=run_config.get('enable_dedup', False)),
                'extracted_video_path_state': run_config.get('video_path', ''),
                'extracted_frames_dir_state': run_config.get('output_folder', ''),
                'analysis_output_dir_state': run_config.get('output_folder', ''),
            }

            scene_seeds_path = session_path / "scene_seeds.json"
            if scene_seeds_path.exists():
                with open(scene_seeds_path, 'r', encoding='utf-8') as f:
                    scenes_from_file = json.load(f)

                scenes_as_dict = []
                for shot_id, scene_data in scenes_from_file.items():
                    scene_data['shot_id'] = int(shot_id)
                    scenes_as_dict.append(scene_data)

                updates['scenes_state'] = scenes_as_dict
                updates['propagate_masks_button'] = gr.update(interactive=True)
                self.logger.info(f"Loaded {len(scenes_as_dict)} scenes from {scene_seeds_path}")

                output_dir = Path(run_config.get('output_folder'))
                frame_map = create_frame_map(output_dir)
                previews = []
                for scene in scenes_as_dict:
                    seed_frame_idx = scene.get('seed_frame_idx')
                    if seed_frame_idx is not None:
                        fname = frame_map.get(seed_frame_idx)
                        if not fname:
                            self.logger.warning(f"Could not find seed_frame_idx {seed_frame_idx} in frame_map for scene {scene['shot_id']}")
                            continue

                        thumb_path = output_dir / "thumbs" / f"{Path(fname).stem}.webp"
                        thumb_rgb = self.thumbnail_manager.get(thumb_path)

                        if thumb_rgb is None:
                            self.logger.warning(f"Could not load thumbnail for seed_frame_idx {seed_frame_idx} at path {thumb_path}")
                            continue

                        caption = f"Scene {scene['shot_id']} (Seed: {seed_frame_idx})"
                        previews.append((thumb_rgb, caption))

                updates['seeding_preview_gallery'] = gr.update(value=previews)
                updates['scene_filter_status'] = get_scene_status_text(scenes_as_dict)

                has_face_sim = any(
                    s.get('seed_metrics', {}).get('best_face_sim') is not None
                    for s in scenes_as_dict
                )
                updates['scene_face_sim_min_input'] = gr.update(visible=has_face_sim)

            metadata_path = session_path / "metadata.json"
            if metadata_path.exists():
                updates['analysis_metadata_path_state'] = str(metadata_path)
                updates['filtering_tab'] = gr.update(interactive=True)
                self.logger.info(f"Found analysis metadata at {metadata_path}. Filtering tab will be enabled.")

            updates['unified_log'] = f"Successfully loaded session from: {session_path}"
            updates['unified_status'] = "Session loaded. You can now proceed from where you left off."

            yield updates

        except Exception as e:
            self.logger.error(f"Error loading session from {session_path}: {e}", exc_info=True)
            yield {
            "unified_log": f"[ERROR] An unexpected error occurred while loading the session: {e}",
            "unified_status": "Session load failed."
            }