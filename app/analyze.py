"""Video frame analysis pipeline."""

import json
import logging
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from dataclasses import asdict

import cv2
import numpy as np
import torch

from app.base import Pipeline
from app.thumb_cache import ThumbnailManager
from app.logging import StructuredFormatter
from app.utils import _to_json_safe
from app.face import get_face_analyzer
from app.person import get_person_detector
from app.subject_masker import SubjectMasker
from app.frames import create_frame_map
from app.models import Frame


class AnalysisPipeline(Pipeline):
    """Pipeline for analyzing extracted video frames."""
    
    def __init__(self, params, progress_queue, cancel_event, 
                 thumbnail_manager=None):
        
        super().__init__(params, progress_queue, cancel_event)
        self.output_dir = Path(self.params.output_folder)
        self.thumb_dir = self.output_dir / "thumbs"
        self.masks_dir = self.output_dir / "masks"
        self.frame_map_path = self.output_dir / "frame_map.json"
        self.metadata_path = self.output_dir / "metadata.jsonl"
        self.write_lock = threading.Lock()
        self.gpu_lock = threading.Lock()
        self.face_analyzer = None
        self.reference_embedding = None
        self.mask_metadata = {}
        self.scene_map = {}
        self.niqe_metric = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.thumbnail_manager = (thumbnail_manager if thumbnail_manager 
                                is not None else ThumbnailManager())

    def _initialize_niqe_metric(self):
        """Initialize NIQE quality metric."""
        if self.niqe_metric is None:
            try:
                import pyiqa
                self.niqe_metric = pyiqa.create_metric('niqe', 
                                                      device=self.device)
                self.logger.info("NIQE metric initialized successfully")
            except Exception as e:
                self.logger.warning("Failed to initialize NIQE metric", 
                                  extra={'error': e})

    def run_full_analysis(self, scenes_to_process):
        """Run complete analysis pipeline."""
        
        run_log_handler = None
        try:
            run_log_path = self.output_dir / "analysis_run.log"
            run_log_handler = logging.FileHandler(run_log_path, mode='w', 
                                                 encoding='utf-8')
            formatter = StructuredFormatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            )
            run_log_handler.setFormatter(formatter)
            self.logger.add_handler(run_log_handler)

            self.metadata_path.unlink(missing_ok=True)
            with self.metadata_path.open('w', encoding='utf-8') as f:
                header = {"params": asdict(self.params)}
                f.write(json.dumps(_to_json_safe(header)) + '\n')

            self.scene_map = {s.shot_id: s for s in scenes_to_process}

            if self.params.enable_face_filter:
                self.face_analyzer = get_face_analyzer(
                    self.params.face_model_name
                )
                if self.params.face_ref_img_path:
                    self._process_reference_face()

            person_detector = get_person_detector(
                self.params.person_detector_model, self.device
            )

            masker = SubjectMasker(
                self.params, self.progress_queue, self.cancel_event,
                self._create_frame_map(), self.face_analyzer,
                self.reference_embedding, person_detector,
                thumbnail_manager=self.thumbnail_manager,
                niqe_metric=self.niqe_metric
            )
            self.mask_metadata = masker.run_propagation(
                str(self.output_dir), scenes_to_process
            )

            self._run_analysis_loop(scenes_to_process)

            if self.cancel_event.is_set():
                return {"log": "Analysis cancelled."}
                
            self.logger.success("Analysis complete.", 
                              extra={'output_dir': self.output_dir})
            return {
                "done": True,
                "metadata_path": str(self.metadata_path),
                "output_dir": str(self.output_dir)
            }
        except Exception as e:
            return self.logger.pipeline_error("analysis", e)
        finally:
            self.logger.remove_handler(run_log_handler)

    def _create_frame_map(self):
        """Create frame mapping from output directory."""
        return create_frame_map(self.output_dir)

    def _process_reference_face(self):
        """Process reference face image for similarity matching."""
        if not self.face_analyzer:
            return
            
        ref_path = Path(self.params.face_ref_img_path)
        if not ref_path.is_file():
            raise FileNotFoundError(
                f"Reference face image not found: {ref_path}"
            )
            
        self.logger.info("Processing reference face...")
        ref_img = cv2.imread(str(ref_path))  # Reads as BGR
        if ref_img is None:
            raise ValueError("Could not read reference image.")
            
        ref_faces = self.face_analyzer.get(ref_img)
        if not ref_faces:
            raise ValueError("No face found in reference image.")
            
        self.reference_embedding = max(
            ref_faces, key=lambda x: x.det_score
        ).normed_embedding
        self.logger.success("Reference face processed.")

    def _run_analysis_loop(self, scenes_to_process):
        """Run the main analysis loop for all frames."""
        frame_map = self._create_frame_map()
        all_frame_nums_to_process = {
            fn for scene in scenes_to_process
            for fn in range(scene.start_frame, scene.end_frame)
            if fn in frame_map
        }

        image_files_to_process = [
            self.thumb_dir / f"{Path(frame_map[fn]).stem}.webp"
            for fn in sorted(list(all_frame_nums_to_process))
        ]

        self.progress_queue.put({
            "total": len(image_files_to_process),
            "stage": "Analysis"
        })
        
        num_workers = (1 if self.params.disable_parallel 
                      else min(os.cpu_count() or 4, 8))
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            list(executor.map(self._process_single_frame, 
                            image_files_to_process))

    def _process_single_frame(self, thumb_path):
        """Process a single frame for analysis."""
        
        if self.cancel_event.is_set():
            return

        frame_num_match = re.search(r'frame_(\d+)', thumb_path.name)
        if not frame_num_match:
            return
        log_context = {'file': thumb_path.name}

        try:
            self._initialize_niqe_metric()

            thumb_image_rgb = self.thumbnail_manager.get(thumb_path)
            if thumb_image_rgb is None:
                raise ValueError("Could not read thumbnail.")

            frame = Frame(thumb_image_rgb, -1)

            base_filename = thumb_path.name.replace('.webp', '.png')
            mask_meta = self.mask_metadata.get(base_filename, {})

            mask_thumb = None
            if mask_meta.get("mask_path"):
                mask_full_path = Path(mask_meta["mask_path"])
                if not mask_full_path.is_absolute():
                    mask_full_path = self.masks_dir / mask_full_path.name
                if mask_full_path.exists():
                    mask_full = cv2.imread(str(mask_full_path), 
                                         cv2.IMREAD_GRAYSCALE)
                    if mask_full is not None:
                        mask_thumb = cv2.resize(
                            mask_full,
                            (thumb_image_rgb.shape[1], thumb_image_rgb.shape[0]),
                            interpolation=cv2.INTER_NEAREST
                        )

            frame.calculate_quality_metrics(
                thumb_image_rgb=thumb_image_rgb, mask=mask_thumb,
                niqe_metric=self.niqe_metric
            )

            if (self.params.enable_face_filter and 
                self.reference_embedding is not None and 
                self.face_analyzer):
                self._analyze_face_similarity(frame, thumb_image_rgb)

            meta = {
                "filename": base_filename,
                "metrics": asdict(frame.metrics)
            }
            if frame.face_similarity_score is not None:
                meta["face_sim"] = frame.face_similarity_score
            if frame.max_face_confidence is not None:
                meta["face_conf"] = frame.max_face_confidence
            meta.update(mask_meta)

            if meta.get("shot_id") is not None:
                scene = self.scene_map.get(meta["shot_id"])
                if scene and scene.seed_metrics:
                    meta['seed_face_sim'] = scene.seed_metrics.get('best_face_sim')

            if self.params.enable_dedup:
                import imagehash
                pil_thumb = rgb_to_pil(thumb_image_rgb)
                meta['phash'] = str(imagehash.phash(pil_thumb))

            if frame.error:
                meta["error"] = frame.error
            if meta.get("mask_path"):
                meta["mask_path"] = Path(meta["mask_path"]).name

            meta = _to_json_safe(meta)
            with self.write_lock, self.metadata_path.open('a', 
                                                         encoding='utf-8') as f:
                json.dump(meta, f)
                f.write('\n')
            self.progress_queue.put({"progress": 1})
            
        except Exception as e:
            self.logger.critical("Error processing frame", exc_info=True,
                               extra={**log_context, 'error': e})
            meta = {
                "filename": thumb_path.name,
                "error": f"processing_failed: {e}"
            }
            with self.write_lock, self.metadata_path.open('a', 
                                                         encoding='utf-8') as f:
                json.dump(meta, f)
                f.write('\n')
            self.progress_queue.put({"progress": 1})

    def _analyze_face_similarity(self, frame, image_rgb):
        """Analyze face similarity for the frame."""
        try:
            # insightface expects BGR
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            with self.gpu_lock:
                faces = self.face_analyzer.get(image_bgr)
            if faces:
                best_face = max(faces, key=lambda x: x.det_score)
                distance = 1 - np.dot(best_face.normed_embedding, 
                                    self.reference_embedding)
                frame.face_similarity_score = 1.0 - float(distance)
                frame.max_face_confidence = float(best_face.det_score)
        except Exception as e:
            frame.error = f"Face similarity failed: {e}"