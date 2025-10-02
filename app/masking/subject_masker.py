"""Subject masking orchestration."""

import cv2
import numpy as np
import torch
from pathlib import Path
from dataclasses import asdict

from app.masking.seed_selector import SeedSelector
from app.masking.propagate import MaskPropagator


class SubjectMasker:
    """Orchestrates subject seeding and mask propagation for video analysis."""
    
    def __init__(self, params, progress_queue, cancel_event, frame_map=None,
                 face_analyzer=None, reference_embedding=None, 
                 person_detector=None, thumbnail_manager=None, 
                 niqe_metric=None):
        from app.core.thumb_cache import ThumbnailManager
        from app.ml.grounding import load_grounding_dino_model
        from app.ml.sam_tracker import initialize_dam4sam_tracker
        
        self.params = params
        self.progress_queue = progress_queue
        self.cancel_event = cancel_event
        self.frame_map = frame_map
        self.face_analyzer = face_analyzer
        self.reference_embedding = reference_embedding
        self.person_detector = person_detector
        self.tracker = None
        self.mask_dir = None
        self.shots = []
        self._gdino = None
        self._sam2_img = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self.thumbnail_manager = (thumbnail_manager if thumbnail_manager 
                                is not None else ThumbnailManager())
        self.niqe_metric = niqe_metric

        # Initialize sub-components
        self._initialize_models()
        self.seed_selector = SeedSelector(params, face_analyzer, 
                                        reference_embedding, person_detector, 
                                        self.tracker, self._gdino)
        self.mask_propagator = MaskPropagator(params, self.tracker, 
                                            cancel_event, progress_queue)

    def _initialize_models(self):
        """Initialize ML models for masking."""
        self._init_grounder()
        self._initialize_tracker()

    def _init_grounder(self):
        """Initialize Grounding DINO model."""
        from app.ml.grounding import load_grounding_dino_model
        
        if self._gdino is not None:
            return True
        self._gdino = load_grounding_dino_model(self.params, self._device)
        return self._gdino is not None

    def _initialize_tracker(self):
        """Initialize DAM4SAM tracker."""
        from app.ml.sam_tracker import initialize_dam4sam_tracker
        
        if self.tracker:
            return True
        self.tracker = initialize_dam4sam_tracker(self.params)
        return self.tracker is not None

    def run_propagation(self, frames_dir: str, scenes_to_process) -> dict:
        """Run mask propagation for all scenes."""
        from app.core.logging import UnifiedLogger
        from app.core.utils import safe_resource_cleanup
        from app.io.frames import create_frame_map
        from app.domain.models import MaskingResult
        
        logger = UnifiedLogger()
        self.mask_dir = Path(frames_dir) / "masks"
        self.mask_dir.mkdir(exist_ok=True)
        logger.info("Starting subject mask propagation...")

        if not self.tracker:
            logger.error("Tracker not initialized; skipping masking.")
            return {}

        self.frame_map = self.frame_map or self._create_frame_map(frames_dir)

        mask_metadata = {}
        total_scenes = len(scenes_to_process)
        for i, scene in enumerate(scenes_to_process):
            with safe_resource_cleanup():
                if self.cancel_event.is_set():
                    break
                self.progress_queue.put({
                    "stage": f"Masking Scene {i+1}/{total_scenes}"
                })
                shot_context = {
                    'shot_id': scene.shot_id,
                    'start_frame': scene.start_frame,
                    'end_frame': scene.end_frame
                }
                logger.info("Masking shot", extra=shot_context)

                seed_frame_num = scene.best_seed_frame
                shot_frames_data = self._load_shot_frames(
                    frames_dir, scene.start_frame, scene.end_frame)
                if not shot_frames_data:
                    continue

                frame_numbers, small_images, dims = zip(*shot_frames_data)

                try:
                    seed_idx_in_shot = frame_numbers.index(seed_frame_num)
                except ValueError:
                    logger.warning(
                        f"Seed frame {seed_frame_num} not found in loaded "
                        f"shot frames for {scene.shot_id}, skipping."
                    )
                    continue

                bbox = scene.seed_result.get('bbox')
                seed_details = scene.seed_result.get('details', {})

                if bbox is None:
                    for fn in frame_numbers:
                        if (fname := self.frame_map.get(fn)):
                            mask_metadata[fname] = asdict(MaskingResult(
                                error="Subject not found", 
                                shot_id=scene.shot_id))
                    continue

                masks, areas, empties, errors = self.mask_propagator.propagate(
                    small_images, seed_idx_in_shot, bbox)

                for i, (original_fn, _, (h, w)) in enumerate(shot_frames_data):
                    frame_fname_webp = self.frame_map.get(original_fn)
                    if not frame_fname_webp:
                        continue

                    frame_fname_png = f"{Path(frame_fname_webp).stem}.png"
                    mask_path = self.mask_dir / frame_fname_png

                    result_args = {
                        "shot_id": scene.shot_id,
                        "seed_type": seed_details.get('type'),
                        "seed_face_sim": seed_details.get('seed_face_sim'),
                        "mask_area_pct": areas[i],
                        "mask_empty": empties[i],
                        "error": errors[i]
                    }
                    if masks[i] is not None and np.any(masks[i]):
                        mask_full_res = cv2.resize(
                            masks[i], (w, h), interpolation=cv2.INTER_NEAREST)
                        if mask_full_res.ndim == 3:
                            mask_full_res = mask_full_res[:, :, 0]
                        cv2.imwrite(str(mask_path), mask_full_res)
                        mask_metadata[frame_fname_png] = asdict(MaskingResult(
                            mask_path=str(mask_path), **result_args))
                    else:
                        mask_metadata[frame_fname_png] = asdict(MaskingResult(
                            mask_path=None, **result_args))
        
        logger.success("Subject masking complete.")
        return mask_metadata

    def _create_frame_map(self, frames_dir):
        """Create frame mapping."""
        from app.io.frames import create_frame_map
        return create_frame_map(Path(frames_dir))

    def _load_shot_frames(self, frames_dir, start, end):
        """Load frames for a shot from thumbnails."""
        frames = []
        if not self.frame_map:
            self.frame_map = self._create_frame_map(frames_dir)

        thumb_dir = Path(frames_dir) / "thumbs"
        for fn in sorted(fn for fn in self.frame_map if start <= fn < end):
            thumb_p = thumb_dir / f"{Path(self.frame_map[fn]).stem}.webp"
            thumb_img = self.thumbnail_manager.get(thumb_p)
            if thumb_img is None:
                continue

            h, w = thumb_img.shape[:2]
            frames.append((fn, thumb_img, (h, w)))
        return frames

    def _select_best_seed_frame_in_scene(self, scene, frames_dir: str):
        """Select the best frame in a scene for seeding."""
        if not self.params.pre_analysis_enabled:
            scene.best_seed_frame = scene.start_frame
            scene.seed_metrics = {'reason': 'pre-analysis disabled'}
            return

        shot_frames = self._load_shot_frames(frames_dir, scene.start_frame, 
                                           scene.end_frame)
        if not shot_frames:
            scene.best_seed_frame = scene.start_frame
            scene.seed_metrics = {'reason': 'no frames loaded'}
            return

        step = max(1, self.params.pre_sample_nth)
        candidates = shot_frames[::step]
        scores = []

        for frame_num, thumb_rgb, _ in candidates:
            niqe_score = 10.0
            if self.niqe_metric:
                img_tensor = (torch.from_numpy(thumb_rgb).float()
                             .permute(2, 0, 1).unsqueeze(0) / 255.0)
                with (torch.no_grad(), 
                      torch.cuda.amp.autocast(enabled=self._device == 'cuda')):
                    niqe_score = float(self.niqe_metric(
                        img_tensor.to(self.niqe_metric.device)))

            face_sim = 0.0
            if (self.face_analyzer and 
                self.reference_embedding is not None):
                thumb_bgr = cv2.cvtColor(thumb_rgb, cv2.COLOR_RGB2BGR)
                faces = self.face_analyzer.get(thumb_bgr)
                if faces:
                    best_face = max(faces, key=lambda x: x.det_score)
                    face_sim = 1.0 - (1 - np.dot(
                        best_face.normed_embedding, self.reference_embedding))

            combined_score = (10 - niqe_score) + (face_sim * 10)
            scores.append(combined_score)

        best_local_idx = int(np.argmax(scores)) if scores else 0
        best_frame_num, _, _ = candidates[best_local_idx]
        scene.best_seed_frame = best_frame_num
        scene.seed_metrics = {
            'reason': 'pre-analysis complete',
            'score': max(scores) if scores else 0,
            'best_niqe': niqe_score,
            'best_face_sim': face_sim
        }

    def get_seed_for_frame(self, frame_rgb: np.ndarray, seed_config: dict):
        """Public method to get a seed for a given frame with overrides."""
        return self.seed_selector._seed_identity(frame_rgb, 
                                               current_params=seed_config)

    def get_mask_for_bbox(self, frame_rgb_small, bbox_xywh):
        """Public method to get a SAM mask for a bounding box."""
        return self.seed_selector._sam2_mask_for_bbox(frame_rgb_small, 
                                                     bbox_xywh)

    def draw_bbox(self, img_rgb, xywh, color=(255, 0, 0), thickness=2):
        """Draw bounding box on image."""
        x, y, w, h = map(int, xywh or [0, 0, 0, 0])
        img_out = img_rgb.copy()
        cv2.rectangle(img_out, (x, y), (x + w, y + h), color, thickness)
        return img_out
