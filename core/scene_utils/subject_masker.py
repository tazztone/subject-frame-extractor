"""
SubjectMasker class for coordinating subject detection and mask propagation.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from queue import Queue
from typing import TYPE_CHECKING, Callable, Optional

import cv2
import numpy as np
import torch

if TYPE_CHECKING:
    from insightface.app import FaceAnalysis
    from mediapipe.tasks.python.vision import FaceLandmarker

    from core.config import Config
    from core.logger import AppLogger
    from core.managers import ModelRegistry, ThumbnailManager
    from core.models import AnalysisParameters, Scene
    from core.progress import AdvancedProgressTracker

from core.scene_utils.mask_propagator import MaskPropagator
from core.scene_utils.seed_selector import SeedSelector
from core.utils import create_frame_map, draw_bbox


class SubjectMasker:
    """
    Coordinates subject detection and mask propagation for video frames.

    This class orchestrates:
    - SAM3 tracker initialization
    - Seed frame selection
    - Bounding box detection via SeedSelector
    - Mask propagation via MaskPropagator
    """

    # TODO: Add support for tracking multiple subjects simultaneously
    # TODO: Implement confidence-based mask rejection
    # TODO: Add mask quality assessment metrics

    def __init__(
        self,
        params: "AnalysisParameters",
        progress_queue: Queue,
        cancel_event: threading.Event,
        config: "Config",
        frame_map: Optional[dict] = None,
        face_analyzer: Optional["FaceAnalysis"] = None,
        reference_embedding: Optional[np.ndarray] = None,
        thumbnail_manager: Optional["ThumbnailManager"] = None,
        niqe_metric: Optional[Callable] = None,
        logger: Optional["AppLogger"] = None,
        face_landmarker: Optional["FaceLandmarker"] = None,
        device: str = "cpu",
        model_registry: "ModelRegistry" = None,
    ):
        """
        Initialize SubjectMasker.

        Args:
            params: Analysis parameters
            progress_queue: Queue for progress updates
            cancel_event: Event to signal cancellation
            config: Application configuration
            frame_map: Optional pre-loaded frame map
            face_analyzer: Optional InsightFace analyzer
            reference_embedding: Optional reference face embedding
            thumbnail_manager: Optional thumbnail cache manager
            niqe_metric: Optional NIQE quality metric
            logger: Application logger
            face_landmarker: Optional MediaPipe face landmarker
            device: Device for computation ('cpu' or 'cuda')
            model_registry: Model registry for loading SAM3
        """
        self.params = params
        self.config = config
        self.progress_queue = progress_queue
        self.cancel_event = cancel_event
        self.logger = logger
        self.frame_map = frame_map
        self.face_analyzer = face_analyzer
        self.reference_embedding = reference_embedding
        self.face_landmarker = face_landmarker
        self.dam_tracker = None
        self.mask_dir = None
        self.shots = []
        self._device = device
        self.thumbnail_manager = thumbnail_manager
        self.niqe_metric = niqe_metric
        self.model_registry = model_registry

        # Initialize child components BEFORE calling initialize_models()
        # because _initialize_tracker() may try to access self.seed_selector
        self.seed_selector = SeedSelector(
            params=params,
            config=self.config,
            face_analyzer=face_analyzer,
            reference_embedding=reference_embedding,
            tracker=self.dam_tracker,  # Will be None at this point, updated later
            logger=self.logger,
            device=self._device,
        )
        self.mask_propagator = MaskPropagator(
            params,
            self.dam_tracker,  # Will be None at this point, updated later
            cancel_event,
            progress_queue,
            config=self.config,
            logger=self.logger,
            device=self._device,
            model_registry=self.model_registry,
        )

        # Now initialize models (may update tracker in child components)
        self.initialize_models()

    def initialize_models(self) -> None:
        """Initialize required models based on parameters."""
        if self.params.enable_face_filter and self.face_analyzer is None:
            self.logger.warning("Face analyzer is not available but face filter is enabled.")

        if getattr(self.params, "need_masks_now", False) or self.params.enable_subject_mask:
            self._initialize_tracker()

    def _initialize_tracker(self) -> bool:
        """
        Initialize the SAM3 tracker.

        Returns:
            True if initialization successful, False otherwise
        """
        if self.dam_tracker:
            return True
        try:
            if not self.model_registry:
                self.logger.error("ModelRegistry not provided to SubjectMasker. Cannot load tracker.")
                return False

            retry_params = (self.config.retry_max_attempts, tuple(self.config.retry_backoff_seconds))
            self.logger.info(f"Initializing SAM3 tracker: {self.params.tracker_model_name}")
            self.dam_tracker = self.model_registry.get_tracker(
                model_name=self.params.tracker_model_name,
                models_path=str(self.config.models_dir),
                user_agent=self.config.user_agent,
                retry_params=retry_params,
                config=self.config,
            )
            if self.dam_tracker is None:
                self.logger.error("SAM3 tracker initialization returned None/failed")
                return False

            # Update child components with the new tracker
            if self.seed_selector:
                self.seed_selector.tracker = self.dam_tracker
            if self.mask_propagator:
                self.mask_propagator.dam_tracker = self.dam_tracker

            self.logger.success("SAM3 tracker initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"Exception during SAM3 tracker initialization: {e}", exc_info=True)
            return False

    def run_propagation(
        self, frames_dir: str, scenes_to_process: list["Scene"], tracker: Optional["AdvancedProgressTracker"] = None
    ) -> dict:
        """
        Run mask propagation for all scenes.

        Args:
            frames_dir: Directory containing extracted frames
            scenes_to_process: List of scenes to process
            tracker: Optional progress tracker

        Returns:
            Dictionary mapping frame filenames to mask metadata
        """
        self.mask_dir = Path(frames_dir) / "masks"
        self.mask_dir.mkdir(exist_ok=True)
        self.logger.info(f"Starting subject mask propagation for {len(scenes_to_process)} scenes...")

        if not self._initialize_tracker():
            self.logger.error("SAM3 tracker could not be initialized; mask propagation failed.")
            return {"error": "SAM3 tracker initialization failed", "completed": False}

        thumb_dir = Path(frames_dir) / "thumbs"
        mask_metadata = {}
        
        # 1. Consolidate all frames and seeds from all scenes
        all_target_frame_numbers = []
        all_seeds = []
        scene_lookup = {} # Map frame_number -> Scene for metadata association
        
        lowres_video_path = Path(frames_dir) / "video_lowres.mp4"
        thumb_dir = Path(frames_dir) / "thumbs"

        if lowres_video_path.exists():
            self.logger.info("Using bounded video-based propagation per scene.")
            
            # Detect actual frame size from thumbnails to ensure correct SAM3 normalization
            frame_size = (640, 480) # Fallback
            if self.frame_map:
                sample_fname = list(self.frame_map.values())[0]
                sample_thumb = self.thumbnail_manager.get(thumb_dir / sample_fname)
                if sample_thumb is not None:
                    frame_size = (sample_thumb.shape[1], sample_thumb.shape[0])
                    self.logger.info(f"Targeting propagation frame size: {frame_size[0]}x{frame_size[1]}", component="propagator")

            # Lock tracker model during the loop to prevent watchdog clearing it
            tracker_key = f"tracker_{self.params.tracker_model_name}"
            with self.model_registry.locked(tracker_key):
                for scene in scenes_to_process:
                    if self.cancel_event.is_set(): break
                    
                    shot_frames_data = self._load_shot_frames(frames_dir, thumb_dir, scene.start_frame, scene.end_frame)
                    if not shot_frames_data: continue
                    
                    target_fns = [f[0] for f in shot_frames_data]
                    bbox = scene.seed_result.get("bbox")
                    if not bbox:
                        self.logger.warning(f"No seed for scene {scene.shot_id}, skipping.")
                        continue

                    prop_seeds = [{"frame": scene.best_frame, "bbox": bbox, "obj_id": 1}]

                    masks_dict, areas_dict, empties_dict, errors_dict = self.mask_propagator.propagate_video(
                        str(lowres_video_path),
                        target_fns,
                        prop_seeds,
                        frame_size,
                        self.frame_map,
                        tracker=tracker,
                    )

                    # Distribute results for this scene
                    for fn in target_fns:
                        fname_webp = self.frame_map.get(fn)
                        if not fname_webp: continue
                        
                        fname_png = f"{Path(fname_webp).stem}.png"
                        mask_path = self.mask_dir / fname_png
                        
                        mask = masks_dict.get(fn)
                        area = areas_dict.get(fn, 0.0)
                        is_empty = empties_dict.get(fn, True)
                        
                        res = {
                            "shot_id": scene.shot_id,
                            "seed_type": scene.seed_result.get("details", {}).get("type"),
                            "seed_face_sim": scene.seed_result.get("details", {}).get("seed_face_sim"),
                            "mask_area_pct": area,
                            "mask_empty": is_empty,
                            "error": errors_dict.get(fn)
                        }
                        
                        if mask is not None and np.any(mask):
                            cv2.imwrite(str(mask_path), mask)
                            res["mask_path"] = str(mask_path)
                        else:
                            res["mask_path"] = None
                            if not errors_dict.get(fn):
                                self.logger.error(f"Failed to generate mask for {fname_png} (frame {fn}). Subject lost.", component="propagator")
                        
                        mask_metadata[fname_png] = res
        else:
            self.logger.warning("video_lowres.mp4 not found, falling back to legacy mode.")
            # Legacy fallback omitted or can be re-implemented if needed.
            pass

        self.logger.success(f"Subject masking complete for {len(mask_metadata)} frames.")
        
        # Save metadata
        try:
            mask_metadata_path = Path(frames_dir) / "mask_metadata.json"
            from core.utils import _to_json_safe
            with mask_metadata_path.open("w", encoding="utf-8") as f:
                json.dump(_to_json_safe(mask_metadata), f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save mask metadata: {e}")

        return mask_metadata

    def _load_shot_frames(
        self, frames_dir: str, thumb_dir: Path, start: int, end: int
    ) -> list[tuple[int, np.ndarray, tuple[int, int]]]:
        """
        Load frames for a shot from disk.

        Args:
            frames_dir: Base frames directory
            thumb_dir: Thumbnails directory
            start: Start frame number
            end: End frame number

        Returns:
            List of (frame_number, thumbnail_rgb, (height, width)) tuples
        """
        frames = []
        if not self.frame_map:
            ext = ".webp" if self.params.thumbnails_only else ".png"
            self.frame_map = create_frame_map(Path(frames_dir), self.logger, ext=ext)

        ext = ".webp" if self.params.thumbnails_only else ".png"
        for fn in sorted(fn for fn in self.frame_map if start <= fn < end):
            thumb_path = thumb_dir / f"{Path(self.frame_map[fn]).stem}{ext}"
            thumb_img = self.thumbnail_manager.get(thumb_path)
            if thumb_img is None:
                continue
            frames.append((fn, thumb_img, thumb_img.shape[:2]))
        return frames

    def _get_thumb_for_frame(self, thumb_dir: Path, frame_num: int) -> Optional[np.ndarray]:
        """Retrieve a thumbnail for a specific frame number."""
        if not self.frame_map:
            return None
        fname = self.frame_map.get(frame_num)
        if not fname:
            return None
        ext = ".webp" if self.params.thumbnails_only else ".png"
        thumb_path = thumb_dir / f"{Path(fname).stem}{ext}"
        return self.thumbnail_manager.get(thumb_path)

    def _select_best_frame_in_scene(self, scene: "Scene", frames_dir: str) -> None:
        """
        Select the best frame in a scene for seeding.

        Uses NIQE quality metric and face similarity if available.

        Args:
            scene: Scene to process
            frames_dir: Frames directory
        """
        if not self.params.pre_analysis_enabled:
            scene.best_frame = scene.start_frame
            scene.seed_metrics = {"reason": "pre-analysis disabled"}
            return

        thumb_dir = Path(frames_dir) / "thumbs"
        shot_frames = self._load_shot_frames(frames_dir, thumb_dir, scene.start_frame, scene.end_frame)
        if not shot_frames:
            scene.best_frame = scene.start_frame
            scene.seed_metrics = {"reason": "no frames loaded"}
            return

        candidates = shot_frames[:: max(1, self.params.pre_sample_nth)]
        scores = []
        niqe_score = 10.0
        face_sim = 0.0

        for frame_num, thumb_rgb, _ in candidates:
            niqe_score = 10.0
            if self.niqe_metric:
                with self.model_registry.locked("niqe"): # Assuming niqe key in future or similar
                    img_tensor = torch.from_numpy(thumb_rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
                    with torch.no_grad(), torch.amp.autocast("cuda", enabled=self._device == "cuda"):
                        niqe_score = float(self.niqe_metric(img_tensor.to(self.niqe_metric.device)))

            face_sim = 0.0
            if self.face_analyzer and self.reference_embedding is not None:
                face_model_key = f"face_analyzer_{self.params.face_model_name}_{self._device}_{tuple(self.config.model_face_analyzer_det_size)}"
                with self.model_registry.locked(face_model_key):
                    faces = self.face_analyzer.get(cv2.cvtColor(thumb_rgb, cv2.COLOR_RGB2BGR))
                    if faces:
                        best_face = max(faces, key=lambda x: x.det_score)
                        face_sim = np.dot(best_face.normed_embedding, self.reference_embedding)

            scores.append((10 - niqe_score) + (face_sim * 10))

        if not scores:
            scene.best_frame = shot_frames[0][0] if shot_frames else scene.start_frame
            scene.seed_metrics = {"reason": "pre-analysis failed, no scores", "score": 0}
            return

        # Select top candidates from the actual thumbnails
        candidates_with_scores = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        best_seeds = []
        min_dist = max(1, (scene.end_frame - scene.start_frame) // 8)

        for candidate, score in candidates_with_scores:
            frame_num = candidate[0]
            if all(abs(frame_num - existing_fn) >= min_dist for existing_fn in best_seeds):
                best_seeds.append(frame_num)
                if len(best_seeds) >= 3:
                    break

        if best_seeds:
            scene.best_frame = best_seeds[0]
            scene.candidate_seed_frames = best_seeds
        else:
            scene.best_frame = candidates[int(np.argmax(scores))][0]
            scene.candidate_seed_frames = [scene.best_frame]

        scene.seed_metrics = {
            "reason": "pre-analysis complete",
            "score": float(max(scores)) if scores else 0.0,
            "best_niqe": float(niqe_score),
            "best_face_sim": float(face_sim),
            "num_candidates": len(scene.candidate_seed_frames),
        }

    def get_seed_for_frame(
        self, frame_rgb: np.ndarray, seed_config: dict = None, scene: Optional["Scene"] = None
    ) -> tuple[Optional[list], dict]:
        """
        Get seed bounding box for a frame.

        Args:
            frame_rgb: RGB frame as numpy array
            seed_config: Optional seed configuration override
            scene: Optional scene context

        Returns:
            Tuple of (bbox_xywh, details_dict)
        """
        if isinstance(seed_config, dict) and seed_config.get("manual_bbox_xywh"):
            return seed_config["manual_bbox_xywh"], {"type": seed_config.get("seed_type", "manual")}

        self._initialize_tracker()

        if scene is not None:
            scene.person_detections = self.seed_selector._get_person_boxes(frame_rgb, scene=None)

        return self.seed_selector.select_seed(frame_rgb, current_params=seed_config, scene=scene)

    def get_mask_for_bbox(self, frame_rgb_small: np.ndarray, bbox_xywh: list) -> Optional[np.ndarray]:
        """
        Generate a mask for a bounding box.

        Args:
            frame_rgb_small: RGB frame
            bbox_xywh: Bounding box in [x, y, w, h] format

        Returns:
            Mask as numpy array or None
        """
        return self.seed_selector._get_mask_for_bbox(frame_rgb_small, bbox_xywh)

    def draw_bbox(
        self,
        img_rgb: np.ndarray,
        xywh: list,
        color: Optional[tuple] = None,
        thickness: Optional[int] = None,
        label: Optional[str] = None,
    ) -> np.ndarray:
        """Draw a bounding box on an image."""
        return draw_bbox(img_rgb, xywh, self.config, color, thickness, label)

    def _create_frame_map(self, output_dir: str) -> dict:
        """Create a frame map for the output directory."""
        ext = ".webp" if self.params.thumbnails_only else ".png"
        return create_frame_map(Path(output_dir), self.logger, ext=ext)
