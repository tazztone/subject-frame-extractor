"""
MaskPropagator class for propagating segmentation masks across video frames.
"""

from __future__ import annotations

import threading
from pathlib import Path
from queue import Queue
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch

if TYPE_CHECKING:
    from core.config import Config
    from core.logger import AppLogger
    from core.managers import SAM3Wrapper
    from core.models import AnalysisParameters
    from core.progress import AdvancedProgressTracker

from core.utils import postprocess_mask


class MaskPropagator:
    """
    Propagates segmentation masks from a seed frame to surrounding frames.

    Uses SAM3 (Segment Anything Model 3) to propagate masks forward and backward
    from a seed frame where the subject was initially identified.
    """

    # TODO: Add temporal consistency smoothing between frames
    # TODO: Implement bidirectional propagation merging (not just forward+backward)
    # TODO: Add adaptive quality thresholds based on propagation distance from seed

    def __init__(
        self,
        params: "AnalysisParameters",
        dam_tracker: "SAM3Wrapper",
        cancel_event: threading.Event,
        progress_queue: Queue,
        config: "Config",
        logger: "AppLogger",
        device: str = "cpu",
        model_registry: Optional["ModelRegistry"] = None,
    ):
        """
        Initialize the MaskPropagator.

        Args:
            params: Analysis parameters
            dam_tracker: SAM3 wrapper for mask prediction
            cancel_event: Event to signal cancellation
            progress_queue: Queue for progress updates
            config: Application configuration
            logger: Application logger
            device: Device to run on ('cpu' or 'cuda')
            model_registry: Optional model registry for memory monitoring
        """
        self.params = params
        self.dam_tracker = dam_tracker
        self.cancel_event = cancel_event
        self.progress_queue = progress_queue
        self.config = config
        self.logger = logger
        self._device = device
        self.model_registry = model_registry

    def propagate_video(
        self,
        video_path: str,
        frame_numbers: list[int],
        prompts: list[dict],
        frame_size: tuple[int, int],
        frame_map: dict[int, str],
        tracker: Optional["AdvancedProgressTracker"] = None,
    ) -> tuple[dict, dict, dict, dict]:
        """
        Propagate masks using the video file directly (no temp JPEG I/O).

        Args:
            video_path: Path to the downscaled video file
            frame_numbers: List of original video frame numbers to get masks for
            prompts: List of prompts [{"frame": int, "bbox": list, "obj_id": int}]
            frame_size: (width, height) of the video frames
            frame_map: Mapping from frame number to filename
            tracker: Optional progress tracker

        Returns:
            Tuple of dicts keyed by frame_number: (masks, area_pcts, is_empty, errors)
        """
        if not self.dam_tracker:
            err_msg = "Tracker not initialized"
            return (
                {fn: None for fn in frame_numbers},
                {fn: 0.0 for fn in frame_numbers},
                {fn: True for fn in frame_numbers},
                {fn: err_msg for fn in frame_numbers},
            )

        w, h = frame_size
        masks = {}
        areas = {}
        empties = {}
        errors = {}
        all_propagated = {}
        
        # We only care about masks for these specific frames
        target_frames = set(frame_numbers)
        
        # Initialize all target frames with None/Empty to ensure alignment
        for fn in frame_numbers:
            all_propagated[fn] = None

        self.logger.info(
            "Propagating masks with SAM3 (video mode)",
            component="propagator",
            user_context={
                "num_targets": len(frame_numbers),
                "num_prompts": len(prompts),
                "video": Path(video_path).name
            },
        )

        if tracker:
            tracker.set_stage(f"Propagating masks for {len(frame_numbers)} frames")

        try:
            # Initialize SAM3 with the video file directly for temporal continuity
            self.dam_tracker.init_video(video_path)

            # Add all prompts
            start_frame_idx = 0
            if prompts:
                start_frame_idx = prompts[0]["frame"]
                for p in prompts:
                    fn = p["frame"]
                    mask = self.dam_tracker.add_bbox_prompt(
                        frame_idx=fn, obj_id=p.get("obj_id", 1), bbox_xywh=p["bbox"], img_size=(w, h)
                    )
                    if mask is not None and fn in target_frames:
                        all_propagated[fn] = mask
                        self.logger.debug(f"Added prompt mask at frame {fn}", component="propagator")

            if tracker:
                tracker.step(1, desc="Prompts added")

            # Propagate in both directions using single call.
            # SAM3 will track through EVERY frame in the video, ensuring stability.
            for frame_idx, obj_id, pred_mask in self.dam_tracker.propagate(start_idx=start_frame_idx, direction="both"):
                if self.cancel_event.is_set():
                    break
                if self.model_registry:
                    self.model_registry.check_memory_usage(self.config)
                
                # Only store results for frames we actually want to analyze
                if frame_idx in target_frames:
                    all_propagated[frame_idx] = pred_mask
                    
                    if tracker:
                        tracker.step(1, desc="Propagation (↔)")
                    
                    if pred_mask is not None and np.any(pred_mask):
                        self.logger.debug(f"Tracked subject at frame {frame_idx}", component="propagator")
                    else:
                        self.logger.debug(f"Lost subject at frame {frame_idx}", component="propagator")

            # Process all gathered masks (seeds + propagated)
            img_area = h * w
            for fn in frame_numbers:
                pred_mask = all_propagated.get(fn)

                if pred_mask is not None and np.any(pred_mask):
                    mask = postprocess_mask(
                        (pred_mask * 255).astype(np.uint8), config=self.config, fill_holes=True, keep_largest_only=True
                    )
                else:
                    mask = np.zeros((h, w), dtype=np.uint8)

                masks[fn] = mask
                area_pct = (np.sum(mask > 0) / img_area) * 100 if img_area > 0 else 0.0
                areas[fn] = area_pct
                empties[fn] = area_pct < self.params.min_mask_area_pct
                errors[fn] = "Empty mask" if empties[fn] else None

        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            err_msg = f"GPU error: {e}"
            self.logger.error(err_msg, component="propagator")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            for fn in frame_numbers:
                if fn not in masks:
                    masks[fn] = np.zeros((h, w), dtype=np.uint8)
                    areas[fn] = 0.0
                    empties[fn] = True
                    errors[fn] = err_msg
        except Exception as e:
            err_msg = f"Propagation error: {e}"
            self.logger.error(err_msg, component="propagator", exc_info=True)
            for fn in frame_numbers:
                if fn not in masks:
                    masks[fn] = np.zeros((h, w), dtype=np.uint8)
                    areas[fn] = 0.0
                    empties[fn] = True
                    errors[fn] = err_msg

        return masks, areas, empties, errors

    # TODO: Consider deprecating legacy method in favor of video-based propagation
    # TODO: Add memory-mapped frame loading for very large sequences
    def propagate(
        self,
        shot_frames_rgb: list[np.ndarray],
        seed_idx: int,
        bbox_xywh: list[int],
        tracker: Optional["AdvancedProgressTracker"] = None,
        additional_seeds: Optional[list[dict]] = None,
        frame_numbers: Optional[list[int]] = None,
    ) -> tuple[list, list, list, list]:
        """
        Legacy method: Propagate masks from a seed frame using in-memory frames.

        This method writes frames to temp JPEGs for SAM3 processing.
        Prefer propagate_video() when a downscaled video is available.

        Args:
            shot_frames_rgb: List of RGB frames as numpy arrays
            seed_idx: Index of the seed frame in the list (relative to shot_frames_rgb)
            bbox_xywh: Bounding box [x, y, width, height] on the seed frame
            tracker: Optional progress tracker
            additional_seeds: Optional list of additional seeds [{"frame": original_fn, "bbox": [x,y,w,h]}]
            frame_numbers: List of original frame numbers matching shot_frames_rgb

        Returns:
            Tuple of (masks, area_percentages, is_empty_flags, error_messages)
        """
        import os
        import tempfile

        from core.utils import rgb_to_pil

        if not self.dam_tracker or not shot_frames_rgb:
            err_msg = "Tracker not initialized" if not self.dam_tracker else "No frames"
            shape = shot_frames_rgb[0].shape[:2] if shot_frames_rgb else (100, 100)
            num_frames = len(shot_frames_rgb)
            return (
                [np.zeros(shape, np.uint8)] * num_frames,
                [0.0] * num_frames,
                [True] * num_frames,
                [err_msg] * num_frames,
            )

        self.logger.info(
            "Propagating masks with SAM3 (legacy temp JPEG mode)",
            component="propagator",
            user_context={
                "num_frames": len(shot_frames_rgb),
                "seed_index": seed_idx,
                "extra_seeds": len(additional_seeds or []),
            },
        )
        h, w = shot_frames_rgb[0].shape[:2]
        masks = [None] * len(shot_frames_rgb)

        if tracker:
            tracker.set_stage(f"Propagating masks for {len(shot_frames_rgb)} frames")

        # Use TemporaryDirectory for automatic cleanup
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # Save frames to temp directory for SAM3
                for i, img in enumerate(shot_frames_rgb):
                    pil_img = rgb_to_pil(img)
                    pil_img.save(os.path.join(temp_dir, f"{i:05d}.jpg"))

                # Initialize video session with new API
                self.dam_tracker.init_video(temp_dir)

                # Add primary seed bbox prompt (index i in temp_dir files)
                seed_mask = self.dam_tracker.add_bbox_prompt(
                    frame_idx=seed_idx, obj_id=1, bbox_xywh=bbox_xywh, img_size=(w, h)
                )

                # Add additional seeds if we can map their frame numbers to indices
                if additional_seeds and frame_numbers:
                    fn_to_idx = {fn: i for i, fn in enumerate(frame_numbers)}
                    for seed in additional_seeds:
                        idx = fn_to_idx.get(seed["frame"])
                        if idx is not None:
                            self.dam_tracker.add_bbox_prompt(
                                frame_idx=idx, obj_id=1, bbox_xywh=seed["bbox"], img_size=(w, h)
                            )

                # Process seed mask
                if seed_mask is not None:
                    mask = postprocess_mask(
                        (seed_mask * 255).astype(np.uint8), config=self.config, fill_holes=True, keep_largest_only=True
                    )
                else:
                    mask = np.zeros((h, w), dtype=np.uint8)
                masks[seed_idx] = mask

                if tracker:
                    tracker.step(1, desc="Propagation (seed)")

                # Propagate in both directions using single call
                for frame_idx, obj_id, pred_mask in self.dam_tracker.propagate(start_idx=seed_idx, direction="both"):
                    if frame_idx == seed_idx:
                        continue
                    if frame_idx < 0 or frame_idx >= len(shot_frames_rgb):
                        continue
                    if self.cancel_event.is_set():
                        break
                    
                    if self.model_registry:
                        self.model_registry.check_memory_usage(self.config)

                    if pred_mask is not None and np.any(pred_mask):
                        mask = postprocess_mask(
                            (pred_mask * 255).astype(np.uint8),
                            config=self.config,
                            fill_holes=True,
                            keep_largest_only=True,
                        )
                        masks[frame_idx] = mask
                    else:
                        masks[frame_idx] = np.zeros((h, w), dtype=np.uint8)

                    if tracker:
                        tracker.step(1, desc="Propagation (↔)")

            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                self.logger.error(f"GPU error in propagation: {e}", component="propagator")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e:
                self.logger.error(f"Propagation error: {e}", component="propagator", exc_info=True)

        # Compute final results
        final_masks = []
        final_areas = []
        final_empties = []
        final_errors = []

        for i, mask in enumerate(masks):
            if self.cancel_event.is_set() or mask is None:
                mask = np.zeros((h, w), dtype=np.uint8)
            img_area = h * w
            area_pct = (np.sum(mask > 0) / img_area) * 100 if img_area > 0 else 0.0
            is_empty = area_pct < self.params.min_mask_area_pct
            error = "Empty mask" if is_empty else None

            final_masks.append(mask)
            final_areas.append(area_pct)
            final_empties.append(is_empty)
            final_errors.append(error)

        return final_masks, final_areas, final_empties, final_errors
