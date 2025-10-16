"""Mask propagation logic for video sequences."""

import numpy as np
import torch

from app.frames import rgb_to_pil, postprocess_mask
from app.logging_enhanced import EnhancedLogger


class MaskPropagator:
    """Handles propagating a mask from a seed frame throughout a scene."""

    def __init__(self, params, dam_tracker, progress_tracker, cancel_event, progress_queue, logger=None):
        self.params = params
        self.dam_tracker = dam_tracker
        self.progress_tracker = progress_tracker
        self.cancel_event = cancel_event
        self.progress_queue = progress_queue
        self.logger = logger or EnhancedLogger()
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    def propagate(self, shot_frames_rgb, seed_idx, bbox_xywh):
        """Propagate mask from seed frame through all frames in the shot."""
        if not self.dam_tracker or not shot_frames_rgb:
            err_msg = ("Tracker not initialized" if not self.dam_tracker
                      else "No frames")
            shape = (shot_frames_rgb[0].shape[:2] if shot_frames_rgb
                    else (100, 100))
            num_frames = len(shot_frames_rgb)
            return (
                [np.zeros(shape, np.uint8)] * num_frames,
                [0.0] * num_frames,
                [True] * num_frames,
                [err_msg] * num_frames
            )

        self.logger.info("Propagating masks",
                         component="propagator",
                         user_context={'num_frames': len(shot_frames_rgb),
                                       'seed_index': seed_idx})
        
        masks = [None] * len(shot_frames_rgb)

        def _propagate_direction(start_idx, end_idx, step):
            """Propagate masks in one direction from seed."""
            for i in range(start_idx, end_idx, step):
                if self.cancel_event.is_set():
                    break
                outputs = self.dam_tracker.track(rgb_to_pil(shot_frames_rgb[i]))
                mask = outputs.get('pred_mask')
                if mask is not None:
                    mask = (mask * 255).astype(np.uint8)
                    mask = postprocess_mask(mask, fill_holes=True,
                                          keep_largest_only=True)
                processed_mask = (mask if mask is not None
                                else np.zeros_like(shot_frames_rgb[i],
                                                  dtype=np.uint8)[:, :, 0])
                masks[i] = processed_mask
                if self.progress_tracker:
                    self.progress_tracker.update_progress()

        try:
            with torch.cuda.amp.autocast(enabled=self._device == 'cuda'):
                # Initialize with seed frame
                outputs = self.dam_tracker.initialize(
                    rgb_to_pil(shot_frames_rgb[seed_idx]), None,
                    bbox=bbox_xywh)
                mask = outputs.get('pred_mask')
                if mask is not None:
                    mask = (mask * 255).astype(np.uint8)
                    mask = postprocess_mask(mask, fill_holes=True,
                                          keep_largest_only=True)
                seed_mask = (mask if mask is not None
                           else np.zeros_like(shot_frames_rgb[seed_idx],
                                            dtype=np.uint8)[:, :, 0])
                masks[seed_idx] = seed_mask
                if self.progress_tracker:
                    self.progress_tracker.update_progress()

                # Propagate forward
                _propagate_direction(seed_idx + 1, len(shot_frames_rgb), 1)

                # Re-initialize and propagate backward
                self.dam_tracker.initialize(rgb_to_pil(shot_frames_rgb[seed_idx]),
                                      None, bbox=bbox_xywh)
                _propagate_direction(seed_idx - 1, -1, -1)
                
            # Process results
            h, w = shot_frames_rgb[0].shape[:2]
            final_results = []
            for i, mask in enumerate(masks):
                if self.cancel_event.is_set() or mask is None:
                    mask = np.zeros((h, w), dtype=np.uint8)
                img_area = h * w
                area_pct = ((np.sum(mask > 0) / img_area) * 100 
                           if img_area > 0 else 0.0)
                is_empty = area_pct < self.params.min_mask_area_pct
                error = "Empty mask" if is_empty else None
                final_results.append((mask, float(area_pct), 
                                    bool(is_empty), error))
            return (tuple(zip(*final_results)) if final_results 
                   else ([], [], [], []))
                   
        except Exception as e:
            self.logger.critical("DAM4SAM propagation failed", component="propagator", exc_info=True)
            h, w = shot_frames_rgb[0].shape[:2]
            error_msg = f"Propagation failed: {e}"
            num_frames = len(shot_frames_rgb)
            return (
                [np.zeros((h, w), np.uint8)] * num_frames,
                [0.0] * num_frames,
                [True] * num_frames,
                [error_msg] * num_frames
            )