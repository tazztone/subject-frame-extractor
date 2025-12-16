"""
MaskPropagator class for propagating segmentation masks across video frames.
"""
from __future__ import annotations
import threading
from typing import Optional, TYPE_CHECKING
from queue import Queue
import numpy as np
import torch

if TYPE_CHECKING:
    from core.config import Config
    from core.logger import AppLogger
    from core.models import AnalysisParameters
    from core.managers import SAM3Wrapper
    from core.progress import AdvancedProgressTracker

from core.utils import rgb_to_pil, postprocess_mask


class MaskPropagator:
    """
    Propagates segmentation masks from a seed frame to surrounding frames.
    
    Uses SAM3 (Segment Anything Model 3) to propagate masks forward and backward
    from a seed frame where the subject was initially identified.
    """
    
    def __init__(
        self,
        params: 'AnalysisParameters',
        dam_tracker: 'SAM3Wrapper',
        cancel_event: threading.Event,
        progress_queue: Queue,
        config: 'Config',
        logger: 'AppLogger',
        device: str = "cpu"
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
        """
        self.params = params
        self.dam_tracker = dam_tracker
        self.cancel_event = cancel_event
        self.progress_queue = progress_queue
        self.config = config
        self.logger = logger
        self._device = device

    def propagate(
        self,
        shot_frames_rgb: list[np.ndarray],
        seed_idx: int,
        bbox_xywh: list[int],
        tracker: Optional['AdvancedProgressTracker'] = None
    ) -> tuple[list, list, list, list]:
        """
        Propagate masks from a seed frame to all frames in a shot.
        
        Args:
            shot_frames_rgb: List of RGB frames as numpy arrays
            seed_idx: Index of the seed frame in the list
            bbox_xywh: Bounding box [x, y, width, height] on the seed frame
            tracker: Optional progress tracker
            
        Returns:
            Tuple of (masks, area_percentages, is_empty_flags, error_messages)
        """
        if not self.dam_tracker or not shot_frames_rgb:
            err_msg = "Tracker not initialized" if not self.dam_tracker else "No frames"
            shape = shot_frames_rgb[0].shape[:2] if shot_frames_rgb else (100, 100)
            num_frames = len(shot_frames_rgb)
            return (
                [np.zeros(shape, np.uint8)] * num_frames,
                [0.0] * num_frames,
                [True] * num_frames,
                [err_msg] * num_frames
            )
        
        self.logger.info(
            "Propagating masks with SAM3",
            component="propagator",
            user_context={'num_frames': len(shot_frames_rgb), 'seed_index': seed_idx}
        )
        masks = [None] * len(shot_frames_rgb)

        if tracker:
            tracker.set_stage(f"Propagating masks for {len(shot_frames_rgb)} frames")

        try:
            pil_images = [rgb_to_pil(img) for img in shot_frames_rgb]

            # Initialize with seed frame
            outputs = self.dam_tracker.initialize(
                pil_images, bbox=bbox_xywh, prompt_frame_idx=seed_idx
            )
            mask = outputs.get('pred_mask')
            if mask is not None:
                mask = postprocess_mask(
                    (mask * 255).astype(np.uint8),
                    config=self.config,
                    fill_holes=True,
                    keep_largest_only=True
                )
            masks[seed_idx] = (
                mask if mask is not None 
                else np.zeros_like(shot_frames_rgb[seed_idx], dtype=np.uint8)[:, :, 0]
            )
            if tracker:
                tracker.step(1, desc="Propagation (seed)")

            # Propagate forward
            for out in self.dam_tracker.propagate_from(seed_idx, direction="forward"):
                frame_idx = out['frame_index']
                if frame_idx == seed_idx:
                    continue
                if frame_idx >= len(shot_frames_rgb):
                    break

                if (out['outputs'] and 'obj_id_to_mask' in out['outputs'] 
                        and len(out['outputs']['obj_id_to_mask']) > 0):
                    pred_mask = list(out['outputs']['obj_id_to_mask'].values())[0]
                    if isinstance(pred_mask, torch.Tensor):
                        pred_mask = pred_mask.cpu().numpy().astype(bool)
                        if pred_mask.ndim == 3:
                            pred_mask = pred_mask[0]

                    mask = (pred_mask * 255).astype(np.uint8)
                    mask = postprocess_mask(
                        mask, config=self.config, fill_holes=True, keep_largest_only=True
                    )
                    masks[frame_idx] = mask
                else:
                    masks[frame_idx] = np.zeros_like(
                        shot_frames_rgb[frame_idx], dtype=np.uint8
                    )[:, :, 0]

                if tracker:
                    tracker.step(1, desc="Propagation (→)")

            # Propagate backward
            for out in self.dam_tracker.propagate_from(seed_idx, direction="backward"):
                frame_idx = out['frame_index']
                if frame_idx == seed_idx:
                    continue
                if frame_idx < 0:
                    break

                if (out['outputs'] and 'obj_id_to_mask' in out['outputs'] 
                        and len(out['outputs']['obj_id_to_mask']) > 0):
                    pred_mask = list(out['outputs']['obj_id_to_mask'].values())[0]
                    if isinstance(pred_mask, torch.Tensor):
                        pred_mask = pred_mask.cpu().numpy().astype(bool)
                        if pred_mask.ndim == 3:
                            pred_mask = pred_mask[0]

                    mask = (pred_mask * 255).astype(np.uint8)
                    mask = postprocess_mask(
                        mask, config=self.config, fill_holes=True, keep_largest_only=True
                    )
                    masks[frame_idx] = mask
                else:
                    masks[frame_idx] = np.zeros_like(
                        shot_frames_rgb[frame_idx], dtype=np.uint8
                    )[:, :, 0]

                if tracker:
                    tracker.step(1, desc="Propagation (←)")

            # Compute final results
            h, w = shot_frames_rgb[0].shape[:2]
            final_results = []
            for i, mask in enumerate(masks):
                if self.cancel_event.is_set() or mask is None:
                    mask = np.zeros((h, w), dtype=np.uint8)
                img_area = h * w
                area_pct = (np.sum(mask > 0) / img_area) * 100 if img_area > 0 else 0.0
                is_empty = area_pct < self.params.min_mask_area_pct
                error = "Empty mask" if is_empty else None
                final_results.append((mask, float(area_pct), bool(is_empty), error))
            
            if not final_results:
                return ([], [], [], [])
            
            masks, areas, empties, errors = map(list, zip(*final_results))
            return masks, areas, empties, errors
            
        except Exception as e:
            self.logger.critical("SAM3 propagation failed", component="propagator", exc_info=True)
            h, w = shot_frames_rgb[0].shape[:2]
            error_msg = f"Propagation failed: {e}"
            num_frames = len(shot_frames_rgb)
            return (
                [np.zeros((h, w), np.uint8)] * num_frames,
                [0.0] * num_frames,
                [True] * num_frames,
                [error_msg] * num_frames
            )
