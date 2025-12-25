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

from core.utils import postprocess_mask


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

    def propagate_video(
        self,
        video_path: str,
        frame_numbers: list[int],
        seed_frame_num: int,
        bbox_xywh: list[int],
        frame_size: tuple[int, int],
        tracker: Optional['AdvancedProgressTracker'] = None
    ) -> tuple[dict, dict, dict, dict]:
        """
        Propagate masks using the video file directly (no temp JPEG I/O).
        
        Args:
            video_path: Path to the downscaled video file
            frame_numbers: List of original video frame numbers to get masks for
            seed_frame_num: Original video frame number to seed from
            bbox_xywh: Bounding box [x, y, width, height] on the seed frame
            frame_size: (width, height) of the video frames
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
                {fn: err_msg for fn in frame_numbers}
            )
        
        w, h = frame_size
        masks = {}
        areas = {}
        empties = {}
        errors = {}
        
        self.logger.info(
            "Propagating masks with SAM3 (video mode)",
            component="propagator",
            user_context={'num_frames': len(frame_numbers), 'seed_frame': seed_frame_num}
        )

        if tracker:
            tracker.set_stage(f"Propagating masks for {len(frame_numbers)} frames")

        try:
            # Initialize SAM3 with the downscaled video directly
            self.dam_tracker.init_video(video_path)
            
            # Add bbox prompt on seed frame (using original video frame index)
            seed_mask = self.dam_tracker.add_bbox_prompt(
                frame_idx=seed_frame_num,
                obj_id=1,
                bbox_xywh=bbox_xywh,
                img_size=(w, h)
            )
            
            # Process seed mask
            if seed_mask is not None:
                mask = postprocess_mask(
                    (seed_mask * 255).astype(np.uint8),
                    config=self.config,
                    fill_holes=True,
                    keep_largest_only=True
                )
            else:
                mask = np.zeros((h, w), dtype=np.uint8)
            
            masks[seed_frame_num] = mask
            img_area = h * w
            area_pct = (np.sum(mask > 0) / img_area) * 100 if img_area > 0 else 0.0
            areas[seed_frame_num] = area_pct
            empties[seed_frame_num] = area_pct < self.params.min_mask_area_pct
            errors[seed_frame_num] = "Empty mask" if empties[seed_frame_num] else None
            
            if tracker:
                tracker.step(1, desc="Propagation (seed)")

            # Collect all propagated masks
            all_propagated = {}
            
            # Propagate forward
            for frame_idx, obj_id, pred_mask in self.dam_tracker.propagate(
                start_idx=seed_frame_num, reverse=False
            ):
                if self.cancel_event.is_set():
                    break
                all_propagated[frame_idx] = pred_mask
                if tracker:
                    tracker.step(1, desc="Propagation (→)")

            # Propagate backward
            for frame_idx, obj_id, pred_mask in self.dam_tracker.propagate(
                start_idx=seed_frame_num, reverse=True
            ):
                if self.cancel_event.is_set():
                    break
                if frame_idx not in all_propagated:  # Don't overwrite forward results
                    all_propagated[frame_idx] = pred_mask
                if tracker:
                    tracker.step(1, desc="Propagation (←)")
            
            # Filter to only the frames we care about
            for fn in frame_numbers:
                if fn == seed_frame_num:
                    continue  # Already processed
                    
                pred_mask = all_propagated.get(fn)
                if pred_mask is not None and np.any(pred_mask):
                    mask = postprocess_mask(
                        (pred_mask * 255).astype(np.uint8),
                        config=self.config,
                        fill_holes=True,
                        keep_largest_only=True
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

    def propagate(
        self,
        shot_frames_rgb: list[np.ndarray],
        seed_idx: int,
        bbox_xywh: list[int],
        tracker: Optional['AdvancedProgressTracker'] = None
    ) -> tuple[list, list, list, list]:
        """
        Legacy method: Propagate masks from a seed frame using in-memory frames.
        
        This method writes frames to temp JPEGs for SAM3 processing.
        Prefer propagate_video() when a downscaled video is available.
        
        Args:
            shot_frames_rgb: List of RGB frames as numpy arrays
            seed_idx: Index of the seed frame in the list
            bbox_xywh: Bounding box [x, y, width, height] on the seed frame
            tracker: Optional progress tracker
            
        Returns:
            Tuple of (masks, area_percentages, is_empty_flags, error_messages)
        """
        import tempfile
        import shutil
        import os
        from core.utils import rgb_to_pil
        
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
            "Propagating masks with SAM3 (legacy temp JPEG mode)",
            component="propagator",
            user_context={'num_frames': len(shot_frames_rgb), 'seed_index': seed_idx}
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
                
                # Add bbox prompt on seed frame
                seed_mask = self.dam_tracker.add_bbox_prompt(
                    frame_idx=seed_idx,
                    obj_id=1,
                    bbox_xywh=bbox_xywh,
                    img_size=(w, h)
                )
                
                # Process seed mask
                if seed_mask is not None:
                    mask = postprocess_mask(
                        (seed_mask * 255).astype(np.uint8),
                        config=self.config,
                        fill_holes=True,
                        keep_largest_only=True
                    )
                else:
                    mask = np.zeros((h, w), dtype=np.uint8)
                masks[seed_idx] = mask
                
                if tracker:
                    tracker.step(1, desc="Propagation (seed)")

                # Propagate forward using new generator API
                for frame_idx, obj_id, pred_mask in self.dam_tracker.propagate(
                    start_idx=seed_idx, reverse=False
                ):
                    if frame_idx == seed_idx:
                        continue
                    if frame_idx >= len(shot_frames_rgb):
                        break
                    if self.cancel_event.is_set():
                        break
                    
                    if pred_mask is not None and np.any(pred_mask):
                        mask = postprocess_mask(
                            (pred_mask * 255).astype(np.uint8),
                            config=self.config,
                            fill_holes=True,
                            keep_largest_only=True
                        )
                        masks[frame_idx] = mask
                    else:
                        masks[frame_idx] = np.zeros((h, w), dtype=np.uint8)

                    if tracker:
                        tracker.step(1, desc="Propagation (→)")

                # Propagate backward using new generator API
                for frame_idx, obj_id, pred_mask in self.dam_tracker.propagate(
                    start_idx=seed_idx, reverse=True
                ):
                    if frame_idx == seed_idx:
                        continue
                    if frame_idx < 0:
                        break
                    if self.cancel_event.is_set():
                        break

                    if pred_mask is not None and np.any(pred_mask):
                        mask = postprocess_mask(
                            (pred_mask * 255).astype(np.uint8),
                            config=self.config,
                            fill_holes=True,
                            keep_largest_only=True
                        )
                        masks[frame_idx] = mask
                    else:
                        masks[frame_idx] = np.zeros((h, w), dtype=np.uint8)

                    if tracker:
                        tracker.step(1, desc="Propagation (←)")

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
