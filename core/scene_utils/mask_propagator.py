"""
MaskPropagator class for propagating segmentation masks across video frames.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Queue
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch

if TYPE_CHECKING:
    import threading

    from core.config import Config
    from core.logger import AppLogger
    from core.managers import ModelRegistry, SAM3Wrapper
    from core.models import AnalysisParameters
    from core.progress import AdvancedProgressTracker

from core.image_utils import postprocess_mask


class MaskPropagator:
    """
    Propagates segmentation masks from a seed frame to surrounding frames.

    Uses a subject tracker (e.g., SAM2.1 or SAM3) to propagate masks forward
    and backward from a seed frame where the subject was initially identified.
    """

    # TODO: Add temporal consistency smoothing between frames
    # TODO: Implement bidirectional propagation merging (not just forward+backward)
    # TODO: Add adaptive quality thresholds based on propagation distance from seed

    def __init__(
        self,
        params: "AnalysisParameters",
        dam_tracker: Optional["SAM3Wrapper"],
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
        from typing import Dict

        masks = {fn: np.zeros((h, w), dtype=np.uint8) for fn in frame_numbers}
        areas: Dict[int, float] = {fn: 0.0 for fn in frame_numbers}
        empties: Dict[int, bool] = {fn: True for fn in frame_numbers}
        errors: Dict[int, Optional[str]] = {fn: None for fn in frame_numbers}
        all_propagated: Dict[int, Optional[np.ndarray]] = {fn: None for fn in frame_numbers}
        target_frames = set(frame_numbers)

        self.logger.info(
            "Propagating masks with SAM3 (video mode)",
            component="propagator",
            user_context={
                "num_targets": len(frame_numbers),
                "num_prompts": len(prompts),
                "video": Path(video_path).name,
            },
        )

        if tracker:
            tracker.set_stage("Propagating masks", substage=f"{len(frame_numbers)} frames")

        try:
            # Initialize SAM3 with the video file
            self.dam_tracker.init_video(video_path)

            # Add all prompts and capture their indices
            start_frame_idx = frame_numbers[0]
            if prompts:
                start_frame_idx = prompts[0]["frame"]
                if self.cancel_event.is_set():
                    return masks, areas, empties, errors

                text_hint = (
                    self.params.text_prompt
                    if (hasattr(self.params, "text_prompt") and self.params.text_prompt)
                    else "person"
                )
                for p in prompts:
                    if self.cancel_event.is_set():
                        return masks, areas, empties, errors
                    fn = p["frame"]
                    mask = self.dam_tracker.add_bbox_prompt(
                        frame_idx=fn, obj_id=p.get("obj_id", 1), bbox_xywh=p["bbox"], img_size=(w, h), text=text_hint
                    )
                    # For seed frames, prioritize the added mask
                    if mask is not None:
                        all_propagated[fn] = mask
                        self.logger.debug(f"Added prompt mask at frame {fn}", component="propagator")

            if tracker:
                tracker.step(1, substage="Prompts added")

            if self.cancel_event.is_set():
                return masks, areas, empties, errors

            # Determine boundaries for progress bar and max_frames
            min_fn = min(frame_numbers)
            max_fn = max(frame_numbers)

            # --- Pass 1: Forward Propagation ---
            fwd_steps = max_fn - start_frame_idx
            if fwd_steps > 0:
                self.logger.debug(
                    f"Tracking forward from {start_frame_idx} to {max_fn} ({fwd_steps} steps)", component="propagator"
                )
                for frame_idx, obj_id, pred_mask, score in self.dam_tracker.propagate(
                    start_idx=start_frame_idx, reverse=False, max_frames=fwd_steps
                ):
                    if self.cancel_event.is_set():
                        break

                    if score < self.params.min_mask_confidence:
                        pred_mask = None  # Reject mask due to low confidence

                    if frame_idx in target_frames:
                        # Protect existing seeds or better masks
                        if pred_mask is not None and np.any(pred_mask):
                            all_propagated[frame_idx] = pred_mask

                        if tracker:
                            tracker.step(1, substage="Propagation (→)")

                        # Heartbeat every 50 frames
                        if frame_idx % 50 == 0:
                            self.logger.info(
                                f"Forward propagation heartbeat: frame {frame_idx}",
                                component="propagator",
                                user_context={"current_frame": frame_idx, "direction": "forward"},
                            )

            if self.cancel_event.is_set():
                return masks, areas, empties, errors

            # --- Pass 2: Backward Propagation ---
            bwd_steps = start_frame_idx - min_fn
            if bwd_steps > 0:
                self.logger.debug(
                    f"Tracking backward from {start_frame_idx} to {min_fn} ({bwd_steps} steps)", component="propagator"
                )
                for frame_idx, obj_id, pred_mask, score in self.dam_tracker.propagate(
                    start_idx=start_frame_idx, reverse=True, max_frames=bwd_steps
                ):
                    if self.cancel_event.is_set():
                        break

                    if score < self.params.min_mask_confidence:
                        pred_mask = None  # Reject mask due to low confidence

                    if frame_idx in target_frames:
                        if pred_mask is not None and np.any(pred_mask):
                            all_propagated[frame_idx] = pred_mask

                        if tracker:
                            tracker.step(1, substage="Propagation (←)")

                        # Heartbeat every 50 frames
                        if frame_idx % 50 == 0:
                            self.logger.info(
                                f"Backward propagation heartbeat: frame {frame_idx}",
                                component="propagator",
                                user_context={"current_frame": frame_idx, "direction": "backward"},
                            )

            # Process all gathered masks (seeds + propagated) in parallel
            img_area = h * w

            def _process_frame(fn):
                try:
                    pred_mask = all_propagated.get(fn)
                    if pred_mask is not None and np.any(pred_mask):
                        mask = postprocess_mask(
                            (pred_mask * 255).astype(np.uint8),
                            config=self.config,
                            fill_holes=True,
                            keep_largest_only=True,
                        )
                    else:
                        mask = np.zeros((h, w), dtype=np.uint8)

                    area_pct = (np.sum(mask > 0) / img_area) * 100 if img_area > 0 else 0.0
                    return fn, mask, float(area_pct), bool(area_pct < self.params.min_mask_area_pct)
                except Exception as e:
                    self.logger.error(f"Parallel mask post-processing failed: {e}")
                    return fn, np.zeros((h, w), dtype=np.uint8), 0.0, True

            with ThreadPoolExecutor(max_workers=min(len(frame_numbers), 8)) as executor:
                for fn, mask, area_pct, is_empty in executor.map(_process_frame, frame_numbers):
                    masks[fn] = mask
                    areas[fn] = area_pct
                    empties[fn] = is_empty
                    errors[fn] = "Empty mask" if is_empty else None

        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            err_msg = f"GPU error: {e}"
            self.logger.error(err_msg, component="propagator")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            for fn in frame_numbers:
                masks[fn] = np.zeros((h, w), dtype=np.uint8)
                areas[fn] = 0.0
                empties[fn] = True
                errors[fn] = err_msg
        except Exception as e:
            err_msg = f"Propagation error: {e}"
            self.logger.error(err_msg, component="propagator", exc_info=True)
            for fn in frame_numbers:
                masks[fn] = np.zeros((h, w), dtype=np.uint8)
                areas[fn] = 0.0
                empties[fn] = True
                errors[fn] = err_msg
        finally:
            try:
                if hasattr(self.dam_tracker, "close_session"):
                    self.dam_tracker.close_session()
            except Exception as e:
                self.logger.debug(f"Error during SAM3 session cleanup: {e}", component="propagator")

        return masks, areas, empties, errors

    def close(self):
        """Release tracker resources."""
        if self.dam_tracker:
            try:
                self.dam_tracker.close_session()
            except Exception:
                pass
