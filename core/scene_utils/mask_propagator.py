"""
MaskPropagator class for propagating segmentation masks across video frames.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Queue
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np

from core.utils.device import empty_cache, is_cuda_available

if TYPE_CHECKING:
    from core.config import Config
    from core.logger import AppLogger
    from core.managers import SAM3Wrapper
    from core.model_loader import ModelRegistry
    from core.models import AnalysisParameters
    from core.progress import AdvancedProgressTracker

from core.image_utils import postprocess_mask


class MaskPropagator:
    """
    Propagates segmentation masks from a seed frame to surrounding frames.

    Uses a subject tracker (e.g., SAM2.1 or SAM3) to propagate masks forward
    and backward through a scene's shots.
    """

    def __init__(
        self,
        params: "AnalysisParameters",
        dam_tracker: Optional["SAM3Wrapper"],
        cancel_event: threading.Event,
        progress_queue: Queue,
        config: "Config",
        logger: "AppLogger",
        device: Optional[str] = None,
        model_registry: Optional["ModelRegistry"] = None,
    ):
        """
        Initialize the MaskPropagator.
        """
        from core.utils.device import get_device

        self.params = params
        self.dam_tracker = dam_tracker
        self.cancel_event = cancel_event
        self.progress_queue = progress_queue
        self.config = config
        self.logger = logger
        self._device = device or get_device()
        self.model_registry = model_registry

    @property
    def device(self) -> str:
        return self._device

    @device.setter
    def device(self, value: str):
        self._device = value

    def propagate_video(
        self,
        video_path: str,
        frame_numbers: List[int],
        prompts: List[Dict],
        frame_size: Tuple[int, int],
        frame_map: Dict[int, str],
        tracker: Optional["AdvancedProgressTracker"] = None,
    ) -> Tuple[Dict[int, np.ndarray], Dict[int, float], Dict[int, bool], Dict[int, Optional[str]]]:
        """
        Propagate masks using the video file directly (no temp JPEG I/O).

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
        masks_dict: Dict[int, np.ndarray] = {fn: np.zeros((h, w), dtype=np.uint8) for fn in frame_numbers}
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
            tracker.set_stage(f"Propagating masks for {len(frame_numbers)} frames")

        try:
            # Initialize SAM3 with the video file
            self.dam_tracker.init_video(video_path)

            # Add all prompts
            start_frame_idx = frame_numbers[0]
            if prompts:
                start_frame_idx = prompts[0]["frame"]
                text_hint = (
                    self.params.text_prompt
                    if (hasattr(self.params, "text_prompt") and self.params.text_prompt)
                    else None
                )
                for p in prompts:
                    if self.cancel_event.is_set():
                        break
                    fn = p["frame"]
                    mask = self.dam_tracker.add_bbox_prompt(
                        frame_idx=fn, obj_id=p.get("obj_id", 1), bbox_xywh=p["bbox"], img_size=(w, h), text=text_hint
                    )
                    if mask is not None:
                        all_propagated[fn] = mask

            if self.cancel_event.is_set():
                return self._final_results(masks_dict, areas, empties, errors, frame_numbers, h, w)

            # Boundaries
            min_fn = min(frame_numbers)
            max_fn = max(frame_numbers)

            # --- Forward Propagation ---
            fwd_steps = max_fn - start_frame_idx
            if fwd_steps > 0:
                for frame_idx, obj_id, pred_mask in self.dam_tracker.propagate(
                    start_idx=start_frame_idx, reverse=False, max_frames=fwd_steps
                ):
                    if self.cancel_event.is_set():
                        break
                    if frame_idx in target_frames and pred_mask is not None:
                        all_propagated[frame_idx] = pred_mask
                    if tracker:
                        tracker.step(1, desc="Propagation (→)")

            if self.cancel_event.is_set():
                return self._final_results(masks_dict, areas, empties, errors, frame_numbers, h, w)

            # --- Backward Propagation ---
            bwd_steps = start_frame_idx - min_fn
            if bwd_steps > 0:
                for frame_idx, obj_id, pred_mask in self.dam_tracker.propagate(
                    start_idx=start_frame_idx, reverse=True, max_frames=bwd_steps
                ):
                    if self.cancel_event.is_set():
                        break
                    if frame_idx in target_frames and pred_mask is not None:
                        all_propagated[frame_idx] = pred_mask
                    if tracker:
                        tracker.step(1, desc="Propagation (←)")

            # --- Parallel Post-processing ---
            img_area = h * w

            def _process_frame(fn):
                try:
                    pred_mask = all_propagated.get(fn)
                    if pred_mask is not None and np.any(pred_mask):
                        # Ensure mask is uint8 for post-processing
                        mask_data = (
                            (pred_mask * 255).astype(np.uint8)
                            if pred_mask.dtype == bool
                            else pred_mask.astype(np.uint8)
                        )
                        res_mask = postprocess_mask(
                            mask_data,
                            config=self.config,
                            fill_holes=True,
                            keep_largest_only=True,
                        )
                    else:
                        res_mask = np.zeros((h, w), dtype=np.uint8)

                    area_pct = (np.sum(res_mask > 0) / img_area) * 100 if img_area > 0 else 0.0
                    return fn, res_mask, float(area_pct), bool(area_pct < self.params.min_mask_area_pct)
                except Exception as e:
                    self.logger.error(f"Parallel mask post-processing failed: {e}")
                    return fn, np.zeros((h, w), dtype=np.uint8), 0.0, True

            with ThreadPoolExecutor(max_workers=min(len(frame_numbers), 8)) as executor:
                for fn, res_mask, area_pct, is_empty in executor.map(_process_frame, frame_numbers):
                    masks_dict[fn] = res_mask
                    areas[fn] = area_pct
                    empties[fn] = is_empty
                    errors[fn] = "Empty mask" if is_empty else None

        except Exception as e:
            import torch as _torch

            _oom_type = getattr(_torch.cuda, "OutOfMemoryError", None)
            is_oom = "out of memory" in str(e).lower() or (
                _oom_type is not None and isinstance(_oom_type, type) and isinstance(e, _oom_type)
            )
            if is_oom:
                self.logger.error(f"GPU error in propagation: {e}", component="propagator")
                if self.device == "cuda" and is_cuda_available():
                    empty_cache()
                # Flag all remaining target frames as GPU error
                for fn in frame_numbers:
                    if errors[fn] is None:
                        errors[fn] = f"GPU error: {e}"
            else:
                self.logger.error(f"Propagation error: {e}", component="propagator", exc_info=True)
                for fn in frame_numbers:
                    if errors[fn] is None:
                        errors[fn] = str(e)

        return masks_dict, areas, empties, errors

    def _final_results(self, masks, areas, empties, errors, frame_numbers, h, w):
        """Helper to return current results on cancellation."""
        for fn in frame_numbers:
            if fn not in masks:
                masks[fn] = np.zeros((h, w), dtype=np.uint8)
                areas[fn] = 0.0
                empties[fn] = True
                errors[fn] = "Cancelled"
        return masks, areas, empties, errors

    def close(self):
        """Release tracker resources."""
        if self.dam_tracker:
            try:
                self.dam_tracker.close_session()
            except Exception:
                pass
