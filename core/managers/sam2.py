# core/managers/sam2.py
from __future__ import annotations

import gc
from typing import Optional, Union

import numpy as np
import torch
from sam2.build_sam import build_sam2_video_predictor


class SAM2Wrapper:
    """SAM2.1 hiera-tiny via pip install sam2. Apache 2.0, ~38MB."""

    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        self.device = device
        if torch.cuda.is_available():
            torch.set_float32_matmul_precision("high")

        # SAM2.1 hiera-tiny config is usually built-in or provided by the package
        # We assume the config name matches the model type.
        self.predictor = build_sam2_video_predictor(
            config_file="configs/sam2.1/sam2.1_hiera_t.yaml",
            ckpt_path=checkpoint_path,
            device=device,
        )
        self._state = None

    def init_video(self, video_resource: Union[str, list]):
        """Accepts a frame-directory path (list → tempdir logic lives in caller)."""
        if self._state is not None:
            self.close_session()
        with torch.inference_mode():
            self._state = self.predictor.init_state(video_path=video_resource)
        return id(self._state)  # synthetic session id

    def add_bbox_prompt(
        self, frame_idx: int, obj_id: int, bbox_xywh: list, img_size: tuple, text: Optional[str] = None
    ) -> np.ndarray:
        x, y, w, h = bbox_xywh
        box = np.array([x, y, x + w, y + h], dtype=np.float32)
        with torch.inference_mode():
            _, _, masks = self.predictor.add_new_points_or_box(
                self._state,
                frame_idx=frame_idx,
                obj_id=obj_id,
                box=box,
            )
        m = masks[0].cpu().numpy()
        # Handle singleton dimensions
        if m.ndim == 3:
            m = m[0]
        return m > 0

    def propagate(self, start_idx: int = 0, max_frames: int = None, reverse: bool = False):
        with torch.inference_mode():
            for frame_idx, ids, masks in self.predictor.propagate_in_video(
                self._state,
                start_frame_idx=start_idx,
                max_frame_num_to_track=max_frames or 9999,
                reverse=reverse,
            ):
                masks_np = masks.cpu().numpy()
                for i, oid in enumerate(ids):
                    m = masks_np[i]
                    if m.ndim == 3:
                        m = m[0]
                    yield frame_idx, oid, m > 0

    def add_point_prompt(self, frame_idx, obj_id, points, labels, img_size):
        pts = np.array(points, dtype=np.float32)
        lbls = np.array(labels, dtype=np.int32)
        with torch.inference_mode():
            _, _, masks = self.predictor.add_new_points_or_box(
                self._state,
                frame_idx=frame_idx,
                obj_id=obj_id,
                points=pts,
                labels=lbls,
            )
        m = masks[0].cpu().numpy()
        if m.ndim == 3:
            m = m[0]
        return m > 0

    def close_session(self):
        if self._state is not None:
            self.predictor.reset_state(self._state)
            self._state = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def reset_session(self):
        self.close_session()

    def clear_prompts(self):
        if self._state:
            self.predictor.reset_state(self._state)

    def remove_object(self, *a, **kw):
        """Not implemented for SAM2.1 wrapper."""
        pass

    def detect_objects(self, *a, **kw):
        """SAM2.1 does not have built-in text-grounded detection like SAM3."""
        return []

    def add_text_prompt(self, *a, **kw):
        """SAM2.1 does not support text prompts."""
        raise NotImplementedError("SAM2.1 has no text prompt support")

    def shutdown(self):
        self.close_session()
        self.predictor = None
        gc.collect()
