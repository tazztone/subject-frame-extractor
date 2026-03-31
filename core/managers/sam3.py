from typing import Optional, Union

import numpy as np
import torch

from core.utils import _setup_triton_mock

_triton_mocked = _setup_triton_mock()


class SAM3Wrapper:
    """SAM3 Tracker using official Sam3VideoPredictor API."""

    def __init__(self, checkpoint_path=None, device="cuda"):
        from sam3.model_builder import build_sam3_video_predictor  # type: ignore

        # Detect mock leakage — only in real GPU mode.
        # Unit tests use device="cpu" with intentional mocks; that's fine.
        # Integration tests use device="cuda" and must have real models.
        if device == "cuda":
            try:
                from unittest.mock import MagicMock as _MagicMock

                if isinstance(build_sam3_video_predictor, _MagicMock):
                    raise RuntimeError(
                        f"SAM3 build_sam3_video_predictor is a MagicMock (device='{device}') — "
                        "conftest.py has injected a mock into sys.modules['sam3.model_builder']. "
                        "Run integration tests with: export PYTEST_INTEGRATION_MODE=true"
                    )
            except ImportError:
                pass

        # Auto-resolve checkpoint path to local models/sam3.pt if not provided.
        # This prevents the SAM3 library from attempting a HuggingFace download
        # (which requires gated repo authentication we don't want in tests/prod).
        if checkpoint_path is None:
            from pathlib import Path

            _project_root = Path(__file__).resolve().parents[2]
            _local_ckpt = _project_root / "models" / "sam3.pt"
            if _local_ckpt.exists():
                checkpoint_path = str(_local_ckpt)

        self.device = device
        if torch.cuda.is_available():
            torch.set_float32_matmul_precision("high")

        # Force single-GPU mode to prevent multi-process resource exhaustion in tests/Gradio
        gpus = [0] if device == "cuda" else None
        self.predictor = build_sam3_video_predictor(checkpoint_path=checkpoint_path, gpus_to_use=gpus)

        if self.predictor is None or getattr(self.predictor, "model", None) is None:
            raise RuntimeError(
                f"SAM3 model failed to load from '{checkpoint_path}'. "
                f"Predictor Type: {type(self.predictor)}. "
                f"Predictor Has Model: {hasattr(self.predictor, 'model')}. "
                "Ensure the checkpoint file exists and is not corrupted."
            )

        # Restore overrides for Subject Extraction workflow:
        # 1. Disable hotstart (delay=0) to ensure immediate masks for all frames
        # 2. Disable confirmation (False) to prevent suppressing masks for unconfirmed tracks
        self.predictor.model.hotstart_delay = 0  # type: ignore
        self.predictor.model.masklet_confirmation_enable = False  # type: ignore
        self.session_id = None

    def init_video(self, video_resource: Union[str, list]):
        if self.session_id:
            self.close_session()

        # Propagate settings through the request to ensure workers receive them
        resp = self.predictor.handle_request(
            dict(
                type="start_session", resource_path=video_resource, hotstart_delay=0, masklet_confirmation_enable=False
            )
        )
        self.session_id = resp["session_id"]
        return self.session_id

    def add_bbox_prompt(
        self, frame_idx: int, obj_id: int, bbox_xywh: list, img_size: tuple, text: Optional[str] = None
    ):
        if not self.session_id:
            raise RuntimeError("init_video must be called before adding prompts")
        """Route through PVS tracker by encoding bbox as two corner points with labels [2, 3]."""
        w, h = img_size
        x, y, bw, bh = bbox_xywh

        # Normalize and convert to [top-left, bottom-right] — the format add_tracker_new_points expects
        points = [
            [max(0.0, min(1.0, x / w)), max(0.0, min(1.0, y / h))],
            [max(0.0, min(1.0, (x + bw) / w)), max(0.0, min(1.0, (y + bh) / h))],
        ]
        point_labels = [2, 3]

        req = dict(
            type="add_prompt",
            session_id=self.session_id,
            frame_index=frame_idx,
            obj_id=obj_id,
            points=points,
            point_labels=point_labels,
        )
        if text:
            req["text"] = text

        resp = self.predictor.handle_request(request=req)
        outputs = resp.get("outputs", {})

        # Low-res masks are in out_binary_masks
        masks = outputs.get("out_binary_masks")
        if masks is not None and len(masks) > 0:
            if hasattr(masks, "cpu"):
                masks = masks.cpu().numpy()
            m = masks[0]
            if m.ndim == 3:
                m = m[0]
            return m > 0
        return np.zeros((h, w), dtype=bool)

    def add_point_prompt(self, frame_idx: int, obj_id: int, points: list, labels: list, img_size: tuple):
        if not self.session_id:
            raise RuntimeError("init_video must be called before adding prompts")
        """Foreground/background points — routes to PVS via points matching"""
        w, h = img_size
        norm_points = [[max(0.0, min(1.0, px / w)), max(0.0, min(1.0, py / h))] for px, py in points]

        req = dict(
            type="add_prompt",
            session_id=self.session_id,
            frame_index=frame_idx,
            obj_id=obj_id,
            points=norm_points,
            point_labels=labels,
        )

        resp = self.predictor.handle_request(request=req)
        outputs = resp.get("outputs", {})
        masks = outputs.get("out_binary_masks")
        if masks is not None and len(masks) > 0:
            if hasattr(masks, "cpu"):
                masks = masks.cpu().numpy()
            m = masks[0]
            if m.ndim == 3:
                m = m[0]
            return m > 0
        return np.zeros((h, w), dtype=bool)

    def propagate(self, start_idx: int = 0, max_frames: Optional[int] = None, reverse: bool = False):
        if not self.session_id:
            raise RuntimeError("init_video must be called before propagation")
        direction = "backward" if reverse else "forward"
        for resp in self.predictor.handle_stream_request(
            dict(
                type="propagate_in_video",
                session_id=self.session_id,
                start_frame_index=start_idx,
                max_frame_num_to_track=max_frames or 9999,
                propagation_direction=direction,
            )
        ):
            frame_idx = resp.get("frame_index")
            out = resp.get("outputs", {})
            masks, ids = out.get("out_binary_masks"), out.get("out_obj_ids")
            if masks is None or ids is None:
                continue
            if hasattr(masks, "cpu"):
                masks = masks.cpu().numpy()
            for i, oid in enumerate(ids):
                m = masks[i]
                if m.ndim == 3:
                    m = m[0]
                yield frame_idx, int(oid), m > 0

    def detect_objects(self, frame_rgb: np.ndarray, prompt: str) -> list:
        """Detect objects in a frame using a text prompt."""
        if not prompt or not prompt.strip():
            return []
        req = dict(type="detect_objects", image=frame_rgb, text=prompt)
        resp = self.predictor.handle_request(req)
        return resp.get("outputs", [])

    def add_text_prompt(self, frame_idx: int, text: str):
        """Add a text prompt to the current session."""
        if not self.session_id:
            raise RuntimeError("init_video must be called before adding prompts")
        req = dict(type="add_prompt", session_id=self.session_id, frame_index=frame_idx, text=text)
        return self.predictor.handle_request(request=req)

    def remove_object(self, obj_id: int):
        """Remove an object from the current session."""
        if not self.session_id:
            return
        req = dict(type="remove_object", session_id=self.session_id, obj_id=obj_id)
        return self.predictor.handle_request(request=req)

    def reset_session(self):
        """Reset the current tracking session."""
        self.close_session()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    def clear_prompts(self):
        """Clear all prompts in the current session."""
        if not self.session_id:
            return
        # Effective clear by resetting the session state
        self.reset_session()

    def close_session(self):
        if self.session_id:
            try:
                self.predictor.handle_request(dict(type="close_session", session_id=self.session_id))
            except Exception:
                pass
            self.session_id = None

    def shutdown(self):
        import gc

        self.close_session()
        if self.predictor is not None:
            if hasattr(self.predictor, "shutdown"):
                self.predictor.shutdown()
            # Deep cleanup of the predictor and its model
            if hasattr(self.predictor, "model"):
                self.predictor.model = None
            self.predictor = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
