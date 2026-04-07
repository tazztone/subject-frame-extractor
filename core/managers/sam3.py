import sys
from typing import TYPE_CHECKING, Optional, Union

if TYPE_CHECKING:
    from core.config import Config

import numpy as np

# torch moved to methods for lazy loading

# Shim pkg_resources for vendored SAM3 that still uses the deprecated API
try:
    import pkg_resources  # noqa: F401
except ImportError:
    import types

    pkg_resources = types.ModuleType("pkg_resources")
    pkg_resources.require = lambda *a, **kw: None
    sys.modules["pkg_resources"] = pkg_resources

from core.utils import _setup_triton_mock
from core.utils.device import empty_cache, is_cuda_available, synchronize

_triton_mocked = _setup_triton_mock()


class SAM3Wrapper:
    """SAM3 Tracker using official Sam3VideoPredictor API."""

    def __init__(self, checkpoint_path=None, device="cuda", config: Optional["Config"] = None):
        from core.sam3_patches import apply_patches

        apply_patches()
        from sam3.model_builder import build_sam3_predictor  # type: ignore

        # Detect mock leakage — only in real GPU mode.
        # Unit tests use device="cpu" with intentional mocks; that's fine.
        # Integration tests use device="cuda" and must have real models.
        if device == "cuda":
            try:
                from unittest.mock import MagicMock as _MagicMock

                if isinstance(build_sam3_predictor, _MagicMock):
                    mod = getattr(build_sam3_predictor, "__module__", "unknown")
                    found_type = type(build_sam3_predictor)
                    raise RuntimeError(
                        f"SAM3 build_sam3_predictor is a MagicMock (type={found_type}, module={mod}, device='{device}') — "
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
            # Try 3.1 first, then 3.0
            _local_ckpt_31 = _project_root / "models" / "sam3.1_multiplex.pt"
            _local_ckpt_30 = _project_root / "models" / "sam3.pt"
            if _local_ckpt_31.exists():
                checkpoint_path = str(_local_ckpt_31)
            elif _local_ckpt_30.exists():
                checkpoint_path = str(_local_ckpt_30)

        self.device = device
        if is_cuda_available():
            import torch

            torch.set_float32_matmul_precision("high")

        # Detect version from checkpoint path
        sam3_version = "sam3.1"
        if checkpoint_path and "sam3.pt" in str(checkpoint_path) and "sam3.1" not in str(checkpoint_path):
            sam3_version = "sam3"

        # Configuration from app config or defaults
        compile_model = config.sam3_compile if config else False
        use_fa3 = config.sam3_use_flash_attention if config else False
        use_rope_real = config.sam3_use_rope_real if config else False

        # Force single-GPU mode for legacy SAM 3.0 (passed via kwargs)
        gpus = [0] if device == "cuda" else None

        self.predictor = build_sam3_predictor(
            checkpoint_path=checkpoint_path,
            version=sam3_version,
            compile=compile_model,
            use_fa3=use_fa3,
            use_rope_real=use_rope_real,
            gpus_to_use=gpus,
        )

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
        if hasattr(self.predictor, "model"):
            setattr(self.predictor.model, "hotstart_delay", 0)
            setattr(self.predictor.model, "masklet_confirmation_enable", False)
        elif hasattr(self.predictor, "tracker"):
            # Some predictor versions might call it 'tracker'
            setattr(self.predictor.tracker, "hotstart_delay", 0)
            setattr(self.predictor.tracker, "masklet_confirmation_enable", False)
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

    def add_mask_prompt(self, frame_idx: int, obj_id: int, mask: np.ndarray):
        """
        Add a mask prompt to the current session (SAM3 support is experimental).
        """
        if not self.session_id:
            raise RuntimeError("init_video must be called before adding prompts")

        # For now, we don't have a direct mask-to-PVS prompt path in the current shims,
        # so we log a warning. SAM3 usually prefers points/bboxes.
        import logging

        from core.logger import log_with_component

        logger = logging.getLogger("app_logger")
        log_with_component(logger, "warning", "SAM3 mask-based seeding is not yet implemented. Falling back.")
        return None

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

        if is_cuda_available():
            synchronize()
            empty_cache()

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

        if is_cuda_available():
            empty_cache()
            synchronize()
