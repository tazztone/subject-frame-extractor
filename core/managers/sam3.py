from typing import Optional, Union

import numpy as np
import torch


# Triton Mocking Logic
def _setup_triton_mock():
    import sys

    try:
        import triton  # noqa: F401

        return False
    except ImportError:
        pass
    import types
    from importlib.machinery import ModuleSpec
    from unittest.mock import MagicMock

    mock_triton = types.ModuleType("triton")
    mock_triton.__spec__ = ModuleSpec("triton", None)
    mock_triton.__path__ = []
    mock_triton.jit = lambda fn: fn
    mock_triton.language = types.ModuleType("triton.language")

    class MockTL:
        constexpr = lambda x: x
        program_id = MagicMock(return_value=0)
        load = MagicMock(return_value=0)
        store = MagicMock()

    for attr in dir(MockTL):
        if not attr.startswith("_"):
            setattr(mock_triton.language, attr, getattr(MockTL, attr))
    sys.modules["triton"] = mock_triton
    sys.modules["triton.language"] = mock_triton.language
    return True


_triton_mocked = _setup_triton_mock()


class SAM3Wrapper:
    """SAM3 Tracker using official Sam3VideoPredictor API."""

    def __init__(self, checkpoint_path=None, device="cuda"):
        from sam3.model_builder import build_sam3_video_predictor

        self.device = device
        if torch.cuda.is_available():
            torch.set_float32_matmul_precision("high")
        gpus = range(torch.cuda.device_count()) if device == "cuda" else None
        from unittest.mock import patch

        with patch("sam3.model_builder.download_ckpt_from_hf", return_value=None):
            self.predictor = build_sam3_video_predictor(checkpoint_path=checkpoint_path, gpus_to_use=gpus)
        if device == "cuda" and hasattr(self.predictor, "model"):
            self.predictor.model.to(dtype=torch.float32)
        self.session_id = None

    def init_video(self, video_resource: Union[str, list]):
        if self.session_id:
            self.close_session()
        resp = self.predictor.handle_request(dict(type="start_session", resource_path=video_resource))
        self.session_id = resp["session_id"]
        return self.session_id

    def add_bbox_prompt(
        self, frame_idx: int, obj_id: int, bbox_xywh: list, img_size: tuple, text: Optional[str] = None
    ):
        w, h = img_size
        x, y, bw, bh = bbox_xywh
        rel_box = [max(0.0, x / w), max(0.0, y / h), min(1.0, bw / w), min(1.0, bh / h)]
        req = dict(
            type="add_prompt",
            session_id=self.session_id,
            frame_index=frame_idx,
            obj_id=obj_id,
            bounding_boxes=np.array([rel_box], dtype=np.float32),
            bounding_box_labels=np.array([1], dtype=np.int32),
        )
        if text:
            req["text"] = text
        resp = self.predictor.handle_request(request=req)
        outputs = resp.get("outputs", {})
        masks, _ = outputs.get("out_binary_masks"), outputs.get("out_obj_ids")
        if masks is not None and len(masks) > 0:
            if hasattr(masks, "cpu"):
                masks = masks.cpu().numpy()
            m = masks[0]
            if m.ndim == 3:
                m = m[0]
            return m > 0
        return np.zeros((h, w), dtype=bool)

    def propagate(self, start_idx: int = 0, max_frames: int = None, direction: str = "forward"):
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
                yield frame_idx, oid, m > 0

    def close_session(self):
        if self.session_id:
            try:
                self.predictor.handle_request(dict(type="close_session", session_id=self.session_id))
            except Exception:
                pass
            self.session_id = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def shutdown(self):
        self.close_session()
        if hasattr(self.predictor, "shutdown"):
            self.predictor.shutdown()
        self.predictor = None
