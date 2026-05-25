# core/managers/person_detector.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

import cv2
import numpy as np
import onnxruntime as ort

if TYPE_CHECKING:
    from core.logger import AppLogger


@dataclass
class SubjectDetection:
    """Standardized detection result for subject tracking."""

    bbox: List[int]  # [x1, y1, x2, y2]
    conf: float
    class_id: int
    type: str  # "yolo26n" | "yolo12l_seg"
    mask: Optional[np.ndarray] = None  # bool H×W, only from yolo12l_seg

    def to_dict(self) -> dict:
        """Compatibility method for legacy SeedSelector dict-based logic."""
        # Manually construct dict to avoid deep-copying the large numpy mask array
        # which asdict() would do.
        return {"bbox": self.bbox, "conf": self.conf, "class_id": self.class_id, "type": self.type}


# Alias for backward compatibility
PersonDetection = SubjectDetection


class SubjectDetector:
    """GPU-accelerated subject detection and segmentation using YOLO ONNX."""

    def __init__(self, model_path: str, logger: "AppLogger", device: str = "cuda"):
        self.logger = logger
        self.model_path = Path(model_path)
        self.device = device
        self.input_size = 640

        # Load ONNX session
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        try:
            self.session = ort.InferenceSession(str(self.model_path), sess_options=sess_options, providers=providers)
            active = self.session.get_providers()[0]
            self.logger.success(f"Subject detector loaded on: {active}", component="person_detector")
        except Exception as e:
            self.logger.error(f"Failed to load subject detector: {e}", exc_info=True)
            raise

        # Check if this is a segmentation model by looking at outputs
        outputs = self.session.get_outputs()
        self.is_seg = len(outputs) > 1
        # Determine number of classes and coefficients
        inputs = self.session.get_inputs()[0]
        output_shape = outputs[0].shape  # YOLO output for seg: (1, 4 + num_classes + num_masks, num_detections)
        # We look at the second dimension of the first output
        self.total_dims = output_shape[1]

        self.logger.debug(
            f"Detector mode: {'Segmentation' if self.is_seg else 'Detection only'} | "
            f"Input: {inputs.shape} | Output: {output_shape}",
            component="person_detector",
        )

    def detect(
        self, frame_bgr: np.ndarray, conf_threshold: float = 0.45, target_class_id: int = 0
    ) -> List[SubjectDetection]:
        """Run inference and return subject detections."""
        h, w = frame_bgr.shape[:2]

        # 1. Preprocess: Letterbox to 640x640
        scale = self.input_size / max(h, w)
        nh, nw = int(h * scale), int(w * scale)
        resized = cv2.resize(frame_bgr, (nw, nh))
        canvas = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
        pad_top = (self.input_size - nh) // 2
        pad_left = (self.input_size - nw) // 2
        canvas[pad_top : pad_top + nh, pad_left : pad_left + nw] = resized

        # BGR→RGB, HWC→CHW, normalize
        blob = canvas[:, :, ::-1].astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)[np.newaxis]  # (1, 3, 640, 640)

        # 2. Inference
        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {input_name: blob})
        self.logger.debug(f"YOLO raw predictions: {len(outputs[0][0].T)} candidates.")

        # 3. Postprocess
        if self.is_seg:
            return self._postprocess_seg(outputs, h, w, scale, pad_top, pad_left, conf_threshold, target_class_id)
        else:
            return self._postprocess_det(outputs, h, w, scale, pad_top, pad_left, conf_threshold, target_class_id)

    def _postprocess_det(
        self,
        outputs: list,
        h: int,
        w: int,
        scale: float,
        pad_top: int,
        pad_left: int,
        conf: float,
        target_class_id: int,
    ) -> List[SubjectDetection]:
        # YOLO26 ONNX output: (1, num_classes+4, num_detections)
        preds = outputs[0][0].T  # (N, C+4)

        # Vectorized class filtering
        class_scores = preds[:, 4:]
        class_ids = np.argmax(class_scores, axis=1)
        scores = class_scores[np.arange(len(class_ids)), class_ids]

        mask = (class_ids == target_class_id) & (scores >= conf)
        valid_preds = preds[mask]
        valid_scores = scores[mask]
        valid_class_ids = class_ids[mask]

        if len(valid_preds) == 0:
            return []

        # Vectorized bbox unscaling
        cxs, cys, bws, bhs = valid_preds[:, 0], valid_preds[:, 1], valid_preds[:, 2], valid_preds[:, 3]
        x1s = np.clip((cxs - bws / 2 - pad_left) / scale, 0, w).astype(int)
        y1s = np.clip((cys - bhs / 2 - pad_top) / scale, 0, h).astype(int)
        x2s = np.clip((cxs + bws / 2 - pad_left) / scale, 0, w).astype(int)
        y2s = np.clip((cys + bhs / 2 - pad_top) / scale, 0, h).astype(int)

        results = []
        for i in range(len(valid_preds)):
            results.append(
                SubjectDetection(
                    bbox=[int(x1s[i]), int(y1s[i]), int(x2s[i]), int(y2s[i])],
                    conf=float(valid_scores[i]),
                    class_id=int(valid_class_ids[i]),
                    type="yolo26n",
                )
            )

        return results

    def _postprocess_seg(
        self,
        outputs: list,
        h: int,
        w: int,
        scale: float,
        pad_top: int,
        pad_left: int,
        conf_threshold: float,
        target_class_id: int,
    ) -> List[SubjectDetection]:
        # YOLO-seg outputs: [0] = detections (1, 116, N), [1] = proto masks (1, 32, 160, 160)
        preds = outputs[0][0].T  # (N, 116)
        proto = outputs[1][0]  # (32, 160, 160)
        # Determine mask coefficient count from proto shape
        num_masks = proto.shape[0]  # Usually 32

        # Vectorized class filtering
        class_scores = preds[:, 4:-num_masks]
        class_ids = np.argmax(class_scores, axis=1)
        scores = class_scores[np.arange(len(class_ids)), class_ids]

        mask = (class_ids == target_class_id) & (scores >= conf_threshold)
        valid_preds = preds[mask]
        valid_scores = scores[mask]
        valid_class_ids = class_ids[mask]

        if len(valid_preds) == 0:
            return []

        # 1. Vectorized bbox unscaling
        cxs, cys, bws, bhs = valid_preds[:, 0], valid_preds[:, 1], valid_preds[:, 2], valid_preds[:, 3]
        x1s = np.clip((cxs - bws / 2 - pad_left) / scale, 0, w).astype(int)
        y1s = np.clip((cys - bhs / 2 - pad_top) / scale, 0, h).astype(int)
        x2s = np.clip((cxs + bws / 2 - pad_left) / scale, 0, w).astype(int)
        y2s = np.clip((cys + bhs / 2 - pad_top) / scale, 0, h).astype(int)

        # 2. Vectorized mask decoding
        c, mh, mw = proto.shape
        coeffs = valid_preds[:, -num_masks:]
        soft_masks = (coeffs @ proto.reshape(c, -1)).reshape(-1, mh, mw)
        soft_masks = 1 / (1 + np.exp(-soft_masks))  # Sigmoid

        # 3. Vectorized mask clipping bounds
        mx1s = np.clip((cxs - bws / 2) * (mw / self.input_size), 0, mw).astype(int)
        my1s = np.clip((cys - bhs / 2) * (mh / self.input_size), 0, mh).astype(int)
        mx2s = np.clip((cxs + bws / 2) * (mw / self.input_size), 0, mw).astype(int)
        my2s = np.clip((cys + bhs / 2) * (mh / self.input_size), 0, mh).astype(int)

        results = []
        for i in range(len(valid_preds)):
            mx1, my1, mx2, my2 = mx1s[i], my1s[i], mx2s[i], my2s[i]

            # Mask post-process (simple cropping to avoid artifacts outside bbox)
            crop_mask = np.zeros((mh, mw), dtype=soft_masks.dtype)
            crop_mask[my1:my2, mx1:mx2] = soft_masks[i, my1:my2, mx1:mx2]

            # 4. Resize mask to original frame size
            full_mask = cv2.resize(crop_mask, (w, h))
            bool_mask = full_mask > 0.5

            results.append(
                SubjectDetection(
                    bbox=[int(x1s[i]), int(y1s[i]), int(x2s[i]), int(y2s[i])],
                    conf=float(valid_scores[i]),
                    class_id=int(valid_class_ids[i]),
                    type="yolo12l_seg",
                    mask=bool_mask,
                )
            )

        return results

    def close(self):
        """Explicitly release the ONNX Runtime session."""
        if hasattr(self, "session"):
            del self.session
            self.logger.debug("SubjectDetector ONNX session released.")

    def __del__(self):
        self.close()
