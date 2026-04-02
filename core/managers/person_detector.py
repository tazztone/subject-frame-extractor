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


class PersonDetector:
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
        self.logger.debug(f"Detector mode: {'Segmentation' if self.is_seg else 'Detection only'}")

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
        results = []
        for det in preds:
            cx, cy, bw, bh = det[:4]
            scores = det[4:]
            class_id = int(np.argmax(scores))
            score = float(scores[class_id])

            if class_id != target_class_id or score < conf:
                continue

            # Unscale letterbox
            x1 = max(0, int((cx - bw / 2 - pad_left) / scale))
            y1 = max(0, int((cy - bh / 2 - pad_top) / scale))
            x2 = min(w, int((cx + bw / 2 - pad_left) / scale))
            y2 = min(h, int((cy + bh / 2 - pad_top) / scale))

            results.append(SubjectDetection(bbox=[x1, y1, x2, y2], conf=score, class_id=class_id, type="yolo26n"))

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
        results = []

        for det in preds:
            cx, cy, bw, bh = det[:4]
            # YOLO-seg typically has 80 classes, then 32 mask coefficients
            # 116 total: 4 coords + 80 class scores + 32 mask coeffs.
            class_scores = det[4:84]
            class_id = int(np.argmax(class_scores))
            score = float(class_scores[class_id])

            if class_id != target_class_id or score < conf_threshold:
                continue

            coeffs = det[84:]  # 32 coefficients

            # 1. Unscale bbox
            x1 = max(0, int((cx - bw / 2 - pad_left) / scale))
            y1 = max(0, int((cy - bh / 2 - pad_top) / scale))
            x2 = min(w, int((cx + bw / 2 - pad_left) / scale))
            y2 = min(h, int((cy + bh / 2 - pad_top) / scale))

            # 2. Decode mask: coeffs (32) @ proto (32, 160, 160) → (160, 160)
            c, mh, mw = proto.shape
            soft_mask = (coeffs @ proto.reshape(c, -1)).reshape(mh, mw)
            soft_mask = 1 / (1 + np.exp(-soft_mask))  # Sigmoid

            # 3. Clip mask to detection box in 160x160 space
            mx1, my1 = int((cx - bw / 2) * (mw / self.input_size)), int((cy - bh / 2) * (mh / self.input_size))
            mx2, my2 = int((cx + bw / 2) * (mw / self.input_size)), int((cy + bh / 2) * (mh / self.input_size))

            # Mask post-process (simple cropping to avoid artifacts outside bbox)
            crop_mask = np.zeros_like(soft_mask)
            my1, my2 = max(0, my1), min(mh, my2)
            mx1, mx2 = max(0, mx1), min(mw, mx2)
            crop_mask[my1:my2, mx1:mx2] = soft_mask[my1:my2, mx1:mx2]

            # 4. Resize mask to original frame size
            full_mask = cv2.resize(crop_mask, (w, h))
            bool_mask = full_mask > 0.5

            results.append(
                SubjectDetection(
                    bbox=[x1, y1, x2, y2], conf=score, class_id=class_id, type="yolo12l_seg", mask=bool_mask
                )
            )

        return results
