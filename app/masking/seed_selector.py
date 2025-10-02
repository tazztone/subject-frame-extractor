"""Seed selection logic for subject masking."""

import math
import cv2
import numpy as np
import torch
from torchvision import transforms
from torchvision.ops import box_convert


class SeedSelector:
    """Handles the logic for selecting the initial seed (bounding box) for a scene."""
    
    def __init__(self, params, face_analyzer, reference_embedding, 
                 person_detector, tracker, gdino_model):
        self.params = params
        self.face_analyzer = face_analyzer
        self.reference_embedding = reference_embedding
        self.person_detector = person_detector
        self.tracker = tracker
        self._gdino = gdino_model
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    def _load_image_from_array(self, image_rgb: np.ndarray):
        """Load image from numpy array for grounding DINO."""
        transform = transforms.Compose([
            transforms.ToPILImage(), transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        image_tensor = transform(image_rgb)
        return image_rgb, image_tensor

    def _ground_first_frame_xywh(self, frame_rgb_small: np.ndarray, text: str,
                                box_th: float, text_th: float):
        """Ground first frame using text prompt to get bounding box."""
        from app.ml.grounding import predict_grounding_dino
        from app.core.logging import UnifiedLogger
        logger = UnifiedLogger()
        
        if not self._gdino:
            return None, {}
        image_source, image_tensor = self._load_image_from_array(frame_rgb_small)
        h, w = image_source.shape[:2]

        with (torch.no_grad(), 
              torch.cuda.amp.autocast(enabled=self._device == 'cuda')):
            boxes, confidences, labels = predict_grounding_dino(
                model=self._gdino, image_tensor=image_tensor, caption=text,
                box_threshold=float(box_th), text_threshold=float(text_th),
                device=self._device
            )

        if boxes is None or len(boxes) == 0:
            return None, {"type": "text_prompt", "error": "no_boxes"}

        scale = torch.tensor([w, h, w, h], device=boxes.device, 
                           dtype=boxes.dtype)
        boxes_abs = (boxes * scale).cpu()
        xyxy = box_convert(boxes=boxes_abs, in_fmt="cxcywh", 
                          out_fmt="xyxy").numpy()
        conf = confidences.cpu().numpy().tolist()

        idx = int(np.argmax(conf))
        x1, y1, x2, y2 = map(float, xyxy[idx])
        xywh = [int(max(0, x1)), int(max(0, y1)), 
                int(max(1, x2 - x1)), int(max(1, y2 - y1))]
        details = {"type": "text_prompt", 
                  "label": labels[idx] if labels else "", 
                  "conf": float(conf[idx])}
        return xywh, details

    def _sam2_mask_for_bbox(self, frame_rgb_small, bbox_xywh):
        """Generate SAM2 mask for bounding box."""
        from app.io.frames import rgb_to_pil, postprocess_mask
        from app.core.logging import UnifiedLogger
        logger = UnifiedLogger()
        
        if not self.tracker or bbox_xywh is None:
            return None
        try:
            outputs = self.tracker.initialize(rgb_to_pil(frame_rgb_small), 
                                            None, bbox=bbox_xywh)
            mask = outputs.get('pred_mask')
            if mask is not None:
                mask = (mask * 255).astype(np.uint8)
                mask = postprocess_mask(mask, fill_holes=True, 
                                      keep_largest_only=True)
            return mask
        except Exception as e:
            logger.warning("DAM4SAM mask generation failed.", extra={'error': e})
            return None

    def _ground_first_frame_mask_xywh(self, frame_rgb_small: np.ndarray, 
                                     text: str, box_th: float, text_th: float):
        """Ground first frame and refine with SAM mask."""
        from app.core.logging import UnifiedLogger
        logger = UnifiedLogger()
        
        xywh, details = self._ground_first_frame_xywh(frame_rgb_small, text, 
                                                     box_th, text_th)
        if xywh is None:
            return None, details
        mask = self._sam2_mask_for_bbox(frame_rgb_small, xywh)
        if mask is None:
            logger.warning("SAM2 mask generation failed. Falling back to box.")
            return xywh, details
        ys, xs = np.where(mask > 128)
        if ys.size == 0:
            return xywh, details
        x1, x2, y1, y2 = xs.min(), xs.max()+1, ys.min(), ys.max()+1
        refined_xywh = [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]
        refined_details = {**details, "type": "text_prompt_mask"}
        return refined_xywh, refined_details

    def _seed_identity(self, frame_rgb, current_params=None):
        """Main seed selection logic."""
        from app.core.logging import UnifiedLogger
        logger = UnifiedLogger()
        
        p = self.params if current_params is None else current_params

        prompt_text = getattr(p, "text_prompt", "")
        if isinstance(current_params, dict):
            prompt_text = current_params.get('text_prompt', prompt_text)

        if prompt_text:
            box_th = getattr(p, "box_threshold", self.params.box_threshold)
            text_th = getattr(p, "text_threshold", self.params.text_threshold)
            if isinstance(current_params, dict):
                box_th = current_params.get('box_threshold', box_th)
                text_th = current_params.get('text_threshold', text_th)

            xywh, details = self._ground_first_frame_mask_xywh(
                frame_rgb, prompt_text, box_th, text_th)
            if xywh is not None:
                logger.info("Text-prompt seed found", extra=details)
                return xywh, details
            else:
                logger.warning("Text-prompt grounding returned no boxes; "
                             "falling back.")

        return self._choose_seed_bbox(frame_rgb, p)

    def _choose_seed_bbox(self, frame_rgb, current_params):
        """Choose seed bounding box using fallback strategies."""
        from app.core.logging import UnifiedLogger
        logger = UnifiedLogger()
        
        frame_bgr_for_face = None
        if self.face_analyzer:
            frame_bgr_for_face = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        # Try face matching first
        if (self.face_analyzer and self.reference_embedding is not None and 
            current_params.enable_face_filter):
            faces = (self.face_analyzer.get(frame_bgr_for_face) 
                    if frame_bgr_for_face is not None else [])
            if faces:
                best_face, best_dist = None, float('inf')
                for face in faces:
                    dist = 1 - np.dot(face.normed_embedding, 
                                    self.reference_embedding)
                    if dist < best_dist:
                        best_dist, best_face = dist, face

                if best_face and best_dist < 0.6:
                    details = {'type': 'face_match', 
                             'seed_face_sim': 1 - best_dist}
                    face_bbox = best_face.bbox.astype(int)
                    final_bbox = self._get_body_box_for_face(frame_rgb, 
                                                           face_bbox, details)
                    return final_bbox, details

        logger.info("No matching face. Applying fallback seeding.",
                   extra={'strategy': current_params.seed_strategy})

        # Try person detection strategies
        if (current_params.seed_strategy in ["Largest Person", 
                                           "Center-most Person"] and 
            self.person_detector):
            boxes = self.person_detector.detect_boxes(frame_rgb)
            if boxes:
                h, w = frame_rgb.shape[:2]
                cx, cy = w / 2, h / 2
                strategy_map = {
                    "Largest Person": lambda b: (b[2] - b[0]) * (b[3] - b[1]),
                    "Center-most Person": lambda b: -math.hypot(
                        (b[0] + b[2]) / 2 - cx, (b[1] + b[3]) / 2 - cy)
                }
                score_func = strategy_map[current_params.seed_strategy]
                x1, y1, x2, y2, _ = sorted(boxes, key=score_func, 
                                         reverse=True)[0]
                strategy_type = (f'person_{current_params.seed_strategy.lower()}'
                               f'.replace(" ", "_")')
                return [x1, y1, x2 - x1, y2 - y1], {'type': strategy_type}

        # Try largest face fallback
        if self.face_analyzer:
            faces = self.face_analyzer.get(frame_bgr_for_face)
            if faces:
                largest_face = max(faces, key=lambda f: (
                    (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])))
                details = {'type': 'face_largest'}
                face_bbox = largest_face.bbox.astype(int)
                final_bbox = self._get_body_box_for_face(frame_rgb, face_bbox, 
                                                       details)
                return final_bbox, details

        # Final fallback
        logger.warning("No faces or persons found to seed shot. "
                      "Using fallback rectangle.")
        h, w, _ = frame_rgb.shape
        return [w // 4, h // 4, w // 2, h // 2], {'type': 'fallback_rect'}

    def _get_body_box_for_face(self, frame_rgb, face_bbox, details_dict):
        """Get body bounding box for a detected face."""
        x1, y1, x2, y2 = face_bbox
        person_bbox = self._pick_person_box_for_face(
            frame_rgb, [x1, y1, x2-x1, y2-y1])
        if person_bbox:
            details_dict['type'] = f'person_box_from_{details_dict["type"]}'
            return person_bbox
        else:
            expanded_box = self._expand_face_to_body([x1, y1, x2-x1, y2-y1], 
                                                   frame_rgb.shape)
            details_dict['type'] = f'expanded_box_from_{details_dict["type"]}'
            return expanded_box

    def _pick_person_box_for_face(self, frame_rgb, face_bbox):
        """Pick best person bounding box that contains the face."""
        from app.core.logging import UnifiedLogger
        logger = UnifiedLogger()
        
        if not self.person_detector:
            return None
        px1, py1, pw, ph = face_bbox
        fx, fy = px1 + pw / 2.0, py1 + ph / 2.0
        try:
            candidates = self.person_detector.detect_boxes(frame_rgb)
        except Exception as e:
            logger.warning("Person detector failed on frame.", 
                          extra={'error': e})
            return None
        if not candidates:
            return None

        def iou(b):
            ix1, iy1 = max(b[0], px1), max(b[1], py1)
            ix2, iy2 = min(b[2], px1 + pw), min(b[3], py1 + ph)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            union = (b[2]-b[0])*(b[3]-b[1]) + pw*ph - inter + 1e-6
            return inter / union

        pool = sorted(candidates, key=lambda b: (
            (b[0] <= fx <= b[2] and b[1] <= fy <= b[3]), iou(b), b[4]), 
            reverse=True)
        best_box = pool[0]
        if (not (best_box[0] <= fx <= best_box[2] and 
                best_box[1] <= fy <= best_box[3]) and iou(best_box) < 0.1):
            return None
        return [best_box[0], best_box[1], 
                best_box[2] - best_box[0], best_box[3] - best_box[1]]

    def _expand_face_to_body(self, face_bbox, img_shape):
        """Expand face bounding box to approximate body region."""
        H, W = img_shape[:2]
        x, y, w, h = face_bbox
        cx = x + w / 2
        new_w = min(W, w * 4.0)
        new_h = min(H, h * 7.0)
        new_x = max(0, cx - new_w / 2)
        new_y = max(0, y - h * 0.75)
        return [int(v) for v in [new_x, new_y, min(new_w, W-new_x), 
                                min(new_h, H-new_y)]]
