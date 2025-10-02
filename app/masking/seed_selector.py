"""Seed selection logic for subject masking."""

import math
import cv2
import numpy as np
import torch
from app.ml.grounding import predict_grounding_dino
from torchvision import transforms
from torchvision.ops import box_convert


class SeedSelector:
    """Handles the logic for selecting the initial seed (bounding box) for a scene."""

    def __init__(self, params, face_analyzer, reference_embedding,
                 person_detector, tracker, gdino_model):
        from app.core.logging import UnifiedLogger
        self.params = params
        self.face_analyzer = face_analyzer
        self.reference_embedding = reference_embedding
        self.person_detector = person_detector
        self.tracker = tracker
        self._gdino = gdino_model
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger = UnifiedLogger()

    def select_seed(self, frame_rgb, current_params=None):
        """
        Main entry point for seed selection.
        Chooses between 'Identity-First' or 'Object-First' mode.
        """
        p = self.params if current_params is None else current_params
        use_face_filter = getattr(p, "enable_face_filter", False)
        
        if isinstance(current_params, dict):
            use_face_filter = current_params.get('enable_face_filter', use_face_filter)

        if self.face_analyzer and self.reference_embedding is not None and use_face_filter:
            self.logger.info("Starting 'Identity-First' seeding.")
            return self._identity_first_seed(frame_rgb, p)
        else:
            self.logger.info("Starting 'Object-First' seeding.")
            return self._object_first_seed(frame_rgb, p)

    def _identity_first_seed(self, frame_rgb, params):
        """Mode 1: Reference Face Provided (The 'Identity-First' Approach)"""
        # Step 1: Find the Anchor (InsightFace)
        target_face, details = self._find_target_face(frame_rgb)
        if not target_face:
            self.logger.warning("Target face not found in scene.", extra=details)
            return None, {"type": "no_subject_found"}

        # Step 2: Gather Corroborating Evidence (YOLO & DINO)
        yolo_boxes = self._get_yolo_boxes(frame_rgb)
        dino_boxes, _ = self._get_dino_boxes(frame_rgb, params)

        # Step 3: Score and Select the Best Candidate
        best_box, best_details = self._score_and_select_candidate(
            target_face, yolo_boxes, dino_boxes
        )

        # Step 4: The Final Seed
        if best_box:
            self.logger.success("Evidence-based seed selected.", extra=best_details)
            return best_box, best_details

        # Fallback: Expand face box if no good candidate is found
        self.logger.warning("No high-confidence body box found, expanding face box as fallback.")
        expanded_box = self._expand_face_to_body(target_face['bbox'], frame_rgb.shape)
        fallback_details = {
            "type": "expanded_box_from_face",
            "seed_face_sim": details.get('seed_face_sim', 0)
        }
        return expanded_box, fallback_details

    def _object_first_seed(self, frame_rgb, params):
        """Mode 2: No Reference Face Provided (The 'Object-First' Approach)"""
        # Step 1: Find the Object (Grounding DINO)
        dino_boxes, dino_details = self._get_dino_boxes(frame_rgb, params)

        if dino_boxes:
            # Step 2: Validate with Person Detection (YOLO)
            yolo_boxes = self._get_yolo_boxes(frame_rgb)
            if yolo_boxes:
                # Find best intersection between DINO and YOLO
                best_iou = -1
                best_match = None
                for d_box in dino_boxes:
                    for y_box in yolo_boxes:
                        iou = self._calculate_iou(d_box['bbox'], y_box['bbox'])
                        if iou > best_iou:
                            best_iou = iou
                            # Combine info, prioritize DINO box but could be averaged
                            best_match = {
                                'bbox': d_box['bbox'],
                                'type': 'dino_yolo_intersect',
                                'iou': iou,
                                'dino_conf': d_box['conf'],
                                'yolo_conf': y_box['conf']
                            }
                if best_match and best_match['iou'] > 0.3: # Confidence threshold for intersection
                    self.logger.info("Found high-confidence DINO+YOLO intersection.", extra=best_match)
                    return self._xyxy_to_xywh(best_match['bbox']), best_match

            # If no good intersection, return the best DINO box
            self.logger.info("Using best DINO box without YOLO validation.", extra=dino_details)
            return self._xyxy_to_xywh(dino_boxes[0]['bbox']), dino_details

        # Step 3: Fallback to General Detection (YOLO-only)
        self.logger.info("No DINO results, falling back to YOLO-only strategy.")
        return self._choose_person_by_strategy(frame_rgb, params)

    def _find_target_face(self, frame_rgb):
        """Find the face in the frame that best matches the reference embedding."""
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        try:
            faces = self.face_analyzer.get(frame_bgr)
        except Exception as e:
            self.logger.error("Face analysis failed.", exc_info=True)
            return None, {"error": str(e)}
            
        if not faces:
            return None, {"error": "no_faces_detected"}

        best_face, best_dist = None, float('inf')
        for face in faces:
            dist = 1 - np.dot(face.normed_embedding, self.reference_embedding)
            if dist < best_dist:
                best_dist, best_face = dist, face

        if best_face and best_dist < 0.6: # Similarity threshold
            details = {'type': 'face_match', 'seed_face_sim': 1 - best_dist}
            face_data = {'bbox': best_face.bbox.astype(int), 'embedding': best_face.normed_embedding}
            return face_data, details
        
        return None, {'error': 'no_matching_face', 'best_dist': best_dist}

    def _get_yolo_boxes(self, frame_rgb):
        """Get all 'person' bounding boxes from YOLO."""
        if not self.person_detector:
            return []
        try:
            boxes = self.person_detector.detect_boxes(frame_rgb)
            # Convert to dicts with xyxy format
            return [{'bbox': b[:4], 'conf': b[4], 'type': 'yolo'} for b in boxes]
        except Exception as e:
            self.logger.warning("YOLO person detector failed.", exc_info=True)
            return []

    def _get_dino_boxes(self, frame_rgb, params):
        """Get all matching bounding boxes from Grounding DINO."""
        prompt = getattr(params, "text_prompt", "")
        if isinstance(params, dict):
            prompt = params.get('text_prompt', prompt)
            
        if not self._gdino or not prompt:
            return [], {}

        box_th = getattr(params, "box_threshold", self.params.box_threshold)
        text_th = getattr(params, "text_threshold", self.params.text_threshold)
        if isinstance(params, dict):
            box_th = params.get('box_threshold', box_th)
            text_th = params.get('text_threshold', text_th)

        image_source, image_tensor = self._load_image_from_array(frame_rgb)
        h, w = image_source.shape[:2]

        try:
            boxes_norm, confs, labels = predict_grounding_dino(
                model=self._gdino, image_tensor=image_tensor, caption=prompt,
                box_threshold=float(box_th), text_threshold=float(text_th),
                device=self._device
            )
        except Exception as e:
            self.logger.error("Grounding DINO prediction failed.", exc_info=True)
            return [], {"error": str(e)}

        if boxes_norm is None or len(boxes_norm) == 0:
            return [], {"type": "text_prompt", "error": "no_boxes"}

        scale = torch.tensor([w, h, w, h], device=boxes_norm.device, dtype=boxes_norm.dtype)
        boxes_abs = (boxes_norm * scale).cpu()
        xyxy_boxes = box_convert(boxes=boxes_abs, in_fmt="cxcywh", out_fmt="xyxy").numpy()
        
        results = []
        for i, box in enumerate(xyxy_boxes):
            results.append({
                'bbox': box.astype(int),
                'conf': confs[i].item(),
                'label': labels[i],
                'type': 'dino'
            })
        
        # Sort by confidence
        results.sort(key=lambda x: x['conf'], reverse=True)
        details = {**results[0], "all_boxes_count": len(results)}
        return results, details

    def _score_and_select_candidate(self, target_face, yolo_boxes, dino_boxes):
        """Scores candidate boxes and selects the best one based on evidence."""
        candidates = yolo_boxes + dino_boxes
        if not candidates:
            return None, {}

        scored_candidates = []
        for cand in candidates:
            score = 0
            details = {'orig_conf': cand['conf'], 'orig_type': cand['type']}

            # 1. Face Containment (Highest Weight)
            face_box = target_face['bbox']
            if self._box_contains(cand['bbox'], face_box):
                score += 100
                details['face_contained'] = True
            
            # 2. Model Confidence (Medium Weight)
            score += cand['conf'] * 20 # Scale confidence to be significant
            
            # Add to list
            scored_candidates.append({'score': score, 'box': cand['bbox'], 'details': details})

        # 3. Model Intersection (High Weight) - check best YOLO/DINO pair
        best_iou = -1
        best_pair = None
        for y_box in yolo_boxes:
            for d_box in dino_boxes:
                iou = self._calculate_iou(y_box['bbox'], d_box['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_pair = (y_box, d_box)
        
        if best_iou > 0.5: # High IoU threshold
            # Find the candidate that is part of this high-IoU pair and boost its score
            for cand in scored_candidates:
                if np.array_equal(cand['box'], best_pair[0]['bbox']) or \
                   np.array_equal(cand['box'], best_pair[1]['bbox']):
                    cand['score'] += 50
                    cand['details']['high_iou_pair'] = True

        if not scored_candidates:
            return None, {}

        # Select the winner
        winner = max(scored_candidates, key=lambda x: x['score'])
        final_details = {
            'type': 'evidence_based_selection',
            'final_score': winner['score'],
            **winner['details']
        }
        return self._xyxy_to_xywh(winner['box']), final_details

    def _choose_person_by_strategy(self, frame_rgb, params):
        """Fallback to choose a person based on a simple strategy (e.g., largest)."""
        boxes = self._get_yolo_boxes(frame_rgb)
        if not boxes:
            self.logger.warning("No persons found for fallback strategy.")
            return self._final_fallback_box(frame_rgb.shape), {'type': 'fallback_rect'}

        strategy = getattr(params, "seed_strategy", "Largest Person")
        if isinstance(params, dict):
            strategy = params.get('seed_strategy', strategy)

        h, w = frame_rgb.shape[:2]
        cx, cy = w / 2, h / 2

        def largest_person_score(b):
            box = b['bbox']
            return (box[2] - box[0]) * (box[3] - box[1])

        def centermost_person_score(b):
            box = b['bbox']
            return -math.hypot((box[0] + box[2]) / 2 - cx, (box[1] + box[3]) / 2 - cy)

        score_func = {
            "Largest Person": largest_person_score,
            "Center-most Person": centermost_person_score
        }.get(strategy, largest_person_score)

        best_person = sorted(boxes, key=score_func, reverse=True)[0]
        details = {
            'type': f'person_{strategy.lower().replace(" ", "_")}',
            'conf': best_person['conf']
        }
        return self._xyxy_to_xywh(best_person['bbox']), details

    # --- Utility and Helper Functions ---

    def _load_image_from_array(self, image_rgb: np.ndarray):
        """Load image from numpy array for grounding DINO."""
        transform = transforms.Compose([
            transforms.ToPILImage(), transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        image_tensor = transform(image_rgb)
        return image_rgb, image_tensor

    def _calculate_iou(self, box1, box2):
        """Calculate Intersection over Union (IoU) for two bounding boxes (xyxy)."""
        x1, y1, x2, y2 = box1
        x1_p, y1_p, x2_p, y2_p = box2

        inter_x1 = max(x1, x1_p)
        inter_y1 = max(y1, y1_p)
        inter_x2 = min(x2, x2_p)
        inter_y2 = min(y2, y2_p)

        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x2_p - x1_p) * (y2_p - y1_p)
        
        union_area = box1_area + box2_area - inter_area
        return inter_area / (union_area + 1e-6)

    def _box_contains(self, container_box, inner_box):
        """Check if container_box (xyxy) fully contains inner_box (xyxy)."""
        return (container_box[0] <= inner_box[0] and
                container_box[1] <= inner_box[1] and
                container_box[2] >= inner_box[2] and
                container_box[3] >= inner_box[3])

    def _expand_face_to_body(self, face_bbox, img_shape):
        """Expand face bounding box (xyxy) to approximate body region."""
        H, W = img_shape[:2]
        x1, y1, x2, y2 = face_bbox
        w, h = x2 - x1, y2 - y1
        cx = x1 + w / 2
        
        new_w = min(W, w * 4.0)
        new_h = min(H, h * 7.0)
        new_x1 = max(0, cx - new_w / 2)
        new_y1 = max(0, y1 - h * 0.75)
        new_x2 = min(W, new_x1 + new_w)
        new_y2 = min(H, new_y1 + new_h)
        
        return [int(v) for v in [new_x1, new_y1, new_x2 - new_x1, new_y2 - new_y1]]

    def _final_fallback_box(self, img_shape):
        """Return a generic center box as a last resort."""
        h, w, _ = img_shape
        return [w // 4, h // 4, w // 2, h // 2]

    def _xyxy_to_xywh(self, box):
        """Convert xyxy bbox to xywh."""
        x1, y1, x2, y2 = box
        return [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]

    # Keep this method if it's used by other parts of the application
    def _ground_first_frame_mask_xywh(self, frame_rgb_small: np.ndarray,
                                     text: str, box_th: float, text_th: float):
        """Ground first frame and refine with SAM mask."""
        dino_boxes, details = self._get_dino_boxes(frame_rgb_small, {
            "text_prompt": text, "box_threshold": box_th, "text_threshold": text_th
        })
        if not dino_boxes:
            return None, details
        
        xywh = self._xyxy_to_xywh(dino_boxes[0]['bbox'])
        
        mask = self._sam2_mask_for_bbox(frame_rgb_small, xywh)
        if mask is None:
            self.logger.warning("SAM2 mask generation failed. Falling back to box.")
            return xywh, details
            
        ys, xs = np.where(mask > 128)
        if ys.size == 0:
            return xywh, details
            
        x1, x2, y1, y2 = xs.min(), xs.max()+1, ys.min(), ys.max()+1
        refined_xywh = [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]
        refined_details = {**details, "type": "text_prompt_mask"}
        return refined_xywh, refined_details

    def _sam2_mask_for_bbox(self, frame_rgb_small, bbox_xywh):
        """Generate SAM2 mask for bounding box."""
        from app.io.frames import rgb_to_pil, postprocess_mask
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
            self.logger.warning("DAM4SAM mask generation failed.", extra={'error': e})
            return None
