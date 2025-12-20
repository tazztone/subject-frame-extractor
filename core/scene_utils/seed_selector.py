"""
SeedSelector class for selecting seed frames and bounding boxes for mask propagation.
"""
from __future__ import annotations
import math
from typing import Optional, Union, Any, TYPE_CHECKING
import numpy as np
import cv2
import torch

if TYPE_CHECKING:
    from core.config import Config
    from core.logger import AppLogger
    from core.models import AnalysisParameters, Scene
    from core.managers import SAM3Wrapper
    from insightface.app import FaceAnalysis

from core.utils import rgb_to_pil, postprocess_mask


class SeedSelector:
    """
    Selects seed frames and bounding boxes for mask propagation.
    
    Supports multiple strategies:
    - Identity-first (face matching)
    - Object-first (text prompt)
    - Face + text fallback
    - Automatic (person detection with various scoring)
    """
    
    def __init__(
        self,
        params: 'AnalysisParameters',
        config: 'Config',
        face_analyzer: 'FaceAnalysis',
        reference_embedding: np.ndarray,
        tracker: 'SAM3Wrapper',
        logger: 'AppLogger',
        device: str = "cpu"
    ):
        """
        Initialize the SeedSelector.
        
        Args:
            params: Analysis parameters
            config: Application configuration
            face_analyzer: InsightFace analyzer for face detection/recognition
            reference_embedding: Reference face embedding for identity matching
            tracker: SAM3 wrapper for object detection
            logger: Application logger
            device: Device to run on ('cpu' or 'cuda')
        """
        self.params = params
        self.config = config
        self.face_analyzer = face_analyzer
        self.reference_embedding = reference_embedding
        self.tracker = tracker
        self._device = device
        self.logger = logger

    def _get_param(self, source: Union[dict, object], key: str, default: Any = None) -> Any:
        """Get a parameter from either a dict or an object."""
        if isinstance(source, dict):
            return source.get(key, default)
        return getattr(source, key, default)

    def select_seed(
        self,
        frame_rgb: np.ndarray,
        current_params: Optional[dict] = None,
        scene: Optional['Scene'] = None
    ) -> tuple[Optional[list], dict]:
        """
        Select a seed bounding box for the given frame.
        
        Args:
            frame_rgb: RGB frame as numpy array
            current_params: Optional override parameters
            scene: Optional scene context
            
        Returns:
            Tuple of (bbox_xywh, details_dict)
        """
        params_source = current_params if current_params is not None else self.params
        p = params_source
        primary_strategy = self._get_param(params_source, 'primary_seed_strategy', "🤖 Automatic")
        use_face_filter = self._get_param(params_source, 'enable_face_filter', False)

        if primary_strategy == "👤 By Face":
            if self.face_analyzer and self.reference_embedding is not None and use_face_filter:
                self.logger.info("Starting 'Identity-First' seeding.")
                return self._identity_first_seed(frame_rgb, p, scene)
            else:
                self.logger.warning("Face strategy selected but no reference face provided.")
                return self._object_first_seed(frame_rgb, p, scene)
        elif primary_strategy == "📝 By Text":
            self.logger.info("Starting 'Object-First' seeding.")
            return self._object_first_seed(frame_rgb, p, scene)
        elif primary_strategy == "🔄 Face + Text Fallback":
            self.logger.info("Starting 'Face-First with Text Fallback' seeding.")
            return self._face_with_text_fallback_seed(frame_rgb, p, scene)
        else:
            self.logger.info("Starting 'Automatic' seeding.")
            return self._choose_person_by_strategy(frame_rgb, p, scene)

    def _face_with_text_fallback_seed(
        self,
        frame_rgb: np.ndarray,
        params: Union[dict, 'AnalysisParameters'],
        scene: Optional['Scene'] = None
    ) -> tuple[Optional[list], dict]:
        """Try face-first, fall back to text prompt if face not found."""
        if self.reference_embedding is None:
            self.logger.warning(
                "No reference face for face-first strategy, falling back to text prompt.",
                extra={'reason': 'no_ref_emb'}
            )
            return self._object_first_seed(frame_rgb, params, scene)
        box, details = self._identity_first_seed(frame_rgb, params, scene)
        if box is not None:
            self.logger.info("Face-first strategy successful.")
            return box, details
        self.logger.warning(
            "Face detection failed or no match found, falling back to text prompt strategy.",
            extra=details
        )
        return self._object_first_seed(frame_rgb, params, scene)

    def _identity_first_seed(
        self,
        frame_rgb: np.ndarray,
        params: Union[dict, 'AnalysisParameters'],
        scene: Optional['Scene'] = None
    ) -> tuple[Optional[list], dict]:
        """Find subject by matching to reference face."""
        target_face, details = self._find_target_face(frame_rgb)
        if not target_face:
            self.logger.warning("Target face not found in scene.", extra=details)
            return None, {"type": "no_subject_found"}
        person_boxes = self._get_person_boxes(frame_rgb, scene)
        text_boxes = self._get_text_prompt_boxes(frame_rgb, params)[0]
        best_box, best_details = self._score_and_select_candidate(target_face, person_boxes, text_boxes)
        if best_box:
            self.logger.success("Evidence-based seed selected.", extra=best_details)
            return best_box, best_details
        self.logger.warning("No high-confidence body box found, expanding face box as fallback.")
        expanded_box = self._expand_face_to_body(target_face['bbox'], frame_rgb.shape)
        return expanded_box, {"type": "expanded_box_from_face", "seed_face_sim": details.get('seed_face_sim', 0)}

    def _object_first_seed(
        self,
        frame_rgb: np.ndarray,
        params: Union[dict, 'AnalysisParameters'],
        scene: Optional['Scene'] = None
    ) -> tuple[Optional[list], dict]:
        """Find subject using text prompt, validated by person detection."""
        text_boxes, text_details = self._get_text_prompt_boxes(frame_rgb, params)
        if text_boxes:
            person_boxes = self._get_person_boxes(frame_rgb, scene)
            if person_boxes:
                best_iou, best_match = -1, None
                for d_box in text_boxes:
                    for y_box in person_boxes:
                        iou = self._calculate_iou(d_box['bbox'], y_box['bbox'])
                        if iou > best_iou:
                            best_iou = iou
                            best_match = {
                                'bbox': d_box['bbox'],
                                'type': 'sam3_intersect',
                                'iou': iou,
                                'text_conf': d_box['conf'],
                                'person_conf': y_box['conf']
                            }
                if best_match and best_match['iou'] > self.config.seeding_yolo_iou_threshold:
                    self.logger.info("Found high-confidence intersection.", extra=best_match)
                    return self._xyxy_to_xywh(best_match['bbox']), best_match
            self.logger.info("Using best text box without validation.", extra=text_details)
            return self._xyxy_to_xywh(text_boxes[0]['bbox']), text_details
        self.logger.info("No text results, falling back to person-only strategy.")
        return self._choose_person_by_strategy(frame_rgb, params, scene)

    def _find_target_face(self, frame_rgb: np.ndarray) -> tuple[Optional[dict], dict]:
        """Find the target face in frame that matches reference embedding."""
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        try:
            faces = self.face_analyzer.get(frame_bgr)
        except Exception as e:
            self.logger.error("Face analysis failed.", exc_info=True)
            return None, {"error": str(e)}
        if not faces:
            return None, {"error": "no_faces_detected"}
        best_face, best_sim = None, 0.0
        for face in faces:
            sim = np.dot(face.normed_embedding, self.reference_embedding)
            if sim > best_sim:
                best_sim, best_face = sim, face
        if best_face and best_sim > self.config.seeding_face_similarity_threshold:
            return {
                'bbox': best_face.bbox.astype(int),
                'embedding': best_face.normed_embedding
            }, {'type': 'face_match', 'seed_face_sim': best_sim}
        return None, {'error': 'no_matching_face', 'best_sim': best_sim}

    def _get_person_boxes(
        self,
        frame_rgb: np.ndarray,
        scene: Optional['Scene'] = None
    ) -> list[dict]:
        """Get person bounding boxes from scene cache or detection."""
        if scene and getattr(scene, 'yolo_detections', None):
            return scene.yolo_detections
        if scene and (scene.selected_bbox or scene.initial_bbox):
            xywh = scene.selected_bbox or scene.initial_bbox
            x, y, w, h = xywh
            xyxy = [x, y, x + w, y + h]
            return [{'bbox': xyxy, 'conf': 1.0, 'type': 'selected'}]
        if not self.tracker:
            return []
        try:
            return self.tracker.detect_objects(frame_rgb, "person")
        except Exception:
            self.logger.warning("Person detection failed.", exc_info=True)
            return []

    def _get_text_prompt_boxes(
        self,
        frame_rgb: np.ndarray,
        params: Union[dict, 'AnalysisParameters']
    ) -> tuple[list[dict], dict]:
        """Get bounding boxes from text prompt detection."""
        prompt = self._get_param(params, "text_prompt", "").strip()
        if not self.tracker or not prompt:
            return [], {}
        try:
            results = self.tracker.detect_objects(frame_rgb, prompt)
        except Exception as e:
            self.logger.error("Text prompt prediction failed.", exc_info=True)
            return [], {"error": str(e)}
        if not results:
            return [], {"type": "text_prompt", "error": "no_boxes"}
        return results, {**results[0], "all_boxes_count": len(results)}

    def _score_and_select_candidate(
        self,
        target_face: dict,
        person_boxes: list[dict],
        text_boxes: list[dict]
    ) -> tuple[Optional[list], dict]:
        """Score and select the best candidate box that contains the target face."""
        candidates = person_boxes + text_boxes
        if not candidates:
            return None, {}
        scored_candidates = []
        for cand in candidates:
            score = 0
            details = {'orig_conf': cand['conf'], 'orig_type': cand['type']}
            if self._box_contains(cand['bbox'], target_face['bbox']):
                score += self.config.seeding_face_contain_score
                details['face_contained'] = True
            score += cand['conf'] * self.config.seeding_confidence_score_multiplier
            scored_candidates.append({'score': score, 'box': cand['bbox'], 'details': details})
        
        # Bonus for high IoU between person and text boxes
        best_iou, best_pair = -1, None
        for y_box in person_boxes:
            for d_box in text_boxes:
                iou = self._calculate_iou(y_box['bbox'], d_box['bbox'])
                if iou > best_iou:
                    best_iou, best_pair = iou, (y_box, d_box)
        if best_iou > self.config.seeding_yolo_iou_threshold:
            for cand in scored_candidates:
                if (np.array_equal(cand['box'], best_pair[0]['bbox']) or 
                        np.array_equal(cand['box'], best_pair[1]['bbox'])):
                    cand['score'] += self.config.seeding_iou_bonus
                    cand['details']['high_iou_pair'] = True
        
        if not scored_candidates:
            return None, {}
        winner = max(scored_candidates, key=lambda x: x['score'])
        return self._xyxy_to_xywh(winner['box']), {
            'type': 'evidence_based_selection',
            'final_score': winner['score'],
            **winner['details']
        }

    def _choose_person_by_strategy(
        self,
        frame_rgb: np.ndarray,
        params: Union[dict, 'AnalysisParameters'],
        scene: Optional['Scene'] = None
    ) -> tuple[list, dict]:
        """Select person using configurable strategy."""
        boxes = self._get_person_boxes(frame_rgb, scene)
        if not boxes:
            self.logger.warning("No people detected in scene - using fallback region")
            fallback_box = self._final_fallback_box(frame_rgb.shape)
            return fallback_box, {
                'type': 'no_people_fallback',
                'reason': 'No people detected in best frame',
                'detection_attempted': True
            }
        strategy = getattr(params, "seed_strategy", "Largest Person")
        if isinstance(params, dict):
            strategy = params.get('seed_strategy', strategy)
        h, w = frame_rgb.shape[:2]
        cx, cy = w / 2, h / 2

        def area(b):
            x1, y1, x2, y2 = b['bbox']
            return (x2 - x1) * (y2 - y1)
        
        def height(b):
            x1, y1, x2, y2 = b['bbox']
            return y2 - y1
        
        def center_dist(b):
            x1, y1, x2, y2 = b['bbox']
            bx, by = (x1 + x2) / 2, (y1 + y2) / 2
            return math.hypot(bx - cx, by - cy)
        
        def thirds_dist(b):
            thirds = [
                (w / 3, h / 3), (2 * w / 3, h / 3),
                (w / 3, 2 * h / 3), (2 * w / 3, 2 * h / 3)
            ]
            x1, y1, x2, y2 = b['bbox']
            bx, by = (x1 + x2) / 2, (y1 + y2) / 2
            return min(math.hypot(bx - tx, by - ty) for tx, ty in thirds)
        
        def min_dist_to_edge(b):
            x1, y1, x2, y2 = b['bbox']
            return min(x1, y1, w - x2, h - y2)
        
        def balanced_score(b):
            weights = self.config.seeding_balanced_score_weights
            norm_area = area(b) / (w * h)
            norm_edge = min_dist_to_edge(b) / (min(w, h) / 2)
            return (weights['area'] * norm_area + 
                    weights['confidence'] * b['conf'] + 
                    weights['edge'] * norm_edge)

        all_faces = None
        if strategy == "Best Face" and self.face_analyzer:
            all_faces = self.face_analyzer.get(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))

        def best_face_score(b):
            if not all_faces:
                return 0.0
            yolo_bbox = b['bbox']
            faces_in_box = []
            for face in all_faces:
                face_cx = face.bbox[0] + face.bbox[2] / 2
                face_cy = face.bbox[1] + face.bbox[3] / 2
                if yolo_bbox[0] <= face_cx < yolo_bbox[2] and yolo_bbox[1] <= face_cy < yolo_bbox[3]:
                    faces_in_box.append(face)
            if not faces_in_box:
                return 0.0
            return max(f.det_score for f in faces_in_box)

        score_funcs = {
            "Largest Person": lambda b: area(b),
            "Center-most Person": lambda b: -center_dist(b),
            "Highest Confidence": lambda b: b['conf'],
            "Tallest Person": lambda b: height(b),
            "Area x Confidence": lambda b: area(b) * b['conf'],
            "Rule-of-Thirds": lambda b: -thirds_dist(b),
            "Edge-avoiding": lambda b: min_dist_to_edge(b),
            "Balanced": balanced_score,
            "Best Face": best_face_score,
        }
        score = score_funcs.get(strategy, score_funcs["Largest Person"])
        best_person = sorted(boxes, key=lambda b: (score(b), b['conf'], area(b)), reverse=True)[0]
        return self._xyxy_to_xywh(best_person['bbox']), {
            'type': f'person_{strategy.lower().replace(" ", "_")}',
            'conf': best_person['conf']
        }

    def _load_image_from_array(self, image_rgb: np.ndarray) -> tuple[np.ndarray, torch.Tensor]:
        """Load image for model input."""
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return image_rgb, transform(image_rgb)

    def _calculate_iou(self, box1: list, box2: list) -> float:
        """Calculate IoU between two boxes in xyxy format."""
        x1, y1, x2, y2 = box1
        x1_p, y1_p, x2_p, y2_p = box2
        inter_x1 = max(x1, x1_p)
        inter_y1 = max(y1, y1_p)
        inter_x2 = min(x2, x2_p)
        inter_y2 = min(y2, y2_p)
        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        union_area = (x2 - x1) * (y2 - y1) + (x2_p - x1_p) * (y2_p - y1_p) - inter_area
        return inter_area / (union_area + 1e-6)

    def _box_contains(self, cb: list, ib: list) -> bool:
        """Check if container box (cb) contains inner box (ib)."""
        return cb[0] <= ib[0] and cb[1] <= ib[1] and cb[2] >= ib[2] and cb[3] >= ib[3]

    def _expand_face_to_body(self, face_bbox: list, img_shape: tuple) -> list[int]:
        """Expand a face bounding box to approximate body bounding box."""
        H, W = img_shape[:2]
        x1, y1, x2, y2 = face_bbox
        w, h = x2 - x1, y2 - y1
        cx = x1 + w / 2
        expansion_factors = self.config.seeding_face_to_body_expansion_factors
        new_w = min(W, w * expansion_factors[0])
        new_h = min(H, h * expansion_factors[1])
        new_x1 = max(0, cx - new_w / 2)
        new_y1 = max(0, y1 - h * expansion_factors[2])
        return [
            int(new_x1),
            int(new_y1),
            int(min(W, new_x1 + new_w) - new_x1),
            int(min(H, new_y1 + new_h) - new_y1)
        ]

    def _final_fallback_box(self, img_shape: tuple) -> list[int]:
        """Return a fallback bounding box when no subject is found."""
        h, w = img_shape[:2]
        fallback_box = self.config.seeding_final_fallback_box
        return [
            int(w * fallback_box[0]),
            int(h * fallback_box[1]),
            int(w * fallback_box[2]),
            int(h * fallback_box[3])
        ]

    def _xyxy_to_xywh(self, box: list) -> list[int]:
        """Convert box from xyxy to xywh format."""
        x1, y1, x2, y2 = box
        return [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]

    def _sam2_mask_for_bbox(
        self,
        frame_rgb_small: np.ndarray,
        bbox_xywh: list
    ) -> Optional[np.ndarray]:
        """Generate a mask for the given bounding box using SAM3."""
        if not self.tracker or bbox_xywh is None:
            return None
        try:
            import tempfile
            import os
            
            # Save frame to temp directory for SAM3 init_state
            temp_dir = tempfile.mkdtemp()
            pil_img = rgb_to_pil(frame_rgb_small)
            pil_img.save(os.path.join(temp_dir, "00000.jpg"))
            
            # Use new SAM3 API
            h, w = frame_rgb_small.shape[:2]
            self.tracker.init_video(temp_dir)
            mask = self.tracker.add_bbox_prompt(
                frame_idx=0, obj_id=1, bbox_xywh=bbox_xywh, img_size=(w, h)
            )
            
            # Cleanup temp directory
            import shutil
            try:
                shutil.rmtree(temp_dir)
            except Exception:
                pass
            
            if mask is not None:
                mask = postprocess_mask(
                    (mask * 255).astype(np.uint8),
                    config=self.config,
                    fill_holes=True,
                    keep_largest_only=True
                )
            return mask
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            self.logger.warning(f"GPU error in mask generation: {e}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error in mask generation: {e}")
            return None
