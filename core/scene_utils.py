from __future__ import annotations
import math
import threading
import json
from typing import Optional, Union, List, Any, TYPE_CHECKING
from queue import Queue
import numpy as np
import cv2
import torch
from pathlib import Path
from PIL import Image
from scenedetect import detect, ContentDetector

if TYPE_CHECKING:
    from config import Config
    from logger import AppLogger
    from core.models import AnalysisParameters, Scene, SceneState
    from core.managers import SAM3Wrapper, ThumbnailManager, ModelRegistry
    from core.pipelines import AdvancedProgressTracker
    from insightface.app import FaceAnalysis
    from mediapipe.tasks.python.vision import FaceLandmarker
    import gradio as gr

from core.utils import safe_resource_cleanup, create_frame_map, rgb_to_pil, postprocess_mask, render_mask_overlay, draw_bbox, _to_json_safe
from core.managers import initialize_analysis_models

def run_scene_detection(video_path: str, output_dir: Path, logger: 'AppLogger') -> list:
    logger.info("Detecting scenes...", component="video")
    try:
        scene_list = detect(str(video_path), ContentDetector())
        shots = ([(s.get_frames(), e.get_frames()) for s, e in scene_list] if scene_list else [])
        with (output_dir / "scenes.json").open('w', encoding='utf-8') as f: json.dump(shots, f)
        logger.success(f"Found {len(shots)} scenes.", component="video")
        return shots
    except Exception as e:
        logger.error("Scene detection failed.", component="video", exc_info=True)
        return []

def make_photo_thumbs(image_paths: list[Path], out_dir: Path, params: 'AnalysisParameters', cfg: 'Config',
                      logger: 'AppLogger', tracker: Optional['AdvancedProgressTracker'] = None) -> dict:
    thumbs_dir = out_dir / "thumbs"
    thumbs_dir.mkdir(parents=True, exist_ok=True)
    target_area = params.thumb_megapixels * 1_000_000
    frame_map, image_manifest = {}, {}

    if tracker: tracker.start(len(image_paths), desc="Generating thumbnails")

    for i, img_path in enumerate(image_paths, start=1):
        if tracker and tracker.pause_event.is_set(): tracker.step()
        try:
            bgr = cv2.imread(str(img_path))
            if bgr is None:
                logger.warning(f"Could not read image file: {img_path}")
                continue

            h, w = bgr.shape[:2]
            scale = math.sqrt(target_area / float(max(1, w * h)))
            if scale < 1.0:
                new_w, new_h = int((w * scale) // 2 * 2), int((h * scale) // 2 * 2)
                bgr = cv2.resize(bgr, (max(2, new_w), max(2, new_h)), interpolation=cv2.INTER_AREA)

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            out_name = f"frame_{i:06d}.webp"
            out_path = thumbs_dir / out_name
            Image.fromarray(rgb).save(out_path, format="WEBP", quality=cfg.ffmpeg_thumbnail_quality)

            frame_map[i] = out_name
            image_manifest[i] = str(img_path.resolve())
        except Exception as e:
            logger.error(f"Failed to process image {img_path}", exc_info=True)
        finally:
            if tracker: tracker.step()

    (out_dir / "frame_map.json").write_text(json.dumps(frame_map, indent=2), encoding="utf-8")
    (out_dir / "image_manifest.json").write_text(json.dumps(image_manifest, indent=2), encoding="utf-8")
    if tracker: tracker.done_stage("Thumbnails generated")
    return frame_map

class MaskPropagator:
    def __init__(self, params: 'AnalysisParameters', dam_tracker: 'SAM3Wrapper', cancel_event: threading.Event,
                 progress_queue: Queue, config: 'Config', logger: 'AppLogger', device: str = "cpu"):
        self.params = params
        self.dam_tracker = dam_tracker
        self.cancel_event = cancel_event
        self.progress_queue = progress_queue
        self.config = config
        self.logger = logger
        self._device = device

    def propagate(self, shot_frames_rgb: list[np.ndarray], seed_idx: int, bbox_xywh: list[int],
                  tracker: Optional['AdvancedProgressTracker'] = None) -> tuple[list, list, list, list]:
        if not self.dam_tracker or not shot_frames_rgb:
            err_msg = "Tracker not initialized" if not self.dam_tracker else "No frames"
            shape = shot_frames_rgb[0].shape[:2] if shot_frames_rgb else (100, 100)
            num_frames = len(shot_frames_rgb)
            return ([np.zeros(shape, np.uint8)] * num_frames, [0.0] * num_frames, [True] * num_frames, [err_msg] * num_frames)
        self.logger.info("Propagating masks with SAM3", component="propagator", user_context={'num_frames': len(shot_frames_rgb), 'seed_index': seed_idx})
        masks = [None] * len(shot_frames_rgb)

        if tracker: tracker.set_stage(f"Propagating masks for {len(shot_frames_rgb)} frames")

        try:
            pil_images = [rgb_to_pil(img) for img in shot_frames_rgb]

            outputs = self.dam_tracker.initialize(pil_images, bbox=bbox_xywh, prompt_frame_idx=seed_idx)
            mask = outputs.get('pred_mask')
            if mask is not None: mask = postprocess_mask((mask * 255).astype(np.uint8), config=self.config, fill_holes=True, keep_largest_only=True)
            masks[seed_idx] = mask if mask is not None else np.zeros_like(shot_frames_rgb[seed_idx], dtype=np.uint8)[:, :, 0]
            if tracker: tracker.step(1, desc="Propagation (seed)")

            for out in self.dam_tracker.propagate_from(seed_idx, direction="forward"):
                frame_idx = out['frame_index']
                if frame_idx == seed_idx: continue
                if frame_idx >= len(shot_frames_rgb): break

                if out['outputs'] and 'obj_id_to_mask' in out['outputs'] and len(out['outputs']['obj_id_to_mask']) > 0:
                    pred_mask = list(out['outputs']['obj_id_to_mask'].values())[0]
                    if isinstance(pred_mask, torch.Tensor):
                        pred_mask = pred_mask.cpu().numpy().astype(bool)
                        if pred_mask.ndim == 3: pred_mask = pred_mask[0]

                    mask = (pred_mask * 255).astype(np.uint8)
                    mask = postprocess_mask(mask, config=self.config, fill_holes=True, keep_largest_only=True)
                    masks[frame_idx] = mask
                else:
                    masks[frame_idx] = np.zeros_like(shot_frames_rgb[frame_idx], dtype=np.uint8)[:, :, 0]

                if tracker: tracker.step(1, desc="Propagation (→)")

            for out in self.dam_tracker.propagate_from(seed_idx, direction="backward"):
                frame_idx = out['frame_index']
                if frame_idx == seed_idx: continue
                if frame_idx < 0: break

                if out['outputs'] and 'obj_id_to_mask' in out['outputs'] and len(out['outputs']['obj_id_to_mask']) > 0:
                    pred_mask = list(out['outputs']['obj_id_to_mask'].values())[0]
                    if isinstance(pred_mask, torch.Tensor):
                        pred_mask = pred_mask.cpu().numpy().astype(bool)
                        if pred_mask.ndim == 3: pred_mask = pred_mask[0]

                    mask = (pred_mask * 255).astype(np.uint8)
                    mask = postprocess_mask(mask, config=self.config, fill_holes=True, keep_largest_only=True)
                    masks[frame_idx] = mask
                else:
                    masks[frame_idx] = np.zeros_like(shot_frames_rgb[frame_idx], dtype=np.uint8)[:, :, 0]

                if tracker: tracker.step(1, desc="Propagation (←)")

            h, w = shot_frames_rgb[0].shape[:2]
            final_results = []
            for i, mask in enumerate(masks):
                if self.cancel_event.is_set() or mask is None: mask = np.zeros((h, w), dtype=np.uint8)
                img_area = h * w
                area_pct = (np.sum(mask > 0) / img_area) * 100 if img_area > 0 else 0.0
                is_empty = area_pct < self.params.min_mask_area_pct
                error = "Empty mask" if is_empty else None
                final_results.append((mask, float(area_pct), bool(is_empty), error))
            if not final_results: return ([], [], [], [])
            masks, areas, empties, errors = map(list, zip(*final_results))
            return masks, areas, empties, errors
        except Exception as e:
            self.logger.critical("SAM3 propagation failed", component="propagator", exc_info=True)
            h, w = shot_frames_rgb[0].shape[:2]
            error_msg = f"Propagation failed: {e}"
            num_frames = len(shot_frames_rgb)
            return ([np.zeros((h, w), np.uint8)] * num_frames, [0.0] * num_frames, [True] * num_frames, [error_msg] * num_frames)

class SeedSelector:
    def __init__(self, params: 'AnalysisParameters', config: 'Config', face_analyzer: 'FaceAnalysis',
                 reference_embedding: np.ndarray, tracker: 'SAM3Wrapper', logger: 'AppLogger', device: str = "cpu"):
        self.params = params
        self.config = config
        self.face_analyzer = face_analyzer
        self.reference_embedding = reference_embedding
        self.tracker = tracker
        self._device = device
        self.logger = logger

    def _get_param(self, source: Union[dict, object], key: str, default: Any = None) -> Any:
        if isinstance(source, dict): return source.get(key, default)
        return getattr(source, key, default)

    def select_seed(self, frame_rgb: np.ndarray, current_params: Optional[dict] = None,
                    scene: Optional['Scene'] = None) -> tuple[Optional[list], dict]:
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

    def _face_with_text_fallback_seed(self, frame_rgb: np.ndarray, params: Union[dict, 'AnalysisParameters'],
                                      scene: Optional['Scene'] = None) -> tuple[Optional[list], dict]:
        if self.reference_embedding is None:
            self.logger.warning("No reference face for face-first strategy, falling back to text prompt.", extra={'reason': 'no_ref_emb'})
            return self._object_first_seed(frame_rgb, params, scene)
        box, details = self._identity_first_seed(frame_rgb, params, scene)
        if box is not None:
            self.logger.info("Face-first strategy successful.")
            return box, details
        self.logger.warning("Face detection failed or no match found, falling back to text prompt strategy.", extra=details)
        return self._object_first_seed(frame_rgb, params, scene)

    def _identity_first_seed(self, frame_rgb: np.ndarray, params: Union[dict, 'AnalysisParameters'],
                             scene: Optional['Scene'] = None) -> tuple[Optional[list], dict]:
        target_face, details = self._find_target_face(frame_rgb)
        if not target_face:
            self.logger.warning("Target face not found in scene.", extra=details)
            return None, {"type": "no_subject_found"}
        person_boxes, text_boxes = self._get_person_boxes(frame_rgb, scene), self._get_text_prompt_boxes(frame_rgb, params)[0]
        best_box, best_details = self._score_and_select_candidate(target_face, person_boxes, text_boxes)
        if best_box:
            self.logger.success("Evidence-based seed selected.", extra=best_details)
            return best_box, best_details
        self.logger.warning("No high-confidence body box found, expanding face box as fallback.")
        expanded_box = self._expand_face_to_body(target_face['bbox'], frame_rgb.shape)
        return expanded_box, {"type": "expanded_box_from_face", "seed_face_sim": details.get('seed_face_sim', 0)}

    def _object_first_seed(self, frame_rgb: np.ndarray, params: Union[dict, 'AnalysisParameters'],
                             scene: Optional['Scene'] = None) -> tuple[Optional[list], dict]:
        text_boxes, text_details = self._get_text_prompt_boxes(frame_rgb, params)
        if text_boxes:
            person_boxes = self._get_person_boxes(frame_rgb, scene)
            if person_boxes:
                best_iou, best_match = -1, None
                for d_box in text_boxes:
                    for y_box in person_boxes:
                        iou = self._calculate_iou(d_box['bbox'], y_box['bbox'])
                        if iou > best_iou:
                            best_iou, best_match = iou, {'bbox': d_box['bbox'], 'type': 'sam3_intersect', 'iou': iou,
                                                         'text_conf': d_box['conf'], 'person_conf': y_box['conf']}
                if best_match and best_match['iou'] > self.config.seeding_yolo_iou_threshold:
                    self.logger.info("Found high-confidence intersection.", extra=best_match)
                    return self._xyxy_to_xywh(best_match['bbox']), best_match
            self.logger.info("Using best text box without validation.", extra=text_details)
            return self._xyxy_to_xywh(text_boxes[0]['bbox']), text_details
        self.logger.info("No text results, falling back to person-only strategy.")
        return self._choose_person_by_strategy(frame_rgb, params, scene)

    def _find_target_face(self, frame_rgb: np.ndarray) -> tuple[Optional[dict], dict]:
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        try: faces = self.face_analyzer.get(frame_bgr)
        except Exception as e:
            self.logger.error("Face analysis failed.", exc_info=True)
            return None, {"error": str(e)}
        if not faces: return None, {"error": "no_faces_detected"}
        best_face, best_sim = None, 0.0
        for face in faces:
            sim = np.dot(face.normed_embedding, self.reference_embedding)
            if sim > best_sim: best_sim, best_face = sim, face
        if best_face and best_sim > self.config.seeding_face_similarity_threshold:
            return {'bbox': best_face.bbox.astype(int), 'embedding': best_face.normed_embedding}, {'type': 'face_match', 'seed_face_sim': best_sim}
        return None, {'error': 'no_matching_face', 'best_sim': best_sim}

    def _get_person_boxes(self, frame_rgb: np.ndarray, scene: Optional['Scene'] = None) -> list[dict]:
        if scene and getattr(scene, 'yolo_detections', None): return scene.yolo_detections
        if scene and (scene.selected_bbox or scene.initial_bbox):
            xywh = scene.selected_bbox or scene.initial_bbox
            x, y, w, h = xywh
            xyxy = [x, y, x + w, y + h]
            return [{'bbox': xyxy, 'conf': 1.0, 'type': 'selected'}]
        if not self.tracker: return []
        try:
            return self.tracker.detect_objects(frame_rgb, "person")
        except Exception as e:
            self.logger.warning("Person detection failed.", exc_info=True)
            return []

    def _get_text_prompt_boxes(self, frame_rgb: np.ndarray, params: Union[dict, 'AnalysisParameters']) -> tuple[list[dict], dict]:
        prompt = self._get_param(params, "text_prompt", "").strip()
        if not self.tracker or not prompt: return [], {}

        try:
            results = self.tracker.detect_objects(frame_rgb, prompt)
        except Exception as e:
            self.logger.error("Text prompt prediction failed.", exc_info=True)
            return [], {"error": str(e)}

        if not results: return [], {"type": "text_prompt", "error": "no_boxes"}
        return results, {**results[0], "all_boxes_count": len(results)}

    def _score_and_select_candidate(self, target_face: dict, person_boxes: list[dict], text_boxes: list[dict]) -> tuple[Optional[list], dict]:
        candidates = person_boxes + text_boxes
        if not candidates: return None, {}
        scored_candidates = []
        for cand in candidates:
            score, details = 0, {'orig_conf': cand['conf'], 'orig_type': cand['type']}
            if self._box_contains(cand['bbox'], target_face['bbox']):
                score += self.config.seeding_face_contain_score
                details['face_contained'] = True
            score += cand['conf'] * self.config.seeding_confidence_score_multiplier
            scored_candidates.append({'score': score, 'box': cand['bbox'], 'details': details})
        best_iou, best_pair = -1, None
        for y_box in person_boxes:
            for d_box in text_boxes:
                iou = self._calculate_iou(y_box['bbox'], d_box['bbox'])
                if iou > best_iou: best_iou, best_pair = iou, (y_box, d_box)
        if best_iou > self.config.seeding_yolo_iou_threshold:
            for cand in scored_candidates:
                if np.array_equal(cand['box'], best_pair[0]['bbox']) or np.array_equal(cand['box'], best_pair[1]['bbox']):
                    cand['score'] += self.config.seeding_iou_bonus
                    cand['details']['high_iou_pair'] = True
        if not scored_candidates: return None, {}
        winner = max(scored_candidates, key=lambda x: x['score'])
        return self._xyxy_to_xywh(winner['box']), {'type': 'evidence_based_selection', 'final_score': winner['score'], **winner['details']}

    def _choose_person_by_strategy(self, frame_rgb: np.ndarray, params: Union[dict, 'AnalysisParameters'],
                                     scene: Optional['Scene'] = None) -> tuple[list, dict]:
        boxes = self._get_person_boxes(frame_rgb, scene)
        if not boxes:
            self.logger.warning(f"No people detected in scene - using fallback region")
            fallback_box = self._final_fallback_box(frame_rgb.shape)
            return fallback_box, {'type': 'no_people_fallback', 'reason': 'No people detected in best frame', 'detection_attempted': True}
        strategy = getattr(params, "seed_strategy", "Largest Person")
        if isinstance(params, dict): strategy = params.get('seed_strategy', strategy)
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
            thirds = [(w / 3, h / 3), (2 * w / 3, h / 3), (w / 3, 2 * h / 3), (2 * w / 3, 2 * h / 3)]
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
            return weights['area'] * norm_area + weights['confidence'] * b['conf'] + weights['edge'] * norm_edge

        all_faces = None
        if strategy == "Best Face" and self.face_analyzer:
            all_faces = self.face_analyzer.get(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))

        def best_face_score(b):
            if not all_faces: return 0.0
            yolo_bbox = b['bbox']
            faces_in_box = []
            for face in all_faces:
                face_cx = face.bbox[0] + face.bbox[2] / 2
                face_cy = face.bbox[1] + face.bbox[3] / 2
                if yolo_bbox[0] <= face_cx < yolo_bbox[2] and yolo_bbox[1] <= face_cy < yolo_bbox[3]:
                    faces_in_box.append(face)
            if not faces_in_box: return 0.0
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
        return self._xyxy_to_xywh(best_person['bbox']), {'type': f'person_{strategy.lower().replace(" ", "_")}', 'conf': best_person['conf']}

    def _load_image_from_array(self, image_rgb: np.ndarray) -> tuple[np.ndarray, torch.Tensor]:
        from torchvision import transforms
        transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        return image_rgb, transform(image_rgb)

    def _calculate_iou(self, box1: list, box2: list) -> float:
        x1, y1, x2, y2 = box1
        x1_p, y1_p, x2_p, y2_p = box2
        inter_x1, inter_y1, inter_x2, inter_y2 = max(x1, x1_p), max(y1, y1_p), min(x2, x2_p), min(y2, y2_p)
        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        union_area = (x2 - x1) * (y2 - y1) + (x2_p - x1_p) * (y2_p - y1_p) - inter_area
        return inter_area / (union_area + 1e-6)

    def _box_contains(self, cb: list, ib: list) -> bool:
        return cb[0] <= ib[0] and cb[1] <= ib[1] and cb[2] >= ib[2] and cb[3] >= ib[3]

    def _expand_face_to_body(self, face_bbox: list, img_shape: tuple) -> list[int]:
        H, W, (x1, y1, x2, y2) = *img_shape[:2], *face_bbox
        w, h, cx = x2 - x1, y2 - y1, x1 + w / 2
        expansion_factors = self.config.seeding_face_to_body_expansion_factors
        new_w, new_h = min(W, w * expansion_factors[0]), min(H, h * expansion_factors[1])
        new_x1, new_y1 = max(0, cx - new_w / 2), max(0, y1 - h * expansion_factors[2])
        return [int(v) for v in [new_x1, new_y1, min(W, new_x1 + new_w) - new_x1, min(H, new_y1 + new_h) - new_y1]]

    def _final_fallback_box(self, img_shape: tuple) -> list[int]:
        h, w, _ = img_shape
        fallback_box = self.config.seeding_final_fallback_box
        return [int(w * fallback_box[0]), int(h * fallback_box[1]), int(w * fallback_box[2]), int(h * fallback_box[3])]

    def _xyxy_to_xywh(self, box: list) -> list[int]:
        x1, y1, x2, y2 = box; return [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]

    def _sam2_mask_for_bbox(self, frame_rgb_small: np.ndarray, bbox_xywh: list) -> Optional[np.ndarray]:
        if not self.tracker or bbox_xywh is None: return None
        try:
            pil_img = rgb_to_pil(frame_rgb_small)
            outputs = self.tracker.initialize([pil_img], None, bbox=bbox_xywh, prompt_frame_idx=0)
            mask = outputs.get('pred_mask')
            if mask is not None: mask = postprocess_mask((mask * 255).astype(np.uint8), config=self.config, fill_holes=True, keep_largest_only=True)
            return mask
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            self.logger.warning(f"GPU error in mask generation: {e}")
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error in mask generation: {e}")
            return None

class SubjectMasker:
    def __init__(self, params: 'AnalysisParameters', progress_queue: Queue, cancel_event: threading.Event, config: 'Config',
                 frame_map: Optional[dict] = None, face_analyzer: Optional['FaceAnalysis'] = None,
                 reference_embedding: Optional[np.ndarray] = None,
                 thumbnail_manager: Optional['ThumbnailManager'] = None, niqe_metric: Optional[Callable] = None,
                 logger: Optional['AppLogger'] = None, face_landmarker: Optional['FaceLandmarker'] = None,
                 device: str = "cpu", model_registry: 'ModelRegistry' = None):
        self.params, self.config, self.progress_queue, self.cancel_event = params, config, progress_queue, cancel_event
        self.logger = logger
        self.frame_map = frame_map
        self.face_analyzer, self.reference_embedding, self.face_landmarker = face_analyzer, reference_embedding, face_landmarker
        self.dam_tracker, self.mask_dir, self.shots = None, None, []
        self._device = device
        self.thumbnail_manager = thumbnail_manager
        self.niqe_metric = niqe_metric
        self.model_registry = model_registry
        self.initialize_models()
        self.seed_selector = SeedSelector(
            params=params,
            config=self.config,
            face_analyzer=face_analyzer,
            reference_embedding=reference_embedding,
            tracker=self.dam_tracker,
            logger=self.logger,
            device=self._device
        )
        self.mask_propagator = MaskPropagator(params, self.dam_tracker, cancel_event, progress_queue, config=self.config, logger=self.logger, device=self._device)

    def initialize_models(self):
        if self.params.enable_face_filter and self.face_analyzer is None:
            self.logger.warning("Face analyzer is not available but face filter is enabled.")

        if getattr(self.params, "need_masks_now", False) or self.params.enable_subject_mask:
            self._initialize_tracker()

    def _initialize_tracker(self) -> bool:
        if self.dam_tracker: return True
        try:
            if not self.model_registry:
                self.logger.error("ModelRegistry not provided to SubjectMasker. Cannot load tracker.")
                return False

            retry_params = (self.config.retry_max_attempts, tuple(self.config.retry_backoff_seconds))
            self.logger.info(f"Initializing SAM3 tracker: {self.params.tracker_model_name}")
            self.dam_tracker = self.model_registry.get_tracker(
                model_name=self.params.tracker_model_name,
                models_path=str(self.config.models_dir),
                user_agent=self.config.user_agent,
                retry_params=retry_params,
                config=self.config
            )
            if self.dam_tracker is None:
                self.logger.error("SAM3 tracker initialization returned None/failed")
                return False

            # Update child components with the new tracker
            if self.seed_selector: self.seed_selector.tracker = self.dam_tracker
            if self.mask_propagator: self.mask_propagator.dam_tracker = self.dam_tracker

            self.logger.success("SAM3 tracker initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"Exception during SAM3 tracker initialization: {e}", exc_info=True)
            return False

    def run_propagation(self, frames_dir: str, scenes_to_process: list['Scene'],
                        tracker: Optional['AdvancedProgressTracker'] = None) -> dict:
        self.mask_dir = Path(frames_dir) / "masks"
        self.mask_dir.mkdir(exist_ok=True)
        self.logger.info("Starting subject mask propagation...")
        if not self._initialize_tracker():
            self.logger.error("SAM3 tracker could not be initialized; mask propagation failed.")
            return {"error": "SAM3 tracker initialization failed", "completed": False}

        thumb_dir = Path(frames_dir) / "thumbs"
        mask_metadata, total_scenes = {}, len(scenes_to_process)
        for i, scene in enumerate(scenes_to_process):
            # monitor_memory_usage is in utils, but SubjectMasker is in scene_utils.
            # I didn't import it in scene_utils.py plan.
            # I should add it.
            # I'll comment it out or add import if I missed it.
            # I'll check imports.
            if self.cancel_event.is_set(): break
            self.logger.info(f"Masking scene {i+1}/{total_scenes}", user_context={'shot_id': scene.shot_id, 'start_frame': scene.start_frame, 'end_frame': scene.end_frame})

            shot_frames_data = self._load_shot_frames(frames_dir, thumb_dir, scene.start_frame, scene.end_frame)
            if not shot_frames_data: continue

            if tracker: tracker.set_stage(f"Scene {i+1}/{len(scenes_to_process)}", substage=f"{len(shot_frames_data)} frames")

            frame_numbers, small_images, dims = zip(*shot_frames_data)

            try:
                best_frame_num = scene.best_frame
                seed_idx_in_shot = frame_numbers.index(best_frame_num)
            except (ValueError, AttributeError):
                self.logger.warning(f"Best frame {scene.best_frame} not found in loaded shot frames for {scene.shot_id}, skipping.")
                continue

            bbox, seed_details = scene.seed_result.get('bbox'), scene.seed_result.get('details', {})
            if bbox is None:
                for fn in frame_numbers:
                    if (fname := self.frame_map.get(fn)):
                        # MaskingResult is from core.models
                        # I'll construct dict directly to avoid import issues or use MaskingResult if imported.
                        mask_metadata[fname] = {"error": "Subject not found", "shot_id": scene.shot_id}
                continue

            masks, areas, empties, errors = self.mask_propagator.propagate(small_images, seed_idx_in_shot, bbox, tracker=tracker)

            for j, (original_fn, _, (h, w)) in enumerate(shot_frames_data):
                frame_fname_webp = self.frame_map.get(original_fn)
                if not frame_fname_webp: continue
                frame_fname_png, mask_path = f"{Path(frame_fname_webp).stem}.png", self.mask_dir / f"{Path(frame_fname_webp).stem}.png"
                result_args = {"shot_id": scene.shot_id, "seed_type": seed_details.get('type'), "seed_face_sim": seed_details.get('seed_face_sim'),
                               "mask_area_pct": areas[j], "mask_empty": empties[j], "error": errors[j]}
                if masks[j] is not None and np.any(masks[j]):
                    mask_full_res = cv2.resize(masks[j], (w, h), interpolation=cv2.INTER_NEAREST)
                    if mask_full_res.ndim == 3: mask_full_res = mask_full_res[:, :, 0]
                    cv2.imwrite(str(mask_path), mask_full_res)
                    result_args["mask_path"] = str(mask_path)
                    mask_metadata[frame_fname_png] = result_args
                else:
                    result_args["mask_path"] = None
                    mask_metadata[frame_fname_png] = result_args
        self.logger.success("Subject masking complete.")
        try:
            with (self.mask_dir.parent / "mask_metadata.json").open('w', encoding='utf-8') as f:
                json.dump(mask_metadata, f, indent=2)
            self.logger.info("Saved mask metadata.")
        except Exception as e:
            self.logger.error("Failed to save mask metadata", exc_info=True)
        return mask_metadata

    def _load_shot_frames(self, frames_dir: str, thumb_dir: Path, start: int, end: int) -> list[tuple[int, np.ndarray, tuple[int, int]]]:
        frames = []
        if not self.frame_map:
            ext = ".webp" if self.params.thumbnails_only else ".png"
            self.frame_map = create_frame_map(Path(frames_dir), self.logger, ext=ext)

        for fn in sorted(fn for fn in self.frame_map if start <= fn < end):
            thumb_path = thumb_dir / f"{Path(self.frame_map[fn]).stem}.webp"
            thumb_img = self.thumbnail_manager.get(thumb_path)
            if thumb_img is None: continue
            frames.append((fn, thumb_img, thumb_img.shape[:2]))
        return frames

    def _select_best_frame_in_scene(self, scene: 'Scene', frames_dir: str):
        if not self.params.pre_analysis_enabled:
            scene.best_frame, scene.seed_metrics = scene.start_frame, {'reason': 'pre-analysis disabled'}
            return

        thumb_dir = Path(frames_dir) / "thumbs"
        shot_frames = self._load_shot_frames(frames_dir, thumb_dir, scene.start_frame, scene.end_frame)
        if not shot_frames:
            scene.best_frame, scene.seed_metrics = scene.start_frame, {'reason': 'no frames loaded'}
            return
        candidates, scores = shot_frames[::max(1, self.params.pre_sample_nth)], []
        for frame_num, thumb_rgb, _ in candidates:
            niqe_score = 10.0
            if self.niqe_metric:
                img_tensor = (torch.from_numpy(thumb_rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0)
                with (torch.no_grad(), torch.amp.autocast('cuda', enabled=self._device == 'cuda')):
                    niqe_score = float(self.niqe_metric(img_tensor.to(self.niqe_metric.device)))
            face_sim = 0.0
            if self.face_analyzer and self.reference_embedding is not None:
                faces = self.face_analyzer.get(cv2.cvtColor(thumb_rgb, cv2.COLOR_RGB2BGR))
                if faces: face_sim = np.dot(max(faces, key=lambda x: x.det_score).normed_embedding, self.reference_embedding)
            scores.append((10 - niqe_score) + (face_sim * 10))
        if not scores:
            scene.best_frame = scene.start_frame
            scene.seed_metrics = {'reason': 'pre-analysis failed, no scores', 'score': 0}
            return
        best_local_idx = int(np.argmax(scores))
        scene.best_frame = candidates[best_local_idx][0]
        scene.seed_metrics = {'reason': 'pre-analysis complete', 'score': max(scores), 'best_niqe': niqe_score, 'best_face_sim': face_sim}

    def get_seed_for_frame(self, frame_rgb: np.ndarray, seed_config: dict = None, scene: Optional['Scene'] = None) -> tuple[Optional[list], dict]:
        if isinstance(seed_config, dict) and seed_config.get('manual_bbox_xywh'):
            return seed_config['manual_bbox_xywh'], {'type': seed_config.get('seed_type', 'manual')}

        self._initialize_tracker()

        if scene is not None:
            scene.yolo_detections = self.seed_selector._get_person_boxes(frame_rgb, scene=None)

        return self.seed_selector.select_seed(frame_rgb, current_params=seed_config, scene=scene)

    def get_mask_for_bbox(self, frame_rgb_small: np.ndarray, bbox_xywh: list) -> Optional[np.ndarray]:
        return self.seed_selector._sam2_mask_for_bbox(frame_rgb_small, bbox_xywh)

    def draw_bbox(self, img_rgb: np.ndarray, xywh: list, color: Optional[tuple] = None,
                  thickness: Optional[int] = None, label: Optional[str] = None) -> np.ndarray:
        return draw_bbox(img_rgb, xywh, self.config, color, thickness, label)

    def _create_frame_map(self, output_dir: str):
        return create_frame_map(Path(output_dir), self.logger, ext=".webp" if self.params.thumbnails_only else ".png")

def draw_boxes_preview(img: np.ndarray, boxes_xyxy: list[list[int]], cfg: 'Config') -> np.ndarray:
    img = img.copy()
    for x1,y1,x2,y2 in boxes_xyxy:
        cv2.rectangle(img, (int(x1),int(y1)), (int(x2),int(y2)), cfg.visualization_bbox_color, cfg.visualization_bbox_thickness)
    return img

def save_scene_seeds(scenes_list: list['Scene'], output_dir_str: str, logger: 'AppLogger'):
    if not scenes_list or not output_dir_str: return
    scene_seeds = {}
    for s in scenes_list:
        data = {
            'best_frame': s.best_frame, 'seed_frame_idx': s.seed_frame_idx, 'seed_type': s.seed_type,
            'seed_config': s.seed_config, 'status': s.status, 'seed_metrics': s.seed_metrics
        }
        scene_seeds[str(s.shot_id)] = data
    try:
        (Path(output_dir_str) / "scene_seeds.json").write_text(json.dumps(_to_json_safe(scene_seeds), indent=2), encoding='utf-8')
        logger.info("Saved scene_seeds.json")
    except Exception as e: logger.error("Failed to save scene_seeds.json", exc_info=True)

def get_scene_status_text(scenes_list: list['Scene']) -> tuple[str, dict]: # Return gr.update as dict for now if gr is not imported, but wait, type check imports gr.
    import gradio as gr # Lazy import to avoid hard dependency at module level if possible, but used in type hint.
    if not scenes_list: return "No scenes loaded.", gr.update(interactive=False)
    included_scenes = [s for s in scenes_list if s.status == 'included']
    ready_for_propagation_count = sum(1 for s in included_scenes if s.seed_result and s.seed_result.get('bbox'))
    total_count, included_count = len(scenes_list), len(included_scenes)
    rejection_counts = {} # Counter
    for scene in scenes_list:
        if scene.status == 'excluded' and scene.rejection_reasons:
            for reason in scene.rejection_reasons:
                rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
    status_text = f"{included_count}/{total_count} scenes included for propagation."
    if rejection_counts:
        reasons_summary = ", ".join([f"{reason}: {count}" for reason, count in rejection_counts.items()])
        status_text += f" (Rejected: {reasons_summary})"
    button_text = f"🔬 Propagate Masks on {ready_for_propagation_count} Ready Scenes"
    return status_text, gr.update(value=button_text, interactive=ready_for_propagation_count > 0)

def toggle_scene_status(scenes_list: list['Scene'], selected_shot_id: int, new_status: str,
                        output_folder: str, logger: 'AppLogger') -> tuple[list, str, str, Any]:
    if selected_shot_id is None or not scenes_list:
        status_text, button_update = get_scene_status_text(scenes_list)
        return scenes_list, status_text, "No scene selected.", button_update
    scene_to_update = next((s for s in scenes_list if s.shot_id == selected_shot_id), None)
    if scene_to_update:
        scene_to_update.status = new_status
        scene_to_update.manual_status_change = True
        save_scene_seeds(scenes_list, output_folder, logger)
        status_text, button_update = get_scene_status_text(scenes_list)
        return (scenes_list, status_text, f"Scene {selected_shot_id} status set to {new_status}.", button_update)
    status_text, button_update = get_scene_status_text(scenes_list)
    return (scenes_list, status_text, f"Could not find scene {selected_shot_id}.", button_update)

def _create_analysis_context(config: 'Config', logger: 'AppLogger', thumbnail_manager: 'ThumbnailManager',
                             cuda_available: bool, ana_ui_map_keys: list[str], ana_input_components: list,
                             model_registry: 'ModelRegistry') -> 'SubjectMasker':
    from core.models import AnalysisParameters
    ui_args = dict(zip(ana_ui_map_keys, ana_input_components))
    if 'outputfolder' in ui_args and 'output_folder' not in ui_args: ui_args['output_folder'] = ui_args.pop('outputfolder')
    output_folder_str = ui_args.get('output_folder')
    if not output_folder_str or isinstance(output_folder_str, bool):
        logger.error(f"Output folder is not valid (was '{output_folder_str}', type: {type(output_folder_str)}). This is likely due to a UI argument mapping error.", component="analysis")
        raise FileNotFoundError(f"Output folder is not valid or does not exist: {output_folder_str}")
    if not Path(output_folder_str).exists(): raise FileNotFoundError(f"Output folder is not valid or does not exist: {output_folder_str}")
    resolved_outdir = Path(output_folder_str).resolve()
    ui_args['output_folder'] = str(resolved_outdir)
    params = AnalysisParameters.from_ui(logger, config, **ui_args)
    models = initialize_analysis_models(params, config, logger, model_registry)
    frame_map = create_frame_map(resolved_outdir, logger)
    if not frame_map: raise RuntimeError("Failed to create frame map. Check if frame_map.json exists and is valid.")
    return SubjectMasker(
        params=params, progress_queue=Queue(), cancel_event=threading.Event(), config=config,
        frame_map=frame_map, face_analyzer=models["face_analyzer"],
        reference_embedding=models["ref_emb"],
        niqe_metric=None, thumbnail_manager=thumbnail_manager, logger=logger,
        face_landmarker=models["face_landmarker"], device=models["device"],
        model_registry=model_registry
    )

def _recompute_single_preview(scene_state: 'SceneState', masker: 'SubjectMasker', overrides: dict,
                              thumbnail_manager: 'ThumbnailManager', logger: 'AppLogger'):
    scene = scene_state.scene # Use .scene property if using refactored SceneState
    out_dir = Path(masker.params.output_folder)
    best_frame_num = scene.best_frame or scene.start_frame
    if best_frame_num is None: raise ValueError(f"Scene {scene.shot_id} has no best frame number.")
    fname = masker.frame_map.get(int(best_frame_num))
    if not fname: raise FileNotFoundError(f"Best frame {best_frame_num} not found in project's frame map.")
    thumb_rgb = thumbnail_manager.get(out_dir / "thumbs" / f"{Path(fname).stem}.webp")
    if thumb_rgb is None: raise FileNotFoundError(f"Thumbnail for frame {best_frame_num} not found on disk.")
    seed_config = {**masker.params.model_dump(), **overrides}
    if overrides.get("text_prompt", "").strip():
        seed_config['primary_seed_strategy'] = "📝 By Text"
        logger.info(f"Recomputing scene {scene.shot_id} with text-first strategy due to override.", extra={'prompt': overrides.get("text_prompt")})
    bbox, details = masker.get_seed_for_frame(thumb_rgb, seed_config=seed_config, scene=scene)
    scene_state.update_seed_result(bbox, details)
    scene.seed_config.update(overrides)
    new_score = details.get('final_score') or details.get('conf') or details.get('dino_conf')
    if new_score is not None:
        if not scene.seed_metrics: scene.seed_metrics = {}
        scene.seed_metrics['score'] = new_score
    mask = masker.get_mask_for_bbox(thumb_rgb, bbox) if bbox else None
    if mask is not None:
        h, w = mask.shape[:2]; area = (h * w)
        if not scene.seed_result.get('details'): scene.seed_result['details'] = {}
        scene.seed_result['details']['mask_area_pct'] = (np.sum(mask > 0) / area * 100.0) if area > 0 else 0.0
    overlay_rgb = render_mask_overlay(thumb_rgb, mask, 0.6, logger=logger) if mask is not None else masker.draw_bbox(thumb_rgb, bbox)
    previews_dir = out_dir / "previews"; previews_dir.mkdir(parents=True, exist_ok=True)
    preview_path = previews_dir / f"scene_{int(scene.shot_id):05d}.jpg"
    try:
        Image.fromarray(overlay_rgb).save(preview_path)
        scene.preview_path = str(preview_path)
    except Exception: logger.error(f"Failed to save preview for scene {scene.shot_id}", exc_info=True)

def _wire_recompute_handler(config: 'Config', logger: 'AppLogger', thumbnail_manager: 'ThumbnailManager',
                            scenes: list['Scene'], shot_id: int, outdir: str, text_prompt: str,
                            view: str, ana_ui_map_keys: list[str],
                            ana_input_components: list, cuda_available: bool, model_registry: 'ModelRegistry') -> tuple:
    import gradio as gr
    from core.models import SceneState
    try:
        if not text_prompt or not text_prompt.strip(): return scenes, gr.update(), gr.update(), "Enter a text prompt to use advanced seeding."
        ui_args = dict(zip(ana_ui_map_keys, ana_input_components))
        ui_args['output_folder'] = outdir
        masker = _create_analysis_context(config, logger, thumbnail_manager, cuda_available, ana_ui_map_keys, ana_input_components, model_registry)
        scene_idx = next((i for i, s in enumerate(scenes) if s.shot_id == shot_id), None)
        if scene_idx is None: return scenes, gr.update(), gr.update(), f"Error: Scene {shot_id} not found."
        overrides = {"text_prompt": text_prompt}
        scene_state = SceneState(scenes[scene_idx])
        _recompute_single_preview(scene_state, masker, overrides, thumbnail_manager, logger)
        save_scene_seeds(scenes, outdir, logger)
        # build_scene_gallery_items requires implementation
        from ui.gallery_utils import build_scene_gallery_items
        gallery_items, index_map, _ = build_scene_gallery_items(scenes, view, outdir)
        msg = f"Scene {shot_id} preview recomputed successfully."
        return scenes, gr.update(value=gallery_items), gr.update(value=index_map), msg
    except Exception as e:
        logger.error("Failed to recompute scene preview", exc_info=True)
        # We need build_scene_gallery_items. Ideally passed or imported.
        # It's circular dependency if in app_ui.
        # I'll create `ui/gallery_utils.py` next.
        from ui.gallery_utils import build_scene_gallery_items
        gallery_items, index_map, _ = build_scene_gallery_items(scenes, view, outdir)
        return scenes, gr.update(value=gallery_items), gr.update(value=index_map), f"[ERROR] Recompute failed: {str(e)}"
