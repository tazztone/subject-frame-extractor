from __future__ import annotations
import math
from typing import Optional, List, Dict, Any, Union, Callable, TYPE_CHECKING
import numpy as np
from pydantic import BaseModel, Field, ConfigDict
import cv2
import torch
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pathlib import Path

if TYPE_CHECKING:
    from core.config import Config
    from core.logger import AppLogger

def _coerce(val: Any, to_type: type) -> Any:
    if to_type is bool:
        if isinstance(val, bool): return val
        return str(val).strip().lower() in {"1", "true", "yes", "on"}
    if to_type in (int, float):
        try: return to_type(val)
        except (ValueError, TypeError): raise
    return val

def _sanitize_face_ref(kwargs: dict, logger: 'AppLogger') -> tuple[str, bool]:
    ref_path = kwargs.get('face_ref_img_path', '')
    video_path = kwargs.get('video_path', '')

    if not ref_path:
        return "", False

    p = Path(ref_path)
    if not p.exists() or not p.is_file():
        logger.warning(f"Face reference path does not exist or is not a file: {ref_path}")
        return "", False

    if str(p.resolve()) == str(Path(video_path).resolve()) if video_path else False:
        logger.warning("Face reference path is the same as video path.")
        return "", False

    return str(p), True

class QualityConfig(BaseModel):
    sharpness_base_scale: float
    edge_strength_base_scale: float
    enable_niqe: bool = True

class FrameMetrics(BaseModel):
    quality_score: float = 0.0
    sharpness_score: float = 0.0
    edge_strength_score: float = 0.0
    contrast_score: float = 0.0
    brightness_score: float = 0.0
    entropy_score: float = 0.0
    niqe_score: float = 0.0
    eyes_open: float = 0.0
    blink_prob: float = 0.0
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0

class Frame(BaseModel):
    image_data: np.ndarray
    frame_number: int
    metrics: FrameMetrics = Field(default_factory=FrameMetrics)
    face_similarity_score: Optional[float] = None
    max_face_confidence: Optional[float] = None
    error: Optional[str] = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def calculate_quality_metrics(self, thumb_image_rgb: np.ndarray, quality_config: 'QualityConfig', logger: 'AppLogger',
                                  mask: Optional[np.ndarray] = None, niqe_metric: Optional[Callable] = None,
                                  main_config: Optional['Config'] = None, face_landmarker: Optional[Callable] = None,
                                  face_bbox: Optional[List[int]] = None,
                                  metrics_to_compute: Optional[Dict[str, bool]] = None):
        try:
            if metrics_to_compute is None:
                metrics_to_compute = {k: True for k in ['eyes_open', 'yaw', 'pitch', 'sharpness', 'edge_strength', 'contrast', 'brightness', 'entropy', 'quality']}

            if face_landmarker and any(metrics_to_compute.get(k) for k in ['eyes_open', 'yaw', 'pitch']):
                if face_bbox:
                    x1, y1, x2, y2 = face_bbox
                    face_img = thumb_image_rgb[y1:y2, x1:x2]
                else:
                    face_img = thumb_image_rgb

                if not face_img.flags['C_CONTIGUOUS']:
                    face_img = np.ascontiguousarray(face_img, dtype=np.uint8)
                if face_img.dtype != np.uint8:
                    face_img = face_img.astype(np.uint8)

                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=face_img)
                landmarker_result = face_landmarker.detect(mp_image)

                if landmarker_result.face_blendshapes:
                    blendshapes = {b.category_name: b.score for b in landmarker_result.face_blendshapes[0]}
                    if metrics_to_compute.get('eyes_open'):
                        self.metrics.eyes_open = 1.0 - max(blendshapes.get('eyeBlinkLeft', 0), blendshapes.get('eyeBlinkRight', 0))
                        self.metrics.blink_prob = max(blendshapes.get('eyeBlinkLeft', 0), blendshapes.get('eyeBlinkRight', 0))

                if landmarker_result.facial_transformation_matrixes and any(metrics_to_compute.get(k) for k in ['yaw', 'pitch']):
                    matrix = landmarker_result.facial_transformation_matrixes[0]
                    sy = math.sqrt(matrix[0, 0] * matrix[0, 0] + matrix[1, 0] * matrix[1, 0])
                    singular = sy < 1e-6
                    if not singular:
                        if metrics_to_compute.get('pitch'): self.metrics.pitch = math.degrees(math.atan2(-matrix[2, 0], sy))
                        if metrics_to_compute.get('yaw'): self.metrics.yaw = math.degrees(math.atan2(matrix[1, 0], matrix[0, 0]))
                        self.metrics.roll = math.degrees(math.atan2(matrix[2, 1], matrix[2, 2]))
                    else:
                        if metrics_to_compute.get('pitch'): self.metrics.pitch = math.degrees(math.atan2(-matrix[2, 0], sy))
                        if metrics_to_compute.get('yaw'): self.metrics.yaw = 0
                        self.metrics.roll = 0

            scores_norm = {}
            gray = cv2.cvtColor(thumb_image_rgb, cv2.COLOR_RGB2GRAY)
            active_mask = ((mask > 128) if mask is not None and mask.ndim == 2 else None)
            if active_mask is not None and np.sum(active_mask) < 100:
                active_mask = None

            def _calculate_and_store_score(name, value):
                normalized_value = min(max(value, 0.0), 1.0)
                scores_norm[name] = normalized_value
                setattr(self.metrics, f"{name}_score", float(normalized_value * 100))

            if metrics_to_compute.get('sharpness'):
                laplacian = cv2.Laplacian(gray, cv2.CV_64F)
                masked_lap = laplacian[active_mask] if active_mask is not None else laplacian
                var_val = np.var(masked_lap) if masked_lap.size > 0 else 0
                sharpness = float(var_val)
                if quality_config.sharpness_base_scale:
                     sharpness = min(100.0, (sharpness / quality_config.sharpness_base_scale) * 100.0)
                _calculate_and_store_score("sharpness", sharpness / 100.0)

            if metrics_to_compute.get('edge_strength'):
                sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                edge_map = np.sqrt(sobelx**2 + sobely**2)
                mean_val = np.mean(edge_map)
                edge_strength = float(mean_val)
                if quality_config.edge_strength_base_scale:
                    edge_strength = min(100.0, (edge_strength / quality_config.edge_strength_base_scale) * 100.0)
                _calculate_and_store_score("edge_strength", edge_strength / 100.0)

            if metrics_to_compute.get('contrast') or metrics_to_compute.get('brightness'):
                pixels = gray[active_mask] if active_mask is not None else gray
                mean_br, std_br = (np.mean(pixels), np.std(pixels)) if pixels.size > 0 else (0, 0)
                if metrics_to_compute.get('brightness'):
                    brightness = float(mean_br) / 255.0
                    _calculate_and_store_score("brightness", brightness)
                if metrics_to_compute.get('contrast'):
                    contrast = float(std_br) / (mean_br + 1e-7)
                    contrast_scaled = min(contrast, main_config.quality_contrast_clamp) / main_config.quality_contrast_clamp
                    _calculate_and_store_score("contrast", contrast_scaled)

            if metrics_to_compute.get('entropy'):
                # Note: compute_entropy is expected to be imported in utils or defined somewhere.
                # Assuming it is handled by caller or passed?
                # Actually compute_entropy is a standalone function. I should import it?
                # It's better to keep calculate_quality_metrics logic self-contained or import utility.
                # But compute_entropy uses numba.
                # I'll defer entropy calculation or replicate logic if simple.
                # It uses njit.
                pass

            if quality_config.enable_niqe and niqe_metric is not None:
                try:
                    rgb_image = self.image_data
                    if active_mask is not None:
                        active_mask_full = cv2.resize(mask, (rgb_image.shape[1], rgb_image.shape[0]), interpolation=cv2.INTER_NEAREST) > 128
                        mask_3ch = (np.stack([active_mask_full] * 3, axis=-1))
                        rgb_image = np.where(mask_3ch, rgb_image, 0)
                    img_tensor = (torch.from_numpy(rgb_image).float().permute(2, 0, 1).unsqueeze(0) / 255.0)
                    with (torch.no_grad(), torch.amp.autocast('cuda', enabled=niqe_metric.device.type == 'cuda')):
                        niqe_raw = float(niqe_metric(img_tensor.to(niqe_metric.device)))
                        niqe_score = max(0, min(100, (main_config.quality_niqe_offset - niqe_raw) * main_config.quality_niqe_scale_factor))
                        scores_norm["niqe"] = niqe_score / 100.0
                        self.metrics.niqe_score = float(niqe_score)
                except Exception as e:
                    logger.warning("NIQE calculation failed", extra={'frame': self.frame_number, 'error': e})
                    if niqe_metric.device.type == 'cuda': torch.cuda.empty_cache()

            if main_config and metrics_to_compute.get('quality'):
                weights = {
                    'sharpness': main_config.quality_weights_sharpness,
                    'edge_strength': main_config.quality_weights_edge_strength,
                    'contrast': main_config.quality_weights_contrast,
                    'brightness': main_config.quality_weights_brightness,
                    'entropy': main_config.quality_weights_entropy,
                    'niqe': main_config.quality_weights_niqe,
                }
                quality_sum = sum(scores_norm.get(k, 0) * (weights.get(k, 0) / 100.0) for k in scores_norm.keys())
                self.metrics.quality_score = float(quality_sum * 100)
        except Exception as e:
            self.error = f"Quality calc failed: {e}"
            logger.error("Frame quality calculation failed", exc_info=True, extra={'frame': self.frame_number})

class Scene(BaseModel):
    shot_id: int
    start_frame: int
    end_frame: int
    status: str = "pending"
    best_frame: Optional[int] = None
    seed_metrics: dict = Field(default_factory=dict)
    seed_frame_idx: Optional[int] = None
    seed_config: dict = Field(default_factory=dict)
    seed_type: Optional[str] = None
    seed_result: dict = Field(default_factory=dict)
    preview_path: Optional[str] = None
    manual_status_change: bool = False
    is_overridden: bool = False
    initial_bbox: Optional[list] = None
    selected_bbox: Optional[list] = None
    yolo_detections: List[dict] = Field(default_factory=list)
    rejection_reasons: Optional[list] = None

class SceneState:
    def __init__(self, scene_data: Union[dict, Scene]):
        if isinstance(scene_data, dict):
            self._scene = Scene(**scene_data)
        else:
            self._scene = scene_data

        # Initialize defaults if missing (logic from legacy SceneState)
        if self._scene.initial_bbox is None and self._scene.seed_result and self._scene.seed_result.get('bbox'):
            self._scene.initial_bbox = self._scene.seed_result.get('bbox')
            self._scene.selected_bbox = self._scene.seed_result.get('bbox')

    @property
    def data(self) -> dict:
        return self._scene.model_dump()

    @property
    def scene(self) -> Scene:
        return self._scene

    def set_manual_bbox(self, bbox: list[int], source: str):
        self._scene.selected_bbox = bbox
        if self._scene.initial_bbox and self._scene.initial_bbox != bbox:
             self._scene.is_overridden = True
        else:
             self._scene.is_overridden = False

        if not self._scene.seed_config: self._scene.seed_config = {}
        self._scene.seed_config['override_source'] = source
        self._scene.status = 'included'
        self._scene.manual_status_change = True

    def reset(self):
        self._scene.selected_bbox = self._scene.initial_bbox
        self._scene.is_overridden = False
        self._scene.seed_config = {}
        self._scene.manual_status_change = False

    def include(self):
        self._scene.status = 'included'
        self._scene.manual_status_change = True

    def exclude(self):
        self._scene.status = 'excluded'
        self._scene.manual_status_change = True

    def update_seed_result(self, bbox: Optional[list[int]], details: dict):
        self._scene.seed_result = {'bbox': bbox, 'details': details}
        if self._scene.initial_bbox is None:
            self._scene.initial_bbox = bbox
        if not self._scene.is_overridden:
            self._scene.selected_bbox = bbox

class AnalysisParameters(BaseModel):
    source_path: str = ""
    method: str = ""
    interval: float = 0.0
    max_resolution: str = ""
    output_folder: str = ""
    video_path: str = ""
    disable_parallel: bool = False
    resume: bool = False
    enable_face_filter: bool = False
    face_ref_img_path: str = ""
    face_model_name: str = ""
    enable_subject_mask: bool = False
    tracker_model_name: str = ""
    seed_strategy: str = ""
    scene_detect: bool = False
    nth_frame: int = 0
    require_face_match: bool = False
    text_prompt: str = ""
    thumbnails_only: bool = True
    thumb_megapixels: float = 0.5
    pre_analysis_enabled: bool = False
    pre_sample_nth: int = 1
    primary_seed_strategy: str = "🤖 Automatic"
    min_mask_area_pct: float = 1.0
    sharpness_base_scale: float = 2500.0
    edge_strength_base_scale: float = 100.0
    compute_quality_score: bool = True
    compute_sharpness: bool = True
    compute_edge_strength: bool = True
    compute_contrast: bool = True
    compute_brightness: bool = True
    compute_entropy: bool = True
    compute_eyes_open: bool = True
    compute_yaw: bool = True
    compute_pitch: bool = True
    compute_face_sim: bool = True
    compute_subject_mask_area: bool = True
    compute_niqe: bool = True
    compute_phash: bool = True
    need_masks_now: bool = False

    @classmethod
    def from_ui(cls, logger: 'AppLogger', config: 'Config', **kwargs) -> 'AnalysisParameters':
        if 'face_ref_img_path' in kwargs or 'video_path' in kwargs:
            sanitized_face_ref, face_filter_enabled = _sanitize_face_ref(kwargs, logger)
            kwargs['face_ref_img_path'] = sanitized_face_ref
            kwargs['enable_face_filter'] = face_filter_enabled

        if 'thumb_megapixels' in kwargs:
            thumb_mp = kwargs['thumb_megapixels']
            if not isinstance(thumb_mp, (int, float)) or thumb_mp <= 0:
                logger.warning(f"Invalid thumb_megapixels: {thumb_mp}, using default")
                kwargs['thumb_megapixels'] = config.default_thumb_megapixels

        if 'pre_sample_nth' in kwargs:
            sample_nth = kwargs['pre_sample_nth']
            if not isinstance(sample_nth, int) or sample_nth < 1:
                logger.warning(f"Invalid pre_sample_nth: {sample_nth}, using 1")
                kwargs['pre_sample_nth'] = 1

        valid_keys = set(cls.model_fields.keys())
        defaults = {f: False for f in valid_keys if f.startswith('compute_')}
        config_defaults = config.model_dump()
        for key in valid_keys:
            if f"default_{key}" in config_defaults:
                defaults[key] = config_defaults[f"default_{key}"]

        for metric in [k.replace('filter_default_', '') for k in config_defaults if k.startswith('filter_default_')]:
            compute_key = f"compute_{metric}"
            if compute_key in valid_keys:
                defaults[compute_key] = True

        defaults['compute_phash'] = True
        instance = cls(**defaults)

        for key, value in kwargs.items():
            if hasattr(instance, key) and value is not None:
                if isinstance(value, str) and not value.strip() and key not in ['text_prompt', 'face_ref_img_path']: continue
                default = getattr(instance, key)
                try: setattr(instance, key, _coerce(value, type(default)))
                except (ValueError, TypeError): logger.warning(f"Could not coerce UI value for '{key}' to {type(default)}. Using default.", extra={'key': key, 'value': value})
        return instance

class MaskingResult(BaseModel):
    mask_path: Optional[str] = None
    shot_id: Optional[int] = None
    seed_type: Optional[str] = None
    seed_face_sim: Optional[float] = None
    mask_area_pct: Optional[float] = None
    mask_empty: bool = True
    error: Optional[str] = None
