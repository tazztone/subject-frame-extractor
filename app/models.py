"""Domain models and data classes for the frame extractor application."""

import json
import hashlib
from dataclasses import dataclass, asdict, field, fields
from pathlib import Path
import numpy as np
import cv2
import torch


@dataclass
class FrameMetrics:
    quality_score: float = 0.0
    sharpness_score: float = 0.0
    edge_strength_score: float = 0.0
    contrast_score: float = 0.0
    brightness_score: float = 0.0
    entropy_score: float = 0.0
    niqe_score: float = 0.0


@dataclass
class Frame:
    image_data: np.ndarray
    frame_number: int
    metrics: FrameMetrics = field(default_factory=FrameMetrics)
    face_similarity_score: float | None = None
    max_face_confidence: float | None = None
    error: str | None = None

    def calculate_quality_metrics(self, thumb_image_rgb: np.ndarray,
                                  config: 'Config',
                                  logger: 'UnifiedLogger',
                                  mask: np.ndarray | None = None,
                                  niqe_metric=None):
        """Calculate quality metrics for this frame."""
        from numba import njit

        @njit
        def compute_entropy(hist):
            prob = hist / (np.sum(hist) + 1e-7)
            entropy = -np.sum(prob[prob > 0] * np.log2(prob[prob > 0]))
            return min(max(entropy / 8.0, 0), 1.0)

        try:
            gray = cv2.cvtColor(thumb_image_rgb, cv2.COLOR_RGB2GRAY)
            active_mask = ((mask > 128) if mask is not None and
                           mask.ndim == 2 else None)
            if active_mask is not None and np.sum(active_mask) < 100:
                raise ValueError("Mask too small.")

            lap = cv2.Laplacian(gray, cv2.CV_64F)
            masked_lap = lap[active_mask] if active_mask is not None else lap
            sharpness = np.var(masked_lap) if masked_lap.size > 0 else 0
            sharpness_scaled = (sharpness /
                                (config.sharpness_base_scale *
                                 (gray.size / 500_000)))

            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            edge_strength = np.mean(np.sqrt(sobelx**2 + sobely**2))
            edge_strength_scaled = (edge_strength /
                                    (config.edge_strength_base_scale *
                                     (gray.size / 500_000)))

            pixels = gray[active_mask] if active_mask is not None else gray
            if pixels.size > 0:
                mean_br, std_br = np.mean(pixels), np.std(pixels)
            else:
                mean_br, std_br = 0, 0
            brightness = mean_br / 255.0
            contrast = std_br / (mean_br + 1e-7)

            gray_full = cv2.cvtColor(self.image_data, cv2.COLOR_RGB2GRAY)
            if mask is not None:
                mask_full = cv2.resize(
                    mask, (gray_full.shape[1], gray_full.shape[0]),
                    interpolation=cv2.INTER_NEAREST
                )
                active_mask_full = (mask_full > 128).astype(np.uint8)
            else:
                mask_full = None
                active_mask_full = None

            hist = cv2.calcHist([gray_full], [0], active_mask_full, [256],
                                [0, 256]).flatten()
            entropy = compute_entropy(hist)

            niqe_score = 0.0
            if niqe_metric is not None:
                try:
                    rgb_image = self.image_data
                    if active_mask_full is not None:
                        mask_3ch = (np.stack([active_mask_full] * 3, axis=-1)
                                    > 0)
                        rgb_image = np.where(mask_3ch, rgb_image, 0)
                    img_tensor = (torch.from_numpy(rgb_image).float()
                                  .permute(2, 0, 1).unsqueeze(0) / 255.0)
                    with (torch.no_grad(),
                          torch.cuda.amp.autocast(
                              enabled=torch.cuda.is_available())):
                        niqe_raw = float(niqe_metric(
                            img_tensor.to(niqe_metric.device)
                        ))
                        niqe_score = max(0, min(100, (10 - niqe_raw) * 10))
                except Exception as e:
                    logger.warning("NIQE calculation failed",
                                   extra={'frame': self.frame_number,
                                          'error': e})

            scores_norm = {
                "sharpness": min(sharpness_scaled, 1.0),
                "edge_strength": min(edge_strength_scaled, 1.0),
                "contrast": min(contrast, 2.0) / 2.0,
                "brightness": brightness,
                "entropy": entropy,
                "niqe": niqe_score / 100.0
            }

            self.metrics = FrameMetrics(**{
                f"{k}_score": float(v * 100) for k, v in scores_norm.items()
            })

            quality_sum = sum(
                scores_norm[k] * (config.quality_weights[k] / 100.0)
                for k in config.QUALITY_METRICS
            )
            self.metrics.quality_score = float(quality_sum * 100)

        except Exception as e:
            self.error = f"Quality calc failed: {e}"
            logger.error("Frame quality calculation failed", exc_info=True,
                         extra={'frame': self.frame_number})


@dataclass
class Scene:
    shot_id: int
    start_frame: int
    end_frame: int
    status: str = "pending"  # pending, included, excluded
    best_seed_frame: int | None = None
    seed_metrics: dict = field(default_factory=dict)
    seed_config: dict = field(default_factory=dict)  # User overrides
    seed_result: dict = field(default_factory=dict)  # Result of seeding
    preview_path: str | None = None  # Path to preview image for UI gallery
    manual_status_change: bool = False


@dataclass
class AnalysisParameters:
    source_path: str = ""
    method: str = ""
    interval: float = 0.0
    max_resolution: str = ""
    fast_scene: bool = False
    use_png: bool = True
    output_folder: str = ""
    video_path: str = ""
    disable_parallel: bool = False
    resume: bool = False
    enable_face_filter: bool = False
    face_ref_img_path: str = ""
    face_model_name: str = ""
    enable_subject_mask: bool = False
    dam4sam_model_name: str = ""
    person_detector_model: str = ""
    seed_strategy: str = ""
    scene_detect: bool = False
    nth_frame: int = 0
    require_face_match: bool = False
    enable_dedup: bool = False
    dedup_thresh: int = 0
    text_prompt: str = ""
    thumbnails_only: bool = True
    thumb_megapixels: float = 0.5
    pre_analysis_enabled: bool = False
    pre_sample_nth: int = 1
    
    # These will be set from config in __post_init__
    gdino_config_path: str = ""
    gdino_checkpoint_path: str = ""
    box_threshold: float = 0.35
    text_threshold: float = 0.25
    min_mask_area_pct: float = 1.0
    sharpness_base_scale: float = 2500.0
    edge_strength_base_scale: float = 100.0

    def __post_init__(self):
        """Set config-dependent defaults after initialization."""
        # This method is called after __init__, so we can't inject config directly.
        # It's used to set defaults from a global config if they aren't provided.
        # This is a slight break in the DI pattern but is localized here.
        # The from_ui factory method *does* use proper DI.
        from app.config import Config
        config = Config()

        if not self.gdino_config_path:
            self.gdino_config_path = str(config.GROUNDING_DINO_CONFIG)
        if not self.gdino_checkpoint_path:
            self.gdino_checkpoint_path = str(config.GROUNDING_DINO_CKPT)
        if self.box_threshold == 0.35:  # Default value
            self.box_threshold = config.GROUNDING_BOX_THRESHOLD
        if self.text_threshold == 0.25:  # Default value
            self.text_threshold = config.GROUNDING_TEXT_THRESHOLD
        if self.min_mask_area_pct == 1.0:  # Default value
            self.min_mask_area_pct = config.min_mask_area_pct
        if self.sharpness_base_scale == 2500.0:  # Default value
            self.sharpness_base_scale = config.sharpness_base_scale
        if self.edge_strength_base_scale == 100.0:  # Default value
            self.edge_strength_base_scale = config.edge_strength_base_scale

    @classmethod
    def from_ui(cls, logger: 'EnhancedLogger', config: 'Config', **kwargs):
        """Create instance from UI parameters."""
        valid_keys = {f.name for f in fields(cls)}
        filtered_defaults = {
            k: v for k, v in config.ui_defaults.items() if k in valid_keys
        }
        instance = cls(**filtered_defaults)

        for key, value in kwargs.items():
            if hasattr(instance, key):
                target_type = type(getattr(instance, key))
                try:
                    if value is not None and value != '':
                        setattr(instance, key, target_type(value))
                except (ValueError, TypeError):
                    logger.warning(
                        f"Could not coerce UI value for '{key}' to "
                        f"{target_type}. Using default.",
                        extra={'key': key, 'value': value}
                    )
        return instance

    def _get_config_hash(self, output_dir: Path) -> str:
        """Create hash of parameters and scene seeds for resume logic."""
        from app.utils import _to_json_safe

        data_to_hash = json.dumps(_to_json_safe(asdict(self)), sort_keys=True)
        scene_seeds_path = output_dir / "scene_seeds.json"
        if scene_seeds_path.exists():
            data_to_hash += scene_seeds_path.read_text(encoding='utf-8')
        return hashlib.sha256(data_to_hash.encode()).hexdigest()


@dataclass
class MaskingResult:
    mask_path: str | None = None
    shot_id: int | None = None
    seed_type: str | None = None
    seed_face_sim: float | None = None
    mask_area_pct: float | None = None
    mask_empty: bool = True
    error: str | None = None
