from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional, Union

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator

if TYPE_CHECKING:
    from core.config import Config
    from core.logger import AppLogger

from core.enums import SceneStatus
from core.pipeline_results import PreAnalysisResult  # noqa: F401 # Backward compatibility shim


def _coerce(val: Any, to_type: Optional[type]) -> Any:
    """Helper to strictly coerce values to the target type."""
    if val is None or to_type is None:
        return val
    if to_type is bool:
        if isinstance(val, bool):
            return val
        return str(val).strip().lower() in {"1", "true", "yes", "on"}
    if to_type in (int, float):
        try:
            return to_type(val)
        except (ValueError, TypeError):
            raise
    return val


def _sanitize_face_ref(kwargs: dict, logger: "AppLogger") -> tuple[str, bool]:
    """Validates the face reference image path."""
    ref_path = kwargs.get("face_ref_img_path", "")
    video_path = kwargs.get("video_path", "")

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


class FrameMetrics(BaseModel):
    """Container for calculated quality scores for a frame."""

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
    """Represents a single video frame and its associated metadata."""

    image_data: np.ndarray
    frame_number: int
    metrics: FrameMetrics = Field(default_factory=FrameMetrics)
    face_similarity_score: Optional[float] = None
    max_face_confidence: Optional[float] = None
    error: Optional[str] = None
    model_config = ConfigDict(arbitrary_types_allowed=True)


class Scene(BaseModel):
    """Represents a detected scene or shot in the video."""

    @property
    def duration_frames(self) -> int:
        return self.end_frame - self.start_frame

    shot_id: int
    start_frame: int
    end_frame: int
    status: SceneStatus = SceneStatus.PENDING
    best_frame: Optional[int] = None
    seed_metrics: dict = Field(default_factory=dict)
    rejection_reasons: Optional[List[str]] = Field(default_factory=list)
    seed_frame_idx: Optional[int] = None
    seed_config: dict = Field(default_factory=dict)
    seed_type: Optional[str] = None
    seed_result: dict = Field(default_factory=dict)
    preview_path: Optional[str] = None
    manual_status_change: bool = False
    is_overridden: bool = False
    initial_bbox: Optional[list] = None
    selected_bbox: Optional[list] = None
    person_detections: List[dict] = Field(default_factory=list)
    additional_seeds: List[dict] = Field(default_factory=list)
    candidate_seed_frames: List[int] = Field(default_factory=list)


class SceneState:
    """Wrapper to manage state transitions and updates for a Scene object."""

    def __init__(self, scene_data: Union[dict, Scene]):
        if isinstance(scene_data, dict):
            self._scene = Scene(**scene_data)
        else:
            self._scene = scene_data

        # Initialize defaults if missing (logic from legacy SceneState)
        if self._scene.initial_bbox is None and self._scene.seed_result and self._scene.seed_result.get("bbox"):
            self._scene.initial_bbox = self._scene.seed_result.get("bbox")
            self._scene.selected_bbox = self._scene.seed_result.get("bbox")

    @property
    def data(self) -> dict:
        """Returns the scene data as a dictionary."""
        return self._scene.model_dump()

    @property
    def scene(self) -> Scene:
        """Returns the underlying Scene object."""
        return self._scene

    def set_manual_bbox(self, bbox: list[int], source: str):
        """Overrides the automatically selected subject bounding box."""
        self._scene.selected_bbox = bbox
        if self._scene.initial_bbox and self._scene.initial_bbox != bbox:
            self._scene.is_overridden = True
        else:
            self._scene.is_overridden = False

        if not self._scene.seed_config:
            self._scene.seed_config = {}
        self._scene.seed_config["override_source"] = source
        self._scene.status = SceneStatus.INCLUDED
        self._scene.manual_status_change = True

    def reset(self):
        """Resets the scene to its initial state (undoes manual overrides)."""
        self._scene.selected_bbox = self._scene.initial_bbox
        self._scene.is_overridden = False
        self._scene.seed_config = {}
        self._scene.manual_status_change = False

    def include(self):
        """Marks the scene as included."""
        self._scene.status = SceneStatus.INCLUDED
        self._scene.manual_status_change = True

    def exclude(self):
        """Marks the scene as excluded."""
        self._scene.status = SceneStatus.EXCLUDED
        self._scene.manual_status_change = True

    def update_seed_result(self, bbox: Optional[list[int]], details: dict):
        """Updates the seeding result (detected subject) for the scene."""
        self._scene.seed_result = {"bbox": bbox, "details": details}
        if self._scene.initial_bbox is None:
            self._scene.initial_bbox = bbox
        if not self._scene.is_overridden:
            self._scene.selected_bbox = bbox


class AnalysisParameters(BaseModel):
    """Aggregates all parameters for the analysis pipeline."""

    model_config = {"extra": "ignore"}

    @field_validator("primary_seed_strategy", mode="before")
    @classmethod
    def strip_emoji_from_strategy(cls, v: Any) -> str:
        """Strip emoji prefix from the strategy string if present."""
        if not isinstance(v, str):
            return str(v)
        import re

        return re.sub(r"^[^\w\s]+\s+", "", v)

    source_path: str = ""
    method: str = ""
    interval: float = 0.0
    max_resolution: str = ""
    output_folder: str = ""
    video_path: str = ""
    disable_parallel: bool = False
    resume: bool = False
    face_ref_img_path: str = ""
    face_model_name: str = ""
    enable_subject_mask: bool = False
    tracker_model_name: str = "sam2"
    seed_strategy: str = ""
    subject_detector_model: str = "YOLO26n"
    subject_detector_class_name: str = "person"
    subject_detector_class_id: int = 0
    subject_detector_threshold: float = 0.45
    scene_detect: bool = False
    nth_frame: int = 0
    require_face_match: bool = False
    text_prompt: str = ""
    thumbnails_only: bool = True
    thumb_megapixels: float = 0.5
    pre_analysis_enabled: bool = False
    pre_sample_nth: int = 1
    primary_seed_strategy: str = "Automatic Detection"
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
    def from_ui(cls, logger: "AppLogger", config: "Config", **kwargs) -> "AnalysisParameters":
        """Factory method to create parameters from UI arguments, handling validation and defaults."""
        if "face_ref_img_path" in kwargs or "video_path" in kwargs:
            sanitized_face_ref, face_filter_enabled = _sanitize_face_ref(kwargs, logger)
            kwargs["face_ref_img_path"] = sanitized_face_ref
            # Unify enable_face_filter with compute_face_sim
            if face_filter_enabled:
                kwargs["compute_face_sim"] = True

        if "thumb_megapixels" in kwargs:
            try:
                thumb_mp = _coerce(kwargs["thumb_megapixels"], float)
                if thumb_mp <= 0:
                    raise ValueError
                kwargs["thumb_megapixels"] = thumb_mp
            except (ValueError, TypeError):
                logger.warning(f"Invalid thumb_megapixels: {kwargs['thumb_megapixels']}, using default")
                kwargs["thumb_megapixels"] = config.default_thumb_megapixels

        if "pre_sample_nth" in kwargs:
            try:
                sample_nth = _coerce(kwargs["pre_sample_nth"], int)
                if sample_nth < 1:
                    raise ValueError
                kwargs["pre_sample_nth"] = sample_nth
            except (ValueError, TypeError):
                logger.warning(f"Invalid pre_sample_nth: {kwargs['pre_sample_nth']}, using 1")
                kwargs["pre_sample_nth"] = 1

        valid_keys = set(cls.model_fields.keys())
        # Initialize defaults for compute metrics
        defaults: dict[str, Any] = {f: False for f in valid_keys if f.startswith("compute_")}

        config_defaults = config.model_dump()
        for key in valid_keys:
            if f"default_{key}" in config_defaults:
                val = config_defaults[f"default_{key}"]
                target_type = cls.model_fields[key].annotation
                if target_type is None:
                    continue
                try:
                    # Pick the first non-None type for Optional fields
                    import types

                    is_union = (hasattr(target_type, "__origin__") and target_type.__origin__ is Union) or isinstance(
                        target_type, types.UnionType
                    )

                    if is_union:
                        if hasattr(target_type, "__args__"):
                            args = target_type.__args__
                            target_type = next((t for t in args if t is not type(None)), target_type)

                    if target_type is not None:
                        defaults[key] = _coerce(val, target_type)
                except (ValueError, TypeError):
                    pass

        for metric in [k.replace("filter_default_", "") for k in config_defaults if k.startswith("filter_default_")]:
            compute_key = f"compute_{metric}"
            if compute_key in valid_keys:
                defaults[compute_key] = True

        defaults["compute_phash"] = True
        instance = cls(**defaults)

        for key, value in kwargs.items():
            if key in valid_keys and value is not None:
                if isinstance(value, str) and not value.strip() and key not in ["text_prompt", "face_ref_img_path"]:
                    continue

                target_type = cls.model_fields[key].annotation
                if target_type is None:
                    continue
                try:
                    # Handle Optional types (extract the inner type)
                    import types

                    is_union = (hasattr(target_type, "__origin__") and target_type.__origin__ is Union) or isinstance(
                        target_type, types.UnionType
                    )
                    if is_union:
                        if hasattr(target_type, "__args__"):
                            args = target_type.__args__
                            target_type = next((t for t in args if t is not type(None)), target_type)

                    if target_type is not None:
                        setattr(instance, key, _coerce(value, target_type))

                except (ValueError, TypeError):
                    logger.warning(
                        f"Could not coerce UI value for '{key}' to {target_type}. Using default.",
                        extra={"key": key, "value": value},
                    )

        # Post-process: Resolve COCO Class ID
        from core.enums import get_coco_id

        instance.subject_detector_class_id = get_coco_id(instance.subject_detector_class_name)

        return instance


class MaskingResult(BaseModel):
    """Result of the mask propagation process for a frame."""

    mask_path: Optional[str] = None
    shot_id: Optional[int] = None
    seed_type: Optional[str] = None
    seed_face_sim: Optional[float] = None
    mask_area_pct: Optional[float] = None
    mask_empty: bool = True
    error: Optional[str] = None
