"""
Event Models for Frame Extractor & Analyzer

Pydantic models representing UI events and data contracts.
"""
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, field_validator, model_validator


class UIEvent(BaseModel):
    model_config = ConfigDict(validate_assignment=True, extra='ignore', str_strip_whitespace=True, arbitrary_types_allowed=True)


class ExtractionEvent(UIEvent):
    source_path: str
    upload_video: Optional[str] = None
    method: str
    interval: str
    nth_frame: str
    max_resolution: str
    thumbnails_only: bool
    thumb_megapixels: float
    scene_detect: bool
    output_folder: Optional[str] = None


class PreAnalysisEvent(UIEvent):
    output_folder: str
    video_path: str
    resume: bool
    enable_face_filter: bool
    face_ref_img_path: str
    face_ref_img_upload: Optional[str]
    face_model_name: str
    enable_subject_mask: bool
    tracker_model_name: str
    best_frame_strategy: str
    scene_detect: bool
    text_prompt: str
    min_mask_area_pct: float
    sharpness_base_scale: float
    edge_strength_base_scale: float
    pre_analysis_enabled: bool
    pre_sample_nth: int
    primary_seed_strategy: str
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

    @field_validator('face_ref_img_path')
    @classmethod
    def validate_face_ref(cls, v: str, info) -> str:
        if not v: return ""
        video_path = info.data.get('video_path', '')
        if v == video_path: return ""
        p = Path(v)
        valid_exts = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}
        if not p.is_file() or p.suffix.lower() not in valid_exts: return ""
        return v

    @model_validator(mode='after')
    def validate_strategy_consistency(self) -> 'PreAnalysisEvent':
        if not self.face_ref_img_path and self.enable_face_filter:
            self.enable_face_filter = False
        return self


class PropagationEvent(UIEvent):
    output_folder: str
    video_path: str
    scenes: list[dict[str, Any]]
    analysis_params: PreAnalysisEvent


class FilterEvent(UIEvent):
    all_frames_data: list[dict[str, Any]]
    per_metric_values: dict[str, Any]
    output_dir: str
    gallery_view: str
    show_overlay: bool
    overlay_alpha: float
    require_face_match: bool
    dedup_thresh: int
    slider_values: dict[str, float]
    dedup_method: str


class ExportEvent(UIEvent):
    all_frames_data: list[dict[str, Any]]
    output_dir: str
    video_path: str
    enable_crop: bool
    crop_ars: str
    crop_padding: int
    filter_args: dict[str, Any]


class SessionLoadEvent(UIEvent):
    session_path: str
