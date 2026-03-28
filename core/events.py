"""
Event Models for Frame Extractor & Analyzer

Pydantic models representing UI events and data contracts.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class UIEvent(BaseModel):
    """Base class for all UI-triggered events."""

    model_config = ConfigDict(
        validate_assignment=True, extra="ignore", str_strip_whitespace=True, arbitrary_types_allowed=True
    )


def validate_writable_directory(v: str, field_name: str) -> str:
    """Helper to validate that a directory path is writable."""
    if not v:
        return v
    path = Path(v)
    # If it doesn't exist, we'll try to create it, so check parent
    target = path if path.exists() else path.parent
    if target.exists() and not os.access(target, os.W_OK):
        raise ValueError(f"{field_name} '{v}' is not writable.")
    return v


class ExtractionEvent(UIEvent):
    """
    Data model for frame extraction events.
    """

    source_path: str = ""
    upload_video: Optional[str] = None
    method: str
    interval: float = Field(default=5.0, gt=0)
    nth_frame: int = Field(default=5, gt=0)
    max_resolution: str
    thumbnails_only: bool = True
    thumb_megapixels: float = Field(default=0.5, gt=0, le=10.0)
    scene_detect: bool
    output_folder: Optional[str] = None

    @field_validator("output_folder")
    @classmethod
    def validate_output_folder(cls, v: Optional[str]) -> Optional[str]:
        if v:
            return validate_writable_directory(v, "Output Folder")
        return v

    @model_validator(mode="after")
    def validate_source(self) -> "ExtractionEvent":
        if not self.source_path and not self.upload_video:
            raise ValueError("Please provide a Source Path or Upload a Video.")
        return self


class PreAnalysisEvent(UIEvent):
    """
    Data model for pre-analysis configuration and execution.
    """

    output_folder: str
    video_path: str
    resume: bool = False
    enable_face_filter: bool = False
    face_ref_img_path: str = ""
    face_ref_img_upload: Optional[str] = None
    face_model_name: str = "buffalo_l"
    enable_subject_mask: bool = False
    tracker_model_name: str = "sam2"
    best_frame_strategy: str = "sharpness"
    scene_detect: bool = True
    text_prompt: str = ""
    min_mask_area_pct: float = Field(default=1.0, gt=0, le=100)
    sharpness_base_scale: float = Field(default=2500.0, gt=0)
    edge_strength_base_scale: float = Field(default=100.0, gt=0)
    pre_analysis_enabled: bool = True
    pre_sample_nth: int = Field(default=1, gt=0)
    primary_seed_strategy: str = "🤖 Automatic"
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

    @field_validator("output_folder")
    @classmethod
    def validate_out(cls, v: str) -> str:
        return validate_writable_directory(v, "Output Folder")

    @field_validator("face_ref_img_path")
    @classmethod
    def validate_face_ref(cls, v: str, info) -> str:
        """Validates that the reference image path is a valid image file."""
        if not v:
            return ""
        video_path = info.data.get("video_path", "")
        if v == video_path:
            return ""
        p = Path(v)
        valid_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
        if not p.is_file() or p.suffix.lower() not in valid_exts:
            return ""
        return v

    @model_validator(mode="after")
    def validate_strategy_consistency(self) -> "PreAnalysisEvent":
        """Ensures that dependent settings (like face filter) are consistent with available data."""
        if not self.face_ref_img_path and self.enable_face_filter:
            self.enable_face_filter = False
        return self


# TODO: Add scene status validation (enum instead of string)
class PropagationEvent(UIEvent):
    """
    Data model for the mask propagation stage.
    """

    output_folder: str
    video_path: str
    scenes: List[Dict[str, Any]]
    analysis_params: PreAnalysisEvent

    @field_validator("output_folder")
    @classmethod
    def validate_out(cls, v: str) -> str:
        return validate_writable_directory(v, "Output Folder")


class FilterEvent(UIEvent):
    """
    Data model for filtering and gallery update events.
    """

    all_frames_data: List[Dict[str, Any]]
    per_metric_values: Dict[str, Any]
    output_dir: str
    gallery_view: str
    show_overlay: bool
    overlay_alpha: float = Field(ge=0, le=1.0)
    require_face_match: bool
    dedup_thresh: int = Field(ge=-1, le=100)
    slider_values: Dict[str, float]
    dedup_method: str

    @field_validator("output_dir")
    @classmethod
    def validate_out(cls, v: str) -> str:
        return validate_writable_directory(v, "Output Directory")


class ExportEvent(UIEvent):
    """
    Data model for exporting filtered frames.
    """

    all_frames_data: List[Dict[str, Any]]
    output_dir: str
    video_path: str
    enable_crop: bool
    crop_ars: str
    crop_padding: int = Field(ge=0)
    enable_xmp_export: bool = False
    filter_args: Dict[str, Any]

    @field_validator("output_dir")
    @classmethod
    def validate_out(cls, v: str) -> str:
        return validate_writable_directory(v, "Output Directory")


class SessionLoadEvent(UIEvent):
    """
    Data model for loading a previous session.
    """

    session_path: str

    @field_validator("session_path")
    @classmethod
    def validate_session_path(cls, v: str) -> str:
        if not v:
            return v
        p = Path(v)
        if not p.exists():
            raise ValueError(f"Session path '{v}' does not exist.")
        if not p.is_dir():
            raise ValueError(f"Session path '{v}' is not a directory.")
        return v
