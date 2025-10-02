from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class UIEvent:
    """Base class for UI events."""
    pass


@dataclass
class ExtractionEvent(UIEvent):
    """Event for triggering the extraction pipeline."""
    source_path: str
    upload_video: Optional[str]
    method: str
    interval: str
    nth_frame: str
    fast_scene: bool
    max_resolution: str
    use_png: bool
    thumbnails_only: bool
    thumb_megapixels: float
    scene_detect: bool


@dataclass
class PreAnalysisEvent(UIEvent):
    """Event for triggering the pre-analysis and seeding pipeline."""
    output_folder: str
    video_path: str
    resume: bool
    enable_face_filter: bool
    face_ref_img_path: str
    face_ref_img_upload: Optional[str]
    face_model_name: str
    enable_subject_mask: bool
    dam4sam_model_name: str
    person_detector_model: str
    seed_strategy: str
    scene_detect: bool
    enable_dedup: bool
    text_prompt: str
    box_threshold: float
    text_threshold: float
    min_mask_area_pct: float
    sharpness_base_scale: float
    edge_strength_base_scale: float
    gdino_config_path: str
    gdino_checkpoint_path: str
    pre_analysis_enabled: bool
    pre_sample_nth: int


@dataclass
class PropagationEvent(UIEvent):
    """Event for triggering the mask propagation pipeline."""
    output_folder: str
    video_path: str
    scenes: list[dict[str, Any]]
    analysis_params: PreAnalysisEvent


@dataclass
class FilterEvent(UIEvent):
    """Event for handling filter changes in the UI."""
    all_frames_data: list[dict[str, Any]]
    per_metric_values: dict[str, Any]
    output_dir: str
    gallery_view: str
    show_overlay: bool
    overlay_alpha: float
    require_face_match: bool
    dedup_thresh: int
    slider_values: dict[str, float]


@dataclass
class ExportEvent(UIEvent):
    """Event for exporting filtered frames."""
    all_frames_data: list[dict[str, Any]]
    output_dir: str
    video_path: str
    enable_crop: bool
    crop_ars: str
    crop_padding: int
    filter_args: dict[str, Any]