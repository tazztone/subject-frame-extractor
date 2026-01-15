"""
Configuration Management for Frame Extractor & Analyzer
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# TODO: Support multiple config file locations (env var, user home, project root)
# TODO: Add config schema version for migration support
# TODO: Consider adding config file hot-reloading for development
def json_config_settings_source() -> Dict[str, Any]:
    """Loads settings from a JSON file for Pydantic settings."""
    try:
        config_path = "config.json"
        if Path(config_path).is_file():
            with open(config_path, "r", encoding="utf-8") as f:
                return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        # TODO: Log config loading errors instead of silently ignoring
        pass
    return {}


class Config(BaseSettings):
    """Manages the application's configuration settings."""

    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="APP_", env_nested_delimiter="_", case_sensitive=False
    )

    # Paths
    logs_dir: str = "logs"
    models_dir: str = "models"
    downloads_dir: str = "downloads"

    # Models
    user_agent: str = "Mozilla/5.0"
    huggingface_token: Optional[str] = None
    face_landmarker_url: str = (
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    )
    face_landmarker_sha256: str = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"
    # TODO: Update SAM3 checkpoint URL when official version is released
    sam3_checkpoint_url: str = "https://huggingface.co/facebook/sam3/resolve/main/sam3.pt"
    # TODO: Verify and update SHA256 hash once official SAM3 model is available
    sam3_checkpoint_sha256: str = "9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e"  # Placeholder, update if known or remove check

    # YouTube-DL
    ytdl_output_template: str = "%(id)s_%(title).40s_%(height)sp.%(ext)s"
    ytdl_format_string: str = (
        "bestvideo[height<={max_res}][ext=mp4]+bestaudio[ext=m4a]/best[height<={max_res}][ext=mp4]/best"
    )

    # FFmpeg
    ffmpeg_log_level: str = "info"
    ffmpeg_thumbnail_quality: int = 80
    ffmpeg_scene_threshold: float = 0.4

    # Cache
    cache_size: int = 200
    cache_eviction_factor: float = 0.2
    cache_cleanup_threshold: float = 0.8

    # Retry
    retry_max_attempts: int = 3
    retry_backoff_seconds: List[float] = Field(default_factory=lambda: [1, 5, 15])

    # Quality Scaling
    quality_entropy_normalization: float = 8.0
    quality_resolution_denominator: int = 500000
    quality_contrast_clamp: float = 2.0
    quality_niqe_offset: float = 10.0
    quality_niqe_scale_factor: float = 10.0

    # Masking
    masking_keep_largest_only: bool = True
    masking_close_kernel_size: int = 5
    masking_open_kernel_size: int = 5

    # UI Defaults
    default_thumbnails_only: bool = True
    default_thumb_megapixels: float = 0.5
    default_scene_detect: bool = True
    default_max_resolution: str = "maximum available"
    default_pre_analysis_enabled: bool = True
    default_pre_sample_nth: int = 5
    default_enable_face_filter: bool = True
    default_face_model_name: str = "buffalo_l"
    default_enable_subject_mask: bool = True
    default_tracker_model_name: str = "sam3"
    default_primary_seed_strategy: str = "🧑‍🤝‍🧑 Find Prominent Person"
    default_seed_strategy: str = "Largest Person"
    default_text_prompt: str = "a person"
    default_resume: bool = False
    default_require_face_match: bool = False
    default_enable_dedup: bool = True
    default_dedup_thresh: int = 5
    default_method: str = "all"
    default_interval: float = 5.0
    default_nth_frame: int = 5
    default_disable_parallel: bool = False

    # Filter Defaults
    filter_default_quality_score: Dict[str, float] = Field(
        default_factory=lambda: {"min": 0.0, "max": 100.0, "step": 0.5, "default_min": 0.0, "default_max": 100.0}
    )
    filter_default_sharpness: Dict[str, float] = Field(
        default_factory=lambda: {"min": 0.0, "max": 100.0, "step": 0.5, "default_min": 0.0, "default_max": 100.0}
    )
    filter_default_edge_strength: Dict[str, float] = Field(
        default_factory=lambda: {"min": 0.0, "max": 100.0, "step": 0.5, "default_min": 0.0, "default_max": 100.0}
    )
    filter_default_contrast: Dict[str, float] = Field(
        default_factory=lambda: {"min": 0.0, "max": 100.0, "step": 0.5, "default_min": 0.0, "default_max": 100.0}
    )
    filter_default_brightness: Dict[str, float] = Field(
        default_factory=lambda: {"min": 0.0, "max": 100.0, "step": 0.5, "default_min": 0.0, "default_max": 100.0}
    )
    filter_default_entropy: Dict[str, float] = Field(
        default_factory=lambda: {"min": 0.0, "max": 100.0, "step": 0.5, "default_min": 0.0, "default_max": 100.0}
    )

    # Missing Defaults
    default_min_mask_area_pct: float = 1.0
    default_sharpness_base_scale: float = 2500.0
    default_edge_strength_base_scale: float = 100.0
    filter_default_niqe: Dict[str, float] = Field(
        default_factory=lambda: {"min": 0.0, "max": 100.0, "step": 0.5, "default_min": 0.0, "default_max": 100.0}
    )
    filter_default_face_sim: Dict[str, float] = Field(
        default_factory=lambda: {"min": 0.0, "max": 1.0, "step": 0.01, "default_min": 0.0}
    )
    filter_default_mask_area_pct: Dict[str, float] = Field(
        default_factory=lambda: {"min": 0.0, "max": 100.0, "step": 0.1, "default_min": 1.0}
    )
    filter_default_dedup_thresh: Dict[str, int] = Field(
        default_factory=lambda: {"min": -1, "max": 32, "step": 1, "default": -1}
    )
    filter_default_eyes_open: Dict[str, float] = Field(
        default_factory=lambda: {"min": 0.0, "max": 1.0, "step": 0.01, "default_min": 0.0}
    )
    filter_default_yaw: Dict[str, float] = Field(
        default_factory=lambda: {"min": -180.0, "max": 180.0, "step": 1, "default_min": -25, "default_max": 25}
    )
    filter_default_pitch: Dict[str, float] = Field(
        default_factory=lambda: {"min": -180.0, "max": 180.0, "step": 1, "default_min": -25, "default_max": 25}
    )

    # Quality Weights
    quality_weights_sharpness: int = 25
    quality_weights_edge_strength: int = 15
    quality_weights_contrast: int = 15
    quality_weights_brightness: int = 10
    quality_weights_entropy: int = 15
    quality_weights_niqe: int = 20

    # Logging Config
    log_level: str = "INFO"
    log_format: str = "%(asctime)s | %(levelname)8s | %(name)s | %(message)s"
    log_colored: bool = True
    log_structured_path: str = "structured_log.jsonl"

    # Monitoring
    # TODO: Implement actual memory monitoring using these thresholds
    # TODO: Add automatic memory cleanup when approaching critical threshold
    monitoring_memory_warning_threshold_mb: int = 8192
    monitoring_memory_critical_threshold_mb: int = 16384
    monitoring_cpu_warning_threshold_percent: float = 90.0
    monitoring_gpu_memory_warning_threshold_percent: int = 90
    monitoring_memory_limit_mb: int = 8192

    # Export Options
    export_enable_crop: bool = True
    export_crop_padding: int = 1
    export_crop_ars: str = "16:9,1:1,9:16"

    # Gradio Defaults
    gradio_auto_pctl_input: int = 25
    gradio_show_mask_overlay: bool = True
    gradio_overlay_alpha: float = 0.6

    # Seeding Defaults
    seeding_face_similarity_threshold: float = 0.4
    seeding_face_contain_score: int = 100
    seeding_confidence_score_multiplier: int = 20
    seeding_iou_threshold: float = 0.5  # IOU threshold for person/text box intersection
    seeding_iou_bonus: float = 50.0  # Bonus score for high IOU matches
    seeding_face_to_body_expansion_factors: List[float] = Field(default_factory=lambda: [4.0, 7.0, 0.75])
    seeding_final_fallback_box: List[float] = Field(default_factory=lambda: [0.25, 0.25, 0.5, 0.5])
    seeding_balanced_score_weights: Dict[str, float] = Field(
        default_factory=lambda: {"area": 0.4, "confidence": 0.4, "edge": 0.2}
    )

    # Utility Defaults
    utility_max_filename_length: int = 50
    utility_video_extensions: List[str] = Field(default_factory=lambda: [".mp4", ".mov", ".mkv", ".avi", ".webm"])
    utility_image_extensions: List[str] = Field(default_factory=lambda: [".png", ".jpg", ".jpeg", ".webp", ".bmp"])

    # PostProcessing
    postprocessing_mask_fill_kernel_size: int = 5

    # Visualization
    visualization_bbox_color: List[int] = Field(default_factory=lambda: [255, 0, 0])
    visualization_bbox_thickness: int = 2

    # Analysis
    analysis_default_batch_size: int = 25
    analysis_default_workers: int = 4

    # Validation
    validation_min_duration_secs: int = 1
    validation_min_frame_count: int = 10

    # Model Defaults
    model_face_analyzer_det_size: List[int] = Field(default_factory=lambda: [640, 640])

    sharpness_base_scale: int = 2500
    edge_strength_base_scale: int = 100

    def model_post_init(self, __context: Any) -> None:
        """Post-initialization hook to validate paths."""
        self._validate_paths()

    def _validate_paths(self):
        """Ensures critical directories exist and are writable."""
        for p in [self.logs_dir, self.models_dir, self.downloads_dir]:
            path = Path(p)
            path.mkdir(parents=True, exist_ok=True)
            # Test writability using pathlib (per project guidelines: never use os.path)
            try:
                test_file = path / ".write_test"
                test_file.touch()
                test_file.unlink()
            except (PermissionError, OSError):
                print(f"WARNING: Directory {p} is not writable.")

    @model_validator(mode="after")
    def _validate_config(self) -> "Config":
        """Validates that at least one quality weight is non-zero."""
        if (
            sum(
                [
                    self.quality_weights_sharpness,
                    self.quality_weights_edge_strength,
                    self.quality_weights_contrast,
                    self.quality_weights_brightness,
                    self.quality_weights_entropy,
                    self.quality_weights_niqe,
                ]
            )
            == 0
        ):
            raise ValueError("The sum of quality_weights cannot be zero.")
        return self

    @property
    def quality_weights(self) -> Dict[str, int]:
        """Returns a dictionary of quality metric weights."""
        return {
            "sharpness": self.quality_weights_sharpness,
            "edge_strength": self.quality_weights_edge_strength,
            "contrast": self.quality_weights_contrast,
            "brightness": self.quality_weights_brightness,
            "entropy": self.quality_weights_entropy,
            "niqe": self.quality_weights_niqe,
        }
