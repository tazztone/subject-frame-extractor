"""Centralized UI locators — single place to update when the UI changes."""


class Selectors:
    """elem_id-based CSS selectors (most resilient to label changes)."""

    # Global
    UNIFIED_LOG = "#unified_log"
    UNIFIED_STATUS = "#unified_status"
    LOG_TEXTAREA = "#unified_log textarea"

    # Extraction Tab (Source)
    SOURCE_INPUT = "#source_input"
    START_EXTRACTION = "#start_extraction_button"
    ADD_TO_QUEUE = "#add_to_queue_button"
    THUMB_MEGAPIXELS = "#thumb_megapixels_input"

    # Subject Tab
    SEED_STRATEGY = "#primary_seed_strategy_input"
    START_PRE_ANALYSIS = "#start_pre_analysis_button"
    SCAN_VIDEO_TAB = "#scan_video_tab"

    # Scenes Tab
    SCENE_GALLERY = "#scene_gallery"
    VIEW_TOGGLE = "#scene_gallery_view_toggle"
    MASK_AREA_MIN = "#scene_mask_area_min_input"
    QUALITY_SCORE_MIN = "#scene_quality_score_min_input"
    PREV_PAGE = "#prev_page_button"
    NEXT_PAGE = "#next_page_button"
    PAGINATION_ROW = "#pagination_row"

    # Metrics Tab
    START_ANALYSIS = "#start_analysis_button"

    # Export Tab
    FILTER_PRESET = "#filter_preset_dropdown"
    DEDUP_THRESH = "#dedup_thresh_input"
    EXPORT_BUTTON = "#export_button"


class Labels:
    """Text/label-based locators (used when elem_id isn't available)."""

    # Tab Names
    TAB_SOURCE = "Source"
    TAB_SUBJECT = "Subject"
    TAB_SCENES = "Scenes"
    TAB_METRICS = "Metrics"
    TAB_EXPORT = "Export"

    # Common Labels
    SOURCE_PLACEHOLDER = "Paste YouTube URL or local path"
    SYSTEM_LOGS = "System Logs"
    HELP_ACCORDION = "Help / Troubleshooting"
    SCAN_VIDEO_BUTTON = "Scan Video Now"

    # Strategy Options
    STRATEGY_FACE = "👤 By Face"
    STRATEGY_TEXT = "📝 By Text"
