import re


class Selectors:
    """elem_id-based CSS selectors (most resilient to label changes)."""

    # Global Components
    UNIFIED_LOG = "#unified_log"
    UNIFIED_STATUS = "#unified_status"
    LOG_TEXTAREA = "#unified_log textarea"
    CANCEL_BUTTON = "button:has-text('Cancel')"
    PAUSE_BUTTON = "button:has-text('Pause')"
    RESET_STATE_BUTTON = "button:has-text('Reset State (MOCKED)')"
    REFRESH_LOGS = "#refresh_logs_button"

    # Status Message Constants (regex for robustness)
    STATUS_READY = "System Reset Ready."
    STATUS_MSG = "#unified_status"
    STATUS_ERROR = "#unified_status"
    STATUS_ERROR_REGEX = re.compile(r"⚠️|Error|Fail|Invalid", re.IGNORECASE)
    STATUS_SUCCESS_EXTRACTION = "Extraction Complete"
    STATUS_SUCCESS_PRE_ANALYSIS = "Pre-Analysis Complete"
    STATUS_SUCCESS_PROPAGATION = "Mask Propagation Complete"
    STATUS_SUCCESS_ANALYSIS = "Analysis Complete"
    STATUS_SUCCESS_EXPORT = "Export Complete"

    # Extraction Tab (Source)
    SOURCE_INPUT = "#source_input textarea, #source_input input"
    START_EXTRACTION = "#start_extraction_button"
    ADD_TO_QUEUE = "#add_to_queue_button"
    THUMB_MEGAPIXELS = "#thumb_megapixels_input"
    SESSION_INPUT = "#session_path_input textarea, #session_path_input input"
    LOAD_SESSION_BUTTON = "#load_session_button"
    MAX_RESOLUTION = "#max_resolution"
    METHOD_INPUT = "#method_input"

    # Subject/Scenes Tab
    SEED_STRATEGY = "#primary_seed_strategy_input"
    BEST_FRAME_STRATEGY = "#best_frame_strategy_input"
    START_PRE_ANALYSIS = "#start_pre_analysis_button"
    SCENE_GALLERY = "#scene_gallery"
    SCENE_GALLERY_VIEW_TOGGLE = "#scene_gallery_view_toggle"
    MASK_AREA_MIN = "input[aria-label='Min Subject Area %']"
    QUALITY_SCORE_MIN = "#slider_quality_score_min"

    # Scene Tab Specific (using elem_id)
    SCENE_MASK_AREA_MIN = "#slider_scene_mask_area_min"
    SCENE_FACE_SIM_MIN = "#slider_scene_face_sim_min"
    SCENE_QUALITY_SCORE_MIN = "#slider_scene_quality_score_min"
    PREV_PAGE_BUTTON = "#prev_page_button"
    NEXT_PAGE_BUTTON = "#next_page_button"

    PROPAGATE_MASKS = "#propagate_masks_button"
    SCENE_FILTER_STATUS = "#scene_filter_status"

    # Metrics & Export Tab
    START_ANALYSIS = "#start_analysis_button"
    FILTER_PRESET = "#filter_preset_dropdown"
    DEDUP_THRESH = "#dedup_thresh_input"
    EXPORT_BUTTON = "#export_button"
    DRY_RUN_BUTTON = "#dry_run_button"


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
    SYSTEM_LOGS = "📋 System Logs"
    HELP_ACCORDION = "❓ Help / Troubleshooting"
    HELP_ACCORDION_BTN = "button:has-text('❓ Help / Troubleshooting')"
    SCAN_VIDEO_BUTTON = "🔍 Scan Video Now"
    TAB_SCAN_VIDEO = "Scan Video for Subjects"
    SESSION_ACCORDION = "Resume previous Session"
    ADVANCED_ACCORDION = "Advanced Processing Settings"
    BATCH_FILTER_ACCORDION = "Batch Filter Scenes"

    # Strategy Options (Match exact UI text including icons)
    STRATEGY_AUTO = "🤖 Automatic Detection"
    STRATEGY_FACE = "👤 Source Face Reference"
    STRATEGY_TEXT = "📝 Text Description (Limited)"
