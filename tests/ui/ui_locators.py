"""Centralized UI locators — single place to update when the UI changes."""


class Selectors:
    """elem_id-based CSS selectors (most resilient to label changes)."""

    # Global Components
    UNIFIED_LOG = "#unified_log"
    UNIFIED_STATUS = "#unified_status"
    LOG_TEXTAREA = "textarea[aria-label='System Logs Output']"
    CANCEL_BUTTON = "button:has-text('Cancel')"
    PAUSE_BUTTON = "button:has-text('Pause')"
    RESET_STATE_BUTTON = "button:has-text('Reset State (MOCKED)')"

    # Status Message Constants (regex for robustness)
    STATUS_READY = "System Reset Ready."
    STATUS_ERROR_REGEX = "Error:|Fail|Invalid"
    STATUS_SUCCESS_EXTRACTION = "Extraction Complete"
    STATUS_SUCCESS_PRE_ANALYSIS = "Pre-Analysis Complete"
    STATUS_SUCCESS_PROPAGATION = "Propagation Complete"
    STATUS_SUCCESS_ANALYSIS = "Analysis Complete"
    STATUS_SUCCESS_EXPORT = "Export Complete"

    # Extraction Tab (Source)
    SOURCE_INPUT = "#source_input textarea, #source_input input"
    START_EXTRACTION = "button:has-text('Start Extraction')"
    ADD_TO_QUEUE = "button:has-text('Queue for Batch')"
    THUMB_MEGAPIXELS = "#thumb_megapixels_input input[data-testid='number-input']"
    SESSION_INPUT = "#session_path_input textarea, #session_path_input input"
    LOAD_SESSION_BUTTON = "button:has-text('Load Session')"
    MAX_RESOLUTION = "#max_resolution input"
    EXTRACTION_METHOD = "#method_input"

    # Subject/Scenes Tab
    SEED_STRATEGY = "input[aria-label='How to find the subject?']"
    START_PRE_ANALYSIS = "button:has-text('Confirm Subject')"
    SCENE_GALLERY = "#scene_gallery"
    VIEW_TOGGLE = "button[aria-label='Toggle Gallery View']"
    MASK_AREA_MIN = "input[aria-label='Min Subject Area %']"
    QUALITY_SCORE_MIN = "input[aria-label='Min Quality Score']"
    PROPAGATE_MASKS = "button:has-text('Propagate Masks')"
    SCENE_FILTER_STATUS = "#scene_filter_status"

    # Metrics & Export Tab
    START_ANALYSIS = "button:has-text('Start Analysis')"
    FILTER_PRESET = "input[aria-label='Filter Preset']"
    DEDUP_THRESH = "input[aria-label='Deduplication Threshold']"
    EXPORT_BUTTON = "button:has-text('Start Export')"
    DRY_RUN_BUTTON = "button:has-text('Dry Run')"


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
    HELP_ACCORDION = "❓ Help / Troubleshooting"
    SCAN_VIDEO_BUTTON = "🔍 Scan Video Now"
    TAB_SCAN_VIDEO = "Scan Video for Subjects"
    SESSION_ACCORDION = "Resume previous Session"
    ADVANCED_ACCORDION = "Advanced Processing Settings"
    BATCH_FILTER_ACCORDION = "Batch Filter Scenes"

    # Strategy Options
    STRATEGY_FACE = "Source Face Reference"
    STRATEGY_TEXT = "Text Description (Limited)"
