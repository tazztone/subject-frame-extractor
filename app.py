# keep app.py Monolithic
import gradio as gr
import cv2
import numpy as np
import os
import json
import re
import shutil
import logging
import threading
import time
import subprocess
import gc
from pathlib import Path
from datetime import datetime
from queue import Queue, Empty
from dataclasses import dataclass, asdict, field
from concurrent.futures import ThreadPoolExecutor
import hashlib

# --- Optional Dependency Imports ---
# These are guarded to allow the app to function without all features installed.
try:
    import yt_dlp as ytdlp
except ImportError:
    ytdlp = None

try:
    from scenedetect import detect, ContentDetector
except ImportError:
    detect, ContentDetector = None, None

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    import torch
except ImportError:
    torch = None

# --- Integrated DAM4SAM Tracker and Dependencies ---
DAM4SAMTracker = None
try:
    # Check for core DAM4SAM dependencies first
    import yaml
    import random
    from collections import OrderedDict
    import torchvision.transforms.functional as F
    from vot.region.raster import calculate_overlaps
    from vot.region.shapes import Mask
    from vot.region import RegionType
    from sam2.build_sam import build_sam2_video_predictor

    # --- Utility functions required by DAM4SAMTracker (integrated) ---

    def keep_largest_component(mask):
        """Keeps only the largest connected component in a binary mask."""
        # Finds all connected components and returns a mask with only the largest one.
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)
        if num_labels > 1:
            # The 0-th label is the background.
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            return (labels == largest_label).astype(np.uint8)
        return mask

    def determine_tracker(tracker_name="sam2.1"):
        """
        Maps a tracker name to its checkpoint and config file paths.
        FIX: Add explicit existence checks for model files to fail early.
        """
        base_path = Path(__file__).parent / "models"
        base_path.mkdir(exist_ok=True)

        checkpoints = {
            "sam2.1": "sam2_hiera_l.pt",
            "sam21pp-L": "sam2_hiera_l.pt",
        }
        model_configs = {
            "default": "sam2_hiera_video.yaml"
        }
        
        checkpoint_filename = checkpoints.get(tracker_name, "sam2_hiera_l.pt")
        config_filename = model_configs["default"]

        checkpoint_path = base_path / checkpoint_filename
        config_path = base_path / config_filename

        # FIX: Verify that the model checkpoint and config files exist before proceeding.
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Model checkpoint file not found: '{checkpoint_filename}'. "
                f"Please ensure it exists in the '{base_path.resolve()}' directory."
            )
        if not config_path.is_file():
            raise FileNotFoundError(
                f"Model config file not found: '{config_filename}'. "
                f"Please ensure it exists in the '{base_path.resolve()}' directory."
            )

        return str(checkpoint_path), str(config_path)

    # --- Embedded DAM4SAM Configuration ---
    dam4sam_config = {"seed": 1234}

    # --- DAM4SAMTracker Class Definition (from dam4sam_tracker.py) ---

    class DAM4SAMTracker():
        def __init__(self, tracker_name="sam21pp-L"):
            """
            Constructor for the DAM4SAM (2 or 2.1) tracking wrapper.
            """
            seed = dam4sam_config["seed"]
            random.seed(seed)
            os.environ['PYTHONHASHSEED'] = str(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)

            self.checkpoint, self.model_cfg = determine_tracker(tracker_name)

            self.input_image_size = 1024
            self.img_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)[:, None, None]
            self.img_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)[:, None, None]

            self.predictor = build_sam2_video_predictor(self.model_cfg, self.checkpoint, device="cuda:0")
            self.tracking_times = []

        def _prepare_image(self, img_pil):
            img = torch.from_numpy(np.array(img_pil)).to(self.inference_state["device"])
            img = img.permute(2, 0, 1).float() / 255.0
            img = F.resize(img, (self.input_image_size, self.input_image_size))
            img = (img - self.img_mean) / self.img_std
            return img

        @torch.inference_mode()
        def init_state_tw(self):
            """Initialize an inference state."""
            compute_device = torch.device("cuda")
            inference_state = {}
            inference_state["images"] = None
            inference_state["num_frames"] = 0
            inference_state["offload_video_to_cpu"] = False
            inference_state["offload_state_to_cpu"] = False
            inference_state["video_height"] = None
            inference_state["video_width"] =  None
            inference_state["device"] = compute_device
            inference_state["storage_device"] = compute_device
            inference_state["point_inputs_per_obj"] = {}
            inference_state["mask_inputs_per_obj"] = {}
            inference_state["adds_in_drm_per_obj"] = {}
            inference_state["cached_features"] = {}
            inference_state["constants"] = {}
            inference_state["obj_id_to_idx"] = OrderedDict()
            inference_state["obj_idx_to_id"] = OrderedDict()
            inference_state["obj_ids"] = []
            inference_state["output_dict"] = {
                "cond_frame_outputs": {},
                "non_cond_frame_outputs": {},
            }
            inference_state["output_dict_per_obj"] = {}
            inference_state["temp_output_dict_per_obj"] = {}
            inference_state["consolidated_frame_inds"] = {
                "cond_frame_outputs": set(),
                "non_cond_frame_outputs": set(),
            }
            inference_state["tracking_has_started"] = False
            inference_state["frames_already_tracked"] = {}
            inference_state["frames_tracked_per_obj"] = {}

            self.img_mean = self.img_mean.to(compute_device)
            self.img_std = self.img_std.to(compute_device)

            return inference_state

        @torch.inference_mode()
        def initialize(self, image, init_mask, bbox=None):
            if type(init_mask) is list:
                init_mask = init_mask[0]
            self.frame_index = 0
            self.object_sizes = []
            self.last_added = -1

            self.img_width = image.width
            self.img_height = image.height
            self.inference_state = self.init_state_tw()
            self.inference_state["images"] = image
            video_width, video_height = image.size
            self.inference_state["video_height"] = video_height
            self.inference_state["video_width"] =  video_width
            prepared_img = self._prepare_image(image)
            self.inference_state["images"] = {0 : prepared_img}
            self.inference_state["num_frames"] = 1
            self.predictor.reset_state(self.inference_state)

            self.predictor._get_image_feature(self.inference_state, frame_idx=0, batch_size=1)

            if init_mask is None:
                if bbox is None:
                    raise ValueError("Initialization state (bbox or mask) is required.")
                init_mask = self.estimate_mask_from_box(bbox)

            _, _, out_mask_logits = self.predictor.add_new_mask(
                inference_state=self.inference_state,
                frame_idx=0,
                obj_id=0,
                mask=init_mask,
            )

            m = (out_mask_logits[0, 0] > 0).float().cpu().numpy().astype(np.uint8)
            self.inference_state["images"].pop(self.frame_index)

            out_dict = {'pred_mask': m}
            return out_dict

        @torch.inference_mode()
        def track(self, image, init=False):
            prepared_img = self._prepare_image(image).unsqueeze(0)
            if not init:
                self.frame_index += 1
                self.inference_state["num_frames"] += 1
            self.inference_state["images"][self.frame_index] = prepared_img

            for out in self.predictor.propagate_in_video(self.inference_state, start_frame_idx=self.frame_index, max_frame_num_to_track=0, return_all_masks=True):
                if len(out) == 3:
                    out_frame_idx, _, out_mask_logits = out
                    m = (out_mask_logits[0][0] > 0.0).float().cpu().numpy().astype(np.uint8)
                else:
                    out_frame_idx, _, out_mask_logits, alternative_masks_ious = out
                    m = (out_mask_logits[0][0] > 0.0).float().cpu().numpy().astype(np.uint8)

                    alternative_masks, out_all_ious = alternative_masks_ious
                    m_idx = np.argmax(out_all_ious)
                    m_iou = out_all_ious[m_idx]
                    alternative_masks = [mask for i, mask in enumerate(alternative_masks) if i != m_idx]

                    n_pixels = (m == 1).sum()
                    self.object_sizes.append(n_pixels)
                    if len(self.object_sizes) > 1 and n_pixels >= 1:
                        obj_sizes_ratio = n_pixels / np.median([
                            size for size in self.object_sizes[-300:] if size >= 1
                        ][-10:])
                    else:
                        obj_sizes_ratio = -1

                    if m_iou > 0.8 and obj_sizes_ratio >= 0.8 and obj_sizes_ratio <= 1.2 and n_pixels >= 1 and (self.frame_index - self.last_added > 5 or self.last_added == -1):
                        alternative_masks = [Mask((m_[0][0] > 0.0).cpu().numpy()).rasterize((0, 0, self.img_width - 1, self.img_height - 1)).astype(np.uint8)
                                         for m_ in alternative_masks]

                        chosen_mask_np = m.copy()
                        chosen_bbox = Mask(chosen_mask_np).convert(RegionType.RECTANGLE)

                        alternative_masks = [np.logical_and(m_, np.logical_not(chosen_mask_np)).astype(np.uint8) for m_ in alternative_masks]
                        alternative_masks = [keep_largest_component(m_) for m_ in alternative_masks if np.sum(m_) >= 1]
                        if len(alternative_masks) > 0:
                            alternative_masks = [np.logical_or(m_, chosen_mask_np).astype(np.uint8) for m_ in alternative_masks]
                            alternative_bboxes = [Mask(m_).convert(RegionType.RECTANGLE) for m_ in alternative_masks]
                            ious = [calculate_overlaps([chosen_bbox], [bbox])[0] for bbox in alternative_bboxes]

                            if np.min(np.array(ious)) <= 0.7:
                                self.last_added = self.frame_index
                                self.predictor.add_to_drm(
                                    inference_state=self.inference_state,
                                    frame_idx=out_frame_idx,
                                    obj_id=0,
                                )

                out_dict = {'pred_mask': m}
                self.inference_state["images"].pop(self.frame_index)
                return out_dict

        def estimate_mask_from_box(self, bbox):
            (
                _,
                _,
                current_vision_feats,
                current_vision_pos_embeds,
                feat_sizes,
            ) = self.predictor._get_image_feature(self.inference_state, 0, 1)

            box = np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])[None, :]
            box = torch.as_tensor(box, dtype=torch.float, device=current_vision_feats[0].device)

            from sam2.utils.transforms import SAM2Transforms
            _transforms = SAM2Transforms(
                resolution=self.predictor.image_size,
                mask_threshold=0.0,
                max_hole_area=0.0,
                max_sprinkle_area=0.0,
            )
            unnorm_box = _transforms.transform_boxes(
                box, normalize=True, orig_hw=(self.img_height, self.img_width)
            )

            box_coords = unnorm_box.reshape(-1, 2, 2)
            box_labels = torch.tensor([[2, 3]], dtype=torch.int, device=unnorm_box.device)
            box_labels = box_labels.repeat(unnorm_box.size(0), 1)
            concat_points = (box_coords, box_labels)

            sparse_embeddings, dense_embeddings = self.predictor.sam_prompt_encoder(
                points=concat_points,
                boxes=None,
                masks=None
            )

            high_res_features = []
            for i in range(2):
                _, b_, c_ = current_vision_feats[i].shape
                high_res_features.append(current_vision_feats[i].permute(1, 2, 0).view(b_, c_, feat_sizes[i][0], feat_sizes[i][1]))
            if self.predictor.directly_add_no_mem_embed:
                img_embed = current_vision_feats[2] + self.predictor.no_mem_embed
            else:
                img_embed = current_vision_feats[2]
            _, b_, c_ = current_vision_feats[2].shape
            img_embed = img_embed.permute(1, 2, 0).view(b_, c_, feat_sizes[2][0], feat_sizes[2][1])
            low_res_masks, iou_predictions, _, _ = self.predictor.sam_mask_decoder(
                image_embeddings=img_embed,
                image_pe=self.predictor.sam_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
                repeat_image=(concat_points is not None and concat_points[0].shape[0] > 1),
                high_res_features=high_res_features,
            )

            masks = _transforms.postprocess_masks(
                low_res_masks, (self.img_height, self.img_width)
            )
            masks = masks > 0

            return masks.squeeze(0).float().detach().cpu().numpy()[0]

except ImportError as e:
    logging.getLogger(__name__).warning("DAM4SAM dependencies missing, tracker disabled: %s", e)
    DAM4SAMTracker = None

try:
    from insightface.app import FaceAnalysis
except ImportError:
    FaceAnalysis = None

try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Provide dummy decorators if numba is not installed
    def njit(func=None, **kwargs):
        if func: return func
        return lambda f: f
    prange = range

# --- Central Configuration & Setup ---
def get_feature_status():
    """Checks for optional dependencies and returns their status."""
    masking_libs_installed = torch is not None and DAM4SAMTracker is not None and Image is not None
    cuda_available = torch is not None and torch.cuda.is_available()
    return {
        'face_analysis': FaceAnalysis is not None,
        'youtube_dl': ytdlp is not None,
        'scene_detection': detect is not None,
        'masking': masking_libs_installed and cuda_available,
        'masking_libs_installed': masking_libs_installed,
        'cuda_available': cuda_available,
        'numba_acceleration': NUMBA_AVAILABLE,
    }

class Config:
    """Centralized configuration management."""
    BASE_DIR = Path(__file__).parent
    LOG_DIR = BASE_DIR / "logs"
    CONFIGS_DIR = BASE_DIR / "configs"
    DOWNLOADS_DIR = BASE_DIR / "downloads"
    LOG_FILE = "frame_extractor.log"
    QUALITY_METRICS = ["sharpness", "edge_strength", "contrast", "brightness", "entropy"]
    QUALITY_WEIGHTS = {"sharpness": 30, "edge_strength": 20, "contrast": 20, "brightness": 10, "entropy": 20}
    NORMALIZATION_CONSTANTS = {"sharpness": 1000, "edge_strength": 100}
    FILTER_MODES = {"OVERALL": "Overall Quality", "INDIVIDUAL": "Individual Metrics"}
    # FIX: Clarify mask area threshold default to 1.0%
    MIN_MASK_AREA_PCT = 1.0
    # FIX: Add configurable downscale factor for quality metrics
    QUALITY_DOWNSCALE_FACTOR = 0.25
    UI_DEFAULTS = {
        "method": "keyframes", "interval": 5.0, "max_resolution": "maximum available",
        "fast_scene": False, "resume": True, "use_png": True, "disable_parallel": False,
        "enable_face_filter": False, "face_model_name": "buffalo_l",
        "min_face_confidence": 0.9, "min_face_area": 5.0, "pre_filter_quality_enabled": False,
        "pre_filter_mode": "Overall Quality", "pre_quality_thresh": 12.0, "pre_filter_face_present": False,
        "quality_thresh": 12.0, "face_thresh": 0.5, "sharpness_thresh": 0.0,
        "edge_strength_thresh": 0.0, "contrast_thresh": 0.0, "brightness_thresh": 0.0, "entropy_thresh": 0.0,
        "enable_subject_mask": False, "scene_detect": True,
        "dam4sam_model_name": "sam2.1",
    }

    @staticmethod
    def setup_directories():
        Config.LOG_DIR.mkdir(exist_ok=True)
        Config.CONFIGS_DIR.mkdir(exist_ok=True)
        Config.DOWNLOADS_DIR.mkdir(exist_ok=True)

# --- Global Instance & Setup ---
config = Config()
config.setup_directories()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(config.LOG_DIR / config.LOG_FILE)],
)
logger = logging.getLogger(__name__)

# --- Utility & Helper Functions ---
def check_dependencies():
    """Checks for presence of essential command-line tools."""
    if not shutil.which("ffmpeg"):
        logger.error("FFMPEG is not installed or not in PATH.")
        raise RuntimeError("FFMPEG is not installed. Please install it to continue.")

def sanitize_filename(name, max_length=50):
    """Sanitizes a string to be a valid filename."""
    return re.sub(r'[^\w\-_.]', '_', name)[:max_length]

# --- Numba Optimized / NumPy Fallback Image Processing ---
if NUMBA_AVAILABLE:
    @njit(parallel=True)
    def compute_edge_strength(sobelx, sobely):
        height, width = sobelx.shape
        total_mag = 0.0
        for i in prange(height):
            for j in range(width):
                total_mag += np.sqrt(sobelx[i, j]**2 + sobely[i, j]**2)
        return total_mag / (height * width) if (height * width) > 0 else 0

    @njit
    def compute_entropy(hist):
        total = np.sum(hist) + 1e-7
        entropy = 0.0
        prob = hist / total
        for p in prob:
            if p > 0:
                entropy -= p * np.log2(p)
        return min(max(entropy / 8.0, 0), 1.0)
else:
    # FIX: Add pure NumPy vectorized fallbacks for better performance without Numba.
    def compute_edge_strength(sobelx, sobely):
        magnitude = np.sqrt(sobelx.astype(np.float64)**2 + sobely.astype(np.float64)**2)
        return np.mean(magnitude)

    def compute_entropy(hist):
        total = np.sum(hist) + 1e-7
        prob = hist / total
        # Use boolean indexing to avoid log(0)
        nz = prob > 0
        entropy = -np.sum(prob[nz] * np.log2(prob[nz]))
        return min(max(entropy / 8.0, 0), 1.0)

# --- Core Data Classes ---
@dataclass
class FrameMetrics:
    quality_score: float = 0.0
    sharpness_score: float = 0.0
    edge_strength_score: float = 0.0
    contrast_score: float = 0.0
    brightness_score: float = 0.0
    entropy_score: float = 0.0

@dataclass
class Frame:
    image_data: np.ndarray
    frame_number: int
    metrics: FrameMetrics | None = None
    face_similarity_score: float | None = None
    max_face_confidence: float | None = None
    error: str | None = None

    def calculate_quality_metrics(self, mask: np.ndarray | None = None):
        try:
            gray = cv2.cvtColor(self.image_data, cv2.COLOR_BGR2GRAY)
            active_mask = None
            if mask is not None and mask.ndim == 2:
                active_mask = (mask > 128).astype(np.uint8)
                if np.sum(active_mask) < 100:
                    raise ValueError("Subject mask is too small for metric calculation.")
            
            gray_for_metrics = cv2.bitwise_and(gray, gray, mask=active_mask) if active_mask is not None else gray
            # Use configurable downscale factor
            dsf = config.QUALITY_DOWNSCALE_FACTOR
            preview = cv2.resize(gray_for_metrics, (0, 0), fx=dsf, fy=dsf, interpolation=cv2.INTER_AREA)
            preview_mask = cv2.resize(active_mask, preview.shape[::-1], interpolation=cv2.INTER_NEAREST) if active_mask is not None else None

            lap = cv2.Laplacian(preview, cv2.CV_64F)
            masked_lap_pixels = lap[preview_mask > 0] if preview_mask is not None else lap
            sharpness = float(np.var(masked_lap_pixels[masked_lap_pixels != 0])) if masked_lap_pixels.size > 0 else 0.0
            sharpness_norm = min(max(sharpness / config.NORMALIZATION_CONSTANTS["sharpness"], 0), 1.0)
            
            sobelx = cv2.Sobel(preview, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(preview, cv2.CV_64F, 0, 1, ksize=3)
            edge_strength = compute_edge_strength(sobelx, sobely)
            edge_strength_norm = min(max(edge_strength / config.NORMALIZATION_CONSTANTS["edge_strength"], 0), 1.0)

            pixel_values = gray[active_mask > 0] if active_mask is not None else gray
            mean_brightness, std_brightness = (float(np.mean(pixel_values)), float(np.std(pixel_values))) if pixel_values.size > 0 else (0,0)
            brightness = mean_brightness / 255.0
            contrast = std_brightness / (mean_brightness + 1e-7)
            contrast_clamped = min(max(contrast, 0), 2.0) / 2.0

            hist = cv2.calcHist([gray], [0], active_mask, [256], [0, 256]).flatten()
            entropy = compute_entropy(hist)

            self.metrics = FrameMetrics(sharpness_score=sharpness_norm * 100, edge_strength_score=edge_strength_norm * 100,
                                        contrast_score=contrast_clamped * 100, brightness_score=brightness * 100, entropy_score=entropy * 100)
            w = [config.QUALITY_WEIGHTS[k] / 100.0 for k in config.QUALITY_METRICS]
            scores = [sharpness_norm, edge_strength_norm, contrast_clamped, brightness, entropy]
            self.metrics.quality_score = max(min(sum(s * weight for s, weight in zip(scores, w)) * 100, 100), 0)
        except Exception as e:
            self.error = f"Quality calculation failed: {e}"
            logger.error(f"Frame {self.frame_number}: {self.error}")
            self.metrics = FrameMetrics()

@dataclass
class AnalysisStats:
    processed: int = 0
    kept: int = 0
    discarded: int = 0
    disc_quality: int = 0
    disc_face: int = 0
    disc_mask: int = 0

@dataclass
class AnalysisParameters:
    source_path: str = ""
    output_folder: str = ""
    method: str = config.UI_DEFAULTS["method"]
    interval: float = config.UI_DEFAULTS["interval"]
    max_resolution: str = config.UI_DEFAULTS["max_resolution"]
    fast_scene: bool = config.UI_DEFAULTS["fast_scene"]
    use_png: bool = config.UI_DEFAULTS["use_png"]
    disable_parallel: bool = config.UI_DEFAULTS["disable_parallel"]
    resume: bool = config.UI_DEFAULTS["resume"]
    enable_face_filter: bool = config.UI_DEFAULTS["enable_face_filter"]
    face_ref_img_path: str = ""
    face_model_name: str = config.UI_DEFAULTS["face_model_name"]
    min_face_confidence: float = config.UI_DEFAULTS["min_face_confidence"]
    min_face_area: float = config.UI_DEFAULTS["min_face_area"]
    pre_filter_quality_enabled: bool = config.UI_DEFAULTS["pre_filter_quality_enabled"]
    pre_filter_mode: str = config.UI_DEFAULTS["pre_filter_mode"]
    pre_quality_thresh: float = config.UI_DEFAULTS["pre_quality_thresh"]
    pre_filter_face_present: bool = config.UI_DEFAULTS["pre_filter_face_present"]
    quality_weights: dict = None
    thresholds: dict = None
    enable_subject_mask: bool = field(default=config.UI_DEFAULTS["enable_subject_mask"])
    dam4sam_model_name: str = field(default=config.UI_DEFAULTS["dam4sam_model_name"])
    scene_detect: bool = field(default=config.UI_DEFAULTS["scene_detect"])

# --- Subject Masking Logic ---
@dataclass
class MaskingResult:
    mask_path: str | None = None
    shot_id: int | None = None
    seed_type: str | None = None
    seed_face_sim: float | None = None
    mask_area_pct: float | None = None
    mask_empty: bool = True
    error: str | None = None

class SubjectMasker:
    def __init__(self, params, progress_queue, cancel_event, frame_map=None, face_analyzer=None, reference_embedding=None):
        self.params = params
        self.progress_queue = progress_queue
        self.cancel_event = cancel_event
        self.mask_dir = None
        self.shots = []
        self.frame_map = frame_map
        self.face_analyzer = face_analyzer
        self.reference_embedding = reference_embedding
        self.tracker = None

    def _initialize_dam4sam_tracker(self):
        """Initializes the DAM4SAM tracker for a single shot."""
        if not DAM4SAMTracker:
            msg = "[ERROR] DAM4SAM dependencies (torch, sam2, vot etc.) are not installed."
            self.progress_queue.put({"log": msg})
            raise ImportError(msg)

        if not torch.cuda.is_available():
            msg = "[ERROR] DAM4SAM masking requires a CUDA-enabled GPU, but CUDA is not available."
            self.progress_queue.put({"log": msg})
            raise RuntimeError(msg)

        try:
            model_name = self.params.dam4sam_model_name
            self.progress_queue.put({"log": f"[INFO] Initializing DAM4SAM tracker with model '{model_name}'..."})
            self.tracker = DAM4SAMTracker(model_name)
            self.progress_queue.put({"log": "[SUCCESS] DAM4SAM tracker initialized."})
            return True
        except Exception as e:
            error_msg = f"Failed to initialize DAM4SAM tracker with model '{model_name}': {e}"
            logger.error(error_msg, exc_info=True)
            self.progress_queue.put({"log": f"[ERROR] {error_msg}"})
            self.tracker = None
            return False

    def run(self, video_path: str, frames_dir: str) -> dict[str, dict]:
        # NOTE: Re-initializing the tracker per shot is safer for state but adds overhead.
        # This is a deliberate trade-off for stability.
        try:
            self.mask_dir = Path(frames_dir) / "masks"
            self.mask_dir.mkdir(exist_ok=True)
            self.progress_queue.put({"log": "[INFO] Starting subject masking..."})

            if self.params.scene_detect:
                self._detect_scenes(video_path, frames_dir)
            else:
                if self.frame_map:
                    last_frame = max(self.frame_map.keys()) if self.frame_map else 0
                    self.shots = [(0, last_frame + 1)]
                else:
                    self.shots = [(0, len(list(Path(frames_dir).glob('frame_*.*'))))]

            mask_metadata = {}
            total_frames_in_shots = sum(end - start for start, end in self.shots)
            self.progress_queue.put({"total": total_frames_in_shots, "stage": "Masking"})

            for shot_id, (start_frame, end_frame) in enumerate(self.shots):
                if self.cancel_event.is_set(): break
                self.progress_queue.put({"log": f"[INFO] Masking shot {shot_id+1}/{len(self.shots)} (Frames {start_frame}-{end_frame})"})

                if not self._initialize_dam4sam_tracker():
                    self.progress_queue.put({"log": f"[ERROR] Could not initialize tracker for shot {shot_id+1}. Skipping."})
                    self.progress_queue.put({"progress": end_frame - start_frame})
                    continue
                
                shot_frames_with_nums = self._load_shot_frames(frames_dir, start_frame, end_frame)
                if not shot_frames_with_nums:
                    self.progress_queue.put({"log": f"[INFO] No extracted frames found for shot {shot_id+1}. Skipping."})
                    self.progress_queue.put({"progress": end_frame - start_frame})
                    continue

                shot_frames_data = [f[1] for f in shot_frames_with_nums]
                seed_frame_local_idx, bbox_xywh, seed_details = self._seed_identity(shot_frames_data)

                if bbox_xywh is None:
                    self.progress_queue.put({"log": f"[WARNING] Could not identify subject in shot {shot_id+1}. Skipping mask generation."})
                    for original_frame_num, _ in shot_frames_with_nums:
                        frame_filename = self.frame_map.get(original_frame_num)
                        if frame_filename:
                            mask_metadata[frame_filename] = asdict(MaskingResult(error="Subject not found in shot", shot_id=shot_id))
                    self.progress_queue.put({"progress": end_frame - start_frame})
                    continue

                masks, mask_area_pcts, mask_empty_flags, mask_errors = self._propagate_masks_dam4sam(shot_frames_data, seed_frame_local_idx, bbox_xywh)

                if len(masks) != len(shot_frames_with_nums):
                    self.progress_queue.put({"log": f"[ERROR] Mask propagation returned {len(masks)} masks for {len(shot_frames_with_nums)} frames in shot {shot_id+1}. This indicates an internal error."})
                    self.progress_queue.put({"progress": end_frame - start_frame})
                    continue

                for i, mask in enumerate(masks):
                    original_frame_num = shot_frames_with_nums[i][0]
                    frame_filename = self.frame_map.get(original_frame_num)
                    if not frame_filename: continue

                    frame_path = Path(frames_dir) / frame_filename
                    mask_path = self.mask_dir / f"{frame_path.stem}.png"

                    current_mask_area_pct = mask_area_pcts[i]
                    current_mask_empty = mask_empty_flags[i]

                    current_mask_error_from_propagation = mask_errors[i]
                    merged_error_messages = []
                    if current_mask_error_from_propagation: merged_error_messages.append(current_mask_error_from_propagation)
                    if current_mask_empty: merged_error_messages.append("Empty mask generated")
                    final_error_string = ", ".join(sorted(list(set(merged_error_messages)))) if merged_error_messages else None

                    result = None
                    if mask is not None and np.any(mask):
                        cv2.imwrite(str(mask_path), mask)
                        result = MaskingResult(
                            mask_path=str(mask_path), shot_id=shot_id,
                            seed_type=seed_details.get('type'), seed_face_sim=seed_details.get('seed_face_sim'),
                            mask_area_pct=current_mask_area_pct,
                            mask_empty=current_mask_empty,
                            error=final_error_string
                        )
                    else:
                        result = MaskingResult(
                            mask_path=None, shot_id=shot_id,
                            seed_type=seed_details.get('type'), seed_face_sim=seed_details.get('seed_face_sim'),
                            mask_area_pct=current_mask_area_pct,
                            mask_empty=current_mask_empty,
                            error=final_error_string
                        )
                    mask_metadata[frame_path.name] = asdict(result)

                newly_processed = end_frame - start_frame
                self.progress_queue.put({"progress": newly_processed})

            self.progress_queue.put({"log": "[SUCCESS] Subject masking complete."})
            return mask_metadata
        finally:
            if hasattr(self, 'tracker') and self.tracker is not None:
                del self.tracker
                self.tracker = None
                gc.collect()

            if torch and torch.cuda.is_available():
                torch.cuda.empty_cache()
                self.progress_queue.put({"log": "[INFO] GPU memory released."})

    def _detect_scenes(self, video_path: str, frames_dir: str):
        if not detect:
            self.progress_queue.put({"log": "[ERROR] PySceneDetect is not installed, but scene detection was enabled for masking. Cannot proceed."})
            raise ImportError("PySceneDetect is required for this operation but is not installed.")

        try:
            self.progress_queue.put({"log": "[INFO] Detecting scene cuts..."})
            scene_list = detect(video_path, ContentDetector())
            if not scene_list:
                cap = cv2.VideoCapture(video_path)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                self.shots = [(0, frame_count)]
            else:
                self.shots = [(s.frame_num, e.frame_num) for s, e in scene_list]

            self.progress_queue.put({"log": f"[INFO] Found {len(self.shots)} shots."})
        except Exception as e:
            logger.error(f"Scene detection failed: {e}")
            self.progress_queue.put({"log": f"[ERROR] Scene detection failed: {e}"})

    def _load_shot_frames(self, frames_dir, start, end):
        frames = []
        if not self.frame_map:
            self.progress_queue.put({"log": "[ERROR] No frame map available for SubjectMasker. Cannot process shot."})
            return []

        shot_frame_numbers = sorted([fn for fn in self.frame_map.keys() if start <= fn < end])

        for fn in shot_frame_numbers:
            p = Path(frames_dir) / self.frame_map[fn]
            if p.exists():
                frame_data = cv2.imread(str(p))
                if frame_data is not None:
                    frames.append((fn, frame_data))
                else:
                    self.progress_queue.put({"log": f"[WARNING] Failed to load frame data from {p.name}. It may be corrupt. Skipping."})
            else:
                self.progress_queue.put({"log": f"[WARNING] Frame file missing for frame {fn}: {p}"})
                
        return frames

    def _seed_identity(self, shot_frames):
        if not shot_frames:
            return None, None, None

        seed_details = {}
        if self.face_analyzer and self.reference_embedding is not None:
            self.progress_queue.put({"log": "[INFO] Searching for reference face in first 5 frames..."})
            for i, frame_img in enumerate(shot_frames[:5]):
                if frame_img is None: continue
                faces = self.face_analyzer.get(frame_img)
                if not faces: continue

                best_match, min_dist = None, float('inf')
                for face in faces:
                    dist = 1 - np.dot(face.normed_embedding, self.reference_embedding)
                    if dist < min_dist:
                        min_dist, best_match = dist, face
                
                if best_match and min_dist < 0.6:
                    self.progress_queue.put({"log": f"[INFO] Found reference face in frame {i} with distance {min_dist:.2f}. Seeding from here."})
                    
                    h, w, _ = frame_img.shape
                    x1, y1, x2, y2 = best_match.bbox.astype(int)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w - 1, x2), min(h - 1, y2)
                    bbox_xywh = [x1, y1, x2 - x1, y2 - y1]
                    
                    seed_details = {'type': 'face_match', 'seed_face_sim': 1 - min_dist}
                    return i, bbox_xywh, seed_details

        self.progress_queue.put({"log": "[INFO] No reference match found. Seeding with the most prominent face in the first frame."})
        first_frame = shot_frames[0]
        if first_frame is None:
             self.progress_queue.put({"log": "[WARNING] First frame of shot is invalid. Cannot seed."})
             return None, None, None

        h, w, _ = first_frame.shape
        if not self.face_analyzer:
             self.progress_queue.put({"log": "[WARNING] Face analyzer not available. Using fallback rectangle on first frame."})
             bbox_xywh = [w // 4, h // 4, w // 2, h // 2]
             seed_details = {'type': 'fallback_rect'}
             return 0, bbox_xywh, seed_details

        faces = self.face_analyzer.get(first_frame)
        if not faces:
            self.progress_queue.put({"log": "[WARNING] No faces found in the first frame to seed shot. Skipping."})
            return None, None, None

        largest_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        
        x1, y1, x2, y2 = largest_face.bbox.astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        bbox_xywh = [x1, y1, x2 - x1, y2 - y1]
        
        self.progress_queue.put({"log": f"[INFO] Seeding with largest face found at {bbox_xywh} in the first frame."})
        
        seed_details = {'type': 'face_largest', 'seed_face_sim': None}
        return 0, bbox_xywh, seed_details 
    
    def _calculate_mask_metrics(self, mask_np, original_image_shape):
        height, width = original_image_shape[:2]
        image_area = height * width

        mask_area_pct = 0.0
        mask_empty = True
        error_message = None

        if mask_np is not None and np.any(mask_np):
            mask_area_pixels = np.sum(mask_np > 0)
            mask_area_pct = (mask_area_pixels / image_area) * 100.0

            if mask_area_pct < config.MIN_MASK_AREA_PCT:
                mask_empty = True
                error_message = f"Mask area ({mask_area_pct:.2f}%) below min threshold ({config.MIN_MASK_AREA_PCT:.2f}%)"
            else:
                mask_empty = False
        else:
            error_message = "Empty mask generated"
        
        return mask_area_pct, mask_empty, error_message

    def _propagate_masks_dam4sam(self, shot_frames, seed_idx, bbox_xywh):
        if not self.tracker:
            self.progress_queue.put({"log": "[ERROR] DAM4SAM tracker is not initialized. Cannot propagate masks."})
            h, w, _ = shot_frames[0].shape
            return ([np.zeros((h, w), dtype=np.uint8) for _ in shot_frames],
                    [0.0] * len(shot_frames),
                    [True] * len(shot_frames),
                    ["DAM4SAM tracker not initialized"] * len(shot_frames))

        if not shot_frames:
            return [], [], [], []

        self.progress_queue.put({"log": f"[INFO] Propagating masks for {len(shot_frames)} frames with DAM4SAM..."})
        
        try:
            all_masks, all_mask_area_pcts, all_mask_empty_flags, all_mask_errors = [], [], [], []

            for i, frame_np in enumerate(shot_frames):
                if self.cancel_event.is_set():
                    self.progress_queue.put({"log": "[INFO] Mask propagation cancelled during shot."})
                    break
                
                frame_pil = Image.fromarray(cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB))
                
                if i == seed_idx:
                    outputs = self.tracker.initialize(frame_pil, None, bbox=bbox_xywh)
                else:
                    outputs = self.tracker.track(frame_pil)
                
                pred_mask = outputs.get('pred_mask')
                mask_uint8 = (pred_mask * 255).astype(np.uint8) if pred_mask is not None else np.zeros(frame_np.shape[:2], dtype=np.uint8)
                
                mask_area_pct, mask_empty, mask_err = self._calculate_mask_metrics(mask_uint8, frame_np.shape)

                all_masks.append(mask_uint8)
                all_mask_area_pcts.append(mask_area_pct)
                all_mask_empty_flags.append(mask_empty)
                all_mask_errors.append(mask_err)
            
            remaining = len(shot_frames) - len(all_masks)
            if remaining > 0:
                h, w, _ = shot_frames[0].shape
                for _ in range(remaining):
                    all_masks.append(np.zeros((h, w), dtype=np.uint8))
                    all_mask_area_pcts.append(0.0)
                    all_mask_empty_flags.append(True)
                    all_mask_errors.append("Propagation cancelled")

            self.progress_queue.put({"log": "[SUCCESS] Mask propagation for shot complete."})
            return all_masks, all_mask_area_pcts, all_mask_empty_flags, all_mask_errors

        except Exception as e:
            logger.error(f"DAM4SAM mask propagation failed: {e}", exc_info=True)
            self.progress_queue.put({"log": f"[ERROR] DAM4SAM mask propagation failed: {e}"})
            h, w, _ = shot_frames[0].shape
            error_msg = f"DAM4SAM propagation failed: {e}"
            return ([np.zeros((h, w), dtype=np.uint8) for _ in shot_frames],
                    [0.0] * len(shot_frames),
                    [True] * len(shot_frames),
                    [error_msg] * len(shot_frames))

# --- Filtering Logic ---
def check_frame_passes_filters(frame, params, video_area, face_analyzer, gpu_lock, mask_meta=None):
    """Consolidated function to check if a frame passes pre-filtering criteria."""
    max_confidence = None
    if not frame.metrics:
        return False, "Discarded - Metric Error", max_confidence

    if params.enable_subject_mask and mask_meta:
        if mask_meta.get('mask_empty', False):
            return False, "Discarded - Invalid Mask", max_confidence

    if params.pre_filter_quality_enabled:
        if params.pre_filter_mode == config.FILTER_MODES["OVERALL"]:
            if frame.metrics.quality_score < params.pre_quality_thresh:
                return False, "Discarded - Low Quality", max_confidence
        else:
            th = params.thresholds or {}
            for metric in config.QUALITY_METRICS:
                thresh = th.get(metric, 0)
                if getattr(frame.metrics, f"{metric}_score") < thresh:
                    return False, f"Discarded - Low {metric.replace('_', ' ').title()}", max_confidence

    if params.pre_filter_face_present:
        if not face_analyzer:
            logger.warning("Face pre-filter enabled but insightface is not installed.")
            return False, "Discarded (face lib missing)", None

        try:
            with gpu_lock:
                faces = face_analyzer.get(frame.image_data)
            if not faces: return False, "Discarded - No Face", 0.0

            max_confidence = max(f.det_score for f in faces)
            min_face_px = params.min_face_area * video_area / 100.0
            passes = any(
                f.det_score > params.min_face_confidence and ((f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1])) > min_face_px
                for f in faces
            )
            if not passes: return False, "Discarded - No Qualifying Face", max_confidence
        except Exception as e:
            logger.warning(f"Face pre-filter failed on frame {frame.frame_number}: {e}")
            return False, "Discarded - Face Error", max_confidence

    return True, "Kept", max_confidence

# --- Backend Analysis Pipeline ---
class VideoManager:
    def __init__(self, source_path, max_resolution="maximum available"):
        self.source_path = source_path
        self.max_resolution = max_resolution
        self.video_path = None
        self.is_youtube = "youtube.com/" in source_path or "youtu.be/" in source_path

    def prepare_video(self):
        if self.is_youtube:
            if not ytdlp: raise ImportError("yt-dlp is not installed. Please install it to download YouTube videos.")
            logger.info(f"Downloading video: {self.source_path}")
            self.video_path = self._download_video()
            return str(self.video_path)
        
        local_path = Path(self.source_path)
        if not local_path.exists(): raise FileNotFoundError(f"Video file not found: {local_path}")
        self.video_path = local_path
        return str(self.video_path)

    def _download_video(self):
        res_filter = f"[height<={self.max_resolution}]" if self.max_resolution != "maximum available" else ""
        ydl_opts = {'outtmpl': str(config.DOWNLOADS_DIR / f"%(id)s_%(title).40s_%(height)sp.%(ext)s"),
                    'format': f'bestvideo{res_filter}[ext=mp4][vcodec^=avc1]+bestaudio[ext=m4a]/best{res_filter}[ext=mp4]/best',
                    'merge_output_format': 'mp4', 'noprogress': True, 'quiet': True, 'logger': logger}
        try:
            with ytdlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(self.source_path, download=True)
                return Path(ydl.prepare_filename(info))
        except ytdlp.utils.DownloadError as e:
            logger.error(f"yt-dlp download failed: {e}")
            raise RuntimeError(f"Download failed. The requested resolution may not be available. Details: {str(e)}")


    def get_video_info(self):
        if not self.video_path or not self.video_path.exists():
            raise ValueError("Video path not set or file does not exist.")
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened(): raise IOError(f"Could not open video file: {self.video_path}")
        info = {"width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "fps": cap.get(cv2.CAP_PROP_FPS), "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}
        cap.release()
        return info

class ExtractionPipeline:
    def __init__(self, params: AnalysisParameters, progress_queue: Queue, cancel_event: threading.Event):
        self.params = params
        self.progress_queue = progress_queue
        self.cancel_event = cancel_event
        self.video_path = None
        self.output_dir = None
        self.video_info = {}

    def run(self):
        try:
            self.progress_queue.put({"log": "[INFO] Preparing video source..."})
            vid_manager = VideoManager(self.params.source_path, self.params.max_resolution)
            self.video_path = Path(vid_manager.prepare_video())
            self.progress_queue.put({"log": f"[INFO] Video ready: {self.video_path.name}"})

            self.video_info = vid_manager.get_video_info()
            self.output_dir = self.params.output_folder or (config.DOWNLOADS_DIR / self.video_path.stem)
            self.output_dir.mkdir(exist_ok=True)
            self.progress_queue.put({"output_dir": str(self.output_dir), "video_path": str(self.video_path)})

            self.progress_queue.put({"log": "[INFO] Starting frame extraction..."})
            self._run_frame_extraction()

            if self.cancel_event.is_set():
                self.progress_queue.put({"log": "[INFO] Extraction cancelled."})
                return {"log": "[INFO] Extraction cancelled."}
            
            self.progress_queue.put({"log": "[SUCCESS] Extraction complete."})
            return {"done": True, "output_dir": str(self.output_dir), "video_path": str(self.video_path)}
        except Exception as e:
            logger.exception("Error in extraction pipeline.")
            return {"error": str(e)}

    def _monitor_ffmpeg_progress(self, process):
        """Reads FFmpeg's stdout line-by-line to parse real-time progress."""
        while process.poll() is None:
            if self.cancel_event.is_set():
                break
            line = process.stdout.readline()
            if not line:
                time.sleep(0.1)
                continue
            
            parts = {k.strip(): v.strip() for k, v in [p.split('=') for p in line.split() if '=' in p]}
            frame = int(parts.get('frame', 0))
            if frame > 0:
                self.progress_queue.put({"progress_abs": frame})
    
    def _run_frame_extraction(self):
        """Uses FFmpeg to extract frames, parses progress, and creates a frame map."""
        use_showinfo = self.params.method != 'all'
        loglevel = 'info' if use_showinfo else 'error'
        ffmpeg_cmd = ['ffmpeg', '-y', '-i', str(self.video_path), '-hide_banner', '-loglevel', loglevel, '-progress', 'pipe:1']
        
        select_filter_map = {'interval': f"fps=1/{self.params.interval}", 'keyframes': "select='eq(pict_type,I)'",
                             'scene': f"select='gt(scene,{0.5 if self.params.fast_scene else 0.4})'",
                             'all': f"fps={self.video_info.get('fps', 30)}"}
        select_filter = select_filter_map.get(self.params.method)
        
        if use_showinfo:
            filter_str = (select_filter + ",showinfo") if select_filter else "showinfo"
            ffmpeg_cmd.extend(['-vf', filter_str, '-vsync', 'vfr'])
        elif select_filter:
            ffmpeg_cmd.extend(['-vf', select_filter, '-vsync', 'vfr'])

        ext = 'png' if self.params.use_png else 'jpg'
        ffmpeg_cmd.extend(['-f', 'image2', str(self.output_dir / f"frame_%06d.{ext}")])
        
        log_file_path = self.output_dir / "ffmpeg_log.txt"
        
        stderr_handle = None
        process = None
        progress_thread = None
        try:
            stderr_handle = open(log_file_path, 'w') if use_showinfo else subprocess.DEVNULL
            
            process = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=stderr_handle, text=True, encoding='utf-8', bufsize=1)
            
            progress_thread = threading.Thread(target=self._monitor_ffmpeg_progress, args=(process,))
            progress_thread.daemon = True
            progress_thread.start()

            total_frames = self.video_info.get('frame_count', 1)
            self.progress_queue.put({"total": total_frames, "stage": "Extraction"})

            process.wait()

        except Exception as e:
            if process: process.terminate()
            logger.error(f"FFmpeg process encountered an unhandled exception: {e}")
            raise RuntimeError(f"FFmpeg process failed unexpectedly. Details: {e}")
        finally:
            # FIX: Implement more robust process termination on cancellation.
            if process and self.cancel_event.is_set() and process.poll() is None:
                self.progress_queue.put({"log": "[INFO] Terminating FFmpeg process..."})
                try:
                    process.terminate()
                    process.wait(timeout=2) # Grace period
                except subprocess.TimeoutExpired:
                    self.progress_queue.put({"log": "[WARNING] FFmpeg did not terminate gracefully, killing."})
                    process.kill()
                logger.warning("FFmpeg process terminated due to cancellation.")
                
            if progress_thread and progress_thread.is_alive():
                progress_thread.join(timeout=1)
            if stderr_handle and stderr_handle != subprocess.DEVNULL:
                stderr_handle.close()
        
        if use_showinfo:
            frame_map_list = []
            try:
                with open(log_file_path, 'r') as f:
                    for line in f:
                        if 'Parsed_showinfo' in line and ' n:' in line:
                            match = re.search(r' n:\s*(\d+)', line)
                            if match:
                                frame_map_list.append(int(match.group(1)))
                
                frame_map_path = self.output_dir / "frame_map.json"
                with open(frame_map_path, 'w') as f:
                    json.dump(frame_map_list, f)
            except FileNotFoundError:
                self.progress_queue.put({"log": "[WARNING] ffmpeg log not found. Could not create frame map."})
            finally:
                log_file_path.unlink(missing_ok=True)
        
        if process.returncode != 0 and not self.cancel_event.is_set():
            logger.error(f"FFmpeg failed with return code {process.returncode}")
            raise RuntimeError("FFmpeg failed. Check logs for details.")

class AnalysisPipeline:
    def __init__(self, params: AnalysisParameters, progress_queue: Queue, cancel_event: threading.Event, video_path: str = ""):
        self.params = params
        self.progress_queue = progress_queue
        self.cancel_event = cancel_event
        self.video_path = video_path
        self.output_dir = Path(params.output_folder)
        self.metadata_path = self.output_dir / "metadata.jsonl"
        self.reference_embedding = None
        self.write_lock = threading.Lock()
        self.gpu_lock = threading.Lock()
        self.face_analyzer = None
        self.stats = AnalysisStats()
        self.last_stats_ts = 0.0
        self.mask_metadata = {}
        self.is_cpu_only = not get_feature_status()['cuda_available']

    def run(self):
        try:
            self.progress_queue.put({"output_dir": str(self.output_dir)})
            config_hash = self._get_config_hash()
            if self.params.resume and self._check_resume_compatibility(config_hash):
                self.progress_queue.put({"log": f"[INFO] Resuming using compatible metadata: {self.metadata_path.name}"})
                return {"done": True, "metadata_path": str(self.metadata_path), "output_dir": str(self.output_dir)}
            
            with self.metadata_path.open('w') as f:
                header = {"config_hash": config_hash, "params": {k:v for k,v in asdict(self.params).items() if k not in ['source_path', 'output_folder']}}
                f.write(json.dumps(header) + '\n')

            if self.params.enable_face_filter or self.params.pre_filter_face_present or self.params.enable_subject_mask:
                self._initialize_face_analyzer()
            
            if self.params.enable_face_filter or (self.params.enable_subject_mask and self.params.face_ref_img_path):
                self._process_reference_face()
            
            if self.params.enable_subject_mask:
                if self.is_cpu_only:
                    self.progress_queue.put({"log": "[WARNING] Subject masking is disabled in CPU-only mode."})
                elif not self.video_path:
                    raise ValueError("A video file path is required for subject masking but was not provided.")
                else:
                    frame_map = self._create_frame_map()
                    masker = SubjectMasker(
                        self.params, self.progress_queue, self.cancel_event, 
                        frame_map, face_analyzer=self.face_analyzer, reference_embedding=self.reference_embedding
                    )
                    self.mask_metadata = masker.run(self.video_path, str(self.output_dir))

            config.QUALITY_WEIGHTS = self.params.quality_weights
            self._run_frame_processing()

            if self.cancel_event.is_set():
                self.progress_queue.put({"log": "[INFO] Analysis cancelled."})
                return {"log": "[INFO] Analysis cancelled."}
            
            self.progress_queue.put({"log": "[SUCCESS] Analysis complete. Go to 'Filtering & Export' tab."})
            return {"done": True, "metadata_path": str(self.metadata_path), "output_dir": str(self.output_dir)}
        except Exception as e:
            logger.exception("Error in analysis pipeline.")
            return {"error": str(e)}
            
    def _create_frame_map(self):
        self.progress_queue.put({"log": "[INFO] Loading frame map..."})
        frame_map_path = self.output_dir / "frame_map.json"
        frame_map = {}

        if not frame_map_path.exists():
            self.progress_queue.put({"log": "[WARNING] frame_map.json not found. Creating map from filenames. This may be inaccurate if 'all' frames weren't extracted."})
            image_files = sorted(list(self.output_dir.glob("frame_*.png")) + list(self.output_dir.glob("frame_*.jpg")))
            for i, f in enumerate(image_files):
                frame_map[i] = f.name
            return frame_map
        
        try:
            with open(frame_map_path, 'r') as f:
                frame_map_list = json.load(f)
        except json.JSONDecodeError as e:
            self.progress_queue.put({"log": f"[ERROR] Failed to parse frame_map.json: {e}. Reverting to filename-based mapping."})
            image_files = sorted(list(self.output_dir.glob("frame_*.png")) + list(self.output_dir.glob("frame_*.jpg")))
            for i, f in enumerate(image_files):
                frame_map[i] = f.name
            return frame_map

        ext = 'png' if self.params.use_png else 'jpg'
        for i, original_frame_num in enumerate(frame_map_list):
            filename = f"frame_{i+1:06d}.{ext}"
            frame_map[original_frame_num] = filename
            
        self.progress_queue.put({"log": "[SUCCESS] Frame map loaded."})
        return frame_map

    def _get_config_hash(self):
        d = asdict(self.params)
        relevant_params = {k: d.get(k) for k in [
            'enable_subject_mask', 'scene_detect', 'enable_face_filter', 'face_model_name',
            'quality_weights', 'dam4sam_model_name', 'pre_filter_quality_enabled',
            'pre_quality_thresh', 'min_face_confidence'
        ]}
        return hashlib.sha1(json.dumps(relevant_params, sort_keys=True).encode()).hexdigest()

    def _check_resume_compatibility(self, current_hash):
        if not self.metadata_path.exists() or self.metadata_path.stat().st_size == 0: return False
        with self.metadata_path.open('r') as f:
            try:
                header = json.loads(f.readline())
                if header.get("config_hash") == current_hash: return True
                self.progress_queue.put({"log": "[WARNING] Config has changed. Re-running analysis for compatibility."})
                return False
            except (json.JSONDecodeError, IndexError):
                self.progress_queue.put({"log": "[WARNING] Invalid metadata header. Re-running analysis."})
                return False

    def _initialize_face_analyzer(self):
        if not FaceAnalysis:
            raise ImportError("insightface library is not installed. Please install it for face analysis.")
        
        if self.is_cpu_only:
            self.progress_queue.put({"log": "[WARNING] Face analysis is disabled in CPU-only mode."})
            return

        self.progress_queue.put({"log": f"[INFO] Loading face model: {self.params.face_model_name}"})
        try:
            self.face_analyzer = FaceAnalysis(
                name=self.params.face_model_name,
                root=str(config.BASE_DIR / 'models'),
                providers=['CUDAExecutionProvider']
            )
            self.face_analyzer.prepare(ctx_id=0, det_size=(640, 640))
            self.progress_queue.put({"log": "[SUCCESS] Face model loaded."})
        except Exception as e:
            logger.error(f"Failed to initialize FaceAnalysis: {e}", exc_info=True)
            raise RuntimeError(f"Could not initialize face analysis model. Check if models are downloaded and CUDA is correctly configured. Error: {e}")

    def _process_reference_face(self):
        if not self.params.face_ref_img_path:
            self.progress_queue.put({"log": "[WARNING] Reference face processing skipped: no image path provided."})
            return
        if not self.face_analyzer:
            self.progress_queue.put({"log": "[WARNING] Reference face processing skipped: Face analyzer not available."})
            return
            
        ref_path = Path(self.params.face_ref_img_path)
        if not ref_path.exists(): raise FileNotFoundError(f"Reference face image not found: {ref_path}")
        self.progress_queue.put({"log": "[INFO] Processing reference face image..."})
        ref_img = cv2.imread(str(ref_path))
        if ref_img is None: raise ValueError("Could not read reference image.")
        ref_faces = self.face_analyzer.get(ref_img)
        if not ref_faces: raise ValueError("No face found in reference image.")
        self.reference_embedding = sorted(ref_faces, key=lambda x: x.det_score, reverse=True)[0].normed_embedding
        self.progress_queue.put({"log": "[SUCCESS] Reference face processed."})

    def _run_frame_processing(self):
        image_files = sorted(list(self.output_dir.glob("frame_*.png")) + list(self.output_dir.glob("frame_*.jpg")))
        self.progress_queue.put({"total": len(image_files), "stage": "Analysis"})

        num_workers = 1 if self.params.disable_parallel else (os.cpu_count() or 4)
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(self._process_single_frame, path, i) for i, path in enumerate(image_files)]
            for f in futures: f.result()

    def _process_single_frame(self, image_path, frame_num):
        if self.cancel_event.is_set(): return
        
        try:
            image_data = cv2.imread(str(image_path))
            if image_data is None: 
                raise ValueError("Could not read image file.")

            frame = Frame(image_data, frame_num)
            mask_meta = self.mask_metadata.get(image_path.name, {})
            mask = cv2.imread(mask_meta["mask_path"], cv2.IMREAD_GRAYSCALE) if mask_meta.get("mask_path") else None
            
            frame.calculate_quality_metrics(mask=mask)

            video_area = frame.image_data.shape[0] * frame.image_data.shape[1]
            keep, status, max_conf = check_frame_passes_filters(frame, self.params, video_area, self.face_analyzer, self.gpu_lock, mask_meta=mask_meta)
            frame.max_face_confidence = max_conf
            
            self.stats.processed += 1
            if keep: self.stats.kept += 1
            else:
                self.stats.discarded += 1
                if "Face" in status: self.stats.disc_face += 1
                elif "Mask" in status: self.stats.disc_mask += 1
                else: self.stats.disc_quality += 1
            
            now = time.time()
            if now - self.last_stats_ts > 0.25:
                self.progress_queue.put({"stats": asdict(self.stats)})
                self.last_stats_ts = now

            if not keep:
                self.progress_queue.put({"progress": 1})
                image_path.unlink()
                if mask is not None and mask_meta.get("mask_path"): Path(mask_meta["mask_path"]).unlink(missing_ok=True)
                meta = {"filename": image_path.name, "error": f"pre-filtered: {status}", "metrics": asdict(frame.metrics) if frame.metrics else {}}
                with self.write_lock, self.metadata_path.open('a') as f:
                    f.write(json.dumps(meta) + '\n')
                return

            if self.params.enable_face_filter and self.reference_embedding is not None and self.face_analyzer:
                self._analyze_face_similarity(frame)
            
            meta = {"filename": image_path.name, "face_sim": frame.face_similarity_score, "face_conf": frame.max_face_confidence,
                    "metrics": asdict(frame.metrics) if frame.metrics else {}, "error": frame.error}
            meta.update(mask_meta)
            
            with self.write_lock, self.metadata_path.open('a') as f:
                f.write(json.dumps(meta) + '\n')
            self.progress_queue.put({"progress": 1})

        except Exception as e:
            logger.error(f"Critical error processing frame {image_path.name}: {e}", exc_info=True)
            meta = {"filename": image_path.name, "error": f"processing_failed: {str(e)}"}
            with self.write_lock, self.metadata_path.open('a') as f:
                f.write(json.dumps(meta) + '\n')
            self.progress_queue.put({"progress": 1, "stats": asdict(self.stats)})


    def _analyze_face_similarity(self, frame):
        try:
            with self.gpu_lock: faces = self.face_analyzer.get(frame.image_data)
            if faces:
                best_face = max(faces, key=lambda x: x.det_score)
                distance = 1 - np.dot(best_face.normed_embedding, self.reference_embedding)
                frame.face_similarity_score = 1.0 - float(distance)
        except Exception as e:
            frame.error = f"Face similarity failed: {e}"
            logger.warning(f"Frame {frame.frame_number}: {frame.error}")

# --- Gradio UI & Event Handlers ---
class AppUI:
    def __init__(self):
        self.components = {}
        self.cancel_event = threading.Event()
        self.param_to_elem_id_map = self._create_param_map()
        self.last_task_result = {}
        self.feature_status = {}

    def build_ui(self):
        """Builds the Gradio UI with a reorganized two-tab layout."""
        self.feature_status = get_feature_status()
        css = """.gradio-container { max-width: 1280px !important; margin: auto !important; }"""
        with gr.Blocks(theme=gr.themes.Default(primary_hue="blue"), css=css) as demo:
            gr.Markdown("# Advanced Frame Extractor & Filter")
            if not self.feature_status['cuda_available']:
                gr.Warning("No CUDA-enabled GPU detected. Running in CPU-only mode. "
                           "Face Analysis and Subject Masking features will be disabled.")
            with gr.Tabs():
                self._create_setup_tab()
                self._create_filtering_tab()
            self._create_event_handlers()
        return demo

    def _create_param_map(self):
        return {
            "source_path": "source_input", "method": "method_input", "interval": "interval_input",
            "max_resolution": "max_resolution", "fast_scene": "fast_scene_input", "use_png": "use_png_input",
            "disable_parallel": "disable_parallel_input", "resume": "resume_input",
            "enable_face_filter": "enable_face_filter_input", "face_ref_img_path": "face_ref_img_path_input",
            "face_model_name": "face_model_name_input", "min_face_confidence": "min_face_confidence_input",
            "min_face_area": "min_face_area_input", "pre_filter_quality_enabled": "pre_filter_quality_enabled_input",
            "pre_filter_mode": "pre_filter_mode_toggle", "pre_quality_thresh": "pre_quality_thresh_input",
            "pre_filter_face_present": "pre_filter_face_present_input",
            "enable_subject_mask": "enable_subject_mask_input",
            "dam4sam_model_name": "dam4sam_model_name_input",
            "scene_detect": "scene_detect_input",
        }

    def _create_setup_tab(self):
        with gr.Tab("1. Setup & Run"):
            with gr.Row():
                with gr.Column(scale=3):
                    gr.Markdown("### 1. Video Source")
                    self.components['source_input'] = gr.Textbox(
                        label="Video URL or Local Path", lines=3,
                        placeholder="Enter YouTube URL or local video file path",
                        elem_id="source_input",
                        info="YouTube downloads require 'yt-dlp'. Status: " + ("Available" if self.feature_status['youtube_dl'] else "Not Installed")
                    )
                    self.components['upload_video_input'] = gr.File(label="Or Upload Video", file_types=["video"], type="filepath", elem_id="upload_video_input")
                    
                    with gr.Accordion("2. Extraction Settings", open=True):
                        with gr.Row():
                            method_choices = ["keyframes", "interval", "all"]
                            if self.feature_status['scene_detection']:
                                method_choices.insert(2, "scene")
                            
                            self.components['method_input'] = gr.Dropdown(
                                method_choices, value=config.UI_DEFAULTS["method"],
                                label="Extraction Method", elem_id="method_input",
                                info="Scene detection requires 'scenedetect'. Status: " + ("Available" if self.feature_status['scene_detection'] else "Not Installed")
                            )
                            self.components['interval_input'] = gr.Number(label="Interval (s)", value=config.UI_DEFAULTS["interval"], visible=False, elem_id="interval_input")
                            self.components['fast_scene_input'] = gr.Checkbox(label="Fast Scene Detect", value=config.UI_DEFAULTS["fast_scene"], visible=False, elem_id="fast_scene_input")
                            self.components['max_resolution'] = gr.Dropdown(["maximum available", "2160", "1080", "720", "480", "360"], value=config.UI_DEFAULTS["max_resolution"], label="DL Res (URL)", elem_id="max_resolution")
                    
                    with gr.Accordion("3. Analysis & Pre-Filtering Settings", open=False):
                        self._create_pre_filtering_ui()

                with gr.Column(scale=2):
                    with gr.Accordion("Face Similarity", open=False):
                        is_face_gpu_ready = self.feature_status['face_analysis'] and self.feature_status['cuda_available']
                        face_info = "Available" if is_face_gpu_ready else ("'insightface' not installed" if not self.feature_status['face_analysis'] else "CUDA not available")
                        
                        self.components['enable_face_filter_input'] = gr.Checkbox(
                            label="Enable Face Similarity",
                            info=f"Compares faces to a reference image. Status: {face_info}",
                            elem_id="enable_face_filter_input",
                            interactive=is_face_gpu_ready
                        )
                        with gr.Group(visible=False) as self.components['face_options_group']:
                            self.components['face_model_name_input'] = gr.Dropdown(["buffalo_l", "buffalo_s", "buffalo_m", "antelopev2"], value=config.UI_DEFAULTS["face_model_name"], label="Model", elem_id="face_model_name_input")
                            self.components['face_ref_img_path_input'] = gr.Textbox(label="Reference Image Path", elem_id="face_ref_img_path_input")
                            self.components['face_ref_img_upload_input'] = gr.File(label="Or Upload Reference", file_types=["image"], type="filepath", elem_id="face_ref_img_upload_input")
                    
                    with gr.Accordion("Subject Masking", open=False):
                        masking_status = "Available"
                        if not self.feature_status['masking_libs_installed']:
                            masking_status = "'torch', 'Pillow', or SAM2 dependencies not installed"
                        elif not self.feature_status['cuda_available']:
                            masking_status = "CUDA not available"

                        self.components['enable_subject_mask_input'] = gr.Checkbox(
                            label="Enable Subject-Only Metrics",
                            info=f"Requires a CUDA GPU and dependencies. Status: {masking_status}",
                            elem_id="enable_subject_mask_input",
                            interactive=self.feature_status['masking']
                        )
                        with gr.Group(visible=False) as self.components['masking_options_group']:
                            with gr.Group(visible=True) as self.components['dam4sam_options_group']:
                                self.components['dam4sam_model_name_input'] = gr.Dropdown(['sam2.1', 'sam21pp-L'], value=config.UI_DEFAULTS["dam4sam_model_name"], label="DAM4SAM Model", elem_id="dam4sam_model_name_input")
                            
                            self.components['scene_detect_input'] = gr.Checkbox(
                                label="Scene Detection for Masking",
                                value=config.UI_DEFAULTS["scene_detect"],
                                elem_id="scene_detect_input",
                                interactive=self.feature_status['scene_detection'],
                                info="Status: " + ("Available" if self.feature_status['scene_detection'] else "'scenedetect' not installed")
                            )

                    with gr.Accordion("Configuration & Advanced Settings", open=False):
                        with gr.Group():
                            self.components['resume_input'] = gr.Checkbox(label="Resume/Use Cache", value=config.UI_DEFAULTS["resume"], elem_id="resume_input")
                            self.components['use_png_input'] = gr.Checkbox(label="Save as PNG (slower, higher quality)", value=config.UI_DEFAULTS["use_png"], elem_id="use_png_input")
                            self.components['disable_parallel_input'] = gr.Checkbox(label="Disable Parallelism (for low memory)", value=config.UI_DEFAULTS["disable_parallel"], elem_id="disable_parallel_input")
                        self._create_config_presets_ui()

            with gr.Row():
                self.components['start_button'] = gr.Button("Start Processing", variant="primary")
                self.components['stop_button'] = gr.Button("Stop", variant="stop", interactive=False)
            
            self.components['process_status'] = gr.Markdown("", elem_id="process_status")
            self.components['process_output_log'] = gr.Textbox(label="Logs", lines=8, interactive=False, autoscroll=True)


    def _create_filtering_tab(self):
        with gr.Tab("2. Filter & Export", interactive=False) as self.components['filtering_tab']:
            self.components['analysis_required_message'] = gr.Markdown("**Run a processing job in the 'Setup & Run' tab to enable filtering.**", visible=True)
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("## Live Filtering")
                    self.components['filter_mode_toggle'] = gr.Radio(list(config.FILTER_MODES.values()), value=config.FILTER_MODES["OVERALL"], label="Filter By")
                    with gr.Group() as self.components['overall_quality_group']:
                        self.components['quality_filter_slider'] = gr.Slider(0, 100, config.UI_DEFAULTS["quality_thresh"], label="Min Quality")
                    with gr.Group(visible=False) as self.components['individual_metrics_group']:
                        self.components['filter_metric_sliders'] = [gr.Slider(0, 100, label=f"Min {k.replace('_',' ').capitalize()}") for k in config.QUALITY_METRICS]
                    self.components['face_filter_slider'] = gr.Slider(0, 1.0, 0.5, label="Min Face Similarity", step=0.01)

                with gr.Column(scale=3):
                    with gr.Row():
                        self.components['filter_stats'] = gr.Textbox(label="Filter Results", lines=4, interactive=False)
                        self.components['export_button'] = gr.Button("Export Kept Frames", variant="primary")
                    self.components['results_gallery'] = gr.Gallery(label="Kept Frames Preview (Max 100)", columns=8, allow_preview=True)
            
            self.components['frame_metadata_path_state'] = gr.State("")
            self.components['analysis_path_state'] = gr.State("")
            
    def _create_pre_filtering_ui(self):
        with gr.Row():
            with gr.Column():
                self.components['pre_filter_quality_enabled_input'] = gr.Checkbox(label="Quality Pre-Filter", elem_id="pre_filter_quality_enabled_input")
                with gr.Group(visible=False) as self.components['pre_quality_filter_group']:
                    self.components['pre_filter_mode_toggle'] = gr.Radio(list(config.FILTER_MODES.values()), value=config.UI_DEFAULTS["pre_filter_mode"], label="Mode", elem_id="pre_filter_mode_toggle")
                    self.components['pre_quality_thresh_input'] = gr.Slider(0, 100, config.UI_DEFAULTS["pre_quality_thresh"], label="Min Overall Quality", elem_id="pre_quality_thresh_input")
                    with gr.Accordion("Customize Quality Weights", open=False):
                        # FIX: Create sliders in a fixed order based on config.QUALITY_METRICS
                        self.components['weight_sliders'] = [gr.Slider(0, 100, config.QUALITY_WEIGHTS[k], step=1, label=k.capitalize(), elem_id=f"weight_{k}") for k in config.QUALITY_METRICS]
                    with gr.Group(visible=False) as self.components['pre_individual_metrics_group']:
                        self.components['pre_metric_sliders'] = [gr.Slider(0, 100, step=0.5, label=f"Min {k.capitalize()}", elem_id=f"pre_{k}_thresh_input") for k in config.QUALITY_METRICS]
            with gr.Column():
                is_face_gpu_ready = self.feature_status['face_analysis'] and self.feature_status['cuda_available']
                self.components['pre_filter_face_present_input'] = gr.Checkbox(
                    label="Face Presence Pre-Filter",
                    info="Requires insightface and CUDA.",
                    elem_id="pre_filter_face_present_input",
                    interactive=is_face_gpu_ready
                )
                with gr.Group(visible=False) as self.components['pre_face_filter_group']:
                    self.components['min_face_confidence_input'] = gr.Slider(0.0, 1.0, config.UI_DEFAULTS["min_face_confidence"], step=0.01, label="Min Face Confidence", elem_id="min_face_confidence_input")
                    self.components['min_face_area_input'] = gr.Slider(0.1, 50.0, config.UI_DEFAULTS["min_face_area"], step=0.1, label="Min Face Area (%)", elem_id="min_face_area_input")
    
    def _create_config_presets_ui(self):
        with gr.Group():
            self.components['config_status'] = gr.Textbox(label="Status", interactive=False, lines=1)
            with gr.Row():
                self.components['config_dropdown'] = gr.Dropdown(label="Select Config", choices=[f.stem for f in config.CONFIGS_DIR.glob("*.json")])
                self.components['load_button'] = gr.Button("Load")
                self.components['delete_button'] = gr.Button("Delete", variant="stop")
            with gr.Row():
                self.components['config_name_input'] = gr.Textbox(label="New Config Name")
                self.components['save_button'] = gr.Button("Save")

    def _create_event_handlers(self):
        self.components['method_input'].change(lambda m: (gr.update(visible=m=='interval'), gr.update(visible=m=='scene')), self.components['method_input'], [self.components['interval_input'], self.components['fast_scene_input']])
        self.components['enable_face_filter_input'].change(lambda e: gr.update(visible=e), self.components['enable_face_filter_input'], self.components['face_options_group'])
        self.components['enable_subject_mask_input'].change(lambda e: gr.update(visible=e), self.components['enable_subject_mask_input'], self.components['masking_options_group'])
        self.components['pre_filter_quality_enabled_input'].change(lambda x: gr.update(visible=x), self.components['pre_filter_quality_enabled_input'], self.components['pre_quality_filter_group'])
        self.components['pre_filter_face_present_input'].change(lambda x: gr.update(visible=x), self.components['pre_filter_face_present_input'], self.components['pre_face_filter_group'])
        
        self.components['pre_filter_mode_toggle'].change(
            lambda m: (gr.update(visible=m == config.FILTER_MODES["OVERALL"]), gr.update(visible=m != config.FILTER_MODES["OVERALL"])),
            self.components['pre_filter_mode_toggle'],
            [self.components['pre_quality_thresh_input'], self.components['pre_individual_metrics_group']]
        )
        self.components['filter_mode_toggle'].change(
            lambda m: (gr.update(visible=m == config.FILTER_MODES["OVERALL"]), gr.update(visible=m != config.FILTER_MODES["OVERALL"])),
            self.components['filter_mode_toggle'],
            [self.components['overall_quality_group'], self.components['individual_metrics_group']]
        )
        
        self._setup_pipeline_handler()
        self._setup_filtering_handlers()
        self._setup_config_handlers()

    def _get_all_param_components(self):
        components = []
        for elem_id in self.param_to_elem_id_map.values():
            if elem_id in self.components:
                components.append(self.components[elem_id])
        components.extend(self.components['weight_sliders'])
        components.extend(self.components['pre_metric_sliders'])
        return components

    def _setup_pipeline_handler(self):
        inputs = self._get_all_param_components()
        inputs.insert(0, self.components['face_ref_img_upload_input'])
        inputs.insert(0, self.components['upload_video_input'])
        
        outputs = [self.components['start_button'], self.components['stop_button'], self.components['process_output_log'], self.components['process_status'],
                   self.components['filtering_tab'], self.components['results_gallery'], self.components['filter_stats'],
                   self.components['frame_metadata_path_state'], self.components['analysis_path_state'], 
                   self.components['analysis_required_message']]

        self.components['start_button'].click(self.run_full_pipeline_wrapper, inputs, outputs)
        self.components['stop_button'].click(lambda: self.cancel_event.set(), [], []).then(lambda: gr.update(interactive=False), None, self.components['stop_button'])

    def _setup_filtering_handlers(self):
        filter_inputs = [self.components['frame_metadata_path_state'], self.components['analysis_path_state'], 
                         self.components['quality_filter_slider'], self.components['face_filter_slider'], 
                         self.components['filter_mode_toggle']] + self.components['filter_metric_sliders']
        filter_outputs = [self.components['results_gallery'], self.components['filter_stats']]
        for c in filter_inputs[2:]: c.change(self.apply_gallery_filters, filter_inputs, filter_outputs)
        self.components['filtering_tab'].select(self.apply_gallery_filters, filter_inputs, filter_outputs)
        self.components['export_button'].click(self.export_kept_frames, filter_inputs, self.components['filter_stats'])
    
    def _setup_config_handlers(self):
        all_param_controls = self._get_all_param_components()
        
        save_inputs = [self.components['config_name_input']] + all_param_controls
        self.components['save_button'].click(self.save_config, save_inputs, [self.components['config_status'], self.components['config_dropdown']])
        
        load_outputs = all_param_controls + [self.components['config_status']]
        self.components['load_button'].click(self.load_config, self.components['config_dropdown'], load_outputs)
        
        self.components['delete_button'].click(self.delete_config, self.components['config_dropdown'], [self.components['config_status'], self.components['config_dropdown']])

    def run_full_pipeline_wrapper(self, upload_video, face_ref_upload, *args):
        yield {
            self.components['start_button']: gr.update(interactive=False), 
            self.components['stop_button']: gr.update(interactive=True), 
            self.components['process_output_log']: "",
            self.components['process_status']: "Starting..."
        }
        self.cancel_event.clear()

        # FIX: Rebuild parameter mapping robustly using config.QUALITY_METRICS order
        params_dict, arg_idx = {}, 0
        
        for key in self.param_to_elem_id_map.keys():
            params_dict[key] = args[arg_idx]
            arg_idx += 1
        
        if upload_video: params_dict['source_path'] = upload_video
        if face_ref_upload: params_dict['face_ref_img_path'] = face_ref_upload
        
        params_dict['quality_weights'] = {k: args[arg_idx + i] for i, k in enumerate(config.QUALITY_METRICS)}
        arg_idx += len(config.QUALITY_METRICS)
        params_dict['thresholds'] = {k: args[arg_idx + i] for i, k in enumerate(config.QUALITY_METRICS)}
        
        params = AnalysisParameters(**params_dict)

        # --- Pre-flight checks ---
        if not params.source_path:
            yield {self.components['process_output_log']: "[ERROR] Video source is required.", self.components['start_button']: gr.update(interactive=True), self.components['stop_button']: gr.update(interactive=False)}
            return
        
        is_url = re.match(r'https?://|youtu\.be', str(params.source_path))
        if not is_url and not Path(params.source_path).exists():
            yield {self.components['process_output_log']: f"[ERROR] Local file path not found: {params.source_path}", self.components['start_button']: gr.update(interactive=True), self.components['stop_button']: gr.update(interactive=False)}
            return

        if params.enable_face_filter and (not params.face_ref_img_path or not Path(params.face_ref_img_path).is_file()):
            yield {self.components['process_output_log']: f"[ERROR] Reference face image is required, but the path is invalid: {params.face_ref_img_path}", self.components['start_button']: gr.update(interactive=True), self.components['stop_button']: gr.update(interactive=False)}
            return
            
        if params.scene_detect and not self.feature_status['scene_detection']:
            yield {self.components['process_output_log']: f"[ERROR] Scene detection was enabled for masking, but PySceneDetect is not installed.", self.components['start_button']: gr.update(interactive=True), self.components['stop_button']: gr.update(interactive=False)}
            return

        # --- Pipeline Execution ---
        progress_queue = Queue()
        extraction_pipeline = ExtractionPipeline(params, progress_queue, self.cancel_event)
        yield from self._run_task(extraction_pipeline.run, self.components['process_output_log'], status_box=self.components['process_status'])
        
        extraction_result = self.last_task_result
        if not extraction_result.get("done") or self.cancel_event.is_set():
            yield {self.components['start_button']: gr.update(interactive=True), self.components['stop_button']: gr.update(interactive=False)}
            return

        output_dir = extraction_result["output_dir"]
        video_path = extraction_result.get("video_path", "")
        params.output_folder = output_dir

        analysis_pipeline = AnalysisPipeline(params, progress_queue, self.cancel_event, video_path=video_path)
        yield from self._run_task(analysis_pipeline.run, self.components['process_output_log'], status_box=self.components['process_status'])

        analysis_result = self.last_task_result
        if analysis_result.get("done") and not self.cancel_event.is_set():
            metadata_path = analysis_result["metadata_path"]
            # FIX: Replace magic number '5' with dynamic length of quality metrics
            num_metrics = len(config.QUALITY_METRICS)
            gallery, stats = self.apply_gallery_filters(metadata_path, output_dir, config.UI_DEFAULTS['quality_thresh'], 0.5, config.FILTER_MODES['OVERALL'], *([0]*num_metrics))
            yield {
                self.components['filtering_tab']: gr.update(interactive=True), 
                self.components['analysis_required_message']: gr.update(visible=False),
                self.components['frame_metadata_path_state']: metadata_path, 
                self.components['analysis_path_state']: output_dir,
                self.components['results_gallery']: gallery, 
                self.components['filter_stats']: stats
            }
        
        yield {self.components['start_button']: gr.update(interactive=True), self.components['stop_button']: gr.update(interactive=False)}


    def _run_task(self, task_func, log_box, status_box=None):
        progress_queue = task_func.__self__.progress_queue
        log_buffer, processed_count, total_frames = [], 0, 1
        start_ts, last_yield_ts, last_stats, current_stage = time.time(), 0.0, {}, "Initializing"
        
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(task_func)
            while future.running():
                if self.cancel_event.is_set():
                    if hasattr(task_func.__self__, 'cancel_event'):
                         task_func.__self__.cancel_event.set()
                    break

                try:
                    msg = progress_queue.get(timeout=0.1)
                    if "log" in msg: log_buffer.append(msg["log"])
                    if "error" in msg: log_buffer.append(f"[ERROR] {msg['error']}")
                    if "stage" in msg: 
                        current_stage, processed_count, start_ts = msg["stage"], 0, time.time()
                    if "total" in msg: total_frames = msg["total"] or 1
                    if "progress" in msg: processed_count += msg["progress"]
                    if "progress_abs" in msg: processed_count = msg["progress_abs"]
                    if "stats" in msg: last_stats = msg["stats"]
                    
                    now = time.time()
                    if now - last_yield_ts > 0.25:
                        ratio = processed_count / max(total_frames, 1)
                        elapsed = max(now - start_ts, 1e-6)
                        fps = processed_count / elapsed if elapsed > 0 else 0
                        eta_s = (total_frames - processed_count) / fps if fps > 0 else 0
                        mm, ss = divmod(int(eta_s), 60)
                        
                        kept, disc = last_stats.get("kept", 0), last_stats.get("discarded", 0)
                        dq, df, dm = last_stats.get("disc_quality", 0), last_stats.get("disc_face", 0), last_stats.get("disc_mask", 0)
                        
                        status_line = f"**{current_stage}:** {processed_count}/{total_frames} ({ratio:.1%}) &nbsp; | &nbsp; Kept: {kept}, Discarded: {disc} (Q:{dq}, F:{df}, M:{dm}) &nbsp; | &nbsp; {fps:.1f} FPS &nbsp; | &nbsp; ETA: {mm:02d}:{ss:02d}"
                        updates = {log_box: "\n".join(log_buffer)}
                        if status_box: updates[status_box] = status_line
                        yield updates
                        last_yield_ts = now

                except Empty:
                    pass

        self.last_task_result = future.result() or {}
        if "log" in self.last_task_result: log_buffer.append(self.last_task_result["log"])
        if "error" in self.last_task_result: log_buffer.append(f"[ERROR] {self.last_task_result['error']}")
        
        final_updates = {log_box: "\n".join(log_buffer)}
        if status_box:
            status_text = "⏹️ Operation cancelled." if self.cancel_event.is_set() else (f"❌ Error: {self.last_task_result.get('error')}" if self.last_task_result.get('error') else "✅ Operation complete.")
            final_updates[status_box] = status_text
        yield final_updates

    @staticmethod
    def apply_gallery_filters(metadata_path, output_dir, quality_thresh, face_thresh, filter_mode, *ind_thresh):
        if not metadata_path: return [], "Run analysis to see results."
        kept_frames, total_kept, _, total_frames = AppUI._load_and_filter_metadata(metadata_path, quality_thresh, face_thresh, filter_mode, ind_thresh)
        preview = [str(Path(output_dir) / f['filename']) for f in kept_frames[:100] if (Path(output_dir) / f['filename']).exists()]
        return preview, f"Kept: {total_kept} / {total_frames} frames (previewing {len(preview)})"

    @staticmethod
    def export_kept_frames(metadata_path, output_dir, quality_thresh, face_thresh, filter_mode, *ind_thresh):
        if not metadata_path: return "No metadata to export."
        try:
            export_dir = Path(output_dir).parent / f"{Path(output_dir).name}_exported_{datetime.now():%Y%m%d_%H%M%S}"
            export_dir.mkdir(exist_ok=True)
            kept_frames, total_kept, _, total_frames = AppUI._load_and_filter_metadata(metadata_path, quality_thresh, face_thresh, filter_mode, ind_thresh)
            
            copied_count = 0
            # FIX: Ensure deterministic export order by sorting
            for frame in sorted(kept_frames, key=lambda x: x['filename']):
                src = Path(output_dir) / frame['filename']
                if src.exists():
                    shutil.copy2(src, export_dir)
                    copied_count += 1
            
            if copied_count != total_kept:
                 return f"Exported {copied_count}/{total_kept} frames to '{export_dir.name}'. Some source files may have been missing."
            
            return f"Exported {total_kept}/{total_frames} frames to '{export_dir.name}'"
        except (IOError, OSError) as e:
            logger.error(f"Failed to export frames: {e}", exc_info=True)
            return f"Error during export: {e}"
    
    @staticmethod
    def _load_and_filter_metadata(path, q_thresh, f_thresh, mode, ind_thresh):
        kept, reasons, total = [], {m: 0 for m in config.QUALITY_METRICS + ['quality', 'face', 'error']}, 0
        p = Path(path)
        if not p.exists(): return [], 0, reasons, 0
        with p.open('r') as f:
            try:
                next(f)
            except StopIteration:
                return [], 0, reasons, 0
            for line in f:
                total += 1
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    reasons['error'] += 1
                    continue
                    
                fail_reason = None

                if "processing_failed" in data.get("error", ""):
                    reasons['error'] += 1
                    continue
                
                if mode == config.FILTER_MODES["OVERALL"]:
                    if data.get('metrics', {}).get('quality_score', 100) < q_thresh:
                        fail_reason = 'quality'
                else:
                    for i, k in enumerate(config.QUALITY_METRICS):
                        if data.get('metrics', {}).get(f'{k}_score', 100) < ind_thresh[i]:
                            fail_reason = k
                            break
                
                if not fail_reason and data.get('face_sim') is not None and data['face_sim'] < f_thresh:
                    fail_reason = 'face'

                if fail_reason:
                    reasons[fail_reason] = reasons.get(fail_reason, 0) + 1
                else:
                    kept.append(data)

        return kept, len(kept), reasons, total

    def save_config(self, name, *values):
        if not name: return "Error: Config name required.", gr.update()
        name = sanitize_filename(name)
        
        settings = {}
        value_idx = 0
        
        for param, elem_id in self.param_to_elem_id_map.items():
            if elem_id in self.components:
                settings[param] = values[value_idx]
                value_idx += 1
        
        # FIX: Save slider values based on the ordered list
        for i, k in enumerate(config.QUALITY_METRICS):
            slider = self.components['weight_sliders'][i]
            settings[slider.elem_id] = values[value_idx]
            value_idx += 1
            
        for i, k in enumerate(config.QUALITY_METRICS):
            slider = self.components['pre_metric_sliders'][i]
            settings[slider.elem_id] = values[value_idx]
            value_idx += 1
            
        with (config.CONFIGS_DIR / f"{name}.json").open('w') as f:
            json.dump(settings, f, indent=2)
            
        return f"Config '{name}' saved.", gr.update(choices=[f.stem for f in config.CONFIGS_DIR.glob("*.json")])

    def load_config(self, name):
        all_param_components = self._get_all_param_components()
        output_components = all_param_components + [self.components['config_status']]
        
        updates = {comp: gr.update() for comp in output_components}
        
        if not name or not (config_path := config.CONFIGS_DIR / f"{name}.json").exists():
            status = "Error: No config selected." if not name else f"Error: Config '{name}' not found."
            updates[self.components['config_status']] = gr.update(value=status)
        else:
            with config_path.open('r') as f:
                settings = json.load(f)
            
            for param, elem_id in self.param_to_elem_id_map.items():
                if param in settings and elem_id in self.components: 
                    updates[self.components[elem_id]] = gr.update(value=settings[param])
            
            for slider in self.components['weight_sliders'] + self.components['pre_metric_sliders']:
                if slider.elem_id in settings:
                    updates[slider] = gr.update(value=settings[slider.elem_id])

            updates[self.components['config_status']] = gr.update(value=f"Loaded config '{name}'.")

        return [updates.get(comp) for comp in output_components]

    def delete_config(self, name):
        if not name: return "Error: No config selected.", gr.update()
        (config.CONFIGS_DIR / f"{name}.json").unlink(missing_ok=True)
        return f"Config '{name}' deleted.", gr.update(choices=[f.stem for f in config.CONFIGS_DIR.glob("*.json")], value=None)

if __name__ == "__main__":
    # FIX: Change hard fail to a warning for CPU-only mode.
    if torch is None or not torch.cuda.is_available():
        print("WARNING: PyTorch or CUDA is not available. Running in CPU-only mode. "
              "GPU-dependent features (Face Analysis, Subject Masking) will be disabled.")
    
    check_dependencies()
    app_ui = AppUI()
    demo = app_ui.build_ui()
    demo.launch()

