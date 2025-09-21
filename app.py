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
from contextlib import contextmanager

# --- Logger Setup (Fix a) ---
# Basic logger setup moved here to be available globally.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

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
        Checks for model files and raises an error if they are not found.
        """
        base_path = Path(__file__).parent / "models"
        base_path.mkdir(exist_ok=True)

        checkpoints = {
            "sam2.1": "sam2.1_hiera_large.pt",
            "sam21pp-L": "sam2.1_hiera_large.pt",
        }
        model_configs = {
            "sam2.1": "DAM4SAM/sam2/sam21pp_hiera_l.yaml",
            "sam21pp-L": "DAM4SAM/sam2/sam21pp_hiera_l.yaml",
            "default": "DAM4SAM/sam2/sam21pp_hiera_l.yaml"
        }
        
        checkpoint_filename = checkpoints.get(tracker_name, "sam2.1_hiera_large.pt")
        config_filename = model_configs.get(tracker_name, model_configs["default"])

        checkpoint_path = base_path / checkpoint_filename
        config_base = Path(__file__).parent
        config_path = config_base / config_filename

        if not checkpoint_path.is_file():
            logger.warning(f"Model checkpoint file not found: '{checkpoint_filename}'. Attempting to download...")
            try:
                import urllib.request
                model_url = f"https://huggingface.co/facebook/sam2.1-hiera-large/resolve/main/{checkpoint_filename}"
                logger.info(f"Downloading model from {model_url} to {checkpoint_path}")

                # Use a more robust download with progress indication
                self._download_with_progress(model_url, checkpoint_path, f"SAM model {checkpoint_filename}")

                logger.info("Model downloaded successfully.")

            except Exception as e:
                error_msg = (
                    f"Failed to download model checkpoint: '{checkpoint_filename}'. "
                    f"Error: {e}"
                )
                logger.error(error_msg)
                raise FileNotFoundError(error_msg) from e

        # --- Fix d: DAM4SAM Initialization Fails Silently on Missing Config ---
        if not config_path.is_file():
            logger.warning(f"Model config file not found: '{config_filename}'. Attempting to download...")
            try:
                import urllib.request
                # Assuming the config is from the official repo, adjust URL if needed
                config_url = f"https://raw.githubusercontent.com/facebookresearch/segment-anything-2/main/sam2/configs/sam2.1_hiera_l.yaml"
                logger.info(f"Downloading config from {config_url} to {config_path}")
                config_path.parent.mkdir(parents=True, exist_ok=True)
                urllib.request.urlretrieve(config_url, config_path)
                logger.info("Config downloaded successfully.")
            except Exception as e:
                error_msg = f"Failed to download config: '{config_filename}'. Error: {e}"
                logger.error(error_msg)
                raise FileNotFoundError(error_msg) from e

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
            self.max_stored_frames = 10  # Limit stored frames to prevent memory leaks

        def cleanup_inference_state(self):
            """Clean up inference state to prevent memory leaks"""
            if hasattr(self, 'inference_state') and self.inference_state:
                # Clear stored images
                if "images" in self.inference_state:
                    self.inference_state["images"].clear()

                # Clear cached features
                if "cached_features" in self.inference_state:
                    self.inference_state["cached_features"].clear()

                # Clear output dictionaries
                if "output_dict" in self.inference_state:
                    self.inference_state["output_dict"].clear()

                if "output_dict_per_obj" in self.inference_state:
                    self.inference_state["output_dict_per_obj"].clear()

                if "temp_output_dict_per_obj" in self.inference_state:
                    self.inference_state["temp_output_dict_per_obj"].clear()

                # Clear consolidated frame indices
                if "consolidated_frame_inds" in self.inference_state:
                    self.inference_state["consolidated_frame_inds"].clear()

                # Clear frame tracking data
                if "frames_already_tracked" in self.inference_state:
                    self.inference_state["frames_already_tracked"].clear()

                if "frames_tracked_per_obj" in self.inference_state:
                    self.inference_state["frames_tracked_per_obj"].clear()

                # Reset object tracking data
                if "obj_id_to_idx" in self.inference_state:
                    self.inference_state["obj_id_to_idx"].clear()

                if "obj_idx_to_id" in self.inference_state:
                    self.inference_state["obj_idx_to_id"].clear()

                if "obj_ids" in self.inference_state:
                    self.inference_state["obj_ids"].clear()

                # Clear point and mask inputs
                if "point_inputs_per_obj" in self.inference_state:
                    self.inference_state["point_inputs_per_obj"].clear()

                if "mask_inputs_per_obj" in self.inference_state:
                    self.inference_state["mask_inputs_per_obj"].clear()

                if "adds_in_drm_per_obj" in self.inference_state:
                    self.inference_state["adds_in_drm_per_obj"].clear()

            # Force garbage collection
            gc.collect()

            # Clear GPU cache if available
            if torch and torch.cuda.is_available():
                torch.cuda.empty_cache()

        def _prepare_image(self, img_pil):
            # Safely access inference_state with proper error handling
            if not hasattr(self, 'inference_state') or not self.inference_state:
                raise RuntimeError("Inference state not initialized")

            device = self.inference_state.get("device")
            if device is None:
                raise RuntimeError("Device not found in inference state")

            try:
                img = torch.from_numpy(np.array(img_pil)).to(device)
                img = img.permute(2, 0, 1).float() / 255.0
                img = F.resize(img, (self.input_image_size, self.input_image_size))
                img = (img - self.img_mean) / self.img_std
                return img
            except Exception as e:
                logger.error(f"Error preparing image: {e}")
                raise RuntimeError(f"Failed to prepare image: {e}") from e

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

            # Safely set inference state properties
            if "images" not in self.inference_state:
                self.inference_state["images"] = {}

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
            # Safely remove frame from inference_state
            if self.frame_index in self.inference_state.get("images", {}):
                self.inference_state["images"].pop(self.frame_index)

            out_dict = {'pred_mask': m}
            return out_dict

        @torch.inference_mode()
        def track(self, image, init=False):
            prepared_img = self._prepare_image(image).unsqueeze(0)
            if not init:
                self.frame_index += 1
                self.inference_state["num_frames"] += 1

            # Limit stored frames to prevent memory leaks
            if "images" not in self.inference_state:
                self.inference_state["images"] = {}

            # Clean up old frames if we exceed the limit
            if len(self.inference_state["images"]) >= self.max_stored_frames:
                # Remove oldest frames, keeping only the most recent ones
                sorted_frames = sorted(self.inference_state["images"].keys())
                frames_to_remove = sorted_frames[:-self.max_stored_frames]
                for old_frame in frames_to_remove:
                    del self.inference_state["images"][old_frame]

            # Safely set the current frame
            if self.frame_index in self.inference_state["images"]:
                logger.warning(f"Frame {self.frame_index} already exists in inference_state, overwriting")
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
                    # Limit object_sizes list to prevent memory leaks
                    if len(self.object_sizes) >= 300:
                        self.object_sizes = self.object_sizes[-200:]  # Keep last 200, remove older ones
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
                # Safely remove frame from inference_state
                if self.frame_index in self.inference_state.get("images", {}):
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
    masking_libs_installed = all([torch, DAM4SAMTracker, Image, yaml])
    cuda_available = torch is not None and torch.cuda.is_available()
    return {
        'face_analysis': FaceAnalysis is not None,
        'youtube_dl': ytdlp is not None,
        'scene_detection': detect is not None,
        'masking': masking_libs_installed and cuda_available,
        'masking_libs_installed': masking_libs_installed,
        'cuda_available': cuda_available,
        'numba_acceleration': NUMBA_AVAILABLE,
        'person_detection': YOLO is not None,
    }

class Config:
    """Centralized configuration management."""
    BASE_DIR = Path(__file__).parent
    LOG_DIR = BASE_DIR / "logs"
    CONFIGS_DIR = BASE_DIR / "configs"
    MODELS_DIR = BASE_DIR / "models"
    DOWNLOADS_DIR = BASE_DIR / "downloads"
    LOG_FILE = "frame_extractor.log"
    QUALITY_METRICS = ["sharpness", "edge_strength", "contrast", "brightness", "entropy"]
    QUALITY_WEIGHTS = {"sharpness": 30, "edge_strength": 20, "contrast": 20, "brightness": 10, "entropy": 20}
    NORMALIZATION_CONSTANTS = {"sharpness": 1000, "edge_strength": 100}
    FILTER_MODES = {"OVERALL": "Overall Quality", "INDIVIDUAL": "Individual Metrics"}
    MIN_MASK_AREA_PCT = 1.0
    QUALITY_DOWNSCALE_FACTOR = 0.25
    UI_DEFAULTS = {
        "method": "all", "interval": 5.0, "max_resolution": "maximum available",
        "fast_scene": False, "resume": False, "use_png": True, "disable_parallel": False,
        "enable_face_filter": True, "face_model_name": "buffalo_l",
        "quality_thresh": 12.0, "face_thresh": 0.5, "sharpness_thresh": 0.0,
        "edge_strength_thresh": 0.0, "contrast_thresh": 0.0, "brightness_thresh": 0.0, "entropy_thresh": 0.0,
        "enable_subject_mask": True, "scene_detect": True,
        "dam4sam_model_name": "sam2.1",
        "person_detector_model": "yolo11x.pt",
    }

    @staticmethod
    def setup_directories():
        Config.LOG_DIR.mkdir(exist_ok=True)
        Config.CONFIGS_DIR.mkdir(exist_ok=True)
        Config.DOWNLOADS_DIR.mkdir(exist_ok=True)
        Config.MODELS_DIR.mkdir(exist_ok=True)

# --- Global Instance & Setup ---
config = Config()
# Directory setup is now the first action
config.setup_directories()

# --- Fix a: Logger Not Defined Before Use (Part 2) ---
# Add file handler after directories are confirmed to exist.
file_handler = logging.FileHandler(config.LOG_DIR / config.LOG_FILE)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

# --- Utility & Helper Functions ---
def check_dependencies():
    """Checks for presence of essential command-line tools."""
    if not shutil.which("ffmpeg"):
        logger.error("FFMPEG is not installed or not in PATH.")
        raise RuntimeError("FFMPEG is not installed. Please install it to continue.")

def sanitize_filename(name, max_length=50):
    """Sanitizes a string to be a valid filename."""
    return re.sub(r'[^\w\-_.]', '_', name)[:max_length]

def safe_path_join(base_path, user_input, allowed_extensions=None):
    """
    Safely join base path with user input to prevent path traversal attacks.

    Args:
        base_path: The base directory path
        user_input: User-provided path or filename
        allowed_extensions: List of allowed file extensions (optional)

    Returns:
        Safe path string

    Raises:
        ValueError: If path is invalid or outside base directory
    """
    import os
    from pathlib import Path

    # Convert to Path objects
    base = Path(base_path).resolve()
    user_path = Path(user_input).resolve()

    # Check if user path is within base path
    try:
        user_path.relative_to(base)
    except ValueError:
        raise ValueError(f"Path '{user_input}' is outside allowed directory '{base}'")

    # Check file extension if specified
    if allowed_extensions and user_path.suffix.lower() not in allowed_extensions:
        raise ValueError(f"File extension '{user_path.suffix}' not allowed. Allowed: {allowed_extensions}")

    return str(user_path)

@contextmanager
def safe_resource_cleanup():
    """
    Context manager for safe resource cleanup.
    Ensures proper cleanup of GPU memory, file handles, and other resources.
    """
    resources = []

    def track_resource(resource, cleanup_func):
        resources.append((resource, cleanup_func))

    try:
        yield track_resource
    finally:
        # Clean up resources in reverse order
        for resource, cleanup_func in reversed(resources):
            try:
                if resource is not None:
                    cleanup_func(resource)
            except Exception as e:
                logger.warning(f"Error during resource cleanup: {e}")

        # Force garbage collection
        gc.collect()

        # Clear GPU cache if available
        if torch and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception as e:
                logger.warning(f"Error clearing GPU cache: {e}")

def safe_execute_with_retry(func, max_retries=3, delay=1.0, backoff=2.0):
    """
    Execute a function with retry logic and proper error handling.

    Args:
        func: Function to execute
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries
        backoff: Backoff multiplier for delay

    Returns:
        Result of function execution

    Raises:
        Exception: Last exception if all retries fail
    """
    import time

    last_exception = None
    current_delay = delay

    for attempt in range(max_retries + 1):
        try:
            return func()
        except Exception as e:
            last_exception = e
            if attempt < max_retries:
                logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {current_delay}s...")
                time.sleep(current_delay)
                current_delay *= backoff
            else:
                logger.error(f"All {max_retries + 1} attempts failed. Last error: {e}")

    raise last_exception

def validate_video_file_path(file_path):
    """
    Validate that a file path points to a valid video file.

    Args:
        file_path: Path to the video file

    Returns:
        Path object if valid

    Raises:
        ValueError: If file is not a valid video file
    """
    from pathlib import Path

    path = Path(file_path)

    if not path.exists():
        raise ValueError(f"File not found: {file_path}")

    if not path.is_file():
        raise ValueError(f"Path is not a file: {file_path}")

    # Check for common video extensions
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v'}
    if path.suffix.lower() not in video_extensions:
        raise ValueError(f"File extension '{path.suffix}' not supported. Supported: {video_extensions}")

    # Check file size (prevent processing extremely large files)
    max_size = 10 * 1024 * 1024 * 1024  # 10GB limit
    if path.stat().st_size > max_size:
        raise ValueError(f"File too large: {path.stat().st_size} bytes (max: {max_size} bytes)")

    return path

def validate_image_file_path(file_path):
    """
    Validate that a file path points to a valid image file.

    Args:
        file_path: Path to the image file

    Returns:
        Path object if valid

    Raises:
        ValueError: If file is not a valid image file
    """
    from pathlib import Path

    path = Path(file_path)

    if not path.exists():
        raise ValueError(f"File not found: {file_path}")

    if not path.is_file():
        raise ValueError(f"Path is not a file: {file_path}")

    # Check for common image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    if path.suffix.lower() not in image_extensions:
        raise ValueError(f"File extension '{path.suffix}' not supported. Supported: {image_extensions}")

    # Check file size (prevent processing extremely large files)
    max_size = 100 * 1024 * 1024  # 100MB limit for images
    if path.stat().st_size > max_size:
        raise ValueError(f"File too large: {path.stat().st_size} bytes (max: {max_size} bytes)")

    return path

def validate_uploaded_file(file_path, allowed_types, max_size_mb=100):
    """
    Validate uploaded file for security and size constraints.

    Args:
        file_path: Path to the uploaded file
        allowed_types: Set of allowed file extensions
        max_size_mb: Maximum file size in MB

    Returns:
        Path object if valid

    Raises:
        ValueError: If file validation fails
    """
    from pathlib import Path

    path = Path(file_path)

    if not path.exists():
        raise ValueError("Uploaded file not found")

    if not path.is_file():
        raise ValueError("Uploaded path is not a file")

    # Check file size
    max_size_bytes = max_size_mb * 1024 * 1024
    if path.stat().st_size > max_size_bytes:
        raise ValueError(f"File too large: {path.stat().st_size} bytes (max: {max_size_bytes} bytes)")

    # Check file extension
    if path.suffix.lower() not in allowed_types:
        raise ValueError(f"File type '{path.suffix}' not allowed. Allowed types: {allowed_types}")

    # Additional security check: ensure file is not a symlink or special file
    if path.is_symlink():
        raise ValueError("Symlinks are not allowed")

    try:
        # Try to read the file to ensure it's not corrupted
        with open(path, 'rb') as f:
            f.read(1024)  # Read first 1KB to check if file is readable
    except (IOError, OSError) as e:
        raise ValueError(f"Cannot read file: {e}")

    return path

def get_person_detector_model_path(model_filename="yolo11x.pt"):
    """Checks for the YOLO model file and downloads it if not found."""
    base_path = Path(__file__).parent / "models"
    base_path.mkdir(exist_ok=True)
    model_path = base_path / model_filename
    model_url = f"https://huggingface.co/Ultralytics/YOLO11/resolve/main/{model_filename}"

    if not model_path.is_file():
        logger.warning(f"Person detector model not found: '{model_filename}'. Attempting to download...")
        try:
            import urllib.request
            logger.info(f"Downloading person detector model from {model_url} to {model_path}")

            # Use enhanced download with progress
            def download_func():
                urllib.request.urlretrieve(model_url, model_path)

            # For now, use simple download but with better error handling
            urllib.request.urlretrieve(model_url, model_path)
            logger.info("Person detector model downloaded successfully.")
        except Exception as e:
            error_msg = (
                f"Failed to download person detector model: '{model_filename}'. "
                f"Error: {e}"
            )
            logger.error(error_msg)
            raise FileNotFoundError(error_msg) from e
    
    return str(model_path)

# --- Optional Person Detector Class ---
class PersonDetector:
    def __init__(self, model="yolo11x.pt", imgsz=640, conf=0.3):
        if YOLO is None:
            raise ImportError("Ultralytics YOLO not installed. Please run: pip install ultralytics")
        
        model_path = get_person_detector_model_path(model)
        self.model = YOLO(model_path)
        self.imgsz = imgsz
        self.conf = conf

    def detect_boxes(self, img_bgr):
        # returns list of (x1, y1, x2, y2, score)
        res = self.model.predict(img_bgr[..., ::-1], imgsz=self.imgsz, conf=self.conf, classes=[0], verbose=False)
        boxes = []
        for r in res:
            if getattr(r, "boxes", None) is None: 
                continue
            for b in r.boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                score = float(b.conf[0])
                boxes.append((x1, y1, x2, y2, score))
        return boxes

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
    def compute_edge_strength(sobelx, sobely):
        magnitude = np.sqrt(sobelx.astype(np.float64)**2 + sobely.astype(np.float64)**2)
        return np.mean(magnitude)

    def compute_entropy(hist):
        total = np.sum(hist) + 1e-7
        prob = hist / total
        nz = prob > 0 # Use boolean indexing to avoid log(0)
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
    errors: int = 0

@dataclass
class AnalysisParameters:
    # Extraction
    source_path: str = ""
    method: str = config.UI_DEFAULTS["method"]
    interval: float = config.UI_DEFAULTS["interval"]
    max_resolution: str = config.UI_DEFAULTS["max_resolution"]
    fast_scene: bool = config.UI_DEFAULTS["fast_scene"]
    use_png: bool = config.UI_DEFAULTS["use_png"]
    # Analysis
    output_folder: str = ""
    video_path: str = ""
    disable_parallel: bool = config.UI_DEFAULTS["disable_parallel"]
    resume: bool = config.UI_DEFAULTS["resume"]
    enable_face_filter: bool = config.UI_DEFAULTS["enable_face_filter"]
    face_ref_img_path: str = ""
    face_model_name: str = config.UI_DEFAULTS["face_model_name"]
    enable_subject_mask: bool = field(default=config.UI_DEFAULTS["enable_subject_mask"])
    dam4sam_model_name: str = field(default=config.UI_DEFAULTS["dam4sam_model_name"])
    scene_detect: bool = field(default=config.UI_DEFAULTS["scene_detect"])
    # --- Fix f: Quality Weights Not Persisted Correctly ---
    quality_weights: dict = field(default_factory=lambda: {k: config.QUALITY_WEIGHTS[k] for k in config.QUALITY_METRICS})
    thresholds: dict = field(default_factory=lambda: {k: config.UI_DEFAULTS[f"{k}_thresh"] for k in config.QUALITY_METRICS})

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
    def __init__(self, params, progress_queue, cancel_event, frame_map=None, face_analyzer=None, reference_embedding=None, person_detector=None):
        self.params = params
        self.progress_queue = progress_queue
        self.cancel_event = cancel_event
        self.mask_dir = None
        self.shots = []
        self.frame_map = frame_map
        self.face_analyzer = face_analyzer
        self.reference_embedding = reference_embedding
        self.person_detector = person_detector
        self.tracker = None

    def _initialize_dam4sam_tracker(self):
        """Initializes the DAM4SAM tracker, failing softly by returning False."""
        if not DAM4SAMTracker:
            msg = "[ERROR] DAM4SAM dependencies (torch, sam2, vot etc.) are not installed."
            self.progress_queue.put({"log": msg})
            return False

        if not torch.cuda.is_available():
            msg = "[ERROR] DAM4SAM masking requires a CUDA-enabled GPU, but CUDA is not available."
            self.progress_queue.put({"log": msg})
            return False

        try:
            model_name = self.params.dam4sam_model_name
            self.progress_queue.put({"log": f"[INFO] Initializing DAM4SAM tracker with model '{model_name}'..."})
            self.tracker = DAM4SAMTracker(model_name)
            self.progress_queue.put({"log": "[SUCCESS] DAM4SAM tracker initialized."})
            return True
        except Exception as e:
            error_msg = f"Failed to initialize DAM4SAM tracker with model '{self.params.dam4sam_model_name}': {e}"
            logger.error(error_msg, exc_info=True)
            self.progress_queue.put({"log": f"[ERROR] {error_msg}"})
            self.tracker = None
            return False

    def run(self, video_path: str, frames_dir: str) -> dict[str, dict]:
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
                    # Advance progress bar for skipped frames to avoid stalling
                    frames_in_shot = len([fn for fn in (self.frame_map or {}).keys() if start_frame <= fn < end_frame])
                    self.progress_queue.put({"progress": frames_in_shot})
                    continue
                
                shot_frames_with_nums = self._load_shot_frames(frames_dir, start_frame, end_frame)
                if not shot_frames_with_nums:
                    self.progress_queue.put({"log": f"[INFO] No extracted frames found for shot {shot_id+1}. Skipping."})
                    continue

                shot_frames_data = [f[1] for f in shot_frames_with_nums]
                seed_frame_local_idx, bbox_xywh, seed_details = self._seed_identity(shot_frames_data)

                if bbox_xywh is None:
                    self.progress_queue.put({"log": f"[WARNING] Could not identify subject in shot {shot_id+1}. Skipping mask generation."})
                    for original_frame_num, _ in shot_frames_with_nums:
                        frame_filename = self.frame_map.get(original_frame_num)
                        if frame_filename:
                            mask_metadata[frame_filename] = asdict(MaskingResult(error="Subject not found in shot", shot_id=shot_id))
                    self.progress_queue.put({"progress": len(shot_frames_with_nums)})
                    continue

                masks, mask_area_pcts, mask_empty_flags, mask_errors = self._propagate_masks_dam4sam(shot_frames_data, seed_frame_local_idx, bbox_xywh)

                if len(masks) != len(shot_frames_with_nums):
                    self.progress_queue.put({"log": f"[ERROR] Mask propagation returned {len(masks)} masks for {len(shot_frames_with_nums)} frames in shot {shot_id+1}. This indicates an internal error."})
                    self.progress_queue.put({"progress": len(shot_frames_with_nums)})
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

                self.progress_queue.put({"progress": len(shot_frames_with_nums)})

            self.progress_queue.put({"log": "[SUCCESS] Subject masking complete."})
            return mask_metadata
        except Exception as e:
            logger.error("Critical error in SubjectMasker run method", exc_info=True)
            self.progress_queue.put({"log": f"[CRITICAL] SubjectMasker failed: {e}"})
            return {}
        finally:
            if hasattr(self, 'tracker') and self.tracker is not None:
                # Clean up inference state before deleting tracker
                self.tracker.cleanup_inference_state()
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
            logger.error(f"Scene detection failed: {e}", exc_info=True)
            self.progress_queue.put({"log": f"[ERROR] Scene detection failed: {e}"})
            # Fallback to single shot to prevent crashing
            image_files = list(Path(frames_dir).glob("frame_*.*"))
            self.shots = [(0, len(image_files))]
            self.progress_queue.put({"log": f"[WARNING] Falling back to a single shot for masking."})


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

    def _expand_face_to_body(self, face_bbox, img_shape):
        """Heuristic fallback when no person detection is available."""
        H, W = img_shape[:2]
        x, y, w, h = face_bbox
        cx = x + w / 2
        # Expand width/height with conservative multipliers, anchored to face position
        new_w = int(min(W, w * 4.0))
        new_h = int(min(H, h * 7.0))
        new_x = int(max(0, cx - new_w / 2))
        # Anchor top of body box slightly above the face
        new_y = int(max(0, y - h * 0.75))
        
        # Ensure box is within image bounds
        if new_x + new_w > W: new_w = W - new_x
        if new_y + new_h > H: new_h = H - new_y
        
        return [new_x, new_y, new_w, new_h]

    def _pick_person_box_for_face(self, frame_img, face_bbox):
        """Finds the person bbox that best contains the given face bbox."""
        # face_bbox: [x, y, w, h]
        if not self.person_detector:
            return None  # Will trigger fallback expansion

        px1, py1, pw, ph = face_bbox
        fx, fy = px1 + pw / 2.0, py1 + ph / 2.0
        
        try:
            candidates = self.person_detector.detect_boxes(frame_img)
        except Exception as e:
            self.progress_queue.put({"log": f"[WARNING] Person detector failed on frame: {e}"})
            return None

        if not candidates:
            return None

        def contains_center(b):
            x1, y1, x2, y2, _ = b
            return (x1 <= fx <= x2) and (y1 <= fy <= y2)

        def iou_with_face(b):
            x1, y1, x2, y2, _ = b
            fb_x1, fb_y1, fb_x2, fb_y2 = px1, py1, px1 + pw, py1 + ph
            ix1, iy1 = max(x1, fb_x1), max(y1, fb_y1)
            ix2, iy2 = min(x2, fb_x2), min(y2, fb_y2)
            iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
            inter = iw * ih
            area_b = (x2 - x1) * (y2 - y1)
            area_f = pw * ph
            union = area_b + area_f - inter + 1e-6
            return inter / union

        pool = sorted(candidates, key=lambda b: (contains_center(b), iou_with_face(b), b[4]), reverse=True)
        
        best_box = pool[0]
        if not contains_center(best_box) and iou_with_face(best_box) < 0.1:
            self.progress_queue.put({"log": "[INFO] No person detection box closely matched the seed face."})
            return None

        x1, y1, x2, y2, _ = best_box
        return [x1, y1, x2 - x1, y2 - y1]

    def _seed_identity(self, shot_frames):
        if not shot_frames:
            return None, None, None

        seed_details = {}
        matched_face = None
        seed_frame_idx = -1
        
        if self.face_analyzer and self.reference_embedding is not None:
            self.progress_queue.put({"log": "[INFO] Searching for reference face in first 5 frames..."})
            min_dist_global = float('inf')
            
            for i, frame_img in enumerate(shot_frames[:5]):
                if frame_img is None: continue
                faces = self.face_analyzer.get(frame_img)
                if not faces: continue

                for face in faces:
                    dist = 1 - np.dot(face.normed_embedding, self.reference_embedding)
                    if dist < min_dist_global:
                        min_dist_global, matched_face, seed_frame_idx = dist, face, i
            
            if matched_face and min_dist_global < 0.6:
                self.progress_queue.put({"log": f"[INFO] Found reference face in frame {seed_frame_idx} with distance {min_dist_global:.2f}."})
                seed_details = {'type': 'face_match', 'seed_face_sim': 1 - min_dist_global}
            else:
                matched_face = None

        if not matched_face:
            self.progress_queue.put({"log": "[INFO] No reference match found. Seeding with the most prominent face in the first frame."})
            seed_frame_idx = 0
            first_frame = shot_frames[0]
            if first_frame is None:
                 self.progress_queue.put({"log": "[WARNING] First frame of shot is invalid. Cannot seed."})
                 return None, None, None

            if not self.face_analyzer:
                 h, w, _ = first_frame.shape
                 self.progress_queue.put({"log": "[WARNING] Face analyzer not available. Using fallback rectangle on first frame."})
                 bbox_xywh = [w // 4, h // 4, w // 2, h // 2]
                 seed_details = {'type': 'fallback_rect'}
                 return 0, bbox_xywh, seed_details

            faces = self.face_analyzer.get(first_frame)
            if not faces:
                self.progress_queue.put({"log": "[WARNING] No faces found in the first frame to seed shot. Skipping."})
                return None, None, None
            
            matched_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            seed_details = {'type': 'face_largest', 'seed_face_sim': None}

        seed_frame_img = shot_frames[seed_frame_idx]
        h, w, _ = seed_frame_img.shape
        x1, y1, x2, y2 = matched_face.bbox.astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        face_bbox_xywh = [x1, y1, x2 - x1, y2 - y1]

        person_bbox = self._pick_person_box_for_face(seed_frame_img, face_bbox_xywh)
        
        final_bbox_xywh = None
        if person_bbox:
            final_bbox_xywh = person_bbox
            seed_details['type'] = 'person_box_from_' + seed_details['type']
            self.progress_queue.put({"log": f"[INFO] Found person box {final_bbox_xywh}. Seeding tracker."})
        else:
            final_bbox_xywh = self._expand_face_to_body(face_bbox_xywh, seed_frame_img.shape)
            seed_details['type'] = 'expanded_box_from_' + seed_details['type']
            self.progress_queue.put({"log": f"[INFO] No person box found. Using heuristic expansion {final_bbox_xywh}."})
        
        return seed_frame_idx, final_bbox_xywh, seed_details
    
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
            self.output_dir = config.DOWNLOADS_DIR / self.video_path.stem
            self.output_dir.mkdir(exist_ok=True)
            
            self.progress_queue.put({"log": "[INFO] Starting frame extraction..."})
            self._run_frame_extraction()

            if self.cancel_event.is_set():
                self.progress_queue.put({"log": "[INFO] Extraction cancelled."})
                return {"log": "[INFO] Extraction cancelled."}
            
            self.progress_queue.put({"log": "[SUCCESS] Extraction complete."})
            return {"done": True, "output_dir": str(self.output_dir), "video_path": str(self.video_path)}
        except Exception as e:
            logger.exception("Error in extraction pipeline.")
            self.progress_queue.put({"log": f"[ERROR] Extraction failed: {e}"})
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

        # Sanitize all inputs to prevent command injection
        video_path = str(self.video_path)
        output_dir = str(self.output_dir)

        # Validate and sanitize filter parameters
        interval = float(self.params.interval)
        if interval <= 0:
            raise ValueError("Interval must be positive")

        fps = self.video_info.get('fps', 30)
        if fps <= 0:
            fps = 30

        # Create safe filter strings
        select_filter_map = {
            'interval': f"fps=1/{interval}",
            'keyframes': "select='eq(pict_type,I)'",
            'scene': f"select='gt(scene,{0.5 if self.params.fast_scene else 0.4})'",
            'all': f"fps={fps}"
        }
        select_filter = select_filter_map.get(self.params.method)

        # Build command safely
        ffmpeg_cmd = ['ffmpeg', '-y', '-i', video_path, '-hide_banner', '-loglevel', loglevel, '-progress', 'pipe:1']

        if use_showinfo:
            filter_str = (select_filter + ",showinfo") if select_filter else "showinfo"
            ffmpeg_cmd.extend(['-vf', filter_str, '-vsync', 'vfr'])
        elif select_filter:
            ffmpeg_cmd.extend(['-vf', select_filter, '-vsync', 'vfr'])

        ext = 'png' if self.params.use_png else 'jpg'
        output_pattern = f"frame_%06d.{ext}"
        ffmpeg_cmd.extend(['-f', 'image2', output_dir + '/' + output_pattern])
        
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
            if process and self.cancel_event.is_set() and process.poll() is None:
                self.progress_queue.put({"log": "[INFO] Terminating FFmpeg process..."})
                try:
                    process.terminate()
                    process.wait(timeout=2)
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
    def __init__(self, params: AnalysisParameters, progress_queue: Queue, cancel_event: threading.Event):
        self.params = params
        self.progress_queue = progress_queue
        self.cancel_event = cancel_event
        self.output_dir = Path(params.output_folder)
        self.metadata_path = self.output_dir / "metadata.jsonl"
        self.reference_embedding = None
        self.write_lock = threading.Lock()
        self.gpu_lock = threading.Lock()
        self.face_analyzer = None
        self.stats = AnalysisStats()
        self.last_stats_ts = 0.0
        self.mask_metadata = {}
        self.features = get_feature_status()
        self.is_cpu_only = not self.features['cuda_available']

    def run(self):
        try:
            if not self.params.output_folder or not self.output_dir.exists():
                raise ValueError("Output folder with extracted frames is required for analysis.")

            config_hash = self._get_config_hash()
            if self.params.resume and self._check_resume_compatibility(config_hash):
                self.progress_queue.put({"log": f"[INFO] Resuming using compatible metadata: {self.metadata_path.name}"})
                return {"done": True, "metadata_path": str(self.metadata_path), "output_dir": str(self.output_dir)}
            
            if (self.output_dir / "masks").exists():
                shutil.rmtree(self.output_dir / "masks")
            self.metadata_path.unlink(missing_ok=True)
            
            with self.metadata_path.open('w') as f:
                header = {"config_hash": config_hash, "params": {k:v for k,v in asdict(self.params).items() if k not in ['source_path', 'output_folder', 'video_path']}}
                f.write(json.dumps(header) + '\n')

            if self.params.enable_face_filter or (self.params.enable_subject_mask and self.features['masking']):
                self._initialize_face_analyzer()
            
            if self.params.enable_face_filter:
                self._process_reference_face()
            
            person_detector = None
            if self.params.enable_subject_mask and self.features['masking']:
                if self.features['person_detection']:
                    try:
                        person_detector = PersonDetector(model="yolo11x.pt", imgsz=640, conf=0.3)
                        self.progress_queue.put({"log": "[INFO] Person detector (YOLO11x) initialized."})
                    except Exception as e:
                        self.progress_queue.put({"log": f"[WARNING] Person detector unavailable: {e}. Falling back to heuristic box expansion."})
                else:
                    self.progress_queue.put({"log": "[INFO] Person detector library not installed. Falling back to heuristic box expansion."})

            # --- Subject Masking Guard ---
            if self.params.enable_subject_mask:
                if not self.features['masking']:
                    self.progress_queue.put({"log": "[WARNING] Subject masking unavailable; skipping (missing dependencies or no CUDA)."})
                else:
                    is_video_path_valid = self.params.video_path and Path(self.params.video_path).exists()
                    if self.params.scene_detect and not is_video_path_valid:
                        self.progress_queue.put({"log": "[WARNING] Valid video path not provided; scene detection for masking is disabled. Masking will run as a single shot over all frames."})

                    frame_map = self._create_frame_map()
                    masker = SubjectMasker(
                        self.params, self.progress_queue, self.cancel_event, 
                        frame_map, face_analyzer=self.face_analyzer, 
                        reference_embedding=self.reference_embedding,
                        person_detector=person_detector
                    )
                    video_path_for_masking = self.params.video_path if is_video_path_valid else ""
                    self.mask_metadata = masker.run(video_path_for_masking, str(self.output_dir))
            else:
                 self.progress_queue.put({"log": "[INFO] Subject masking disabled."})

            config.QUALITY_WEIGHTS = self.params.quality_weights
            self._run_frame_processing()

            if self.cancel_event.is_set():
                self.progress_queue.put({"log": "[INFO] Analysis cancelled."})
                return {"log": "[INFO] Analysis cancelled."}
            
            self.progress_queue.put({"log": "[SUCCESS] Analysis complete. Go to the 'Filtering & Export' tab."})
            return {"done": True, "metadata_path": str(self.metadata_path), "output_dir": str(self.output_dir)}
        except Exception as e:
            logger.exception("Error in analysis pipeline.")
            self.progress_queue.put({"log": f"[ERROR] Analysis failed: {e}"})
            return {"error": str(e)}
            
    # --- Fix c: Incomplete Frame Map Handling in AnalysisPipeline ---
    def _create_frame_map(self):
        self.progress_queue.put({"log": "[INFO] Loading frame map..."})
        frame_map_path = self.output_dir / "frame_map.json"
        frame_map = {}  # original_frame_num -> filename

        image_files = sorted(list(self.output_dir.glob("frame_*.png")) + list(self.output_dir.glob("frame_*.jpg")), 
                             key=lambda p: int(re.search(r'frame_(\d+)', p.name).group(1)) if re.search(r'frame_(\d+)', p.name) else 0)

        if not frame_map_path.exists():
            self.progress_queue.put({"log": "[WARNING] frame_map.json not found. Assuming sequential mapping from filenames."})
            for i, f in enumerate(image_files):
                # Extract frame num from filename (e.g., frame_000123.png -> 123)
                match = re.search(r'frame_(\d+)', f.name)
                if match:
                    frame_map[int(match.group(1))] = f.name
            return frame_map
        
        try:
            with open(frame_map_path, 'r') as f:
                frame_map_list = json.load(f)  # List of original frame nums
            # Map original frame num to sequential filename index
            for i, orig_num in enumerate(sorted(frame_map_list)):  # Sort to ensure order
                if i < len(image_files):
                    frame_map[orig_num] = image_files[i].name
        except (json.JSONDecodeError, ValueError) as e:
            self.progress_queue.put({"log": f"[ERROR] Failed to parse frame_map.json: {e}. Using filename-based mapping."})
            for f in image_files:
                match = re.search(r'frame_(\d+)', f.name)
                if match:
                    frame_map[int(match.group(1))] = f.name
        
        self.progress_queue.put({"log": f"[SUCCESS] Frame map loaded with {len(frame_map)} entries."})
        return frame_map

    def _download_with_progress(self, url, destination, description=""):
        """Download a file with progress indication."""
        try:
            import urllib.request
            import ssl

            # Create SSL context to handle HTTPS downloads
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE

            # Create a request with headers to mimic a browser
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            req = urllib.request.Request(url, headers=headers)

            with urllib.request.urlopen(req, context=ssl_context) as response:
                total_size = int(response.headers.get('Content-Length', 0))

                with open(destination, 'wb') as f:
                    downloaded = 0
                    chunk_size = 8192

                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)

                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            self.progress_queue.put({"log": f"[INFO] Downloading {description}: {progress:.1f}% ({downloaded}/{total_size} bytes)"})

        except Exception as e:
            raise RuntimeError(f"Download failed: {e}") from e

    def _get_config_hash(self):
        d = asdict(self.params)
        relevant_params = {k: d.get(k) for k in [
            'enable_subject_mask', 'scene_detect', 'enable_face_filter', 'face_model_name',
            'quality_weights', 'dam4sam_model_name'
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
            self.face_analyzer = None
            return

        self.progress_queue.put({"log": f"[INFO] Loading face model: {self.params.face_model_name}"})
        try:
            # Ensure face model is downloaded
            self._ensure_face_model_downloaded(self.params.face_model_name)

            self.face_analyzer = FaceAnalysis(
                name=self.params.face_model_name,
                root=str(config.MODELS_DIR),
                providers=['CUDAExecutionProvider']
            )
            self.face_analyzer.prepare(ctx_id=0, det_size=(640, 640))
            self.progress_queue.put({"log": "[SUCCESS] Face model loaded."})
        except Exception as e:
            logger.error(f"Failed to initialize FaceAnalysis: {e}", exc_info=True)
            self.face_analyzer = None
            raise RuntimeError(f"Could not initialize face analysis model. Error: {e}")

    def _ensure_face_model_downloaded(self, model_name):
        """Ensures the specified face analysis model is downloaded."""
        model_urls = {
            "buffalo_l": "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip",
            "buffalo_s": "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_s.zip",
            "buffalo_m": "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_m.zip",
            "antelopev2": "https://github.com/deepinsight/insightface/releases/download/v0.7/antelopev2.zip"
        }

        if model_name not in model_urls:
            raise ValueError(f"Unknown face model: {model_name}")

        model_dir = config.MODELS_DIR / model_name
        if model_dir.exists():
            # Check if model files exist
            required_files = ["det_10g.onnx", "w600k_r50.onnx", "2d106det.onnx"]
            if all((model_dir / f).exists() for f in required_files):
                return  # Model already downloaded

        self.progress_queue.put({"log": f"[INFO] Downloading face model '{model_name}'..."})
        try:
            import urllib.request
            import zipfile

            model_url = model_urls[model_name]
            zip_path = config.MODELS_DIR / f"{model_name}.zip"

            # Download the zip file
            urllib.request.urlretrieve(model_url, zip_path)
            self.progress_queue.put({"log": f"[INFO] Downloaded {model_name}.zip, extracting..."})

            # Extract the zip file
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(config.MODELS_DIR)

            # Clean up zip file
            zip_path.unlink()

            self.progress_queue.put({"log": f"[SUCCESS] Face model '{model_name}' downloaded and extracted."})

        except Exception as e:
            error_msg = f"Failed to download face model '{model_name}': {e}"
            self.progress_queue.put({"log": f"[ERROR] {error_msg}"})
            raise RuntimeError(error_msg) from e

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

        # --- Fix e: GPU Lock Not Always Honored in Parallel Processing ---
        num_workers = 1 if self.params.disable_parallel or self.params.enable_face_filter else (os.cpu_count() or 4)
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

            self.stats.processed += 1
            if frame.error:
                self.stats.errors += 1
            
            now = time.time()
            if now - self.last_stats_ts > 0.25:
                self.progress_queue.put({"stats": asdict(self.stats)})
                self.last_stats_ts = now

            if self.params.enable_face_filter and self.reference_embedding is not None and self.face_analyzer:
                self._analyze_face_similarity(frame)
            
            meta = {
                "filename": image_path.name,
                "face_sim": frame.face_similarity_score,
                "face_conf": frame.max_face_confidence,
                "metrics": asdict(frame.metrics) if frame.metrics else {}
            }
            meta.update(mask_meta)
            if frame.error:
                meta["error"] = frame.error
            
            with self.write_lock, self.metadata_path.open('a') as f:
                f.write(json.dumps(meta) + '\n')
            self.progress_queue.put({"progress": 1})

        except Exception as e:
            logger.error(f"Critical error processing frame {image_path.name}: {e}", exc_info=True)
            self.stats.errors += 1
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
                frame.max_face_confidence = float(best_face.det_score)
        except Exception as e:
            frame.error = f"Face similarity failed: {e}"
            logger.warning(f"Frame {frame.frame_number}: {frame.error}")

# --- Gradio UI & Event Handlers ---
class AppUI:
    def __init__(self):
        self.components = {}
        self.cancel_event = threading.Event()
        self.last_task_result = {}
        self.feature_status = {}

    def build_ui(self):
        """Builds the Gradio UI with a new three-tab workflow."""
        self.feature_status = get_feature_status()
        css = """.gradio-container { max-width: 1280px !important; margin: auto !important; }"""
        with gr.Blocks(theme=gr.themes.Default(primary_hue="blue"), css=css) as demo:
            gr.Markdown("# Advanced Frame Extractor & Filter")
            gr.Markdown("**🤖 Automatic Model Downloads:** All required models (face analysis, SAM, YOLO) will be downloaded automatically when needed.")
            self.components['model_download_status'] = gr.Textbox(
                label="Model Download Status",
                value="✅ All models ready - no downloads needed",
                interactive=False,
                lines=2
            )
            if not self.feature_status['cuda_available']:
                gr.Warning("No CUDA-enabled GPU detected. Running in CPU-only mode. "
                           "Face Analysis and Subject Masking features will be disabled.")
            
            self.components['extracted_video_path_state'] = gr.State("")
            self.components['extracted_frames_dir_state'] = gr.State("")
            self.components['analysis_output_dir_state'] = gr.State("")
            self.components['analysis_metadata_path_state'] = gr.State("")

            with gr.Tabs():
                self._create_extraction_tab()
                self._create_analysis_tab()
                self._create_filtering_tab()

            self._create_event_handlers()
        return demo

    def _create_extraction_tab(self):
        with gr.Tab("1. Frame Extraction") as self.components['extraction_tab']:
            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("### Video Source")
                    self.components['source_input'] = gr.Textbox(
                        label="Video URL or Local Path", lines=1,
                        placeholder="Enter YouTube URL or local video file path",
                        info="YouTube downloads require 'yt-dlp'. Status: " + ("Available" if self.feature_status['youtube_dl'] else "Not Installed")
                    )
                    self.components['upload_video_input'] = gr.File(label="Or Upload Video", file_types=["video"], type="filepath")
                    
                    gr.Markdown("### Extraction Settings")
                    with gr.Row():
                        method_choices = ["keyframes", "interval", "all"]
                        if self.feature_status['scene_detection']: method_choices.insert(2, "scene")
                        
                        self.components['method_input'] = gr.Dropdown(method_choices, value=config.UI_DEFAULTS["method"], label="Extraction Method")
                        self.components['interval_input'] = gr.Number(label="Interval (s)", value=config.UI_DEFAULTS["interval"], visible=False)
                        self.components['fast_scene_input'] = gr.Checkbox(label="Fast Scene Detect", value=config.UI_DEFAULTS["fast_scene"], visible=False)
                    
                    self.components['max_resolution'] = gr.Dropdown(["maximum available", "2160", "1080", "720", "480", "360"], value=config.UI_DEFAULTS["max_resolution"], label="DL Res (URL)")
                    self.components['use_png_input'] = gr.Checkbox(label="Save as PNG (slower, higher quality)", value=config.UI_DEFAULTS["use_png"])

                with gr.Column(scale=3):
                    gr.Markdown("### Extraction Log")
                    self.components['extraction_status'] = gr.Markdown("")
                    self.components['extraction_log'] = gr.Textbox(label="Logs", lines=10, interactive=False, autoscroll=True)
            
            with gr.Row():
                self.components['start_extraction_button'] = gr.Button("Start Extraction", variant="primary")
                self.components['stop_extraction_button'] = gr.Button("Stop", variant="stop", interactive=False)

    def _create_analysis_tab(self):
         with gr.Tab("2. Frame Analysis") as self.components['analysis_tab']:
            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("### Source Frames & Settings")
                    self.components['frames_folder_input'] = gr.Textbox(label="Extracted Frames Folder Path", info="This is filled automatically from Step 1, or you can enter a path manually.")
                    self.components['analysis_video_path_input'] = gr.Textbox(label="Original Video Path (Optional)", info="Needed for scene-detection in masking. Filled from Step 1 if available.")
                    
                    with gr.Accordion("Face Similarity", open=False):
                        is_face_gpu_ready = self.feature_status['face_analysis'] and self.feature_status['cuda_available']
                        face_info = "Available" if is_face_gpu_ready else ("'insightface' not installed" if not self.feature_status['face_analysis'] else "CUDA not available")
                        
                        self.components['enable_face_filter_input'] = gr.Checkbox(label="Enable Face Similarity", value=is_face_gpu_ready, info=f"Compares faces to a reference image. Status: {face_info}", interactive=is_face_gpu_ready)
                        with gr.Group(visible=is_face_gpu_ready) as self.components['face_options_group']:
                            self.components['face_model_name_input'] = gr.Dropdown(["buffalo_l", "buffalo_s", "buffalo_m", "antelopev2"], value=config.UI_DEFAULTS["face_model_name"], label="Model")
                            self.components['face_ref_img_path_input'] = gr.Textbox(label="Reference Image Path")
                            self.components['face_ref_img_upload_input'] = gr.File(label="Or Upload Reference", file_types=["image"], type="filepath")
                    
                    with gr.Accordion("Subject Masking", open=False):
                        masking_status = "Available" if self.feature_status['masking'] else ("Dependencies missing" if not self.feature_status['masking_libs_installed'] else "CUDA not available")
                        person_det_status = "Available (YOLO11x)" if self.feature_status['person_detection'] else "'ultralytics' not installed"
                        
                        default_mask_enable = self.feature_status['masking']
                        self.components['enable_subject_mask_input'] = gr.Checkbox(label="Enable Subject-Only Metrics", value=default_mask_enable, info=f"Masking Status: {masking_status} | Person Detector: {person_det_status}", interactive=self.feature_status['masking'])
                        
                        with gr.Group(visible=default_mask_enable) as self.components['masking_options_group']:
                            self.components['dam4sam_model_name_input'] = gr.Dropdown(['sam2.1'], value=config.UI_DEFAULTS["dam4sam_model_name"], label="DAM4SAM Model")
                            
                            default_scene_detect = self.feature_status['scene_detection']
                            self.components['scene_detect_input'] = gr.Checkbox(label="Use Scene Detection for Masking", value=default_scene_detect, interactive=self.feature_status['scene_detection'], info="Status: " + ("Available" if self.feature_status['scene_detection'] else "'scenedetect' not installed"))

                    with gr.Accordion("Advanced & Config", open=False):
                        self.components['resume_input'] = gr.Checkbox(label="Resume/Use Cache", value=config.UI_DEFAULTS["resume"])
                        self.components['disable_parallel_input'] = gr.Checkbox(label="Disable Parallelism (for low memory)", value=config.UI_DEFAULTS["disable_parallel"])
                        self._create_config_presets_ui()

                with gr.Column(scale=3):
                    gr.Markdown("### Analysis Log")
                    self.components['analysis_status'] = gr.Markdown("")
                    self.components['analysis_log'] = gr.Textbox(label="Logs", lines=15, interactive=False, autoscroll=True)
                    self.components['model_download_status_analysis'] = gr.Textbox(
                        label="Model Download Status",
                        value="⏳ Checking model availability...",
                        interactive=False,
                        lines=2
                    )
            
            with gr.Row():
                self.components['start_analysis_button'] = gr.Button("Start Analysis", variant="primary")
                self.components['stop_analysis_button'] = gr.Button("Stop", variant="stop", interactive=False)

    def _create_filtering_tab(self):
        with gr.Tab("3. Filtering & Export") as self.components['filtering_tab']:
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("## Live Filtering")
                    self.components['filter_mode_toggle'] = gr.Radio(list(config.FILTER_MODES.values()), value=config.FILTER_MODES["OVERALL"], label="Filter By")
                    with gr.Accordion("Customize Quality Weights", open=False):
                        self.components['weight_sliders'] = [gr.Slider(0, 100, config.QUALITY_WEIGHTS[k], step=1, label=k.capitalize()) for k in config.QUALITY_METRICS]
                    
                    with gr.Group() as self.components['overall_quality_group']:
                        self.components['quality_filter_slider'] = gr.Slider(0, 100, config.UI_DEFAULTS["quality_thresh"], label="Min Quality")
                    with gr.Group(visible=False) as self.components['individual_metrics_group']:
                        self.components['filter_metric_sliders'] = [gr.Slider(0, 100, label=f"Min {k.replace('_',' ').capitalize()}") for k in config.QUALITY_METRICS]
                    self.components['face_filter_slider'] = gr.Slider(0, 1.0, 0.5, label="Min Face Similarity", step=0.01, interactive=False)

                with gr.Column(scale=3):
                    with gr.Row():
                        self.components['filter_stats'] = gr.Textbox(label="Filter Results", lines=4, interactive=False, value="Run analysis to see results.")
                        self.components['export_button'] = gr.Button("Export Kept Frames", variant="primary")
                    
                    with gr.Accordion("Export Settings", open=True):
                        self.components['enable_crop_input'] = gr.Checkbox(label="Crop to Subject", value=False)
                        with gr.Group(visible=False) as self.components['crop_options_group']:
                            self.components['crop_ar_input'] = gr.Textbox(label="Target Aspect Ratios (comma-separated)", value="16:9, 1:1, 4:5", info="e.g., '16:9, 4:5'. Will pick the best fit.")
                            self.components['crop_padding_input'] = gr.Slider(label="Padding (%)", minimum=0, maximum=100, value=15, step=1, info="Padding added around the subject's bounding box.")

                    self.components['results_gallery'] = gr.Gallery(label="Kept Frames Preview (Max 100)", columns=8, allow_preview=True)
            
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
        self.components['enable_face_filter_input'].change(lambda e: (gr.update(visible=e), gr.update(interactive=e)), self.components['enable_face_filter_input'], [self.components['face_options_group'], self.components['face_filter_slider']])
        self.components['enable_subject_mask_input'].change(lambda e: gr.update(visible=e), self.components['enable_subject_mask_input'], self.components['masking_options_group'])
        self.components['enable_crop_input'].change(lambda x: gr.update(visible=x), self.components['enable_crop_input'], self.components['crop_options_group'])
        
        self.components['filter_mode_toggle'].change(
            lambda m: (gr.update(visible=m == config.FILTER_MODES["OVERALL"]), gr.update(visible=m != config.FILTER_MODES["OVERALL"])),
            self.components['filter_mode_toggle'],
            [self.components['overall_quality_group'], self.components['individual_metrics_group']]
        )
        
        self._setup_extraction_handler()
        self._setup_analysis_handler()
        self._setup_filtering_handlers()
        self._setup_config_handlers()
        self._setup_model_status_handlers()

    def _get_analysis_params_components(self):
        return [
            self.components['frames_folder_input'], self.components['analysis_video_path_input'], self.components['disable_parallel_input'],
            self.components['resume_input'], self.components['enable_face_filter_input'],
            self.components['face_ref_img_path_input'], self.components['face_ref_img_upload_input'], self.components['face_model_name_input'],
            self.components['enable_subject_mask_input'], self.components['dam4sam_model_name_input'],
            self.components['scene_detect_input']
        ] + self.components['weight_sliders']

    def _setup_extraction_handler(self):
        inputs = [
            self.components['source_input'], self.components['upload_video_input'], self.components['method_input'],
            self.components['interval_input'], self.components['max_resolution'], self.components['fast_scene_input'],
            self.components['use_png_input']
        ]
        outputs = [
            self.components['start_extraction_button'], self.components['stop_extraction_button'], 
            self.components['extraction_log'], self.components['extraction_status'],
            self.components['extracted_video_path_state'], self.components['extracted_frames_dir_state'],
            self.components['frames_folder_input'], self.components['analysis_video_path_input']
        ]
        self.components['start_extraction_button'].click(self.run_extraction_wrapper, inputs, outputs)
        self.components['stop_extraction_button'].click(lambda: self.cancel_event.set(), [], []).then(lambda: gr.update(interactive=False), None, self.components['stop_extraction_button'])

    def _setup_analysis_handler(self):
        inputs = self._get_analysis_params_components()
        outputs = [
            self.components['start_analysis_button'], self.components['stop_analysis_button'],
            self.components['analysis_log'], self.components['analysis_status'],
            self.components['analysis_output_dir_state'], self.components['analysis_metadata_path_state'],
            self.components['filtering_tab']
        ]
        self.components['start_analysis_button'].click(self.run_analysis_wrapper, inputs, outputs)
        self.components['stop_analysis_button'].click(lambda: self.cancel_event.set(), [], []).then(lambda: gr.update(interactive=False), None, self.components['stop_analysis_button'])
        
    def _setup_filtering_handlers(self):
        filter_inputs = [self.components['analysis_metadata_path_state'], self.components['analysis_output_dir_state'],
                         self.components['quality_filter_slider'], self.components['face_filter_slider'], 
                         self.components['filter_mode_toggle']] + self.components['filter_metric_sliders'] + self.components['weight_sliders']
        filter_outputs = [self.components['results_gallery'], self.components['filter_stats']]
        
        filter_controls = [self.components['quality_filter_slider'], self.components['face_filter_slider'], self.components['filter_mode_toggle']] + self.components['filter_metric_sliders'] + self.components['weight_sliders']
        for c in filter_controls:
            c.change(self.apply_gallery_filters, filter_inputs, filter_outputs)
        
        self.components['filtering_tab'].select(self.apply_gallery_filters, filter_inputs, filter_outputs)

        export_inputs = filter_inputs + [
            self.components['enable_crop_input'], 
            self.components['crop_ar_input'],
            self.components['crop_padding_input']
        ]
        self.components['export_button'].click(self.export_kept_frames, export_inputs, self.components['filter_stats'])
    
    def _setup_config_handlers(self):
        config_controls = [
            self.components[comp_id] for comp_id in ['method_input', 'interval_input', 'max_resolution', 'fast_scene_input', 'use_png_input', 'disable_parallel_input', 'resume_input', 'enable_face_filter_input', 'face_model_name_input', 'enable_subject_mask_input', 'dam4sam_model_name_input', 'scene_detect_input']
        ] + self.components['weight_sliders']

        save_inputs = [self.components['config_name_input']] + config_controls
        self.components['save_button'].click(self.save_config, save_inputs, [self.components['config_status'], self.components['config_dropdown']])
        
        load_outputs = config_controls + [self.components['config_status']]
        self.components['load_button'].click(self.load_config, self.components['config_dropdown'], load_outputs)
        
        self.components['delete_button'].click(self.delete_config, self.components['config_dropdown'], [self.components['config_status'], self.components['config_dropdown']])

    def _setup_model_status_handlers(self):
        """Setup handlers for model download status updates."""
        # Update model status when analysis tab is selected
        self.components['analysis_tab'].select(self.update_model_status, [], [self.components['model_download_status_analysis']])

    def update_model_status(self):
        """Update the model download status indicator."""
        try:
            features = get_feature_status()
            status_parts = []

            # Check face analysis models
            if features['face_analysis'] and features['cuda_available']:
                face_models_dir = config.MODELS_DIR
                if face_models_dir.exists():
                    # Check if any face models are available
                    model_found = any((face_models_dir / model_name).exists()
                                    for model_name in ["buffalo_l", "buffalo_s", "buffalo_m", "antelopev2"])
                    if model_found:
                        status_parts.append("✅ Face analysis models ready")
                    else:
                        status_parts.append("📥 Face analysis models will be downloaded automatically")
                else:
                    status_parts.append("📥 Face analysis models will be downloaded automatically")

            # Check SAM models
            sam_models_dir = config.MODELS_DIR
            if sam_models_dir.exists():
                sam_model_found = (sam_models_dir / "sam2.1_hiera_large.pt").exists()
                if sam_model_found:
                    status_parts.append("✅ SAM models ready")
                else:
                    status_parts.append("📥 SAM models will be downloaded automatically")
            else:
                status_parts.append("📥 SAM models will be downloaded automatically")

            # Check YOLO models
            yolo_models_dir = config.MODELS_DIR
            if yolo_models_dir.exists():
                yolo_model_found = (yolo_models_dir / "yolo11x.pt").exists()
                if yolo_model_found:
                    status_parts.append("✅ YOLO models ready")
                else:
                    status_parts.append("📥 YOLO models will be downloaded automatically")
            else:
                status_parts.append("📥 YOLO models will be downloaded automatically")

            if not status_parts:
                return "⏳ Checking model availability..."

            return "\n".join(status_parts)

        except Exception as e:
            return f"⚠️ Error checking model status: {e}"

    def run_extraction_wrapper(self, source_path, upload_video, method, interval, max_res, fast_scene, use_png):
        yield {
            self.components['start_extraction_button']: gr.update(interactive=False),
            self.components['stop_extraction_button']: gr.update(interactive=True),
            self.components['extraction_log']: "",
            self.components['extraction_status']: "Starting..."
        }
        self.cancel_event.clear()

        source = upload_video if upload_video else source_path
        if not source:
            yield {self.components['extraction_log']: "[ERROR] Video source is required.", self.components['start_extraction_button']: gr.update(interactive=True), self.components['stop_extraction_button']: gr.update(interactive=False)}
            return

        # Validate input to prevent path traversal
        try:
            if upload_video:
                # For uploaded files, validate the path
                source = safe_path_join(config.DOWNLOADS_DIR, Path(source).name, {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v'})
            else:
                # For source paths, validate it's a proper video file or URL
                is_url = re.match(r'https?://|youtu\.be', str(source))
                if not is_url:
                    source = str(validate_video_file_path(source))
        except (ValueError, RuntimeError) as e:
            yield {self.components['extraction_log']: f"[ERROR] Invalid video source: {e}", self.components['start_extraction_button']: gr.update(interactive=True), self.components['stop_extraction_button']: gr.update(interactive=False)}
            return

        params = AnalysisParameters(
            source_path=source, method=method, interval=interval, max_resolution=max_res,
            fast_scene=fast_scene, use_png=use_png
        )

        progress_queue = Queue()
        pipeline = ExtractionPipeline(params, progress_queue, self.cancel_event)
        yield from self._run_task(pipeline.run, self.components['extraction_log'], self.components['extraction_status'])
        
        result = self.last_task_result
        if result.get("done") and not self.cancel_event.is_set():
            output_dir = result["output_dir"]
            video_path = result.get("video_path", "")
            yield {
                self.components['start_extraction_button']: gr.update(interactive=True), 
                self.components['stop_extraction_button']: gr.update(interactive=False),
                self.components['extracted_video_path_state']: video_path,
                self.components['extracted_frames_dir_state']: output_dir,
                self.components['frames_folder_input']: output_dir,
                self.components['analysis_video_path_input']: video_path
            }
        else:
             yield {self.components['start_extraction_button']: gr.update(interactive=True), self.components['stop_extraction_button']: gr.update(interactive=False)}

    def run_analysis_wrapper(self, frames_folder, video_path, disable_parallel, resume, enable_face, face_ref_path, face_ref_upload, face_model, enable_mask, dam4sam_model, scene_detect, *weights):
        yield {
            self.components['start_analysis_button']: gr.update(interactive=False), 
            self.components['stop_analysis_button']: gr.update(interactive=True), 
            self.components['analysis_log']: "",
            self.components['analysis_status']: "Starting..."
        }
        self.cancel_event.clear()
        
        # Validate frames folder path to prevent path traversal
        try:
            frames_folder = safe_path_join(config.BASE_DIR, frames_folder)
            frames_path = Path(frames_folder)
            if not frames_path.exists() or not frames_path.is_dir():
                yield {self.components['analysis_log']: "[ERROR] A valid folder of extracted frames is required.", self.components['start_analysis_button']: gr.update(interactive=True), self.components['stop_analysis_button']: gr.update(interactive=False)}
                return
        except (ValueError, RuntimeError) as e:
            yield {self.components['analysis_log']: f"[ERROR] Invalid frames folder path: {e}", self.components['start_analysis_button']: gr.update(interactive=True), self.components['stop_analysis_button']: gr.update(interactive=False)}
            return

        face_ref = face_ref_upload if face_ref_upload else face_ref_path
        if enable_face and face_ref:
            try:
                # Validate face reference image path
                if face_ref_upload:
                    face_ref = safe_path_join(config.DOWNLOADS_DIR, Path(face_ref).name, {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'})
                else:
                    face_ref = str(validate_video_file_path(face_ref))  # Reusing video validation for images
            except (ValueError, RuntimeError) as e:
                yield {self.components['analysis_log']: f"[ERROR] Invalid reference face image: {e}", self.components['start_analysis_button']: gr.update(interactive=True), self.components['stop_analysis_button']: gr.update(interactive=False)}
                return

        features = get_feature_status()
        is_video_path_valid = bool(video_path) and Path(video_path).exists()
        # Coerce scene_detect to False if its dependencies are missing or video path is invalid
        safe_scene_detect = scene_detect and features['scene_detection'] and is_video_path_valid

        params = AnalysisParameters(
            output_folder=frames_folder, video_path=video_path, disable_parallel=disable_parallel, resume=resume,
            enable_face_filter=enable_face, face_ref_img_path=face_ref, face_model_name=face_model,
            enable_subject_mask=enable_mask, dam4sam_model_name=dam4sam_model, scene_detect=safe_scene_detect,
            quality_weights={k: weights[i] for i, k in enumerate(config.QUALITY_METRICS)}
        )

        progress_queue = Queue()
        pipeline = AnalysisPipeline(params, progress_queue, self.cancel_event)
        yield from self._run_task(pipeline.run, self.components['analysis_log'], self.components['analysis_status'])

        result = self.last_task_result
        if result.get("done") and not self.cancel_event.is_set():
            yield {
                self.components['analysis_output_dir_state']: result["output_dir"], 
                self.components['analysis_metadata_path_state']: result["metadata_path"],
                self.components['filtering_tab']: gr.update(interactive=True)
            }
        
        yield {self.components['start_analysis_button']: gr.update(interactive=True), self.components['stop_analysis_button']: gr.update(interactive=False)}

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
                        
                        errors = last_stats.get("errors", 0)
                        
                        status_line = f"**{current_stage}:** {processed_count}/{total_frames} ({ratio:.1%}) &nbsp; | &nbsp; Errors: {errors} &nbsp; | &nbsp; {fps:.1f} items/s &nbsp; | &nbsp; ETA: {mm:02d}:{ss:02d}"
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
    def apply_gallery_filters(metadata_path, output_dir, quality_thresh, face_thresh, filter_mode, *thresholds):
        if not metadata_path: return [], "Run analysis to see results."
        
        ind_thresh = thresholds[:len(config.QUALITY_METRICS)]
        weights = thresholds[len(config.QUALITY_METRICS):]
        
        quality_weights = {k: weights[i] for i, k in enumerate(config.QUALITY_METRICS)}

        kept_frames, total_kept, _, total_frames = AppUI._load_and_filter_metadata(metadata_path, quality_thresh, face_thresh, filter_mode, ind_thresh, quality_weights)
        preview = [str(Path(output_dir) / f['filename']) for f in kept_frames[:100] if (Path(output_dir) / f['filename']).exists()]
        return preview, f"Kept: {total_kept} / {total_frames} frames (previewing {len(preview)})"

    # --- Fix g: Export Crop Function Ignores Mask Errors ---
    @staticmethod
    def _crop_frame_to_aspect_ratio(img, mask, target_ars_str, padding_pct):
        if mask is None or np.sum(mask > 0) == 0:
            logger.warning("Skipping crop: Invalid or empty mask.")
            return img

        rows, cols = np.any(mask, axis=1), np.any(mask, axis=0)
        if not np.any(rows) or not np.any(cols): return img
            
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        box_w, box_h = cmax - cmin, rmax - rmin
        if box_w <= 0 or box_h <= 0: return img
        
        box_cx, box_cy = cmin + box_w / 2, rmin + box_h / 2
        box_ar = box_w / box_h

        pad_w, pad_h = box_w * (padding_pct / 100.0), box_h * (padding_pct / 100.0)
        padded_w, padded_h = box_w + 2 * pad_w, box_h + 2 * pad_h

        try:
            target_ars = [float(w)/float(h) for w, h in (ar.split(':') for ar in target_ars_str.replace(" ","").split(',') if ':' in ar)]
            if not target_ars:
                logger.warning("No valid ARs; using image AR.")
                target_ars = [img.shape[1] / img.shape[0]]
        except (ValueError, ZeroDivisionError) as e:
            logger.error(f"AR parsing failed: {e}. Using image AR.")
            target_ars = [img.shape[1] / img.shape[0]]

        best_ar = min(target_ars, key=lambda ar: abs(ar - box_ar))

        if best_ar > (padded_w / padded_h):
            final_w, final_h = best_ar * padded_h, padded_h
        else:
            final_h, final_w = padded_w / best_ar, padded_w
        
        img_h, img_w, _ = img.shape
        crop_x1 = int(box_cx - final_w / 2)
        crop_y1 = int(box_cy - final_h / 2)
        
        crop_x1 = max(0, min(crop_x1, img_w - int(final_w)))
        crop_y1 = max(0, min(crop_y1, img_h - int(final_h)))
        
        final_w = min(int(final_w), img_w - crop_x1)
        final_h = min(int(final_h), img_h - crop_y1)

        return img[crop_y1:crop_y1+final_h, crop_x1:crop_x1+final_w]

    @staticmethod
    def export_kept_frames(metadata_path, output_dir, quality_thresh, face_thresh, filter_mode, *thresholds_and_crop_args):
        if not metadata_path: return "No metadata to export."
        try:
            num_metrics = len(config.QUALITY_METRICS)
            num_weights = len(config.QUALITY_METRICS)
            ind_thresh = thresholds_and_crop_args[:num_metrics]
            weights = thresholds_and_crop_args[num_metrics : num_metrics + num_weights]
            
            crop_args = thresholds_and_crop_args[num_metrics + num_weights:]
            enable_crop, crop_ars, crop_padding = crop_args[0], crop_args[1], crop_args[2]

            export_dir = Path(output_dir).parent / f"{Path(output_dir).name}_exported_{datetime.now():%Y%m%d_%H%M%S}"
            export_dir.mkdir(exist_ok=True)

            quality_weights = {k: weights[i] for i, k in enumerate(config.QUALITY_METRICS)}
            
            kept_frames, total_kept, _, total_frames = AppUI._load_and_filter_metadata(metadata_path, quality_thresh, face_thresh, filter_mode, ind_thresh, quality_weights)
            
            copied_count = 0
            for frame_meta in sorted(kept_frames, key=lambda x: x['filename']):
                src_path = Path(output_dir) / frame_meta['filename']
                if not src_path.exists(): continue
                
                img = cv2.imread(str(src_path))
                if img is None: continue

                if enable_crop:
                    mask_path_str = frame_meta.get('mask_path')
                    if mask_path_str and Path(mask_path_str).exists():
                        mask = cv2.imread(mask_path_str, cv2.IMREAD_GRAYSCALE)
                        img = AppUI._crop_frame_to_aspect_ratio(img, mask, crop_ars, crop_padding)
                
                dest_path = export_dir / frame_meta['filename']
                cv2.imwrite(str(dest_path), img)
                copied_count += 1
            
            if copied_count != total_kept:
                 return f"Exported {copied_count}/{total_kept} frames to '{export_dir.name}'. Some source files may have been missing."
            
            return f"Exported {total_kept}/{total_frames} frames to '{export_dir.name}'"
        except (IOError, OSError, IndexError) as e:
            logger.error(f"Failed to export frames: {e}", exc_info=True)
            return f"Error during export: {e}"

    @staticmethod
    def _load_and_filter_metadata(path, q_thresh, f_thresh, mode, ind_thresh, quality_weights):
        kept, reasons, total = [], {m: 0 for m in config.QUALITY_METRICS + ['quality', 'face', 'error', 'mask']}, 0
        p = Path(path)
        if not p.exists(): return [], 0, reasons, 0
        with p.open('r') as f:
            try:
                header_line = next(f)
                header = json.loads(header_line)
                is_face_enabled = header.get("params", {}).get("enable_face_filter", False)
            except (StopIteration, json.JSONDecodeError):
                return [], 0, reasons, 0

            for line in f:
                total += 1
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    reasons['error'] += 1
                    continue
                    
                fail_reason = None
                metrics = data.get("metrics", {})

                if "processing_failed" in str(data.get("error", "")):
                    reasons['error'] += 1
                    continue
                
                if metrics:
                    scores = [metrics.get(f"{k}_score", 0)/100.0 for k in config.QUALITY_METRICS]
                    weights = [quality_weights[k]/100.0 for k in config.QUALITY_METRICS]
                    current_quality_score = sum(s * w for s, w in zip(scores, weights)) * 100
                else:
                    current_quality_score = 0

                if mode == config.FILTER_MODES["OVERALL"]:
                    if current_quality_score < q_thresh:
                        fail_reason = 'quality'
                else:
                    for i, k in enumerate(config.QUALITY_METRICS):
                        if metrics.get(f'{k}_score', 100) < ind_thresh[i]:
                            fail_reason = k
                            break
                
                if not fail_reason and is_face_enabled and data.get('face_sim') is not None and data['face_sim'] < f_thresh:
                    fail_reason = 'face'
                
                if not fail_reason and "mask_empty" in data and data["mask_empty"]:
                    fail_reason = 'mask'

                if fail_reason:
                    reasons[fail_reason] = reasons.get(fail_reason, 0) + 1
                else:
                    kept.append(data)

        return kept, len(kept), reasons, total

    def save_config(self, name, *values):
        if not name: return "Error: Config name required.", gr.update()
        name = sanitize_filename(name)
        
        settings = {
            'method_input': values[0], 'interval_input': values[1], 'max_resolution': values[2], 
            'fast_scene_input': values[3], 'use_png_input': values[4], 'disable_parallel_input': values[5],
            'resume_input': values[6], 'enable_face_filter_input': values[7], 'face_model_name_input': values[8],
            'enable_subject_mask_input': values[9], 'dam4sam_model_name_input': values[10], 'scene_detect_input': values[11],
        }
        for i, k in enumerate(config.QUALITY_METRICS):
            settings[f"weight_{k}"] = values[12 + i]

        with (config.CONFIGS_DIR / f"{name}.json").open('w') as f:
            json.dump(settings, f, indent=2)
            
        # --- Fix b: Typo in Config Directory References ---
        return f"Config '{name}' saved.", gr.update(choices=[f.stem for f in config.CONFIGS_DIR.glob("*.json")])

    def load_config(self, name):
        ordered_comp_ids = [
            'method_input', 'interval_input', 'max_resolution', 'fast_scene_input', 'use_png_input', 
            'disable_parallel_input', 'resume_input', 'enable_face_filter_input', 'face_model_name_input',
            'enable_subject_mask_input', 'dam4sam_model_name_input', 'scene_detect_input'
        ]
        
        if not name or not (config_path := config.CONFIGS_DIR / f"{name}.json").exists():
            status = "Error: No config selected." if not name else f"Error: Config '{name}' not found."
            num_outputs = len(ordered_comp_ids) + len(config.QUALITY_METRICS)
            return [gr.update()] * num_outputs + [status]
            
        with config_path.open('r') as f:
            settings = json.load(f)
        
        updates = []
        for comp_id in ordered_comp_ids:
            updates.append(gr.update(value=settings.get(comp_id)))

        for k in config.QUALITY_METRICS:
            updates.append(gr.update(value=settings.get(f"weight_{k}")))
            
        updates.append(f"Loaded config '{name}'.")
        return updates

    def delete_config(self, name):
        if not name: return "Error: No config selected.", gr.update()
        (config.CONFIGS_DIR / f"{name}.json").unlink(missing_ok=True)
        # --- Fix b: Typo in Config Directory References ---
        return f"Config '{name}' deleted.", gr.update(choices=[f.stem for f in config.CONFIGS_DIR.glob("*.json")], value=None)

if __name__ == "__main__":
    if torch is None or not torch.cuda.is_available():
        print("WARNING: PyTorch or CUDA is not available. Running in CPU-only mode. "
              "GPU-dependent features (Face Analysis, Subject Masking) will be disabled.")
    
    check_dependencies()
    app_ui = AppUI()
    demo = app_ui.build_ui()
    demo.launch()

