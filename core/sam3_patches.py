"""
SAM3 Compatibility Patches

Provides fallback implementations for SAM3 operations that require Triton,
fixes image processing issues, and addresses deprecation warnings.
"""

import hashlib
import logging
from pathlib import Path

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)

# Expected hash of SAM3_repo/sam3/model/sam3_video_predictor.py
# to ensure manual patches align with the library version.
SAM3_PREDICTOR_HASH = "a229bb66a1b3ee4a8db9b51b5a2b9b998cb7b87546cd3cf7612f31e42ce578db"


def edt_triton_fallback(data):
    """OpenCV-based fallback for Euclidean Distance Transform when Triton unavailable"""
    assert data.dim() == 3
    device = data.device
    data_cpu = data.cpu().numpy().astype(np.uint8)
    B, H, W = data_cpu.shape
    output = np.zeros_like(data_cpu, dtype=np.float32)
    for b in range(B):
        dist = cv2.distanceTransform(data_cpu[b], cv2.DIST_L2, 0)
        output[b] = dist
    return torch.from_numpy(output).to(device)


def connected_components_fallback(input_tensor):
    """CPU-based fallback for connected components when Triton unavailable"""
    from skimage.measure import label as sk_label

    if input_tensor.dim() == 3:
        input_tensor = input_tensor.unsqueeze(1)
    assert input_tensor.dim() == 4 and input_tensor.shape[1] == 1

    device = input_tensor.device
    data_cpu = input_tensor.squeeze(1).cpu().numpy().astype(np.uint8)
    B, H, W = data_cpu.shape

    labels_list, counts_list = [], []
    for b in range(B):
        labels, num = sk_label(data_cpu[b], return_num=True)
        counts = np.zeros_like(labels)
        for i in range(1, num + 1):
            cur_mask = labels == i
            counts[cur_mask] = cur_mask.sum()
        labels_list.append(labels)
        counts_list.append(counts)

    labels_tensor = torch.from_numpy(np.stack(labels_list)).unsqueeze(1).to(device)
    counts_tensor = torch.from_numpy(np.stack(counts_list)).unsqueeze(1).to(device)
    return labels_tensor, counts_tensor


def set_image_patched(self, image):
    """
    Patched version of Sam3Processor.set_image to handle HWC inputs correctly.
    Original implementation assumes CHW or HW, failing on standard HWC.
    """
    if isinstance(image, np.ndarray):
        # Check for channel-last format (H, W, C) typical for OpenCV/Pillow
        if image.ndim == 3 and image.shape[2] <= 4:
            height, width = image.shape[:2]
        else:
            height, width = image.shape[-2:]
    else:
        height, width = image.shape[-2:]

    self.orig_h = height
    self.orig_w = width
    return self.transform(image)


def patch_sam3_dtype():
    """
    Force SAM3 to use float32 to avoid BFloat16/float32 bias mismatch on Ampere+ GPUs.
    This patches the model builder and the high-level predictor.
    """
    try:
        import sam3.model_builder as mb
        from sam3.model.sam3_video_predictor import Sam3VideoPredictor

        # 1. Patch build_sam3_video_model in model_builder
        original_build_model = mb.build_sam3_video_model

        def build_sam3_video_model_patched(*args, **kwargs):
            model = original_build_model(*args, **kwargs)
            # Ensure model is float32
            if hasattr(model, "to"):
                # We don't change device here, just ensure dtype
                model = model.to(dtype=torch.float32)
            return model

        mb.build_sam3_video_model = build_sam3_video_model_patched

        # 2. Patch Sam3VideoPredictor.__init__ to avoid hardcoded .cuda() call
        # which might reset the dtype if not careful, and to be safer.
        original_predictor_init = Sam3VideoPredictor.__init__

        def predictor_init_patched(self, *args, **kwargs):
            # We need to temporarily patch build_sam3_video_model AGAIN inside here
            # because Sam3VideoPredictor imports it locally in __init__
            import sam3.model.sam3_video_predictor as svp

            svp.build_sam3_video_model = build_sam3_video_model_patched

            # Now call original init
            original_predictor_init(self, *args, **kwargs)

            # After init, ensure the model is indeed float32 and on the right device
            if hasattr(self, "model") and hasattr(self.model, "to"):
                self.model = self.model.to(device="cuda", dtype=torch.float32)

        if not hasattr(Sam3VideoPredictor, "_mock_name"):
            Sam3VideoPredictor.__init__ = predictor_init_patched

    except ImportError:
        pass


def patch_sam3_resources():
    """
    Monkey patch pkg_resources within sam3.model_builder to use importlib.resources.
    This avoids the deprecation warning without modifying the 3rd party code.
    """
    try:
        from importlib import resources as importlib_resources

        import pkg_resources
        import sam3.model_builder as mb

        # Define a replacement for resource_filename
        def patched_resource_filename(package, resource):
            if package == "sam3" and "assets" in resource:
                try:
                    return str(importlib_resources.files(package).joinpath(resource))
                except Exception:
                    # Fallback to original if something fails
                    return pkg_resources.resource_filename(package, resource)
            return pkg_resources.resource_filename(package, resource)

        # Apply the patch to the model_builder module's reference to pkg_resources
        # Note: We patch the attribute on the module object itself
        if hasattr(mb, "pkg_resources"):
            mb.pkg_resources.resource_filename = patched_resource_filename

    except (ImportError, AttributeError):
        pass


def _check_sam3_version(predictor_path: Path) -> bool:
    """
    Check if the SAM3 predictor file matches the expected hash.
    Returns True if version matches or file doesn't exist, False on mismatch.
    """
    if not predictor_path.exists():
        return True

    try:
        with open(predictor_path, "rb") as f:
            current_hash = hashlib.sha256(f.read()).hexdigest()

        if current_hash != SAM3_PREDICTOR_HASH:
            logger.warning(
                f"SAM3 version mismatch (hash: {current_hash[:8]}...). "
                f"Monkey patches may be unstable. Expected: {SAM3_PREDICTOR_HASH[:8]}...",
                extra={"component": "patcher"},
            )
            return False
    except Exception as e:
        logger.debug(f"SAM3 version check failed to read file: {e}")
        return True

    return True


def patch_sam3_bf16_stability():
    """
    Patch TransformerDecoderLayer.forward_ffn to handle BFloat16 inputs
    correctly when autocast is disabled.
    """
    try:
        from sam3.model.decoder import TransformerDecoderLayer

        original_forward_ffn = TransformerDecoderLayer.forward_ffn

        def forward_ffn_patched(self, tgt):
            # If input is bfloat16, ensure it's float32 before entering
            # the disabled-autocast block to avoid type mismatch with float32 weights
            if isinstance(tgt, torch.Tensor) and tgt.dtype == torch.bfloat16:
                tgt = tgt.to(torch.float32)
            return original_forward_ffn(self, tgt)

        TransformerDecoderLayer.forward_ffn = forward_ffn_patched
    except ImportError:
        pass


def patch_sam3_detect_objects():
    """Add detect_objects capability to Sam3VideoPredictor classes."""
    try:
        from PIL import Image
        from sam3.model.sam3_video_predictor import Sam3VideoPredictor, Sam3VideoPredictorMultiGPU

        def detect_objects(self, image: np.ndarray, text: str):
            """Detect objects in a frame using a text prompt."""
            # Use init_state with a list containing one PIL image
            pil_img = Image.fromarray(image)
            # Important: init_state is available on self.model
            inference_state = self.model.init_state(resource_path=[pil_img])

            # Add the text prompt
            _, outputs = self.model.add_prompt(inference_state=inference_state, frame_idx=0, text_str=text)

            # outputs contains out_boxes_xywh (normalized)
            boxes_xywh = outputs.get("out_boxes_xywh", [])
            probs = outputs.get("out_probs", [])

            # Convert to absolute xyxy as expected by seed_selector.py
            h, w = image.shape[:2]
            results = []
            for box, prob in zip(boxes_xywh, probs):
                x, y, bw, bh = box
                xyxy = [x * w, y * h, (x + bw) * w, (y + bh) * h]
                results.append({"bbox": xyxy, "conf": float(prob), "type": "detection"})

            # Clean up session/inference_state
            self.model.reset_state(inference_state)
            return {"outputs": results}

        # Patch the base handle_request to support the new type
        original_handle_request = Sam3VideoPredictor.handle_request

        @torch.inference_mode()
        def handle_request_patched(self, request):
            if request["type"] == "detect_objects":
                return self.detect_objects(request["image"], request["text"])
            return original_handle_request(self, request)

        Sam3VideoPredictor.detect_objects = detect_objects
        Sam3VideoPredictor.handle_request = handle_request_patched

        # Sam3VideoPredictorMultiGPU also overrides handle_request, so patch it too
        if hasattr(Sam3VideoPredictorMultiGPU, "handle_request"):
            original_multi_handle_request = Sam3VideoPredictorMultiGPU.handle_request

            @torch.inference_mode()
            def multi_handle_request_patched(self, request):
                if request["type"] == "detect_objects":
                    # For detect_objects, we use the base implementation on rank 0
                    if self.world_size > 1 and self.rank == 0:
                        for rank in range(1, self.world_size):
                            self.command_queues[rank].put((request, False))

                    return Sam3VideoPredictor.handle_request(self, request)

                return original_multi_handle_request(self, request)

            Sam3VideoPredictorMultiGPU.handle_request = multi_handle_request_patched

    except ImportError:
        pass


def patch_sam3_pvs_initialization():
    """
    SAM3's Sam3VideoInference._build_tracker_output assumes that
    the detector has processed a frame before PVS (Promptable Visual Segmentation) tracker points
    are added (PCS first workflow). This causes an AssertionError if adding points/boxes first.
    This patch allows PVS to optionally initialize the cache as an empty dict.
    """
    try:
        from sam3.model.sam3_video_inference import Sam3VideoInferenceWithInstanceInteractivity

        orig_build_tracker = Sam3VideoInferenceWithInstanceInteractivity._build_tracker_output

        def _build_tracker_output_patched(self, inference_state, frame_idx, refined_obj_id_to_mask=None):
            if "cached_frame_outputs" not in inference_state:
                inference_state["cached_frame_outputs"] = {}

            if frame_idx not in inference_state["cached_frame_outputs"]:
                # Bypass the assertion by populating an empty cache
                inference_state["cached_frame_outputs"][frame_idx] = {}

            return orig_build_tracker(self, inference_state, frame_idx, refined_obj_id_to_mask)

        Sam3VideoInferenceWithInstanceInteractivity._build_tracker_output = _build_tracker_output_patched
        logger.debug("Applied SAM3 PVS Initialization patch.")
    except Exception as e:
        logger.warning(f"Failed to apply PVS patch: {e}")


def apply_patches():
    """Apply all monkey patches to SAM3 with version safety check."""
    # 0. Version Safety Check
    try:
        import sam3.model.sam3_video_predictor as svp

        _check_sam3_version(Path(svp.__file__))
    except Exception as e:
        logger.debug(f"SAM3 version check skipped: {e}")

    # 1. Resource patches (pkg_resources deprecation)
    patch_sam3_resources()

    # 2. Image processor patches (HWC handling)
    try:
        from sam3.model.sam3_image_processor import Sam3Processor

        Sam3Processor.set_image = set_image_patched
    except ImportError:
        pass

    # 3. Dtype and stability patches
    patch_sam3_dtype()
    patch_sam3_bf16_stability()

    # 4. Feature and Workflow patches
    patch_sam3_detect_objects()
    patch_sam3_pvs_initialization()

    # 5. Triton fallbacks
    try:
        import triton  # noqa: F401
    except ImportError:
        try:
            import sam3.model.edt as edt_module
            import sam3.perflib.connected_components as cc_module

            edt_module.edt_triton = edt_triton_fallback
            cc_module.connected_components = connected_components_fallback
        except ImportError:
            pass
