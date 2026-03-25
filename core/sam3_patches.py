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

    # 3. Dtype patches (Ampere+ stability)
    patch_sam3_dtype()

    # 4. Triton fallbacks
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
