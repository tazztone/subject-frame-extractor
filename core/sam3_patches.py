"""
SAM3 Compatibility Patches for Windows

Provides fallback implementations for SAM3 operations that require Triton,
which is not available on Windows.
"""

import cv2
import numpy as np
import torch


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


def apply_patches():
    """Apply monkey patches to SAM3 if Triton is not available, AND fix image processing."""
    # Always patch the image processor to fix HWC handling
    try:
        from sam3.model.sam3_image_processor import Sam3Processor
        Sam3Processor.set_image = set_image_patched
    except ImportError:
        pass

    try:
        import triton  # noqa: F401
        # Triton is available, no patching needed
    except ImportError:
        # Triton not available - apply monkey patches
        import sam3.model.edt as edt_module
        import sam3.perflib.connected_components as cc_module

        edt_module.edt_triton = edt_triton_fallback
        cc_module.connected_components = connected_components_fallback
