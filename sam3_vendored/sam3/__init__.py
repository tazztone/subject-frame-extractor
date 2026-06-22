# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def patch_torch_load_for_safetensors():
    """
    Patch torch.load to transparently load safetensors checkpoints when requested.
    This allows loading Comfy-Org's sam3.1_multiplex_fp16.safetensors directly.
    """
    try:
        import safetensors.torch
        import torch

        original_load = torch.load

        def patched_load(f, *args, **kwargs):
            filepath = None
            if isinstance(f, (str, Path)):
                filepath = str(f)
            elif hasattr(f, "name"):
                filepath = str(f.name)

            if filepath and filepath.endswith(".safetensors"):
                state_dict = safetensors.torch.load_file(filepath, device="cpu")
                if "model" in state_dict and isinstance(state_dict["model"], dict):
                    return state_dict["model"]
                return state_dict

            return original_load(f, *args, **kwargs)

        torch.load = patched_load
        logger.debug("Applied safetensors torch.load patch.")
    except (ImportError, AttributeError) as e:
        logger.debug(f"Failed to apply safetensors torch.load patch: {e}")


# Apply the torch.load patch automatically when the sam3 package is imported
patch_torch_load_for_safetensors()

from .model_builder import build_sam3_image_model, build_sam3_predictor

__version__ = "0.1.0"

__all__ = ["build_sam3_image_model", "build_sam3_predictor"]
