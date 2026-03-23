# core/managers/tracker_factory.py
from __future__ import annotations

from typing import Literal

TrackerBackend = Literal["sam2", "sam3"]


def build_tracker(backend: TrackerBackend, checkpoint_path: str, device: str = "cuda"):
    """
    Factory function to build a subject tracker based on the selected backend.

    Args:
        backend: The tracker backend to use ("sam2" or "sam3")
        checkpoint_path: Path to the model checkpoint
        device: Device to run on ('cpu' or 'cuda')

    Returns:
        An instance of the selected tracker wrapper.
    """
    if backend == "sam2":
        from .sam21 import SAM21Wrapper

        return SAM21Wrapper(checkpoint_path, device)
    elif backend == "sam3":
        # Lazy import ensures SAM3's Triton mocks only run when sam3 is selected
        from .sam3 import SAM3Wrapper

        return SAM3Wrapper(checkpoint_path, device)
    raise ValueError(f"Unknown tracker backend: {backend!r}")
