# core/managers/tracker_factory.py
from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Optional

if TYPE_CHECKING:
    from core.config import Config

TrackerBackend = Literal["sam3"]


def build_tracker(backend: str, checkpoint_path: str, device: str = "cuda", config: Optional["Config"] = None):
    """
    Factory function to build a subject tracker based on the selected backend.

    Args:
        backend: The tracker backend to use ("sam2" or "sam3")
        checkpoint_path: Path to the model checkpoint
        device: Device to run on ('cpu' or 'cuda')
        config: Optional application configuration

    Returns:
        An instance of the selected tracker wrapper.
    """
    if backend == "sam2":
        raise ValueError("SAM2.1 has been retired. Please use SAM3.1 instead.")
    elif backend == "sam3":
        # Lazy import ensures SAM3's Triton mocks only run when sam3 is selected
        from .sam3 import SAM3Wrapper

        return SAM3Wrapper(checkpoint_path, device, config=config)
    raise ValueError(f"Unknown tracker backend: {backend!r}")
