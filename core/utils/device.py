def get_device() -> str:
    """Returns 'cuda' if available, else 'cpu'."""
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def is_cuda_available() -> bool:
    """Returns True if CUDA is available."""
    import torch

    # Using a simple check to allow easy patching in tests
    return torch.cuda.is_available()


def get_gpu_memory_pressure() -> float:
    """Returns fraction of reserved/total VRAM (0.0 to 1.0), or 0.0 if not available."""
    try:
        import torch

        if not torch.cuda.is_available():
            return 0.0
        # total_memory for device 0
        total = torch.cuda.get_device_properties(0).total_memory
        if total == 0:
            return 0.0
        reserved = torch.cuda.memory_reserved(0)
        return reserved / total
    except Exception:
        return 0.0


def empty_cache():
    """Empty CUDA cache if available."""
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def synchronize():
    """Synchronize CUDA if available."""
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()
