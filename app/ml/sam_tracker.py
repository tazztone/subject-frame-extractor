"""SAM and DAM4SAM tracker utilities."""

import shutil
from pathlib import Path
import torch
from DAM4SAM.dam4sam_tracker import DAM4SAMTracker


def initialize_dam4sam_tracker(params):
    """Initialize DAM4SAM tracker with model downloading."""
    from app.core.config import Config
    from app.core.logging import UnifiedLogger
    from app.ml.downloads import download_model
    
    config = Config()
    logger = UnifiedLogger()
    
    if not all([DAM4SAMTracker, torch, torch.cuda.is_available()]):
        logger.error("DAM4SAM dependencies or CUDA not available.")
        return None
        
    try:
        model_name = params.dam4sam_model_name
        logger.info("Initializing DAM4SAM tracker",
                   extra={'model': model_name})
                   
        model_urls = {
            "sam21pp-T": ("https://dl.fbaipublicfiles.com/"
                         "segment_anything_2/092824/sam2.1_hiera_tiny.pt"),
            "sam21pp-S": ("https://dl.fbaipublicfiles.com/"
                         "segment_anything_2/092824/sam2.1_hiera_small.pt"),
            "sam21pp-B+": ("https://dl.fbaipublicfiles.com/"
                          "segment_anything_2/092824/sam2.1_hiera_base_plus.pt"),
            "sam21pp-L": ("https://dl.fbaipublicfiles.com/"
                         "segment_anything_2/092824/sam2.1_hiera_large.pt")
        }
        
        checkpoint_path = config.DIRS['models'] / Path(model_urls[model_name]).name
        download_model(model_urls[model_name], checkpoint_path,
                      f"{model_name} model", 100_000_000)

        from DAM4SAM.utils.utils import determine_tracker
        actual_path, _ = determine_tracker(model_name)
        if not Path(actual_path).exists():
            Path(actual_path).parent.mkdir(exist_ok=True, parents=True)
            shutil.copy(checkpoint_path, actual_path)

        tracker = DAM4SAMTracker(model_name)
        logger.success("DAM4SAM tracker initialized.")
        return tracker
    except Exception as e:
        logger.error("Failed to initialize DAM4SAM tracker", exc_info=True)
        return None
