import sys
import threading
import shutil
from pathlib import Path
from queue import Queue
from collections import deque
import pytest

# Add root to path so we can import core modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.config import Config
from core.logger import AppLogger
from core.managers import ModelRegistry, ThumbnailManager
from core.pipelines import (
    execute_extraction,
    execute_pre_analysis,
    execute_propagation,
    execute_analysis,
)
from core.events import (
    ExtractionEvent,
    PreAnalysisEvent,
    PropagationEvent,
)

# Constants from the manual script
VIDEO_PATH = Path("downloads/example clip (2).mp4")
FACE_PATH = Path("downloads/example face.png")

@pytest.mark.integration
@pytest.mark.slow
def test_real_end_to_end_workflow(tmp_path):
    """
    Automated version of verification/e2e_run.py.
    Runs the full extraction -> pre-analysis -> propagation -> analysis pipeline
    on a real video file.
    """
    
    # 1. Setup & Checks
    if not VIDEO_PATH.exists():
        pytest.skip(f"Test video not found at {VIDEO_PATH}. Skip integration test.")
    if not FACE_PATH.exists():
        pytest.skip(f"Test face image not found at {FACE_PATH}. Skip integration test.")

    print(f"\n🚀 Starting E2E Verification with real data at {tmp_path}...")
    
    # Use tmp_path for output to avoid cluttering the repo, 
    # but we can optionally use a fixed path if debugging is needed.
    output_dir = tmp_path / "e2e_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    config = Config()
    # Disable memory watchdog for E2E test to prevent model unloading
    config.monitoring_memory_critical_threshold_mb = 64000
    
    # Disable file logging for dry run
    logger = AppLogger(config, log_to_file=False)
    progress_queue = Queue()
    cancel_event = threading.Event()
    model_registry = ModelRegistry(logger)
    thumbnail_manager = ThumbnailManager(logger, config)
    
    # 2. Extraction
    print("\n--- [STAGE 1: EXTRACTION] ---")
    ext_event = ExtractionEvent(
        source_path=str(VIDEO_PATH),
        method="interval",
        interval="1.0",
        nth_frame=5,
        max_resolution="720",
        thumbnails_only=True,
        thumb_megapixels=0.2,
        scene_detect=True,
        output_folder=str(output_dir)
    )
    
    ext_gen = execute_extraction(
        ext_event, progress_queue, cancel_event, logger, config, thumbnail_manager, model_registry=model_registry
    )
    
    ext_result = deque(ext_gen, maxlen=1)[0]
    assert ext_result.get("done"), f"Extraction failed: {ext_result.get('unified_log')}"
    print("✅ Extraction complete.")
    
    # 3. Pre-Analysis
    print("\n--- [STAGE 2: PRE-ANALYSIS] ---")
    pre_ana_event = PreAnalysisEvent(
        output_folder=str(output_dir),
        video_path=str(VIDEO_PATH),
        resume=False,
        scene_detect=True,
        pre_analysis_enabled=True,
        pre_sample_nth=1,
        primary_seed_strategy="👤 By Face",
        face_ref_img_path=str(FACE_PATH),
        face_model_name="buffalo_l",
        tracker_model_name="sam3",
        best_frame_strategy="Largest Person",
        enable_face_filter=True,
        enable_subject_mask=True,
        min_mask_area_pct=1.0,
        sharpness_base_scale=2500.0,
        edge_strength_base_scale=100.0,
        compute_quality_score=True,
        compute_sharpness=True,
        compute_edge_strength=True,
        compute_contrast=True,
        compute_brightness=True,
        compute_entropy=True,
        compute_eyes_open=True,
        compute_yaw=True,
        compute_pitch=True,
        compute_face_sim=True,
        compute_subject_mask_area=True,
        compute_niqe=True,
        compute_phash=True
    )
    
    # Check for CUDA availability for the test
    import torch
    cuda_available = torch.cuda.is_available()
    if not cuda_available:
        print("⚠️ CUDA not available, some models might be slow or fail if they require GPU.")

    pre_ana_gen = execute_pre_analysis(
        pre_ana_event, progress_queue, cancel_event, logger, config, thumbnail_manager, 
        cuda_available=cuda_available,
        model_registry=model_registry
    )
    
    pre_ana_result = deque(pre_ana_gen, maxlen=1)[0]
    assert pre_ana_result.get("done"), f"Pre-analysis failed: {pre_ana_result.get('unified_log')}"
    
    scenes = pre_ana_result["scenes"]
    print(f"✅ Pre-analysis complete. Found {len(scenes)} scenes.")
    assert len(scenes) > 0, "No scenes found during pre-analysis"
    
    # 4. Propagation
    print("\n--- [STAGE 3: MASK PROPAGATION] ---")
    prop_event = PropagationEvent(
        output_folder=str(output_dir),
        video_path=str(VIDEO_PATH),
        scenes=scenes,
        analysis_params=pre_ana_event
    )
    
    prop_gen = execute_propagation(
        prop_event, progress_queue, cancel_event, logger, config, thumbnail_manager, 
        cuda_available=cuda_available, model_registry=model_registry
    )
    
    prop_result = deque(prop_gen, maxlen=1)[0]
    assert prop_result.get("done"), f"Propagation failed: {prop_result.get('unified_log')}"
    print("✅ Propagation complete.")
    
    # 5. Analysis
    print("\n--- [STAGE 4: FRAME ANALYSIS] ---")
    ana_gen = execute_analysis(
        prop_event, progress_queue, cancel_event, logger, config, thumbnail_manager, 
        cuda_available=cuda_available, model_registry=model_registry
    )
    
    ana_result = deque(ana_gen, maxlen=1)[0]
    assert ana_result.get("done"), f"Analysis failed: {ana_result.get('unified_log')}"
    print("✅ Analysis complete.")
    
    # 6. Final Verification
    print("\n--- [STAGE 5: FINAL VERIFICATION] ---")
    db_path = output_dir / "metadata.db"
    assert db_path.exists(), "Metadata database MISSING!"
        
    mask_dir = output_dir / "masks"
    assert mask_dir.exists(), "Mask directory missing!"
    assert any(mask_dir.iterdir()), "No masks generated!"
        
    print("\n🎉 E2E VERIFICATION SUCCESSFUL!")

if __name__ == "__main__":
    # Allow running this file directly for debugging
    sys.exit(pytest.main([__file__, "-v", "-s", "-m", "integration"]))
