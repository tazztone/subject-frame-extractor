import os
import sys
import threading
import json
import time
from pathlib import Path
from queue import Queue
from collections import deque

# Add root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

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
from core.models import Scene

def run_e2e_verification():
    print("🚀 Starting E2E Verification with real data...")
    
    config = Config()
    # Disable file logging for dry run
    logger = AppLogger(config, log_to_file=False)
    progress_queue = Queue()
    cancel_event = threading.Event()
    model_registry = ModelRegistry(logger)
    thumbnail_manager = ThumbnailManager(logger, config)
    
    video_path = "downloads/example clip (2).mp4"
    face_path = "downloads/example face.png"
    
    output_dir = Path("verification/e2e_output")
    if output_dir.exists():
        import shutil
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Extraction
    print("\n--- [STAGE 1: EXTRACTION] ---")
    ext_event = ExtractionEvent(
        source_path=video_path,
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
    if not ext_result.get("done"):
        print(f"❌ Extraction failed: {ext_result.get('unified_log')}")
        return False
    print("✅ Extraction complete.")
    
    # 2. Pre-Analysis
    print("\n--- [STAGE 2: PRE-ANALYSIS] ---")
    pre_ana_event = PreAnalysisEvent(
        output_folder=str(output_dir),
        video_path=video_path,
        resume=False,
        scene_detect=True,
        pre_analysis_enabled=True,
        pre_sample_nth=1,
        primary_seed_strategy="👤 By Face",
        face_ref_img_path=face_path,
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
    
    pre_ana_gen = execute_pre_analysis(
        pre_ana_event, progress_queue, cancel_event, logger, config, thumbnail_manager, 
        cuda_available=True, # Assume true for E2E
        model_registry=model_registry
    )
    
    pre_ana_result = deque(pre_ana_gen, maxlen=1)[0]
    if not pre_ana_result.get("done"):
        print(f"❌ Pre-analysis failed: {pre_ana_result.get('unified_log')}")
        return False
    
    scenes = pre_ana_result["scenes"]
    print(f"✅ Pre-analysis complete. Found {len(scenes)} scenes.")
    
    # 3. Propagation
    print("\n--- [STAGE 3: MASK PROPAGATION] ---")
    prop_event = PropagationEvent(
        output_folder=str(output_dir),
        video_path=video_path,
        scenes=scenes,
        analysis_params=pre_ana_event
    )
    
    prop_gen = execute_propagation(
        prop_event, progress_queue, cancel_event, logger, config, thumbnail_manager, 
        cuda_available=True, model_registry=model_registry
    )
    
    prop_result = deque(prop_gen, maxlen=1)[0]
    if not prop_result.get("done"):
        print(f"❌ Propagation failed: {prop_result.get('unified_log')}")
        return False
    print("✅ Propagation complete.")
    
    # 4. Analysis
    print("\n--- [STAGE 4: FRAME ANALYSIS] ---")
    ana_gen = execute_analysis(
        prop_event, progress_queue, cancel_event, logger, config, thumbnail_manager, 
        cuda_available=True, model_registry=model_registry
    )
    
    ana_result = deque(ana_gen, maxlen=1)[0]
    if not ana_result.get("done"):
        print(f"❌ Analysis failed: {ana_result.get('unified_log')}")
        return False
    print("✅ Analysis complete.")
    
    # 5. Final Check
    print("\n--- [STAGE 5: FINAL VERIFICATION] ---")
    db_path = output_dir / "metadata.db"
    if db_path.exists():
        print(f"✅ Metadata database found at {db_path}")
    else:
        print("❌ Metadata database MISSING!")
        return False
        
    mask_dir = output_dir / "masks"
    if mask_dir.exists() and any(mask_dir.iterdir()):
        print(f"✅ Masks generated in {mask_dir}")
    else:
        print("❌ No masks generated!")
        return False
        
    print("\n🎉 E2E VERIFICATION SUCCESSFUL!")
    return True

if __name__ == "__main__":
    success = run_e2e_verification()
    sys.exit(0 if success else 1)