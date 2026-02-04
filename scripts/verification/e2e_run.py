import os
import sys
import threading
import json
import time
import shutil
from pathlib import Path
from queue import Queue
from collections import deque

import torch
import numpy as np

# Add root to path
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
from core.models import Scene

# --- ADDED: LOGGER REDIRECTION ---
class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def run_e2e_verification():
    output_dir = Path("verification/e2e_output")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Redirect stdout and stderr to terminal_log.txt
    log_file = output_dir / "terminal_log.txt"
    sys.stdout = Logger(log_file)
    sys.stderr = sys.stdout
    
    print("🚀 Starting E2E Verification with real data...")
    print(f"📝 Logging to: {log_file}")
    
    config = Config()
    # Disable memory watchdog for E2E test to prevent model unloading
    config.monitoring_memory_critical_threshold_mb = 64000
    config.log_level = "DEBUG" # Enable verbose logging
    
    # AppLogger now automatically creates 'run.log' in the output_dir
    logger = AppLogger(config, log_dir=output_dir, log_to_file=True)
    print(f"📝 Logs and results will be saved to: {output_dir}")
    
    progress_queue = Queue()
    cancel_event = threading.Event()
    model_registry = ModelRegistry(logger)
    thumbnail_manager = ThumbnailManager(logger, config)
    
    video_path = "downloads/example clip (2).mp4"
    face_path = "downloads/example face.png"
    
    # 1. Extraction
    print("\n--- [STAGE 1: EXTRACTION] ---")
    ext_event = ExtractionEvent(
        source_path=video_path,
        method="every_nth_frame", # CORRECTED: Get every 3rd frame
        interval="1.0",
        nth_frame=3,
        max_resolution="480", # 480p source for high-quality thumbs
        thumbnails_only=True,
        thumb_megapixels=0.5, # INCREASED: 0.5MP for better face recognition
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
    
    # Debug: Check frame map
    frame_map_path = output_dir / "frame_map.json"
    if frame_map_path.exists():
        frame_map = json.loads(frame_map_path.read_text())
        print(f"✅ Extraction complete. Mapped {len(frame_map)} frames.")
    else:
        print("❌ frame_map.json missing!")
        return False
    
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
    
    # Enable Half-Precision for E2E if CUDA is available
    if torch.cuda.is_available():
        print("💡 Enabling half-precision (FP16/BF16) optimizations for verification.")
        torch.set_float32_matmul_precision("medium") # Balance speed/precision
    
    pre_ana_gen = execute_pre_analysis(
        pre_ana_event, progress_queue, cancel_event, logger, config, thumbnail_manager, 
        cuda_available=torch.cuda.is_available(),
        model_registry=model_registry
    )
    
    pre_ana_result = deque(pre_ana_gen, maxlen=1)[0]
    if not pre_ana_result.get("done"):
        print(f"❌ Pre-analysis failed: {pre_ana_result.get('unified_log')}")
        return False
    
    scenes = pre_ana_result["scenes"]
    print(f"✅ Pre-analysis complete. Found {len(scenes)} scenes.")
    
    # Debug: Print seeding results
    for i, scene in enumerate(scenes):
        seed = scene.get("seed_result", {})
        bbox = seed.get("bbox")
        details = seed.get("details", {})
        print(f"  Scene {i}: best_frame={scene.get('best_frame')}, bbox={bbox}, type={details.get('type')}, face_sim={details.get('seed_face_sim')}")

    
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
    print(f"📝 Full terminal log saved to: {log_file}")
    return True

if __name__ == "__main__":
    success = run_e2e_verification()
    sys.exit(0 if success else 1)
