from pathlib import Path

import numpy as np

from core.managers.registry import ModelRegistry

REAL_VIDEO_PATH = Path("downloads/example clip 720p 2x.mp4")
REAL_BBOX = [77, 51, 102, 153]


def test_debug_sam3():
    registry = ModelRegistry()
    wrapper = registry.get_tracker("sam3")
    wrapper.init_video(str(REAL_VIDEO_PATH))
    mask = wrapper.add_bbox_prompt(frame_idx=0, obj_id=1, bbox_xywh=REAL_BBOX, img_size=(1280, 720))
    print(f"Mask obtained from add_prompt? {np.any(mask)}")

    propagated = list(wrapper.propagate(start_idx=0, max_frames=2, direction="forward"))
    print(f"Propagated len: {len(propagated)}")
