# Mock triton
import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.modules["triton"] = MagicMock()
sys.modules["triton.language"] = MagicMock()

from core.managers.sam3 import SAM3Wrapper

REAL_VIDEO_PATH = Path("downloads/example clip 720p 2x.mp4")
REAL_BBOX = [77, 51, 102, 153]


def test_debug_sam3():
    wrapper = SAM3Wrapper()
    print("Wrapper initialized.")
    wrapper.init_video(str(REAL_VIDEO_PATH))
    print(f"Video initialized. Session ID: {wrapper.session_id}")

    mask = wrapper.add_bbox_prompt(frame_idx=0, obj_id=1, bbox_xywh=REAL_BBOX, img_size=(1280, 720))
    print(f"Mask obtained from add_prompt? {mask.any()}")

    print("Starting propagation...")
    propagated = list(wrapper.propagate(start_idx=0, max_frames=2, reverse=False))
    print(f"Propagated frames count: {len(propagated)}")
    for f, oid, m in propagated:
        print(f"Frame {f}, Obj {oid}, Mask valid? {m.any()}")


if __name__ == "__main__":
    test_debug_sam3()
