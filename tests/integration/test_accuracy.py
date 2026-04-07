from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

pytestmark = [
    pytest.mark.gpu_e2e,
    pytest.mark.slow,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available for accuracy tests"),
]


@pytest.fixture(scope="module")
def real_video():
    """Provides a guaranteed real-world video from SAM3 assets."""
    project_root = Path(__file__).parents[2]
    video_path = project_root / "SAM3_repo" / "assets" / "videos" / "bedroom.mp4"

    if not video_path.exists():
        pytest.skip(f"Could not find {video_path}")

    # The bedroom.mp4 asset is 960x540.
    # Central object (bed/pillow) is roughly in this region.
    bbox = [240, 135, 480, 270]  # x, y, w, h
    return str(video_path), bbox, (960, 540)


def test_mask_iou_accuracy(real_video, module_model_registry):
    """
    Verify that SAM3 generates a mask that reasonably matches the input bbox.
    This serves as a baseline check for model correctness.
    """
    video_path, bbox, img_size = real_video
    wrapper = module_model_registry.get_tracker("sam3")

    try:
        # 1. Initialize
        wrapper.reset_session()
        wrapper.init_video(video_path)

        # 2. Add Prompt (img_size is now mandatory for normalization)
        mask = wrapper.add_bbox_prompt(frame_idx=0, obj_id=1, bbox_xywh=bbox, img_size=img_size)

        assert mask is not None
        assert mask.any(), "Mask should not be empty"

        # 3. Calculate IoU between mask (as bbox) and input bbox
        # mask is (H, W) boolean. Get bounding box of mask.
        rows, cols = np.where(mask)
        if len(rows) == 0:
            pytest.fail("Generated mask is empty")

        m_y1, m_y2 = rows.min(), rows.max()
        m_x1, m_x2 = cols.min(), cols.max()
        mask_bbox = [m_x1, m_y1, m_x2 - m_x1, m_y2 - m_y1]  # xywh

        def get_iou(boxA, boxB):
            xA = max(boxA[0], boxB[0])
            yA = max(boxA[1], boxB[1])
            xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
            yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
            interArea = max(0, xB - xA) * max(0, yB - yA)
            boxAArea = boxA[2] * boxA[3]
            boxBArea = boxB[2] * boxB[3]
            iou = interArea / float(boxAArea + boxBArea - interArea)
            return iou

        iou = get_iou(bbox, mask_bbox)
        print(f"Mask IoU: {iou:.4f}")

        # SAM3 should be very precise for a clean bbox on frame 0
        assert iou > 0.75, f"Mask IoU too low ({iou:.4f}). Potential model regression."

    finally:
        wrapper.close_session()


def test_propagation_stability(real_video, module_model_registry, tmp_path):
    """
    Verify that propagation maintains a non-empty mask over several frames.
    """
    import threading
    from queue import Queue

    from core.config import Config
    from core.logger import AppLogger

    video_path, bbox, img_size = real_video
    wrapper = module_model_registry.get_tracker("sam3")

    config = Config(logs_dir=str(tmp_path / "logs"))
    logger = AppLogger(config, log_to_console=False, log_to_file=False)
    from core.models import AnalysisParameters
    from core.scene_utils.mask_propagator import MaskPropagator

    params = AnalysisParameters(source_path=video_path, output_folder=str(tmp_path))

    # Initial Add Prompt for propagation
    wrapper.reset_session()

    try:
        propagator = MaskPropagator(
            params=params,
            dam_tracker=wrapper,
            cancel_event=threading.Event(),
            progress_queue=Queue(),
            config=config,
            logger=logger,
            device="cuda",
        )

        # Read first 5 frames for a quick but real test
        cap = cv2.VideoCapture(video_path)
        frames = []
        for _ in range(5):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()

        # Propagate
        frame_numbers = list(range(len(frames)))
        h, w = frames[0].shape[:2]
        res_masks, res_areas, res_empties, res_errors = propagator.propagate_video(
            video_path=str(video_path),
            frame_numbers=frame_numbers,
            prompts=[{"frame": 0, "bbox": bbox}],
            frame_size=(w, h),
            frame_map={},
        )

        # Convert dictionaries to lists for compatibility with existing assertions
        masks = [res_masks[i] for i in frame_numbers]
        areas = [res_areas[i] for i in frame_numbers]
        errors = [res_errors[i] for i in frame_numbers]

        # Assertions
        assert len(masks) == len(frames)
        assert not any(errors), f"Errors during propagation: {errors}"

        # Check that masks don't disappear completely
        for i, m in enumerate(masks):
            assert m is not None, f"Frame {i} has no mask"
            assert m.any(), f"Frame {i} mask is empty"
            # area is percentage (0-100)
            assert areas[i] > 0.5, f"Frame {i} mask area too small ({areas[i]:.2f}%)"

    finally:
        wrapper.reset_session()
