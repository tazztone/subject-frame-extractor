import threading
import time
from pathlib import Path
from queue import Queue
from unittest.mock import MagicMock, patch

import numpy as np

from core.application_state import ApplicationState
from core.managers import ThumbnailManager
from core.managers.analysis import AnalysisPipeline
from core.models import AnalysisParameters


def test_application_state_concurrent_updates():
    """Test concurrent updates to ApplicationState (stress test)."""
    state = ApplicationState()
    num_threads = 10
    updates_per_thread = 100

    def worker(thread_id):
        for i in range(updates_per_thread):
            data = {"id": f"{thread_id}-{i}", "score": i}
            state.all_frames_data.append(data)
            time.sleep(0.0001)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(state.all_frames_data) <= num_threads * updates_per_thread


def test_thumbnail_manager_concurrent_access(tmp_path):
    """Test concurrent access to ThumbnailManager's LRU cache."""
    from core.config import Config

    config = Config()
    config.cache_size = 5
    manager = ThumbnailManager(MagicMock(), config)

    num_threads = 5
    accesses_per_thread = 50

    files = []
    for i in range(10):
        p = tmp_path / f"{i}.jpg"
        p.touch()
        files.append(p)

    def worker():
        import random
        from unittest.mock import patch

        with patch("PIL.Image.open") as mock_open:
            mock_img = MagicMock()
            mock_img.convert.return_value = np.zeros((10, 10, 3), dtype=np.uint8)
            mock_open.return_value.__enter__.return_value = mock_img

            for _ in range(accesses_per_thread):
                f = random.choice(files)
                manager.get(f)

    threads = [threading.Thread(target=worker) for _ in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(manager.cache) <= 5


def test_analysis_pipeline_concurrency(tmp_path):
    """Test that AnalysisPipeline handles concurrent frame processing correctly."""
    config = MagicMock()
    config.retry_max_attempts = 1
    config.retry_backoff_seconds = [1]

    params = AnalysisParameters(
        output_folder=str(tmp_path), video_path="test.mp4", compute_face_sim=True, compute_niqe=True
    )

    logger = MagicMock()
    tm = MagicMock()
    registry = MagicMock()

    with patch("core.managers.analysis.Database") as mock_db_cls, patch("core.managers.analysis.OperatorRegistry"):
        pipeline = AnalysisPipeline(config, logger, params, Queue(), threading.Event(), tm, registry)
        pipeline.db = mock_db_cls.return_value
        pipeline.face_analyzer = MagicMock()

        num_frames = 20
        num_threads = 4

        def worker(indices):
            for i in indices:
                path = Path(f"frame_{i:06d}.png")
                # Mock preloaded data
                preloaded = {"img": np.zeros((10, 10, 3), dtype=np.uint8), "mask_thumb": None, "mask_meta": {}}
                pipeline._process_single_frame(path, {"quality": True}, preloaded=preloaded)

        with (
            patch("core.managers.analysis.run_operators") as mock_run_ops,
            patch("core.managers.analysis.cv2.cvtColor"),
        ):
            mock_run_ops.return_value = {}  # Mock results

            threads = []
            chunk_size = num_frames // num_threads
            for i in range(num_threads):
                indices = range(i * chunk_size, (i + 1) * chunk_size)
                t = threading.Thread(target=worker, args=(indices,))
                threads.append(t)
                t.start()

            for t in threads:
                t.join()

        # Each frame should have inserted metadata once
        assert pipeline.db.insert_metadata.call_count == num_frames
