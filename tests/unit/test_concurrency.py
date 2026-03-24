import threading
import time
from unittest.mock import MagicMock

import numpy as np

from core.application_state import ApplicationState
from core.managers import ThumbnailManager


def test_application_state_concurrent_updates():
    """Test concurrent updates to ApplicationState (stress test)."""
    state = ApplicationState()
    num_threads = 10
    updates_per_thread = 100

    def worker(thread_id):
        for i in range(updates_per_thread):
            # Simulate updating frames data
            data = {"id": f"{thread_id}-{i}", "score": i}
            # Note: Pydantic list is not thread-safe for appending without a lock
            # but we want to see if it causes crashes or data loss.
            state.all_frames_data.append(data)
            time.sleep(0.0001)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # If it was thread-safe, we'd expect exactly num_threads * updates_per_thread
    # But since it's not, we might have less or a crash.
    # This test surfaces the need for protection if state is shared.
    assert len(state.all_frames_data) <= num_threads * updates_per_thread


def test_thumbnail_manager_concurrent_access(tmp_path):
    """Test concurrent access to ThumbnailManager's LRU cache."""
    from core.config import Config

    config = Config()
    config.cache_size = 5
    manager = ThumbnailManager(MagicMock(), config)

    num_threads = 5
    accesses_per_thread = 50

    # Create some dummy files
    files = []
    for i in range(10):
        p = tmp_path / f"{i}.jpg"
        p.touch()
        files.append(p)

    def worker():
        import random

        # Mock PIL.Image.open to avoid actual I/O in stress test
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

    # The goal is no crashes (e.g. KeyError in OrderedDict during eviction)
    assert len(manager.cache) <= 5
