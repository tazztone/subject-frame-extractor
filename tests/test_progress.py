import pytest
from unittest.mock import MagicMock
from queue import Queue
from core.progress import AdvancedProgressTracker

def test_progress_tracker():
    progress_func = MagicMock()
    queue = Queue()
    logger = MagicMock()

    tracker = AdvancedProgressTracker(progress_func, queue, logger, "Test")
    tracker.throttle_interval = 0 # Disable throttle to ensure every update is sent

    tracker.start(100, "Starting")
    assert tracker.total == 100
    assert tracker.done == 0

    tracker.step(10, "Stepping")
    assert tracker.done == 10

    messages = []
    while not queue.empty():
        messages.append(queue.get())

    assert len(messages) >= 1
    # Check the last message which should correspond to the step
    last_msg = messages[-1]
    assert last_msg['progress']['fraction'] == 0.1

    tracker.set(50)
    assert tracker.done == 50

    tracker.set_stage("New Stage")
    assert tracker.stage == "New Stage"

    tracker.done_stage("Done")
    assert tracker.done == 100
