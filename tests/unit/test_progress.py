from queue import Queue
from unittest.mock import MagicMock, patch

import pytest

from core.progress import AdvancedProgressTracker, ProgressEvent


def test_progress_event_model():
    """Test the Pydantic model for progress events."""
    event = ProgressEvent(stage="Testing", done=5, total=10, fraction=0.5)
    assert event.stage == "Testing"
    assert event.fraction == 0.5
    assert event.eta_formatted == "—"


def test_tracker_initialization():
    """Test tracker initialization and default values."""
    progress_mock = MagicMock()
    queue = Queue()
    logger = MagicMock()
    tracker = AdvancedProgressTracker(progress_mock, queue, logger, ui_stage_name="Initial")

    assert tracker.stage == "Initial"
    assert tracker.total == 1
    assert tracker.done == 0


def test_tracker_start():
    """Test the start method resets state."""
    progress_mock = MagicMock()
    queue = Queue()
    logger = MagicMock()
    tracker = AdvancedProgressTracker(progress_mock, queue, logger)

    tracker.start(100, desc="Starting")
    assert tracker.total == 100
    assert tracker.done == 0
    assert tracker.stage == "Starting"

    # Check that it emitted an update immediately (force=True)
    assert not queue.empty()
    event = queue.get()["progress"]
    assert event["stage"] == "Starting"
    assert event["total"] == 100


def test_tracker_step_and_eta():
    """Test stepping through progress and ETA estimation."""
    progress_mock = MagicMock()
    queue = Queue()
    logger = MagicMock()

    # We need to account for multiple time.time() calls:
    # 1. __init__ (1 call)
    # 2. start (2 calls: one for _t0, one for _overlay)
    # 3. step (2 calls: one for now, one for _overlay)
    # To keep dt=1.0, we want:
    # start's _t0 = 101.0
    # step's now = 102.0

    times = [
        100.0,  # __init__
        101.0,
        101.1,  # start (_t0, _overlay)
        102.1,
        102.2,  # step 1 (now, _overlay)
        103.2,
        103.3,  # step 2 (now, _overlay)
    ]

    with patch("time.time", side_effect=times):
        tracker = AdvancedProgressTracker(progress_mock, queue, logger)
        tracker.throttle_interval = 0

        tracker.start(10)
        # self._last_ts is now 101.0

        # Step 1: now=102.1, dt = 102.1 - 101.0 = 1.1
        tracker.step(1)
        assert tracker.done == 1
        assert tracker._ema_dt == pytest.approx(1.1)

        # Step 2: now=103.2, dt = 103.2 - 102.1 = 1.1
        tracker.step(1)
        # EMA: 0.2 * 1.1 + 0.8 * 1.1 = 1.1
        assert tracker._ema_dt == pytest.approx(1.1)

        # ETA: 8 items remaining * 1.1s = 8.8s
        events = list(queue.queue)
        progress_events = [q["progress"] for q in events if "progress" in q]
        event = progress_events[-1]
        assert event["eta_seconds"] == pytest.approx(8.8)


def test_tracker_throttling():
    """Test that updates are throttled."""
    progress_mock = MagicMock()
    queue = Queue()
    logger = MagicMock()
    tracker = AdvancedProgressTracker(progress_mock, queue, logger)
    tracker.throttle_interval = 1.0  # 1 second throttle

    tracker.start(100)
    # Clear queue after start()'s forced update
    while not queue.empty():
        queue.get()

    # Multiple steps in quick succession
    tracker.step(1)
    tracker.step(1)
    tracker.step(1)

    # Queue should be empty because 1 second hasn't passed
    assert queue.empty()

    # Force an update
    tracker._overlay(force=True)
    assert not queue.empty()


def test_tracker_set_and_done():
    """Test set() and done_stage() methods."""
    progress_mock = MagicMock()
    queue = Queue()
    logger = MagicMock()
    tracker = AdvancedProgressTracker(progress_mock, queue, logger)
    tracker.throttle_interval = 0

    tracker.start(10)
    tracker.set(5, substage="Halfway")
    assert tracker.done == 5
    assert tracker.substage == "Halfway"

    tracker.done_stage("All finished")
    assert tracker.done == 10
    logger.info.assert_called_with("All finished", component="progress")


def test_fmt_eta_static():
    """Test the static _fmt_eta helper."""
    assert AdvancedProgressTracker._fmt_eta(None) == "—"
    assert AdvancedProgressTracker._fmt_eta(45) == "45s"
    assert AdvancedProgressTracker._fmt_eta(125) == "2m 5s"
    assert AdvancedProgressTracker._fmt_eta(3665) == "1h 1m"


def test_tracker_pause_resume():
    """Test that stepping honors the pause event."""
    progress_mock = MagicMock()
    queue = Queue()
    logger = MagicMock()
    tracker = AdvancedProgressTracker(progress_mock, queue, logger)

    tracker.pause_event.clear()  # Paused

    # Running step in a thread because it should block
    import threading

    def do_step():
        tracker.step(1)

    thread = threading.Thread(target=do_step)
    thread.start()

    import time as _t

    deadline = _t.monotonic() + 1.0
    while thread.is_alive() and tracker.done == 0 and _t.monotonic() < deadline:
        _t.sleep(0.001)
    assert tracker.done == 0
    assert thread.is_alive()

    tracker.pause_event.set()  # Resume
    thread.join(timeout=1.0)
    assert tracker.done == 1
    assert not thread.is_alive()
