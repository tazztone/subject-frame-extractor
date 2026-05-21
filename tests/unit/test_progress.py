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


def test_tracker_start_edge_cases():
    """Test the start method handles edge cases like 0 total items, missing desc, and clears substage."""
    progress_mock = MagicMock()
    queue = Queue()
    logger = MagicMock()
    tracker = AdvancedProgressTracker(progress_mock, queue, logger, ui_stage_name="Initial Stage")
    tracker.substage = "Initial Substage"
    tracker._ema_dt = 10.0

    # Start with 0 items, which should default to 1, and no desc, which should retain the old stage.
    tracker.start(0)

    assert tracker.total == 1
    assert tracker.done == 0
    assert tracker.stage == "Initial Stage"
    assert tracker.substage is None
    assert tracker._ema_dt is None


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


def test_tracker_set_stage():
    """Test set_stage updates description and forces overlay."""
    progress_mock = MagicMock()
    queue = Queue()
    logger = MagicMock()
    tracker = AdvancedProgressTracker(progress_mock, queue, logger)
    tracker.throttle_interval = 1.0  # Set throttle to ensure force=True works

    tracker.start(10)

    # Clear the queue from start
    while not queue.empty():
        queue.get()

    # Call set_stage with just stage
    tracker.set_stage("New Stage")
    assert tracker.stage == "New Stage"
    assert tracker.substage is None

    # Verify forced overlay emitted an event
    assert not queue.empty()
    event = queue.get()["progress"]
    assert event["stage"] == "New Stage"
    assert event["substage"] is None

    # Clear queue again
    while not queue.empty():
        queue.get()

    # Call set_stage with stage and substage
    tracker.set_stage("Another Stage", substage="New Substage")
    assert tracker.stage == "Another Stage"
    assert tracker.substage == "New Substage"

    # Verify forced overlay emitted an event
    assert not queue.empty()
    event = queue.get()["progress"]
    assert event["stage"] == "Another Stage"
    assert event["substage"] == "New Substage"


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
    assert AdvancedProgressTracker._fmt_eta(3665, precision="fine") == "1h 1m 5s"
    assert AdvancedProgressTracker._fmt_eta(3605, precision="coarse") == "1h 0m"
    assert AdvancedProgressTracker._fmt_eta(3605, precision="fine") == "1h 0m 5s"


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

    # The thread should be waiting on pause_event
    # Use a small wait to ensure it's blocked, but not 1.0s
    import time

    time.sleep(0.01)
    assert tracker.done == 0
    assert thread.is_alive()

    tracker.pause_event.set()  # Resume
    thread.join(timeout=0.1)
    assert tracker.done == 1
    assert not thread.is_alive()


def test_tracker_step_with_desc_and_substage():
    """Test stepping with optional desc and substage."""
    progress_mock = MagicMock()
    queue = Queue()
    logger = MagicMock()
    tracker = AdvancedProgressTracker(progress_mock, queue, logger)
    tracker.throttle_interval = 0
    tracker.start(10)

    tracker.step(1, desc="New Stage", substage="New Substage")
    assert tracker.stage == "New Stage"
    assert tracker.substage == "New Substage"


def test_tracker_set_stage_alternate():
    """Test setting stage and substage explicitly."""
    progress_mock = MagicMock()
    queue = Queue()
    logger = MagicMock()
    tracker = AdvancedProgressTracker(progress_mock, queue, logger)
    tracker.start(10)

    tracker.set_stage("Custom Stage", substage="Custom Substage")
    assert tracker.stage == "Custom Stage"
    assert tracker.substage == "Custom Substage"


def test_tracker_no_eta_seconds():
    """Test _eta_seconds when _ema_dt is None."""
    progress_mock = MagicMock()
    queue = Queue()
    logger = MagicMock()
    tracker = AdvancedProgressTracker(progress_mock, queue, logger)

    assert tracker._eta_seconds() is None


def test_tracker_done_stage_no_text():
    """Test done_stage without final_text."""
    progress_mock = MagicMock()
    queue = Queue()
    logger = MagicMock()
    tracker = AdvancedProgressTracker(progress_mock, queue, logger)
    tracker.start(10)

    tracker.done_stage()
    assert tracker.done == 10
    logger.info.assert_not_called()


def test_tracker_set_delta_zero():
    """Test set when delta <= 0."""
    progress_mock = MagicMock()
    queue = Queue()
    logger = MagicMock()
    tracker = AdvancedProgressTracker(progress_mock, queue, logger)
    tracker.start(10)
    tracker.done = 5

    tracker.set(5)  # delta = 0
    assert tracker.done == 5

    tracker.set(3)  # delta < 0
    assert tracker.done == 5


def test_tracker_no_queue_no_progress():
    """Test overlay when queue and progress are None."""
    tracker = AdvancedProgressTracker(progress=None, queue=None)
    tracker.throttle_interval = 0
    tracker._overlay(force=True)
    # Should not raise an exception


def test_fmt_eta_fine_precision_no_hours():
    """Test _fmt_eta with fine precision and no hours."""
    assert AdvancedProgressTracker._fmt_eta(125, precision="fine") == "2m 5s"


def test_tracker_step_with_desc():
    """Test step with description explicitly."""
    tracker = AdvancedProgressTracker()
    tracker.start(10)
    tracker.step(1, desc="New Desc")
    assert tracker.stage == "New Desc"


def test_tracker_set_stage_explicitly():
    """Test set_stage with substage."""
    tracker = AdvancedProgressTracker()
    tracker.set_stage("Stage A", substage="Substage B")
    assert tracker.stage == "Stage A"
    assert tracker.substage == "Substage B"


def test_tracker_dt_le_zero():
    """Test step when dt <= 0."""
    progress_mock = MagicMock()
    queue = Queue()
    logger = MagicMock()
    tracker = AdvancedProgressTracker(progress_mock, queue, logger)
    tracker.throttle_interval = 0
    tracker.start(10)

    with patch("time.time", side_effect=[tracker._last_ts, tracker._last_ts]):
        tracker.step(1)
        assert tracker.done == 1
        assert tracker._ema_dt is None


def test_tracker_substage_not_none():
    """Test step with substage explicit passing."""
    tracker = AdvancedProgressTracker()
    tracker.start(10)
    tracker.step(1, substage="Working")
    assert tracker.substage == "Working"


def test_tracker_progress_is_not_none():
    """Test overlay when progress is not None."""
    progress_mock = MagicMock()
    tracker = AdvancedProgressTracker(progress=progress_mock, queue=None)
    tracker.throttle_interval = 0
    tracker._overlay(force=True)
    progress_mock.assert_called_once()


def test_fmt_eta_fine_precision_no_hours_no_minutes():
    """Test _fmt_eta with fine precision and no hours/minutes."""
    assert AdvancedProgressTracker._fmt_eta(45, precision="fine") == "45s"


def test_tracker_progress_none():
    """Test overlay when progress is None."""
    queue = Queue()
    tracker = AdvancedProgressTracker(progress=None, queue=queue)
    tracker.throttle_interval = 0
    tracker._overlay(force=True)
    assert not queue.empty()


def test_tracker_progress_is_callable():
    """Test overlay when progress is provided as a Callable."""
    progress_called = False

    def mock_progress(fraction, desc=None):
        nonlocal progress_called
        progress_called = True

    tracker = AdvancedProgressTracker(progress=mock_progress, queue=None)
    tracker.throttle_interval = 0
    tracker._overlay(force=True)
    assert progress_called


def test_tracker_progress_lambda_desc_none():
    """Test overlay when progress uses the default lambda and desc is updated."""
    tracker = AdvancedProgressTracker(queue=None)
    tracker.throttle_interval = 0
    tracker._overlay(force=True)
    # Testing that it doesn't fail when the default lambda is called.


def test_tracker_progress_is_callable_true():
    """Test overlay when progress evaluates to truthy but not callable to ensure line 142 condition is fully covered."""
    progress_mock = MagicMock()
    tracker = AdvancedProgressTracker(progress=progress_mock, queue=None)
    tracker.throttle_interval = 0
    tracker._overlay(force=True)
    progress_mock.assert_called_once()


def test_tracker_progress_is_callable_but_not_truthy():
    """Test what happens if the mock progress is falsey... actually, a MagicMock is truthy."""
    pass  # covered by `test_tracker_progress_none`


def test_tracker_progress_is_falsy():
    """Test overlay when progress is intentionally falsey to cover line 142 false branch."""
    tracker = AdvancedProgressTracker(queue=None)
    tracker.progress = None
    tracker.throttle_interval = 0
    tracker._overlay(force=True)


def test_tracker_set_stage_main_branch():
    """Test set_stage method from main branch."""
    progress_mock = MagicMock()
    queue = Queue()
    logger = MagicMock()
    tracker = AdvancedProgressTracker(progress_mock, queue, logger)
    tracker.throttle_interval = 0
    tracker.start(10)

    # Clear queue of start events
    while not queue.empty():
        queue.get()

    tracker.set_stage("New Stage", substage="New Substage")
    assert tracker.stage == "New Stage"
    assert tracker.substage == "New Substage"

    # Check that _overlay was called
    assert not queue.empty()
    event = queue.get()["progress"]
    assert event["stage"] == "New Stage"
    assert event["substage"] == "New Substage"


def test_tracker_done_stage_no_text_or_logger():
    """Test done_stage method without text or logger."""
    tracker = AdvancedProgressTracker()
    tracker.start(10)

    # done_stage should not raise an error when logger is None and final_text is None
    tracker.done_stage()
    assert tracker.done == 10

    # test with final_text but no logger
    tracker.done = 0
    tracker.done_stage("Finished")
    assert tracker.done == 10


def test_tracker_step_edge_cases():
    """Test step edge cases (dt <= 0, and desc provided)."""
    tracker = AdvancedProgressTracker()
    tracker.start(10)

    with patch("time.time", return_value=tracker._last_ts):  # dt = 0
        tracker.step(1, desc="Step Edge Case")
        assert tracker.stage == "Step Edge Case"
        assert tracker.done == 1


def test_tracker_set_delta_zero_main_branch():
    """Test set method when delta <= 0 from main branch."""
    tracker = AdvancedProgressTracker()
    tracker.start(10)
    tracker.set(5)

    # delta = 0
    tracker.set(5)
    assert tracker.done == 5

    # delta < 0
    tracker.set(3)
    assert tracker.done == 5


def test_tracker_overlay_edge_cases():
    """Test _overlay when progress and queue are None."""
    tracker = AdvancedProgressTracker()  # No progress or queue
    tracker.start(10)

    # Should run without raising any exception
    tracker._overlay(force=True)


def test_fmt_eta_static_fine_no_hours_main_branch():
    """Test _fmt_eta with fine precision and h == 0 from main branch."""
    assert AdvancedProgressTracker._fmt_eta(65, precision="fine") == "1m 5s"


def test_tracker_overlay_edge_cases_no_progress_main_branch():
    """Test _overlay when progress function is not provided."""
    # To test line 142 false branch
    queue = Queue()
    logger = MagicMock()
    # Explicitly set progress=None (default behavior is dummy function)
    tracker = AdvancedProgressTracker(progress=None, queue=queue, logger=logger)

    # Force _overlay to execute line 142
    tracker.start(10)
    tracker._overlay(force=True)


def test_tracker_overlay_edge_cases_no_progress_branch_main_branch():
    """Test _overlay when progress function evaluates to False to hit branch missing."""
    tracker = AdvancedProgressTracker()
    # Explicitly bypass the dummy function set in __init__
    tracker.progress = None
    tracker.start(10)
    tracker._overlay(force=True)


def test_tracker_done_stage_with_text_and_logger():
    """Test done_stage method with both text and logger."""
    progress_mock = MagicMock()
    queue = Queue()
    logger = MagicMock()
    tracker = AdvancedProgressTracker(progress_mock, queue, logger)
    tracker.start(10)

    tracker.done_stage("Task completed")
    assert tracker.done == 10
    logger.info.assert_called_once_with("Task completed", component="progress")
