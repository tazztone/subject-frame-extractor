import time

import pytest

from core.batch_manager import BatchManager, BatchStatus


@pytest.fixture
def bm():
    return BatchManager()


def test_batch_manager_add(bm):
    bm.add_paths(["test1.mp4", "test2.mp4"])
    assert len(bm.queue) == 2
    assert bm.queue[0].path == "test1.mp4"
    assert bm.queue[1].status == BatchStatus.PENDING
    assert isinstance(bm.queue[0].id, str)


def test_batch_manager_get_queue_snapshot(bm):
    bm.add_paths(["test1.mp4"])
    snapshot = bm.get_queue_snapshot()
    assert len(snapshot) == 1
    assert snapshot[0].path == "test1.mp4"
    # Ensure it's a new list but contains same objects
    assert snapshot is not bm.queue
    assert snapshot[0] is bm.queue[0]


def test_batch_manager_get_status_list(bm):
    bm.add_paths(["test1.mp4"])
    status_list = bm.get_status_list()
    assert len(status_list) == 1
    assert status_list[0] == ["test1.mp4", "Pending", 0.0, "Waiting..."]


def test_batch_manager_clear_completed(bm):
    bm.add_paths(["p1.mp4", "p2.mp4", "p3.mp4", "p4.mp4"])
    bm.queue[0].status = BatchStatus.COMPLETED
    bm.queue[1].status = BatchStatus.FAILED
    bm.queue[2].status = BatchStatus.CANCELLED
    bm.queue[3].status = BatchStatus.PENDING

    bm.clear_completed()
    assert len(bm.queue) == 1
    assert bm.queue[0].path == "p4.mp4"


def test_batch_manager_clear_all(bm):
    bm.add_paths(["test.mp4"])
    bm.clear_all()
    assert len(bm.queue) == 0


def test_batch_manager_update_progress(bm):
    bm.add_paths(["test.mp4"])
    item_id = bm.queue[0].id
    bm.update_progress(item_id, 0.75, "Almost there")
    assert bm.queue[0].progress == 0.75
    assert bm.queue[0].message == "Almost there"


def test_batch_manager_set_status(bm):
    bm.add_paths(["test.mp4"])
    item_id = bm.queue[0].id
    bm.set_status(item_id, BatchStatus.PROCESSING, "Busy")
    assert bm.queue[0].status == BatchStatus.PROCESSING
    assert bm.queue[0].message == "Busy"


def test_batch_manager_processing(bm):
    bm.add_paths(["test1.mp4"])

    def processor(item, progress):
        progress(0.5, "Halfway")
        return {"message": "Done"}

    bm.start_processing(processor)

    # Wait for completion
    timeout = 5
    start = time.time()
    while bm.is_running and time.time() - start < timeout:
        time.sleep(0.1)

    assert bm.queue[0].status == BatchStatus.COMPLETED
    assert bm.queue[0].progress == 0.5
    assert bm.queue[0].message == "Done"


def test_batch_manager_failure(bm):
    bm.add_paths(["fail.mp4"])

    def processor(item, progress):
        raise ValueError("Error")

    bm.start_processing(processor)

    # Wait
    timeout = 5
    start = time.time()
    while bm.is_running and time.time() - start < timeout:
        time.sleep(0.1)

    assert bm.queue[0].status == BatchStatus.FAILED
    assert bm.queue[0].message == "Error"


def test_batch_manager_stop_processing(bm):
    bm.add_paths(["p1.mp4", "p2.mp4"])

    def slow_processor(item, progress):
        time.sleep(1)
        return {"message": "Slow"}

    bm.start_processing(slow_processor)
    time.sleep(0.1)  # Let it start
    assert bm.is_running

    bm.stop_processing()

    # Wait for scheduler to stop
    timeout = 5
    start = time.time()
    while bm.is_running and time.time() - start < timeout:
        time.sleep(0.1)

    assert not bm.is_running
    assert bm.stop_event.is_set()


def test_batch_manager_start_empty(bm):
    # Should not crash or hang
    bm.start_processing(lambda x, y: None)
    assert not bm.is_running
