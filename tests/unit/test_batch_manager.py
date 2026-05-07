import time

from core.batch_manager import BatchManager, BatchStatus


def test_batch_manager_add():
    bm = BatchManager()
    bm.add_paths(["test1.mp4", "test2.mp4"])
    assert len(bm.queue) == 2
    items = list(bm.queue.values())
    assert any(item.path == "test1.mp4" for item in items)
    assert items[1].status == BatchStatus.PENDING


def test_batch_manager_processing():
    bm = BatchManager()
    bm.add_paths(["test1.mp4"])
    item_id = list(bm.queue.keys())[0]

    def processor(item, progress):
        progress(0.5, "Halfway")
        return {"message": "Done"}

    bm.start_processing(processor)

    # Wait for completion
    timeout = 5
    start = time.time()
    while bm.is_running and time.time() - start < timeout:
        with bm.lock:
            if bm.queue[item_id].status == BatchStatus.COMPLETED:
                break
        time.sleep(0.1)

    assert bm.queue[item_id].status == BatchStatus.COMPLETED
    assert bm.queue[item_id].progress == 0.5
    assert bm.queue[item_id].message == "Done"


def test_batch_manager_failure():
    bm = BatchManager()
    bm.add_paths(["fail.mp4"])
    item_id = list(bm.queue.keys())[0]

    def processor(item, progress):
        raise ValueError("Error")

    bm.start_processing(processor)

    # Wait
    timeout = 5
    start = time.time()
    while bm.is_running and time.time() - start < timeout:
        with bm.lock:
            if bm.queue[item_id].status == BatchStatus.FAILED:
                break
        time.sleep(0.1)

    assert bm.queue[item_id].status == BatchStatus.FAILED
    assert bm.queue[item_id].message == "Error"
