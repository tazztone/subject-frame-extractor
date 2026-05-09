import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional

import psutil
import torch


class BatchStatus(Enum):
    PENDING = "Pending"
    PROCESSING = "Processing"
    COMPLETED = "Completed"
    FAILED = "Failed"
    CANCELLED = "Cancelled"


@dataclass
class BatchItem:
    """Represents a single item in the batch processing queue."""

    id: str
    path: str
    params: Dict = field(default_factory=dict)
    status: BatchStatus = BatchStatus.PENDING
    progress: float = 0.0
    message: str = "Waiting..."
    output_path: str = ""
    error: str = ""


class BatchManager:
    """Manages a queue of batch processing tasks."""

    # TODO: Add resource-aware scheduling (wait for GPU or RAM availability)
    def __init__(self, logger=None):
        """Initializes the BatchManager."""
        self.queue: List[BatchItem] = []
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.executor: Optional[ThreadPoolExecutor] = None
        self.is_running = False
        self.active_items: Dict[str, BatchItem] = {}
        self.logger = logger

    def add_paths(self, paths: List[str]):
        """Adds a list of file paths to the batch queue."""
        with self.lock:
            for p in paths:
                item = BatchItem(id=str(uuid.uuid4()), path=p)
                self.queue.append(item)

    def get_queue_snapshot(self) -> List[BatchItem]:
        """Returns a thread-safe snapshot of the current queue."""
        with self.lock:
            return list(self.queue)

    def get_status_list(self) -> List[List]:
        """Returns a simplified list of status data for the UI."""
        with self.lock:
            return [[item.path, item.status.value, item.progress, item.message] for item in self.queue]

    def clear_completed(self):
        """Removes completed, failed, and cancelled items from the queue."""
        with self.lock:
            self.queue = [
                item
                for item in self.queue
                if item.status not in (BatchStatus.COMPLETED, BatchStatus.FAILED, BatchStatus.CANCELLED)
            ]

    def clear_all(self):
        """Clears all items from the queue."""
        with self.lock:
            self.queue = []

    def update_progress(self, item_id: str, fraction: float, message: Optional[str] = None):
        """Updates the progress of a specific batch item."""
        with self.lock:
            for item in self.queue:
                if item.id == item_id:
                    item.progress = fraction
                    if message:
                        item.message = message
                    break

    def set_status(self, item_id: str, status: BatchStatus, message: Optional[str] = None, error: Optional[str] = None):
        """Updates the status, message, and error of a specific batch item."""
        with self.lock:
            for item in self.queue:
                if item.id == item_id:
                    item.status = status
                    if message:
                        item.message = message
                    if error is not None:
                        item.error = error
                    else:
                        item.error = ""
                    break

    def start_processing(self, processor_func: Callable, max_workers: int = 1):
        """
        Starts processing the batch queue in a background thread.

        Args:
            processor_func: Function to process each item.
            max_workers: Number of concurrent worker threads.
        """
        self.stop_event.clear()
        self.is_running = True

        with self.lock:
            pending_items = [item for item in self.queue if item.status == BatchStatus.PENDING]
        if not pending_items:
            self.is_running = False
            return

        threading.Thread(target=self._run_scheduler, args=(processor_func, max_workers), daemon=True).start()

    def _run_scheduler(self, processor_func, max_workers):
        try:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                self.executor = executor
                futures = []
                submitted_ids = set()

                while not self.stop_event.is_set():
                    candidate = None
                    with self.lock:
                        for item in self.queue:
                            if item.status == BatchStatus.PENDING and item.id not in submitted_ids:
                                candidate = item
                                break

                    if candidate:
                        # Resource-aware scheduling pre-check
                        vram_avail_mb = 0
                        if torch.cuda.is_available():
                            try:
                                props = torch.cuda.get_device_properties(0)
                                reserved = torch.cuda.memory_reserved(0)
                                val = (props.total_memory - reserved) / 1024 / 1024
                                vram_avail_mb = val if isinstance(val, (int, float)) else 8000
                            except (AttributeError, TypeError, RuntimeError):
                                vram_avail_mb = 8000  # Fallback for mocks/failures

                        try:
                            raw_ram = psutil.virtual_memory().available
                            ram_avail_mb = float(raw_ram) / 1024 / 1024 if isinstance(raw_ram, (int, float)) else 4096
                        except (AttributeError, TypeError):
                            ram_avail_mb = 4096  # Fallback

                        # Thresholds (could be made configurable in Config)
                        MIN_VRAM_MB = 1024
                        MIN_RAM_MB = 2048

                        if (torch.cuda.is_available() and vram_avail_mb < MIN_VRAM_MB) or ram_avail_mb < MIN_RAM_MB:
                            if self.logger:
                                self.logger.debug(
                                    f"Waiting for resources... (RAM: {ram_avail_mb:.0f}MB, VRAM: {vram_avail_mb:.0f}MB)",
                                    component="batch_manager",
                                )
                            time.sleep(2.0)
                            continue

                        submitted_ids.add(candidate.id)

                        def task(item=candidate):
                            if self.stop_event.is_set():
                                return

                            self.set_status(item.id, BatchStatus.PROCESSING, "Starting...")

                            class ProgressAdapter:
                                def __init__(self, manager, item_id):
                                    self.manager = manager
                                    self.item_id = item_id

                                def __call__(self, fraction, desc=None):
                                    self.manager.update_progress(self.item_id, fraction, desc)

                            import traceback

                            max_retries = 3
                            retry_delay = 1.0

                            for attempt in range(max_retries):
                                try:
                                    result = processor_func(item, ProgressAdapter(self, item.id))
                                    msg = "Completed"
                                    if isinstance(result, dict):
                                        if "message" in result:
                                            msg = result["message"]
                                        if "output_path" in result:
                                            item.output_path = result["output_path"]
                                    self.set_status(item.id, BatchStatus.COMPLETED, msg)
                                    break  # Success, exit retry loop
                                except Exception as e:
                                    if attempt < max_retries - 1:
                                        self.set_status(
                                            item.id,
                                            BatchStatus.PROCESSING,
                                            f"Retrying... ({attempt + 1}/{max_retries})",
                                        )
                                        time.sleep(retry_delay)
                                    else:
                                        stack_trace = traceback.format_exc()
                                        if self.logger:
                                            # Depending on logger type, it might support component= kwargs or just standard args
                                            if hasattr(self.logger, "error"):
                                                try:
                                                    self.logger.error(
                                                        f"Task failed after {max_retries} attempts: {str(e)}",
                                                        exc_info=True,
                                                        component="batch_manager",
                                                    )
                                                except TypeError:
                                                    self.logger.error(
                                                        f"Task failed after {max_retries} attempts: {str(e)}",
                                                        exc_info=True,
                                                    )

                                        self.set_status(item.id, BatchStatus.FAILED, str(e), error=stack_trace)

                        futures.append(executor.submit(task))
                    else:
                        time.sleep(0.5)

                        all_done = True
                        with self.lock:
                            for item in self.queue:
                                if item.status in (BatchStatus.PENDING, BatchStatus.PROCESSING):
                                    all_done = False
                                    break
                        if all_done:
                            break

                if self.stop_event.is_set():
                    for f in futures:
                        f.cancel()
        finally:
            self.executor = None
            self.is_running = False

    def stop_processing(self):
        """Signals the scheduler to stop processing."""
        self.stop_event.set()
