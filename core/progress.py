"""
Progress Tracking Infrastructure for Frame Extractor & Analyzer
"""

import threading
import time
from queue import Queue
from typing import TYPE_CHECKING, Callable, Optional

from pydantic import BaseModel

if TYPE_CHECKING:
    from core.logger import AppLogger


class ProgressEvent(BaseModel):
    stage: str
    substage: Optional[str] = None
    done: int = 0
    total: int = 1
    fraction: float = 0.0
    eta_seconds: Optional[float] = None
    eta_formatted: str = "—"


class AdvancedProgressTracker:
    """
    Tracks and estimates progress for long-running operations.

    Calculates ETA using exponential moving average (EMA) and updates the UI.
    """

    def __init__(self, progress: Callable, queue: Queue, logger: "AppLogger", ui_stage_name: str = ""):
        """
        Initializes the progress tracker.

        Args:
            progress: Gradio progress callback.
            queue: Queue for sending progress events.
            logger: Application logger.
            ui_stage_name: Initial stage name.
        """
        self.progress = progress
        self.queue = queue
        self.logger = logger
        self.stage = ui_stage_name or "Working"
        self.substage: Optional[str] = None
        self.total = 1
        self.done = 0
        self._t0 = time.time()
        self._last_ts = self._t0
        self._ema_dt = None
        self._alpha = 0.2
        self._last_update_ts: float = 0.0
        self.throttle_interval: float = 0.1
        self.pause_event = threading.Event()
        self.pause_event.set()

    def start(self, total_items: int, desc: Optional[str] = None):
        """Resets the tracker for a new operation."""
        self.total = max(1, int(total_items))
        self.done = 0
        if desc:
            self.stage = desc
        self.substage = None
        self._t0 = time.time()
        self._last_ts = self._t0
        self._ema_dt = None
        self._overlay(force=True)

    def step(self, n: int = 1, desc: Optional[str] = None, substage: Optional[str] = None):
        """
        Increments progress by 'n' steps.

        Args:
            n: Number of steps completed.
            desc: Optional stage description update.
            substage: Optional substage description update.
        """
        self.pause_event.wait()
        now = time.time()
        dt = now - self._last_ts
        self._last_ts = now
        if dt > 0:
            if self._ema_dt is None:
                self._ema_dt = dt / max(1, n)
            else:
                self._ema_dt = self._alpha * (dt / max(1, n)) + (1 - self._alpha) * self._ema_dt
        self.done = min(self.total, self.done + n)
        if desc:
            self.stage = desc
        if substage is not None:
            self.substage = substage
        self._overlay()

    def set(self, done: int, desc: Optional[str] = None, substage: Optional[str] = None):
        """Sets the absolute number of completed steps."""
        delta = max(0, done - self.done)
        if delta > 0:
            self.step(delta, desc=desc, substage=substage)

    def set_stage(self, stage: str, substage: Optional[str] = None):
        """Updates the current stage description without changing progress."""
        self.stage = stage
        self.substage = substage
        self._overlay(force=True)

    def done_stage(self, final_text: Optional[str] = None):
        """Marks the current operation as complete."""
        self.done = self.total
        self._overlay(force=True)
        if final_text:
            self.logger.info(final_text, component="progress")

    def _overlay(self, force: bool = False):
        """Emits a progress update if enough time has passed (throttling)."""
        now = time.time()
        fraction = self.done / max(1, self.total)
        if not force and (now - self._last_update_ts < self.throttle_interval):
            return
        self._last_update_ts = now
        eta_s = self._eta_seconds()
        eta_str = self._fmt_eta(eta_s)
        desc_parts = [f"{self.stage} ({self.done}/{self.total})"]
        if self.substage:
            desc_parts.append(self.substage)
        desc_parts.append(f"ETA {eta_str}")
        gradio_desc = " • ".join(desc_parts)
        if self.progress:
            self.progress(fraction, desc=gradio_desc)
        progress_event = ProgressEvent(
            stage=self.stage,
            substage=self.substage,
            done=self.done,
            total=self.total,
            fraction=fraction,
            eta_seconds=eta_s,
            eta_formatted=eta_str,
        )
        self.queue.put({"progress": progress_event.model_dump()})

    def _eta_seconds(self) -> Optional[float]:
        """Calculates estimated seconds remaining based on EMA."""
        if self._ema_dt is None:
            return None
        remaining = max(0, self.total - self.done)
        return self._ema_dt * remaining

    @staticmethod
    def _fmt_eta(eta_s: Optional[float]) -> str:
        """Formats seconds into a human-readable string."""
        if eta_s is None:
            return "—"
        if eta_s < 60:
            return f"{int(eta_s)}s"
        m, s = divmod(int(eta_s), 60)
        if m < 60:
            return f"{m}m {s}s"
        h, m = divmod(m, 60)
        return f"{h}h {m}m"
