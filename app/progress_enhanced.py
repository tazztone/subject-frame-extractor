import time
import threading
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, Callable, List
from queue import Queue
import math

@dataclass
class ProgressState:
    """Comprehensive progress state information."""
    operation: str
    stage: str
    current: int
    total: int
    stage_current: int = 0
    stage_total: int = 0
    start_time: float = 0.0
    stage_start_time: float = 0.0
    substages: List[str] = None
    current_substage: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class AdvancedProgressTracker:
    """Advanced progress tracking with detailed analytics and prediction."""

    def __init__(self, progress_queue: Queue, logger=None):
        self.progress_queue = progress_queue
        self.logger = logger
        self.current_state: Optional[ProgressState] = None
        self.history: List[Dict[str, Any]] = []
        self.stage_history: Dict[str, List[float]] = {}
        self.lock = threading.Lock()

    def start_operation(self,
                       operation: str,
                       total_items: int,
                       stages: List[str] = None,
                       metadata: Dict[str, Any] = None):
        """Start a new operation with comprehensive tracking."""
        with self.lock:
            self.current_state = ProgressState(
                operation=operation,
                stage="initializing",
                current=0,
                total=total_items,
                start_time=time.time(),
                substages=stages or [],
                metadata=metadata or {}
            )

        self._update_ui()
        if self.logger:
            self.logger.info(f"Started operation: {operation}",
                           component="progress",
                           operation=operation,
                           custom_fields={'total_items': total_items})

    def start_stage(self,
                   stage_name: str,
                   stage_items: int = None,
                   substage: str = None):
        """Start a new stage within the current operation."""
        if not self.current_state:
            return

        with self.lock:
            # Record previous stage timing for prediction
            if self.current_state.stage != "initializing":
                stage_duration = time.time() - self.current_state.stage_start_time
                if self.current_state.stage not in self.stage_history:
                    self.stage_history[self.current_state.stage] = []
                self.stage_history[self.current_state.stage].append(stage_duration)

            self.current_state.stage = stage_name
            self.current_state.stage_current = 0
            self.current_state.stage_total = stage_items or 0
            self.current_state.stage_start_time = time.time()
            self.current_state.current_substage = substage

        self._update_ui()
        if self.logger:
            self.logger.info(f"Started stage: {stage_name}",
                           component="progress",
                           operation=self.current_state.operation,
                           custom_fields={'stage_items': stage_items})

    def update_progress(self,
                       items_processed: int = 1,
                       stage_items_processed: int = None,
                       substage: str = None,
                       metadata: Dict[str, Any] = None):
        """Update progress with optional metadata."""
        if not self.current_state:
            return

        with self.lock:
            self.current_state.current = min(
                self.current_state.current + items_processed,
                self.current_state.total
            )

            if stage_items_processed is not None:
                self.current_state.stage_current = min(
                    stage_items_processed,
                    self.current_state.stage_total
                )

            if substage:
                self.current_state.current_substage = substage

            if metadata:
                self.current_state.metadata.update(metadata)

        self._update_ui()

    def complete_operation(self, success: bool = True, message: str = None):
        """Mark the operation as complete."""
        if not self.current_state:
            return

        total_duration = time.time() - self.current_state.start_time

        with self.lock:
            # Record final stage timing
            if self.current_state.stage != "initializing":
                stage_duration = time.time() - self.current_state.stage_start_time
                if self.current_state.stage not in self.stage_history:
                    self.stage_history[self.current_state.stage] = []
                self.stage_history[self.current_state.stage].append(stage_duration)

            # Add to history
            self.history.append({
                'operation': self.current_state.operation,
                'total_items': self.current_state.total,
                'duration': total_duration,
                'success': success,
                'timestamp': time.time(),
                'stages': self.current_state.substages,
                'final_metadata': self.current_state.metadata
            })

            if success:
                self.current_state.current = self.current_state.total
                self.current_state.stage = "completed"
            else:
                self.current_state.stage = "failed"

        self._update_ui(force_complete=True)

        if self.logger:
            status = "SUCCESS" if success else "ERROR"
            self.logger._log_event(self.logger._create_log_event(
                status,
                message or f"Operation {self.current_state.operation} {'completed' if success else 'failed'}",
                "progress",
                operation=self.current_state.operation,
                duration_ms=total_duration * 1000,
                custom_fields={
                    'total_items_processed': self.current_state.current,
                    'success': success
                }
            ))

    def _calculate_eta(self) -> tuple[float, str]:
        """Calculate ETA using historical data and current progress."""
        if not self.current_state or self.current_state.current == 0:
            return float('inf'), "calculating..."

        current_time = time.time()
        elapsed = current_time - self.current_state.start_time

        # Calculate based on overall progress
        progress_ratio = self.current_state.current / self.current_state.total
        if progress_ratio > 0:
            estimated_total_time = elapsed / progress_ratio
            eta_seconds = estimated_total_time - elapsed
        else:
            eta_seconds = float('inf')

        # Refine ETA using stage history if available
        if self.current_state.stage in self.stage_history:
            stage_history = self.stage_history[self.current_state.stage]
            avg_stage_time = sum(stage_history) / len(stage_history)

            # Estimate remaining stages
            remaining_stages = len(self.current_state.substages) - self.current_state.substages.index(self.current_state.stage) - 1 if self.current_state.stage in self.current_state.substages else 0
            stage_eta = avg_stage_time * remaining_stages

            # Current stage progress
            if self.current_state.stage_total > 0:
                stage_progress = self.current_state.stage_current / self.current_state.stage_total
                current_stage_remaining = avg_stage_time * (1 - stage_progress)
                stage_eta += current_stage_remaining

            # Use the more conservative estimate
            eta_seconds = min(eta_seconds, stage_eta)

        # Format ETA string
        if eta_seconds == float('inf'):
            return eta_seconds, "calculating..."
        elif eta_seconds < 60:
            return eta_seconds, f"{int(eta_seconds)}s"
        elif eta_seconds < 3600:
            minutes = int(eta_seconds / 60)
            seconds = int(eta_seconds % 60)
            return eta_seconds, f"{minutes}m {seconds}s"
        else:
            hours = int(eta_seconds / 3600)
            minutes = int((eta_seconds % 3600) / 60)
            return eta_seconds, f"{hours}h {minutes}m"

    def _calculate_rate(self) -> float:
        """Calculate processing rate (items per second)."""
        if not self.current_state or self.current_state.current == 0:
            return 0.0

        elapsed = time.time() - self.current_state.start_time
        return self.current_state.current / elapsed if elapsed > 0 else 0.0

    def _update_ui(self, force_complete: bool = False):
        """Send progress update to UI."""
        if not self.current_state:
            return

        progress_ratio = self.current_state.current / self.current_state.total if self.current_state.total > 0 else 0
        eta_seconds, eta_str = self._calculate_eta()
        rate = self._calculate_rate()

        # Stage progress
        stage_progress = ""
        if self.current_state.stage_total > 0:
            stage_ratio = self.current_state.stage_current / self.current_state.stage_total
            stage_progress = f" | Stage: {self.current_state.stage_current}/{self.current_state.stage_total} ({stage_ratio:.1%})"

        # Substage info
        substage_info = f" | {self.current_state.current_substage}" if self.current_state.current_substage else ""

        # Progress message for UI
        progress_message = {
            "stage": self.current_state.stage,
            "progress": progress_ratio,
            "current": self.current_state.current,
            "total": self.current_state.total,
            "eta": eta_str,
            "rate": f"{rate:.1f}/s" if rate > 0 else "0.0/s",
            "detailed_status": f"{self.current_state.operation} | {self.current_state.stage} | {self.current_state.current}/{self.current_state.total} ({progress_ratio:.1%}) | ETA: {eta_str} | Rate: {rate:.1f}/s{stage_progress}{substage_info}"
        }

        # Add metadata to progress message
        if self.current_state.metadata:
            progress_message["metadata"] = self.current_state.metadata

        self.progress_queue.put(progress_message)