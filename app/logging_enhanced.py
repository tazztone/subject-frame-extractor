import logging
import json
import time
import traceback
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from contextlib import contextmanager
import psutil
import GPUtil

@dataclass
class LogEvent:
    """Structured log event with standardized fields."""
    timestamp: str
    level: str
    message: str
    component: str
    operation: Optional[str] = None
    duration_ms: Optional[float] = None
    memory_mb: Optional[float] = None
    gpu_memory_mb: Optional[float] = None
    error_type: Optional[str] = None
    stack_trace: Optional[str] = None
    user_context: Optional[Dict[str, Any]] = None
    performance_metrics: Optional[Dict[str, Any]] = None
    custom_fields: Optional[Dict[str, Any]] = None

class PerformanceMonitor:
    """System and application performance monitoring."""

    def __init__(self):
        self.process = psutil.Process()
        self.gpu_available = self._check_gpu_availability()

    def _check_gpu_availability(self) -> bool:
        try:
            GPUtil.getGPUs()
            return True
        except:
            return False

    def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system performance metrics."""
        metrics = {
            'cpu_percent': psutil.cpu_percent(interval=0.1),
            'memory_percent': psutil.virtual_memory().percent,
            'memory_available_mb': psutil.virtual_memory().available / (1024**2),
            'disk_usage_percent': psutil.disk_usage('/').percent,
            'process_memory_mb': self.process.memory_info().rss / (1024**2),
            'process_cpu_percent': self.process.cpu_percent(),
        }

        if self.gpu_available:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]  # Primary GPU
                    metrics.update({
                        'gpu_memory_used_mb': gpu.memoryUsed,
                        'gpu_memory_total_mb': gpu.memoryTotal,
                        'gpu_memory_percent': gpu.memoryUtil * 100,
                        'gpu_load_percent': gpu.load * 100,
                        'gpu_temperature': gpu.temperature
                    })
            except Exception:
                pass

        return metrics

class EnhancedLogger:
    """Advanced logging system with structured output and performance monitoring."""

    def __init__(self,
                 log_dir: Optional[Path] = None,
                 enable_performance_monitoring: bool = True,
                 log_to_file: bool = True,
                 log_to_console: bool = True):

        self.log_dir = log_dir or Path("logs")
        self.log_dir.mkdir(exist_ok=True, parents=True)
        self.progress_queue = None
        self.performance_monitor = PerformanceMonitor() if enable_performance_monitoring else None

        # Create session-specific log file
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_log_file = self.log_dir / f"session_{self.session_id}.log"
        self.structured_log_file = self.log_dir / f"structured_{self.session_id}.jsonl"

        # Setup Python logger
        self.logger = logging.getLogger(f'enhanced_logger_{self.session_id}')
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False

        # Clear any existing handlers
        self.logger.handlers.clear()

        if log_to_console:
            self._setup_console_handler()

        if log_to_file:
            self._setup_file_handlers()

        # Operation timing context
        self._operation_stack: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    def _setup_console_handler(self):
        """Setup enhanced console logging with colors."""
        console_handler = logging.StreamHandler()
        console_formatter = ColoredFormatter(
            '%(asctime)s | %(levelname)8s | %(name)s | %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(logging.INFO)
        self.logger.addHandler(console_handler)

    def _setup_file_handlers(self):
        """Setup file-based logging handlers."""
        # Human-readable log file
        file_handler = logging.FileHandler(self.session_log_file, encoding='utf-8')
        file_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)8s | %(name)s | %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.DEBUG)
        self.logger.addHandler(file_handler)

        # Structured JSON log file
        self.structured_handler = logging.FileHandler(
            self.structured_log_file, encoding='utf-8'
        )
        self.structured_handler.setLevel(logging.DEBUG)
        self.logger.addHandler(self.structured_handler)

    def set_progress_queue(self, queue):
        """Set the queue for real-time UI updates."""
        self.progress_queue = queue

    @contextmanager
    def operation_context(self,
                         operation_name: str,
                         component: str,
                         user_context: Optional[Dict[str, Any]] = None):
        """Context manager for timing operations and automatic error handling."""
        start_time = time.time()
        start_metrics = self.performance_monitor.get_system_metrics() if self.performance_monitor else {}

        operation_data = {
            'operation': operation_name,
            'component': component,
            'start_time': start_time,
            'start_metrics': start_metrics,
            'user_context': user_context or {}
        }

        with self._lock:
            self._operation_stack.append(operation_data)

        try:
            self.info(f"Starting {operation_name}",
                     component=component,
                     operation=operation_name,
                     user_context=user_context)
            yield operation_data

            duration = (time.time() - start_time) * 1000
            self.success(f"Completed {operation_name}",
                        component=component,
                        operation=operation_name,
                        duration_ms=duration,
                        user_context=user_context)

        except Exception as e:
            duration = (time.time() - start_time) * 1000
            self.error(f"Failed {operation_name}: {str(e)}",
                      component=component,
                      operation=operation_name,
                      duration_ms=duration,
                      error_type=type(e).__name__,
                      stack_trace=traceback.format_exc(),
                      user_context=user_context)
            raise
        finally:
            with self._lock:
                if self._operation_stack:
                    self._operation_stack.pop()

    def _create_log_event(self,
                         level: str,
                         message: str,
                         component: str,
                         **kwargs) -> LogEvent:
        """Create a structured log event."""
        current_metrics = self.performance_monitor.get_system_metrics() if self.performance_monitor else {}

        return LogEvent(
            timestamp=datetime.now().isoformat(),
            level=level,
            message=message,
            component=component,
            memory_mb=current_metrics.get('process_memory_mb'),
            gpu_memory_mb=current_metrics.get('gpu_memory_used_mb'),
            **kwargs
        )

    def _log_event(self, event: LogEvent):
        """Log a structured event to all configured outputs."""
        # Log to Python logger (files and console)
        log_level = getattr(logging, event.level.upper(), logging.INFO)
        extra_info = f" [{event.component}]"
        if event.operation:
            extra_info += f" [{event.operation}]"
        if event.duration_ms:
            extra_info += f" ({event.duration_ms:.1f}ms)"

        self.logger.log(log_level, f"{event.message}{extra_info}")

        # Write structured JSON log
        if hasattr(self, 'structured_handler'):
            json_line = json.dumps(asdict(event), default=str, ensure_ascii=False)
            with open(self.structured_log_file, 'a', encoding='utf-8') as f:
                f.write(json_line + '\n')

        # Send to UI progress queue
        if self.progress_queue:
            ui_message = f"[{event.level}] {event.message}"
            if event.operation:
                ui_message = f"[{event.operation}] {ui_message}"
            self.progress_queue.put({"log": ui_message})

    def debug(self, message: str, component: str = "system", **kwargs):
        event = self._create_log_event("DEBUG", message, component, **kwargs)
        self._log_event(event)

    def info(self, message: str, component: str = "system", **kwargs):
        event = self._create_log_event("INFO", message, component, **kwargs)
        self._log_event(event)

    def warning(self, message: str, component: str = "system", **kwargs):
        event = self._create_log_event("WARNING", message, component, **kwargs)
        self._log_event(event)

    def error(self, message: str, component: str = "system", **kwargs):
        event = self._create_log_event("ERROR", message, component, **kwargs)
        self._log_event(event)

    def success(self, message: str, component: str = "system", **kwargs):
        event = self._create_log_event("SUCCESS", message, component, **kwargs)
        self._log_event(event)

    def critical(self, message: str, component: str = "system", **kwargs):
        event = self._create_log_event("CRITICAL", message, component, **kwargs)
        self._log_event(event)

class ColoredFormatter(logging.Formatter):
    """Console formatter with color support."""

    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[37m',      # White
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'SUCCESS': '\033[32m',   # Green
        'RESET': '\033[0m'
    }

    def format(self, record):
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        record.levelname = f"{color}{record.levelname}{self.COLORS['RESET']}"
        return super().format(record)