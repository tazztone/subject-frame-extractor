import time
import threading
from typing import Dict, Any, Optional, Callable
import psutil
import gc

class AdaptiveResourceManager:
    """Automatically adjusts processing parameters based on system resources."""

    def __init__(self, logger, config):
        self.logger = logger
        self.config = config
        self.monitoring_active = False
        self.current_limits = {
            'batch_size': getattr(config, 'default_batch_size', 32),
            'num_workers': getattr(config, 'default_workers', 4),
            'memory_limit_mb': getattr(config, 'memory_limit_mb', 8192)
        }
        self.performance_history = []
        self._monitor_thread = None

    def start_monitoring(self):
        """Start continuous resource monitoring."""
        self.monitoring_active = True
        self._monitor_thread = threading.Thread(target=self._monitor_resources, daemon=True)
        self._monitor_thread.start()

        self.logger.info("Started adaptive resource monitoring", component="resource_manager")

    def stop_monitoring(self):
        """Stop resource monitoring."""
        self.monitoring_active = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1)

        self.logger.info("Stopped adaptive resource monitoring", component="resource_manager")

    def _monitor_resources(self):
        """Background monitoring thread."""
        while self.monitoring_active:
            try:
                metrics = self._get_resource_metrics()
                self._adjust_parameters(metrics)
                self.performance_history.append({
                    'timestamp': time.time(),
                    'metrics': metrics,
                    'limits': self.current_limits.copy()
                })

                # Keep only last hour of history
                cutoff_time = time.time() - 3600
                self.performance_history = [
                    h for h in self.performance_history
                    if h['timestamp'] > cutoff_time
                ]

                time.sleep(5)  # Monitor every 5 seconds

            except Exception as e:
                self.logger.error(f"Resource monitoring error: {e}", component="resource_manager")
                time.sleep(10)  # Back off on errors

    def _get_resource_metrics(self) -> Dict[str, Any]:
        """Get current resource utilization metrics."""
        process = psutil.Process()

        metrics = {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'memory_available_mb': psutil.virtual_memory().available / (1024**2),
            'process_memory_mb': process.memory_info().rss / (1024**2),
            'disk_io': psutil.disk_io_counters()._asdict() if psutil.disk_io_counters() else {},
            'load_average': psutil.getloadavg() if hasattr(psutil, 'getloadavg') else [0, 0, 0]
        }

        # Add GPU metrics if available
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                metrics.update({
                    'gpu_memory_percent': gpu.memoryUtil * 100,
                    'gpu_load_percent': gpu.load * 100,
                    'gpu_temperature': gpu.temperature
                })
        except Exception:
            pass

        return metrics

    def _adjust_parameters(self, metrics: Dict[str, Any]):
        """Automatically adjust processing parameters based on metrics."""
        monitoring_config = self.config.settings.get('monitoring', {})
        # Memory-based adjustments
        if metrics['process_memory_mb'] > monitoring_config.get('memory_warning_threshold_mb', 8192):
            # Reduce batch size if using too much memory
            if self.current_limits['batch_size'] > 1:
                old_batch_size = self.current_limits['batch_size']
                self.current_limits['batch_size'] = max(1, int(old_batch_size * 0.7))

                self.logger.warning(
                    f"High memory usage detected, reducing batch size from {old_batch_size} to {self.current_limits['batch_size']}",
                    component="resource_manager",
                    custom_fields={'memory_mb': metrics['process_memory_mb']}
                )

                # Force garbage collection
                gc.collect()

        # CPU-based adjustments
        if metrics['cpu_percent'] > monitoring_config.get('cpu_warning_threshold_percent', 90):
            # Reduce number of workers if CPU is overloaded
            if self.current_limits['num_workers'] > 1:
                old_workers = self.current_limits['num_workers']
                self.current_limits['num_workers'] = max(1, int(old_workers * 0.8))

                self.logger.warning(
                    f"High CPU usage detected, reducing workers from {old_workers} to {self.current_limits['num_workers']}",
                    component="resource_manager",
                    custom_fields={'cpu_percent': metrics['cpu_percent']}
                )

        # GPU memory adjustments
        if 'gpu_memory_percent' in metrics and metrics['gpu_memory_percent'] > monitoring_config.get('gpu_memory_warning_threshold_percent', 90):
            self.logger.warning(
                "High GPU memory usage detected, consider reducing batch size or switching to CPU",
                component="resource_manager",
                custom_fields={'gpu_memory_percent': metrics['gpu_memory_percent']}
            )

    def get_current_limits(self) -> Dict[str, Any]:
        """Get current adaptive limits."""
        return self.current_limits.copy()

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for the current session."""
        if not self.performance_history:
            return {}

        recent_metrics = self.performance_history[-10:]  # Last 10 measurements

        avg_cpu = sum(m['metrics']['cpu_percent'] for m in recent_metrics) / len(recent_metrics)
        avg_memory = sum(m['metrics']['process_memory_mb'] for m in recent_metrics) / len(recent_metrics)

        return {
            'avg_cpu_percent': avg_cpu,
            'avg_memory_mb': avg_memory,
            'current_batch_size': self.current_limits['batch_size'],
            'current_workers': self.current_limits['num_workers'],
            'monitoring_duration_minutes': (time.time() - self.performance_history[0]['timestamp']) / 60
        }