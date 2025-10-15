import pytest
import time
import sys
from unittest.mock import MagicMock, patch

# Mock heavy dependencies
sys.modules['GPUtil'] = MagicMock()

from app.logging_enhanced import EnhancedLogger, PerformanceMonitor

@patch('psutil.Process')
def test_enhanced_logger_basic_functionality(mock_process):
    """Test basic logging functionality."""
    logger = EnhancedLogger(log_to_console=False, log_to_file=False)

    # Test different log levels
    logger.info("Test info message", component="test")
    logger.warning("Test warning message", component="test")
    logger.error("Test error message", component="test")
    logger.success("Test success message", component="test")

@patch('psutil.Process')
def test_operation_context_timing(mock_process):
    """Test operation context manager timing."""
    logger = EnhancedLogger(log_to_console=False, log_to_file=False)

    with logger.operation_context("test_operation", "test_component") as ctx:
        time.sleep(0.1)  # Simulate work
        assert 'operation' in ctx
        assert ctx['operation'] == "test_operation"

@patch('psutil.Process')
def test_performance_monitor(mock_process):
    """Test performance monitoring."""
    monitor = PerformanceMonitor()
    metrics = monitor.get_system_metrics()

    assert 'cpu_percent' in metrics
    assert 'memory_percent' in metrics
    assert 'process_memory_mb' in metrics

if __name__ == "__main__":
    pytest.main([__file__])