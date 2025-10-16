import unittest
import itertools
import threading
import time
from unittest.mock import MagicMock, patch
from app.performance_optimizer import AdaptiveResourceManager

class TestAdaptiveResourceManager(unittest.TestCase):
    @patch('app.config.Config')
    @patch('app.logging_enhanced.EnhancedLogger')
    def setUp(self, mock_logger, mock_config):
        self.mock_config = mock_config
        self.mock_config.settings = {
            'monitoring': {
                'cpu_threshold': 80,
                'mem_threshold': 80,
                'high_threshold_duration': 10,
                'cooldown_duration': 30,
                'memory_warning_threshold_mb': 8192,
                'cpu_warning_threshold_percent': 90,
                'gpu_memory_warning_threshold_percent': 90
            }
        }
        self.mock_logger = mock_logger
        self.resource_manager = AdaptiveResourceManager(logger=self.mock_logger, config=self.mock_config)

    def test_initialization(self):
        self.assertEqual(self.resource_manager.logger, self.mock_logger)
        self.assertEqual(self.resource_manager.config, self.mock_config)

    def test_start_and_stop_monitoring(self):
        with patch('app.performance_optimizer.threading.Thread') as mock_thread:
            self.resource_manager.start_monitoring()
            self.assertTrue(self.resource_manager.monitoring_active)
            mock_thread.assert_called_once()

            self.resource_manager.stop_monitoring()
            self.assertFalse(self.resource_manager.monitoring_active)

    @patch('app.performance_optimizer.time.sleep')
    @patch('app.performance_optimizer.psutil.cpu_percent')
    @patch('app.performance_optimizer.psutil.virtual_memory')
    def test_monitor_resources(self, mock_virtual_memory, mock_cpu_percent, mock_sleep):
        mock_cpu_percent.return_value = 90
        mock_virtual_memory.return_value.percent = 90

        with patch.object(self.resource_manager, '_get_resource_metrics') as mock_get_metrics:
            mock_get_metrics.return_value = {
                'cpu_percent': 90,
                'memory_percent': 90,
                'process_memory_mb': 9000
            }
            with patch.object(self.resource_manager, '_adjust_parameters') as mock_adjust:
                # Set up a threading event to control the loop
                stop_event = threading.Event()
                self.resource_manager._stop_event = stop_event
                self.resource_manager.monitoring_active = True

                # Use a side effect to stop the loop after the first iteration
                def sleep_and_stop(*args, **kwargs):
                    stop_event.set()

                mock_sleep.side_effect = sleep_and_stop

                # Run the monitoring loop
                self.resource_manager._monitor_resources()

                # Assert that _adjust_parameters was called
                mock_adjust.assert_called_once()
                mock_get_metrics.assert_called_once()

    @patch('app.performance_optimizer.psutil.cpu_percent')
    @patch('app.performance_optimizer.psutil.virtual_memory')
    def test_adjust_parameters(self, mock_virtual_memory, mock_cpu_percent):
        mock_cpu_percent.return_value = 95
        mock_virtual_memory.return_value.percent = 95

        self.resource_manager.current_limits['batch_size'] = 32
        self.resource_manager.current_limits['num_workers'] = 4

        metrics = {
            'cpu_percent': 95,
            'process_memory_mb': 9000
        }

        self.resource_manager._adjust_parameters(metrics)

        self.assertEqual(self.resource_manager.current_limits['batch_size'], 22)
        self.assertEqual(self.resource_manager.current_limits['num_workers'], 3)

if __name__ == '__main__':
    unittest.main()