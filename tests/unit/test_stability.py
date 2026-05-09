import unittest
from unittest.mock import MagicMock, patch

from core.batch_manager import BatchManager, BatchStatus
from core.config import Config
from core.enums import SceneStatus
from core.events import PreAnalysisEvent, PropagationEvent
from core.utils import MemoryWatchdog


class TestStability(unittest.TestCase):
    def setUp(self):
        self.config = Config()
        self.logger = MagicMock()

    @patch("core.batch_manager.psutil.virtual_memory")
    @patch("core.batch_manager.time.sleep")
    def test_batch_manager_resource_aware_wait(self, mock_sleep, mock_ram):
        # Mock low RAM
        mock_ram.return_value.available = 100 * 1024 * 1024  # 100MB

        bm = BatchManager(logger=self.logger)
        bm.add_paths(["test.mp4"])

        # We'll make it succeed after one sleep
        def side_effect(seconds):
            mock_ram.return_value.available = 4000 * 1024 * 1024  # 4GB
            return None

        mock_sleep.side_effect = side_effect

        # Start processing
        processor = MagicMock(return_value={"message": "Success", "output_path": "/out"})
        bm.start_processing(processor)

        # Wait for item to complete (max 5s)
        import time

        start_time = time.time()
        while time.time() - start_time < 5:
            with bm.lock:
                if bm.queue and bm.queue[0].status == BatchStatus.COMPLETED:
                    break
            time.sleep(0.1)

        with bm.lock:
            item = bm.queue[0]
            self.assertEqual(item.status, BatchStatus.COMPLETED)
            self.assertEqual(item.output_path, "/out")

        self.assertTrue(mock_sleep.called)

    @patch("time.sleep", side_effect=[StopIteration])
    @patch("core.utils.psutil.virtual_memory")
    def test_memory_watchdog(self, mock_ram, mock_sleep):
        # Setup thresholds
        self.config.monitoring_memory_warning_threshold_mb = 1000
        self.config.monitoring_memory_critical_threshold_mb = 2000
        self.config.monitoring_memory_watchdog_enabled = True

        # Mock high RAM usage
        mock_ram.return_value.total = 8000 * 1024 * 1024
        mock_ram.return_value.available = 500 * 1024 * 1024  # 7.5GB used

        watchdog = MemoryWatchdog(self.config, self.logger)

        try:
            watchdog._run()
        except StopIteration:
            pass

        # Check if critical warning was logged
        self.logger.critical.assert_called()
        args, _ = self.logger.critical.call_args
        self.assertIn("CRITICAL: System RAM usage", args[0])

    def test_propagation_event_enum_validation(self):
        analysis_params = PreAnalysisEvent(output_folder=".", video_path="test.mp4")

        # Test with string status
        event = PropagationEvent(
            output_folder=".",
            video_path="test.mp4",
            scenes=[{"shot_id": 1, "status": "excluded"}],
            analysis_params=analysis_params,
        )

        self.assertIsInstance(event.scenes[0]["status"], SceneStatus)
        self.assertEqual(event.scenes[0]["status"], SceneStatus.EXCLUDED)

        # Test with invalid string status (should remain string or fallback)
        event2 = PropagationEvent(
            output_folder=".",
            video_path="test.mp4",
            scenes=[{"shot_id": 1, "status": "invalid"}],
            analysis_params=analysis_params,
        )
        self.assertEqual(event2.scenes[0]["status"], "invalid")


if __name__ == "__main__":
    unittest.main()
