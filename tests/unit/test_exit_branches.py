import sqlite3
from unittest.mock import MagicMock, patch

import pytest

from core.batch_manager import BatchManager
from core.db_schema import migrate_database
from core.managers.sam2 import SAM2Wrapper
from core.managers.session import validate_session_dir


class TestExitBranches:
    # --- BatchManager ---
    def test_batch_manager_cancel_before_loop(self, mock_logger):
        """Test BatchManager scheduler exits if stop event is set."""
        bm = BatchManager()
        bm.stop_event.set()

        processor_fn = MagicMock()

        # Add a pending item
        bm.add_paths(["test.mp4"])

        # _run_scheduler is the internal loop
        bm._run_scheduler(processor_fn, max_workers=1)

        # Should not have called processor_fn because stop_event was set
        processor_fn.assert_not_called()

    # --- DB Schema ---
    def test_migrate_database_failure(self, mock_logger):
        """Test migrate_database handles exceptions and triggers rollback."""
        mock_conn = MagicMock()
        mock_cursor = mock_conn.cursor.return_value

        # 1. First call to get MAX(version) returns 0 to trigger migration
        # 2. Second call during _detect_legacy_version returns None (no metadata table)
        mock_cursor.fetchone.side_effect = [(0,), None]

        # Force exception during migration body (e.g. during _migration_v1_initial_schema)
        # We need to find a way to make it fail inside the try block
        # We can patch _migration_v1_initial_schema
        with patch("core.db_schema._migration_v1_initial_schema", side_effect=sqlite3.OperationalError("locked")):
            with pytest.raises(sqlite3.OperationalError):
                migrate_database(mock_conn, mock_logger)

        mock_conn.rollback.assert_called_once()
        mock_logger.error.assert_called()

    # --- SAM2 Wrapper ---
    def test_sam2_wrapper_init_failure(self, mock_config):
        """Test SAM2Wrapper initialization failure handling."""
        with patch("core.managers.sam2.build_sam2_video_predictor", side_effect=RuntimeError("build failed")):
            with pytest.raises(RuntimeError, match="build failed"):
                SAM2Wrapper(checkpoint_path="missing.pt", device="cpu")

    # --- Session Manager ---
    def test_validate_session_dir_is_file(self, tmp_path):
        """Test validation fails if path is a file."""
        p = tmp_path / "some_file.txt"
        p.touch()

        res_path, error = validate_session_dir(str(p))
        assert res_path is None
        assert "does not exist" in error.lower()

    def test_validate_session_dir_non_existent(self):
        """Test validation fails if path does not exist."""
        res_path, error = validate_session_dir("/non/existent/path")
        assert res_path is None
        assert "does not exist" in error.lower()
