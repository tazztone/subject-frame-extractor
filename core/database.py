from __future__ import annotations

import atexit
import json
import logging
import sqlite3
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

if TYPE_CHECKING:
    from core.error_handling import ErrorHandler
    from core.logger import AppLogger

from core.db_schema import migrate_database


class Database:
    def __init__(
        self,
        db_path: Path,
        batch_size: int = 100,
        logger: Optional[Union[logging.Logger, "AppLogger"]] = None,
    ):
        """Initializes the Database manager."""
        self.db_path = db_path
        self.conn = None
        self.buffer = []
        self.batch_size = batch_size
        self.logger = logger or logging.getLogger(__name__)
        self.lock = threading.Lock()
        self.error_handler: Optional["ErrorHandler"] = None
        self.columns = [
            "filename",
            "face_sim",
            "face_conf",
            "shot_id",
            "seed_type",
            "seed_face_sim",
            "mask_area_pct",
            "mask_empty",
            "error",
            "error_severity",
            "phash",
            "dedup_thresh",
            "metrics",
        ]
        self.connect()
        self.migrate()
        # Register atexit handler to ensure flush on shutdown
        atexit.register(self.close)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def connect(self):
        """Connects to the SQLite database."""
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=30.0)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.row_factory = sqlite3.Row

    def migrate(self):
        """Migrates the database to the latest schema version."""
        if not self.conn:
            self.connect()
        assert self.conn is not None
        migrate_database(self.conn, self.logger)

    def close(self):
        """Closes the database connection."""
        if self.conn:
            try:
                self.flush()
            except Exception:
                # Avoid logging here as it might fail during interpreter shutdown
                pass

            try:
                self.conn.close()
            except Exception:
                pass
            self.conn = None

    def clear_metadata(self):
        """Deletes all records from the metadata table."""
        with self.lock:
            self.buffer.clear()
            if not self.conn:
                self.connect()
            assert self.conn is not None
            cursor = self.conn.cursor()
            cursor.execute("DELETE FROM metadata")
            self.conn.commit()

    def insert_metadata(self, metadata: Dict[str, Any]):
        """Inserts or replaces a metadata record."""
        data_copy = metadata.copy()
        keys_to_extract = [
            "filename",
            "face_sim",
            "face_conf",
            "shot_id",
            "seed_type",
            "seed_face_sim",
            "mask_area_pct",
            "mask_empty",
            "error",
            "error_severity",
            "phash",
            "dedup_thresh",
        ]
        base_metadata = {key: data_copy.pop(key, None) for key in keys_to_extract}

        if "metrics" in data_copy and isinstance(data_copy["metrics"], dict):
            metrics_dict = data_copy.pop("metrics")
            if data_copy:
                metrics_dict.update(data_copy)
            base_metadata["metrics"] = json.dumps(metrics_dict)
        else:
            base_metadata["metrics"] = json.dumps(data_copy)

        if (
            "mask_empty" in base_metadata
            and base_metadata["mask_empty"] is not None
            and not isinstance(base_metadata["mask_empty"], int)
        ):
            base_metadata["mask_empty"] = int(base_metadata["mask_empty"])

        row_values = [base_metadata.get(col) for col in self.columns]

        with self.lock:
            self.buffer.append(row_values)
            if len(self.buffer) >= self.batch_size:
                self._flush_buffer()

    def flush(self):
        """Manually flush the buffer."""
        with self.lock:
            self._flush_buffer()

    def _flush_buffer(self):
        """Internal method to write buffered records to the database."""
        if not self.buffer:
            return

        def _execute_flush():
            if not self.conn:
                self.connect()
            assert self.conn is not None
            cursor = self.conn.cursor()
            placeholders = ", ".join(["?"] * len(self.columns))
            columns_str = ", ".join(self.columns)
            cursor.executemany(
                f"""
                INSERT OR REPLACE INTO metadata ({columns_str})
                VALUES ({placeholders})
            """,
                self.buffer,
            )
            self.conn.commit()
            self.buffer.clear()

        try:
            if self.error_handler:
                self.error_handler.with_retry(recoverable_exceptions=(sqlite3.OperationalError, sqlite3.DatabaseError))(
                    _execute_flush
                )()
            else:
                _execute_flush()
        except sqlite3.Error as e:
            try:
                self.logger.error(f"Failed to flush database buffer: {e}", exc_info=True)
            except Exception:
                # Logger might be closed during shutdown
                pass
            raise

    def load_all_metadata(self) -> List[Dict[str, Any]]:
        """Loads all metadata from the database."""
        self.flush()
        if not self.conn:
            self.connect()
        assert self.conn is not None
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM metadata")
        rows = cursor.fetchall()

        results = []
        for row in rows:
            row_dict = dict(row)
            if "metrics" in row_dict and isinstance(row_dict["metrics"], str):
                try:
                    row_dict.update(json.loads(row_dict["metrics"]))
                except json.JSONDecodeError:
                    pass
            results.append(row_dict)
        return results

    def count_errors(self) -> int:
        """Counts the number of records with errors."""
        self.flush()
        if not self.conn:
            self.connect()
        assert self.conn is not None
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM metadata WHERE error IS NOT NULL")
        result = cursor.fetchone()
        return result[0] if result else 0
