import json
import sqlite3
import threading
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional


class Database:
    CURRENT_VERSION = 2

    def __init__(self, db_path: Path, batch_size: int = 50, logger: Optional[logging.Logger] = None):
        """
        Initializes the Database manager.

        Args:
            db_path: Path to the SQLite database file.
            batch_size: Number of records to buffer before writing.
            logger: Optional logger for errors and info.
        """
        self.db_path = db_path
        self.conn = None
        self.buffer = []
        self.batch_size = batch_size
        self.logger = logger or logging.getLogger(__name__)
        self.lock = threading.Lock()
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

    def connect(self):
        """Connects to the SQLite database."""
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=30.0)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.row_factory = sqlite3.Row

    def close(self):
        """Closes the database connection."""
        self.flush()
        if self.conn:
            self.conn.close()

    def migrate(self):
        """Applies database migrations to reach the current version."""
        cursor = self.conn.cursor()

        # Ensure schema_versions table exists
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS schema_versions (
                version INTEGER PRIMARY KEY,
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.commit()

        # Get current version
        cursor.execute("SELECT MAX(version) FROM schema_versions")
        row = cursor.fetchone()
        current_version = row[0] if row and row[0] is not None else 0

        # Handle legacy databases without schema_versions
        if current_version == 0:
            current_version = self._detect_legacy_version(cursor)
            # Record the detected starting version
            if current_version > 0:
                cursor.execute("INSERT INTO schema_versions (version) VALUES (?)", (current_version,))
                self.conn.commit()

        # Apply migrations
        if current_version < self.CURRENT_VERSION:
            self.logger.info(f"Migrating database from version {current_version} to {self.CURRENT_VERSION}...")

            try:
                if current_version < 1:
                    self._migration_v1_initial_schema(cursor)
                    cursor.execute("INSERT INTO schema_versions (version) VALUES (1)")
                    self.conn.commit()
                    self.logger.info("Applied migration v1")

                if current_version < 2:
                    self._migration_v2_add_error_severity(cursor)
                    cursor.execute("INSERT INTO schema_versions (version) VALUES (2)")
                    self.conn.commit()
                    self.logger.info("Applied migration v2")

            except Exception as e:
                self.conn.rollback()
                self.logger.error(f"Migration failed: {e}")
                raise

    def _detect_legacy_version(self, cursor) -> int:
        """Detects the schema version of a legacy database."""
        # Check if metadata table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='metadata'")
        if not cursor.fetchone():
            return 0

        # Check if error_severity column exists
        cursor.execute("PRAGMA table_info(metadata)")
        columns = [info[1] for info in cursor.fetchall()]
        if "error_severity" in columns:
            return 2

        return 1

    def _migration_v1_initial_schema(self, cursor):
        """Migration v1: Create initial metadata table and indexes."""
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT UNIQUE,
                metrics TEXT,
                face_sim REAL,
                face_conf REAL,
                shot_id INTEGER,
                seed_type TEXT,
                seed_face_sim REAL,
                mask_area_pct REAL,
                mask_empty INTEGER,
                error TEXT,
                phash TEXT,
                dedup_thresh INTEGER
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_shot_id ON metadata (shot_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_filename ON metadata (filename)")

    def _migration_v2_add_error_severity(self, cursor):
        """Migration v2: Add error_severity column."""
        # Check again just in case, though _detect_legacy_version should handle it
        cursor.execute("PRAGMA table_info(metadata)")
        columns = [info[1] for info in cursor.fetchall()]
        if "error_severity" not in columns:
            cursor.execute("ALTER TABLE metadata ADD COLUMN error_severity TEXT")

    # Removed create_tables as it is replaced by migrate()

    def clear_metadata(self):
        """Deletes all records from the metadata table."""
        with self.lock:
            self.buffer.clear()
            if not self.conn:
                self.connect()
            cursor = self.conn.cursor()
            cursor.execute("DELETE FROM metadata")
            self.conn.commit()

    def insert_metadata(self, metadata: Dict[str, Any]):
        """Inserts or replaces a metadata record."""
        # Work on a copy to avoid mutating the original dictionary passed by the caller
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

        # Use the metrics key if it already exists as a dict, otherwise use remaining data
        if "metrics" in data_copy and isinstance(data_copy["metrics"], dict):
            metrics_dict = data_copy.pop("metrics")
            # If there are other remaining keys, merge them into metrics_dict
            if data_copy:
                metrics_dict.update(data_copy)
            base_metadata["metrics"] = json.dumps(metrics_dict)
        else:
            # The rest of the metadata is a dictionary that we will store as a JSON string
            base_metadata["metrics"] = json.dumps(data_copy)

        # Make sure that the mask_empty field is an integer
        if (
            "mask_empty" in base_metadata
            and base_metadata["mask_empty"] is not None
            and not isinstance(base_metadata["mask_empty"], int)
        ):
            base_metadata["mask_empty"] = int(base_metadata["mask_empty"])

        # Ensure we have values for all columns in the correct order
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

        # Helper to ensure connection - though it should be open
        if not self.conn:
            self.connect()

        cursor = self.conn.cursor()
        placeholders = ", ".join(["?"] * len(self.columns))
        columns_str = ", ".join(self.columns)

        try:
            cursor.executemany(
                f"""
                INSERT OR REPLACE INTO metadata ({columns_str})
                VALUES ({placeholders})
            """,
                self.buffer,
            )
            self.conn.commit()
            self.buffer.clear()
        except sqlite3.Error as e:
            # TODO: Implement proper error recovery strategy (retry with backoff)
            self.logger.error(f"Database error during flush: {e}", exc_info=True)
            # Optional: Decide whether to clear buffer or retry?
            # For now, we clear to avoid getting stuck, but log error.
            self.buffer.clear()

    def load_all_metadata(self) -> List[Dict[str, Any]]:
        """Loads all metadata from the database."""
        self.flush()  # Ensure everything is written before reading
        if not self.conn:
            self.connect()
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
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM metadata WHERE error IS NOT NULL")
        result = cursor.fetchone()
        return result[0] if result else 0
