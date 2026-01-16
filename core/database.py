import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, List


class Database:
    CURRENT_VERSION = 2

    def __init__(self, db_path: Path, batch_size: int = 50):
        """
        Initializes the Database manager.

        Args:
            db_path: Path to the SQLite database file.
            batch_size: Number of records to buffer before writing.
        """
        self.db_path = db_path
        self.conn = None
        self.buffer = []
        self.batch_size = batch_size
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
            print(f"Migrating database from version {current_version} to {self.CURRENT_VERSION}...")

            try:
                if current_version < 1:
                    self._migration_v1_initial_schema(cursor)
                    cursor.execute("INSERT INTO schema_versions (version) VALUES (1)")
                    self.conn.commit()
                    print("Applied migration v1")

                if current_version < 2:
                    self._migration_v2_add_error_severity(cursor)
                    cursor.execute("INSERT INTO schema_versions (version) VALUES (2)")
                    self.conn.commit()
                    print("Applied migration v2")

            except Exception as e:
                self.conn.rollback()
                print(f"Migration failed: {e}")
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
        # Pop extra fields to be able to use the spreak operator ** later
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
        base_metadata = {key: metadata.pop(key, None) for key in keys_to_extract}

        # The rest of the metadata is a dictionary that we will store as a JSON string
        base_metadata["metrics"] = json.dumps(metadata)

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
            # TODO: Use logger instead of print for error reporting
            # TODO: Consider saving failed records to a recovery file
            print(f"Database error during flush: {e}")
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
