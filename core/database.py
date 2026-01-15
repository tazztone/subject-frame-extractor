import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, List


# TODO: Add database schema versioning and automatic migration support
# TODO: Consider using SQLAlchemy for more robust ORM features
# TODO: Implement connection pooling for concurrent access
class Database:
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

    def connect(self):
        """Connects to the SQLite database."""
        # TODO: Add connection timeout configuration
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.row_factory = sqlite3.Row

    def close(self):
        """Closes the database connection."""
        self.flush()
        if self.conn:
            self.conn.close()

    def create_tables(self):
        """Creates the necessary tables if they don't exist."""
        if not self.conn:
            self.connect()
        cursor = self.conn.cursor()
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
                error_severity TEXT,
                phash TEXT,
                dedup_thresh INTEGER
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_shot_id ON metadata (shot_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_filename ON metadata (filename)")

        # Migration: Add error_severity if missing
        cursor.execute("PRAGMA table_info(metadata)")
        columns = [info[1] for info in cursor.fetchall()]
        if "error_severity" not in columns:
            try:
                cursor.execute("ALTER TABLE metadata ADD COLUMN error_severity TEXT")
            except sqlite3.OperationalError:
                pass  # Column might have been added concurrently

        self.conn.commit()

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
