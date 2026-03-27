"""
Database schema and migration logic for SQLite.
"""

import logging
import sqlite3
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from core.logger import AppLogger

CURRENT_VERSION = 2


def migrate_database(conn: sqlite3.Connection, logger: Union[logging.Logger, "AppLogger"]):
    """Applies database migrations to reach the current version."""
    cursor = conn.cursor()

    # Ensure schema_versions table exists
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS schema_versions (
            version INTEGER PRIMARY KEY,
            applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()

    # Get current version
    cursor.execute("SELECT MAX(version) FROM schema_versions")
    row = cursor.fetchone()
    current_version = row[0] if row and row[0] is not None else 0

    # Handle legacy databases without schema_versions
    if current_version == 0:
        current_version = _detect_legacy_version(cursor)
        # Record the detected starting version
        if current_version > 0:
            cursor.execute("INSERT INTO schema_versions (version) VALUES (?)", (current_version,))
            conn.commit()

    # Apply migrations
    if current_version < CURRENT_VERSION:
        logger.info(f"Migrating database from version {current_version} to {CURRENT_VERSION}...")

        try:
            if current_version < 1:
                _migration_v1_initial_schema(cursor)
                cursor.execute("INSERT INTO schema_versions (version) VALUES (1)")
                conn.commit()
                logger.info("Applied migration v1")

            if current_version < 2:
                _migration_v2_add_error_severity(cursor)
                cursor.execute("INSERT INTO schema_versions (version) VALUES (2)")
                conn.commit()
                logger.info("Applied migration v2")

        except Exception as e:
            conn.rollback()
            logger.error(f"Migration failed: {e}")
            raise


def _detect_legacy_version(cursor) -> int:
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


def _migration_v1_initial_schema(cursor):
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


def _migration_v2_add_error_severity(cursor):
    """Migration v2: Add error_severity column."""
    cursor.execute("PRAGMA table_info(metadata)")
    columns = [info[1] for info in cursor.fetchall()]
    if "error_severity" not in columns:
        cursor.execute("ALTER TABLE metadata ADD COLUMN error_severity TEXT")
