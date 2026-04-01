import sqlite3
from unittest.mock import MagicMock, patch

import pytest

from core.db_schema import CURRENT_VERSION, migrate_database


def test_migrate_database_new(tmp_path):
    """Test migration on a fresh database."""
    db_path = tmp_path / "new.db"
    conn = sqlite3.connect(db_path)
    logger = MagicMock()

    migrate_database(conn, logger)

    # Verify current version is recorded
    cursor = conn.cursor()
    cursor.execute("SELECT MAX(version) FROM schema_versions")
    assert cursor.fetchone()[0] == CURRENT_VERSION

    # Verify table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='metadata'")
    assert cursor.fetchone() is not None
    conn.close()


def test_migrate_database_legacy_v1(tmp_path):
    """Test migration from a legacy v1 database (no schema_versions table)."""
    db_path = tmp_path / "legacy_v1.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create v1 schema manually
    cursor.execute("""
        CREATE TABLE metadata (
            id INTEGER PRIMARY KEY,
            filename TEXT UNIQUE,
            metrics TEXT
        )
    """)
    conn.commit()

    logger = MagicMock()
    migrate_database(conn, logger)

    # Should detect v1, record it, then apply v2 (or latest)
    cursor.execute("SELECT MAX(version) FROM schema_versions")
    assert cursor.fetchone()[0] == CURRENT_VERSION  # Current version

    # Check if error_severity was added
    cursor.execute("PRAGMA table_info(metadata)")
    columns = [info[1] for info in cursor.fetchall()]
    assert "error_severity" in columns
    conn.close()


def test_migrate_database_rollback_on_failure(tmp_path):
    """Test that migration rolls back on failure."""
    db_path = tmp_path / "fail.db"
    conn = sqlite3.connect(db_path)
    logger = MagicMock()

    # Mock cursor to fail on a specific migration step
    # This is slightly complex with real sqlite, let's use a mock connection if needed
    # but we can also just try to trigger a real error if possible.

    with patch("core.db_schema._migration_v1_initial_schema", side_effect=RuntimeError("Migration Fail")):
        with pytest.raises(RuntimeError, match="Migration Fail"):
            migrate_database(conn, logger)

    # schema_versions table should still exist but version 1 should not be recorded
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='schema_versions'")
    assert cursor.fetchone() is not None

    cursor.execute("SELECT MAX(version) FROM schema_versions")
    row = cursor.fetchone()
    assert row[0] is None or row[0] == 0
    conn.close()
