import sqlite3
from unittest.mock import MagicMock, patch

import pytest

from core.db_schema import CURRENT_VERSION, _migration_v2_add_error_severity, migrate_database


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

    # Should detect v1, record it, then apply latest migrations
    cursor.execute("SELECT MAX(version) FROM schema_versions")
    assert cursor.fetchone()[0] == CURRENT_VERSION  # Ensure it reached latest version

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


def test_migrate_database_legacy_v2(tmp_path):
    """Test migration from a legacy v2 database (no schema_versions table, but has error_severity)."""
    db_path = tmp_path / "legacy_v2.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create v2 schema manually without schema_versions
    cursor.execute("""
        CREATE TABLE metadata (
            id INTEGER PRIMARY KEY,
            filename TEXT UNIQUE,
            metrics TEXT,
            error_severity TEXT
        )
    """)
    conn.commit()

    logger = MagicMock()
    migrate_database(conn, logger)

    # Should detect v2, record it, and since it is CURRENT_VERSION it doesn't apply migrations
    cursor.execute("SELECT MAX(version) FROM schema_versions")
    assert cursor.fetchone()[0] == CURRENT_VERSION

    # Verify error_severity is present
    cursor.execute("PRAGMA table_info(metadata)")
    columns = [info[1] for info in cursor.fetchall()]
    assert "error_severity" in columns
    conn.close()


def test_migrate_database_future_version_mocked(tmp_path):
    """Test migration path when a future version requires migrations but skips past ones."""
    db_path = tmp_path / "future.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE schema_versions (version INTEGER PRIMARY KEY)")
    cursor.execute("INSERT INTO schema_versions (version) VALUES (?)", (2,))
    conn.commit()

    logger = MagicMock()
    with patch("core.db_schema.CURRENT_VERSION", 3):
        migrate_database(conn, logger)

    # Note: Because there's no actual logic for CURRENT_VERSION=3 in db_schema.py,
    # it just completes execution without error and leaves version at 2.
    # We are just testing it handles skipping the inner blocks successfully.
    cursor.execute("SELECT MAX(version) FROM schema_versions")
    assert cursor.fetchone()[0] == 2
    conn.close()


def test_migration_v2_column_already_exists(tmp_path):
    """Test _migration_v2_add_error_severity when the column is already present."""
    db_path = tmp_path / "v2_already.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create metadata table with error_severity
    cursor.execute("""
        CREATE TABLE metadata (
            id INTEGER PRIMARY KEY,
            error_severity TEXT
        )
    """)
    conn.commit()

    # Should not raise any exception about column already existing
    _migration_v2_add_error_severity(cursor)

    # Verify column is still there
    cursor.execute("PRAGMA table_info(metadata)")
    columns = [info[1] for info in cursor.fetchall()]
    assert "error_severity" in columns
    conn.close()


def test_migrate_database_from_v1_with_schema_versions(tmp_path):
    """Test migration from v1 to v2 when schema_versions already exists and is at version 1."""
    db_path = tmp_path / "v1_with_schema.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create v1 schema and initialize schema_versions to 1
    cursor.execute("""
        CREATE TABLE metadata (
            id INTEGER PRIMARY KEY,
            filename TEXT UNIQUE,
            metrics TEXT
        )
    """)
    cursor.execute("CREATE TABLE schema_versions (version INTEGER PRIMARY KEY)")
    cursor.execute("INSERT INTO schema_versions (version) VALUES (1)")
    conn.commit()

    logger = MagicMock()
    migrate_database(conn, logger)

    # Should update to CURRENT_VERSION and add error_severity column
    cursor.execute("SELECT MAX(version) FROM schema_versions")
    assert cursor.fetchone()[0] == CURRENT_VERSION

    cursor.execute("PRAGMA table_info(metadata)")
    columns = [info[1] for info in cursor.fetchall()]
    assert "error_severity" in columns
    conn.close()


def test_migrate_database_already_updated(tmp_path):
    """Test migration on a database that is already at the current version."""
    db_path = tmp_path / "updated.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE schema_versions (version INTEGER PRIMARY KEY)")
    cursor.execute("INSERT INTO schema_versions (version) VALUES (?)", (CURRENT_VERSION,))
    conn.commit()

    logger = MagicMock()
    migrate_database(conn, logger)

    # Verify no new versions added and version is still CURRENT_VERSION
    cursor.execute("SELECT MAX(version) FROM schema_versions")
    assert cursor.fetchone()[0] == CURRENT_VERSION
    conn.close()


def test_migrate_database_failure(mock_logger):
    """Test migrate_database handles exceptions and triggers rollback."""
    mock_conn = MagicMock()
    mock_cursor = mock_conn.cursor.return_value

    # 1. First call to get MAX(version) returns 0 to trigger migration
    # 2. Second call during _detect_legacy_version returns None (no metadata table)
    mock_cursor.fetchone.side_effect = [(0,), None]

    # Force exception during migration body
    with patch("core.db_schema._migration_v1_initial_schema", side_effect=sqlite3.OperationalError("locked")):
        with pytest.raises(sqlite3.OperationalError):
            migrate_database(mock_conn, mock_logger)

    mock_conn.rollback.assert_called_once()
    mock_logger.error.assert_called()
