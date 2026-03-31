import logging
import sqlite3
from unittest.mock import MagicMock, patch

import pytest

from core.database import Database
from core.db_schema import CURRENT_VERSION, migrate_database


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "metadata.db"


@pytest.fixture
def db(db_path):
    db = Database(db_path, batch_size=2)
    yield db
    db.close()


def test_initial_schema(db, db_path):
    assert db_path.exists()
    cursor = db.conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='metadata'")
    assert cursor.fetchone() is not None

    # Check current version in schema_versions
    cursor.execute("SELECT MAX(version) FROM schema_versions")
    assert cursor.fetchone()[0] == CURRENT_VERSION


def test_insert_metadata_and_flush(db):
    data = {"filename": "test.jpg", "face_sim": 0.9, "some_metric": 123}
    db.insert_metadata(data)
    # buffer not full yet (batch_size=2)
    assert len(db.buffer) == 1

    db.flush()
    assert len(db.buffer) == 0

    results = db.load_all_metadata()
    assert len(results) == 1
    assert results[0]["filename"] == "test.jpg"
    assert results[0]["face_sim"] == 0.9
    assert results[0]["some_metric"] == 123


def test_insert_metadata_replaces_duplicate(db):
    db.insert_metadata({"filename": "dup.jpg", "face_sim": 0.5})
    db.flush()
    db.insert_metadata({"filename": "dup.jpg", "face_sim": 0.8})
    db.flush()

    results = db.load_all_metadata()
    assert len(results) == 1
    assert results[0]["face_sim"] == 0.8


def test_insert_metadata_batch_flush(db):
    data1 = {"filename": "test1.jpg"}
    data2 = {"filename": "test2.jpg"}
    db.insert_metadata(data1)
    db.insert_metadata(data2)
    # Should auto flush because batch_size=2
    assert len(db.buffer) == 0

    results = db.load_all_metadata()
    assert len(results) == 2


def test_clear_metadata(db):
    db.insert_metadata({"filename": "test.jpg"})
    db.flush()
    db.clear_metadata()
    results = db.load_all_metadata()
    assert len(results) == 0


def test_count_errors(db):
    db.insert_metadata({"filename": "err.jpg", "error": "Something went wrong"})
    db.flush()
    assert db.count_errors() == 1

    db.clear_metadata()
    assert db.count_errors() == 0


def test_metrics_json_parsing(db):
    # Store complex dict in metrics
    metrics = {"nested": {"value": 42}, "list": [1, 2, 3]}
    db.insert_metadata({"filename": "metrics.jpg", "metrics": metrics})
    db.flush()

    results = db.load_all_metadata()
    assert results[0]["nested"]["value"] == 42
    assert results[0]["list"] == [1, 2, 3]


def test_load_all_metadata_json_error(db):
    # Manually insert invalid JSON into metrics
    cursor = db.conn.cursor()
    cursor.execute("INSERT INTO metadata (filename, metrics) VALUES (?, ?)", ("invalid.jpg", "{invalid json"))
    db.conn.commit()

    results = db.load_all_metadata()
    assert len(results) == 1
    assert results[0]["filename"] == "invalid.jpg"
    assert results[0]["metrics"] == "{invalid json"


def test_error_handler_integration(db_path):
    mock_handler = MagicMock()
    # Mock with_retry to return the original function (no-op decorator)
    mock_handler.with_retry.return_value = lambda x: x

    db = Database(db_path, batch_size=1)
    db.error_handler = mock_handler

    db.insert_metadata({"filename": "retry.jpg"})
    # Should call with_retry during _flush_buffer
    assert mock_handler.with_retry.called


def test_flush_exception_logging(db_path):
    logger = MagicMock()

    with patch("sqlite3.connect") as mock_connect:
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn
        mock_cursor = mock_conn.cursor.return_value

        # Mock schema version check
        mock_cursor.fetchone.return_value = [CURRENT_VERSION]

        # Trigger failure during executemany
        mock_cursor.executemany.side_effect = sqlite3.Error("Mock error")

        db = Database(db_path, logger=logger)
        db.insert_metadata({"filename": "error.jpg"})
        with pytest.raises(sqlite3.Error):
            db.flush()
        assert logger.error.called


def test_close_calls_flush(db_path):
    db = Database(db_path, batch_size=10)
    db.insert_metadata({"filename": "close.jpg"})
    assert len(db.buffer) == 1

    db.close()

    # Reopen and check
    db2 = Database(db_path)
    results = db2.load_all_metadata()
    assert len(results) == 1
    assert results[0]["filename"] == "close.jpg"
    db2.close()


def test_database_corruption_resilience(db_path):
    """Test behavior when the database file is corrupted (truncated)."""
    db = Database(db_path, batch_size=1)
    db.insert_metadata({"filename": "good.jpg"})
    db.flush()
    db.close()

    # Corrupt the file by truncating it
    with open(db_path, "wb") as f:
        f.write(b"NOT A SQLITE FILE")

    # Attempting to open and use it should raise an error or handle it
    with pytest.raises(sqlite3.DatabaseError):
        db2 = Database(db_path)
        db2.load_all_metadata()


def test_database_partial_write_failure(db_path):
    """Test behavior when flush fails mid-way."""
    with patch("sqlite3.connect") as mock_connect:
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn

        # Mock cursor and schema check
        mock_cursor = mock_conn.cursor.return_value
        mock_cursor.fetchone.return_value = [CURRENT_VERSION]

        db = Database(db_path, batch_size=2)
        db.insert_metadata({"filename": "1.jpg"})

        # Set side effect BEFORE it's triggered
        mock_conn.commit.side_effect = sqlite3.OperationalError("Disk full")

        with pytest.raises(sqlite3.OperationalError):
            db.insert_metadata({"filename": "2.jpg"})  # This triggers flush

        mock_conn.commit.side_effect = None  # Clear for teardown
        assert mock_conn.commit.called
        # Buffer should still have the data if flush failed before clearing
        assert len(db.buffer) == 2


def test_migration_v1_v2_logic(tmp_path):
    """Test the migration logic in db_schema directly."""
    db_path = tmp_path / "migration_test.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Setup v1 schema manually
    cursor.execute("""
        CREATE TABLE metadata (
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
    conn.commit()

    # Run migration
    logger = logging.getLogger("test")
    migrate_database(conn, logger)

    # Verify v2 column exists
    cursor.execute("PRAGMA table_info(metadata)")
    columns = [info[1] for info in cursor.fetchall()]
    assert "error_severity" in columns

    # Verify version recorded
    cursor.execute("SELECT MAX(version) FROM schema_versions")
    assert cursor.fetchone()[0] == CURRENT_VERSION
    conn.close()


def test_database_default_batch_size(db_path):
    """Test that Database initializes with the optimized default batch size."""
    db = Database(db_path)
    # Phase 2 goal: Increase from 50 to 100
    assert db.batch_size == 100
    db.close()
