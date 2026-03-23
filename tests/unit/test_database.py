import sqlite3
from unittest.mock import MagicMock, patch

import pytest

from core.database import Database


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "metadata.db"


@pytest.fixture
def db(db_path):
    db = Database(db_path, batch_size=2)
    # connect and migrate are called in __init__
    yield db
    db.close()


def test_initial_schema(db, db_path):
    assert db_path.exists()
    cursor = db.conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='metadata'")
    assert cursor.fetchone() is not None

    # Check current version in schema_versions
    cursor.execute("SELECT MAX(version) FROM schema_versions")
    assert cursor.fetchone()[0] == Database.CURRENT_VERSION


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


def test_migration_v1_v2(tmp_path):
    db_path = tmp_path / "v1.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    # Manual v1 schema
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
    conn.close()

    db = Database(db_path)
    # It should detect version 1 and migrate to CURRENT_VERSION
    cursor = db.conn.cursor()
    cursor.execute("PRAGMA table_info(metadata)")
    columns = [info[1] for info in cursor.fetchall()]
    assert "error_severity" in columns

    cursor.execute("SELECT MAX(version) FROM schema_versions")
    assert cursor.fetchone()[0] == Database.CURRENT_VERSION
    db.close()


@patch.object(Database, "migrate")
def test_detect_legacy_version_0(mock_migrate, tmp_path):
    db_path = tmp_path / "empty.db"
    db = Database(db_path)
    cursor = db.conn.cursor()
    # Should be 0 because no tables were created
    assert db._detect_legacy_version(cursor) == 0
    db.close()


def test_metrics_json_parsing(db):
    data = {"filename": "test.jpg", "custom_data": {"nested": "value"}}
    db.insert_metadata(data)
    db.flush()

    results = db.load_all_metadata()
    assert results[0]["custom_data"] == {"nested": "value"}


def test_load_all_metadata_json_error(db):
    # Manually insert invalid JSON
    db.flush()
    cursor = db.conn.cursor()
    cursor.execute("INSERT INTO metadata (filename, metrics) VALUES (?, ?)", ("bad.jpg", "{invalid json}"))
    db.conn.commit()

    results = db.load_all_metadata()
    assert len(results) == 1
    assert results[0]["filename"] == "bad.jpg"
    assert results[0]["metrics"] == "{invalid json}"


def test_error_handler_integration(db_path):
    mock_handler = MagicMock()
    # with_retry should return a decorator
    mock_handler.with_retry.return_value = lambda x: x

    db = Database(db_path)
    db.error_handler = mock_handler

    db.insert_metadata({"filename": "retry.jpg"})
    db.flush()

    mock_handler.with_retry.assert_called()


def test_flush_exception_logging(db_path):
    # Cause a flush error by closing connection prematurely or mocking
    db = Database(db_path)
    db.buffer = [["test.jpg"] + [None] * 12]  # Correct length

    with patch.object(db, "conn") as mock_conn:
        mock_conn.cursor.side_effect = sqlite3.Error("Test error")
        with pytest.raises(sqlite3.Error):
            db.flush()
    db.close()


def test_close_calls_flush(db_path):
    db = Database(db_path)
    with patch.object(db, "_flush_buffer") as mock_flush:
        db.close()
        mock_flush.assert_called_once()
