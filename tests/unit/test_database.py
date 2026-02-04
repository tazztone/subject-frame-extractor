import sqlite3

import pytest

from core.database import Database


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "metadata.db"


@pytest.fixture
def db(db_path):
    db = Database(db_path, batch_size=2)
    db.connect()
    db.migrate()
    yield db
    db.close()


def test_initial_schema(db, db_path):
    assert db_path.exists()
    cursor = db.conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='metadata'")
    assert cursor.fetchone() is not None


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


def test_migration_adds_column(tmp_path):
    # Setup old schema
    db_path = tmp_path / "old.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
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
    """)  # Missing error_severity
    conn.commit()
    conn.close()

    db = Database(db_path)
    db.migrate()  # Should run migration

    cursor = db.conn.cursor()
    cursor.execute("PRAGMA table_info(metadata)")
    columns = [info[1] for info in cursor.fetchall()]
    assert "error_severity" in columns
    db.close()


def test_metrics_json_parsing(db):
    data = {"filename": "test.jpg", "custom_data": {"nested": "value"}}
    db.insert_metadata(data)
    db.flush()

    results = db.load_all_metadata()
    assert results[0]["custom_data"] == {"nested": "value"}


def test_mask_empty_conversion(db):
    data = {
        "filename": "test.jpg",
        "mask_empty": "1",  # String
    }
    db.insert_metadata(data)
    db.flush()

    results = db.load_all_metadata()
    assert results[0]["mask_empty"] == 1
    assert isinstance(results[0]["mask_empty"], int)


def test_wal_mode_enabled(db):
    cursor = db.conn.cursor()
    cursor.execute("PRAGMA journal_mode;")
    mode = cursor.fetchone()[0]
    assert mode.lower() == "wal"
