import pytest

from core.database import Database


@pytest.fixture
def database(tmp_path):
    """Provides a fresh in-memory or temporary SQLite database for tests."""
    db_path = tmp_path / "test.db"
    db = Database(str(db_path))
    db.connect()
    yield db
    db.close()
