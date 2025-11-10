import sqlite3
import json
from pathlib import Path
from typing import List, Dict, Any

class Database:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.conn = None

    def connect(self):
        """Connects to the SQLite database."""
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row

    def close(self):
        """Closes the database connection."""
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
                phash TEXT,
                dedup_thresh INTEGER
            )
        """)
        self.conn.commit()

    def clear_metadata(self):
        """Deletes all records from the metadata table."""
        if not self.conn:
            self.connect()
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM metadata")
        self.conn.commit()

    def insert_metadata(self, metadata: Dict[str, Any]):
        """Inserts or replaces a metadata record."""
        if not self.conn:
            self.connect()

        #Pop extra fields to be able to use the spreak operator ** later
        keys_to_extract = ['filename', 'face_sim', 'face_conf', 'shot_id', 'seed_type', 'seed_face_sim', 'mask_area_pct', 'mask_empty', 'error', 'phash', 'dedup_thresh']
        base_metadata = {key: metadata.pop(key, None) for key in keys_to_extract}

        #The rest of the metadata is a dictionary that we will store as a JSON string
        base_metadata['metrics'] = json.dumps(metadata)

        #Make sure that the mask_empty field is an integer
        if 'mask_empty' in base_metadata and base_metadata['mask_empty'] is not None and not isinstance(base_metadata['mask_empty'], int) :
            base_metadata['mask_empty'] = int(base_metadata['mask_empty'])


        cursor = self.conn.cursor()
        cursor.execute(f"""
            INSERT OR REPLACE INTO metadata ({', '.join(base_metadata.keys())})
            VALUES ({', '.join('?' * len(base_metadata))})
        """, tuple(base_metadata.values()))
        self.conn.commit()

    def load_all_metadata(self) -> List[Dict[str, Any]]:
        """Loads all metadata from the database."""
        if not self.conn:
            self.connect()
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM metadata")
        rows = cursor.fetchall()

        results = []
        for row in rows:
            row_dict = dict(row)
            if 'metrics' in row_dict and isinstance(row_dict['metrics'], str) :
                row_dict.update(json.loads(row_dict['metrics']))
            results.append(row_dict)
        return results
