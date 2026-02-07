"""
E2E CLI Tests for Photo Mode.
Verified by running cli.py directly.
"""

import json
import subprocess
import sys
from pathlib import Path
import pytest
import shutil

@pytest.fixture
def photo_test_dir(tmp_path):
    """Creates a temporary directory with dummy JPEG photos."""
    photo_dir = tmp_path / "photos"
    photo_dir.mkdir()
    
    # Create 3 dummy JPEGs
    for i in range(1, 4):
        p = photo_dir / f"test_{i}.jpg"
        p.write_bytes(b"dummy jpeg content")
        
    return photo_dir

def run_cli(args):
    """Helper to run cli.py."""
    cmd = [sys.executable, "cli.py"] + args
    return subprocess.run(cmd, capture_output=True, text=True)

class TestPhotoCLI:
    """End-to-end tests for the 'photo' command group."""

    def test_full_photo_cli_workflow(self, photo_test_dir, tmp_path):
        session_dir = tmp_path / "session"
        
        # 1. Ingest
        print("Running photo ingest...")
        res = run_cli(["photo", "ingest", "--folder", str(photo_test_dir), "--output", str(session_dir)])
        assert res.returncode == 0
        assert "Ingested 3 photos" in res.stdout
        
        photos_json = session_dir / "photos.json"
        assert photos_json.exists()
        
        with open(photos_json) as f:
            photos = json.load(f)
        assert len(photos) == 3
        assert photos[0]["type"] == "jpeg"
        
        # 2. Score
        print("Running photo score...")
        # Use simple weights
        res = run_cli(["photo", "score", "--session", str(session_dir), "--weights", '{"sharpness": 1.0}'])
        assert res.returncode == 0
        assert "Scoring complete" in res.stdout
        
        with open(photos_json) as f:
            scored_photos = json.load(f)
        assert "quality_score" in scored_photos[0]["scores"]
        
        # 3. Export
        print("Running photo export...")
        res = run_cli(["photo", "export", "--session", str(session_dir)])
        assert res.returncode == 0
        assert "Exported 3 XMP files" in res.stdout
        
        # Verify XMPs exist next to source
        for i in range(1, 4):
            xmp = photo_test_dir / f"test_{i}.xmp"
            assert xmp.exists()
            content = xmp.read_text()
            assert "xmp:Rating" in content
            assert "xmp:Label" in content

    def test_ingest_invalid_folder(self, tmp_path):
        res = run_cli(["photo", "ingest", "--folder", "/non/existent/path", "--output", str(tmp_path)])
        # Click validates Path(exists=True) and exits with 2
        assert res.returncode == 2
        assert "does not exist" in res.stderr

    def test_score_missing_session(self, tmp_path):
        res = run_cli(["photo", "score", "--session", str(tmp_path)])
        assert res.returncode != 0
        assert "No ingested photos found" in res.stderr or "No ingested photos found" in res.stdout
