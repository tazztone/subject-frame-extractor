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
    
    # Create 3 dummy JPEGs (actual binary content to avoid PIL errors)
    from PIL import Image
    import numpy as np
    
    for i in range(1, 4):
        p = photo_dir / f"test_{i}.jpg"
        img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
        img.save(p)
        
    return photo_dir

def run_cli(args):
    """Helper to run cli.py."""
    cmd = [sys.executable, "cli.py"] + args
    res = subprocess.run(cmd, capture_output=True, text=True)
    # Always print output for debugging in -s mode
    if res.stdout: print(f"CLI STDOUT:\n{res.stdout}")
    if res.stderr: print(f"CLI STDERR:\n{res.stderr}")
    return res

class TestPhotoCLI:
    """End-to-end tests for the modern unified CLI workflow."""

    def test_full_photo_cli_workflow(self, photo_test_dir, tmp_path):
        session_dir = tmp_path / "session"
        
        # 1. Ingestion (using 'extract' command on a folder)
        print("Running photo ingestion...")
        res = run_cli(["extract", "--source", str(photo_test_dir), "--output", str(session_dir)])
        assert res.returncode == 0
        assert "Ingestion complete" in res.stdout
        
        frame_map_json = session_dir / "frame_map.json"
        assert frame_map_json.exists()
        
        # 2. Analysis (using 'analyze' command)
        print("Running photo analysis...")
        res = run_cli(["analyze", "--session", str(session_dir), "--source", str(photo_test_dir), "--verbose"])
        assert res.returncode == 0
        assert "Analysis complete" in res.stdout
        
        db_path = session_dir / "metadata.db"
        assert db_path.exists()
        
        # Verify database content (basic check)
        from core.database import Database
        db = Database(db_path)
        metadata = db.load_all_metadata()
        db.close()
        
        assert len(metadata) == 3
        # Check that metrics were computed (sharpness etc)
        assert "quality_score" in metadata[0]["metrics"]

    def test_extract_invalid_folder(self, tmp_path):
        res = run_cli(["extract", "--source", "/non/existent/path", "--output", str(tmp_path)])
        # Click validates Path(exists=True) and exits with 2
        assert res.returncode == 2
        assert "does not exist" in res.stderr

    def test_analyze_missing_session(self, tmp_path):
        # Point to an empty dir
        session_dir = tmp_path / "empty_session"
        session_dir.mkdir()
        res = run_cli(["analyze", "--session", str(session_dir), "--source", str(tmp_path)])
        # Should fail because frame_map.json is missing
        assert res.returncode != 0