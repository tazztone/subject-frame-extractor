import os
from unittest.mock import patch

from core.fingerprint import RunFingerprint, create_fingerprint, fingerprints_match, load_fingerprint, save_fingerprint


def test_run_fingerprint_init():
    """Test RunFingerprint initialization and post_init."""
    fp = RunFingerprint(video_path="v.mp4", video_size=100, video_mtime=123.45, extraction_hash="abc")
    assert fp.video_path == "v.mp4"
    assert fp.created_at != ""


@patch("os.stat")
@patch("os.path.abspath")
def test_create_fingerprint(mock_abspath, mock_stat):
    """Test create_fingerprint function."""
    mock_stat.return_value.st_size = 1000
    mock_stat.return_value.st_mtime = 1234567.0
    mock_abspath.side_effect = lambda x: f"/abs/{x}"

    settings = {"method": "all"}
    fp = create_fingerprint("video.mp4", settings)

    assert fp.video_path == "/abs/video.mp4"
    assert fp.video_size == 1000
    assert fp.extraction_hash is not None


def test_save_load_fingerprint(tmp_path):
    """Test saving and loading fingerprint."""
    output_dir = str(tmp_path)
    fp = RunFingerprint(video_path="v.mp4", video_size=100, video_mtime=123.45, extraction_hash="abc")

    save_fingerprint(fp, output_dir)
    assert os.path.exists(os.path.join(output_dir, "run_fingerprint.json"))

    loaded = load_fingerprint(output_dir)
    assert loaded.video_path == fp.video_path
    assert loaded.video_size == fp.video_size
    assert loaded.extraction_hash == fp.extraction_hash


def test_load_fingerprint_none():
    """Test loading non-existent fingerprint."""
    assert load_fingerprint("/nonexistent") is None


def test_load_fingerprint_invalid_json(tmp_path):
    """Test loading invalid fingerprint JSON."""
    output_dir = str(tmp_path)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "run_fingerprint.json"), "w") as f:
        f.write("invalid json")

    assert load_fingerprint(output_dir) is None


def test_fingerprints_match():
    """Test fingerprints_match logic."""
    fp1 = RunFingerprint(video_path="v.mp4", video_size=100, video_mtime=123.45, extraction_hash="abc")
    fp2 = RunFingerprint(video_path="v.mp4", video_size=100, video_mtime=123.46, extraction_hash="abc")
    fp3 = RunFingerprint(video_path="v.mp4", video_size=100, video_mtime=125.0, extraction_hash="abc")
    fp4 = RunFingerprint(video_path="v.mp4", video_size=200, video_mtime=123.45, extraction_hash="abc")

    assert fingerprints_match(fp1, fp2) is True  # Within 1s slack
    assert fingerprints_match(fp1, fp3) is False  # Outside 1s slack
    assert fingerprints_match(fp1, fp4) is False  # Different size
