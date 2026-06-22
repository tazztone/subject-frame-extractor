import os

from core.fingerprint import RunFingerprint, create_fingerprint, fingerprints_match, load_fingerprint, save_fingerprint


def test_run_fingerprint_init():
    """Test RunFingerprint initialization and post_init."""
    fp = RunFingerprint(video_path="v.mp4", video_size=100, video_mtime=123.45, extraction_hash="abc")
    assert fp.video_path == "v.mp4"
    assert fp.created_at != ""


def test_create_fingerprint(tmp_path):
    """Test create_fingerprint function."""
    video_path = tmp_path / "video.mp4"
    video_path.write_text("dummy video content")

    settings = {"method": "all"}
    fp = create_fingerprint(str(video_path), settings)

    assert fp.video_path == str(video_path.resolve())
    assert fp.video_size == video_path.stat().st_size
    assert fp.video_mtime == video_path.stat().st_mtime
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


def test_create_fingerprint_deterministic(tmp_path):
    """Test that fingerprint hashes are deterministic regardless of dict insertion order."""
    video_path = tmp_path / "video.mp4"
    video_path.write_text("dummy video content")

    # Dictionaries with same content but different insertion order
    settings_a = {"a": 1, "b": {"c": 2, "d": [3, 4]}}
    settings_b = {"b": {"d": [3, 4], "c": 2}, "a": 1}

    fp_a = create_fingerprint(str(video_path), settings_a)
    fp_b = create_fingerprint(str(video_path), settings_b)

    assert fp_a.extraction_hash == fp_b.extraction_hash

    # Test with types that require default=str serialization
    from datetime import datetime

    now = datetime.now()

    settings_c = {"time": now, "path": tmp_path}
    settings_d = {"path": tmp_path, "time": now}

    fp_c = create_fingerprint(str(video_path), settings_c)
    fp_d = create_fingerprint(str(video_path), settings_d)

    assert fp_c.extraction_hash == fp_d.extraction_hash


def test_create_fingerprint_analysis_settings(tmp_path):
    """Test create_fingerprint with and without analysis_settings."""
    video_path = tmp_path / "video.mp4"
    video_path.write_text("dummy video content")

    settings = {"method": "all"}

    # Without analysis_settings
    fp_no_analysis = create_fingerprint(str(video_path), settings)
    assert fp_no_analysis.analysis_hash is None

    # With analysis_settings
    analysis_settings = {"face_detection": True}
    fp_with_analysis = create_fingerprint(str(video_path), settings, analysis_settings)
    assert fp_with_analysis.analysis_hash is not None

    # Ensure extraction hash is unchanged
    assert fp_no_analysis.extraction_hash == fp_with_analysis.extraction_hash
