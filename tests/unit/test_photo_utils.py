from pathlib import Path
from unittest.mock import MagicMock, patch

from core.photo_utils import extract_preview, ingest_folder


@patch("core.photo_utils.shutil.which")
@patch("core.photo_utils.subprocess.run")
def test_extract_preview_success(mock_run, mock_which, tmp_path):
    """Test successful preview extraction using ExifTool."""
    mock_which.return_value = "/usr/bin/exiftool"

    # Mock subprocess.run to return a fake binary JPEG (size > 25KB)
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = b"\xff\xd8" + b"\x00" * (30 * 1024)  # 30KB
    mock_run.return_value = mock_result

    raw_path = tmp_path / "test.CR2"
    raw_path.touch()
    output_dir = tmp_path / "previews"

    # Mock PIL Image to avoid actual resizing overhead
    with patch("core.photo_utils.Image.open") as mock_img_open:
        mock_img = MagicMock()
        mock_img.size = (2000, 2000)
        mock_img_open.return_value.__enter__.return_value = mock_img

        result = extract_preview(raw_path, output_dir)

        assert result is not None
        assert result.exists()
        assert result.name == "test_preview.jpg"
        assert mock_run.called


@patch("core.photo_utils.shutil.which")
def test_extract_preview_no_exiftool(mock_which, tmp_path):
    """Test extraction failure when ExifTool is missing."""
    mock_which.return_value = None
    raw_path = tmp_path / "test.CR2"
    output_dir = tmp_path / "previews"

    result = extract_preview(raw_path, output_dir)
    assert result is None


@patch("core.photo_utils.shutil.which")
@patch("core.photo_utils.subprocess.run")
def test_extract_preview_fail_all_tags(mock_run, mock_which, tmp_path):
    """Test extraction failure when all tags fail or return small data."""
    mock_which.return_value = "/usr/bin/exiftool"
    mock_run.return_value.returncode = 1
    mock_run.return_value.stdout = b""

    raw_path = tmp_path / "test.CR2"
    raw_path.touch()
    output_dir = tmp_path / "previews"

    result = extract_preview(raw_path, output_dir)
    assert result is None


def test_ingest_folder(tmp_path):
    """Test ingesting a folder with mixed JPEG and RAW files."""
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "img1.jpg").touch()
    (source_dir / "img2.CR2").touch()
    (source_dir / "random.txt").touch()

    output_dir = tmp_path / "output"

    with patch("core.photo_utils.extract_preview") as mock_extract:
        mock_extract.return_value = output_dir / "img2_preview.jpg"

        photos = ingest_folder(source_dir, output_dir)

        assert len(photos) == 2
        # Check JPEG
        assert any(p["id"] == "img1" and p["type"] == "jpeg" for p in photos)
        # Check RAW
        assert any(p["id"] == "img2" and p["type"] == "raw" for p in photos)


def test_extract_preview_exists(tmp_path):
    """Test extract_preview when output file already exists."""
    raw_path = tmp_path / "test.CR2"
    output_dir = tmp_path / "previews"
    output_dir.mkdir()
    preview_path = output_dir / "test_preview.jpg"
    preview_path.touch()

    result = extract_preview(raw_path, output_dir)
    assert result == preview_path


@patch("core.photo_utils.shutil.which")
@patch("core.photo_utils.subprocess.run")
def test_extract_preview_high_quality(mock_run, mock_which, tmp_path):
    """Test extract_preview with thumbnails_only=False."""
    mock_which.return_value = "/usr/bin/exiftool"
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = b"\xff\xd8" + b"\x00" * (30 * 1024)
    mock_run.return_value = mock_result

    raw_path = tmp_path / "test.CR2"
    raw_path.touch()
    output_dir = tmp_path / "previews"

    with patch("core.photo_utils.Image.open") as mock_img_open:
        mock_img = mock_img_open.return_value.__enter__.return_value
        mock_img.size = (100, 100)  # Small size, no resize

        result = extract_preview(raw_path, output_dir, thumbnails_only=False)
        assert result is not None


def test_ingest_folder_recursive(tmp_path):
    """Test recursive folder ingestion."""
    source_dir = tmp_path / "source"
    sub_dir = source_dir / "sub"
    sub_dir.mkdir(parents=True)
    (source_dir / "img1.jpg").touch()
    (sub_dir / "img2.jpg").touch()

    output_dir = tmp_path / "output"

    photos = ingest_folder(source_dir, output_dir, recursive=True)
    assert len(photos) == 2


def test_extract_preview_resize_exception(tmp_path):
    """Test handling of resizing exceptions."""
    raw_path = tmp_path / "test.CR2"
    raw_path.touch()
    output_dir = tmp_path / "previews"

    with patch("core.photo_utils.shutil.which", return_value="/bin/exiftool"):
        with patch("core.photo_utils.subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = b"\xff\xd8" + b"\x00" * (30 * 1024)

            with patch("core.photo_utils.Image.open") as mock_open:
                mock_open.side_effect = Exception("PIL error")
                result = extract_preview(raw_path, output_dir)
                assert result is not None  # Still returns path even if resize fails


def test_ingest_folder_nonexistent():
    """Test ingesting a non-existent folder."""
    assert ingest_folder(Path("/nonexistent"), Path("/out")) == []
