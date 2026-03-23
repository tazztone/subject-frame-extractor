import xml.etree.ElementTree as ET
from pathlib import Path

from core.xmp_writer import export_xmps_for_photos, write_xmp_sidecar


def test_write_xmp_sidecar(tmp_path):
    """Test writing a single XMP sidecar file."""
    source_path = tmp_path / "test.jpg"
    source_path.touch()

    xmp_path = write_xmp_sidecar(source_path, 5, "Green")

    assert xmp_path is not None
    assert xmp_path.exists()
    assert xmp_path.suffix == ".xmp"

    # Verify XML content
    tree = ET.parse(xmp_path)
    root = tree.getroot()
    # Check Rating and Label
    # Namespaces are tricky in ET, but we can search by tag name
    desc = root.find(".//{http://www.w3.org/1999/02/22-rdf-syntax-ns#}Description")
    assert desc.attrib["{http://ns.adobe.com/xap/1.0/}Rating"] == "5"
    assert desc.attrib["{http://ns.adobe.com/xap/1.0/}Label"] == "Green"


def test_export_xmps_for_photos(tmp_path):
    """Test exporting XMPs for multiple photos."""
    p1 = tmp_path / "p1.jpg"
    p1.touch()
    p2 = tmp_path / "p2.jpg"
    p2.touch()

    photos = [
        {"source": p1, "scores": {"quality_score": 95}, "status": "kept"},
        {"source": p2, "scores": {"quality_score": 30}, "status": "review"},
        {"source": tmp_path / "p3.jpg", "scores": {"quality_score": 10}, "status": "rejected"},
    ]
    (tmp_path / "p3.jpg").touch()

    count = export_xmps_for_photos(photos)
    assert count == 3
    assert (tmp_path / "p1.xmp").exists()
    assert (tmp_path / "p2.xmp").exists()


def test_write_xmp_sidecar_fail(tmp_path):
    """Test handling of write failures."""
    # Read-only directory or invalid path
    source_path = Path("/nonexistent/test.jpg")
    result = write_xmp_sidecar(source_path, 1, "Red")
    assert result is None
