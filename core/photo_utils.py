from __future__ import annotations

import subprocess
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Union
import logging

# Configure logger
logger = logging.getLogger(__name__)

def extract_preview(raw_path: Path, output_dir: Path) -> Optional[Path]:
    """
    Extracts the embedded JPEG preview from a RAW file using ExifTool.

    Args:
        raw_path: Path to the RAW image file.
        output_dir: Directory to save the extracted preview.

    Returns:
        Path to the extracted preview file, or None if extraction failed.
    """
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    # Output filename: original_stem + _preview.jpg
    preview_path = output_dir / f"{raw_path.stem}_preview.jpg"

    # If preview already exists, return it (basic caching)
    if preview_path.exists():
        return preview_path

    exiftool_path = shutil.which("exiftool")
    if not exiftool_path:
        logger.error("ExifTool not found. Cannot extract preview.")
        return None

    # ExifTool command to extract the binary preview
    # -b: Binary output
    # -JpgFromRaw: Try to extract the JpgFromRaw tag (common in Nikon/Canon)
    # -PreviewImage: Fallback tag
    # -ThumbnailImage: Fallback tag
    # -w: Write to file
    # %d: directory, %f: filename
    
    # We use a simplified approach: pipe stdout to file or use specific tag extraction
    # The most robust way is asking for -JpgFromRaw first, then others.
    
    # Command: exiftool -b -JpgFromRaw src.CR2 > dst.jpg
    # If that fails (empty), try PreviewImage
    
    tags_to_try = ["-JpgFromRaw", "-PreviewImage", "-ThumbnailImage"]
    
    for tag in tags_to_try:
        try:
            # We redirect stdout to the file manually to handle the binary stream check
            cmd = [exiftool_path, "-b", tag, str(raw_path)]
            result = subprocess.run(cmd, capture_output=True, check=False)
            
            if result.returncode == 0 and len(result.stdout) > 0:
                # Success, write to file
                with open(preview_path, "wb") as f:
                    f.write(result.stdout)
                return preview_path
                
        except Exception as e:
            logger.debug(f"Failed to extract {tag} from {raw_path}: {e}")
            continue

    logger.warning(f"Could not extract any preview from {raw_path}")
    return None

def ingest_folder(folder_path: Path, output_dir: Path) -> List[Dict[str, Any]]:
    """
    Scans a folder for images, extracting previews for RAW files.

    Args:
        folder_path: Path to the source folder.
        output_dir: Directory to store extracted previews.

    Returns:
        List of dictionaries with photo metadata:
        [
            {
                "id": "file_stem",
                "source": Path(...),
                "preview": Path(...),
                "type": "raw" | "jpeg"
            },
            ...
        ]
    """
    if not folder_path.exists():
        logger.error(f"Source folder not found: {folder_path}")
        return []

    # Supported extensions
    raw_exts = {".CR2", ".NEF", ".ARW", ".DNG", ".ORF", ".RAF"}
    jpeg_exts = {".JPG", ".JPEG"}
    
    ingested_photos = []
    
    # Ensure output dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sort files for consistent order
    files = sorted([f for f in folder_path.iterdir() if f.is_file()])

    for file_path in files:
        ext = file_path.suffix.upper()
        photo_id = file_path.stem
        
        if ext in jpeg_exts:
            # For JPEGs, the source is the preview (or we could copy it if needed)
            # We'll just define the preview as the source itself to save space/time
            ingested_photos.append({
                "id": photo_id,
                "source": file_path,
                "preview": file_path,
                "type": "jpeg",
                "status": "unreviewed",
                "scores": {}
            })
            
        elif ext in raw_exts:
            # Extract preview
            preview = extract_preview(file_path, output_dir)
            if preview:
                ingested_photos.append({
                    "id": photo_id,
                    "source": file_path,
                    "preview": preview,
                    "type": "raw",
                    "status": "unreviewed",
                    "scores": {}
                })
        
    logger.info(f"Ingested {len(ingested_photos)} photos from {folder_path}")
    return ingested_photos
