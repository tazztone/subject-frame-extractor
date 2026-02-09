from __future__ import annotations

import subprocess
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import logging
from PIL import Image

# Configure logger
logger = logging.getLogger(__name__)

def extract_preview(raw_path: Path, output_dir: Path, thumbnails_only: bool = True) -> Optional[Path]:
    """
    Extracts the embedded JPEG preview from a RAW file using ExifTool.

    Args:
        raw_path: Path to the RAW image file.
        output_dir: Directory to save the extracted preview.
        thumbnails_only: If True, prioritizes smaller ThumbnailImage over JpgFromRaw.

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
    # We want a usable preview. 'PreviewImage' is usually the best balance (~100-500KB).
    # 'JpgFromRaw' is high quality but large (multi-MB).
    # 'ThumbnailImage' is often too small (<20KB) for a main UI preview.
    
    if thumbnails_only:
        # Prioritize balance: PreviewImage first
        tags_to_try = ["-PreviewImage", "-ThumbnailImage", "-JpgFromRaw"]
    else:
        # Prioritize quality: JpgFromRaw is full resolution
        tags_to_try = ["-JpgFromRaw", "-PreviewImage", "-ThumbnailImage"]
    
    MIN_USABLE_SIZE = 25 * 1024  # 25KB threshold to skip tiny thumbnails
    
    for tag in tags_to_try:
        try:
            # We redirect stdout to the file manually to handle the binary stream check
            cmd = [exiftool_path, "-b", tag, str(raw_path)]
            result = subprocess.run(cmd, capture_output=True, check=False)
            
            if result.returncode == 0 and len(result.stdout) > MIN_USABLE_SIZE:
                # Success, write to file
                with open(preview_path, "wb") as f:
                    f.write(result.stdout)
                
                # Task 2.1: Resize if too large
                try:
                    with Image.open(preview_path) as img:
                        if max(img.size) > 1000:
                            img.thumbnail((1000, 1000), Image.Resampling.LANCZOS)
                            img.save(preview_path, quality=85, optimize=True)
                            logger.debug(f"Resized preview {preview_path} to max 1000px")
                except Exception as e:
                    logger.warning(f"Failed to resize extracted preview {preview_path}: {e}")
                
                return preview_path
                
        except Exception as e:
            logger.debug(f"Failed to extract {tag} from {raw_path}: {e}")
            continue

    logger.warning(f"Could not extract any preview from {raw_path}")
    return None

def ingest_folder(folder_path: Path, output_dir: Path, recursive: bool = False, thumbnails_only: bool = True) -> List[Dict[str, Any]]:
    """
    Scans a folder for images, extracting previews for RAW files.

    Args:
        folder_path: Path to the source folder.
        output_dir: Directory to store extracted previews.
        recursive: Whether to scan subdirectories.
        thumbnails_only: Passed to extract_preview.

    Returns:
        List of dictionaries with photo metadata.
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
    if recursive:
        files = sorted(list(folder_path.rglob("*")))
    else:
        files = sorted(list(folder_path.iterdir()))

    for file_path in files:
        if not file_path.is_file():
            continue
            
        ext = file_path.suffix.upper()
        photo_id = file_path.stem
        
        if ext in jpeg_exts:
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
            preview = extract_preview(file_path, output_dir, thumbnails_only=thumbnails_only)
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
