from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

def write_xmp_sidecar(source_path: Path, rating: int, label: str) -> Optional[Path]:
    """
    Writes an XMP sidecar file compatible with Adobe Lightroom/Bridge.

    Args:
        source_path: Path to the original image file (e.g., .CR2).
        rating: Star rating (0-5).
        label: Color label (e.g., "Red", "Green", "Blue", "Purple", "Yellow").

    Returns:
        Path to the written XMP file, or None if failed.
    """
    xmp_path = source_path.with_suffix(".xmp")
    
    # Namespaces
    NS_RDF = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
    NS_XMP = "http://ns.adobe.com/xap/1.0/"
    
    ET.register_namespace("rdf", NS_RDF)
    ET.register_namespace("x", "adobe:ns:meta/")
    ET.register_namespace("xmp", NS_XMP)

    # Root: <x:xmpmeta>
    xmpmeta = ET.Element("{adobe:ns:meta/}xmpmeta", attrib={"{adobe:ns:meta/}xmptk": "Adobe XMP Core 5.6-c140 79.160451, 2017/05/06-01:08:21"})
    
    # RDF Wrapper
    rdf = ET.SubElement(xmpmeta, f"{{{NS_RDF}}}RDF")
    
    # Description
    desc = ET.SubElement(rdf, f"{{{NS_RDF}}}Description", attrib={
        f"{{{NS_RDF}}}about": "",
        f"{{{NS_XMP}}}Rating": str(rating),
        f"{{{NS_XMP}}}Label": label
    })
    
    try:
        tree = ET.ElementTree(xmpmeta)
        tree.write(xmp_path, encoding="utf-8", xml_declaration=True)
        return xmp_path
    except Exception as e:
        logger.error(f"Failed to write XMP for {source_path}: {e}")
        return None

def export_xmps_for_photos(photos: List[Dict], star_thresholds: List[int] = None) -> int:
    """
    Writes XMP sidecars for all photos in the list.
    
    Args:
        photos: List of photo metadata dicts.
        star_thresholds: Optional list of score thresholds for 1-5 stars.
                         Video-centric default: [20, 40, 60, 80, 90]
    
    Returns:
        Count of successfully written XMP files.
    """
    if star_thresholds is None:
        star_thresholds = [20, 40, 60, 80, 90]
        
    count = 0
    for p in photos:
        score = p.get("scores", {}).get("quality_score", 0.0)
        status = p.get("status", "unreviewed")
        
        # Calculate Rating (Stars)
        rating = 0
        for i, thresh in enumerate(star_thresholds):
            if score >= thresh:
                rating = i + 1
        
        # Calculate Label (Color)
        # Lightroom standard: Red (6), Yellow (7), Green (8), Blue (9), Purple (?)
        # Typically mapped via string names in XMP: "Red", "Green", etc.
        label = ""
        if status == "kept":
            label = "Green"
        elif status == "rejected":
            label = "Red"
        elif status == "review":
            label = "Yellow"
            
        if write_xmp_sidecar(p["source"], rating, label):
            count += 1
            
    return count
