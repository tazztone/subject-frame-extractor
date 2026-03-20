"""
Core module for Subject Frame Extractor.
"""

from core.sam3_patches import apply_patches

# Apply all runtime patches for SAM3 (Resource patching, HWC fixing, Dtype forcing)
# This MUST happen early to avoid deprecation warnings and stability issues.
apply_patches()
