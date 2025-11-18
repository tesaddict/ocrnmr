"""Filename generation utilities."""

import re
import unicodedata
from pathlib import Path
from typing import Optional, Tuple


def sanitize_filename(name: str) -> str:
    """
    Sanitize a filename by removing/replacing characters unsafe on common filesystems.
    
    Removes or replaces:
    - Path separators: / \
    - Windows reserved: : * ? " < > |
    - Control characters (0x00-0x1F, 0x7F)
    - Leading/trailing spaces and dots (Windows doesn't allow these)
    - Very long filenames (truncate to 255 chars, leaving room for extension)
    """
    # Characters to replace with space or dash
    # Path separators and Windows reserved characters
    unsafe_replace = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
    for ch in unsafe_replace:
        name = name.replace(ch, ' - ')
    
    # Remove control characters (0x00-0x1F, 0x7F)
    # Also remove other problematic Unicode characters
    cleaned = []
    for char in name:
        # Skip control characters
        if ord(char) < 32 or ord(char) == 127:
            continue
        # Skip zero-width and other problematic Unicode characters
        category = unicodedata.category(char)
        if category.startswith('C'):  # Control, Format, Surrogate, Private Use
            continue
        cleaned.append(char)
    name = ''.join(cleaned)
    
    # Normalize Unicode (e.g., convert é to e, or keep as é depending on normalization)
    # Use NFKC to decompose and recompose, which helps with some special characters
    try:
        name = unicodedata.normalize('NFKC', name)
    except Exception:
        pass  # If normalization fails, continue with original
    
    # Collapse repeated spaces and dashes
    name = ' '.join(name.split())
    # Clean up multiple dashes/spaces combinations
    name = re.sub(r'\s*-\s*-\s*', ' - ', name)  # Multiple dashes
    name = re.sub(r'\s{2,}', ' ', name)  # Multiple spaces
    
    # Strip leading/trailing spaces, dots, and dashes (Windows doesn't allow these)
    name = name.strip(' .-')
    
    # Handle empty names (shouldn't happen, but be safe)
    if not name:
        name = "unnamed"
    
    # Truncate very long filenames (max 255 chars total, but we'll be conservative)
    # Leave room for extension (typically 4 chars: .mkv)
    # So limit base name to ~250 chars
    if len(name) > 250:
        name = name[:250].rstrip(' .-')
    
    # Windows doesn't allow filenames ending with space or dot
    # Also remove any remaining trailing dots/spaces
    name = name.rstrip(' .')
    
    return name
