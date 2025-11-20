import pytest
from ocrnmr.filename import sanitize_filename

def test_sanitize_basic():
    assert sanitize_filename("Normal File") == "Normal File"

def test_sanitize_illegal_chars():
    # / \ : * ? " < > |
    # Replaces with " - "
    assert sanitize_filename("File/Name") == "File - Name"
    assert sanitize_filename("File\\Name") == "File - Name"
    assert sanitize_filename("File:Name") == "File - Name"
    assert sanitize_filename("File*Name") == "File - Name"
    assert sanitize_filename("File?Name") == "File - Name"
    assert sanitize_filename('File"Name') == "File - Name"
    assert sanitize_filename("File<Name") == "File - Name"
    assert sanitize_filename("File>Name") == "File - Name"
    assert sanitize_filename("File|Name") == "File - Name"

def test_sanitize_collapse_spaces():
    assert sanitize_filename("File  Name") == "File Name"
    assert sanitize_filename("File   Name") == "File Name"
    assert sanitize_filename("File -  - Name") == "File - Name"

def test_sanitize_trim():
    assert sanitize_filename(" File ") == "File"
    assert sanitize_filename("File.") == "File"
    assert sanitize_filename(".File") == "File"

def test_sanitize_unicode_control():
    # Control characters should be removed
    assert sanitize_filename("File\x00Name") == "FileName"
    assert sanitize_filename("File\nName") == "FileName"

def test_sanitize_length():
    # Create a very long filename
    long_name = "a" * 300
    sanitized = sanitize_filename(long_name)
    assert len(sanitized) <= 255
    assert sanitized == "a" * 250

def test_sanitize_empty():
    assert sanitize_filename("") == "unnamed"
    assert sanitize_filename("   ") == "unnamed"

