# OCR Name Matcher (ocrnmr)

A CLI-driven tool for extracting episode titles from video files using OCR and automatically renaming them.

## Installation

```bash
pip install -e .
```

Or install dependencies manually:
```bash
pip install easyocr ffmpeg-python pillow rich rapidfuzz requests
```

## Quick Start

```bash
# Basic usage with TMDB (no config file needed):
ocrnmr --show "Star Trek: The Next Generation" --season 1 --input ~/videos/season1/

# Preview changes without renaming:
ocrnmr --show "The Office" --season 1 --input ~/videos/ --dry-run
```

## Usage

### Required Arguments

- `--show`: TV show name (e.g., "Star Trek: The Next Generation")
- `--season`: Season number (e.g., 1)
- `--input`: Input directory path containing video files

### Optional Arguments

- `--episodes-file`: Path to JSON file containing custom episodes array (only needed when TMDB doesn't have complete info)
- `--tmdb-key`: TMDB API key (optional, can also use `TMDB_API_KEY` environment variable)
- `--dry-run`: Preview renames without executing

### OCR Settings (all optional with defaults)

- `--max-dimension`: Maximum frame dimension in pixels (default: 800, use 0 for full resolution)
- `--duration`: Maximum duration to process in seconds (default: 600 = 10 minutes, use 0 for full video)
- `--frame-interval`: Interval between frames in seconds (default: 2.0)
- `--match-threshold`: Episode matching threshold 0.0-1.0 (default: 0.6, lower = more lenient)
- `--hwaccel`: Hardware acceleration: `videotoolbox` (macOS), `vaapi` (Linux), `d3d11va`/`dxva2` (Windows). Default: None (software)

### Examples

```bash
# Basic usage with TMDB:
ocrnmr --show "Stargate SG-1" --season 2 --input ~/videos/

# With custom episodes (episodes file required):
ocrnmr --show "Dragon Ball Z Kai" --season 5 --input ~/videos/ --episodes-file episodes.json

# Override OCR settings for better accuracy:
ocrnmr --show "The Office" --season 1 --input ~/videos/ --max-dimension 1200 --duration 900

# Use hardware acceleration on macOS:
ocrnmr --show "Star Trek: TNG" --season 1 --input ~/videos/ --hwaccel videotoolbox

# More lenient matching (lower threshold):
ocrnmr --show "The Simpsons" --season 1 --input ~/videos/ --match-threshold 0.5
```

## Episodes File Format

The episodes file (`--episodes-file`) is **optional** and only needed when TMDB doesn't have complete episode information. It should contain:

```json
{
  "episodes": [
    {"episode": 1, "title": "Episode Title 1"},
    {"episode": 1, "title": "Episode Title Variant"},
    {"episode": 2, "title": "Episode Title 2"}
  ]
}
```

**Key points:**
- Each episode can have multiple title variants (useful for fuzzy matching)
- Episode numbers are explicit (prevents errors)
- Episodes file is ONLY for custom episodes - all other settings come from CLI arguments

## How It Works

1. **Frame Extraction**: Uses FFMPEG to extract frames from video files at specified intervals
2. **OCR Processing**: Uses EasyOCR to extract text from frames
3. **Episode Matching**: Matches extracted text against episode titles (from TMDB or config) using fuzzy matching
4. **Rename Preview**: Shows preview of all matches before renaming
5. **File Renaming**: Automatically renames files with format: `Show Name - S01E01 - Episode Title.mkv`

## Features

- **CLI-first interface**: No config file needed for basic usage
- **TMDB integration**: Automatically fetches episode titles from TMDB
- **Custom episodes**: Support for manual episode lists when TMDB is incomplete
- **Pipelined processing**: Extracts frames in background while OCR processes current file
- **Interactive display**: Real-time progress display with split panes for OCR and FFMPEG status
- **Rename preview**: Always shows preview before applying changes
- **Hardware acceleration**: Supports VideoToolbox (macOS), VAAPI (Linux), D3D11VA/DXVA2 (Windows)
- **Dry-run mode**: Preview changes without renaming files

## Project Structure

```
ocrnmr/
├── ocrnmr/              # Package directory
│   ├── __init__.py
│   ├── __main__.py      # Entry point
│   ├── cli.py           # CLI interface
│   ├── display.py       # Display module
│   ├── episode_matcher.py
│   ├── exit_flag.py
│   ├── filename.py      # Filename utilities
│   ├── frame_extractor.py
│   ├── ocr_engine.py
│   ├── processor.py
│   ├── title_card_scanner.py
│   └── tmdb_client.py  # TMDB API client
├── test_ocr_single.py   # Test script
└── README.md
```

## License

Same as parent renamer project.
