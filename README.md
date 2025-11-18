# OCR Name Matcher (ocrnmr)

A standalone application for extracting episode titles from video files using OCR and generating rename previews.

## Installation

```bash
cd /Users/methos/source/ocrnmr
pip install -e .
```

Or install dependencies manually:
```bash
pip install easyocr ffmpeg-python pillow prompt-toolkit rich
```

## Usage

### Basic Usage

```bash
python3 -m ocrnmr
```

Or if installed as a package:
```bash
ocrnmr
```

### Configuration

Edit `ocr_config.json` to configure:
- OCR parameters (scene detection, frame intervals, etc.)
- Hardware acceleration settings
- Test directory and episode titles
- Debug options

### Example Configuration

```json
{
  "ocr": {
    "use_scene_detection": false,
    "max_dimension": 800,
    "duration": 600,
    "frame_interval": 2.0,
    "match_threshold": 0.6,
    "hwaccel": "videotoolbox"
  },
  "test": {
    "test_directory": "~/rips/Dragon_Ball_Z_Kai/S5/",
    "episode_titles": [
      "Episode Title 1",
      "Episode Title 2"
    ]
  }
}
```

## Features

- **Pipelined FFMPEG extraction**: Extracts frames in background while OCR processes current file
- **Interactive display**: Real-time progress display with split panes for OCR and FFMPEG status
- **Rename preview**: Shows preview of all renamed files before applying changes
- **Hardware acceleration**: Supports VideoToolbox (macOS), VAAPI (Linux), D3D11VA/DXVA2 (Windows)
- **Scene detection**: Optional scene change detection to reduce frames processed

## Project Structure

```
ocrnmr/
├── ocrnmr/              # Package directory
│   ├── __init__.py
│   ├── __main__.py      # Entry point
│   ├── filename.py      # Filename utilities
│   └── ocr/             # OCR module
│       ├── __init__.py
│       ├── episode_matcher.py
│       ├── frame_extractor.py
│       ├── ocr_engine.py
│       └── title_card_scanner.py
├── ocrnmr_display.py    # Display module
├── test_ocr_single.py   # Main application script
├── ocr_config.json      # Configuration file
└── README.md
```

## License

Same as parent renamer project.

