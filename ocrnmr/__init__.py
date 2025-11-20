"""OCR Name Matcher - Standalone OCR application for renaming video files."""

__version__ = "1.0.0"

# Export OCR functionality (previously in ocr submodule)
from ocrnmr.title_card_scanner import find_episode_by_title_card
from ocrnmr.frame_extractor import extract_frames_batch
from ocrnmr.ocr_engine import extract_text_from_frame, initialize_reader
from ocrnmr.episode_matcher import match_episode, match_episode_with_scores
from ocrnmr.processor import process_video_files

__all__ = [
    'find_episode_by_title_card',
    'extract_frames_batch',
    'extract_text_from_frame',
    'initialize_reader',
    'match_episode',
    'match_episode_with_scores',
    'process_video_files',
]
