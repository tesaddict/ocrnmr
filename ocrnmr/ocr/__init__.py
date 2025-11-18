"""OCR module for title card detection and episode matching."""

from .title_card_scanner import find_episode_by_title_card
from .frame_extractor import extract_frames, extract_frames_batch
from .ocr_engine import extract_text_from_frame, initialize_reader
from .episode_matcher import match_episode, match_episode_with_scores
from .processor import PipelinedOCRProcessor

__all__ = [
    'find_episode_by_title_card',
    'extract_frames',
    'extract_frames_batch',
    'extract_text_from_frame',
    'initialize_reader',
    'match_episode',
    'match_episode_with_scores',
    'PipelinedOCRProcessor',
]

