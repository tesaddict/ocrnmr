"""Episode name matching using fuzzy string matching."""

from typing import List, Optional, Dict, Tuple

try:
    from rapidfuzz import fuzz
except ImportError:
    fuzz = None


def match_episode(
    extracted_text: str,
    episode_names: List[str],
    threshold: float = 0.55
) -> Optional[str]:
    """
    Match extracted OCR text against a list of episode names using fuzzy matching.
    
    This function is thread-safe and can be used in parallel processing.
    
    Args:
        extracted_text: Text extracted from OCR
        episode_names: List of episode names to match against
        threshold: Minimum similarity ratio (0.0 to 1.0) to consider a match (default: 0.55)
    
    Returns:
        Best matching episode name if above threshold, else None
    
    Raises:
        RuntimeError: If rapidfuzz is not available
    """
    if fuzz is None:
        raise RuntimeError(
            "rapidfuzz is not installed. Please install it with: pip install rapidfuzz"
        )
    
    if not extracted_text or not episode_names:
        return None
    
    # Normalize extracted text - remove extra whitespace, normalize punctuation
    # Remove common punctuation that OCR might misinterpret (e.g., ";" vs "!", "," vs ";")
    normalized_extracted = extracted_text.lower()
    # Replace common punctuation variations with spaces for better matching
    # This handles cases like "Today;" vs "Today," vs "Today!"
    for punct in '!?;:.,':
        normalized_extracted = normalized_extracted.replace(punct, ' ')
    normalized_extracted = ' '.join(normalized_extracted.split())
    
    # Filter out very short words (common words like "the", "a", "an" that are noise)
    # But allow single-word episode titles like "Urgo" - they'll be validated by threshold and word overlap
    words = [w for w in normalized_extracted.split() if len(w) > 2]
    if len(words) == 0:  # Require at least one meaningful word (> 2 chars)
        return None
    
    # Note: We allow single-word matches (e.g., "URGO" matching "Urgo")
    # Noise filtering is handled by:
    # 1. Minimum word length (> 2 chars) filters out "a", "an", "the", etc.
    # 2. Fuzzy matching threshold filters low-confidence matches
    # 3. Word overlap check (30% minimum) ensures episode words appear in extracted text
    
    best_match = None
    best_score = 0.0
    
    # Try multiple matching strategies and use the best score
    for episode_name in episode_names:
        if not episode_name:
            continue
        
        # Normalize episode name similarly
        normalized_episode = episode_name.lower()
        # Replace punctuation with spaces for consistent matching
        for punct in '!?;:.,':
            normalized_episode = normalized_episode.replace(punct, ' ')
        normalized_episode = ' '.join(normalized_episode.split())
        
        # Special handling for single-word episode titles to avoid false positives
        # (e.g., "haven" matching "haven't" or "behaven")
        episode_words_list = normalized_episode.split()
        is_single_word = len(episode_words_list) == 1
        
        if is_single_word:
            episode_word = episode_words_list[0]
            extracted_words_list = normalized_extracted.split()
            word_found_as_standalone = False
            
            # Check if episode word appears as a standalone word
            for ext_word in extracted_words_list:
                # Exact match
                if ext_word == episode_word:
                    word_found_as_standalone = True
                    break
                # Fuzzy match for OCR errors (require high similarity)
                if len(ext_word) >= len(episode_word) * 0.8 and len(ext_word) <= len(episode_word) * 1.2:
                    word_ratio = fuzz.ratio(episode_word, ext_word) / 100.0
                    if word_ratio >= 0.85:  # Require 85% similarity
                        word_found_as_standalone = True
                        break
            
            # If single-word title doesn't appear as standalone, skip it
            # (avoids false positives like "haven" in "haven't")
            if not word_found_as_standalone:
                continue
        
        # Use 2-3 simple matching strategies
        # 1. Token set ratio - best for jumbled words (most robust for OCR)
        token_set_score = fuzz.token_set_ratio(normalized_extracted, normalized_episode) / 100.0
        
        # 2. Token sort ratio - good for reordered words
        token_sort_score = fuzz.token_sort_ratio(normalized_extracted, normalized_episode) / 100.0
        
        # 3. Full ratio - exact string matching (bonus for perfect matches)
        full_score = fuzz.ratio(normalized_extracted, normalized_episode) / 100.0
        
        # Use the best score from these strategies
        score = max(token_set_score, token_sort_score, full_score * 0.9)
        
        if score > best_score:
            best_score = score
            best_match = episode_name
    
    # Return best match if above threshold
    if best_score >= threshold:
        return best_match
    
    return None


def match_episode_with_scores(
    extracted_text: str,
    episode_names: List[str],
    threshold: float = 0.55
) -> Tuple[Optional[str], Dict[str, float]]:
    """
    Match extracted OCR text against a list of episode names and return detailed scores.
    
    This function is similar to match_episode but returns a tuple of (best_match, scores_dict)
    where scores_dict contains the match score for each episode name.
    
    Args:
        extracted_text: Text extracted from OCR
        episode_names: List of episode names to match against
        threshold: Minimum similarity ratio (0.0 to 1.0) to consider a match (default: 0.55)
    
    Returns:
        Tuple of (best_match_or_None, dict_of_scores) where scores_dict maps episode_name -> score
    """
    if fuzz is None:
        raise RuntimeError(
            "rapidfuzz is not installed. Please install it with: pip install rapidfuzz"
        )
    
    scores = {}
    
    if not extracted_text or not episode_names:
        return (None, scores)
    
    # Normalize extracted text - remove extra whitespace, normalize punctuation
    normalized_extracted = extracted_text.lower()
    for punct in '!?;:.,':
        normalized_extracted = normalized_extracted.replace(punct, ' ')
    normalized_extracted = ' '.join(normalized_extracted.split())
    
    # Filter out very short words
    words = [w for w in normalized_extracted.split() if len(w) > 2]
    if len(words) == 0:
        return (None, scores)
    
    best_match = None
    best_score = 0.0
    
    # Try multiple matching strategies and use the best score
    for episode_name in episode_names:
        if not episode_name:
            continue
        
        # Normalize episode name similarly
        normalized_episode = episode_name.lower()
        for punct in '!?;:.,':
            normalized_episode = normalized_episode.replace(punct, ' ')
        normalized_episode = ' '.join(normalized_episode.split())
        
        # Special handling for single-word episode titles (same as match_episode)
        episode_words_list = normalized_episode.split()
        is_single_word = len(episode_words_list) == 1
        
        if is_single_word:
            episode_word = episode_words_list[0]
            extracted_words_list = normalized_extracted.split()
            word_found_as_standalone = False
            
            for ext_word in extracted_words_list:
                if ext_word == episode_word:
                    word_found_as_standalone = True
                    break
                if len(ext_word) >= len(episode_word) * 0.8 and len(ext_word) <= len(episode_word) * 1.2:
                    word_ratio = fuzz.ratio(episode_word, ext_word) / 100.0
                    if word_ratio >= 0.85:
                        word_found_as_standalone = True
                        break
            
            if not word_found_as_standalone:
                scores[episode_name] = 0.0  # Set score to 0 instead of skipping
                continue
        
        # Use same 2-3 simple matching strategies as match_episode
        token_set_score = fuzz.token_set_ratio(normalized_extracted, normalized_episode) / 100.0
        token_sort_score = fuzz.token_sort_ratio(normalized_extracted, normalized_episode) / 100.0
        full_score = fuzz.ratio(normalized_extracted, normalized_episode) / 100.0
        
        score = max(token_set_score, token_sort_score, full_score * 0.9)
        scores[episode_name] = score
        
        if score > best_score:
            best_score = score
            best_match = episode_name
    
    # Return best match if above threshold
    if best_score >= threshold:
        return (best_match, scores)
    
    return (None, scores)

