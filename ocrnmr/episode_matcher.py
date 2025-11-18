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
        
        # Calculate multiple similarity scores
        # 1. Full ratio - exact string matching
        full_score = fuzz.ratio(normalized_extracted, normalized_episode) / 100.0
        
        # 2. Token sort ratio - order-independent matching (good for OCR errors)
        token_sort_score = fuzz.token_sort_ratio(normalized_extracted, normalized_episode) / 100.0
        
        # 3. Token set ratio - best for matching when words are jumbled
        token_set_score = fuzz.token_set_ratio(normalized_extracted, normalized_episode) / 100.0
        
        # 4. Partial ratio - but only use if extracted text is substantial
        # Only use partial_ratio if we have a reasonable amount of text
        partial_score = 0.0
        if len(normalized_extracted) >= 20:  # Only use partial for longer extracts
            partial_score = fuzz.partial_ratio(normalized_extracted, normalized_episode) / 100.0
            # Penalize partial matches - they're less reliable
            partial_score = partial_score * 0.75
        
        # Check if this is a single-word episode title
        episode_words_list = normalized_episode.split()
        is_single_word = len(episode_words_list) == 1
        
        # For single-word titles, require stricter matching to avoid false positives
        # (e.g., "haven" matching "haven't" or "behaven")
        if is_single_word:
            episode_word = episode_words_list[0]
            # Check if the episode word appears as a complete word (with word boundaries)
            # Split extracted text into words and check for exact or fuzzy match
            extracted_words_list = normalized_extracted.split()
            word_found_as_standalone = False
            
            for ext_word in extracted_words_list:
                # Check exact match first
                if ext_word == episode_word:
                    word_found_as_standalone = True
                    break
                # Check fuzzy match (for OCR errors) but require high similarity
                if len(ext_word) >= len(episode_word) * 0.8 and len(ext_word) <= len(episode_word) * 1.2:
                    word_ratio = fuzz.ratio(episode_word, ext_word) / 100.0
                    if word_ratio >= 0.85:  # Require 85% similarity for single-word matches
                        word_found_as_standalone = True
                        break
            
            # If single-word title doesn't appear as standalone word, heavily penalize
            if not word_found_as_standalone:
                # Still allow partial matches but with heavy penalty
                # This handles cases where OCR might have added/removed characters
                partial_word_score = 0.0
                for ext_word in extracted_words_list:
                    if episode_word in ext_word or ext_word in episode_word:
                        # Check if it's a reasonable substring match (not just "haven" in "haven't")
                        if abs(len(ext_word) - len(episode_word)) <= 2:
                            partial_word_score = max(partial_word_score, fuzz.ratio(episode_word, ext_word) / 100.0)
                
                if partial_word_score < 0.9:  # Require very high similarity for substring matches
                    # This is likely a false positive (e.g., "haven" in "haven't")
                    # Skip this episode entirely
                    continue
        
        # Use weighted combination - prefer token_set_ratio and token_sort_ratio
        # as they're more robust to OCR errors, but require substantial overlap
        # For Tesseract which makes more character-level errors, be more lenient
        score = max(
            token_set_score * 1.0,      # Best for jumbled words
            token_sort_score * 0.95,    # Good for reordered words
            partial_score,              # Good for partial matches (if substantial)
            full_score * 0.9            # Exact match bonus
        )
        
        # Additional check: require that at least 30% of episode words appear in extracted text
        # But use fuzzy word matching to handle OCR character errors (e.g., "moday" vs "today")
        episode_words = set(w for w in normalized_episode.split() if len(w) > 2)
        extracted_words = set(w for w in normalized_extracted.split() if len(w) > 2)
        
        if episode_words:
            # Use fuzzy word matching to handle OCR errors
            matched_words = 0
            for ep_word in episode_words:
                # Check if word appears exactly
                if ep_word in extracted_words:
                    matched_words += 1
                else:
                    # Try fuzzy matching for OCR errors (e.g., "moday" vs "today")
                    best_fuzzy_score = 0
                    for ext_word in extracted_words:
                        # Use partial_ratio for character-level OCR errors
                        fuzzy_score = fuzz.partial_ratio(ep_word, ext_word) / 100.0
                        # Also check if words are similar length (OCR errors usually preserve length)
                        if abs(len(ep_word) - len(ext_word)) <= 2:
                            fuzzy_score = max(fuzzy_score, fuzz.ratio(ep_word, ext_word) / 100.0)
                        best_fuzzy_score = max(best_fuzzy_score, fuzzy_score)
                    
                    # Consider it a match if fuzzy score > 0.7 (handles OCR errors like "moday"/"today")
                    if best_fuzzy_score > 0.7:
                        matched_words += best_fuzzy_score  # Partial credit for fuzzy matches
            
            word_overlap = matched_words / len(episode_words)
            # Require at least 30% word overlap, or scale down the score
            if word_overlap < 0.3:
                score = score * 0.5  # Heavily penalize low word overlap
            elif word_overlap < 0.5:
                # Give partial boost for moderate word overlap
                score = score * (0.7 + word_overlap * 0.6)  # Scale between 0.7 and 1.0
        
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
        
        # Calculate multiple similarity scores
        full_score = fuzz.ratio(normalized_extracted, normalized_episode) / 100.0
        token_sort_score = fuzz.token_sort_ratio(normalized_extracted, normalized_episode) / 100.0
        token_set_score = fuzz.token_set_ratio(normalized_extracted, normalized_episode) / 100.0
        
        partial_score = 0.0
        if len(normalized_extracted) >= 20:
            partial_score = fuzz.partial_ratio(normalized_extracted, normalized_episode) / 100.0
            partial_score = partial_score * 0.75
        
        # Check if this is a single-word episode title
        episode_words_list = normalized_episode.split()
        is_single_word = len(episode_words_list) == 1
        
        # For single-word titles, require stricter matching to avoid false positives
        # (e.g., "haven" matching "haven't" or "behaven")
        if is_single_word:
            episode_word = episode_words_list[0]
            # Check if the episode word appears as a complete word (with word boundaries)
            # Split extracted text into words and check for exact or fuzzy match
            extracted_words_list = normalized_extracted.split()
            word_found_as_standalone = False
            
            for ext_word in extracted_words_list:
                # Check exact match first
                if ext_word == episode_word:
                    word_found_as_standalone = True
                    break
                # Check fuzzy match (for OCR errors) but require high similarity
                if len(ext_word) >= len(episode_word) * 0.8 and len(ext_word) <= len(episode_word) * 1.2:
                    word_ratio = fuzz.ratio(episode_word, ext_word) / 100.0
                    if word_ratio >= 0.85:  # Require 85% similarity for single-word matches
                        word_found_as_standalone = True
                        break
            
            # If single-word title doesn't appear as standalone word, heavily penalize
            if not word_found_as_standalone:
                # Still allow partial matches but with heavy penalty
                # This handles cases where OCR might have added/removed characters
                partial_word_score = 0.0
                for ext_word in extracted_words_list:
                    if episode_word in ext_word or ext_word in episode_word:
                        # Check if it's a reasonable substring match (not just "haven" in "haven't")
                        if abs(len(ext_word) - len(episode_word)) <= 2:
                            partial_word_score = max(partial_word_score, fuzz.ratio(episode_word, ext_word) / 100.0)
                
                if partial_word_score < 0.9:  # Require very high similarity for substring matches
                    # This is likely a false positive (e.g., "haven" in "haven't")
                    # Skip this episode entirely
                    scores[episode_name] = 0.0  # Set score to 0 instead of skipping
                    continue
        
        score = max(
            token_set_score * 1.0,
            token_sort_score * 0.95,
            partial_score,
            full_score * 0.9
        )
        
        # Word overlap check
        episode_words = set(w for w in normalized_episode.split() if len(w) > 2)
        extracted_words = set(w for w in normalized_extracted.split() if len(w) > 2)
        
        if episode_words:
            matched_words = 0
            for ep_word in episode_words:
                if ep_word in extracted_words:
                    matched_words += 1
                else:
                    best_fuzzy_score = 0
                    for ext_word in extracted_words:
                        fuzzy_score = fuzz.partial_ratio(ep_word, ext_word) / 100.0
                        if abs(len(ep_word) - len(ext_word)) <= 2:
                            fuzzy_score = max(fuzzy_score, fuzz.ratio(ep_word, ext_word) / 100.0)
                        best_fuzzy_score = max(best_fuzzy_score, fuzzy_score)
                    
                    if best_fuzzy_score > 0.7:
                        matched_words += best_fuzzy_score
            
            word_overlap = matched_words / len(episode_words)
            if word_overlap < 0.3:
                score = score * 0.5
            elif word_overlap < 0.5:
                score = score * (0.7 + word_overlap * 0.6)
        
        scores[episode_name] = score
        
        if score > best_score:
            best_score = score
            best_match = episode_name
    
    # Return best match if above threshold
    if best_score >= threshold:
        return (best_match, scores)
    
    return (None, scores)

