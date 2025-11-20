import pytest
from ocrnmr.episode_matcher import match_episode

def test_match_episode_exact():
    titles = ["Episode One", "Episode Two", "Episode Three"]
    assert match_episode("Episode One", titles) == "Episode One"

def test_match_episode_fuzzy():
    titles = ["Episode One", "Episode Two", "Episode Three"]
    # OCR often introduces noise
    assert match_episode("Episode On", titles) == "Episode One"
    assert match_episode("Episde Two", titles) == "Episode Two"

def test_match_episode_case_insensitive():
    titles = ["Episode One"]
    assert match_episode("episode one", titles) == "Episode One"
    assert match_episode("EPISODE ONE", titles) == "Episode One"

def test_match_episode_no_match():
    titles = ["Episode One", "Episode Two"]
    assert match_episode("Zebra", titles) is None
    # "Episode" matches "Episode One" via token_set_ratio which handles subsets.
    # Instead, test something that has some common letters but is wrong
    assert match_episode("Totally Different", titles) is None

def test_match_episode_punctuation():
    titles = ["Don't Stop Believin'"]
    # OCR might miss punctuation or see it as spaces
    assert match_episode("Dont Stop Believin", titles) == "Don't Stop Believin'"
    assert match_episode("Don t Stop Believin", titles) == "Don't Stop Believin'"
    assert match_episode("Don't Stop Believin'", titles) == "Don't Stop Believin'"

def test_match_episode_single_word():
    titles = ["Run", "Hide", "Seek"]
    # Single words need stricter matching in the implementation
    assert match_episode("Run", titles) == "Run"
    assert match_episode("Running", titles) is None  # Should likely not match if strictly single word
    
def test_match_episode_threshold():
    titles = ["Episode One"]
    # "Episd On" is actually quite close to "Episode One" (high ratio)
    # Use a stricter threshold to ensure rejection
    assert match_episode("Episd On", titles, threshold=0.95) is None
    # But might pass lower threshold
    assert match_episode("Episd On", titles, threshold=0.4) == "Episode One"

