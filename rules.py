"""
Rule-Based Segment Processing Module (NEW in GHTSP).

Configurable post-processing rules applied to tweet segments:
1. MERGE: Combines adjacent segments that together form a known Wikipedia title.
2. NORMALIZE: Lowercases, strips punctuation, and reduces repeated characters (e.g., 'cooolll' -> 'cool').
3. FILTER: Removes stop-word-only segments, single-character segments, and purely numeric segments.

Each rule can be toggled on or off individually or collectively via config.py.
When USE_RULES is False, this module acts as a pass-through (SEDTWik baseline).
"""

import re
import string
from typing import List, Optional, Set
import config
from hashtag_tokenizer import load_wikipedia_vocabulary


# Fallback English stop words in case NLTK corpus is not downloaded in offline environments
FALLBACK_STOP_WORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your",
    "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she",
    "her", "hers", "herself", "it", "its", "itself", "they", "them", "their",
    "theirs", "themselves", "what", "which", "who", "whom", "this", "that",
    "these", "those", "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an",
    "the", "and", "but", "if", "or", "because", "as", "until", "while", "of",
    "at", "by", "for", "with", "about", "against", "between", "into", "through",
    "during", "before", "after", "above", "below", "to", "from", "up", "down",
    "in", "out", "on", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "any",
    "both", "each", "few", "more", "most", "other", "some", "such", "no",
    "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s",
    "t", "can", "will", "just", "don", "should", "now"
}

_STOP_WORDS: Optional[Set[str]] = None


def get_stop_words() -> Set[str]:
    """Retrieve stop words using NLTK if available, otherwise fallback."""
    global _STOP_WORDS
    if _STOP_WORDS is not None:
        return _STOP_WORDS

    try:
        import nltk
        from nltk.corpus import stopwords
        try:
            words = set(stopwords.words("english"))
        except LookupError:
            nltk.download("stopwords", quiet=True)
            words = set(stopwords.words("english"))
        _STOP_WORDS = words
    except Exception:
        _STOP_WORDS = set(FALLBACK_STOP_WORDS)

    return _STOP_WORDS


# Regex to identify 3+ repeated characters
RE_REPEATED_CHARS = re.compile(r"(.)\1{2,}")


def reduce_repeated_characters(text: str, vocab: Optional[Set[str]] = None) -> str:
    """
    Reduce runs of 3+ repeated characters down to 1 or 2, checking vocab for valid words.
    Example: 'cooolll' -> 'cool'
             'yaaaay' -> 'yay'
    """
    matches = list(RE_REPEATED_CHARS.finditer(text))
    if not matches:
        return text

    v = vocab if vocab is not None else load_wikipedia_vocabulary()

    candidates = [""]
    last_idx = 0
    for m in matches:
        char = m.group(1)
        prefix = text[last_idx:m.start()]
        new_candidates = []
        for c in candidates:
            # Try 2 repeats (e.g. 'oo' or 'll')
            new_candidates.append(c + prefix + (char * 2))
            # Try 1 repeat (e.g. 'o' or 'l')
            new_candidates.append(c + prefix + char)
        candidates = new_candidates
        last_idx = m.end()

    suffix = text[last_idx:]
    final_candidates = [c + suffix for c in candidates]

    # Return first candidate recognized in vocabulary
    for cand in final_candidates:
        if cand in v:
            return cand

    # Default fallback: collapse to single character
    return RE_REPEATED_CHARS.sub(r"\1", text)


def normalize_segment(segment: str, vocab: Optional[Set[str]] = None) -> str:
    """
    NORMALIZE rule:
    - Lowercase
    - Strip leading/trailing punctuation and whitespace
    - Reduce 3 or more repeated characters (e.g. 'cooolll' -> 'cool', 'yaaaay' -> 'yay')
    """
    if not segment:
        return ""

    s = segment.lower().strip()
    # Strip punctuation from edges
    s = s.strip(string.punctuation)
    # Reduce repeated characters using vocab check
    s = reduce_repeated_characters(s, vocab=vocab)
    return s.strip()


def should_filter_segment(segment: str, stop_words: Set[str]) -> bool:
    """
    FILTER rule:
    Returns True if the segment should be DISCARDED:
    - Empty or whitespace
    - Single-character (e.g. 'a', 's')
    - Purely numeric (e.g. '123', '2012')
    - Stop-word-only (every word in segment is a stop word, e.g. 'in the', 'and')
    """
    if not segment:
        return True

    s = segment.strip()
    if len(s) <= 1:
        return True

    if s.isdigit():
        return True

    words = s.split()
    if words and all(w in stop_words for w in words):
        return True

    return False


def merge_adjacent_segments(segments: List[str], vocab: Set[str]) -> List[str]:
    """
    MERGE rule:
    Combines adjacent segments if their concatenation forms a known Wikipedia title.
    Example: ['presidential', 'debate'] -> ['presidential debate']
             ['united', 'states'] -> ['united states']
    """
    if len(segments) <= 1:
        return segments

    merged: List[str] = []
    i = 0
    n = len(segments)

    while i < n:
        if i < n - 1:
            candidate = f"{segments[i]} {segments[i+1]}"
            if candidate in vocab:
                merged.append(candidate)
                i += 2
                continue
        merged.append(segments[i])
        i += 1

    return merged


def apply_rules(
    segments: List[str],
    vocab: Optional[Set[str]] = None,
    use_rules: bool = config.USE_RULES,
    rule_merge: bool = config.RULE_MERGE_ADJACENT,
    rule_normalize: bool = config.RULE_NORMALIZE,
    rule_filter: bool = config.RULE_FILTER
) -> List[str]:
    """
    Apply configured rule-based processing to segments:
    1. Normalize
    2. Merge adjacent Wikipedia entities
    3. Filter out noise (stop words, single chars, pure numbers)

    If use_rules is False, returns segments unmodified.
    """
    if not use_rules or not segments:
        return segments

    v = vocab if vocab is not None else load_wikipedia_vocabulary()
    stop_words = get_stop_words()

    processed = segments

    # Step 1: Normalize
    if rule_normalize:
        normalized = []
        for s in processed:
            norm_s = normalize_segment(s, vocab=v)
            if norm_s:
                normalized.append(norm_s)
        processed = normalized

    # Step 2: Merge adjacent segments forming Wikipedia titles
    if rule_merge:
        processed = merge_adjacent_segments(processed, v)

    # Step 3: Filter invalid/noise segments
    if rule_filter:
        filtered = []
        for s in processed:
            if not should_filter_segment(s, stop_words):
                filtered.append(s)
        processed = filtered

    return processed
