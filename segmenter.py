"""
Tweet Segmentation Module for GHTSP / SEDTWik.

Responsibilities:
1. Segment tweet text (with hashtags, URLs, and mentions removed) using
   longest-match-first against the Wikipedia title lexicon.
2. Replace @mentions with real names using a configurable username->name map,
   and segment the resulting names.
3. Tokenize hashtags (via hashtag_tokenizer) and repeat each hashtag segment
   H times (default H=3) to confer the specified frequency weighting.
"""

import re
from typing import Dict, List, Optional, Set
import config
from hashtag_tokenizer import load_wikipedia_vocabulary, tokenize_hashtag


RE_WORD_TOKENS = re.compile(r"\b[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?\b")


def longest_match_segmentation(
    text: str,
    vocab: Set[str],
    max_ngram_length: int = 5
) -> List[str]:
    """
    Segment input text using longest-match-first against a vocabulary of Wikipedia titles.

    Args:
        text: Input text string.
        vocab: Set of lowercase Wikipedia titles.
        max_ngram_length: Maximum n-gram span to test for multi-word titles.

    Returns:
        List of segmented strings (lowercase).
    """
    tokens = RE_WORD_TOKENS.findall(text)
    if not tokens:
        return []

    tokens_lower = [t.lower() for t in tokens]
    n = len(tokens_lower)
    segments: List[str] = []
    i = 0

    while i < n:
        matched = False
        # Try longest n-gram down to 1
        max_j = min(n, i + max_ngram_length)
        for j in range(max_j, i, -1):
            ngram = " ".join(tokens_lower[i:j])
            if ngram in vocab:
                segments.append(ngram)
                i = j
                matched = True
                break

        if not matched:
            # Fall back to single token
            segments.append(tokens_lower[i])
            i += 1

    return segments


def process_mentions(
    mentions: List[str],
    vocab: Set[str],
    mention_map: Optional[Dict[str, str]] = None
) -> List[str]:
    """
    Replace @mentions with real names and segment them against Wikipedia vocabulary.
    """
    mapping = mention_map or config.MENTION_NAME_MAP
    segments = []

    for m in mentions:
        m_lower = m.lower()
        if m_lower in mapping:
            real_name = mapping[m_lower]
            # Segment the real name (e.g. 'Joe Biden' -> ['joe biden'])
            name_segs = longest_match_segmentation(real_name, vocab)
            segments.extend(name_segs)
        else:
            # Strip '@' and add raw username
            cleaned = m.lstrip("@").lower()
            if cleaned:
                segments.append(cleaned)

    return segments


def segment_tweet(
    clean_text: str,
    hashtags: List[str],
    mentions: List[str],
    vocab: Optional[Set[str]] = None,
    h_weight: int = config.HASHTAG_WEIGHT_H,
    use_hashtag_tokenizer: bool = config.USE_HASHTAG_TOKENIZER,
    mention_map: Optional[Dict[str, str]] = None
) -> List[str]:
    """
    Produce the complete list of segments for a single tweet:
    - Text segments via longest-match-first against Wikipedia titles.
    - Mention segments replaced by real names.
    - Hashtag segments repeated H times.

    Returns:
        Combined list of segments.
    """
    v = vocab if vocab is not None else load_wikipedia_vocabulary()

    # 1. Longest-match segmentation on the cleaned tweet body
    text_segments = longest_match_segmentation(clean_text, v)

    # 2. Process mentions with real-name substitution
    mention_segments = process_mentions(mentions, v, mention_map=mention_map)

    # 3. Process hashtags and repeat each segment H times
    hashtag_segments = []
    for ht in hashtags:
        toks = tokenize_hashtag(ht, vocab=v, use_tokenizer=use_hashtag_tokenizer)
        for tok in toks:
            # Repeat H times
            hashtag_segments.extend([tok] * h_weight)

    return text_segments + mention_segments + hashtag_segments
