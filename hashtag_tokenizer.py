"""
Hashtag Tokenizer Module (NEW in GHTSP).

Responsibilities:
1. Split camelCase hashtags when capital letters exist.
2. Handle embedded digits (e.g., '#Oct12Debate' -> ['oct', '12', 'debate']).
3. When lowercase (e.g., '#nationalcomingoutday'), use dynamic programming over
   the Wikipedia-title vocabulary, preferring fewer and longer known words.
4. Fall back to a single unigram if no valid split is found.
5. Provide switchable behavior to support baseline SEDTWik mode.
"""

import gzip
import re
from pathlib import Path
from typing import List, Optional, Set, Tuple
import config


# Global cached vocabulary
_WIKI_VOCAB: Optional[Set[str]] = None


def load_wikipedia_vocabulary(
    gz_path: Optional[Path] = None,
    fallback_path: Optional[Path] = None
) -> Set[str]:
    """
    Load Wikipedia titles from enwiki-latest-all-titles-in-ns0.gz if present;
    otherwise use the bundled fallback file. Lowercase all titles and return as a set.
    """
    global _WIKI_VOCAB
    if _WIKI_VOCAB is not None:
        return _WIKI_VOCAB

    gz_file = gz_path or config.WIKI_GZ_FILE
    fb_file = fallback_path or config.WIKI_TITLES_FILE

    vocab = set()

    if gz_file and Path(gz_file).exists():
        print(f"Loading Wikipedia titles from {gz_file}...")
        try:
            with gzip.open(gz_file, "rt", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    t = line.strip().lower()
                    if t:
                        vocab.add(t)
                        # Also index individual words in multi-word titles
                        for part in t.split():
                            vocab.add(part)
        except Exception as e:
            print(f"Warning: Failed to load {gz_file}: {e}. Falling back to text file.")

    if not vocab and fb_file and Path(fb_file).exists():
        with open(fb_file, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                t = line.strip().lower()
                if t:
                    vocab.add(t)
                    for part in t.split():
                        vocab.add(part)

    # Ensure essential digits and base tokens are recognized
    for d in range(100):
        vocab.add(str(d))

    _WIKI_VOCAB = vocab
    return _WIKI_VOCAB


def set_custom_vocabulary(vocab: Set[str]) -> None:
    """Helper for testing or custom lexicon overrides."""
    global _WIKI_VOCAB
    _WIKI_VOCAB = {w.lower() for w in vocab}


# Regular expressions for camelCase and digit boundary splitting
RE_HASHTAG_PREFIX = re.compile(r"^#+")
# Boundaries: lowercase followed by uppercase, or acronyms followed by lowercase
RE_CAMEL_CASE = re.compile(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
# Boundaries between letters and digits
RE_DIGIT_SPLIT = re.compile(r"(?<=[a-zA-Z])(?=\d)|(?<=\d)(?=[a-zA-Z])")


def split_camel_case_and_digits(text: str) -> List[str]:
    """
    Split a string with camelCase and/or digit transitions into lowercase tokens.
    Example: 'Oct12Debate' -> ['oct', '12', 'debate']
             'BreakingNews' -> ['breaking', 'news']
             'VpDebate' -> ['vp', 'debate']
    """
    # 1. Split across letter/digit boundaries
    chunks = RE_DIGIT_SPLIT.split(text)
    tokens = []
    for chunk in chunks:
        # 2. Split across camelCase boundaries
        sub_chunks = RE_CAMEL_CASE.split(chunk)
        for sc in sub_chunks:
            cleaned = sc.strip().lower()
            if cleaned:
                tokens.append(cleaned)
    return tokens


def dp_word_segmentation(text: str, vocab: Set[str]) -> Optional[List[str]]:
    """
    Dynamic programming word segmentation over vocabulary.
    Prefers fewer and longer known words by maximizing sum of squared lengths.

    dp[i] = (max_score, token_count, split_index) for text[0:i]
    """
    n = len(text)
    # dp[i] stores (best_score, min_tokens, split_point)
    # best_score = sum(len(word)^2), min_tokens = count of words
    dp: List[Optional[Tuple[float, int, int]]] = [None] * (n + 1)
    dp[0] = (0.0, 0, 0)

    # Max word length to search
    max_len = min(n, 35)

    for i in range(1, n + 1):
        best_tuple = None
        for length in range(1, min(i, max_len) + 1):
            j = i - length
            if dp[j] is None:
                continue

            word = text[j:i]
            # Check if word is known in Wikipedia vocabulary or is purely numeric
            if word in vocab or word.isdigit():
                prev_score, prev_count, _ = dp[j]
                # Reward longer words quadratically: e.g. 8^2 = 64
                current_score = prev_score + (len(word) ** 2)
                current_count = prev_count + 1

                # We prefer:
                # 1. Higher score (heavily favors longer words)
                # 2. Fewer tokens
                candidate = (current_score, -current_count, j)
                if best_tuple is None or candidate > best_tuple:
                    best_tuple = candidate

        if best_tuple is not None:
            score, neg_count, split_idx = best_tuple
            dp[i] = (score, -neg_count, split_idx)

    # If the entire string could not be segmented into known words
    if dp[n] is None:
        return None

    # Reconstruct words from backpointers
    tokens = []
    curr = n
    while curr > 0:
        _, _, prev_idx = dp[curr]
        tokens.append(text[prev_idx:curr])
        curr = prev_idx

    tokens.reverse()
    return tokens


def tokenize_hashtag(
    hashtag: str,
    vocab: Optional[Set[str]] = None,
    use_tokenizer: bool = config.USE_HASHTAG_TOKENIZER
) -> List[str]:
    """
    Tokenize a single hashtag according to GHTSP requirements.

    Args:
        hashtag: Raw hashtag string (e.g., '#BreakingNews', '#Oct12Debate', '#nationalcomingoutday').
        vocab: Set of lowercase Wikipedia titles. If None, loaded automatically.
        use_tokenizer: If False, runs in SEDTWik baseline mode (unigram strip).

    Returns:
        List of segmented lowercase token strings.
    """
    raw = RE_HASHTAG_PREFIX.sub("", hashtag).strip()
    if not raw:
        return []

    # If tokenizer is disabled (SEDTWik baseline)
    if not use_tokenizer:
        return [raw.lower()]

    # Case 1: Capitals or digits exist -> split camelCase and digits
    has_capitals = any(c.isupper() for c in raw)
    has_digits = any(c.isdigit() for c in raw)

    if has_capitals or has_digits:
        tokens = split_camel_case_and_digits(raw)
        if tokens:
            return tokens

    # Case 2: All lowercase -> dynamic programming over Wikipedia vocabulary
    v = vocab if vocab is not None else load_wikipedia_vocabulary()
    lower_raw = raw.lower()

    split_result = dp_word_segmentation(lower_raw, v)
    if split_result and len(split_result) > 0:
        return split_result

    # Fall back to single unigram if no split is found
    return [lower_raw]


def tokenize_hashtags_list(
    hashtags: List[str],
    vocab: Optional[Set[str]] = None,
    use_tokenizer: bool = config.USE_HASHTAG_TOKENIZER
) -> List[str]:
    """Tokenize a list of hashtags and flatten the resulting tokens."""
    tokens = []
    for ht in hashtags:
        tokens.extend(tokenize_hashtag(ht, vocab=vocab, use_tokenizer=use_tokenizer))
    return tokens
