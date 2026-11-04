"""
Unit tests for Rule-Based Segment Processing module.

Verifies:
- MERGE: adjacent segments forming Wikipedia title are combined.
- NORMALIZE: lowercase, strip punctuation, strip repeated characters (e.g. 'cooolll' -> 'cool').
- FILTER: remove stop-word-only segments, single-character, and purely numeric segments.
- Switchable behavior (use_rules=False).
"""

import pytest
from rules import (
    apply_rules,
    normalize_segment,
    should_filter_segment,
    merge_adjacent_segments,
    get_stop_words
)
from hashtag_tokenizer import load_wikipedia_vocabulary


@pytest.fixture(scope="module")
def wiki_vocab():
    return load_wikipedia_vocabulary()


@pytest.fixture(scope="module")
def stop_words():
    return get_stop_words()


def test_normalize_repeated_chars_and_punct(wiki_vocab):
    # 'cooolll' -> 'cool'
    assert normalize_segment("cooolll", vocab=wiki_vocab) == "cool"
    # 'debate!!!' -> 'debate'
    assert normalize_segment("debate!!!", vocab=wiki_vocab) == "debate"
    # '...GREAT...' -> 'great'
    assert normalize_segment("...GREAT...", vocab=wiki_vocab) == "great"


def test_filter_stop_words_numbers_and_single_char(stop_words):
    # Single character
    assert should_filter_segment("a", stop_words) is True
    assert should_filter_segment("s", stop_words) is True
    # Pure numeric
    assert should_filter_segment("123", stop_words) is True
    assert should_filter_segment("2012", stop_words) is True
    # Stop-word-only
    assert should_filter_segment("in the", stop_words) is True
    assert should_filter_segment("and", stop_words) is True
    # Valid informative segment
    assert should_filter_segment("presidential debate", stop_words) is False
    assert should_filter_segment("biden", stop_words) is False


def test_merge_adjacent_segments(wiki_vocab):
    # 'presidential' + 'debate' form 'presidential debate' which is a known title
    raw_segments = ["presidential", "debate", "tonight"]
    merged = merge_adjacent_segments(raw_segments, wiki_vocab)
    assert merged == ["presidential debate", "tonight"]

    # 'united' + 'states' form 'united states'
    raw_segments2 = ["the", "united", "states", "economy"]
    merged2 = merge_adjacent_segments(raw_segments2, wiki_vocab)
    assert merged2 == ["the", "united states", "economy"]


def test_apply_rules_pipeline(wiki_vocab):
    # Combine normalize, merge, and filter
    input_segments = ["the", "cooolll", "presidential", "debate!!!", "2012", "x"]
    processed = apply_rules(input_segments, vocab=wiki_vocab, use_rules=True)

    # 'the' -> filtered (stop word)
    # 'cooolll' -> 'cool'
    # 'presidential' + 'debate' -> 'presidential debate'
    # '2012' -> filtered (numeric)
    # 'x' -> filtered (single char)
    assert "presidential debate" in processed
    assert "cool" in processed
    assert "the" not in processed
    assert "2012" not in processed
    assert "x" not in processed


def test_use_rules_disabled(wiki_vocab):
    # When disabled, segments pass through untouched
    input_segments = ["the", "2012", "x"]
    processed = apply_rules(input_segments, vocab=wiki_vocab, use_rules=False)
    assert processed == ["the", "2012", "x"]
