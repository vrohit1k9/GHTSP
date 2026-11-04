"""
Unit tests for Tweet Segmentation module.

Verifies:
- Longest-match-first segmentation against Wikipedia titles.
- Mention replacement with real names.
- Hashtag segments repeated H times (default H=3).
"""

import pytest
from segmenter import segment_tweet, longest_match_segmentation
from hashtag_tokenizer import load_wikipedia_vocabulary


@pytest.fixture(scope="module")
def wiki_vocab():
    return load_wikipedia_vocabulary()


def test_longest_match_multiword(wiki_vocab):
    # 'vice presidential debate' should be matched as a single multi-word segment
    text = "The vice presidential debate was very intense"
    segments = longest_match_segmentation(text, wiki_vocab)
    assert "vice presidential debate" in segments
    # 'intense' is a single word
    assert "intense" in segments


def test_hashtag_weight_x3(wiki_vocab):
    clean_text = "economy and jobs"
    hashtags = ["#VpDebate"]
    mentions = []

    # With H=3, 'vp' and 'debate' should each appear 3 times
    segments = segment_tweet(
        clean_text=clean_text,
        hashtags=hashtags,
        mentions=mentions,
        vocab=wiki_vocab,
        h_weight=3,
        use_hashtag_tokenizer=True
    )

    assert segments.count("vp") == 3
    assert segments.count("debate") == 3
    assert segments.count("economy") == 1
    assert segments.count("jobs") == 1


def test_mention_replacement(wiki_vocab):
    clean_text = "great speech"
    hashtags = []
    mentions = ["@joebiden"]

    # @joebiden maps to 'Joe Biden', which is a Wikipedia entity 'joe biden'
    segments = segment_tweet(
        clean_text=clean_text,
        hashtags=hashtags,
        mentions=mentions,
        vocab=wiki_vocab,
        h_weight=3,
        use_hashtag_tokenizer=True
    )

    assert "joe biden" in segments
