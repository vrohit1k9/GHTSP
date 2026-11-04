"""
Unit tests for Event Summarization and LexRank module.

Verifies:
- Near-duplicate tweet deduplication.
- LexRank power iteration algorithm.
- Central tweet identification on a star/hub-and-spoke toy graph.
"""

import numpy as np
import pytest
from summarizer import deduplicate_tweets, run_lexrank


def test_deduplicate_tweets():
    tweets = [
        {"id": "1", "clean_text": "breaking presidential debate in kentucky tonight", "raw_text": ""},
        {"id": "2", "clean_text": "breaking presidential debate in kentucky tonight", "raw_text": ""},  # duplicate
        {"id": "3", "clean_text": "different topic about sports soccer league", "raw_text": ""}
    ]
    deduped = deduplicate_tweets(tweets, sim_threshold=0.85)
    # Tweet 2 should be discarded as near-duplicate
    assert len(deduped) == 2
    ids = [t["id"] for t in deduped]
    assert "1" in ids
    assert "3" in ids


def test_lexrank_most_central_tweet():
    # Construct a star / hub-and-spoke graph:
    # Tweet 0 is the central hub, containing keywords connected to tweets 1, 2, and 3.
    # Tweet 4 is completely isolated / irrelevant.
    texts = [
        "presidential debate foreign policy middle east economy medicare", # Central hub (Tweet 0)
        "foreign policy and middle east strategy discussion",              # Spoke 1 (Tweet 1)
        "economy jobs and medicare debate issues",                         # Spoke 2 (Tweet 2)
        "presidential debate debate clash on policy",                      # Spoke 3 (Tweet 3)
        "unrelated recipe for homemade chocolate cake"                     # Outlier (Tweet 4)
    ]

    scores = run_lexrank(
        texts=texts,
        damping=0.85,
        sim_threshold=0.05
    )

    assert len(scores) == 5
    # The central hub (Tweet 0) must have the highest LexRank score
    assert int(np.argmax(scores)) == 0
    # Outlier must have lower score than central hub
    assert scores[0] > scores[4]
