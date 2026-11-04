"""
Unit tests for Bursty Segment Extraction module.

Verifies:
- Statistical burst calculation: E = N_t * p_s, sigma = sqrt(N_t * p_s * (1 - p_s))
- Condition f >= E
- Bursty probability Pb via sigmoid
- Bursty weight wb scaling with distinct users, retweets, and followers
- Ranking and top-K extraction on toy index
"""

import math
import pytest
from segment_index import SegmentIndex
from bursty import (
    calculate_bursty_weight,
    extract_bursty_segments,
    sigmoid
)


def test_sigmoid_properties():
    assert math.isclose(sigmoid(0.0), 0.5, rel_tol=1e-5)
    assert sigmoid(10.0) > 0.999
    assert sigmoid(-10.0) < 0.001


def test_bursty_weight_monotonicity():
    # Weight should increase monotonically with users, retweets, followers
    pb = 0.95
    w1 = calculate_bursty_weight(pb=pb, distinct_users=5, sum_retweets=10, sum_followers=500)
    w2 = calculate_bursty_weight(pb=pb, distinct_users=50, sum_retweets=100, sum_followers=5000)
    assert w2 > w1 > 0.0


def test_extract_bursty_toy_example():
    index = SegmentIndex(window_id="test_window")

    # Add 100 tweets
    # Segment 'bursty_event' appears in 40 tweets from 40 users with high engagement
    for i in range(40):
        index.add_tweet(
            tweet_id=f"t_burst_{i}",
            user=f"user_{i}",
            raw_text="breaking event news",
            clean_text="breaking event news",
            segments=["bursty_event", "news"],
            timestamp="2012-10-11 12:00:00",
            retweet_count=10,
            follower_count=200
        )

    # Segment 'normal_word' appears in 3 tweets
    for i in range(40, 100):
        segs = ["normal_word"] if i < 43 else ["other_word"]
        index.add_tweet(
            tweet_id=f"t_{i}",
            user=f"user_{i}",
            raw_text="regular chat",
            clean_text="regular chat",
            segments=segs,
            timestamp="2012-10-11 12:00:00",
            retweet_count=0,
            follower_count=50
        )

    # Corpus priors
    corpus_priors = {
        "bursty_event": 0.01,  # expected in 100 tweets: 1, observed: 40 -> huge burst!
        "normal_word": 0.05,   # expected in 100 tweets: 5, observed: 3 -> below E (filtered)
        "other_word": 0.60
    }

    candidates = extract_bursty_segments(index, corpus_priors, k_bursty=2)

    seg_names = [c["segment"] for c in candidates]
    assert "bursty_event" in seg_names
    # 'normal_word' should NOT be in candidates because f (3) < E (5)
    assert "normal_word" not in seg_names

    # The top candidate must be bursty_event with high Pb
    top_c = candidates[0]
    assert top_c["segment"] == "bursty_event"
    assert top_c["Pb"] > 0.90
    assert top_c["wb"] > 0.0
