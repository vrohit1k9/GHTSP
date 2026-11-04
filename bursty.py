"""
Bursty Segment Extraction Module for GHTSP / SEDTWik.

Mathematical Formulation:
- f_st: Observed frequency of segment s in window t.
- N_t: Total number of tweets in window t (or total segment occurrences).
- p_s: Prior probability of segment s estimated from the whole corpus.
- E = N_t * p_s (expected frequency).
- sigma = sqrt(N_t * p_s * (1 - p_s)) (binomial standard deviation).
- Condition: Keep candidate segments with f_st >= E.
- Bursty Probability:
    Pb(s) = sigmoid(10 * (f_st - (E + sigma)) / sigma)
- Bursty Weight:
    w_b = Pb * log(u) * log(src) * log(log(sfc))
    Where:
      u = distinct users tweeting s
      src = summed retweet count of tweets containing s
      sfc = summed follower count of users tweeting s
    Safe log guards are applied to guarantee positive real numbers without zero/negative log errors.
- Top-K selection: Keep top K = int(sqrt(N_t)) segments ranked by w_b.
"""

import math
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from segment_index import SegmentIndex


def sigmoid(x: float) -> float:
    """Numerically stable sigmoid function."""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)


def compute_corpus_priors(
    all_segmented_tweets: List[List[str]],
    min_prob: float = 1e-6
) -> Dict[str, float]:
    """
    Estimate prior probability p_s for all segments from the whole corpus.
    p_s = (count of segment s) / (total segment tokens in corpus)
    """
    counts: Dict[str, int] = {}
    total_tokens = 0

    for tweet_segs in all_segmented_tweets:
        for s in tweet_segs:
            counts[s] = counts.get(s, 0) + 1
            total_tokens += 1

    priors: Dict[str, float] = {}
    denom = max(1, total_tokens)

    for s, c in counts.items():
        priors[s] = max(min_prob, c / denom)

    return priors


def calculate_bursty_weight(
    pb: float,
    distinct_users: int,
    sum_retweets: int,
    sum_followers: int
) -> float:
    """
    Calculate w_b = Pb * log(u) * log(src) * log(log(sfc)) with safe guards.

    Assumptions & Guards:
    - log(u): guard with math.log(max(distinct_users, 1) + 1.0). Since u >= 1, log(u+1) >= log(2) > 0.
    - log(src): retweets may be 0. We guard with math.log(max(sum_retweets, 0) + 2.0) to ensure log >= log(2) > 0.
    - log(log(sfc)): follower counts can be small. We guard with math.log(math.log(max(sum_followers, 10) + math.e)).
      Since sfc >= 10, inner log >= log(10+e) > 2.5, outer log > 0.
    This guarantees that wb is always strictly positive when Pb > 0, and scales monotonically with engagement.
    """
    # Safe guards
    term_u = math.log(max(distinct_users, 1) + 1.0)
    term_src = math.log(max(sum_retweets, 0) + 2.0)
    inner_sfc = math.log(max(sum_followers, 10) + math.e)
    term_sfc = math.log(max(inner_sfc, 1.1))

    return float(pb * term_u * term_src * term_sfc)


def extract_bursty_segments(
    index: SegmentIndex,
    corpus_priors: Dict[str, float],
    k_bursty: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Extract top bursty segments from a 24-hour SegmentIndex.

    Args:
        index: Populated SegmentIndex for the window.
        corpus_priors: Global segment prior probabilities p_s.
        k_bursty: Number of bursty segments to keep. If None, computes sqrt(N_t).

    Returns:
        List of dictionaries containing segment burst metrics, sorted descending by w_b.
    """
    n_t = max(1, index.total_tweets)
    # Default K = int(sqrt(N_t)) as defined in architecture; ensure minimum candidate pool for small corpora
    k = k_bursty if k_bursty is not None else max(40, int(math.isqrt(n_t)))

    candidates: List[Dict[str, Any]] = []

    for s in index.all_segments():
        stats = index.get_segment_stats(s)
        f = stats["frequency"]
        p_s = corpus_priors.get(s, 1e-6)

        # Expected frequency and standard deviation under Binomial model
        # E = N_t * p_s
        expected_e = n_t * p_s
        # sigma = sqrt(N_t * p_s * (1 - p_s))
        variance = n_t * p_s * max(0.0, (1.0 - p_s))
        sigma = math.sqrt(variance)
        # Guard sigma against 0
        sigma = max(sigma, 1e-4)

        # Condition: Keep segments with observed frequency >= expected frequency
        if f >= expected_e:
            # Pb = sigmoid(10 * (f - (E + sigma)) / sigma)
            z = 10.0 * (f - (expected_e + sigma)) / sigma
            # Clip z to prevent extreme float overflow
            z = max(-50.0, min(50.0, z))
            pb = sigmoid(z)

            wb = calculate_bursty_weight(
                pb=pb,
                distinct_users=stats["distinct_users"],
                sum_retweets=stats["sum_retweets"],
                sum_followers=stats["sum_followers"]
            )

            candidates.append({
                "segment": s,
                "frequency": f,
                "expected": expected_e,
                "sigma": sigma,
                "Pb": pb,
                "wb": wb,
                "distinct_users": stats["distinct_users"],
                "sum_retweets": stats["sum_retweets"],
                "sum_followers": stats["sum_followers"],
                "tweet_ids": stats["tweet_ids"]
            })

    # Sort descending by bursty weight w_b
    candidates.sort(key=lambda x: x["wb"], reverse=True)

    # Keep top K
    return candidates[:k]
