"""
Event Newsworthiness Filter Module for GHTSP / SEDTWik.

Mathematical Formulation:
- Q(s): Wikipedia Keyphraseness score of string s (probability of s being an anchor link in Wikipedia).
- Segment newsworthiness mu(s):
    mu(s) = exp(Q(s))                         if s is a single word
    mu(s) = max_{l in subphrases(s)} (exp(Q(l)) - 1)  if s is a multi-word phrase
- Cluster newsworthiness mu(e):
    mu(e) = ( sum_{s in e_s} mu(s) / |e_s| ) * ( sum_{(u,v) in edges} sim(u,v) / |e_s| )
    where |e_s| is the number of segments in cluster e.
- Filter Rule:
    Let mu_max = max_{e in candidate_clusters} mu(e).
    Keep cluster e if and only if:
        mu_max / mu(e) < T (default T=4.0)

NOTE ON APPROXIMATION:
If no precomputed keyphraseness file (wikipedia_keyphraseness.json) is provided,
an analytical approximation is computed based on Wikipedia presence:
- Recognized multi-word Wikipedia entities: Q(s) approx 0.70 - 0.85
- Recognized single-word Wikipedia titles: Q(s) approx 0.40 - 0.55
- Out-of-vocabulary / common tokens: Q(s) approx 0.05
This follows the empirical anchor distribution documented in Milne & Witten (2008).
"""

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import config
from hashtag_tokenizer import load_wikipedia_vocabulary


_KEYPHRASENESS_CACHE: Optional[Dict[str, float]] = None


def load_keyphraseness_data(
    file_path: Optional[Path] = None
) -> Dict[str, float]:
    """Load precomputed keyphraseness file if present."""
    global _KEYPHRASENESS_CACHE
    if _KEYPHRASENESS_CACHE is not None:
        return _KEYPHRASENESS_CACHE

    target = file_path or config.KEYPHRASENESS_FILE
    if target and Path(target).exists():
        try:
            with open(target, "r", encoding="utf-8") as f:
                _KEYPHRASENESS_CACHE = json.load(f)
                return _KEYPHRASENESS_CACHE
        except Exception as e:
            print(f"Warning: Could not read {target}: {e}. Using approximation.")

    _KEYPHRASENESS_CACHE = {}
    return _KEYPHRASENESS_CACHE


def approximate_keyphraseness(s: str, wiki_vocab: Set[str]) -> float:
    """
    Analytical approximation of Wikipedia keyphraseness Q(s).
    Marked explicitly in code as required by project constraints.
    """
    s_lower = s.strip().lower()
    words = s_lower.split()

    if s_lower in wiki_vocab:
        if len(words) >= 2:
            # Multi-word Wikipedia concept (e.g. 'vice presidential debate', 'coming out day')
            return 0.80
        else:
            # Single-word Wikipedia entity (e.g. 'debate', 'obama', 'equality')
            return 0.45
    else:
        # Check if words are known titles
        known_count = sum(1 for w in words if w in wiki_vocab)
        if known_count > 0:
            return 0.20 * (known_count / len(words))
        return 0.05


def get_keyphraseness(
    s: str,
    keyphraseness_dict: Dict[str, float],
    wiki_vocab: Set[str]
) -> float:
    """Look up exact keyphraseness if available, otherwise return approximation."""
    s_lower = s.strip().lower()
    if s_lower in keyphraseness_dict:
        return keyphraseness_dict[s_lower]
    return approximate_keyphraseness(s_lower, wiki_vocab)


def get_subphrases(phrase: str) -> List[str]:
    """Generate all contiguous non-empty sub-phrases of a phrase."""
    words = phrase.split()
    subphrases = []
    n = len(words)
    for length in range(1, n + 1):
        for i in range(n - length + 1):
            subphrases.append(" ".join(words[i:i + length]))
    return subphrases


def compute_segment_newsworthiness(
    segment: str,
    keyphraseness_dict: Dict[str, float],
    wiki_vocab: Set[str]
) -> float:
    """
    Compute mu(s):
    - Single word: exp(Q(s))
    - Multi-word: max_{l in subphrases(s)} (exp(Q(l)) - 1)
    """
    words = segment.strip().split()
    if len(words) <= 1:
        q = get_keyphraseness(segment, keyphraseness_dict, wiki_vocab)
        return math.exp(q)
    else:
        subphrases = get_subphrases(segment)
        scores = []
        for l in subphrases:
            ql = get_keyphraseness(l, keyphraseness_dict, wiki_vocab)
            scores.append(math.exp(ql) - 1.0)
        return max(scores) if scores else 0.0


def compute_cluster_newsworthiness(
    cluster: Dict[str, Any],
    keyphraseness_dict: Dict[str, float],
    wiki_vocab: Set[str]
) -> float:
    """
    mu(e) = ( sum mu(s) / |e_s| ) * ( sum edge sims / |e_s| )
    """
    segments: List[str] = cluster["segments"]
    e_s_count = len(segments)
    if e_s_count == 0:
        return 0.0

    # 1. Sum mu(s) over all segments in cluster
    sum_mu_s = sum(
        compute_segment_newsworthiness(s, keyphraseness_dict, wiki_vocab)
        for s in segments
    )
    avg_mu_s = sum_mu_s / e_s_count

    # 2. Sum edge similarities
    edges: List[Tuple[str, str, float]] = cluster.get("edges", [])
    sum_edge_sims = sum(sim for _, _, sim in edges)
    # Average connectivity density
    # Guard against 0 edge similarity
    edge_term = max(0.01, sum_edge_sims / e_s_count)

    mu_e = avg_mu_s * edge_term
    return float(mu_e)


def filter_newsworthy_events(
    candidate_clusters: List[Dict[str, Any]],
    threshold_t: float = config.THRESHOLD_T,
    keyphraseness_dict: Optional[Dict[str, float]] = None,
    wiki_vocab: Optional[Set[str]] = None
) -> List[Dict[str, Any]]:
    """
    Evaluate cluster newsworthiness mu(e) across all candidate clusters.
    Keep only clusters satisfying: mu_max / mu(e) < threshold_t.
    """
    if not candidate_clusters:
        return []

    vocab = wiki_vocab if wiki_vocab is not None else load_wikipedia_vocabulary()
    q_dict = keyphraseness_dict if keyphraseness_dict is not None else load_keyphraseness_data()

    # Calculate mu(e) for each cluster
    for c in candidate_clusters:
        c["mu_score"] = compute_cluster_newsworthiness(c, q_dict, vocab)

    mu_max = max(c["mu_score"] for c in candidate_clusters)

    surviving = []
    for c in candidate_clusters:
        mu_e = c["mu_score"]
        # If mu_e is extremely small or zero, ratio is infinite (fails threshold)
        if mu_e > 1e-9:
            ratio = mu_max / mu_e
            c["ratio"] = ratio
            c["mu_max"] = mu_max
            if ratio < threshold_t:
                c["is_newsworthy"] = True
                surviving.append(c)
            else:
                c["is_newsworthy"] = False
        else:
            c["ratio"] = float("inf")
            c["is_newsworthy"] = False

    return surviving
