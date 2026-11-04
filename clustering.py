"""
Bursty Segment Clustering Module for GHTSP / SEDTWik.

Algorithm:
1. Split the 24-hour window into M sub-windows (default M=12, each 2 hours).
2. For each bursty segment s:
   - Compute sub-window frequency f_t(s, m).
   - Compute temporal sub-window weight: w_t(s, m) = f_t(s, m) / f_st.
   - Aggregate tweet texts in sub-window m containing segment s: T(s, m).
3. Compute pairwise segment similarity:
   sim(a, b) = sum_{m=1}^M [ w_t(a, m) * w_t(b, m) * tfidf_cosine(T(a, m), T(b, m)) ]
4. Build a Jarvis-Patrick graph:
   - An undirected edge (a, b) exists if and only if a is in b's k nearest neighbors (k-NN)
     AND b is in a's k nearest neighbors, with sim(a, b) > 0.
5. Extract connected components as candidate clusters; drop isolated nodes (size < 2).
"""

from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Set, Tuple
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import config
from segment_index import SegmentIndex


def partition_tweets_into_subwindows(
    index: SegmentIndex,
    subwindows_m: int = config.SUBWINDOWS_M
) -> List[List[Dict[str, Any]]]:
    """
    Split the tweets in index into M equal temporal sub-windows based on timestamp.
    """
    tweets = list(index.tweets.values())
    if not tweets:
        return [[] for _ in range(subwindows_m)]

    # Parse timestamps
    timestamps = [pd.to_datetime(t["timestamp"]) for t in tweets]
    min_ts = min(timestamps)
    max_ts = max(timestamps)

    total_duration = (max_ts - min_ts).total_seconds()
    if total_duration <= 0:
        # Single point in time
        subwindows = [[] for _ in range(subwindows_m)]
        subwindows[0] = tweets
        return subwindows

    subwindow_duration = max(1.0, total_duration / subwindows_m)
    subwindows: List[List[Dict[str, Any]]] = [[] for _ in range(subwindows_m)]

    for t, ts in zip(tweets, timestamps):
        delta = (ts - min_ts).total_seconds()
        idx = min(subwindows_m - 1, int(delta // subwindow_duration))
        subwindows[idx].append(t)

    return subwindows


def compute_subwindow_representations(
    bursty_segments: List[str],
    subwindows: List[List[Dict[str, Any]]],
    subwindows_m: int,
    index: SegmentIndex
) -> Tuple[Dict[str, List[float]], Dict[str, List[str]]]:
    """
    Compute for each segment s:
    1. w_t(s, m) = f_t(s, m) / f_st across m in 0..M-1
    2. T(s, m) = aggregated text of tweets in subwindow m containing segment s
    """
    seg_sub_counts: Dict[str, List[int]] = {s: [0] * subwindows_m for s in bursty_segments}
    seg_sub_texts: Dict[str, List[List[str]]] = {s: [[] for _ in range(subwindows_m)] for s in bursty_segments}

    for m_idx, sub_tweets in enumerate(subwindows):
        for tw in sub_tweets:
            tw_text = tw["clean_text"] or tw["raw_text"]
            tw_segs = tw["segments"]
            for s in bursty_segments:
                if s in tw_segs:
                    # Count occurrences in this tweet
                    cnt = tw_segs.count(s)
                    seg_sub_counts[s][m_idx] += cnt
                    seg_sub_texts[s][m_idx].append(tw_text)

    # Calculate normalized weights w_t(s, m) = f_t(s, m) / f_st
    weights: Dict[str, List[float]] = {}
    aggregated_texts: Dict[str, List[str]] = {}

    for s in bursty_segments:
        stats = index.get_segment_stats(s)
        f_st = max(1, stats["frequency"])
        weights[s] = [cnt / f_st for cnt in seg_sub_counts[s]]
        aggregated_texts[s] = [" ".join(texts) for texts in seg_sub_texts[s]]

    return weights, aggregated_texts


def compute_pairwise_similarities(
    bursty_segments: List[str],
    weights: Dict[str, List[float]],
    aggregated_texts: Dict[str, List[str]],
    subwindows_m: int
) -> Dict[Tuple[str, str], float]:
    """
    sim(a, b) = sum_{m=1}^M [ w_t(a, m) * w_t(b, m) * tfidf_cosine(T(a, m), T(b, m)) ]
    """
    similarities: Dict[Tuple[str, str], float] = defaultdict(float)
    n = len(bursty_segments)

    for m in range(subwindows_m):
        # Gather all non-empty texts for subwindow m
        active_segs = [s for s in bursty_segments if weights[s][m] > 0 and aggregated_texts[s][m].strip()]
        if len(active_segs) < 2:
            continue

        corpus_m = [aggregated_texts[s][m] for s in active_segs]
        try:
            vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
            tfidf_mat = vectorizer.fit_transform(corpus_m)
            cos_sim = cosine_similarity(tfidf_mat)

            for i in range(len(active_segs)):
                for j in range(i + 1, len(active_segs)):
                    seg_a = active_segs[i]
                    seg_b = active_segs[j]
                    sim_val = float(cos_sim[i, j])
                    if sim_val > 0:
                        term = weights[seg_a][m] * weights[seg_b][m] * sim_val
                        similarities[(seg_a, seg_b)] += term
                        similarities[(seg_b, seg_a)] += term
        except Exception:
            continue

    return similarities


def build_jarvis_patrick_graph(
    bursty_segments: List[str],
    similarities: Dict[Tuple[str, str], float],
    k: int = config.KNN_K
) -> nx.Graph:
    """
    Build Jarvis-Patrick graph:
    An edge (a, b) exists if a in kNN(b) AND b in kNN(a) with sim(a, b) > 0.
    """
    g = nx.Graph()
    for s in bursty_segments:
        g.add_node(s)

    if len(bursty_segments) < 2:
        return g

    # Find k nearest neighbors for each segment
    knn_dict: Dict[str, Set[str]] = {}
    for a in bursty_segments:
        # Candidates for a
        candidates = []
        for b in bursty_segments:
            if a == b:
                continue
            sim = similarities.get((a, b), 0.0)
            if sim > 0:
                candidates.append((b, sim))
        # Sort descending by similarity
        candidates.sort(key=lambda x: x[1], reverse=True)
        top_k = {item[0] for item in candidates[:k]}
        knn_dict[a] = top_k

    # Mutual kNN condition
    for i in range(len(bursty_segments)):
        for j in range(i + 1, len(bursty_segments)):
            a = bursty_segments[i]
            b = bursty_segments[j]
            if b in knn_dict.get(a, set()) and a in knn_dict.get(b, set()):
                sim_weight = similarities.get((a, b), 0.0)
                if sim_weight > 0:
                    g.add_edge(a, b, weight=sim_weight)

    return g


def cluster_bursty_segments(
    bursty_candidates: List[Dict[str, Any]],
    index: SegmentIndex,
    subwindows_m: int = config.SUBWINDOWS_M,
    knn_k: int = config.KNN_K
) -> Tuple[List[Dict[str, Any]], nx.Graph]:
    """
    Cluster bursty segments into candidate events using temporal subwindow TF-IDF
    and Jarvis-Patrick graph connected components.

    Returns:
        candidate_clusters: list of clusters with size >= 2
        graph: full networkx Graph
    """
    bursty_segs = [b["segment"] for b in bursty_candidates]
    if len(bursty_segs) < 2:
        empty_g = nx.Graph()
        for s in bursty_segs:
            empty_g.add_node(s)
        return [], empty_g

    # 1. Partition tweets into M sub-windows
    subwindows = partition_tweets_into_subwindows(index, subwindows_m=subwindows_m)

    # 2. Compute weights and texts per subwindow
    weights, agg_texts = compute_subwindow_representations(
        bursty_segments=bursty_segs,
        subwindows=subwindows,
        subwindows_m=subwindows_m,
        index=index
    )

    # 3. Compute pairwise similarities
    similarities = compute_pairwise_similarities(
        bursty_segments=bursty_segs,
        weights=weights,
        aggregated_texts=agg_texts,
        subwindows_m=subwindows_m
    )

    # 4. Build Jarvis-Patrick graph
    graph = build_jarvis_patrick_graph(
        bursty_segments=bursty_segs,
        similarities=similarities,
        k=knn_k
    )

    # 5. Extract connected components; drop isolated nodes (size < 2)
    clusters = []
    cluster_idx = 1

    # Map segment to bursty info
    bursty_map = {b["segment"]: b for b in bursty_candidates}

    for component in nx.connected_components(graph):
        if len(component) >= 2:
            seg_list = list(component)
            # Sort segments inside cluster by bursty weight wb descending
            seg_list.sort(key=lambda s: bursty_map[s]["wb"] if s in bursty_map else 0.0, reverse=True)

            subgraph = graph.subgraph(component)
            edges_info = [
                (u, v, data.get("weight", 0.0))
                for u, v, data in subgraph.edges(data=True)
            ]

            clusters.append({
                "cluster_id": f"cluster_{cluster_idx}",
                "segments": seg_list,
                "subgraph": subgraph,
                "edges": edges_info,
                "size": len(seg_list)
            })
            cluster_idx += 1

    return clusters, graph
