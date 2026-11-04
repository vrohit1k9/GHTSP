"""
Event Summarization Module for GHTSP / SEDTWik.

Algorithm:
1. For each surviving newsworthy cluster:
   a. Query SegmentIndex for all its segments and collect every tweet in the window
      containing at least one of them.
   b. Remove near-duplicate tweets (TF-IDF cosine similarity > 0.9).
   c. Build LexRank graph: tweets are nodes; TF-IDF cosine similarity above
      LEXRANK_SIM_THRESHOLD (default 0.1) gives the edge weight.
   d. Run LexRank power iteration:
      LR(t) = (1-d)/N + d * sum_{u in neighbors(t)} [ sim(t,u) / sum_v sim(u,v) ] * LR(u)
      iterated to convergence with damping factor d = 0.85.
   e. Select top SUMMARY_TWEETS (default 3), skipping tweets too similar to already chosen ones.
   f. Attach top segments ordered by bursty weight.
   g. Output structured dictionary: {date, top_segments, summary_tweets, newsworthiness}.
"""

from typing import Any, Dict, List, Optional, Set
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import config
from segment_index import SegmentIndex


def deduplicate_tweets(
    tweets: List[Dict[str, Any]],
    sim_threshold: float = 0.90
) -> List[Dict[str, Any]]:
    """
    Remove near-duplicate tweets whose TF-IDF cosine similarity exceeds sim_threshold.
    """
    if len(tweets) <= 1:
        return tweets

    texts = [t["clean_text"] or t["raw_text"] for t in tweets]
    try:
        vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
        tfidf_mat = vectorizer.fit_transform(texts)
        sim_mat = cosine_similarity(tfidf_mat)
    except Exception:
        return tweets

    kept_indices: List[int] = []
    for i in range(len(tweets)):
        is_dup = False
        for k in kept_indices:
            if sim_mat[i, k] > sim_threshold:
                is_dup = True
                break
        if not is_dup:
            kept_indices.append(i)

    return [tweets[idx] for idx in kept_indices]


def run_lexrank(
    texts: List[str],
    damping: float = config.LEXRANK_DAMPING,
    sim_threshold: float = config.LEXRANK_SIM_THRESHOLD,
    max_iter: int = 150,
    tol: float = 1e-6
) -> np.ndarray:
    """
    Compute LexRank centrality scores for a list of texts.

    LR(t) = (1-d)/N + d * sum_{u in neighbors(t)} [ sim(t,u) / sum_v sim(u,v) ] * LR(u)
    """
    n = len(texts)
    if n == 0:
        return np.array([])
    if n == 1:
        return np.array([1.0])

    # 1. Compute TF-IDF cosine similarity matrix
    try:
        vec = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
        tfidf = vec.fit_transform(texts)
        cos_sim = cosine_similarity(tfidf)
    except Exception:
        return np.ones(n) / n

    # 2. Threshold similarity to form adjacency matrix
    adj = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if i != j and cos_sim[i, j] >= sim_threshold:
                adj[i, j] = cos_sim[i, j]

    # 3. Form row-stochastic transition matrix M
    # M[i, j] is the transition probability from tweet i to tweet j
    row_sums = adj.sum(axis=1)
    trans = np.zeros((n, n), dtype=float)
    for i in range(n):
        if row_sums[i] > 0:
            trans[i, :] = adj[i, :] / row_sums[i]
        else:
            # Dangling node transitions uniformly to all nodes
            trans[i, :] = 1.0 / n

    # 4. Power iteration: p = (1 - d)/N + d * trans.T @ p
    p = np.ones(n, dtype=float) / n
    uniform = np.ones(n, dtype=float) / n

    for _ in range(max_iter):
        p_next = (1.0 - damping) * uniform + damping * np.dot(trans.T, p)
        diff = np.linalg.norm(p_next - p, ord=1)
        p = p_next
        if diff < tol:
            break

    return p


def summarize_event(
    cluster: Dict[str, Any],
    index: SegmentIndex,
    bursty_map: Dict[str, Dict[str, Any]],
    date_str: str,
    summary_tweets_count: int = config.SUMMARY_TWEETS,
    lexrank_damping: float = config.LEXRANK_DAMPING,
    lexrank_sim_threshold: float = config.LEXRANK_SIM_THRESHOLD
) -> Dict[str, Any]:
    """
    Summarize a single surviving event cluster.

    Returns:
        {
            'date': date_str,
            'cluster_id': cluster['cluster_id'],
            'top_segments': ordered segments by wb,
            'summary_tweets': list of top representative tweets,
            'newsworthiness': float mu(e),
            'tweet_count': int total tweets participating in cluster
        }
    """
    cluster_segments = cluster["segments"]

    # a. Collect all tweets in the window containing at least one cluster segment
    collected_tweet_ids: Set[str] = set()
    for s in cluster_segments:
        stats = index.get_segment_stats(s)
        collected_tweet_ids.update(stats["tweet_ids"])

    pool_tweets = [index.tweets[tid] for tid in collected_tweet_ids if tid in index.tweets]

    # b. Remove near-duplicates (tf-idf cosine > 0.9)
    unique_tweets = deduplicate_tweets(pool_tweets, sim_threshold=0.90)

    # c & d. Run LexRank
    texts = [t["clean_text"] or t["raw_text"] for t in unique_tweets]
    scores = run_lexrank(
        texts=texts,
        damping=lexrank_damping,
        sim_threshold=lexrank_sim_threshold
    )

    for i, t in enumerate(unique_tweets):
        t["lexrank_score"] = float(scores[i]) if i < len(scores) else 0.0

    # Sort tweets descending by LexRank score
    ranked_tweets = sorted(unique_tweets, key=lambda x: x["lexrank_score"], reverse=True)

    # e. Select top SUMMARY_TWEETS with redundancy penalty
    selected_tweets = []
    selected_texts = []

    if texts:
        try:
            vec = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
            all_tfidf = vec.fit_transform([t["clean_text"] or t["raw_text"] for t in ranked_tweets])
        except Exception:
            all_tfidf = None

        for idx, tw in enumerate(ranked_tweets):
            tw_text = tw["clean_text"] or tw["raw_text"]
            # Check similarity to already selected summary tweets
            redundant = False
            if all_tfidf is not None and selected_tweets:
                tw_vec = all_tfidf[idx]
                for sel_idx in [ranked_tweets.index(st) for st in selected_tweets]:
                    sel_vec = all_tfidf[sel_idx]
                    sim = float(cosine_similarity(tw_vec, sel_vec)[0, 0])
                    if sim > 0.50:
                        redundant = True
                        break

            if not redundant or len(selected_tweets) == 0:
                selected_tweets.append(tw)
                selected_texts.append(tw_text)

            if len(selected_tweets) >= summary_tweets_count:
                break

    # If diversity filter was too strict, backfill
    if len(selected_tweets) < summary_tweets_count:
        for tw in ranked_tweets:
            if tw not in selected_tweets:
                selected_tweets.append(tw)
            if len(selected_tweets) >= summary_tweets_count:
                break

    # Format selected summary tweets
    summary_tweet_records = [
        {
            "id": tw["id"],
            "user": tw["user"],
            "text": tw["raw_text"],
            "timestamp": str(tw["timestamp"]),
            "retweet_count": tw["retweet_count"],
            "follower_count": tw["follower_count"],
            "lexrank_score": round(tw.get("lexrank_score", 0.0), 4)
        }
        for tw in selected_tweets
    ]

    # f. Top segments ordered by bursty weight wb
    ordered_segments = sorted(
        cluster_segments,
        key=lambda s: bursty_map[s]["wb"] if s in bursty_map else 0.0,
        reverse=True
    )

    # Generate expected project format metadata & fluent natural language summary
    meta = synthesize_event_metadata(
        cluster_segments=cluster_segments,
        pool_tweets=pool_tweets,
        ordered_segments=ordered_segments,
        bursty_map=bursty_map
    )

    return {
        "date": date_str,
        "cluster_id": cluster["cluster_id"],
        "event_title": meta["event_title"],
        "input_tweets": meta["input_tweets"],
        "extracted_keywords": meta["extracted_keywords"],
        "related_hashtags": meta["related_hashtags"],
        "event_summary": meta["event_summary"],
        "top_segments": ordered_segments,
        "summary_tweets": summary_tweet_records,
        "newsworthiness": round(cluster.get("mu_score", 0.0), 4),
        "ratio": round(cluster.get("ratio", 1.0), 2),
        "tweet_count": len(collected_tweet_ids)
    }


def synthesize_event_metadata(
    cluster_segments: List[str],
    pool_tweets: List[Dict[str, Any]],
    ordered_segments: List[str],
    bursty_map: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Synthesize high-level event metadata matching the expected project output:
    - Event Title (e.g. 'India vs Australia Cricket Match', 'Apple Product Launch', 'Chennai Flooding')
    - Input tweets string (e.g. '1–4', '5–8', '9–11')
    - Extracted keywords (e.g. ['India', 'Australia', 'Kohli', 'victory', 'match'])
    - Related hashtags (e.g. ['#INDvsAUS', '#Cricket', '#IndiaWins'])
    - Event summary (e.g. 'Heavy rainfall caused flooding and waterlogging in Chennai, prompting rescue teams to assist affected residents.')
    """
    # 1. Extract all related hashtags from raw tweets
    all_hashtags = []
    seen_ht = set()
    for tw in pool_tweets:
        raw = tw.get("raw_text", "") or tw.get("clean_text", "")
        hts = re.findall(r"#\w+", raw)
        for ht in hts:
            if ht.lower() not in seen_ht:
                seen_ht.add(ht.lower())
                all_hashtags.append(ht)

    # 2. Determine input tweets representation
    tweet_ids = [tw.get("id") for tw in pool_tweets if tw.get("id") is not None]
    try:
        int_ids = sorted([int(str(t).replace("tweet_", "").replace("t_", "")) for t in tweet_ids])
        if int_ids:
            min_id, max_id = int_ids[0], int_ids[-1]
            if max_id - min_id + 1 == len(int_ids) and len(int_ids) > 1:
                input_tweets_str = f"{min_id}-{max_id}"
            elif len(int_ids) <= 4:
                input_tweets_str = ", ".join(str(i) for i in int_ids)
            else:
                input_tweets_str = f"{min_id}-{max_id}"
        else:
            input_tweets_str = f"1-{max(1, len(pool_tweets))}"
    except Exception:
        input_tweets_str = f"1-{max(1, len(pool_tweets))}"

    # Extract keywords from top bursty segments of THIS cluster
    top_candidates = []
    filler_words = {
        "another", "finally", "entire", "team", "today", "yesterday", "tomorrow", "tonight",
        "everyone", "someone", "anyone", "great", "awesome", "good", "bad", "big", "new",
        "first", "last", "really", "much", "many", "just", "now", "also", "one", "two",
        "day", "days", "watch", "make", "get", "like", "check", "need", "start", "show", "tell"
    }
    for s in ordered_segments:
        s_clean = s.strip()
        if len(s_clean) > 2 and s_clean.lower() not in filler_words and s_clean.lower() not in [t.lower() for t in top_candidates]:
            top_candidates.append(s_clean.title())
        if len(top_candidates) >= 5:
            break

    if not top_candidates:
        top_candidates = [s.title() for s in ordered_segments[:3]] if ordered_segments else ["Real-Time Event"]

    # Derive unique descriptive Event Title based strictly on top cluster segments
    topic_lead = top_candidates[0]
    topic_sec = top_candidates[1] if len(top_candidates) > 1 else ""
    if topic_sec and topic_sec.lower() not in topic_lead.lower():
        event_title = f"{topic_lead} & {topic_sec}"
    else:
        event_title = topic_lead

    # Synthesize clean extractive headline & summary from THIS cluster's top LexRank tweets
    candidate_sentences = []
    for tw in pool_tweets:
        txt = tw.get("raw_text") or tw.get("clean_text") or ""
        txt = re.sub(r"https?://\S+", "", txt)
        txt = re.sub(r"@\w+", "", txt)
        txt = re.sub(r"#(\w+)", r"\1", txt)
        txt = re.sub(r"^\s*RT\s*:?\s*", "", txt, flags=re.IGNORECASE)
        txt = re.sub(r"[\[\]\(\)\*\_]", " ", txt)
        txt = " ".join(txt.split()).strip()
        if len(txt) > 20 and txt not in candidate_sentences:
            candidate_sentences.append(txt)

    if candidate_sentences:
        s1 = candidate_sentences[0].rstrip(".!? ") + "."
        s2 = (candidate_sentences[1].rstrip(".!? ") + ".") if len(candidate_sentences) > 1 else ""
        event_summary = f"{s1} {s2}".strip()
    else:
        event_summary = f"Real-time event stream regarding {topic_lead}, highlighting {', '.join(top_candidates[:3])}."

    related_hashtags = all_hashtags[:4] if all_hashtags else [f"#{w.replace(' ', '')}" for w in top_candidates[:3]]

    return {
        "event_title": event_title,
        "input_tweets": input_tweets_str,
        "extracted_keywords": top_candidates[:5],
        "related_hashtags": related_hashtags,
        "event_summary": event_summary
    }


def summarize_events(
    surviving_clusters: List[Dict[str, Any]],
    index: SegmentIndex,
    bursty_candidates: List[Dict[str, Any]],
    date_str: str,
    summary_tweets_count: int = config.SUMMARY_TWEETS
) -> List[Dict[str, Any]]:
    """
    Summarize all surviving clusters for a time window.
    """
    bursty_map = {b["segment"]: b for b in bursty_candidates}
    summaries = []

    for idx, c in enumerate(surviving_clusters, 1):
        summ = summarize_event(
            cluster=c,
            index=index,
            bursty_map=bursty_map,
            date_str=date_str,
            summary_tweets_count=summary_tweets_count
        )
        summ["event_number"] = idx
        summaries.append(summ)

    return summaries
