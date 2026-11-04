"""
End-to-End Pipeline Module for GHTSP & SEDTWik.

Orchestrates the entire architectural flow:
Twitter stream -> Preprocessing -> Hashtag Tokenization (NEW) -> Tweet Segmentation
-> Rule-Based Segment Processing (NEW) -> Segment Index -> Bursty Segment Extraction
-> Bursty Segment Clustering -> Event Newsworthiness Filter -> Event Summarization
-> Event summary with top segments.

Both NEW modules are switchable:
- use_hashtag_tokenizer=True, use_rules=True: GHTSP (extended system)
- use_hashtag_tokenizer=False, use_rules=False: SEDTWik (baseline)
"""

from typing import Any, Dict, List, Optional, Set, Tuple, Union
import pandas as pd
import config
from preprocessing import preprocess_tweets
from hashtag_tokenizer import load_wikipedia_vocabulary
from segmenter import segment_tweet
from rules import apply_rules
from segment_index import build_window_index, SegmentIndex
from bursty import compute_corpus_priors, extract_bursty_segments
from clustering import cluster_bursty_segments
from newsworthiness import filter_newsworthy_events, load_keyphraseness_data
from summarizer import summarize_events


class GHTSPPipeline:
    """
    End-to-End pipeline executor for GHTSP and SEDTWik baseline.
    """

    def __init__(
        self,
        use_hashtag_tokenizer: bool = config.USE_HASHTAG_TOKENIZER,
        use_rules: bool = config.USE_RULES,
        hashtag_weight_h: int = config.HASHTAG_WEIGHT_H,
        subwindows_m: int = config.SUBWINDOWS_M,
        knn_k: int = config.KNN_K,
        threshold_t: float = config.THRESHOLD_T,
        summary_tweets_count: int = config.SUMMARY_TWEETS,
        lexrank_damping: float = config.LEXRANK_DAMPING,
        lexrank_sim_threshold: float = config.LEXRANK_SIM_THRESHOLD
    ):
        self.use_hashtag_tokenizer = use_hashtag_tokenizer
        self.use_rules = use_rules
        self.hashtag_weight_h = hashtag_weight_h
        self.subwindows_m = subwindows_m
        self.knn_k = knn_k
        self.threshold_t = threshold_t
        self.summary_tweets_count = summary_tweets_count
        self.lexrank_damping = lexrank_damping
        self.lexrank_sim_threshold = lexrank_sim_threshold

        # Load shared resources
        self.wiki_vocab = load_wikipedia_vocabulary()
        self.keyphraseness_dict = load_keyphraseness_data()

    def process_tweets_corpus(
        self,
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, List[List[str]]]:
        """
        Step 1 to 4: Preprocessing, Tokenization, Segmentation, and Rule Processing
        across the entire corpus to enable global prior estimation.
        """
        # Step 1: Preprocessing
        clean_df = preprocess_tweets(df, window_hours=config.WINDOW_HOURS)

        all_segmented_tweets: List[List[str]] = []

        # Steps 2, 3, 4: Segmentation + Rules
        for _, row in clean_df.iterrows():
            # Step 2 & 3: Hashtag Tokenization + Longest-Match Segmentation
            segs = segment_tweet(
                clean_text=row["clean_text"],
                hashtags=row["hashtags"],
                mentions=row["mentions"],
                vocab=self.wiki_vocab,
                h_weight=self.hashtag_weight_h,
                use_hashtag_tokenizer=self.use_hashtag_tokenizer
            )

            # Step 4: Rule-Based Segment Processing
            segs_processed = apply_rules(
                segments=segs,
                vocab=self.wiki_vocab,
                use_rules=self.use_rules
            )

            all_segmented_tweets.append(segs_processed)

        return clean_df, all_segmented_tweets

    def run(
        self,
        data: Union[pd.DataFrame, str],
        target_window: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run the full GHTSP / SEDTWik pipeline on a dataset.

        Args:
            data: Input DataFrame or file path to CSV/TSV/JSON.
            target_window: Optional specific window_id to process. If None, processes all.

        Returns:
            Dictionary containing:
                - 'windows': dict of window_id -> window pipeline results
                - 'all_events': flat list of all summarized events across windows
                - 'mode': 'GHTSP' or 'SEDTWik Baseline'
        """
        if isinstance(data, str):
            df = pd.read_csv(data)
        else:
            df = data.copy()

        # Step 1-4: Corpus-level preprocessing & segmentation
        clean_df, all_segmented = self.process_tweets_corpus(df)
        clean_df["segments"] = all_segmented

        # Compute corpus priors p_s
        corpus_priors = compute_corpus_priors(all_segmented)

        window_results: Dict[str, Any] = {}
        all_events: List[Dict[str, Any]] = []

        # Group by 24h window
        unique_windows = clean_df["window_id"].unique()
        if target_window is not None:
            unique_windows = [w for w in unique_windows if w == target_window]

        for w_id in sorted(unique_windows):
            w_df = clean_df[clean_df["window_id"] == w_id].reset_index(drop=True)
            w_segments = w_df["segments"].tolist()

            # Step 5: Build Segment Index
            index = build_window_index(
                window_df=w_df,
                segmented_tweets=w_segments,
                window_id=str(w_id)
            )

            # Step 6: Bursty Segment Extraction
            bursty_candidates = extract_bursty_segments(
                index=index,
                corpus_priors=corpus_priors
            )

            # Step 7: Bursty Segment Clustering (M subwindows & Jarvis-Patrick)
            candidate_clusters, graph = cluster_bursty_segments(
                bursty_candidates=bursty_candidates,
                index=index,
                subwindows_m=self.subwindows_m,
                knn_k=self.knn_k
            )

            # Step 8: Event Newsworthiness Filter
            surviving_clusters = filter_newsworthy_events(
                candidate_clusters=candidate_clusters,
                threshold_t=self.threshold_t,
                keyphraseness_dict=self.keyphraseness_dict,
                wiki_vocab=self.wiki_vocab
            )

            # Step 9: Event Summarization (LexRank + top segments)
            summaries = summarize_events(
                surviving_clusters=surviving_clusters,
                index=index,
                bursty_candidates=bursty_candidates,
                date_str=str(w_id),
                summary_tweets_count=self.summary_tweets_count
            )

            all_events.extend(summaries)

            window_results[str(w_id)] = {
                "window_id": str(w_id),
                "df": w_df,
                "index": index,
                "bursty_candidates": bursty_candidates,
                "candidate_clusters": candidate_clusters,
                "surviving_clusters": surviving_clusters,
                "graph": graph,
                "summaries": summaries
            }

        # Ensure globally unique cluster IDs across all windows
        import re
        for idx, ev in enumerate(all_events, 1):
            top_segs = ev.get("top_segments", [])
            slug = re.sub(r"[^a-zA-Z0-9]+", "_", top_segs[0]).strip("_").lower() if top_segs else "event"
            ev["cluster_id"] = f"cluster_{idx}_{slug}"
            ev["event_number"] = idx

        mode_name = "GHTSP (Extended)" if (self.use_hashtag_tokenizer and self.use_rules) else (
            "SEDTWik Baseline" if (not self.use_hashtag_tokenizer and not self.use_rules) else "Custom GHTSP Hybrid"
        )

        return {
            "mode": mode_name,
            "windows": window_results,
            "all_events": all_events,
            "total_tweets_processed": len(clean_df),
            "total_windows": len(window_results)
        }
