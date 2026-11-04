"""
Segment Index Module for GHTSP / SEDTWik.

Responsibilities:
Build an inverted index for a given time window:
Maps each segment -> {
    'tweet_ids': list of tweet IDs containing s,
    'user_ids': set of distinct user IDs tweeting s,
    'retweet_counts': list of retweet counts for tweets with s,
    'follower_counts': list of follower counts for tweets with s,
    'frequency': f_st (total occurrences in window, including hashtag weight H),
    'distinct_users': count of unique users (u),
    'sum_retweets': sum of retweet counts (src),
    'sum_followers': sum of follower counts (sfc)
}
"""

from collections import defaultdict
from typing import Any, Dict, List, Set
import pandas as pd


def _default_segment_entry() -> Dict[str, Any]:
    return {
        "tweet_ids": [],
        "user_ids": set(),
        "retweet_counts": [],
        "follower_counts": [],
        "frequency": 0
    }


class SegmentIndex:
    """
    Inverted index for a 24-hour time window.
    """

    def __init__(self, window_id: str):
        self.window_id = window_id
        # segment -> postings dict
        self.index: Dict[str, Dict[str, Any]] = defaultdict(_default_segment_entry)
        # tweet_id -> tweet record for quick lookup
        self.tweets: Dict[str, Dict[str, Any]] = {}
        # total count of tweets in this window (N_t)
        self.total_tweets: int = 0
        # total count of segment tokens in this window
        self.total_segment_tokens: int = 0

    def add_tweet(
        self,
        tweet_id: str,
        user: str,
        raw_text: str,
        clean_text: str,
        segments: List[str],
        timestamp: Any,
        retweet_count: int,
        follower_count: int
    ) -> None:
        """Add a processed tweet and its segments to the index."""
        self.tweets[tweet_id] = {
            "id": tweet_id,
            "user": user,
            "raw_text": raw_text,
            "clean_text": clean_text,
            "segments": segments,
            "timestamp": timestamp,
            "retweet_count": retweet_count,
            "follower_count": follower_count
        }
        self.total_tweets += 1
        self.total_segment_tokens += len(segments)

        # Count occurrences in this tweet
        seg_counts: Dict[str, int] = defaultdict(int)
        for s in segments:
            seg_counts[s] += 1

        for s, count in seg_counts.items():
            entry = self.index[s]
            entry["tweet_ids"].append(tweet_id)
            entry["user_ids"].add(user)
            entry["retweet_counts"].append(retweet_count)
            entry["follower_counts"].append(follower_count)
            entry["frequency"] += count

    def get_segment_stats(self, segment: str) -> Dict[str, Any]:
        """
        Return aggregated statistics for a segment in this window:
        - f_st: observed frequency
        - u: distinct user count
        - src: summed retweet count
        - sfc: summed follower count
        """
        entry = self.index.get(segment)
        if not entry:
            return {
                "frequency": 0,
                "distinct_users": 0,
                "sum_retweets": 0,
                "sum_followers": 0,
                "tweet_ids": []
            }

        return {
            "frequency": entry["frequency"],
            "distinct_users": len(entry["user_ids"]),
            "sum_retweets": sum(entry["retweet_counts"]),
            "sum_followers": sum(entry["follower_counts"]),
            "tweet_ids": entry["tweet_ids"],
            "user_ids": entry["user_ids"]
        }

    def all_segments(self) -> List[str]:
        """List all unique segments indexed in this window."""
        return list(self.index.keys())


def build_window_index(
    window_df: pd.DataFrame,
    segmented_tweets: List[List[str]],
    window_id: str
) -> SegmentIndex:
    """
    Construct a SegmentIndex for a dataframe of tweets belonging to one window.
    """
    idx = SegmentIndex(window_id=window_id)

    for i, (_, row) in enumerate(window_df.iterrows()):
        segments = segmented_tweets[i]
        idx.add_tweet(
            tweet_id=str(row["id"]),
            user=str(row["user"]),
            raw_text=str(row.get("raw_text", row.get("text", ""))),
            clean_text=str(row.get("clean_text", "")),
            segments=segments,
            timestamp=row["timestamp"],
            retweet_count=int(row.get("retweet_count", 0)),
            follower_count=int(row.get("follower_count", 100))
        )

    return idx
