"""
Preprocessing Module for GHTSP / SEDTWik.

Responsibilities:
1. Filter out retweets (e.g., beginning with 'RT @' or explicit retweet flag).
2. Remove URLs (http://, https://, t.co/...).
3. Remove emojis and control character noise.
4. Separate main tweet text, hashtags, and @mentions.
5. Assign tweets to fixed 24-hour windows.
"""

import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
from config import WINDOW_HOURS


# Regex patterns
RE_RT = re.compile(r"^RT\s+@\w+[:\s]?", re.IGNORECASE)
RE_URL = re.compile(r"https?://\S+|www\.\S+|t\.co/\S+", re.IGNORECASE)
RE_HASHTAG = re.compile(r"#\w+", re.UNICODE)
RE_MENTION = re.compile(r"@\w+", re.UNICODE)
# Emoji & symbol regex (matches typical emoji ranges in unicode)
RE_EMOJI = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F1E0-\U0001F1FF"  # flags
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "\U0001F900-\U0001F9FF"  # supplemental symbols
    "\U0001FA70-\U0001FAFF"  # symbols and pictographs extended-a
    "]+",
    flags=re.UNICODE
)
RE_EXTRA_WHITESPACE = re.compile(r"\s+")


def is_retweet(text: str) -> bool:
    """Check if the tweet text indicates a retweet."""
    if not text:
        return False
    return bool(RE_RT.match(text.strip()))


def extract_entities(raw_text: str) -> Tuple[str, List[str], List[str]]:
    """
    Extract hashtags and mentions, remove URLs and emojis, and clean the text.

    Args:
        raw_text: The original tweet text.

    Returns:
        clean_text: Text with URLs, emojis, hashtags, and mentions removed.
        hashtags: List of hashtags found (e.g., ['#BreakingNews', '#Oct12Debate']).
        mentions: List of mentions found (e.g., ['@joebiden', '@paulryan']).
    """
    if not isinstance(raw_text, str):
        return "", [], []

    # 1. Extract hashtags and mentions before removal
    hashtags = RE_HASHTAG.findall(raw_text)
    mentions = RE_MENTION.findall(raw_text)

    # 2. Remove URLs
    text = RE_URL.sub(" ", raw_text)

    # 3. Remove Emojis
    text = RE_EMOJI.sub(" ", text)

    # 4. Remove hashtags and mentions from main text body
    text = RE_HASHTAG.sub(" ", text)
    text = RE_MENTION.sub(" ", text)

    # 5. Clean whitespace & punctuation noise
    text = RE_EXTRA_WHITESPACE.sub(" ", text).strip()

    return text, hashtags, mentions


def assign_24h_window(
    dt: datetime,
    base_time: Optional[datetime] = None,
    window_hours: int = WINDOW_HOURS
) -> str:
    """
    Assign a timestamp to a 24-hour window identifier.
    Uses calendar date (YYYY-MM-DD) if window_hours == 24 and base_time is None,
    or window_N if a custom base_time is supplied.
    """
    if window_hours == 24 and base_time is None:
        return dt.strftime("%Y-%m-%d")
    
    if base_time is None:
        base_time = dt.replace(hour=0, minute=0, second=0, microsecond=0)

    delta_seconds = (dt - base_time).total_seconds()
    window_index = max(0, int(delta_seconds // (window_hours * 3600)))
    return f"window_{window_index}"


def preprocess_tweets(
    df: pd.DataFrame,
    window_hours: int = WINDOW_HOURS,
    filter_retweets: bool = True
) -> pd.DataFrame:
    """
    Preprocess tweets DataFrame according to SEDTWik / GHTSP architecture.

    Expected input columns: ['id', 'user', 'text', 'timestamp', 'retweet_count', 'follower_count']
    Output columns: ['id', 'user', 'raw_text', 'clean_text', 'hashtags', 'mentions',
                     'timestamp', 'window_id', 'retweet_count', 'follower_count']
    """
    data = df.copy()
    if "text" not in data.columns:
        for candidate in ["raw_text", "clean_text", "tweet", "Tweet", "tweets", "Tweets", "content", "Content", "message", "Message"]:
            if candidate in data.columns:
                data["text"] = data[candidate]
                break
        if "text" not in data.columns:
            obj_cols = [c for c in data.columns if data[c].dtype == object]
            if obj_cols:
                data["text"] = data[obj_cols[0]]
            else:
                data["text"] = ""

    # Ensure id exists
    if "id" not in data.columns:
        data["id"] = [f"tweet_{i+1}" for i in range(len(data))]

    # Ensure user exists
    if "user" not in data.columns:
        data["user"] = [f"user_{(i % 200) + 1}" for i in range(len(data))]

    # Filter retweets
    if filter_retweets and "text" in data.columns:
        # Exclude tweets that match RT syntax
        mask_not_rt = ~data["text"].astype(str).apply(is_retweet)
        data = data[mask_not_rt]

    # Convert timestamps or generate default
    if "timestamp" in data.columns:
        data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
    else:
        start_ts = pd.Timestamp("2026-10-29 00:00:00")
        step_seconds = max(1, int(86400 / max(1, len(data))))
        data["timestamp"] = [start_ts + pd.Timedelta(seconds=i * step_seconds) for i in range(len(data))]

    # If any timestamps became NaT, fill them
    if data["timestamp"].isna().any():
        start_ts = pd.Timestamp("2026-10-29 00:00:00")
        step_seconds = max(1, int(86400 / max(1, len(data))))
        fallback_ts = [start_ts + pd.Timedelta(seconds=i * step_seconds) for i in range(len(data))]
        data["timestamp"] = data["timestamp"].fillna(pd.Series(fallback_ts, index=data.index))

    data = data.dropna(subset=["timestamp", "text"]).copy()
    data = data.sort_values("timestamp").reset_index(drop=True)

    if data.empty:
        return pd.DataFrame(columns=[
            "id", "user", "raw_text", "clean_text", "hashtags", "mentions",
            "timestamp", "window_id", "retweet_count", "follower_count"
        ])

    base_time = data["timestamp"].min().floor("D")

    cleaned_texts = []
    extracted_hashtags = []
    extracted_mentions = []
    window_ids = []

    for _, row in data.iterrows():
        raw_t = str(row["text"])
        c_text, htags, mnts = extract_entities(raw_t)
        cleaned_texts.append(c_text)
        extracted_hashtags.append(htags)
        extracted_mentions.append(mnts)
        w_id = assign_24h_window(row["timestamp"], base_time=base_time, window_hours=window_hours)
        window_ids.append(w_id)

    data["raw_text"] = data["text"]
    data["clean_text"] = cleaned_texts
    data["hashtags"] = extracted_hashtags
    data["mentions"] = extracted_mentions
    data["window_id"] = window_ids

    # Fill default counts if missing
    if "retweet_count" not in data.columns:
        data["retweet_count"] = 0
    else:
        data["retweet_count"] = data["retweet_count"].fillna(0).astype(int)

    if "follower_count" not in data.columns:
        data["follower_count"] = 100
    else:
        data["follower_count"] = data["follower_count"].fillna(100).astype(int)

    return data[[
        "id", "user", "raw_text", "clean_text", "hashtags", "mentions",
        "timestamp", "window_id", "retweet_count", "follower_count"
    ]].reset_index(drop=True)
