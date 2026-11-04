"""
Events2012 Corpus Loader (McMinn et al., 2013).

Provides functions to load and filter the Events2012 evaluation dataset,
specifically focusing on the standard evaluation window: October 11 - 17, 2012.
"""

from datetime import datetime
from pathlib import Path
from typing import Optional, Union
import pandas as pd


# Default evaluation window for Events2012 benchmark
EVAL_START = datetime(2012, 10, 11, 0, 0, 0)
EVAL_END = datetime(2012, 10, 17, 23, 59, 59)


def load_events2012(
    file_path: Union[str, Path],
    start_date: Optional[datetime] = EVAL_START,
    end_date: Optional[datetime] = EVAL_END,
    max_tweets: Optional[int] = None
) -> pd.DataFrame:
    """
    Load Events2012 corpus tweets from CSV, TSV, or JSON lines.

    Expected columns:
        - id: unique tweet identifier
        - user: screen name or user id
        - text: tweet raw text
        - timestamp: datetime string or unix epoch
        - retweet_count: integer (defaults to 0 if absent)
        - follower_count: integer (defaults to 100 if absent)

    Args:
        file_path: Path to the Events2012 file.
        start_date: Start of evaluation window (default 2012-10-11).
        end_date: End of evaluation window (default 2012-10-17).
        max_tweets: Optional upper limit on loaded rows.

    Returns:
        pd.DataFrame standardized to GHTSP schema.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Events2012 file not found: {path}")

    # Determine file format
    if path.suffix.lower() == ".tsv" or ".tsv" in path.name.lower():
        df = pd.read_csv(path, sep="\t", nrows=max_tweets)
    elif path.suffix.lower() in [".json", ".jsonl"]:
        df = pd.read_json(path, lines=True, nrows=max_tweets)
    else:
        df = pd.read_csv(path, nrows=max_tweets)

    # Standardize column names
    col_map = {
        "tweet_id": "id",
        "tweetid": "id",
        "user_id": "user",
        "screen_name": "user",
        "username": "user",
        "created_at": "timestamp",
        "time": "timestamp",
        "retweets": "retweet_count",
        "followers": "follower_count"
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    # Ensure required columns
    required = ["id", "user", "text", "timestamp"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' missing from Events2012 file {path.name}")

    if "retweet_count" not in df.columns:
        df["retweet_count"] = 0
    if "follower_count" not in df.columns:
        df["follower_count"] = 100

    # Parse timestamps
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp", "text"])

    # Filter to evaluation window
    if start_date is not None:
        df = df[df["timestamp"] >= start_date]
    if end_date is not None:
        df = df[df["timestamp"] <= end_date]

    # Clean types
    df["id"] = df["id"].astype(str)
    df["user"] = df["user"].astype(str)
    df["text"] = df["text"].astype(str)
    df["retweet_count"] = df["retweet_count"].fillna(0).astype(int)
    df["follower_count"] = df["follower_count"].fillna(100).astype(int)

    return df.sort_values("timestamp").reset_index(drop=True)
