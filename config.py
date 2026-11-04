"""
Configuration parameters for GHTSP (Hashtag Tokenization and Rule-Based Segment Processing
for Tweet Segmentation, Extending SEDTWik).

All parameters live here as specified in the architecture.
"""

from pathlib import Path

# --- Pipeline Parameters ---
WINDOW_HOURS: int = 24
SUBWINDOWS_M: int = 12
HASHTAG_WEIGHT_H: int = 3
KNN_K: int = 3
THRESHOLD_T: float = 4.0
LEXRANK_DAMPING: float = 0.85
LEXRANK_SIM_THRESHOLD: float = 0.1
SUMMARY_TWEETS: int = 3

# --- Module Toggles (Baseline SEDTWik vs Extended GHTSP) ---
USE_HASHTAG_TOKENIZER: bool = True
USE_RULES: bool = True

# --- Fine-grained Rule Toggles (active when USE_RULES is True) ---
RULE_MERGE_ADJACENT: bool = True
RULE_NORMALIZE: bool = True
RULE_FILTER: bool = True

# --- Paths ---
BASE_DIR: Path = Path(__file__).resolve().parent
DATA_DIR: Path = BASE_DIR / "data"
WIKI_TITLES_FILE: Path = DATA_DIR / "wikipedia_titles_fallback.txt"
WIKI_GZ_FILE: Path = DATA_DIR / "enwiki-latest-all-titles-in-ns0.gz"
KEYPHRASENESS_FILE: Path = DATA_DIR / "wikipedia_keyphraseness.json"
SYNTHETIC_DATA_FILE: Path = DATA_DIR / "synthetic_tweets.csv"
ANNOTATIONS_FILE: Path = DATA_DIR / "ground_truth_annotations.csv"

# --- Mention to Real Name Mapping ---
MENTION_NAME_MAP = {
    "@barackobama": "Barack Obama",
    "@mittromney": "Mitt Romney",
    "@joebiden": "Joe Biden",
    "@paulryan": "Paul Ryan",
    "@cnn": "CNN",
    "@bbcbreaking": "BBC Breaking News",
    "@reuters": "Reuters",
    "@whitehouse": "White House",
    "@gop": "Republican Party",
    "@thedemocrats": "Democratic Party"
}
