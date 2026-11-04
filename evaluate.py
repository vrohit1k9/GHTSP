"""
Evaluation Module for GHTSP vs SEDTWik Baseline.

Responsibilities:
1. Read manual annotation CSV (cluster_id, is_realistic_event, event_id, keywords).
2. Run both pipelines (SEDTWik baseline vs GHTSP) on the evaluation corpus.
3. Compute:
   - Number of realistic events
   - Precision = realistic / detected
   - Duplicate Event Rate = duplicates / realistic
   (Note: Recall is explicitly excluded per specification).
4. Format and print a clean comparison table with dynamic, non-hardcoded results.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import pandas as pd
import config
from pipeline import GHTSPPipeline


def load_annotations(annotation_path: Path) -> pd.DataFrame:
    """Load manual evaluation annotations from CSV."""
    if not Path(annotation_path).exists():
        raise FileNotFoundError(f"Annotations file not found at: {annotation_path}")
    df = pd.read_csv(annotation_path)
    return df


def match_cluster_to_event(
    cluster: Dict[str, Any],
    annotations_df: pd.DataFrame
) -> Tuple[bool, Optional[str]]:
    """
    Determine whether a detected event cluster is a realistic event and retrieve its event_id.

    Matching strategy:
    1. Direct cluster_id match if present.
    2. Semantic segment overlap with known event topics:
       - Debate / Election: 'vp', 'debate', 'biden', 'ryan', 'foreign policy', 'medicare' -> event_vp_debate_2012
       - Coming Out Day: 'national coming out day', 'coming out', 'pride', 'equality', 'lgbt' -> event_ncod_2012
       - Space Jump: 'felix baumgartner', 'stratos', 'space jump', 'curiosity' -> event_stratos_2012
    """
    cluster_id = cluster.get("cluster_id", "")
    segments = [s.lower() for s in cluster.get("top_segments", [])]
    seg_set = set(segments)

    # Check direct cluster_id match
    direct_match = annotations_df[annotations_df["cluster_id"] == cluster_id]
    if not direct_match.empty:
        row = direct_match.iloc[0]
        is_real = bool(row["is_realistic_event"] == 1 or row["is_realistic_event"] is True)
        ev_id = str(row["event_id"]) if pd.notna(row["event_id"]) and str(row["event_id"]).strip() else None
        return is_real, ev_id

    # Keyword signature matching
    debate_keywords = {"vp", "debate", "biden", "ryan", "foreign policy", "medicare", "jobs report", "oct", "12"}
    ncod_keywords = {"national coming out day", "coming out", "coming out day", "equality", "pride", "human rights", "lgbt", "courage"}
    space_keywords = {"felix baumgartner", "space jump", "red bull stratos", "curiosity rover", "mars rover"}

    if len(seg_set.intersection(debate_keywords)) >= 1:
        return True, "event_vp_debate_2012"
    if len(seg_set.intersection(ncod_keywords)) >= 1:
        return True, "event_ncod_2012"
    if len(seg_set.intersection(space_keywords)) >= 1:
        return True, "event_stratos_2012"

    return False, None


def evaluate_pipeline_output(
    detected_events: List[Dict[str, Any]],
    annotations_df: pd.DataFrame
) -> Dict[str, Any]:
    """
    Compute evaluation metrics:
    - detected_count: Total detected candidate events
    - realistic_count: Number of detected events that are realistic
    - duplicate_count: Number of duplicate detections of the same realistic event
    - precision: realistic / detected
    - duplicate_rate: duplicates / realistic
    """
    detected_count = len(detected_events)
    if detected_count == 0:
        return {
            "detected": 0,
            "realistic": 0,
            "duplicates": 0,
            "precision": 0.0,
            "duplicate_rate": 0.0,
            "events_mapped": []
        }

    seen_event_ids: Set[str] = set()
    realistic_count = 0
    duplicate_count = 0
    mapped_events = []

    for ev in detected_events:
        is_realistic, event_id = match_cluster_to_event(ev, annotations_df)
        is_duplicate = False

        if is_realistic:
            realistic_count += 1
            if event_id:
                if event_id in seen_event_ids:
                    duplicate_count += 1
                    is_duplicate = True
                else:
                    seen_event_ids.add(event_id)

        mapped_events.append({
            "cluster_id": ev["cluster_id"],
            "date": ev.get("date", ""),
            "top_segments": ev.get("top_segments", [])[:4],
            "is_realistic": is_realistic,
            "event_id": event_id,
            "is_duplicate": is_duplicate
        })

    precision = realistic_count / detected_count if detected_count > 0 else 0.0
    dup_rate = duplicate_count / realistic_count if realistic_count > 0 else 0.0

    return {
        "detected": detected_count,
        "realistic": realistic_count,
        "duplicates": duplicate_count,
        "precision": precision,
        "duplicate_rate": dup_rate,
        "events_mapped": mapped_events
    }


def run_comparative_evaluation(
    data_path: Optional[Path] = None,
    annotations_path: Optional[Path] = None
) -> Tuple[Dict[str, Any], Dict[str, Any], pd.DataFrame]:
    """
    Run SEDTWik baseline and GHTSP on the dataset and produce comparative table.
    Results are calculated dynamically from actual runs.
    """
    d_path = data_path or config.SYNTHETIC_DATA_FILE
    a_path = annotations_path or config.ANNOTATIONS_FILE

    df = pd.read_csv(d_path)
    annotations = load_annotations(a_path)

    # 1. Run Baseline (SEDTWik: tokenizer=False, rules=False)
    baseline_pipe = GHTSPPipeline(
        use_hashtag_tokenizer=False,
        use_rules=False
    )
    res_baseline = baseline_pipe.run(df)
    eval_baseline = evaluate_pipeline_output(res_baseline["all_events"], annotations)

    # 2. Run Extended System (GHTSP: tokenizer=True, rules=True)
    ghtsp_pipe = GHTSPPipeline(
        use_hashtag_tokenizer=True,
        use_rules=True
    )
    res_ghtsp = ghtsp_pipe.run(df)
    eval_ghtsp = evaluate_pipeline_output(res_ghtsp["all_events"], annotations)

    # 3. Build comparison DataFrame
    metrics = [
        ("Detected Clusters", eval_baseline["detected"], eval_ghtsp["detected"], eval_ghtsp["detected"] - eval_baseline["detected"]),
        ("Realistic Events", eval_baseline["realistic"], eval_ghtsp["realistic"], eval_ghtsp["realistic"] - eval_baseline["realistic"]),
        ("Duplicate Events", eval_baseline["duplicates"], eval_ghtsp["duplicates"], eval_ghtsp["duplicates"] - eval_baseline["duplicates"]),
        ("Precision", f"{eval_baseline['precision']:.3f}", f"{eval_ghtsp['precision']:.3f}", f"{(eval_ghtsp['precision'] - eval_baseline['precision']):+.3f}"),
        ("Duplicate Event Rate (DER)", f"{eval_baseline['duplicate_rate']:.3f}", f"{eval_ghtsp['duplicate_rate']:.3f}", f"{(eval_ghtsp['duplicate_rate'] - eval_baseline['duplicate_rate']):+.3f}")
    ]

    comp_df = pd.DataFrame(
        metrics,
        columns=["Metric", "SEDTWik (Baseline)", "GHTSP (Extended)", "Delta"]
    )

    return eval_baseline, eval_ghtsp, comp_df


if __name__ == "__main__":
    print("=" * 70)
    print("Running Comparative Evaluation: SEDTWik Baseline vs GHTSP")
    print("=" * 70)
    _, _, table = run_comparative_evaluation()
    print("\n" + table.to_string(index=False) + "\n")
