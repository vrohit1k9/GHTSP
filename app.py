"""
Streamlit Web Application for GHTSP (Hashtag Tokenization and Rule-Based
Segment Processing for Tweet Segmentation, Extending SEDTWik).

6 Tabs:
1. Hashtag Tokenization before/after table
2. Segments per tweet with x3 hashtag weights
3. Bursty segments with Pb and w_b metrics
4. Jarvis-Patrick Cluster Graph visualization
5. Events with top segments and LexRank summaries
6. GHTSP vs SEDTWik Baseline comparative ablation
"""

import io
from pathlib import Path
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import streamlit as st

import config
from pipeline import GHTSPPipeline
from evaluate import load_annotations, evaluate_pipeline_output


st.set_page_config(
    page_title="GHTSP: Tweet Segmentation & Event Detection",
    page_icon="🐦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #1DA1F2, #0056b3, #9b59b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #555;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        border-left: 4px solid #1DA1F2;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .event-card {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        border: 1px solid #e1e8ed;
        box-shadow: 0 4px 6px rgba(0,0,0,0.03);
    }
    .tag-badge {
        display: inline-block;
        background: #e8f5fd;
        color: #1da1f2;
        border: 1px solid #c8e1f5;
        border-radius: 14px;
        padding: 3px 10px;
        margin: 2px 4px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .tag-burst {
        display: inline-block;
        background: #fdf2e9;
        color: #e67e22;
        border: 1px solid #fadbd8;
        border-radius: 14px;
        padding: 3px 10px;
        margin: 2px 4px;
        font-size: 0.85rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# -------------------------------------------------------------
# SIDEBAR CONTROLS
# -------------------------------------------------------------
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/6/6f/Logo_of_Twitter.svg", width=50)
st.sidebar.title("⚙️ Pipeline Configuration")

# 1. Dataset Upload or Sample Choice
data_source = st.sidebar.radio(
    "Data Source",
    options=["Synthetic Sample (2,100 tweets, planted events)", "Upload Custom CSV/TSV"],
    index=0
)

uploaded_df = None
if data_source == "Upload Custom CSV/TSV":
    file = st.sidebar.file_uploader("Upload Twitter Data (CSV/TSV)", type=["csv", "tsv", "json"])
    if file is not None:
        try:
            if file.name.endswith(".tsv"):
                uploaded_df = pd.read_csv(file, sep="\t")
            elif file.name.endswith(".json"):
                uploaded_df = pd.read_json(file)
            else:
                uploaded_df = pd.read_csv(file)
            st.sidebar.success(f"Loaded {len(uploaded_df)} tweets")
        except Exception as e:
            st.sidebar.error(f"Error loading file: {e}")
else:
    if config.SYNTHETIC_DATA_FILE.exists():
        uploaded_df = pd.read_csv(config.SYNTHETIC_DATA_FILE)
    else:
        st.sidebar.error("Synthetic dataset not found. Generating...")
        from data.generate_synthetic_data import generate_dataset
        generate_dataset()
        uploaded_df = pd.read_csv(config.SYNTHETIC_DATA_FILE)

# 2. Hyperparameters
st.sidebar.subheader("Hyperparameters")
h_weight = st.sidebar.slider("Hashtag Frequency Weight (H)", min_value=1, max_value=5, value=config.HASHTAG_WEIGHT_H)
knn_k = st.sidebar.slider("Jarvis-Patrick Neighbors (k)", min_value=1, max_value=8, value=config.KNN_K)
threshold_t = st.sidebar.slider("Newsworthiness Ratio Filter (T)", min_value=1.0, max_value=10.0, value=config.THRESHOLD_T, step=0.5)

# 3. Module Toggles
st.sidebar.subheader("Extended Modules (GHTSP vs Baseline)")
use_hashtag_tok = st.sidebar.checkbox("Hashtag Tokenizer (NEW)", value=True, help="Disable to run SEDTWik baseline")
use_rules = st.sidebar.checkbox("Rule-Based Segment Processing (NEW)", value=True, help="Disable to run SEDTWik baseline")

rule_merge = True
rule_norm = True
rule_filt = True
if use_rules:
    with st.sidebar.expander("Fine-grained Rule Switches", expanded=False):
        rule_merge = st.checkbox("MERGE Adjacent Entities", value=True)
        rule_norm = st.checkbox("NORMALIZE (strip punctuation & repeats)", value=True)
        rule_filt = st.checkbox("FILTER (stop words, single chars, pure numbers)", value=True)


# -------------------------------------------------------------
# MAIN APP HEADER & RUNNER
# -------------------------------------------------------------
st.markdown('<div class="main-title">GHTSP: Tweet Segmentation & Event Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Extending <b>SEDTWik</b> with Hashtag Tokenization & Rule-Based Segment Processing</div>', unsafe_allow_html=True)

if uploaded_df is None or uploaded_df.empty:
    st.warning("Please select or upload a valid dataset.")
    st.stop()


@st.cache_resource(show_spinner="Running GHTSP Pipeline...")
def execute_pipeline(df_data, use_ht, use_r, h_val, k_val, t_val):
    pipe = GHTSPPipeline(
        use_hashtag_tokenizer=use_ht,
        use_rules=use_r,
        hashtag_weight_h=h_val,
        knn_k=k_val,
        threshold_t=t_val
    )
    return pipe.run(df_data)


pipeline_output = execute_pipeline(uploaded_df, use_hashtag_tok, use_rules, h_weight, knn_k, threshold_t)
windows_dict = pipeline_output["windows"]

# Window Selector
available_windows = list(windows_dict.keys())
selected_window = st.selectbox("📅 Select Time Window (24h)", options=available_windows, index=0)
active_window_data = windows_dict[selected_window]

# Top Metrics Row
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Tweets in Window", len(active_window_data["df"]))
col2.metric("Bursty Segments (Top √Nt)", len(active_window_data["bursty_candidates"]))
col3.metric("Candidate Clusters", len(active_window_data["candidate_clusters"]))
col4.metric("Newsworthy Events Detected", len(active_window_data["surviving_clusters"]))


# -------------------------------------------------------------
# 6 TABS INTERFACE
# -------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "1️⃣ Hashtag Tokenization",
    "2️⃣ Segments & xH Weights",
    "3️⃣ Bursty Extraction (Pb, wb)",
    "4️⃣ Cluster Graph",
    "5️⃣ Events & LexRank Summaries",
    "6️⃣ GHTSP vs Baseline Comparison"
])


# -------------------------------------------------------------
# TAB 1: Hashtag Tokenization Before / After
# -------------------------------------------------------------
with tab1:
    st.subheader("Hashtag Tokenization: Before vs After")
    st.write("Examines how camelCase splitting, digit partitioning, and Wikipedia DP decompose unsegmented hashtags into semantic tokens.")

    # Collect hashtags from the window
    w_df = active_window_data["df"]
    raw_ht_records = []
    seen_hts = set()

    from hashtag_tokenizer import tokenize_hashtag, load_wikipedia_vocabulary
    wiki_v = load_wikipedia_vocabulary()

    for _, row in w_df.iterrows():
        for ht in row["hashtags"]:
            if ht not in seen_hts:
                seen_hts.add(ht)
                toks_with = tokenize_hashtag(ht, vocab=wiki_v, use_tokenizer=True)
                toks_without = tokenize_hashtag(ht, vocab=wiki_v, use_tokenizer=False)
                raw_ht_records.append({
                    "Original Hashtag": ht,
                    "SEDTWik Baseline (No Split)": ", ".join(toks_without),
                    "GHTSP Tokenizer (Split)": ", ".join(toks_with),
                    "Token Count": len(toks_with)
                })

    if raw_ht_records:
        ht_df = pd.DataFrame(raw_ht_records)
        st.dataframe(ht_df, use_container_width=True, hide_index=True)
    else:
        st.info("No hashtags found in this time window.")


# -------------------------------------------------------------
# TAB 2: Segments per Tweet with x3 Hashtag Weights
# -------------------------------------------------------------
with tab2:
    st.subheader(f"Tweet Segmentation & Hashtag Weighting (H = {h_weight})")
    st.write(f"Hashtag tokens are repeated {h_weight}x to give high prior salience to user-tagged concepts.")

    sample_tweets = active_window_data["df"].head(20)
    for idx, row in sample_tweets.iterrows():
        st.markdown(f"""
        <div class="event-card">
            <b>User:</b> @{row['user']} &nbsp;&nbsp;|&nbsp;&nbsp; <b>Time:</b> {row['timestamp']}<br>
            <b>Raw Text:</b> {row['raw_text']}<br>
            <b>Cleaned Text:</b> {row['clean_text']}<br>
            <b>Hashtags:</b> {' '.join(row['hashtags']) if row['hashtags'] else '<i>None</i>'} &nbsp;&nbsp;|&nbsp;&nbsp;
            <b>Mentions:</b> {' '.join(row['mentions']) if row['mentions'] else '<i>None</i>'}<br>
            <div style="margin-top: 8px;"><b>Generated Segments:</b><br>
                {' '.join([f'<span class="tag-badge">{s}</span>' for s in row['segments']])}
            </div>
        </div>
        """, unsafe_allow_html=True)


# -------------------------------------------------------------
# TAB 3: Bursty Segments (Pb, wb)
# -------------------------------------------------------------
with tab3:
    st.subheader("Bursty Segment Extraction")
    st.latex(r"E = N_t \cdot p_s, \quad \sigma = \sqrt{N_t \cdot p_s (1 - p_s)}, \quad P_b = \sigma\left(10 \cdot \frac{f - (E + \sigma)}{\sigma}\right)")
    st.latex(r"w_b = P_b \cdot \log(u) \cdot \log(src) \cdot \log(\log(sfc))")

    bursty_data = active_window_data["bursty_candidates"]
    if bursty_data:
        b_df = pd.DataFrame([
            {
                "Segment": b["segment"],
                "Frequency (f)": b["frequency"],
                "Expected (E)": round(b["expected"], 2),
                "Sigma (σ)": round(b["sigma"], 2),
                "Bursty Prob (Pb)": round(b["Pb"], 4),
                "Bursty Weight (wb)": round(b["wb"], 4),
                "Distinct Users (u)": b["distinct_users"],
                "Retweets (src)": b["sum_retweets"],
                "Followers (sfc)": b["sum_followers"]
            }
            for b in bursty_data
        ])
        st.dataframe(b_df, use_container_width=True, hide_index=True)

        st.subheader("Top Segments by Bursty Weight (wb)")
        top_plot_df = b_df.head(15)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.barh(top_plot_df["Segment"][::-1], top_plot_df["Bursty Weight (wb)"][::-1], color="#1DA1F2")
        ax.set_xlabel("Bursty Weight (w_b)")
        ax.grid(axis="x", linestyle="--", alpha=0.5)
        st.pyplot(fig)
    else:
        st.info("No bursty segments identified for this window.")


# -------------------------------------------------------------
# TAB 4: Jarvis-Patrick Cluster Graph
# -------------------------------------------------------------
with tab4:
    st.subheader(f"Jarvis-Patrick Graph (Mutual {knn_k}-NN)")
    st.write(f"Segments are linked if each is in the other's top {knn_k} nearest neighbors based on M={config.SUBWINDOWS_M} temporal subwindow TF-IDF similarity.")

    graph = active_window_data["graph"]
    candidate_clusters = active_window_data["candidate_clusters"]

    if graph.number_of_nodes() > 0:
        fig, ax = plt.subplots(figsize=(12, 7))
        pos = nx.spring_layout(graph, seed=42, k=0.45)

        # Color nodes by cluster membership
        node_colors = []
        cluster_color_map = {}
        colors_palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

        for idx, c in enumerate(candidate_clusters):
            c_color = colors_palette[idx % len(colors_palette)]
            for s in c["segments"]:
                cluster_color_map[s] = c_color

        for node in graph.nodes():
            node_colors.append(cluster_color_map.get(node, "#cccccc"))

        nx.draw_networkx_nodes(graph, pos, node_color=node_colors, node_size=800, alpha=0.9, ax=ax)
        nx.draw_networkx_labels(graph, pos, font_size=9, font_family="sans-serif", font_weight="bold", ax=ax)
        nx.draw_networkx_edges(graph, pos, width=1.5, alpha=0.6, edge_color="#7f8c8d", ax=ax)

        ax.set_title(f"Cluster Graph: {len(candidate_clusters)} Candidate Components", fontsize=14)
        ax.axis("off")
        st.pyplot(fig)
    else:
        st.info("No connected components formed in this window.")


# -------------------------------------------------------------
# TAB 5: Events with Top Segments & LexRank Summaries
# -------------------------------------------------------------
with tab5:
    st.subheader("Newsworthy Events & LexRank Summaries")
    st.write(f"Surviving events pass the threshold test (mu_max / mu(e) < {threshold_t}), summarized with LexRank damping={config.LEXRANK_DAMPING}.")

    summaries = active_window_data["summaries"]
    if summaries:
        for idx, ev in enumerate(summaries, 1):
            title = ev.get("event_title") or f"Detected Event {idx}: {', '.join(ev.get('top_segments', [])[:3]).title()}"
            inp_tweets = ev.get("input_tweets", f"1-{ev.get('tweet_count', 1)}")
            kw_list = ev.get("extracted_keywords") or [s.title() for s in ev.get("top_segments", [])[:5]]
            ht_list = ev.get("related_hashtags") or []
            summ = ev.get("event_summary")
            if not summ:
                stws = ev.get("summary_tweets", [])
                summ = stws[0].get("text", "") if stws else "Event summary generated."

            with st.expander(f"📌 {title}", expanded=True):
                st.markdown(f"**Input tweets:** `{inp_tweets}`")
                st.markdown(f"**Extracted keywords:** {', '.join(kw_list)}")
                if ht_list:
                    st.markdown(f"**Related hashtags:** {' '.join([f'`{h}`' for h in ht_list])}")
                st.markdown(f"""
                <div style="background:#e8f4fd; border-radius:8px; padding:12px 16px; margin:10px 0; border-left:4px solid #0284c7;">
                    <b style="color:#0369a1; font-size:1.02rem;">Event summary:</b><br>
                    <span style="color:#1e293b; font-size:0.98rem; line-height:1.6;">{summ}</span>
                </div>
                """, unsafe_allow_html=True)
                st.markdown(f"**Top Bursty Segments:** {' '.join([f'<span class=\"tag-burst\">{s}</span>' for s in ev['top_segments']])}", unsafe_allow_html=True)
                st.markdown(f"<small style='color:#64748b;'>Newsworthiness μ(e) = {ev['newsworthiness']} | Ratio = {ev['ratio']} | Tweets participating: {ev['tweet_count']}</small>", unsafe_allow_html=True)
    else:
        st.warning("No candidate clusters survived the newsworthiness threshold.")


# -------------------------------------------------------------
# TAB 6: GHTSP vs Baseline Comparison
# -------------------------------------------------------------
with tab6:
    st.subheader("GHTSP vs SEDTWik Baseline Comparative Ablation")
    st.write("Dynamic side-by-side evaluation against manual annotations. (Recall is excluded per specification).")

    if st.button("🚀 Run Live Comparative Evaluation", key="btn_eval"):
        with st.spinner("Executing SEDTWik baseline and GHTSP pipelines..."):
            annotations_df = load_annotations(config.ANNOTATIONS_FILE)

            # Baseline run
            pipe_baseline = GHTSPPipeline(use_hashtag_tokenizer=False, use_rules=False, hashtag_weight_h=1)
            out_baseline = pipe_baseline.run(uploaded_df)
            eval_b = evaluate_pipeline_output(out_baseline["all_events"], annotations_df)

            # GHTSP run
            pipe_ghtsp = GHTSPPipeline(use_hashtag_tokenizer=True, use_rules=True, hashtag_weight_h=h_weight)
            out_ghtsp = pipe_ghtsp.run(uploaded_df)
            eval_g = evaluate_pipeline_output(out_ghtsp["all_events"], annotations_df)

            # Metric cards
            mcol1, mcol2, mcol3, mcol4 = st.columns(4)
            mcol1.metric("Baseline Precision", f"{eval_b['precision']:.3f}")
            mcol2.metric("GHTSP Precision", f"{eval_g['precision']:.3f}", f"{(eval_g['precision'] - eval_b['precision']):+.3f}")
            mcol3.metric("Baseline DER", f"{eval_b['duplicate_rate']:.3f}")
            mcol4.metric("GHTSP DER", f"{eval_g['duplicate_rate']:.3f}", f"{(eval_g['duplicate_rate'] - eval_b['duplicate_rate']):+.3f}")

            # Comparison Table
            comp_metrics = [
                {"Metric": "Detected Clusters", "SEDTWik Baseline": eval_b["detected"], "GHTSP (Extended)": eval_g["detected"], "Delta": eval_g["detected"] - eval_b["detected"]},
                {"Metric": "Realistic Events", "SEDTWik Baseline": eval_b["realistic"], "GHTSP (Extended)": eval_g["realistic"], "Delta": eval_g["realistic"] - eval_b["realistic"]},
                {"Metric": "Duplicate Events", "SEDTWik Baseline": eval_b["duplicates"], "GHTSP (Extended)": eval_g["duplicates"], "Delta": eval_g["duplicates"] - eval_b["duplicates"]},
                {"Metric": "Precision (Realistic / Detected)", "SEDTWik Baseline": f"{eval_b['precision']:.3f}", "GHTSP (Extended)": f"{eval_g['precision']:.3f}", "Delta": f"{(eval_g['precision'] - eval_b['precision']):+.3f}"},
                {"Metric": "Duplicate Event Rate (DER)", "SEDTWik Baseline": f"{eval_b['duplicate_rate']:.3f}", "GHTSP (Extended)": f"{eval_g['duplicate_rate']:.3f}", "Delta": f"{(eval_g['duplicate_rate'] - eval_b['duplicate_rate']):+.3f}"}
            ]
            st.table(pd.DataFrame(comp_metrics))
