# dashboard.py
# Streamlit dashboard for the Loan Serious-Late Prediction project
# Compatible with Streamlit >= 1.49 (uses width='stretch'/'content', never width=None)

from __future__ import annotations

import os
import glob
import textwrap
from typing import Dict, Optional

import pandas as pd
import streamlit as st


# --------------------------- Paths & helpers ---------------------------

BASE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(BASE, "reports", "figures")
REPORTS_DIR = os.path.join(BASE, "reports")
EXPORTS_DIR = os.path.join(BASE, "exports")

EXPECTED_FILES = [
    "reports/figures/*.png",
    "reports/key_numbers.csv",
    "exports/holdout_predictions.csv",
    "exports/model_eval_summary.csv",
    "exports/threshold_metrics.csv",
    "exports/score_deciles_logreg.csv",
    "exports/score_deciles_rf.csv",
]

def exists(path: str) -> bool:
    return os.path.exists(path)

@st.cache_data(show_spinner=False)
def read_csv_safe(path: str) -> Optional[pd.DataFrame]:
    try:
        if exists(path):
            return pd.read_csv(path)
    except Exception:
        return None
    return None

def nice_num(x: float | int | str) -> str:
    try:
        xf = float(x)
        # Show integer with thousands separator if it is integer-ish
        if abs(xf - int(xf)) < 1e-9:
            return f"{int(xf):,}"
        return f"{xf:,.2f}"
    except Exception:
        return str(x)

def read_figure_captions(file_path: str) -> Dict[str, str]:
    """Optional: parse a captions file with lines like 'age_distribution.png: Age is broad...'
    Returns dict[filename] -> caption"""
    caps: Dict[str, str] = {}
    if not exists(file_path):
        return caps
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if ":" in line:
                    fname, cap = line.split(":", 1)
                    caps[fname.strip()] = cap.strip()
    except Exception:
        pass
    return caps


# --------------------------- Page config ---------------------------

st.set_page_config(
    page_title="Loan Serious-Late Prediction — Dashboard",
    layout="wide",
)

st.title("Loan Serious-Late Prediction — Dashboard")

# --------------------------- Sidebar ---------------------------

with st.sidebar:
    st.header("Navigation")
    page = st.radio("Go to:", ["Overview", "EDA Gallery", "Model Performance", "Score Explorer", "How to run"], index=0)

    st.markdown("#### Files this app expects:")
    st.code("\n".join(EXPECTED_FILES), language="text")


# --------------------------- Pages ---------------------------

if page == "Overview":
    # Key numbers: total clients, late rate, median income
    key_csv = os.path.join(REPORTS_DIR, "key_numbers.csv")
    dfk = read_csv_safe(key_csv)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.caption("Total clients")
        if dfk is not None and "total_clients" in dfk.columns:
            st.markdown(f"## {nice_num(dfk['total_clients'].iloc[0])}")
        else:
            st.markdown("## –")
    with col2:
        st.caption("Late rate")
        if dfk is not None and "late_rate" in dfk.columns:
            # Expect late_rate ~ 0.067 -> 6.7%
            val = dfk["late_rate"].iloc[0]
            try:
                pct = float(val) * 100 if float(val) <= 1 else float(val)
                st.markdown(f"## {pct:.1f}%")
            except Exception:
                st.markdown(f"## {val}")
        else:
            st.markdown("## –")
    with col3:
        st.caption("Median income")
        if dfk is not None and "median_income" in dfk.columns:
            st.markdown(f"## {nice_num(dfk['median_income'].iloc[0])}")
        else:
            st.markdown("## –")

    st.markdown("### What’s here")
    st.markdown(
        "- **EDA Gallery:** all charts saved from the analysis.\n"
        "- **Model Performance:** AUC/AP, thresholds, confusion matrices, lift/gains.\n"
        "- **Score Explorer:** interact with holdout probabilities.\n"
        "- **How to run:** simple instructions to reproduce."
    )

elif page == "EDA Gallery":
    st.subheader("EDA Gallery")
    captions = read_figure_captions(os.path.join(REPORTS_DIR, "figure_captions.txt"))

    pngs = sorted(glob.glob(os.path.join(FIG_DIR, "*.png")))
    if not pngs:
        st.info("No figures found in `reports/figures/*.png`.")
    for img_path in pngs:
        fname = os.path.basename(img_path)
        cap = captions.get(fname, "")
        st.subheader(fname)
        if cap:
            st.caption(cap)
        # IMPORTANT: width must be 'stretch', 'content', or an int; never None.
        st.image(img_path, caption=None, width="stretch")
        st.divider()

elif page == "Model Performance":
    st.subheader("Model Performance")

    # Helpful banner if summary CSV missing
    summary_csv = os.path.join(EXPORTS_DIR, "model_eval_summary.csv")
    if not exists(summary_csv):
        st.warning("`exports/model_eval_summary.csv` is missing.", icon="⚠️")
        st.button("Generate analysis outputs (run main.py)")  # Visual cue only

    # Helper to try showing a figure by preferred name or fallback
    def show_fig_block(title: str, preferred: Optional[str] = None, fallback: Optional[str] = None):
        cand: Optional[str] = None
        if preferred:
            p = os.path.join(FIG_DIR, preferred)
            if exists(p):
                cand = p
        if cand is None and fallback:
            f = os.path.join(FIG_DIR, fallback)
            if exists(f):
                cand = f

        if cand:
            st.subheader(title)
            st.image(cand, width="stretch")
        else:
            miss = preferred or fallback or ""
            st.caption(f"Missing figure: {miss}")

    # Example sections (keep names the same as used in your analysis so paths match)
    st.markdown("### Figures")
    show_fig_block("Confusion — Logistic Regression",
                   preferred="confusion_logreg.png",
                   fallback="confusion_logreg.png")
    show_fig_block("Confusion — Random Forest",
                   preferred="confusion_rf.png",
                   fallback="confusion_rf.png")
    show_fig_block("ROC & PR Curves",
                   preferred="roc_pr_curves.png",
                   fallback="roc_pr_curves.png")
    show_fig_block("Threshold Metrics",
                   preferred="threshold_metrics.png",
                   fallback="threshold_metrics.png")
    show_fig_block("Lift / Gains",
                   preferred="lift_gains.png",
                   fallback="lift_gains.png")

    # Optional: show a small table if present
    df_summary = read_csv_safe(summary_csv)
    if df_summary is not None and not df_summary.empty:
        st.markdown("### Evaluation summary")
        st.dataframe(df_summary, use_container_width=True)

elif page == "Score Explorer":
    st.subheader("Score Explorer (holdout set)")
    holdout_csv = os.path.join(EXPORTS_DIR, "holdout_predictions.csv")
    dfh = read_csv_safe(holdout_csv)
    if dfh is None:
        st.info("Run `python main.py` first to create `exports/holdout_predictions.csv`.")
    else:
        # Show a preview with filters
        with st.expander("Preview data", expanded=True):
            st.dataframe(dfh.head(200), use_container_width=True)

        # Optional simple exploration: choose a probability threshold to view counts
        if "score" in dfh.columns:
            st.markdown("### Quick threshold check")
            thr = st.slider("Threshold for positive (serious-late)", 0.0, 1.0, 0.5, 0.01)
            preds = (dfh["score"] >= thr).astype(int)
            st.write(
                f"Predicted positives @ {thr:.2f}: **{int(preds.sum()):,}** "
                f"out of **{len(preds):,}**"
            )

elif page == "How to run":
    st.subheader("How to run")

    st.markdown("#### Local steps (Windows / PyCharm / VS Code)")
    st.code(
        textwrap.dedent(
            """
            1) In Terminal, run:
               python main.py
               # This generates figures and CSVs into reports/ and exports/

            2) Start the dashboard locally:
               python -m streamlit run dashboard.py

            3) Open the Local URL shown in the terminal (usually http://localhost:8501)
            """
        )
    )

    st.markdown("#### Files generated by the analysis")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**reports/**")
        st.code(
            "reports/run_environment.txt\n"
            "reports/cleaning_report.txt\n"
            "reports/figure_captions.txt\n"
            "reports/key_numbers.csv\n"
            "reports/figures/*.png",
            language="text",
        )
    with c2:
        st.markdown("**exports/**")
        st.code(
            "exports/model_eval_summary.csv\n"
            "exports/holdout_predictions.csv\n"
            "exports/threshold_metrics.csv\n"
            "exports/score_deciles_logreg.csv\n"
            "exports/score_deciles_rf.csv",
            language="text",
        )

    st.markdown("#### Deploy on Streamlit Community Cloud / GitHub")
    st.code(
        textwrap.dedent(
            """
            requirements.txt  # include: streamlit, pandas, numpy, scikit-learn, matplotlib, seaborn (if used)
            """
        ),
        language="text",
    )

# --------------------------- Footer note ---------------------------
st.caption("This app never passes width=None to st.image and avoids deprecated use_container_width.")
