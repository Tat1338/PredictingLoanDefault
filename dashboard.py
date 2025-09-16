# dashboard.py
# Streamlit dashboard for Loan Serious-Late Prediction
# Works with Streamlit >= 1.49 (uses width='content' / 'stretch')

import os
from os.path import exists, join
import glob
import textwrap

import pandas as pd
import numpy as np
import streamlit as st

# --------------------------------------------------------------------------------------
# Paths & constants
# --------------------------------------------------------------------------------------
ROOT = os.getcwd()
REP_DIR = "reports"
FIG_DIR = join(REP_DIR, "figures")
EXP_DIR = "exports"

KEY_NUMBERS_CSV = join(REP_DIR, "key_numbers.csv")
MODEL_EVAL_CSV = join(EXP_DIR, "model_eval_summary.csv")
THRESHOLD_METRICS_CSV = join(EXP_DIR, "threshold_metrics.csv")
HOLDOUT_PREDICTIONS_CSV = join(EXP_DIR, "holdout_predictions.csv")
SCORE_DECILES_LOGREG_CSV = join(EXP_DIR, "score_deciles_logreg.csv")
SCORE_DECILES_RF_CSV = join(EXP_DIR, "score_deciles_rf.csv")
FIGURE_CAPTIONS_TXT = join(REP_DIR, "figure_captions.txt")

st.set_page_config(
    page_title="Loan Serious-Late — Dashboard",
    page_icon="📊",
    layout="wide",
)

# --------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------
def show_image(obj, caption=None, full=False):
    """Wrapper around st.image using the new width API."""
    st.image(obj, caption=caption, width=("stretch" if full else "content"))

def read_csv_safe(path, **kwargs):
    try:
        if exists(path):
            return pd.read_csv(path, **kwargs)
    except Exception as e:
        st.warning(f"Could not read `{path}`: {e}")
    return None

def read_figure_captions(path=FIGURE_CAPTIONS_TXT):
    """Optional captions: file format is 'filename: caption...' per line."""
    mapping = {}
    if not exists(path):
        return mapping
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if ":" in line:
                    fname, cap = line.split(":", 1)
                    mapping[fname.strip()] = cap.strip()
    except Exception as e:
        st.warning(f"Could not parse figure captions: {e}")
    return mapping

def metric_fmt(x):
    if pd.isna(x):
        return "—"
    try:
        x = float(x)
        if abs(x) >= 1000 and abs(x) < 1_000_000:
            return f"{x:,.0f}"
        if abs(x) >= 1_000_000:
            return f"{x/1_000_000:,.1f}M"
        if x.is_integer():
            return f"{x:,.0f}"
        return f"{x:,.2f}"
    except Exception:
        return str(x)

def show_confusion_table(y_true, y_pred):
    """Simple confusion matrix counts as a small table."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    df = pd.DataFrame(
        {
            "": ["Pred 0", "Pred 1"],
            "True 0": [tn, fp],
            "True 1": [fn, tp],
        }
    )
    st.table(df)

def pill(text, color="violet"):
    st.markdown(
        f"<span style='background:{color};color:white;padding:2px 8px;border-radius:10px;font-size:0.8rem'>{text}</span>",
        unsafe_allow_html=True,
    )

# --------------------------------------------------------------------------------------
# Sidebar navigation
# --------------------------------------------------------------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to:",
    ["Overview", "EDA Gallery", "Model Performance", "Score Explorer", "How to run"],
    index=0,
)

# Left sidebar: expected files
st.sidebar.markdown("### Files this app expects:")
expected = [
    "reports/figures/*.png",
    "reports/key_numbers.csv",
    "exports/holdout_predictions.csv",
    "exports/model_eval_summary.csv",
    "exports/threshold_metrics.csv",
    "exports/score_deciles_logreg.csv",
    "exports/score_deciles_rf.csv",
]
st.sidebar.code("\n".join(expected), language="text")

# --------------------------------------------------------------------------------------
# PAGES
# --------------------------------------------------------------------------------------

# ---------------------------------- Overview -----------------------------------------
if page == "Overview":
    st.title("Loan Serious-Late Prediction — Dashboard")

    # Key numbers (if available)
    kn = read_csv_safe(KEY_NUMBERS_CSV)
    total_clients = late_rate = median_income = None

    if kn is not None:
        # try common column names
        cols_lower = {c.lower(): c for c in kn.columns}
        total_clients = kn.iloc[0][cols_lower.get("total_clients", kn.columns[0])] if not kn.empty else None
        late_rate = kn.iloc[0][cols_lower.get("late_rate", kn.columns[min(1, len(kn.columns)-1)])] if not kn.empty else None
        median_income = kn.iloc[0][cols_lower.get("median_income", kn.columns[min(2, len(kn.columns)-1)])] if not kn.empty else None

    c1, c2, c3 = st.columns(3)
    with c1:
        st.caption("Total clients")
        st.metric(label="", value=metric_fmt(total_clients) if total_clients is not None else "—")
    with c2:
        st.caption("Late rate")
        if late_rate is not None and not pd.isna(late_rate):
            try:
                st.metric(label="", value=f"{float(late_rate)*100:.1f}%")
            except Exception:
                st.metric(label="", value=str(late_rate))
        else:
            st.metric(label="", value="—")
    with c3:
        st.caption("Median income")
        st.metric(label="", value=metric_fmt(median_income) if median_income is not None else "—")

    st.markdown("### What’s here")
    st.markdown(
        "- **EDA Gallery**: all charts saved from the analysis.\n"
        "- **Model Performance**: AUC/AP, thresholds, confusion matrices, lift/gains.\n"
        "- **Score Explorer**: interact with holdout probabilities.\n"
        "- **How to run**: simple instructions to reproduce."
    )

# -------------------------------- EDA Gallery ----------------------------------------
elif page == "EDA Gallery":
    st.title("EDA Gallery")

    captions = read_figure_captions()
    pngs = sorted(glob.glob(join(FIG_DIR, "*.png")))
    if not pngs:
        st.info(f"No figures found in `{FIG_DIR}`. Generate them with `python main.py`.")
    else:
        for p in pngs:
            fname = os.path.basename(p)
            cap = captions.get(fname, "")
            st.subheader(fname)
            if cap:
                st.caption(cap)
            show_image(p, caption=None, full=False)
            st.divider()

# ------------------------------ Model Performance ------------------------------------
elif page == "Model Performance":
    st.title("Model Performance")
    # Banner if critical CSV missing
    if not exists(MODEL_EVAL_CSV):
        st.warning(f"`{MODEL_EVAL_CSV}` is missing.")
        st.button("Generate analysis outputs (run main.py)", help="From your terminal: `python main.py`")

    st.markdown("### Figures")

    # Show any confusion matrix figures if present
    # We attempt a couple of helpful fallbacks
    candidates = [
        ("Confusion — Logistic Regression", ["confusion_logreg.png", "confusion_lr.png"]),
        ("Confusion — Random Forest", ["confusion_rf.png", "confusion_random_forest.png"]),
        ("ROC & PR curves", ["roc_pr_curves.png", "roc_curve.png"]),
        ("Lift & Gain", ["lift_gain.png", "lift_curve.png", "gain_curve.png"]),
    ]

    for title, names in candidates:
        path = None
        for n in names:
            candidate = join(FIG_DIR, n)
            if exists(candidate):
                path = candidate
                break
        if path:
            st.subheader(title)
            show_image(path, full=False)
            st.divider()

    # Model eval table (if available)
    eval_df = read_csv_safe(MODEL_EVAL_CSV)
    if eval_df is not None:
        st.subheader("Model evaluation summary")
        st.dataframe(eval_df, height=400)

    thr_df = read_csv_safe(THRESHOLD_METRICS_CSV)
    if thr_df is not None:
        st.subheader("Threshold sweep metrics")
        st.dataframe(thr_df, height=400)

# --------------------------------- Score Explorer ------------------------------------
elif page == "Score Explorer":
    st.title("Score Explorer (holdout set)")
    if not exists(HOLDOUT_PREDICTIONS_CSV):
        st.info(f"Run `python main.py` first to create `{HOLDOUT_PREDICTIONS_CSV}`.")
    else:
        df = read_csv_safe(HOLDOUT_PREDICTIONS_CSV)
        if df is None or df.empty:
            st.info("Holdout predictions file is empty or unreadable.")
        else:
            # Try typical column names
            # y_true can be 'y', 'target', 'actual', 'label'
            ycol = None
            for c in ["y", "target", "actual", "label", "serious_late"]:
                if c in df.columns:
                    ycol = c
                    break

            # pick first score-like column if not found
            score_cols = [c for c in df.columns if c.lower().startswith("p_") or c.lower().endswith("_prob") or c.lower().endswith("_score")]
            if not score_cols:
                score_cols = [c for c in df.columns if df[c].dtype.kind in "fc" and c != ycol]

            if ycol is None or not score_cols:
                st.warning("Could not identify label/score columns automatically.")
                st.write("Columns:", list(df.columns))
            else:
                model = st.selectbox("Score column", options=score_cols, index=0)
                thr = st.slider("Decision threshold", 0.0, 1.0, 0.5, 0.01)

                # safety
                y_true = (df[ycol].astype(float) > 0.5).astype(int).values
                y_prob = df[model].astype(float).clip(0, 1).values
                y_pred = (y_prob >= thr).astype(int)

                # Metrics
                tp = int(((y_true == 1) & (y_pred == 1)).sum())
                fp = int(((y_true == 0) & (y_pred == 1)).sum())
                tn = int(((y_true == 0) & (y_pred == 0)).sum())
                fn = int(((y_true == 1) & (y_pred == 0)).sum())
                tpr = tp / (tp + fn) if (tp + fn) > 0 else np.nan
                fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan
                prec = tp / (tp + fp) if (tp + fp) > 0 else np.nan
                acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else np.nan

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Accuracy", f"{acc*100:,.1f}%" if pd.notna(acc) else "—")
                c2.metric("Recall (TPR)", f"{tpr*100:,.1f}%" if pd.notna(tpr) else "—")
                c3.metric("Fall-out (FPR)", f"{fpr*100:,.1f}%" if pd.notna(fpr) else "—")
                c4.metric("Precision", f"{prec*100:,.1f}%" if pd.notna(prec) else "—")

                st.markdown("#### Confusion matrix")
                show_confusion_table(y_true, y_pred)

                # Optional decile summaries
                dec_lr = read_csv_safe(SCORE_DECILES_LOGREG_CSV)
                dec_rf = read_csv_safe(SCORE_DECILES_RF_CSV)
                if dec_lr is not None or dec_rf is not None:
                    st.markdown("#### Score deciles (if available)")
                    tabs = []
                    names = []
                    if dec_lr is not None:
                        names.append("LogReg deciles")
                    if dec_rf is not None:
                        names.append("RF deciles")
                    tabs = st.tabs(names)
                    idx = 0
                    if dec_lr is not None:
                        with tabs[idx]:
                            st.dataframe(dec_lr, height=350)
                        idx += 1
                    if dec_rf is not None:
                        with tabs[idx]:
                            st.dataframe(dec_rf, height=350)

# ----------------------------------- How to run --------------------------------------
elif page == "How to run":
    st.title("How to run")

    st.markdown("#### Local steps (Windows / PyCharm)")
    st.code(
        textwrap.dedent(
            """
            1) In Terminal, run:  python main.py
               - This generates figures and CSVs into reports/ and exports/

            2) Start the dashboard:
               python -m streamlit run dashboard.py

            3) Open the Local URL shown in the terminal (usually http://localhost:8501)
            """
        )
    )

    st.markdown("### Files generated by the analysis")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**reports/**")
        st.code(
            "\n".join(
                [
                    "reports/run_environment.txt",
                    "reports/cleaning_report.txt",
                    "reports/figure_captions.txt",
                    "reports/key_numbers.csv",
                    "reports/figures/*.png",
                ]
            ),
            language="text",
        )
    with c2:
        st.markdown("**exports/**")
        st.code(
            "\n".join(
                [
                    "exports/model_eval_summary.csv",
                    "exports/holdout_predictions.csv",
                    "exports/threshold_metrics.csv",
                    "exports/score_deciles_logreg.csv",
                    "exports/score_deciles_rf.csv",
                ]
            ),
            language="text",
        )

    st.markdown("### Deploy on Streamlit Community Cloud / GitHub")
    st.code(
        textwrap.dedent(
            """
            requirements.txt   # include: streamlit, pandas, numpy, scikit-learn, matplotlib, seaborn (if used)
            """
        ),
        language="text",
    )
