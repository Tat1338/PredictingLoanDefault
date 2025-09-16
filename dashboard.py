# dashboard.py
# Streamlit dashboard for Loan Serious-Late Prediction
# Safe with Streamlit >= 1.49 and Python 3.13
# - No width=None / use_container_width
# - Robust guards around file/column issues
# - Exceptions are shown in the app instead of killing it

import os
from os.path import exists, join
import glob
import textwrap

import numpy as np
import pandas as pd
import streamlit as st

# --------------------------------------------------------------------------------------
# Basic setup
# --------------------------------------------------------------------------------------
st.set_page_config(
    page_title="Loan Serious-Late — Dashboard",
    page_icon="📊",
    layout="wide",
)

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


# --------------------------------------------------------------------------------------
# Helpers (robust / won’t throw)
# --------------------------------------------------------------------------------------
def show_image(obj, caption=None, full=False):
    """Wrapper using the new width API: 'content' or 'stretch' only."""
    try:
        st.image(obj, caption=caption, width=("stretch" if full else "content"))
    except Exception as e:
        st.warning(f"Could not render image {obj}: {e}")

def read_csv_safe(path, **kwargs):
    try:
        if exists(path):
            return pd.read_csv(path, **kwargs)
        return None
    except Exception as e:
        st.warning(f"Could not read `{path}`: {e}")
        return None

def read_figure_captions(path=FIGURE_CAPTIONS_TXT):
    mapping = {}
    if not exists(path):
        return mapping
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or ":" not in line:
                    continue
                fname, cap = line.split(":", 1)
                mapping[fname.strip()] = cap.strip()
    except Exception as e:
        st.warning(f"Could not parse figure captions: {e}")
    return mapping

def metric_fmt(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    try:
        x = float(x)
        if abs(x) >= 1_000_000:
            return f"{x/1_000_000:,.1f}M"
        if abs(x) >= 1000:
            return f"{x:,.0f}"
        return f"{x:,.2f}" if not x.is_integer() else f"{x:,.0f}"
    except Exception:
        return str(x)

def to_num(series, default=0.0):
    """Convert to numeric safely."""
    s = pd.to_numeric(series, errors="coerce")
    return s.fillna(default)

def safe_bool01(series):
    """Map common labels (0/1, true/false, yes/no) to 0/1 numbers safely."""
    s = series.copy()
    if s.dtype.kind in "biufc":
        return (to_num(s) > 0.5).astype(int)
    s = s.astype(str).str.strip().str.lower()
    true_vals = {"1", "true", "t", "yes", "y"}
    return s.isin(true_vals).astype(int)

def confusion_counts(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    return tn, fp, fn, tp


# --------------------------------------------------------------------------------------
# Main app (all wrapped so exceptions are shown inside the app)
# --------------------------------------------------------------------------------------
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to:",
        ["Overview", "EDA Gallery", "Model Performance", "Score Explorer", "How to run"],
        index=0,
    )

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

    # ------------------------------ Overview -----------------------------------------
    if page == "Overview":
        st.title("Loan Serious-Late Prediction — Dashboard")

        kn = read_csv_safe(KEY_NUMBERS_CSV)
        total_clients = late_rate = median_income = None
        try:
            if kn is not None and not kn.empty:
                cols_lower = {c.lower(): c for c in kn.columns}

                def pick(col, fallback_idx):
                    c = cols_lower.get(col)
                    if c:
                        return kn.iloc[0][c]
                    idx = min(fallback_idx, len(kn.columns) - 1)
                    return kn.iloc[0][kn.columns[idx]]

                total_clients = pick("total_clients", 0)
                late_rate = pick("late_rate", 1)
                median_income = pick("median_income", 2)
        except Exception as e:
            st.info(f"key_numbers.csv present but unexpected format: {e}")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.caption("Total clients")
            st.metric(label="", value=metric_fmt(total_clients))
        with c2:
            st.caption("Late rate")
            try:
                v = float(late_rate)
                st.metric(label="", value=f"{v*100:.1f}%")
            except Exception:
                st.metric(label="", value="—")
        with c3:
            st.caption("Median income")
            st.metric(label="", value=metric_fmt(median_income))

        st.markdown("### What’s here")
        st.markdown(
            "- **EDA Gallery**: all charts saved from the analysis.\n"
            "- **Model Performance**: AUC/AP, thresholds, confusion matrices, lift/gains.\n"
            "- **Score Explorer**: interact with holdout probabilities.\n"
            "- **How to run**: simple instructions to reproduce."
        )

    # ------------------------------- EDA Gallery --------------------------------------
    elif page == "EDA Gallery":
        st.title("EDA Gallery")

        captions = read_figure_captions()
        pngs = sorted(glob.glob(join(FIG_DIR, "*.png")))
        if not pngs:
            st.info(f"No figures found in `{FIG_DIR}`. Generate them with `python main.py`.")
        else:
            for p in pngs:
                fname = os.path.basename(p)
                st.subheader(fname)
                cap = captions.get(fname)
                if cap:
                    st.caption(cap)
                show_image(p, caption=None, full=False)
                st.divider()

    # ---------------------------- Model Performance -----------------------------------
    elif page == "Model Performance":
        st.title("Model Performance")

        if not exists(MODEL_EVAL_CSV):
            st.warning(f"`{MODEL_EVAL_CSV}` is missing.")
            st.button("Generate analysis outputs (run main.py)")
        st.markdown("### Figures")

        figure_sets = [
            ("Confusion — Logistic Regression", ["confusion_logreg.png", "confusion_lr.png"]),
            ("Confusion — Random Forest", ["confusion_rf.png", "confusion_random_forest.png"]),
            ("ROC & PR curves", ["roc_pr_curves.png", "roc_curve.png", "pr_curve.png"]),
            ("Lift & Gain", ["lift_gain.png", "lift_curve.png", "gain_curve.png"]),
        ]
        for title, names in figure_sets:
            path = next((join(FIG_DIR, n) for n in names if exists(join(FIG_DIR, n))), None)
            if path:
                st.subheader(title)
                show_image(path, full=False)
                st.divider()

        eval_df = read_csv_safe(MODEL_EVAL_CSV)
        if eval_df is not None:
            st.subheader("Model evaluation summary")
            st.dataframe(eval_df, height=400)

        thr_df = read_csv_safe(THRESHOLD_METRICS_CSV)
        if thr_df is not None:
            st.subheader("Threshold sweep metrics")
            st.dataframe(thr_df, height=400)

    # ------------------------------- Score Explorer -----------------------------------
    elif page == "Score Explorer":
        st.title("Score Explorer (holdout set)")
        if not exists(HOLDOUT_PREDICTIONS_CSV):
            st.info(f"Run `python main.py` first to create `{HOLDOUT_PREDICTIONS_CSV}`.")
        else:
            df = read_csv_safe(HOLDOUT_PREDICTIONS_CSV)
            if df is None or df.empty:
                st.info("Holdout predictions file is empty or unreadable.")
            else:
                # Detect columns
                ycol = next((c for c in ["y", "target", "actual", "label", "serious_late"] if c in df.columns), None)
                score_cols = [c for c in df.columns if c.lower().startswith("p_") or c.lower().endswith("_prob") or c.lower().endswith("_score")]
                if not score_cols:
                    score_cols = [c for c in df.columns if c != ycol and df[c].dtype.kind in "fc"]

                if ycol is None or not score_cols:
                    st.warning("Could not identify label/score columns automatically.")
                    st.write("Columns:", list(df.columns))
                else:
                    model_col = st.selectbox("Score column", options=score_cols, index=0)
                    thr = st.slider("Decision threshold", 0.0, 1.0, 0.5, 0.01)

                    y_true = safe_bool01(df[ycol])
                    y_prob = to_num(df[model_col]).clip(0, 1)
                    y_pred = (y_prob >= thr).astype(int)

                    tn, fp, fn, tp = confusion_counts(y_true, y_pred)
                    acc = (tp + tn) / max(tn + fp + fn + tp, 1)
                    tpr = tp / max(tp + fn, 1)
                    fpr = fp / max(fp + tn, 1)
                    prec = tp / max(tp + fp, 1)

                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Accuracy", f"{acc*100:,.1f}%")
                    c2.metric("Recall (TPR)", f"{tpr*100:,.1f}%")
                    c3.metric("Fall-out (FPR)", f"{fpr*100:,.1f}%")
                    c4.metric("Precision", f"{prec*100:,.1f}%")

                    st.markdown("#### Confusion matrix (counts)")
                    st.table(pd.DataFrame(
                        {"": ["Pred 0", "Pred 1"], "True 0": [tn, fp], "True 1": [fn, tp]}
                    ))

                    # Optional deciles
                    dec_lr = read_csv_safe(SCORE_DECILES_LOGREG_CSV)
                    dec_rf = read_csv_safe(SCORE_DECILES_RF_CSV)
                    if dec_lr is not None or dec_rf is not None:
                        names, tabs = [], []
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

    # --------------------------------- How to run -------------------------------------
    elif page == "How to run":
        st.title("How to run")
        st.markdown("#### Local steps (Windows / PyCharm)")
        st.code(textwrap.dedent("""
            1) In Terminal, run:  python main.py
               - This generates figures and CSVs into reports/ and exports/

            2) Start the dashboard:
               python -m streamlit run dashboard.py

            3) Open the Local URL shown in the terminal (usually http://localhost:8501)
        """))
        st.markdown("### Files generated by the analysis")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**reports/**")
            st.code("\n".join([
                "reports/run_environment.txt",
                "reports/cleaning_report.txt",
                "reports/figure_captions.txt",
                "reports/key_numbers.csv",
                "reports/figures/*.png",
            ]), language="text")
        with c2:
            st.markdown("**exports/**")
            st.code("\n".join([
                "exports/model_eval_summary.csv",
                "exports/holdout_predictions.csv",
                "exports/threshold_metrics.csv",
                "exports/score_deciles_logreg.csv",
                "exports/score_deciles_rf.csv",
            ]), language="text")


# Run safely so any unexpected error is shown inside the app (not white screen)
try:
    main()
except Exception as _e:
    st.error("An unexpected error occurred in the app. Details:")
    st.exception(_e)
