# dashboard.py
# Streamlit dashboard for the Loan Default Exam Project
# - Clean widths (no use_container_width warnings)
# - Inline threshold metrics in Score Explorer
# - Gentle fallbacks + downloads (no width warnings)

from __future__ import annotations

import os
import glob
import textwrap
from typing import Optional

import pandas as pd
import numpy as np
import streamlit as st

FIG_DIR = "reports/figures"
EXPORTS_DIR = "exports"

st.set_page_config(page_title="Loan Default — Dashboard", layout="wide")

# ---------- helpers ----------
def exists(p: str) -> bool:
    return os.path.exists(p)

@st.cache_data(show_spinner=False)
def safe_read_csv(path: str) -> Optional[pd.DataFrame]:
    try:
        if exists(path):
            return pd.read_csv(path)
    except Exception:
        return None
    return None

def show_table(df: Optional[pd.DataFrame], caption: str | None = None, height: int = 360):
    if df is None or df.empty:
        st.info("No data found for this section.")
        return
    # Avoid deprecated/changed width flags
    st.dataframe(df, height=height)
    if caption:
        st.caption(caption)

def find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first existing column from candidates."""
    for c in candidates:
        if c in df.columns:
            return c
    return None

def safe_div(n: float, d: float) -> float:
    return (n / d) if d else np.nan

# ---------- sidebar ----------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to:",
    ["Overview", "EDA Gallery", "Model Performance", "Score Explorer", "How to run"]
)

st.sidebar.divider()
st.sidebar.markdown("**Files this app expects:**")
st.sidebar.code(
    "reports/figures/*.png\n"
    "reports/key_numbers.csv\n"
    "exports/holdout_predictions.csv\n"
    "exports/model_eval_summary.csv\n"
    "exports/threshold_metrics.csv\n"
    "exports/score_deciles_logreg.csv\n"
    "exports/score_deciles_rf.csv",
    language="text"
)

# ---------- pages ----------
if page == "Overview":
    st.title("Loan Serious-Late Prediction — Dashboard")

    col1, col2, col3 = st.columns(3)
    key_numbers = safe_read_csv("reports/key_numbers.csv")
    if key_numbers is not None and not key_numbers.empty:
        row = key_numbers.iloc[0].to_dict()
        with col1:
            st.metric("Total clients", f"{int(row.get('total_clients', 0)):,}")
        with col2:
            rr = row.get("late_rate", np.nan)
            st.metric("Late rate", f"{float(rr):.1%}" if pd.notna(rr) else "—")
        with col3:
            med_inc = row.get("median_income_all", np.nan)
            st.metric("Median income", f"{float(med_inc):,.0f}" if pd.notna(med_inc) else "—")
    else:
        with col1: st.metric("Total clients", "—")
        with col2: st.metric("Late rate", "—")
        with col3: st.metric("Median income", "—")

    st.markdown("### What’s here")
    st.markdown(
        "- **EDA Gallery**: all charts saved from the analysis.\n"
        "- **Model Performance**: AUC/AP, thresholds, confusion matrices, lift/gains.\n"
        "- **Score Explorer**: interact with holdout probabilities.\n"
        "- **How to run**: simple instructions to reproduce."
    )

elif page == "EDA Gallery":
    st.title("EDA Gallery")
    if not os.path.isdir(FIG_DIR):
        st.warning(f"Folder not found: `{FIG_DIR}`")
    else:
        images = sorted(glob.glob(os.path.join(FIG_DIR, "*.png")))
        if not images:
            st.info("No figures found yet. Run `python main.py` to generate them.")
        else:
            # Optional captions file
            captions = {}
            cap_path = os.path.join("reports", "figure_captions.txt")
            if exists(cap_path):
                try:
                    with open(cap_path, "r", encoding="utf-8") as f:
                        for line in f:
                            if ": " in line and not line.strip().startswith("==="):
                                k, v = line.strip().split(": ", 1)
                                captions[k] = v
                except Exception:
                    pass

            for img in images:
                fname = os.path.basename(img)
                cap = captions.get(fname, "")
                st.subheader(fname)
                if cap:
                    st.caption(cap)
                # IMPORTANT: width must be 'stretch', 'content', or an int; never None.
                st.image(img, caption=None, width="stretch")  # responsive, no warnings
                st.divider()

elif page == "Model Performance":
    st.title("Model Performance")

    # Top metrics table
    perf = safe_read_csv(os.path.join(EXPORTS_DIR, "model_eval_summary.csv"))
    if perf is not None and not perf.empty:
        st.markdown("**Holdout metrics**")
        show_table(perf, height=260)

        # Small summary KPIs
        c1, c2 = st.columns(2)
        with c1:
            if "AUC" in perf.columns and "model" in perf.columns:
                auc_txt = " • ".join(f"{m}: {a:.3f}" for m, a in zip(perf["model"], perf["AUC"]))
                st.success(f"AUC — {auc_txt}")
        with c2:
            if "AP" in perf.columns and "model" in perf.columns:
                ap_txt = " • ".join(f"{m}: {a:.3f}" for m, a in zip(perf["model"], perf["AP"]))
                st.info(f"Average Precision — {ap_txt}")
    else:
        st.info("Run `python main.py` to generate `exports/model_eval_summary.csv`.")

    # Confusion matrices & curves if the PNGs exist
    figs = [
        ("Confusion — Logistic Regression", "cm_logreg_tuned.png", "cm_logreg.png"),
        ("Confusion — Random Forest", "cm_rf_tuned.png", "cm_rf.png"),
        ("ROC — Logistic Regression", "roc_logreg.png", None),
        ("ROC — Random Forest", "roc_rf.png", None),
        ("PR — Logistic Regression", "pr_logreg.png", None),
        ("PR — Random Forest", "pr_rf.png", None),
        ("Lift — Logistic Regression", "lift_logreg.png", None),
        ("Lift — Random Forest", "lift_rf.png", None),
        ("Gains — Logistic Regression", "gains_logreg.png", None),
        ("Gains — Random Forest", "gains_rf.png", None),
        ("Calibration — Logistic Regression", "calibration_logreg.png", None),
        ("Permutation importance — LogReg", "pi_logreg.png", None),
        ("Permutation importance — RF", "pi_rf.png", None),
    ]
    st.markdown("### Figures")
    for title, preferred, fallback in figs:
        path = os.path.join(FIG_DIR, preferred)
        if not exists(path) and fallback:
            path = os.path.join(FIG_DIR, fallback)
        if exists(path):
            st.subheader(title)
            st.image(path, width="stretch")  # responsive, no warnings
        else:
            st.caption(f"Missing: {preferred if preferred else ''}")

elif page == "Score Explorer":
    st.title("Score Explorer (holdout set)")

    holdout = safe_read_csv(os.path.join(EXPORTS_DIR, "holdout_predictions.csv"))
    thr_metrics = safe_read_csv(os.path.join(EXPORTS_DIR, "threshold_metrics.csv"))

    if holdout is None or holdout.empty:
        st.info("Run `python main.py` first to create `exports/holdout_predictions.csv`.")
    else:
        st.markdown("Use the threshold slider to see how many would be flagged.")

        # Detect available proba columns
        model_map: dict[str, str] = {}
        if "proba_logreg" in holdout.columns:
            model_map["Logistic Regression"] = "proba_logreg"
        if "proba_rf" in holdout.columns:
            model_map["Random Forest"] = "proba_rf"
        # Fallback: any column starting with 'proba'
        if not model_map:
            for c in holdout.columns:
                if c.lower().startswith("proba"):
                    model_map[c] = c

        if not model_map:
            st.error("No probability columns found (expected e.g. 'proba_logreg', 'proba_rf').")
        else:
            model_choice = st.selectbox("Model", list(model_map.keys()))
            prob_col = model_map[model_choice]
            thr = st.slider("Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

            probs = holdout[prob_col].to_numpy(dtype=float)
            flagged = int((probs >= thr).sum())
            total = len(probs)

            cA, cB, cC = st.columns(3)
            with cA:
                st.metric("Flagged (>= threshold)", f"{flagged:,}")
                st.caption(f"Out of {total:,} holdout cases")
            with cB:
                st.metric("Mean score", f"{np.nanmean(probs):.3f}")
            with cC:
                st.metric("Std. dev. score", f"{np.nanstd(probs):.3f}")

            # quick histogram
            hist, edges = np.histogram(probs, bins=30, range=(0, 1))
            st.bar_chart(pd.DataFrame({"count": hist}, index=pd.Index(edges[:-1], name="p")), height=240)

            # Inline metrics if we have y_true
            y_col = find_column(holdout, ["y_true", "target", "SeriousDlqin2yrs"])
            if y_col is not None:
                y = holdout[y_col].astype(int).to_numpy()
                preds = (probs >= thr).astype(int)

                tp = int(((preds == 1) & (y == 1)).sum())
                tn = int(((preds == 0) & (y == 0)).sum())
                fp = int(((preds == 1) & (y == 0)).sum())
                fn = int(((preds == 0) & (y == 1)).sum())

                precision = safe_div(tp, tp + fp)
                recall = safe_div(tp, tp + fn)
                specificity = safe_div(tn, tn + fp)
                accuracy = safe_div(tp + tn, total)
                f1 = safe_div(2 * precision * recall, precision + recall)

                st.markdown("#### Threshold metrics (on holdout)")
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Precision", f"{precision:.3f}" if pd.notna(precision) else "—")
                m2.metric("Recall (TPR)", f"{recall:.3f}" if pd.notna(recall) else "—")
                m3.metric("Specificity (TNR)", f"{specificity:.3f}" if pd.notna(specificity) else "—")
                m4.metric("Accuracy", f"{accuracy:.3f}" if pd.notna(accuracy) else "—")
                m5.metric("F1", f"{f1:.3f}" if pd.notna(f1) else "—")

                st.caption(f"Confusion matrix @ {thr:.2f}: TP={tp:,}, FP={fp:,}, TN={tn:,}, FN={fn:,}")

            # If precomputed threshold_metrics.csv exists, show nearest row for current thr
            if thr_metrics is not None and not thr_metrics.empty and "threshold" in thr_metrics.columns:
                nearest = thr_metrics.iloc[(thr_metrics["threshold"] - thr).abs().argsort()[:1]]
                st.markdown("##### Nearest precomputed row from `threshold_metrics.csv`")
                show_table(nearest, height=120)

            # Show a few top-scored rows
            st.markdown("#### Top scores")
            top_n = st.number_input("Show top N scores", min_value=5, max_value=200, value=20, step=5)
            top_df = pd.DataFrame({"probability": probs}).sort_values("probability", ascending=False).head(int(top_n))
            show_table(top_df, caption="Highest-risk holdout predictions", height=300)

            # Download (omit width/use_container_width to avoid warnings)
            st.download_button(
                "Download holdout with scores (CSV)",
                data=holdout.to_csv(index=False).encode("utf-8"),
                file_name="holdout_predictions.csv",
                mime="text/csv",
            )

elif page == "How to run":
    st.title("How to run")

    st.markdown("**Local steps (Windows / PyCharm)**")
    st.code(
        "1) In Terminal, run: python main.py\n"
        "   - This generates figures and CSVs into reports/ and exports/\n"
        "2) Start the dashboard:\n"
        "   python -m streamlit run dashboard.py\n"
        "3) Open the Local URL shown in the terminal (usually http://localhost:8501)\n",
        language="bash"
    )

    st.markdown("**Files generated by the analysis**")
    colA, colB = st.columns(2)
    with colA:
        st.write("**reports/**")
        st.code(
            "reports/run_environment.txt\n"
            "reports/cleaning_report.txt\n"
            "reports/figure_captions.txt\n"
            "reports/key_numbers.csv\n"
            "reports/figures/*.png\n",
            language="text"
        )
    with colB:
        st.write("**exports/**")
        st.code(
            "exports/model_eval_summary.csv\n"
            "exports/holdout_predictions.csv\n"
            "exports/threshold_metrics.csv\n"
            "exports/score_deciles_logreg.csv\n"
            "exports/score_deciles_rf.csv\n",
            language="text"
        )

    st.markdown("**Deploy on Streamlit Community Cloud / GitHub**")
    st.code(
        textwrap.dedent(
            """
            requirements.txt  # include: streamlit, pandas, numpy, scikit-learn, matplotlib, seaborn (if used)
            .streamlit/config.toml  # optional theming
            Repo contains: main.py, dashboard.py, /reports, /exports (or code that creates them)
            App entry point: dashboard.py
            """
        ),
        language="text"
    )

    st.info("Tip: If you change code, just refresh the Streamlit page or rerun the command.")
