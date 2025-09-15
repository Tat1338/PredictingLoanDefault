# dashboard.py
# Streamlit dashboard for the Loan Default Exam Project
# - Uses width="stretch" (no use_container_width warnings)

import os
import glob
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
def safe_read_csv(path: str) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path)
    except Exception:
        return None

def show_table(df: pd.DataFrame, caption: str | None = None, height: int = 360):
    if df is None or df.empty:
        st.info("No data found for this section.")
        return
    st.dataframe(df, height=height, use_container_width=False)  # width is controlled by columns
    if caption:
        st.caption(caption)

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
    "exports/holdout_predictions.csv\n"
    "exports/model_eval_summary.csv\n"
    "exports/threshold_metrics.csv",
    language="text"
)

# ---------- pages ----------
if page == "Overview":
    st.title("Loan Serious-Late Prediction — Dashboard")

    col1, col2, col3 = st.columns(3)
    # Key numbers (if available)
    key_numbers = safe_read_csv("reports/key_numbers.csv")
    if key_numbers is not None and not key_numbers.empty:
        row = key_numbers.iloc[0].to_dict()
        with col1:
            st.metric("Total clients", f"{int(row.get('total_clients', 0)):,}")
        with col2:
            rr = row.get("late_rate", np.nan)
            st.metric("Late rate", f"{rr:.1%}" if pd.notna(rr) else "—")
        with col3:
            st.metric("Median income", f"{row.get('median_income_all', float('nan')):,.0f}")
    else:
        with col1:
            st.metric("Total clients", "—")
        with col2:
            st.metric("Late rate", "—")
        with col3:
            st.metric("Median income", "—")

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
                st.image(img, caption=None, width=None)  # draws at natural width
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
            st.image(path, width=None)
        else:
            st.caption(f"Missing: {preferred if preferred else ''}")

elif page == "Score Explorer":
    st.title("Score Explorer (holdout set)")

    holdout = safe_read_csv(os.path.join(EXPORTS_DIR, "holdout_predictions.csv"))
    if holdout is None or holdout.empty:
        st.info("Run `python main.py` first to create `exports/holdout_predictions.csv`.")
    else:
        st.markdown("Use the threshold slider to see how many would be flagged.")
        model_choice = st.selectbox("Model", ["Logistic Regression", "Random Forest"])
        thr = st.slider("Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

        prob_col = "proba_logreg" if "logreg" in model_choice.lower() else "proba_rf"
        if prob_col not in holdout.columns:
            st.error(f"Column not found: {prob_col}")
        else:
            probs = holdout[prob_col].values
            flagged = int((probs >= thr).sum())
            total = len(probs)
            st.metric("Flagged (>= threshold)", f"{flagged:,}", delta=None)
            st.caption(f"Out of {total:,} holdout cases")

            # quick histogram
            hist, edges = np.histogram(probs, bins=30, range=(0, 1))
            st.bar_chart(pd.DataFrame({"count": hist}, index=pd.Index(edges[:-1], name="p")), height=240)

            # show a few top-scored rows
            top_n = st.number_input("Show top N scores", min_value=5, max_value=200, value=20, step=5)
            top_df = pd.DataFrame({"probability": probs}).sort_values("probability", ascending=False).head(int(top_n))
            show_table(top_df, caption="Highest-risk holdout predictions", height=300)

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

    st.info("Tip: If you change code, just refresh the Streamlit page or rerun the command.")
