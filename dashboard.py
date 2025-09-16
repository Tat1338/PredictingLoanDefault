# dashboard.py
# Streamlit dashboard for the Loan Default Exam Project
# - Uses use_container_width (no deprecation warnings)
# - Optional one-click generator to run main.py on Streamlit Cloud

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

def show_table(df: pd.DataFrame | None, caption: str | None = None, height: int = 360):
    if df is None or df.empty:
        st.info("No data found for this section.")
        return
    st.dataframe(df, height=height, use_container_width=False)
    if caption:
        st.caption(caption)

def find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def safe_div(n: float, d: float) -> float:
    return (n / d) if d else np.nan

def offer_generator(note: str = ""):
    """Show a button that runs main.py to generate reports/ and exports/."""
    st.warning(note or "Required files are missing. You can generate them here.")
    if st.button("Generate analysis outputs (run main.py)", type="primary"):
        import subprocess, sys
        with st.status("Running main.py…", expanded=True) as status:
            try:
                # Run the analysis script with the same Python interpreter
                proc = subprocess.run(
                    [sys.executable, "main.py"],
                    capture_output=True, text=True, check=True
                )
                st.write(proc.stdout[-4000:] or "(no stdout)")
                if proc.stderr:
                    st.write("stderr:")
                    st.code(proc.stderr[-2000:], language="text")
                status.update(label="Done", state="complete")
            except subprocess.CalledProcessError as e:
                st.error("main.py failed. See logs below.")
                st.code((e.stdout or "")[-2000:] + "\n" + (e.stderr or "")[-2000:], language="text")
                return
        st.success("Files generated. Reloading…")
        st.rerun()

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
            st.metric("Late rate", f"{rr:.1%}" if pd.notna(rr) else "—")
        with col3:
            med_inc = row.get("median_income_all", np.nan)
            st.metric("Median income", f"{med_inc:,.0f}" if pd.notna(med_inc) else "—")
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
                # ✅ modern, no deprecation
                st.image(img, use_container_width=True)
                st.divider()

elif page == "Model Performance":
    st.title("Model Performance")

    perf = safe_read_csv(os.path.join(EXPORTS_DIR, "model_eval_summary.csv"))
    if perf is not None and not perf.empty:
        st.markdown("**Holdout metrics**")
        show_table(perf, height=260)
        st.download_button(
            "Download metrics (CSV)",
            data=perf.to_csv(index=False).encode("utf-8"),
            file_name="model_eval_summary.csv",
            mime="text/csv",
            use_container_width=True,
        )
    else:
        offer_generator("`exports/model_eval_summary.csv` is missing.")

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
            st.image(path, use_container_width=True)  # ✅ modern
        else:
            st.caption(f"Missing: {preferred if preferred else ''}")

elif page == "Score Explorer":
    st.title("Score Explorer (holdout set)")

    holdout = safe_read_csv(os.path.join(EXPORTS_DIR, "holdout_predictions.csv"))
    thr_metrics = safe_read_csv(os.path.join(EXPORTS_DIR, "threshold_metrics.csv"))

    if holdout is None or holdout.empty:
        offer_generator("`exports/holdout_predictions.csv` is missing.")
    else:
        st.markdown("Use the threshold slider to see how many would be flagged.")
        model_map = {}
        if "proba_logreg" in holdout.columns:
            model_map["Logistic Regression"] = "proba_logreg"
        if "proba_rf" in holdout.columns:
            model_map["Random Forest"] = "proba_rf"
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

            probs = holdout[prob_col].to_numpy()
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

            hist, edges = np.histogram(probs, bins=30, range=(0, 1))
            st.bar_chart(pd.DataFrame({"count": hist}, index=pd.Index(edges[:-1], name="p")), height=240)

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

            if thr_metrics is not None and not thr_metrics.empty and "threshold" in thr_metrics.columns:
                nearest = thr_metrics.iloc[(thr_metrics["threshold"] - thr).abs().argsort()[:1]]
                st.markdown("##### Nearest precomputed row from `threshold_metrics.csv`")
                show_table(nearest, height=120)

            st.markdown("#### Top scores")
            top_n = st.number_input("Show top N scores", min_value=5, max_value=200, value=20, step=5)
            top_df = pd.DataFrame({"probability": probs}).sort_values("probability", ascending=False).head(int(top_n))
            show_table(top_df, caption="Highest-risk holdout predictions", height=300)

            st.download_button(
                "Download holdout with scores (CSV)",
                data=holdout.to_csv(index=False).encode("utf-8"),
                file_name="holdout_predictions.csv",
                mime="text/csv",
                use_container_width=True,
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
        "requirements.txt  # include: streamlit, pandas, numpy, scikit-learn, matplotlib, seaborn (if used)\n"
        ".streamlit/config.toml  # optional theming\n"
        "Repo contains: main.py, dashboard.py, /reports, /exports (or code that creates them)\n"
        "App entry point: dashboard.py\n",
        language="text"
    )

    st.info("Tip: You can also generate files from the buttons on 'Model Performance' or 'Score Explorer'.")
