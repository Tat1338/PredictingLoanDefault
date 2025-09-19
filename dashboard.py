# dashboard.py
# Streamlit dashboard for the Loan Default Exam Project
# - Same layout you had
# - Beginner-friendly, *result-focused* analysis under each chart/section
# - Robust to missing files (skips gracefully)

from __future__ import annotations

import os
import glob
import textwrap
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

FIG_DIR = "reports/figures"
EXPORTS_DIR = "exports"

st.set_page_config(page_title="Loan Default Prediction — Dashboard", layout="wide")


# ---------- small utils ----------
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
    st.dataframe(df, height=height)
    if caption:
        st.caption(caption)

def find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def safe_div(n: float, d: float) -> float:
    return float(n) / float(d) if d else np.nan

def num(x: float | int | None, d: int = 2) -> str:
    if x is None or pd.isna(x): return "—"
    try:
        return f"{float(x):.{d}f}"
    except Exception:
        return str(x)

def pct(x: float | int | None, d: int = 1) -> str:
    if x is None or pd.isna(x): return "—"
    try:
        return f"{100.0 * float(x):.{d}f}%"
    except Exception:
        return "—"

def ks_statistic(y: np.ndarray, s: np.ndarray) -> float:
    """Two-sample KS on [0,1] scores."""
    pos = s[y == 1]
    neg = s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    xp = np.sort(pos)
    xn = np.sort(neg)
    grid = np.linspace(0.0, 1.0, 1001)
    cdf_p = np.searchsorted(xp, grid, side="right") / xp.size
    cdf_n = np.searchsorted(xn, grid, side="right") / xn.size
    return float(np.max(np.abs(cdf_p - cdf_n)))

# ---------- sidebar ----------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to:",
    ["Overview", "EDA Gallery", "Model Performance", "Score Explorer", "Executive Summary", "How to run"]
)


st.sidebar.divider()
st.sidebar.markdown("**Files this app expects:**")
st.sidebar.code(
    "reports/figures/*.png\n"
    "reports/key_numbers.csv\n"
    "exports/model_eval_summary.csv\n"
    "exports/holdout_predictions.csv\n"
    "exports/threshold_metrics.csv\n"
    "exports/score_deciles_logreg.csv\n"
    "exports/score_deciles_rf.csv",
    language="text"
)

# ---------- pages ----------
if page == "Overview":
    st.title("Loan Default Prediction — Dashboard")

    col1, col2, col3 = st.columns(3)
    key_numbers = safe_read_csv("reports/key_numbers.csv")
    if key_numbers is not None and not key_numbers.empty:
        row = key_numbers.iloc[0].to_dict()
        with col1:
            st.metric("Total clients", f"{int(row.get('total_clients', 0)):,}")
        with col2:
            rr = row.get("late_rate", np.nan)
            st.metric("Late rate", pct(rr))
        with col3:
            med_inc = row.get("median_income_all", np.nan)
            st.metric("Median income", f"{float(med_inc):,.0f}" if pd.notna(med_inc) else "—")

        # Analysis paragraph if extra fields exist
        pieces = []
        if "median_income_all" in row:
            pieces.append(f"Median income in the portfolio sits at **{float(row['median_income_all']):,.0f}**.")
        if "min_income_all" in row and "max_income_all" in row:
            pieces.append(
                f"Reported income ranges from **{float(row['min_income_all']):,.0f}** to **{float(row['max_income_all']):,.0f}**."
            )
        if "late_rate" in row:
            pieces.append(f"Overall late/default rate is **{pct(row['late_rate'])}**.")
        if "default_rate_no_dependents" in row and "default_rate_with_dependents" in row:
            nd = pct(row["default_rate_no_dependents"])
            wd = pct(row["default_rate_with_dependents"])
            pieces.append(f"Clients **without dependants** show a default rate of **{nd}**, versus **{wd}** with dependants.")

        if pieces:
            st.markdown(
                "#### Summary insight\n" + " ".join(pieces)
            )
    else:
        with col1: st.metric("Total clients", "—")
        with col2: st.metric("Late rate", "—")
        with col3: st.metric("Median income", "—")

    st.markdown("### What’s here")
    st.markdown(
        "- **EDA Gallery**: figures from the analysis.\n"
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
            # Optional captions authored by you
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
                st.subheader(fname)
                cap = captions.get(fname, "")
                if cap:
                    st.caption(cap)  # your own narrative if present
                st.image(img, caption=None, width="stretch")
                # If you want computed commentary here, export per-figure stats to a CSV and print them.
                st.divider()

elif page == "Model Performance":
    st.title("Model Performance")

    perf = safe_read_csv(os.path.join(EXPORTS_DIR, "model_eval_summary.csv"))
    holdout = safe_read_csv(os.path.join(EXPORTS_DIR, "holdout_predictions.csv"))
    thr_metrics = safe_read_csv(os.path.join(EXPORTS_DIR, "threshold_metrics.csv"))
    base_rate = None
    if holdout is not None:
        y_col = find_column(holdout, ["y_true", "target", "SeriousDlqin2yrs"])
        if y_col:
            base_rate = float(pd.to_numeric(holdout[y_col], errors="coerce").mean())

    # --- metrics table
    if perf is not None and not perf.empty:
        st.markdown("**Holdout metrics**")
        show_table(perf, height=260)

        # AUC/AP ribbons
        c1, c2 = st.columns(2)
        with c1:
            if "AUC" in perf.columns and "model" in perf.columns:
                auc_txt = " • ".join(f"{m}: {float(a):.3f}" for m, a in zip(perf["model"], perf["AUC"]))
                st.success(f"AUC — {auc_txt}")
        with c2:
            if "AP" in perf.columns and "model" in perf.columns:
                ap_txt = " • ".join(f"{m}: {float(a):.3f}" for m, a in zip(perf["model"], perf["AP"]))
                st.info(f"Average Precision — {ap_txt}")

        # --- ANALYSIS: who is best and by how much
        msgs = []
        if "AP" in perf.columns:
            best_ap_idx = int(perf["AP"].astype(float).idxmax())
            best_ap_row = perf.iloc[best_ap_idx]
            best_ap_name = str(best_ap_row.get("model", "Best"))
            best_ap = float(best_ap_row["AP"])
            second_ap = float(perf["AP"].nlargest(2).iloc[-1]) if len(perf) >= 2 else np.nan
            lift_over_base = (best_ap / base_rate) if base_rate else np.nan
            msgs.append(
                f"**Ranking by AP:** {best_ap_name} leads with **AP {best_ap:.3f}** "
                + (f"(next best {second_ap:.3f}). " if pd.notna(second_ap) else ". ")
                + (f"Given a portfolio base default rate of **{pct(base_rate)}**, the model’s AP represents about **{num(lift_over_base,2)}×** the random baseline." if base_rate else "")
            )
        if "AUC" in perf.columns:
            best_auc_idx = int(perf["AUC"].astype(float).idxmax())
            best_auc_row = perf.iloc[best_auc_idx]
            best_auc_name = str(best_auc_row.get("model", "Best"))
            best_auc = float(best_auc_row["AUC"])
            msgs.append(f"**Ranking by AUC:** {best_auc_name} shows the strongest overall ranking (**AUC {best_auc:.3f}**).")

        if msgs:
            st.markdown("#### What the metrics say\n" + " ".join(msgs))
    else:
        st.info("Run `python main.py` to generate `exports/model_eval_summary.csv`.")

    # --- Figures with commentary (use CSVs when available)
    figs = [
        ("Confusion — Logistic Regression", "cm_logreg_tuned.png", "cm_logreg.png", "Logistic Regression"),
        ("Confusion — Random Forest", "cm_rf_tuned.png", "cm_rf.png", "Random Forest"),
        ("ROC — Logistic Regression", "roc_logreg.png", None, "Logistic Regression"),
        ("ROC — Random Forest", "roc_rf.png", None, "Random Forest"),
        ("PR — Logistic Regression", "pr_logreg.png", None, "Logistic Regression"),
        ("PR — Random Forest", "pr_rf.png", None, "Random Forest"),
        ("Lift — Logistic Regression", "lift_logreg.png", None, "Logistic Regression"),
        ("Lift — Random Forest", "lift_rf.png", None, "Random Forest"),
        ("Gains — Logistic Regression", "gains_logreg.png", None, "Logistic Regression"),
        ("Gains — Random Forest", "gains_rf.png", None, "Random Forest"),
        ("Calibration — Logistic Regression", "calibration_logreg.png", None, "Logistic Regression"),
        ("Permutation importance — LogReg", "pi_logreg.png", None, "Logistic Regression"),
        ("Permutation importance — RF", "pi_rf.png", None, "Random Forest"),
    ]

    # helper to write per-model paragraph using threshold_metrics or deciles
    def write_model_paragraph(model_name: str):
        texts = []
        # From threshold_metrics.csv (if present and has rows per model)
        if thr_metrics is not None and not thr_metrics.empty:
            # Try to choose a "best" row per model (F1 max, else recall max, else first)
            dfm = thr_metrics.copy()
            if "model" in dfm.columns:
                dfm = dfm[dfm["model"].astype(str).str.lower() == model_name.lower()]
            if not dfm.empty:
                cand_cols = ["F1", "f1", "Recall", "recall"]
                use_col = next((c for c in cand_cols if c in dfm.columns), None)
                if use_col:
                    row = dfm.sort_values(use_col, ascending=False).iloc[0]
                else:
                    row = dfm.iloc[0]
                tp = int(row.get("tp", np.nan)) if pd.notna(row.get("tp", np.nan)) else None
                fp = int(row.get("fp", np.nan)) if pd.notna(row.get("fp", np.nan)) else None
                tn = int(row.get("tn", np.nan)) if pd.notna(row.get("tn", np.nan)) else None
                fn = int(row.get("fn", np.nan)) if pd.notna(row.get("fn", np.nan)) else None
                prec = row.get("precision") if "precision" in row else row.get("Precision")
                rec = row.get("recall") if "recall" in row else row.get("Recall")
                thr = row.get("threshold") if "threshold" in row else None

                line = f"At a working cutoff {num(thr,2) if thr is not None else ''} the confusion mix is "
                parts = []
                if tp is not None: parts.append(f"TP={tp:,}")
                if fp is not None: parts.append(f"FP={fp:,}")
                if tn is not None: parts.append(f"TN={tn:,}")
                if fn is not None: parts.append(f"FN={fn:,}")
                if parts: line += ", ".join(parts) + ". "
                if pd.notna(prec): line += f"Precision **{num(prec,3)}**. "
                if pd.notna(rec):  line += f"Recall **{num(rec,3)}**. "
                if base_rate is not None and pd.notna(rec):
                    line += f"Base default rate is **{pct(base_rate)}**, so recall at this cutoff captures that share of positives."
                texts.append(line)

        # From score deciles (capture in top buckets) if present
        for fname in ["score_deciles_logreg.csv", "score_deciles_rf.csv"]:
            p = os.path.join(EXPORTS_DIR, fname)
            if exists(p):
                dec = safe_read_csv(p)
                if dec is not None and not dec.empty:
                    # expected columns: decile (1..10), capture_rate or cum_capture_rate, defaults, count
                    if "decile" in dec.columns:
                        d1 = dec.sort_values("decile").iloc[0]
                        # try both names
                        cap1 = d1.get("capture_rate", d1.get("cum_capture_rate", np.nan))
                        if pd.notna(cap1):
                            texts.append(f"The top **10%** of scores contain about **{pct(cap1)}** of all defaulters, "
                                         f"which is a strong prioritisation band for manual review.")
                        break

        if texts:
            st.markdown("**What this figure tells us:** " + " ".join(texts))

    st.markdown("### Figures")
    for title, preferred, fallback, model_name in figs:
        path = os.path.join(FIG_DIR, preferred)
        if not exists(path) and fallback:
            path = os.path.join(FIG_DIR, fallback)
        if exists(path):
            st.subheader(title)
            st.image(path, width="stretch")
            write_model_paragraph(model_name)
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

        # Detect probability columns
        model_map: dict[str, str] = {}
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

            scores = pd.to_numeric(holdout[prob_col], errors="coerce").clip(0, 1).to_numpy()
            flagged = int((scores >= thr).sum())
            total = int(np.isfinite(scores).sum())

            cA, cB, cC = st.columns(3)
            with cA:
                st.metric("Flagged (>= threshold)", f"{flagged:,}")
                st.caption(f"Out of {total:,} holdout cases")
            with cB:
                st.metric("Mean score", num(np.nanmean(scores), 3))
            with cC:
                st.metric("Std. dev. score", num(np.nanstd(scores), 3))

            # Histogram of scores
            hist, edges = np.histogram(scores, bins=30, range=(0, 1))
            st.bar_chart(pd.DataFrame({"count": hist}, index=pd.Index(edges[:-1], name="p")), height=240)

            # Compute analytics with labels if available
            y_col = find_column(holdout, ["y_true", "target", "SeriousDlqin2yrs"])
            if y_col is not None:
                y = pd.to_numeric(holdout[y_col], errors="coerce").fillna(0).astype(int).to_numpy()
                preds = (scores >= thr).astype(int)

                tp = int(((preds == 1) & (y == 1)).sum())
                tn = int(((preds == 0) & (y == 0)).sum())
                fp = int(((preds == 1) & (y == 0)).sum())
                fn = int(((preds == 0) & (y == 1)).sum())

                precision = safe_div(tp, tp + fp)
                recall = safe_div(tp, tp + fn)
                specificity = safe_div(tn, tn + fp)
                accuracy = safe_div(tp + tn, len(y))
                f1 = safe_div(2 * precision * recall, precision + recall)

                st.markdown("#### Threshold metrics (on holdout)")
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Precision", num(precision, 3))
                m2.metric("Recall (TPR)", num(recall, 3))
                m3.metric("Specificity (TNR)", num(specificity, 3))
                m4.metric("Accuracy", num(accuracy, 3))
                m5.metric("F1", num(f1, 3))

                st.caption(f"Confusion @ {thr:.2f}: TP={tp:,}, FP={fp:,}, TN={tn:,}, FN={fn:,}")

                # ---- ANALYSIS: what these numbers mean
                base = float(y.mean()) if len(y) else np.nan
                share_flagged = flagged / total if total else np.nan
                mean_pos = float(np.nanmean(scores[y == 1])) if (y == 1).any() else np.nan
                mean_neg = float(np.nanmean(scores[y == 0])) if (y == 0).any() else np.nan
                ks = ks_statistic(y, scores)

                st.markdown(
                    textwrap.dedent(
                        f"""
                        **What the distribution and cutoff are telling us:**  
                        • About **{pct(share_flagged)}** of the portfolio would be flagged at this cutoff (**{thr:.2f}**).  
                        • Portfolio default rate is **{pct(base)}**. Scores average **{num(mean_pos,3)}** for defaulters vs **{num(mean_neg,3)}** for non-defaulters.  
                        • Separation (KS) is **{num(ks,3)}** — values around 0.3–0.5 indicate useful rank ordering for credit risk.  
                        • With precision **{num(precision,3)}** and recall **{num(recall,3)}**, this setting balances how many risky cases we catch versus the extra reviews created. Shift the slider left for more coverage (higher recall) or right for fewer false alarms (higher precision).
                        """
                    ).strip()
                )

            # Show nearest precomputed row if available
            if thr_metrics is not None and not thr_metrics.empty and "threshold" in thr_metrics.columns:
                nearest = thr_metrics.iloc[(thr_metrics["threshold"] - thr).abs().argsort()[:1]]
                st.markdown("##### Nearest precomputed row from `threshold_metrics.csv`")
                show_table(nearest, height=120)

            # Top scores table
            st.markdown("#### Top scores")
            top_n = st.number_input("Show top N scores", min_value=5, max_value=200, value=20, step=5)
            top_df = pd.DataFrame({"probability": scores}).sort_values("probability", ascending=False).head(int(top_n))
            show_table(top_df, caption="Highest-risk holdout predictions", height=300)

            st.download_button(
                "Download holdout with scores (CSV)",
                data=holdout.to_csv(index=False).encode("utf-8"),
                file_name="holdout_predictions.csv",
                mime="text/csv",
            )
elif page == "Executive Summary":
    md_path = os.path.join("reports", "executive_summary.md")
    if exists(md_path):
        # Read and drop a leading "# Executive Summary" line if present
        with open(md_path, "r", encoding="utf-8") as f:
            content = f.read().lstrip("\ufeff")
        lines = content.splitlines()
        if lines and lines[0].strip().lower().startswith("# executive summary"):
            content = "\n".join(lines[1:]).lstrip()

        st.title("Executive Summary")
        st.markdown(content)

        with open(md_path, "rb") as fh:
            st.download_button(
                "Download executive_summary.md",
                data=fh.read(),
                file_name="executive_summary.md",
                mime="text/markdown",
            )
    else:
        st.title("Executive Summary")
        st.info("No executive summary found. In a second terminal, run: .\\.venv\\Scripts\\python.exe exec_summary.py")

elif page == "How to run":
    st.title("How to run")

    st.code(
        "1) Run: python main.py  # generates reports/ and exports/\n"
        "2) Start: python -m streamlit run dashboard.py\n"
        "3) Open the Local URL in the terminal (usually http://localhost:8501)\n",
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

    st.info("Tip: Refresh Streamlit after you change code or re-run `main.py`.")
