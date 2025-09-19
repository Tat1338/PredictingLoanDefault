"""
================= Loan Default Prediction — Exam Project =================
Author: Tatjana Aasgaard

This project delivers a clear EDA, robust modeling, and standard risk metrics
It includes: reproducibility logs; EDA
(distributions, age groups, income ECDF/deciles/box, debt ratio & utilization,
dependents/mortgage breakdowns, correlation heatmap, high-corr);
risk monotonicity checks; models (LogReg + RandomForest) with stratified CV;
holdout ROC/PR; threshold tuning (F2 + simple cost); calibration;
permutation importance; and extras (Brier, Gini, KS, AUC 95% CI via
bootstrap, lift & gains, score deciles). Operating points include F2-optimal
and a capacity-aware (top 5%) option. Fairness audits cover age groups,
dependents, and mortgage status. Outputs include an HTML figure gallery,
CSV manifests, saved pipelines, a CLI scorer, and a concise model card.

Palette: blue = not late (0), coral = late (1).
===========================================================================
"""


import os
import sys
import platform
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from matplotlib import image as mpimg  # for figure sizes in manifest

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix,
    precision_recall_curve, average_precision_score
)
from sklearn.inspection import permutation_importance
from sklearn.calibration import CalibrationDisplay, CalibratedClassifierCV

# -------------------- Configuration ----------------------------------------
DATA_PATH    = "cs-training.csv"        # CSV path
TARGET       = "SeriousDlqin2yrs"
OUTPUT_DIR   = "reports/figures"
RANDOM_SEED  = 42
TEST_SIZE    = 0.20
SHOW_WINDOWS = False                    # True to pop up figure windows
BOOTSTRAP_AUC_N = 300                   # modest bootstrap for AUC CI
# ----------------------------------------------------------------------------

# ------------------ Palette (color-vision friendly) ------------------------
COLOR_NO   = "#3B5BA5"  # deep blue  — "No late"
COLOR_YES  = "#E45756"  # coral      — "Late"
LINE_MED   = "#FFB000"  # gold       — median
LINE_MEAN  = "#6B7280"  # slate      — mean
LINE_MODE  = "#8E6AC8"  # purple     — mode
HEATMAP_CMAP = "PuOr"   # purple <-> orange
ACCENT     = "#D97706"  # amber — for cost lines
# ----------------------------------------------------------------------------

sns.set_theme(
    style="whitegrid",
    rc={
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "axes.titlepad": 10,
        "legend.frameon": False,
        "figure.dpi": 110,
        "axes.facecolor": "white",
        "grid.color": "#EEF2F5",
        "grid.linewidth": 0.8,
        "font.family": "DejaVu Sans",
    }
)

# ---------------------- Formatters -----------------------------------------
PCT = FuncFormatter(lambda v, _: f"{v*100:.0f}%")  # [0,1] → "xx%"
def _fmt_thousands(x, _):
    try:
        return f"{int(x):,}"
    except Exception:
        return str(x)
FMT_THOUSANDS = FuncFormatter(_fmt_thousands)

# ============================== Small helpers ==============================

def make_output_folder() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def new_fig(figsize=(7, 4)):
    return plt.subplots(figsize=figsize, constrained_layout=True)

def save_and_show(fig: plt.Figure, filename: str) -> None:
    out = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    if SHOW_WINDOWS:
        plt.show()
    plt.close(fig)
    print(f"[SAVE] {out}")

def cap_series(s: pd.Series, q_low: float = 0.001, q_high: float = 0.999) -> pd.Series:
    lo, hi = s.quantile(q_low), s.quantile(q_high)
    return s.clip(lower=lo, upper=hi)

def _first_mode(s: pd.Series) -> float:
    m = s.mode(dropna=True)
    return float(m.iloc[0]) if not m.empty else np.nan

def add_stats_box(ax, s: pd.Series, round_to: int | None = None):
    s = pd.to_numeric(s, errors="coerce").dropna()
    s_for_mode = (s / round_to).round() * round_to if round_to else s
    mean_v   = float(s.mean()) if len(s) else np.nan
    median_v = float(s.median()) if len(s) else np.nan
    mode_v   = _first_mode(s_for_mode) if len(s_for_mode) else np.nan
    lines = []
    if np.isfinite(mean_v):   lines.append(f"Mean: {mean_v:,.0f}")
    if np.isfinite(median_v): lines.append(f"Median: {median_v:,.0f}")
    if np.isfinite(mode_v):   lines.append(f"Mode: {mode_v:,.0f}")
    if lines:
        ax.text(
            0.02, 0.98, "\n".join(lines),
            transform=ax.transAxes, ha="left", va="top", fontsize=9,
            bbox=dict(facecolor="white", edgecolor="#D1D5DB", boxstyle="round,pad=0.3")
        )
    return mean_v, median_v, mode_v

def save_run_environment():
    os.makedirs("reports", exist_ok=True)
    path = os.path.join("reports", "run_environment.txt")
    import sklearn, pandas, numpy, matplotlib, seaborn
    with open(path, "w", encoding="utf-8") as f:
        f.write("=== Run Environment ===\n")
        f.write(f"Python        : {sys.version.split()[0]} ({platform.system()})\n")
        f.write(f"numpy         : {numpy.__version__}\n")
        f.write(f"pandas        : {pandas.__version__}\n")
        f.write(f"scikit-learn  : {sklearn.__version__}\n")
        f.write(f"matplotlib    : {matplotlib.__version__}\n")
        f.write(f"seaborn       : {seaborn.__version__}\n")
    print(f"[SAVE] Environment -> {path}")

def write_min_requirements():
    os.makedirs("reports", exist_ok=True)
    try:
        import sklearn, pandas, numpy, matplotlib, seaborn
        with open("reports/requirements_min.txt", "w", encoding="utf-8") as f:
            f.write(f"numpy=={numpy.__version__}\n")
            f.write(f"pandas=={pandas.__version__}\n")
            f.write(f"scikit-learn=={sklearn.__version__}\n")
            f.write(f"matplotlib=={matplotlib.__version__}\n")
            f.write(f"seaborn=={seaborn.__version__}\n")
        print("[SAVE] Minimal requirements -> reports/requirements_min.txt")
    except Exception:
        pass

# -------- Figures index + manifest, and key numbers -------------------

def _read_captions(cap_path: str) -> dict:
    caps = {}
    if os.path.exists(cap_path):
        with open(cap_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip().startswith("===") or ": " not in line:
                    continue
                k, v = line.strip().split(": ", 1)
                caps[k.strip()] = v.strip()
    return caps

def build_figures_index():
    """Create reports/figures/index.html + manifest.csv."""
    fig_dir = OUTPUT_DIR
    os.makedirs(fig_dir, exist_ok=True)

    captions = _read_captions(os.path.join("reports", "figure_captions.txt"))
    files = sorted([f for f in os.listdir(fig_dir) if f.lower().endswith(".png")])

    # manifest.csv
    rows = []
    for fname in files:
        fpath = os.path.join(fig_dir, fname)
        try:
            arr = mpimg.imread(fpath)
            h, w = int(arr.shape[0]), int(arr.shape[1])
        except Exception:
            w, h = None, None
        size_kb = round(os.path.getsize(fpath) / 1024, 1) if os.path.exists(fpath) else None
        rows.append({
            "filename": fname,
            "caption": captions.get(fname, ""),
            "width_px": w,
            "height_px": h,
            "size_kb": size_kb
        })
    manifest_path = os.path.join(fig_dir, "manifest.csv")
    pd.DataFrame(rows).to_csv(manifest_path, index=False)

    # index.html
    html = [
        "<!doctype html><meta charset='utf-8'>",
        "<title>Figures Index</title>",
        "<style>",
        "body{font-family:Arial, sans-serif;max-width:1100px;margin:24px auto;padding:0 12px;}",
        ".card{margin:16px 0;padding:12px 16px;border:1px solid #e5e7eb;border-radius:10px;}",
        "img{max-width:100%;height:auto;border:1px solid #e5e7eb;border-radius:8px;}",
        "h1{margin:0 0 10px 0} h3{margin:6px 0 8px;} p{margin:6px 0;}",
        "code{background:#f3f4f6;padding:2px 6px;border-radius:6px;}",
        "</style>",
        "<h1>Figures Index</h1>",
        "<p>This page lists all PNGs from <code>reports/figures</code> with short captions.</p>"
    ]
    for r in rows:
        cap = r["caption"]
        html += [
            "<div class='card'>",
            f"<h3>{r['filename']}</h3>",
            f"<p>{cap}</p>" if cap else "<p></p>",
            f"<img src='{r['filename']}' alt='{r['filename']}'>",
            f"<p style='color:#6b7280;font-size:12px'>{r['width_px']}×{r['height_px']} px • {r['size_kb']} KB</p>",
            "</div>"
        ]
    index_path = os.path.join(fig_dir, "index.html")
    with open(index_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html))

    print(f"[SAVE] Manifest   -> {manifest_path}")
    print(f"[SAVE] Index page -> {index_path}")
    if os.name == "nt":
        try:
            os.startfile(os.path.abspath(fig_dir))
        except Exception:
            pass

def build_key_numbers(df: pd.DataFrame, metrics: dict):
    """Save key summary numbers (data + model) to reports/key_numbers.csv."""
    total = len(df)
    late_yes = int(df[TARGET].sum()) if TARGET in df.columns else np.nan
    late_rate = late_yes / total if total else np.nan

    med_inc_all = float(df["MonthlyIncome"].median()) if "MonthlyIncome" in df.columns else np.nan
    med_inc_no  = float(df.loc[df[TARGET]==0, "MonthlyIncome"].median()) if {"MonthlyIncome", TARGET}.issubset(df.columns) else np.nan
    med_inc_yes = float(df.loc[df[TARGET]==1, "MonthlyIncome"].median()) if {"MonthlyIncome", TARGET}.issubset(df.columns) else np.nan
    debtratio_med = float(df["DebtRatio"].median()) if "DebtRatio" in df.columns else np.nan
    util_p99 = float(np.percentile(df["RevolvingUtilizationOfUnsecuredLines"].dropna(), 99)) \
               if "RevolvingUtilizationOfUnsecuredLines" in df.columns else np.nan
    age_med = float(df["age"].median()) if "age" in df.columns else np.nan

    out = pd.DataFrame([{
        "total_clients": total,
        "late_yes": late_yes,
        "late_rate": late_rate,
        "median_income_all": med_inc_all,
        "median_income_no": med_inc_no,
        "median_income_yes": med_inc_yes,
        "debt_ratio_median": debtratio_med,
        "utilization_p99": util_p99,
        "age_median": age_med,
        # model metrics (holdout)
        **metrics
    }])
    os.makedirs("reports", exist_ok=True)
    path = os.path.join("reports", "key_numbers.csv")
    out.to_csv(path, index=False)
    print(f"[SAVE] Key numbers -> {path}")

# =============== Exam extras: metrics & plots most rubrics expect ==========

def brier_score(y_true, y_prob):
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    return float(np.mean((y_true - y_prob)**2))

def gini_from_auc(auc):
    return 2*float(auc) - 1.0

def ks_statistic(y_true, y_prob):
    order = np.argsort(-y_prob)
    y = np.asarray(y_true)[order]
    pos = np.cumsum(y)
    neg = np.cumsum(1 - y)
    P = pos[-1] if pos[-1] > 0 else 1.0
    N = neg[-1] if neg[-1] > 0 else 1.0
    tpr = pos / P
    fpr = neg / N
    return float(np.max(np.abs(tpr - fpr)))

def bootstrap_auc_ci(y_true, y_prob, n=BOOTSTRAP_AUC_N, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true); y_prob = np.asarray(y_prob)
    aucs = []
    m = len(y_true)
    for _ in range(n):
        idx = rng.integers(0, m, size=m)
        try:
            aucs.append(roc_auc_score(y_true[idx], y_prob[idx]))
        except Exception:
            continue
    if not aucs:
        return (np.nan, np.nan)
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return float(lo), float(hi)

def lift_and_gains_plot(y_true, y_prob, model_name, fname_prefix):
    df = pd.DataFrame({"y": y_true, "p": y_prob}).sort_values("p", ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(len(df))
    df["decile"] = pd.qcut(df["rank"], 10, labels=False) + 1  # 1..10
    agg = df.groupby("decile", observed=False).agg(clients=("y","size"),
                                                   events=("y","sum"),
                                                   avg_p=("p","mean")).reset_index()
    agg = agg.sort_values("decile")
    agg["cum_events"] = agg["events"].cumsum()
    total_events = agg["events"].sum()
    baseline = np.arange(1, 11) / 10.0
    gains = agg["cum_events"] / (total_events if total_events > 0 else 1)
    lift = gains / baseline

    # Save decile table
    dec_tbl = agg.copy()
    dec_tbl["cum_events_pct"] = gains
    dec_tbl["lift"] = lift
    os.makedirs("exports", exist_ok=True)
    dec_tbl.to_csv(f"exports/score_deciles_{fname_prefix}.csv", index=False)

    # Gains chart
    fig, ax = new_fig(figsize=(6.2, 4.2))
    ax.plot(np.arange(1,11)/10.0, gains, marker="o", label=f"{model_name}")
    ax.plot(np.arange(1,11)/10.0, baseline, linestyle="--", label="Baseline")
    ax.set_xlabel("Share of population (by score, high→low)")
    ax.set_ylabel("Cumulative share of late payers")
    ax.set_title(f"Cumulative gains — {model_name}")
    ax.legend()
    save_and_show(fig, f"gains_{fname_prefix}.png")

    # Lift chart
    fig, ax = new_fig(figsize=(6.2, 4.2))
    ax.plot(np.arange(1,11)/10.0, lift, marker="o")
    ax.axhline(1.0, linestyle="--", color="#9CA3AF")
    ax.set_xlabel("Share of population (by score)")
    ax.set_ylabel("Lift vs baseline")
    ax.set_title(f"Lift — {model_name}")
    save_and_show(fig, f"lift_{fname_prefix}.png")

def summarize_model(y_true, y_prob, y_hat_tuned, name, thr):
    auc = roc_auc_score(y_true, y_prob)
    ap  = average_precision_score(y_true, y_prob)
    bri = brier_score(y_true, y_prob)
    gin = gini_from_auc(auc)
    ks  = ks_statistic(y_true, y_prob)
    prec = precision_score(y_true, y_hat_tuned, zero_division=0)
    rec  = recall_score(y_true, y_hat_tuned, zero_division=0)
    f1   = f1_score(y_true, y_hat_tuned, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_hat_tuned, labels=[0,1]).ravel()
    lo, hi = bootstrap_auc_ci(y_true, y_prob)
    return {
        "model": name, "threshold": float(thr),
        "AUC": float(auc), "AUC_CI_low": lo, "AUC_CI_high": hi,
        "AP": float(ap), "Brier": float(bri), "Gini": float(gin), "KS": float(ks),
        "Precision_at_thr": float(prec), "Recall_at_thr": float(rec), "F1_at_thr": float(f1),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "Flagged_at_thr": int((y_hat_tuned == 1).sum())
    }

# ===== Business operating point: capacity-aware thresholds =====
def threshold_for_top_rate(y_prob, top_rate: float) -> float:
    """
    Threshold so that approximately 'top_rate' fraction are flagged positive.
    Example: top_rate=0.05 -> top 5% highest risk are flagged.
    """
    probs = np.asarray(y_prob, dtype=float)
    top_rate = float(np.clip(top_rate, 0.0, 1.0))
    if top_rate <= 0 or len(probs) == 0:
        return 1.0  # nobody flagged
    q = np.quantile(probs, 1.0 - top_rate)  # right-tail cut
    return float(q)

def eval_at_threshold(y_true, y_prob, thr: float) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_hat  = (np.asarray(y_prob) >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_hat, labels=[0,1]).ravel()
    prec = precision_score(y_true, y_hat, zero_division=0)
    rec  = recall_score(y_true, y_hat, zero_division=0)
    f1   = f1_score(y_true, y_hat, zero_division=0)
    return {
        "threshold": float(thr), "flagged": int(y_hat.sum()),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "precision": float(prec), "recall": float(rec), "f1": float(f1)
    }

# ===== Fairness & error-balance audit =====
def _mk_age_groups(s: pd.Series):
    bins   = [18, 25, 35, 45, 55, 65, 200]
    labels = ["18–24","25–34","35–44","45–54","55–64","65+"]
    s = pd.to_numeric(s, errors="coerce")
    return pd.cut(s, bins=bins, labels=labels, right=False)

def _mk_dep_buckets(s: pd.Series):
    s = pd.to_numeric(s, errors="coerce")
    b = s.clip(0, 3).fillna(0).astype(int).astype(str).replace({"3":"3+"})
    return b

def _mk_mortgage_flag(s: pd.Series):
    s = pd.to_numeric(s, errors="coerce").fillna(0)
    return (s > 0).map({False:"No mortgage", True:"≥1 mortgage"})

def fairness_table(y_true, y_prob, groups: pd.Series, thr: float) -> pd.DataFrame:
    """
    Returns per-group: size, prevalence, flagged rate, TPR, FPR, PPV.
    """
    df = pd.DataFrame({
        "y": np.asarray(y_true).astype(int),
        "p": np.asarray(y_prob, dtype=float),
        "g": groups.astype(str).fillna("Missing")
    }).dropna()
    df["yhat"] = (df["p"] >= thr).astype(int)

    def _safe_rate(num, den): return float(num/den) if den > 0 else np.nan

    rows = []
    for g, sub in df.groupby("g", observed=False):
        y = sub["y"].values; yhat = sub["yhat"].values
        tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0,1]).ravel()
        rows.append({
            "group": g,
            "n": int(len(sub)),
            "prevalence": _safe_rate(y.sum(), len(sub)),
            "flagged_rate": _safe_rate(yhat.sum(), len(sub)),
            "TPR": _safe_rate(tp, tp+fn),   # recall / sensitivity
            "FPR": _safe_rate(fp, fp+tn),
            "PPV": precision_score(y, yhat, zero_division=0)
        })
    out = pd.DataFrame(rows).sort_values("n", ascending=False)
    return out

def save_fairness_audits(y_true, y_prob, X: pd.DataFrame, thr: float, prefix: str):
    os.makedirs("reports/fairness", exist_ok=True)
    audits = []
    # Age groups
    if "age" in X.columns:
        age_g = _mk_age_groups(X["age"])
        tbl = fairness_table(y_true, y_prob, age_g, thr)
        tbl.to_csv(f"reports/fairness/{prefix}_age.csv", index=False)
        audits.append(("Age groups", tbl))
    # Dependents
    if "NumberOfDependents" in X.columns:
        dep_g = _mk_dep_buckets(X["NumberOfDependents"])
        tbl = fairness_table(y_true, y_prob, dep_g, thr)
        tbl.to_csv(f"reports/fairness/{prefix}_dependents.csv", index=False)
        audits.append(("Dependents", tbl))
    # Mortgage status
    if "NumberRealEstateLoansOrLines" in X.columns:
        mort_g = _mk_mortgage_flag(X["NumberRealEstateLoansOrLines"])
        tbl = fairness_table(y_true, y_prob, mort_g, thr)
        tbl.to_csv(f"reports/fairness/{prefix}_mortgage.csv", index=False)
        audits.append(("Mortgage status", tbl))
    # Quick TPR bars for the first audit (usually age)
    if audits:
        title, tbl = audits[0]
        fig, ax = new_fig(figsize=(7.2, 4.2))
        sns.barplot(data=tbl, x="group", y="TPR", ax=ax, color=COLOR_YES)
        ax.set_title(f"TPR by group — {title}")
        ax.set_xlabel(title); ax.set_ylabel("True Positive Rate")
        ax.yaxis.set_major_formatter(PCT)
        ax.tick_params(axis="x", labelrotation=45)
        for lbl in ax.get_xticklabels():
            lbl.set_ha("right")
        save_and_show(fig, f"fairness_{prefix}_tpr.png")

# ============================ Load & Clean data ============================

def load_csv() -> pd.DataFrame:
    path_abs = os.path.abspath(DATA_PATH)
    if not os.path.exists(DATA_PATH):
        print("\n[ERROR] Can’t find the dataset.")
        print(f"Looked for: {path_abs}")
        print("➡ Put 'cs-training.csv' next to this script or change DATA_PATH.")
        raise FileNotFoundError(path_abs)
    df = pd.read_csv(DATA_PATH)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"] )
    return df

def clean_data(df: pd.DataFrame):
    """
    Minimal, explainable cleaning:
      - Replace +/-inf with NaN; drop duplicates
      - Fill MonthlyIncome and NumberOfDependents with median (if present)
      - Drop rows with age <= 0
      - Drop rows where target is missing
    Returns: (clean_df, summary_dict)
    """
    summary = {}
    rows_before = len(df)
    dup_before = df.duplicated().sum()
    inf_count = 0
    for c in df.select_dtypes(include=[np.number]).columns:
        inf_count += np.isinf(df[c]).sum()

    out = df.replace([np.inf, -np.inf], np.nan)
    out = out.drop_duplicates()
    summary["duplicates_removed"] = int(dup_before)

    if "MonthlyIncome" in out.columns:
        mi_na_before = out["MonthlyIncome"].isna().sum()
        out["MonthlyIncome"] = out["MonthlyIncome"].fillna(out["MonthlyIncome"].median())
        summary["monthlyincome_filled"] = int(mi_na_before)

    if "NumberOfDependents" in out.columns:
        dep_na_before = out["NumberOfDependents"].isna().sum()
        out["NumberOfDependents"] = out["NumberOfDependents"].fillna(out["NumberOfDependents"].median())
        summary["dependents_filled"] = int(dep_na_before)

    if "age" in out.columns:
        bad_age = int((out["age"] <= 0).sum())
        out = out[out["age"] > 0]
        summary["nonpositive_age_removed"] = bad_age

    if TARGET in out.columns:
        tgt_na = int(out[TARGET].isna().sum())
        out = out.dropna(subset=[TARGET])
        summary["target_missing_dropped"] = tgt_na

    summary["infinite_values_found"] = int(inf_count)
    summary["rows_before"] = int(rows_before)
    summary["rows_after"]  = int(len(out))
    summary["rows_removed_total"] = int(rows_before - len(out))
    return out, summary

def save_cleaning_report(summary: dict):
    os.makedirs("reports", exist_ok=True)
    path = os.path.join("reports", "cleaning_report.txt")
    lines = [
        "=== Data Cleaning Summary ===\n",
        f"Rows before cleaning       : {summary.get('rows_before', 0):,}\n",
        f"Rows after cleaning        : {summary.get('rows_after', 0):,}\n",
        f"Total rows removed         : {summary.get('rows_removed_total', 0):,}\n",
        f"Duplicates removed         : {summary.get('duplicates_removed', 0):,}\n",
        f"Infinite values found      : {summary.get('infinite_values_found', 0):,}\n",
        f"Filled MonthlyIncome (NaN) : {summary.get('monthlyincome_filled', 0):,}\n",
        f"Filled Dependents (NaN)    : {summary.get('dependents_filled', 0):,}\n",
        f"Non-positive ages removed  : {summary.get('nonpositive_age_removed', 0):,}\n",
        f"Missing targets dropped    : {summary.get('target_missing_dropped', 0):,}\n",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"[SAVE] Cleaning summary -> {path}")

def write_figure_captions():
    os.makedirs("reports", exist_ok=True)
    path = os.path.join("reports", "figure_captions.txt")
    captions = {
        "serious_late_payment_counts.png": "Class balance: ~6–7% became 90+ days late within 2 years.",
        "age_distribution.png": "Age is broad; median and mode marked.",
        "age_groups_counts.png": "Most clients are between 35–64.",
        "age_group_default_rate.png": "Share late by age group (labels include group size).",
        "monthly_income_focus.png": "Income distribution (log scale, 1–99th pct) with mean/median.",
        "monthly_income_ecdf.png": "ECDF: share of clients earning ≤ X (median/P90/P99 marked).",
        "monthly_income_deciles.png": "Income deciles: clients (bars) & late rate (line).",
        "monthly_income_by_default_box.png": "Median income slightly lower among late payers (0–40k view).",
        "debtratio_zoom.png": "Debt-to-income peaks below 100%; 100% & 200% lines shown (0–200%).",
        "debtratio_log.png": "Log scale reveals a long right tail of extreme ratios.",
        "dependents_default_counts.png": "Counts by dependents with % late on red bars.",
        "dependents_default_rate.png": "Share late by dependents (labels include clients).",
        "mortgage_counts.png": "How many clients with vs without mortgages.",
        "mortgage_default_rate.png": "Share late by mortgage status (labels include clients).",
        "mortgage_owners_breakdown.png": "Among mortgage owners, split of late vs no late.",
        "revolving_utilization_zoom.png": "Credit utilization mostly under 100%; 99th pct marked.",
        "revolving_utilization_log.png": "Log view shows rare very-high utilization.",
        "correlation_heatmap.png": "Correlation heatmap (numeric features).",
        "gains_logreg.png": "Cumulative gains — Logistic Regression.",
        "gains_rf.png": "Cumulative gains — Random Forest.",
        "lift_logreg.png": "Lift vs baseline — Logistic Regression.",
        "lift_rf.png": "Lift vs baseline — Random Forest.",
        "debtratio_risk_bins.png": "DebtRatio bins: clients and serious-late rate.",
        "utilization_risk_bins.png": "Utilization bins: clients and serious-late rate.",
        "cm_logreg.png": "Confusion matrix — Logistic Regression.",
        "cm_rf.png": "Confusion matrix — Random Forest.",
        "roc_logreg.png": "ROC — Logistic Regression; diagonal is random.",
        "roc_rf.png": "ROC — Random Forest; diagonal is random.",
        "pr_logreg.png": "Precision–Recall — Logistic Regression.",
        "pr_rf.png": "Precision–Recall — Random Forest.",
        "cm_logreg_tuned.png": "Confusion at chosen threshold — Logistic Regression.",
        "cm_rf_tuned.png": "Confusion at chosen threshold — Random Forest.",
        "pi_logreg.png": "Permutation importance — top drivers (LogReg).",
        "pi_rf.png": "Permutation importance — top drivers (RF).",
        "thr_tuning_logistic_regression.png": "Precision/Recall/F-scores & cost vs threshold (LogReg).",
        "thr_tuning_random_forest.png": "Precision/Recall/F-scores & cost vs threshold (RF).",
        "calibration_logreg.png": "Reliability curve — predicted vs observed risk (LogReg).",
        "fairness_logreg_tuned_tpr.png": "TPR by group — Logistic Regression (tuned threshold).",
        "fairness_rf_tuned_tpr.png": "TPR by group — Random Forest (tuned threshold)."
    }
    with open(path, "w", encoding="utf-8") as f:
        f.write("=== Figure Captions (short, ready for the report) ===\n")
        for k, v in captions.items():
            f.write(f"{k}: {v}\n")
    print(f"[SAVE] Figure captions -> {path}")

# ============================== Diagnostic EDA =============================

def missingness_table(df: pd.DataFrame):
    miss = df.isna().sum().rename("missing")
    pct = (df.isna().mean()*100).round(2).rename("missing_pct")
    out = pd.concat([miss, pct], axis=1).sort_values("missing_pct", ascending=False)
    os.makedirs("reports", exist_ok=True)
    out.to_csv("reports/missingness.csv")
    print("[SAVE] Missingness -> reports/missingness.csv")

def high_corr_report(df: pd.DataFrame, thr: float = 0.80):
    num = df.select_dtypes(include=[np.number])
    if num.shape[1] < 2:
        return
    corr = num.corr(numeric_only=True)
    pairs = []
    cols = corr.columns
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            r = corr.iloc[i, j]
            if abs(r) >= thr:
                pairs.append((cols[i], cols[j], float(r)))
    out = pd.DataFrame(pairs, columns=["feature_1", "feature_2", "corr"])
    os.makedirs("reports", exist_ok=True)
    out.to_csv("reports/high_corr_pairs.csv", index=False)
    print("[SAVE] High-corr pairs (|r|>=0.80) -> reports/high_corr_pairs.csv")

def risk_by_bins_plot(df: pd.DataFrame, col: str, title: str, fname: str, bins=None, x_fmt=None):
    if col not in df.columns or TARGET not in df.columns:
        return
    s = pd.to_numeric(df[col], errors="coerce")
    tmp = pd.DataFrame({col: s, TARGET: df[TARGET]}).dropna()
    if bins is None:
        bins = [0, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]  # generic ratios
    tmp["bin"] = pd.cut(tmp[col], bins=bins, include_lowest=True)
    grp = tmp.groupby("bin", observed=False)[TARGET].agg(late="mean", clients="count").reset_index()
    grp["label"] = grp["bin"].astype(str)
    fig, ax1 = new_fig(figsize=(8.5, 4.2))
    ax1.bar(grp["label"], grp["clients"], color=COLOR_NO)
    ax1.set_ylabel("Number of clients"); ax1.yaxis.set_major_formatter(FMT_THOUSANDS)
    ax1.tick_params(axis="x", labelrotation=45)
    for lbl in ax1.get_xticklabels():
        lbl.set_ha("right")
    ax2 = ax1.twinx()
    ax2.plot(grp["label"], grp["late"], marker="o", linewidth=2, color=COLOR_YES)
    ax2.set_ylabel("Serious-late rate"); ax2.yaxis.set_major_formatter(PCT)
    fig.suptitle(title)
    save_and_show(fig, fname)

# ================================ EDA ======================================

def make_eda_charts(df: pd.DataFrame) -> None:
    make_output_folder()
    plt.close("all")
    print("\n[EDA] Columns:", list(df.columns))

    col_y   = TARGET
    col_age = "age"
    col_inc = "MonthlyIncome"
    col_dr  = "DebtRatio"
    col_ru  = "RevolvingUtilizationOfUnsecuredLines"
    col_re  = "NumberRealEstateLoansOrLines"
    col_dep = "NumberOfDependents"

    # Missingness and high-corr reports for the examiner
    missingness_table(df)
    high_corr_report(df, thr=0.80)

    # 1) Target balance (counts + %)
    if col_y in df.columns:
        counts = df[col_y].value_counts().sort_index()
        total  = counts.sum()
        data = pd.DataFrame({"label": counts.index.astype(str), "n": counts.values})
        fig, ax = new_fig(figsize=(6.2, 4))
        sns.barplot(
            data=data, x="label", y="n", hue="label",
            palette={"0": COLOR_NO, "1": COLOR_YES}, dodge=False, ax=ax
        )
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()
        ax.set_title("Number of clients by serious late (90+ days)")
        ax.set_xlabel("Serious late (0 = No, 1 = Yes)")
        ax.set_ylabel("Number of clients"); ax.yaxis.set_major_formatter(FMT_THOUSANDS)
        for i, v in enumerate(counts.values):
            ax.text(i, v, f"{v:,}\n({v/total:.1%})", ha="center", va="bottom", fontsize=9)
        ax.margins(y=0.10)
        save_and_show(fig, "serious_late_payment_counts.png")
        print(f"[EDA] Serious late payment: {int(counts.get(1,0)):,} of {total:,} = {counts.get(1,0)/total:.1%}")

    # 2) Age distribution + mean/median/mode
    if col_age in df.columns:
        age = df[col_age].dropna()
        fig, ax = new_fig()
        sns.histplot(age, bins=40, kde=True, ax=ax, color=COLOR_NO)
        ax.set_title("Age distribution")
        ax.set_xlabel("Age")
        ax.set_ylabel("Number of clients"); ax.yaxis.set_major_formatter(FMT_THOUSANDS)
        mean_v, median_v, mode_v = add_stats_box(ax, age)
        if np.isfinite(median_v): ax.axvline(median_v, linestyle="--", linewidth=2, color=LINE_MED,  label=f"Median {median_v:,.0f}")
        if np.isfinite(mean_v):   ax.axvline(mean_v,   linestyle=":",  linewidth=2, color=LINE_MEAN, label=f"Mean {mean_v:,.0f}")
        if np.isfinite(mode_v):   ax.axvline(mode_v,   linestyle="-.", linewidth=2, color=LINE_MODE, label=f"Mode {mode_v:,.0f}")
        ax.legend()
        save_and_show(fig, "age_distribution.png")

    # 3) Age groups — counts + % late
    if {col_age, col_y}.issubset(df.columns):
        bins = [18, 25, 35, 45, 55, 65, 100]
        labels = ["18–24", "25–34", "35–44", "45–54", "55–64", "65+"]
        tmp = df[[col_age, col_y]].dropna().copy()
        tmp = tmp[tmp[col_age] >= 18]
        tmp["age_group"] = pd.cut(tmp[col_age], bins=bins, labels=labels, right=False)

        grp_counts = tmp["age_group"].value_counts().reindex(labels).fillna(0).astype(int)
        fig, ax = new_fig(figsize=(7.5, 4))
        sns.barplot(x=grp_counts.index, y=grp_counts.values, ax=ax, color=COLOR_NO)
        ax.set_title("Number of clients by age group")
        ax.set_xlabel("Age group"); ax.set_ylabel("Number of clients")
        ax.yaxis.set_major_formatter(FMT_THOUSANDS)
        for i, v in enumerate(grp_counts.values):
            ax.text(i, v, f"{int(v):,}", ha="center", va="bottom", fontsize=9, color="#374151")
        save_and_show(fig, "age_groups_counts.png")

        rate_df = (
            tmp.groupby("age_group", observed=False)[col_y]
               .agg(late="sum", clients="count")
               .reindex(labels)
               .assign(rate=lambda d: d["late"]/d["clients"])
        )
        fig, ax = new_fig(figsize=(7.5, 4))
        sns.barplot(x=rate_df.index, y=rate_df["rate"], ax=ax, color=COLOR_YES)
        ax.set_title("Share late by age group")
        ax.set_xlabel("Age group"); ax.set_ylabel("% of clients")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
        for i, (r, n) in enumerate(zip(rate_df["rate"], rate_df["clients"])):
            ax.text(i, r, f"{r*100:.1f}%  |  {n:,} clients",
                    ha="center", va="bottom", fontsize=9, color="#374151")
        save_and_show(fig, "age_group_default_rate.png")

    # 4) Monthly income — log distribution + ECDF + deciles + box
    if "MonthlyIncome" in df.columns:
        inc_raw = pd.to_numeric(df["MonthlyIncome"], errors="coerce").dropna()
        inc_cap = cap_series(inc_raw, 0.01, 0.99)
        inc_pos = inc_cap[inc_cap > 0]

        fig, ax = new_fig()
        sns.histplot(inc_pos, bins=60, stat="percent", ax=ax, color=COLOR_NO)
        ax.set_xscale("log")
        xticks = [500, 1000, 2000, 5000, 10_000, 20_000, 40_000]
        ax.set_xticks(xticks)
        ax.set_xticklabels([f"{int(t):,}" for t in xticks])
        ax.set_title("Monthly income — distribution (log scale, 1–99th pct)")
        ax.set_xlabel("Monthly income"); ax.set_ylabel("% of clients")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
        median_v = float(inc_pos.median()); mean_v = float(inc_pos.mean())
        ax.axvline(median_v, linestyle="--", linewidth=2, color=LINE_MED,  label=f"Median {median_v:,.0f}")
        ax.axvline(mean_v,   linestyle=":",  linewidth=2, color=LINE_MEAN, label=f"Mean {mean_v:,.0f}")
        ax.legend(frameon=False, loc="upper left")
        save_and_show(fig, "monthly_income_focus.png")

        # ECDF
        x = np.sort(inc_cap.values)
        y = np.arange(1, len(x) + 1) / len(x)
        fig, ax = new_fig()
        ax.plot(x, y, linewidth=2, color=COLOR_NO)
        ax.set_xlim(0, 40_000); ax.set_ylim(0, 1)
        ax.set_title("Monthly income — ECDF"); ax.set_xlabel("Monthly income")
        ax.set_ylabel("Share of clients"); ax.yaxis.set_major_formatter(PCT)
        for px, label in [(np.percentile(x,50), "Median"),
                          (np.percentile(x,90), "P90"),
                          (np.percentile(x,99), "P99")]:
            ax.axvline(px, linestyle="--", linewidth=1.5,
                       color=LINE_MED if label=="Median" else "#9CA3AF")
            ax.text(px, 0.02, f"{label} {px:,.0f}", rotation=90, va="bottom",
                    ha="right", fontsize=9, color="#374151")
        save_and_show(fig, "monthly_income_ecdf.png")

        # Deciles combo (clients + late rate)
        y_nonnull = df.loc[inc_raw.index, TARGET] if TARGET in df.columns else pd.Series(index=inc_raw.index, dtype=int)
        bins = pd.qcut(inc_raw, 10, duplicates="drop")
        cats = bins.cat.categories
        summary = (
            pd.DataFrame({"bin": bins, TARGET: y_nonnull})
            .groupby("bin", observed=False)
            .agg(clients=("bin", "size"), late=(TARGET, "sum"))
            .assign(late_rate=lambda d: d["late"] / d["clients"])
            .reindex(cats)
        )
        pretty_ranges = [f"{int(iv.left):,}–{int(iv.right):,}" for iv in summary.index]
        summary.insert(0, "range", pretty_ranges)
        os.makedirs("reports", exist_ok=True)
        summary.reset_index(drop=True).to_csv("reports/monthly_income_deciles.csv", index=False)
        print("[SAVE] reports/monthly_income_deciles.csv")

        fig, ax1 = new_fig(figsize=(9, 4))
        ax1.bar(summary["range"], summary["clients"], color=COLOR_NO)
        ax1.set_ylabel("Number of clients"); ax1.yaxis.set_major_formatter(FMT_THOUSANDS)
        ax1.tick_params(axis="x", labelrotation=45)
        for lbl in ax1.get_xticklabels():
            lbl.set_ha("right")
        ax2 = ax1.twinx()
        ax2.plot(summary["range"], summary["late_rate"], marker="o", linewidth=2, color=COLOR_YES)
        ax2.set_ylabel("Serious-late rate"); ax2.yaxis.set_major_formatter(PCT)
        fig.suptitle("Monthly income deciles: clients & late rate")
        save_and_show(fig, "monthly_income_deciles.png")

        # Box by late
        if TARGET in df.columns:
            box = df[["MonthlyIncome", TARGET]].dropna().copy()
            box["MonthlyIncome"] = cap_series(box["MonthlyIncome"])
            box[TARGET] = box[TARGET].astype(int).astype(str)
            n0 = (box[TARGET] == "0").sum(); n1 = (box[TARGET] == "1").sum()
            order = ["0", "1"]; labels = [f"No (n={n0:,})", f"Yes (n={n1:,})"]
            fig, ax = new_fig()
            sns.boxplot(
                data=box, x=TARGET, y="MonthlyIncome",
                order=order, hue=TARGET, hue_order=order,
                palette={"0": COLOR_NO, "1": COLOR_YES},
                dodge=False, showfliers=False, ax=ax
            )
            leg = ax.get_legend()
            if leg is not None:
                leg.remove()
            ax.set_ylim(0, 40_000)
            ax.set_title("Monthly income by serious late (0–40,000)")
            ax.set_xlabel("Serious late")
            ax.set_ylabel("Monthly income"); ax.yaxis.set_major_formatter(FMT_THOUSANDS)
            ax.set_xticklabels(labels)
            med0 = float(box.loc[box[TARGET] == "0", "MonthlyIncome"].median())
            med1 = float(box.loc[box[TARGET] == "1", "MonthlyIncome"].median())
            ax.annotate(f"Median {med0:,.0f}", xy=(0, med0), xytext=(-20, 10),
                        textcoords="offset points", ha="right", va="bottom", fontsize=9, color="#374151",
                        arrowprops=dict(arrowstyle="-", color="#9CA3AF", lw=1))
            ax.annotate(f"Median {med1:,.0f}", xy=(1, med1), xytext=(20, 10),
                        textcoords="offset points", ha="left", va="bottom", fontsize=9, color="#374151",
                        arrowprops=dict(arrowstyle="-", color="#9CA3AF", lw=1))
            save_and_show(fig, "monthly_income_by_default_box.png")

    # 5) Debt ratio & Utilization — risk-by-bins (monotonicity check)
    risk_by_bins_plot(df, "DebtRatio",
                      "Debt-to-income ratio bins — clients & late rate",
                      "debtratio_risk_bins.png",
                      bins=[0, 0.25, 0.5, 1.0, 2.0, 5.0, 50.0])
    risk_by_bins_plot(df, "RevolvingUtilizationOfUnsecuredLines",
                      "Credit utilization bins — clients & late rate",
                      "utilization_risk_bins.png",
                      bins=[0, 0.25, 0.5, 1.0, 2.0, 5.0, 50.0])

    # 6) Dependents — counts + % on red bars; rate chart
    if {"NumberOfDependents", TARGET}.issubset(df.columns):
        ddf = df["NumberOfDependents"].to_frame().join(df[TARGET]).dropna().copy()
        ddf["dep_bucket"] = ddf["NumberOfDependents"].clip(0, 3).astype(int).astype(str).replace({"3": "3+"})

        ctab = (
            ddf.groupby(["dep_bucket", TARGET]).size()
               .unstack(fill_value=0)
               .reindex(["0", "1", "2", "3+"])
        )
        ctab.columns = ["No", "Yes"]
        totals = ctab.sum(axis=1)
        rates = (ctab["Yes"] / totals).fillna(0.0)

        melted = ctab.reset_index().melt(id_vars="dep_bucket", value_vars=["No","Yes"],
                                         var_name="Late", value_name="Count")
        fig, ax = new_fig(figsize=(8.8, 4.8))
        sns.barplot(
            data=melted, x="dep_bucket", y="Count", hue="Late",
            hue_order=["No","Yes"],
            palette={"No": COLOR_NO, "Yes": COLOR_YES},
            ax=ax
        )
        ax.set_title("Serious late by dependents (counts and % late)")
        ax.set_xlabel("Dependents"); ax.set_ylabel("Number of clients")
        ax.yaxis.set_major_formatter(FMT_THOUSANDS)
        ax.legend(title="Late status", ncol=2, frameon=False)
        blue_bars, red_bars = ax.containers[0], ax.containers[1]
        for i, p in enumerate(blue_bars):
            cnt = ctab["No"].iloc[i]
            ax.annotate(f"{cnt:,}", (p.get_x()+p.get_width()/2, p.get_height()),
                        ha="center", va="bottom", fontsize=9, color="#374151",
                        xytext=(0, 2), textcoords="offset points")
        for i, p in enumerate(red_bars):
            cnt = ctab["Yes"].iloc[i]
            pct = rates.iloc[i] * 100
            ax.annotate(f"{pct:.1f}%  ({cnt:,})", (p.get_x()+p.get_width()/2, p.get_height()),
                        ha="center", va="bottom", fontsize=9, color=COLOR_YES, fontweight="bold",
                        xytext=(0, 2), textcoords="offset points")
        save_and_show(fig, "dependents_default_counts.png")

        rate_tbl = pd.DataFrame({
            "dep_bucket": totals.index,
            "clients": totals.values,
            "late_rate": rates.values
        })
        fig, ax = new_fig()
        sns.barplot(x="dep_bucket", y="late_rate", data=rate_tbl, ax=ax, color=COLOR_YES)
        ax.set_title("Share late by dependents")
        ax.set_xlabel("Dependents"); ax.set_ylabel("% of clients")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
        for i, row in rate_tbl.iterrows():
            ax.text(i, row["late_rate"], f"{row['late_rate']*100:.1f}%  |  {int(row['clients']):,} clients",
                    ha="center", va="bottom", fontsize=9, color="#374151")
        save_and_show(fig, "dependents_default_rate.png")

        out_tbl = pd.DataFrame({
            "dep_bucket": totals.index,
            "clients": totals.values,
            "late": ctab["Yes"].values,
            "late_rate": rates.values
        }).set_index("dep_bucket")
        os.makedirs("reports", exist_ok=True)
        out_tbl.to_csv("reports/dependents_summary.csv")
        print("\n[Table] Dependents summary (clients, late, late_rate):")
        print(out_tbl)
        print("[SAVE] reports/dependents_summary.csv")

    # 7) Mortgages (proxy: NumberRealEstateLoansOrLines)
    if {"NumberRealEstateLoansOrLines", TARGET}.issubset(df.columns):
        re_df = df[["NumberRealEstateLoansOrLines", TARGET]].dropna().copy()
        re_df["has_mortgage"] = (re_df["NumberRealEstateLoansOrLines"] > 0).astype(int)

        re_counts = re_df["has_mortgage"].value_counts().sort_index()
        fig, ax = new_fig(figsize=(6, 4))
        sns.barplot(x=["No mortgage", "≥1 mortgage"], y=re_counts.values, ax=ax, color=COLOR_NO)
        ax.set_title("Number of clients with mortgages")
        ax.set_xlabel("Mortgage status"); ax.set_ylabel("Number of clients"); ax.yaxis.set_major_formatter(FMT_THOUSANDS)
        for i, v in enumerate(re_counts.values):
            ax.text(i, v, f"{int(v):,}", ha="center", va="bottom", fontsize=9)
        save_and_show(fig, "mortgage_counts.png")

        rate_tbl = (
            re_df.groupby("has_mortgage")[TARGET]
                 .agg(late="mean", clients="count")
                 .reindex([0, 1])
                 .rename(index={0: "No mortgage", 1: "≥1 mortgage"})
        )
        fig, ax = new_fig(figsize=(6.4, 4))
        sns.barplot(x=rate_tbl.index, y=rate_tbl["late"], ax=ax, color=COLOR_YES)
        ax.set_title("Serious late rate by mortgage status")
        ax.set_xlabel("Mortgage status"); ax.set_ylabel("Share of clients")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
        for i, (r, n) in enumerate(zip(rate_tbl["late"], rate_tbl["clients"])):
            ax.text(i, r, f"{r*100:.1f}%  |  {n:,} clients", ha="center", va="bottom", fontsize=9, color="#374151")
        save_and_show(fig, "mortgage_default_rate.png")

        mort = re_df[re_df["has_mortgage"] == 1]
        if not mort.empty:
            ctab = mort[TARGET].value_counts().reindex([0, 1]).fillna(0).astype(int)
            total_mort = int(ctab.sum()); shares = ctab / total_mort if total_mort else ctab
            fig, ax = new_fig(figsize=(6, 4))
            ax.bar(["Mortgage owners"], [ctab.loc[0]], color=COLOR_NO,  label="No late")
            ax.bar(["Mortgage owners"], [ctab.loc[1]], bottom=[ctab.loc[0]], color=COLOR_YES, label="Serious late")
            ax.set_title("Mortgage owners — breakdown")
            ax.set_ylabel("Number of clients"); ax.yaxis.set_major_formatter(FMT_THOUSANDS)
            if total_mort:
                ax.text(0, ctab.loc[0]/2,              f"No late\n{ctab.loc[0]:,}\n({shares.loc[0]:.1%})",
                        ha="center", va="center", color="white", fontsize=10)
                ax.text(0, ctab.loc[0]+ctab.loc[1]/2, f"Serious late\n{ctab.loc[1]:,}\n({shares.loc[1]:.1%})",
                        ha="center", va="center", color="white", fontsize=10)
            ax.legend()
            save_and_show(fig, "mortgage_owners_breakdown.png")

    # 8) Credit utilization (0–200%) + log view
    if "RevolvingUtilizationOfUnsecuredLines" in df.columns:
        ru = pd.to_numeric(df["RevolvingUtilizationOfUnsecuredLines"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        fig, ax = new_fig()
        sns.histplot(ru, bins=100, kde=True, ax=ax, color=COLOR_NO)
        ax.set_xlim(0, 2)
        ax.set_title("Credit utilization — 0–200%")
        ax.set_xlabel("Credit utilization"); ax.set_ylabel("Number of clients")
        ax.xaxis.set_major_formatter(PCT); ax.yaxis.set_major_formatter(FMT_THOUSANDS)
        ax.axvline(1.0, color="#9CA3AF", linestyle="--", label="100% (maxed out)")
        u99 = float(ru.quantile(0.99))
        if u99 <= 2:
            ax.axvline(u99, color=COLOR_YES, linestyle="--", label=f"99th pct ≈ {u99*100:.0f}%")
        else:
            ax.text(0.98, 0.90, f"99th pct ≈ {u99*100:.0f}% (off-chart)",
                    transform=ax.transAxes, ha="right", va="top", fontsize=9, color=COLOR_YES)
        mean_v, median_v, mode_v = add_stats_box(ax, ru)
        if np.isfinite(median_v): ax.axvline(median_v, linestyle="--", linewidth=2, color=LINE_MED,  label=f"Median {median_v*100:.0f}%")
        if np.isfinite(mean_v):   ax.axvline(mean_v,   linestyle=":",  linewidth=2, color=LINE_MEAN, label=f"Mean {mean_v*100:.0f}%")
        if np.isfinite(mode_v):   ax.axvline(mode_v,   linestyle="-.", linewidth=2, color=LINE_MODE, label=f"Mode {mode_v*100:.0f}%")
        ax.legend()
        save_and_show(fig, "revolving_utilization_zoom.png")

        fig, ax = new_fig()
        sns.histplot(ru[ru > 0], bins=120, kde=False, ax=ax, color=COLOR_NO)
        ax.set_xscale("log")
        ax.set_title("Credit utilization (log scale)")
        ax.set_xlabel("Credit utilization (log)"); ax.set_ylabel("Number of clients")
        save_and_show(fig, "revolving_utilization_log.png")

    # 9) Correlation heatmap
    num_df = df.select_dtypes(include=[np.number]).copy()
    if not num_df.empty and num_df.shape[1] >= 2:
        corr = num_df.corr(numeric_only=True)
        upper = corr.where(np.triu(np.ones_like(corr, dtype=bool), k=1))
        vals = upper.unstack().dropna().abs().values
        lim = float(np.quantile(vals, 0.95)) if len(vals) else 0.35
        lim = min(max(lim, 0.25), 0.55)
        mask = np.triu(np.ones_like(corr, dtype=bool))
        label_thr = 0.30
        labels = corr.round(2).astype(str).where(np.abs(corr) >= label_thr, "")
        labels = labels.mask(mask, "")
        fig, ax = new_fig(figsize=(9, 7))
        sns.heatmap(
            corr, mask=mask, cmap=HEATMAP_CMAP, center=0,
            vmin=-lim, vmax=lim, square=True,
            linewidths=0.5, linecolor="#E6ECF2",
            cbar_kws={"shrink": 0.8, "ticks": np.linspace(-lim, lim, 7)},
            annot=labels, fmt="", annot_kws={"fontsize": 8, "fontweight": "bold", "color": "#1F2937"},
            ax=ax
        )
        ax.set_title("Correlation heatmap (numeric features)")
        save_and_show(fig, "correlation_heatmap.png")

# ================================ Modeling =================================

def build_pipes(X: pd.DataFrame):
    scale_cols = [c for c in ["RevolvingUtilizationOfUnsecuredLines", "age", "DebtRatio", "MonthlyIncome"] if c in X.columns]
    prep = ColumnTransformer([("scaler", StandardScaler(), scale_cols)], remainder="passthrough")
    logreg = Pipeline([("prep", prep),
                       ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED))])
    rf = Pipeline([("prep", prep),
                   ("clf", RandomForestClassifier(n_estimators=300, min_samples_split=4, min_samples_leaf=2,
                                                 class_weight="balanced", n_jobs=-1, random_state=RANDOM_SEED))])
    return logreg, rf

def kfold_report(model, X, y, name: str):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    auc = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    f1  = cross_val_score(model, X, y, cv=cv, scoring="f1", n_jobs=-1)
    ap  = cross_val_score(model, X, y, cv=cv, scoring="average_precision", n_jobs=-1)
    print(f"\n[CV] {name} — 5-fold")
    print(f"  AUC: {auc.mean():.3f} ± {auc.std():.3f}")
    print(f"  F1 : {f1.mean():.3f} ± {f1.std():.3f}")
    print(f"  AP : {ap.mean():.3f} ± {ap.std():.3f}")

def pr_curve(y_true, y_prob, title, fname):
    p, r, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    fig, ax = new_fig(figsize=(6, 5))
    ax.plot(r, p, label=f"AP = {ap:.3f}", color=COLOR_YES)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title(title); ax.legend()
    save_and_show(fig, fname)

def tune_threshold(y_true, y_prob, model_name: str) -> float:
    grid = np.linspace(0.05, 0.95, 37)
    rows = []
    for t in grid:
        y_hat = (y_prob >= t).astype(int)
        prec = precision_score(y_true, y_hat, zero_division=0)
        rec  = recall_score(y_true, y_hat, zero_division=0)
        f1s  = f1_score(y_true, y_hat, zero_division=0)
        beta = 2.0
        f2   = (1 + beta**2) * (prec * rec) / (beta**2 * prec + rec + 1e-12)
        tn, fp, fn, tp = confusion_matrix(y_true, y_hat).ravel()
        cost = 5*fn + 1*fp
        rows.append((t, prec, rec, f1s, f2, cost))
    df_thr = pd.DataFrame(rows, columns=["thr","precision","recall","f1","f2","cost"])
    fig, ax1 = new_fig(figsize=(7, 4))
    ax1.plot(df_thr["thr"], df_thr["precision"], label="Precision", color=COLOR_NO)
    ax1.plot(df_thr["thr"], df_thr["recall"],    label="Recall",    color=COLOR_YES)
    ax1.plot(df_thr["thr"], df_thr["f1"],        label="F1",        color=LINE_MEAN)
    ax1.plot(df_thr["thr"], df_thr["f2"],        label="F2",        color=LINE_MODE)
    ax1.set_xlabel("Threshold"); ax1.set_ylabel("Score"); ax1.set_ylim(0,1)
    ax1.legend(loc="upper right")
    ax2 = ax1.twinx()
    ax2.plot(df_thr["thr"], df_thr["cost"], color=ACCENT, linestyle="--", label="Cost (FN*5 + FP)")
    ax2.set_ylabel("Cost")
    fig.suptitle(f"Threshold tuning — {model_name}")
    save_and_show(fig, f"thr_tuning_{model_name.lower().replace(' ','_')}.png")
    best = df_thr.loc[df_thr["f2"].idxmax()]
    t_star = float(best["thr"])
    print(f"[TUNE] {model_name}: best F2 at thr={t_star:.3f} | "
          f"P={best['precision']:.3f} R={best['recall']:.3f} F2={best['f2']:.3f} Cost={best['cost']:.0f}")
    return t_star

def nice_confusion(y_true, y_pred, title, fname):
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    df_cm = pd.DataFrame(cm,
                         index=["True: No serious late", "True: Serious late"],
                         columns=["Pred: No serious late", "Pred: Serious late"])
    fig, ax = new_fig(figsize=(6, 5))
    sns.heatmap(df_cm, annot=True, fmt="d", cmap="Purples", cbar=False, ax=ax)
    ax.set_title(title)
    ax.text(0.0, -0.25,
            "Rows = actual, Columns = predicted. FP = flagged but not late. FN = missed serious late.",
            transform=ax.transAxes, ha="left", va="top", fontsize=9, color="#374151")
    save_and_show(fig, fname)
    tn, fp, fn, tp = cm.ravel()
    print(f"[EXPLAIN] {title}: TN={tn:,}, FP={fp:,}, FN={fn:,}, TP={tp:,}")

def feature_names_from_ct(ct: ColumnTransformer, X: pd.DataFrame):
    names, used = [], set()
    for name, trans, cols in ct.transformers_:
        if name == "remainder" and trans == "drop":
            continue
        if name != "remainder":
            cols_list = list(cols) if isinstance(cols, (list, tuple, np.ndarray, pd.Index)) else [cols]
            names.extend([str(c) for c in cols_list]); used.update(cols_list)
        else:
            rest = [c for c in X.columns if c not in used]
            names.extend(rest)
    return names

def perm_importance_plot(model, X_test, y_test, feat_names, title, fname, n_repeats=10):
    r = permutation_importance(model, X_test, y_test, n_repeats=n_repeats,
                               random_state=RANDOM_SEED, n_jobs=-1, scoring="roc_auc")
    imp = pd.DataFrame({"feature": feat_names, "importance": r.importances_mean}).sort_values("importance", ascending=False).head(12)
    fig, ax = new_fig(figsize=(8, 5))
    sns.barplot(data=imp, x="importance", y="feature", ax=ax, color=COLOR_NO)
    ax.set_title(title); ax.set_xlabel("Permutation importance (AUC drop)")
    save_and_show(fig, fname)
    return imp

def run_models(df: pd.DataFrame) -> dict:
    assert TARGET in df.columns, f"Missing target column '{TARGET}'."
    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(int).values

    logreg, rf = build_pipes(X)
    kfold_report(logreg, X, y, "Logistic Regression")
    kfold_report(rf, X, y, "Random Forest")

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y)

    # Logistic Regression
    logreg.fit(Xtr, ytr)
    yhat_lr  = logreg.predict(Xte)
    yprob_lr = logreg.predict_proba(Xte)[:, 1]
    auc_lr   = roc_auc_score(yte, yprob_lr)

    print("\n=== Logistic Regression (holdout) ===")
    print(f"Accuracy : {accuracy_score(yte, yhat_lr):.4f}")
    print(f"Precision: {precision_score(yte, yhat_lr, zero_division=0):.4f}")
    print(f"Recall   : {recall_score(yte, yhat_lr, zero_division=0):.4f}")
    print(f"F1-score : {f1_score(yte, yhat_lr, zero_division=0):.4f}")
    print(f"ROC AUC  : {auc_lr:.4f}")

    nice_confusion(yte, yhat_lr, "Confusion matrix — Logistic Regression", "cm_logreg.png")

    fpr, tpr, thr = roc_curve(yte, yprob_lr)
    fig, ax = new_fig(figsize=(6, 5))
    ax.plot(fpr, tpr, color=COLOR_NO, label=f"LogReg (AUC={auc_lr:.3f})")
    ax.plot([0,1],[0,1], linestyle="--", color="#9CA3AF")
    idx = (np.abs(thr - 0.5)).argmin()
    ax.scatter([fpr[idx]], [tpr[idx]], s=60, label="Threshold ≈ 0.5", zorder=5)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve — Logistic Regression"); ax.legend()
    save_and_show(fig, "roc_logreg.png")

    pr_curve(yte, yprob_lr, "Precision–Recall — Logistic Regression", "pr_logreg.png")
    t_star_lr = tune_threshold(yte, yprob_lr, "Logistic Regression")
    yhat_lr_tuned = (yprob_lr >= t_star_lr).astype(int)
    nice_confusion(yte, yhat_lr_tuned, f"Confusion (LogReg @ {t_star_lr:.2f})", "cm_logreg_tuned.png")

    feats_lr = feature_names_from_ct(logreg.named_steps["prep"], Xtr)
    perm_importance_plot(logreg, Xte, yte, feats_lr, "Permutation importance — Logistic Regression", "pi_logreg.png")

    cal_lr = CalibratedClassifierCV(estimator=logreg, method="sigmoid", cv=3)
    cal_lr.fit(Xtr, ytr)
    fig, ax = new_fig(figsize=(5, 5))
    CalibrationDisplay.from_estimator(cal_lr, Xte, yte, n_bins=10, ax=ax)
    ax.set_title("Reliability curve — Logistic Regression (sigmoid)")
    save_and_show(fig, "calibration_logreg.png")

    # Random Forest
    rf.fit(Xtr, ytr)
    yhat_rf  = rf.predict(Xte)
    yprob_rf = rf.predict_proba(Xte)[:, 1]
    auc_rf   = roc_auc_score(yte, yprob_rf)

    print("\n=== Random Forest (holdout) ===")
    print(f"Accuracy : {accuracy_score(yte, yhat_rf):.4f}")
    print(f"Precision: {precision_score(yte, yhat_rf, zero_division=0):.4f}")
    print(f"Recall   : {recall_score(yte, yhat_rf, zero_division=0):.4f}")
    print(f"F1-score : {f1_score(yte, yhat_rf, zero_division=0):.4f}")
    print(f"ROC AUC  : {auc_rf:.4f}")

    nice_confusion(yte, yhat_rf, "Confusion matrix — Random Forest", "cm_rf.png")

    fpr2, tpr2, thr2 = roc_curve(yte, yprob_rf)
    fig, ax = new_fig(figsize=(6, 5))
    ax.plot(fpr2, tpr2, color=COLOR_YES, label=f"RF (AUC={auc_rf:.3f})")
    ax.plot([0,1],[0,1], linestyle="--", color="#9CA3AF")
    idx2 = (np.abs(thr2 - 0.5)).argmin()
    ax.scatter([fpr2[idx2]], [tpr2[idx2]], s=60, label="Threshold ≈ 0.5", zorder=5)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve — Random Forest"); ax.legend()
    save_and_show(fig, "roc_rf.png")

    pr_curve(yte, yprob_rf, "Precision–Recall — Random Forest", "pr_rf.png")
    t_star_rf = tune_threshold(yte, yprob_rf, "Random Forest")
    yhat_rf_tuned = (yprob_rf >= t_star_rf).astype(int)
    nice_confusion(yte, yhat_rf_tuned, f"Confusion (RF @ {t_star_rf:.2f})", "cm_rf_tuned.png")

    feats_rf = feature_names_from_ct(rf.named_steps["prep"], Xtr)
    perm_importance_plot(rf, Xte, yte, feats_rf, "Permutation importance — Random Forest", "pi_rf.png")

    # Exports for dashboard
    os.makedirs("exports", exist_ok=True)
    pd.DataFrame({
        "y_true": yte,
        "proba_logreg": yprob_lr,
        "proba_rf": yprob_rf
    }).to_csv("exports/holdout_predictions.csv", index=False)

    grid = np.linspace(0.05, 0.95, 37)
    rows = []
    for t in grid:
        for name, probs in [("LogReg", yprob_lr), ("RandomForest", yprob_rf)]:
            yhat = (probs >= t).astype(int)
            rows.append({
                "model": name,
                "threshold": t,
                "precision": precision_score(yte, yhat, zero_division=0),
                "recall": recall_score(yte, yhat, zero_division=0),
                "f1": f1_score(yte, yhat, zero_division=0)
            })
    pd.DataFrame(rows).to_csv("exports/threshold_metrics.csv", index=False)
    print("[SAVE] Exports written to ./exports for the dashboard.")

    # === Exam extras: model summary + lifts/gains + decile CSVs ===
    summ_lr = summarize_model(yte, yprob_lr, yhat_lr_tuned, "Logistic Regression", t_star_lr)
    summ_rf = summarize_model(yte, yprob_rf, yhat_rf_tuned, "Random Forest", t_star_rf)
    summary_df = pd.DataFrame([summ_lr, summ_rf])
    summary_df.to_csv("exports/model_eval_summary.csv", index=False)
    print("[SAVE] exports/model_eval_summary.csv")

    lift_and_gains_plot(yte, yprob_lr, "Logistic Regression", "logreg")
    lift_and_gains_plot(yte, yprob_rf, "Random Forest", "rf")

    # Capacity-aware operating point (example: top 5%)
    cap_rate = 0.05  # change this to your review capacity
    thr_lr_cap = threshold_for_top_rate(yprob_lr, cap_rate)
    thr_rf_cap = threshold_for_top_rate(yprob_rf, cap_rate)
    cap_lr = eval_at_threshold(yte, yprob_lr, thr_lr_cap)
    cap_rf = eval_at_threshold(yte, yprob_rf, thr_rf_cap)
    pd.DataFrame([
        {"model":"LogReg (top 5%)", **cap_lr},
        {"model":"RF (top 5%)",     **cap_rf},
    ]).to_csv("exports/capacity_5pct_summary.csv", index=False)
    print("[SAVE] exports/capacity_5pct_summary.csv")
    print("[CAPACITY] Top 5% flagged — LogReg:",
          f"flagged={cap_lr['flagged']:,}, prec={cap_lr['precision']:.2f}, recall={cap_lr['recall']:.2f}")
    print("[CAPACITY] Top 5% flagged — RF    :",
          f"flagged={cap_rf['flagged']:,}, prec={cap_rf['precision']:.2f}, recall={cap_rf['recall']:.2f}")

    # Fairness audits at tuned thresholds (you can also audit at capacity thr if needed)
    save_fairness_audits(yte, yprob_lr, Xte, t_star_lr, prefix="logreg_tuned")
    save_fairness_audits(yte, yprob_rf, Xte, t_star_rf, prefix="rf_tuned")

    # Save fitted models for scoring
    try:
        from joblib import dump
        os.makedirs("artifacts", exist_ok=True)
        dump(logreg, "artifacts/logreg_pipeline.joblib")
        dump(rf,     "artifacts/rf_pipeline.joblib")
        print("[SAVE] artifacts/logreg_pipeline.joblib")
        print("[SAVE] artifacts/rf_pipeline.joblib")
    except Exception as e:
        print("[WARN] Could not save joblib artifacts:", e)

    # Return metrics for key_numbers + model card
    return {
        # AUC/AP & CIs
        "auc_logreg_holdout": float(summ_lr["AUC"]),
        "auc_rf_holdout": float(summ_rf["AUC"]),
        "auc_logreg_ci_low": float(summ_lr["AUC_CI_low"]),
        "auc_logreg_ci_high": float(summ_lr["AUC_CI_high"]),
        "auc_rf_ci_low": float(summ_rf["AUC_CI_low"]),
        "auc_rf_ci_high": float(summ_rf["AUC_CI_high"]),
        "ap_logreg": float(summ_lr["AP"]),
        "ap_rf": float(summ_rf["AP"]),
        # Brier, Gini, KS
        "brier_logreg": float(summ_lr["Brier"]),
        "brier_rf": float(summ_rf["Brier"]),
        "gini_logreg": float(summ_lr["Gini"]),
        "gini_rf": float(summ_rf["Gini"]),
        "ks_logreg": float(summ_lr["KS"]),
        "ks_rf": float(summ_rf["KS"]),
        # thresholds + ops
        "threshold_logreg": float(summ_lr["threshold"]),
        "threshold_rf": float(summ_rf["threshold"]),
        "flagged_logreg": int(summ_lr["Flagged_at_thr"]),
        "flagged_rf": int(summ_rf["Flagged_at_thr"]),
        "tp_logreg": int(summ_lr["TP"]),
        "fp_logreg": int(summ_lr["FP"]),
        "fn_logreg": int(summ_lr["FN"]),
        "tp_rf": int(summ_rf["TP"]),
        "fp_rf": int(summ_rf["FP"]),
        "fn_rf": int(summ_rf["FN"]),
        "recall_logreg": float(summ_lr["Recall_at_thr"]),
        "recall_rf": float(summ_rf["Recall_at_thr"]),
        "precision_logreg": float(summ_lr["Precision_at_thr"]),
        "precision_rf": float(summ_rf["Precision_at_thr"]),
        "f1_logreg": float(summ_lr["F1_at_thr"]),
        "f1_rf": float(summ_rf["F1_at_thr"]),
    }

# ========================== CLI Scorer & Model Card ========================

def write_predict_cli_script():
    os.makedirs("tools", exist_ok=True)
    code = r'''#!/usr/bin/env python3
import sys, os
import pandas as pd
from joblib import load

def main():
    if len(sys.argv) < 3:
        print("Usage: python tools/predict_cli.py <model: logreg|rf> <input_csv> [output_csv]")
        sys.exit(1)
    model_name = sys.argv[1].lower()
    in_csv     = sys.argv[2]
    out_csv    = sys.argv[3] if len(sys.argv) > 3 else "predictions.csv"

    model_key = "logreg" if model_name == "logreg" else "rf"
    model_path = os.path.join("artifacts", f"{model_key}_pipeline.joblib")
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}. Run main.py first.")
        sys.exit(2)

    pipe = load(model_path)
    df = pd.read_csv(in_csv)
    # If target present, drop it for scoring
    if "SeriousDlqin2yrs" in df.columns:
        df = df.drop(columns=["SeriousDlqin2yrs"])
    proba = pipe.predict_proba(df)[:,1]
    out = pd.DataFrame({"proba": proba})
    out.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

if __name__ == "__main__":
    main()
'''
    with open("tools/predict_cli.py", "w", encoding="utf-8") as f:
        f.write(code)
    print("[SAVE] tools/predict_cli.py")

def write_model_card(df: pd.DataFrame, metrics: dict):
    os.makedirs("reports", exist_ok=True)
    total = len(df); late = int(df[TARGET].sum()) if TARGET in df.columns else None
    prev  = (late/total) if (late is not None and total) else None

    lines = []
    lines.append("# Model Card: Loan Default Prediction\n")
    lines.append("**Owner:** Tatjana Aasgaard  \n**Date:** autogenerated\n")
    lines.append("## 1. Intended Use\nPredict probability that a client becomes 90+ days late within 2 years, to prioritize manual review and adjust credit decisions.\n")
    lines.append("## 2. Data\n")
    lines.append(f"- Samples: **{total:,}**; Target prevalence: **{prev:.1%}** (if available)\n" if prev is not None else f"- Samples: **{total:,}**\n")
    lines.append("- Key fields used: utilization, age, debt ratio, monthly income, credit lines/loans, delinquencies, mortgages, dependents.\n")
    lines.append("## 3. Training/Validation\nStratified 80/20 split; 5-fold CV for AUC/AP. Models: Logistic Regression (balanced) and Random Forest (balanced class_weight).\n")
    lines.append("## 4. Performance (holdout)\n")
    for m in ["logreg","rf"]:
        lines.append(f"### {m.upper()}\n")
        lines.append(f"- AUC: **{metrics.get(f'auc_{m}_holdout', float('nan')):.3f}** "
                     f"(95% CI {metrics.get(f'auc_{m}_ci_low', float('nan')):.3f}–{metrics.get(f'auc_{m}_ci_high', float('nan')):.3f})\n")
        lines.append(f"- AP: **{metrics.get(f'ap_{m}', float('nan')):.3f}**; "
                     f"Brier: **{metrics.get(f'brier_{m}', float('nan')):.3f}**; "
                     f"Gini: **{metrics.get(f'gini_{m}', float('nan')):.3f}**; "
                     f"KS: **{metrics.get(f'ks_{m}', float('nan')):.3f}**\n")
        lines.append(f"- Tuned threshold: **{metrics.get(f'threshold_{m}', float('nan')):.3f}**  \n")
        lines.append(f"  Precision: **{metrics.get(f'precision_{m}', float('nan')):.3f}**, "
                     f"Recall: **{metrics.get(f'recall_{m}', float('nan')):.3f}**, "
                     f"F1: **{metrics.get(f'f1_{m}', float('nan')):.3f}**\n")
        lines.append(f"  Flagged: **{metrics.get(f'flagged_{m}', 0):,}**; "
                     f"TP: **{metrics.get(f'tp_{m}', 0):,}**, FP: **{metrics.get(f'fp_{m}', 0):,}**, FN: **{metrics.get(f'fn_{m}', 0):,}**\n")
    lines.append("## 5. Operating Point Selection\nTwo options considered: (a) best F2 (recall-oriented), (b) capacity-aware (e.g., top 5% highest risk). Final selection should match review capacity and cost of FN vs FP.\n")
    lines.append("## 6. Fairness & Error Balance\nAudits run over age groups, dependents, and mortgage status (see `reports/fairness/`). We monitor TPR/FPR gaps and adjust thresholding/policies if disparities exceed 5–10%.\n")
    lines.append("## 7. Interpretability\nPermutation importance charts provided. For policy changes, review feature constraints to avoid disadvantaging protected groups.\n")
    lines.append("## 8. Risk & Limitations\n- Historical bias may exist.  \n- Not all drivers are causal.  \n- Model may drift; schedule quarterly re-validation.  \n- Use human oversight on edge cases.\n")
    lines.append("## 9. Deployment & Monitoring\nExported pipelines in `artifacts/`. For production, log score drift, prevalence, calibration, and fairness deltas; retrain upon material change.\n")

    path = os.path.join("reports", "model_card.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[SAVE] {path}")

# ================================ Main =====================================

def main() -> None:
    make_output_folder()
    df = load_csv()
    save_run_environment()
    write_min_requirements()
    print("Raw shape:", df.shape)
    print("\n[Preview] First 5 rows:\n", df.head())

    df, clean_summary = clean_data(df)
    print("\n=== Cleaning summary ===")
    for k, v in clean_summary.items():
        print(f"{k:>26}: {v:,}")
    save_cleaning_report(clean_summary)
    print("\nClean shape:", df.shape)

    make_eda_charts(df)
    write_figure_captions()
    metrics = run_models(df)  # includes exam extras, capacity, fairness, saves models

    print("\nAll done. Figures saved in:", OUTPUT_DIR)
    build_figures_index()
    build_key_numbers(df, metrics)
    write_model_card(df, metrics)
    write_predict_cli_script()

    if os.name == "nt":
        try:
            os.startfile(os.path.abspath(OUTPUT_DIR))
        except Exception:
            pass

if __name__ == "__main__":
    main()
