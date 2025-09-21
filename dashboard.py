# dashboard.py
# Loan Default Dashboard — Core + Advanced (Option A)
# - Core pages: Introduction, Feature Views, Model & Threshold, Risk Buckets (A–D), Client Credit Check
# - Advanced page: Relationships & Segments, Correlation, KS/Lift, Permutation Importance, Saved Figures, Data Quality
# - Teal styling, aligned sidebar, integer Age/Dependents, whole-number Income
# - Client Credit Check includes utilization & debt-ratio mini-calculators + hardened number_input ranges

from __future__ import annotations

import os, glob
from typing import Optional, List
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_fscore_support,
    precision_recall_curve, auc as sk_auc,
    confusion_matrix, accuracy_score, brier_score_loss
)
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance

# ---------------- Page + Styles ----------------
st.set_page_config(page_title="Loan Default — Dashboard", layout="wide")

st.markdown("""
<style>
:root {
  --teal: #007c82;
  --teal-light: #e6f6f7;
  --ink: #0f172a;
  --muted: #475569;
  --radius: 16px;
  --shadow: 0 6px 18px rgba(0,0,0,0.08);
}
/* Titles */
.big-title { background: var(--teal); color:#fff!important; padding:18px 22px;
  border-radius: var(--radius); box-shadow: var(--shadow); font-size:1.6rem;
  font-weight:700; margin:8px 0 18px 0; letter-spacing:.2px; }
.section-title { background: var(--teal-light); border:2px solid var(--teal); color:var(--ink);
  padding:10px 14px; border-radius:12px; box-shadow: var(--shadow); font-size:1.05rem;
  font-weight:700; margin:8px 0 10px 0; }
.note { color: var(--muted); font-size:.95rem; margin:6px 2px 14px 2px; }
.block { background:#fff; border-radius: var(--radius); box-shadow: var(--shadow); padding:14px; }
hr.separator { border:none; height:1px; background:#e2e8f0; margin:18px 0; }
.stMetric > div { justify-content: center; }

/* Sidebar polish */
.css-1d391kg, .css-1d391kg e1fqkh3o4 { width: 300px; } /* older Streamlit fallback */
section[data-testid="stSidebar"] { min-width: 300px; max-width: 300px; }
[data-testid="stSidebar"] .stRadio > label, [data-testid="stSidebar"] .stRadio div { width: 100%; }
[data-testid="stSidebar"] .stRadio label { font-weight: 600; }
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label { padding: 2px 0; }
</style>
""", unsafe_allow_html=True)

def big_title(t: str): st.markdown(f'<div class="big-title">{t}</div>', unsafe_allow_html=True)
def section_title(t: str): st.markdown(f'<div class="section-title">{t}</div>', unsafe_allow_html=True)
def pct(x: float) -> str: return f"{100*x:.1f}%"

DISPLAY = {
    "age": "Age",
    "MonthlyIncome": "Monthly Income",
    "DebtRatio": "Debt Ratio",
    "RevolvingUtilizationOfUnsecuredLines": "Card/Line Utilization",
    "NumberOfOpenCreditLinesAndLoans": "Open Credit Lines/Loans",
    "NumberOfTimes90DaysLate": "Times 90+ Days Late",
    "NumberOfTime60-89DaysPastDueNotWorse": "Times 60–89 Days Late",
    "NumberOfTime30-59DaysPastDueNotWorse": "Times 30–59 Days Late",
    "NumberRealEstateLoansOrLines": "Real Estate Loans/Lines",
    "NumberOfDependents": "Dependents",
    "SeriousDlqin2yrs": "Serious Delinquency (within 2 Years)"
}

# ---------------- Data helpers ----------------
REQUIRED_COLS = {
    "SeriousDlqin2yrs","RevolvingUtilizationOfUnsecuredLines","age",
    "NumberOfTime30-59DaysPastDueNotWorse","DebtRatio","MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans","NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines","NumberOfTime60-89DaysPastDueNotWorse","NumberOfDependents"
}

@st.cache_data(show_spinner=False)
def find_dataset() -> Optional[str]:
    patterns = ["data/*.csv","datasets/*.csv","*.csv","input/*.csv","inputs/*.csv"]
    found: List[str] = []
    for p in patterns: found += glob.glob(p)
    priority = ["data/clean_credit.csv","data/credit_clean.csv","data/credit.csv","cs-training.csv","train.csv"]
    ordered, seen = [], set()
    for p in priority:
        if p in found and p not in seen: ordered.append(p); seen.add(p)
    for f in found:
        if f not in seen: ordered.append(f); seen.add(f)
    for path in ordered:
        try: sample = pd.read_csv(path, nrows=5)
        except Exception: continue
        cols = set(c.strip() for c in sample.columns)
        if "SeriousDlqin2yrs" in cols and len(cols & REQUIRED_COLS) >= max(7, int(.6*len(REQUIRED_COLS))):
            return path
    return None

def _to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # clean display (no decimals for Age/Income)
    if "age" in df.columns:
        df["age"] = df["age"].round().astype("Int64")
    if "MonthlyIncome" in df.columns:
        df["MonthlyIncome"] = df["MonthlyIncome"].round(0).astype("Int64")
    return df

@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df = _to_numeric(df)
    if "SeriousDlqin2yrs" in df.columns: df = df.dropna(subset=["SeriousDlqin2yrs"])
    if "age" in df.columns: df = df[df["age"].fillna(0) > 0]
    return df

csv_path = find_dataset()
if not csv_path:
    big_title("Loan Default — Dashboard")
    st.error("No compatible CSV found. Add e.g. `data/clean_credit.csv` or `cs-training.csv` and refresh.")
    st.stop()
df_full = load_data(csv_path)

def safe_rate_by_bin(frame: pd.DataFrame, col: str, bins: int = 8) -> Optional[go.Figure]:
    try:
        if col not in frame.columns or frame[col].dropna().empty: return None
        tmp = frame[[col,"SeriousDlqin2yrs"]].dropna()
        if tmp[col].nunique() < 2: return None
        q = min(bins, max(2, tmp[col].nunique()))
        tmp["bin"] = pd.qcut(tmp[col], q=q, duplicates="drop")
        grp = tmp.groupby("bin")["SeriousDlqin2yrs"].mean().reset_index()
        grp["rate"] = 100*grp["SeriousDlqin2yrs"]
        grp["bin"] = grp["bin"].astype(str)
        fig = px.bar(grp, x="bin", y="rate",
                     labels={"bin": DISPLAY.get(col, col), "rate": "Default rate (%)"})
        fig.update_layout(margin=dict(l=10,r=10,t=8,b=10), height=320)
        return fig
    except Exception:
        return None

# ---------------- Sidebar nav ----------------
with st.sidebar:
    st.title("Navigation")
    st.caption(f"Data source: `{csv_path}`")
    page = st.radio(
        "Core",
        [
            "Introduction",
            "Feature Views",
            "Model & Threshold",
            "Risk Buckets (A–D)",
            "Client Credit Check",
            "Advanced (EDA & Diagnostics)",
            "Summary & Conclusion",
        ],
        index=0,
    )

# ---------------- 1) Introduction ----------------
if page == "Introduction":
    big_title("Loan Default Dashboard")

    # Optional hero image: place a JPG/PNG in assets/ or images/
    hero_paths = []
    for pat in ["assets/*.jpg","assets/*.png","images/*.jpg","images/*.png"]:
        hero_paths.extend(sorted(glob.glob(pat)))
    if hero_paths:
        st.image(hero_paths[0], use_container_width=True)

    section_title("Purpose")
    st.markdown("""
This dashboard helps a lender identify borrowers who are more likely to fall into **serious delinquency within two years**.
The aim is practical: use a small set of familiar signals (utilization, debt ratio, late history, income, age, exposure)
with a transparent model to support **portfolio monitoring** and **case-by-case decisions**.
""")

    section_title("How to use this")
    st.markdown("""
1. **Feature Views** — see the patterns most linked with risk.  
2. **Model & Threshold** — check discrimination (ROC/PR), probability quality (calibration), and pick a threshold.  
3. **Risk Buckets (A–D)** — translate probabilities into policy buckets.  
4. **Client Credit Check** — enter a few fields and get a **GO / CAUTION / STOP** verdict.
""")

    section_title("Target & key terms")
    st.markdown(
        "- **Serious Delinquency (within 2 Years)**: target (0/1). *1 = serious delinquency occurred within two years.*\n"
        "- **Card/Line Utilization**: revolving balances ÷ credit limits (values near 1.0 mean lines are nearly maxed).\n"
        "- **Debt Ratio**: total monthly debt payments ÷ gross monthly income.\n"
    )

    section_title("Portfolio snapshot")
    c1,c2,c3,c4 = st.columns(4)
    rows = len(df_full)
    dr = df_full["SeriousDlqin2yrs"].mean() if rows else 0.0
    mi_missing = df_full["MonthlyIncome"].isna().sum() if "MonthlyIncome" in df_full.columns else 0
    age_min = int(df_full["age"].min()) if "age" in df_full.columns and df_full["age"].notna().any() else None
    age_max = int(df_full["age"].max()) if "age" in df_full.columns and df_full["age"].notna().any() else None
    c1.metric("Rows", f"{rows:,}")
    c2.metric("Default rate", pct(dr))
    c3.metric("Missing Monthly Income", f"{mi_missing:,}" if "MonthlyIncome" in df_full.columns else "—")
    c4.metric("Age span", f"{age_min}–{age_max}" if age_min is not None else "—")

    section_title("Scope & limits")
    st.markdown("""
This is a proof-of-concept using a historical credit dataset and a simple, explainable model (logistic regression with class balancing).
Probabilities are estimates and should be combined with policy rules, affordability checks, and manual review.
With more features (employment length, loan-to-income details, bureau scores), accuracy would improve; the workflow remains the same.
""")

# ---------------- 2) Feature Views ----------------
elif page == "Feature Views":
    big_title("Feature Views")

    section_title("Serious Delinquency within 2 Years (0/1)")
    counts = df_full["SeriousDlqin2yrs"].value_counts().sort_index()
    fig = px.bar(x=[str(int(k)) for k in counts.index], y=counts.values,
                 labels={"x": DISPLAY["SeriousDlqin2yrs"], "y":"Count"})
    fig.update_layout(margin=dict(l=10,r=10,t=8,b=10), height=320)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('<div class="note">0 = no serious delinquency within two years; 1 = serious delinquency occurred.</div>', unsafe_allow_html=True)

    essentials = [c for c in [
        "RevolvingUtilizationOfUnsecuredLines","DebtRatio","NumberOfTimes90DaysLate",
        "MonthlyIncome","age"
    ] if c in df_full.columns]

    for c in essentials:
        section_title(DISPLAY.get(c, c))
        col1, col2 = st.columns([2,1])
        with col1:
            h = px.histogram(df_full, x=c, nbins=60, labels={c: DISPLAY.get(c, c)})
            h.update_layout(height=300, margin=dict(l=10,r=10,t=8,b=10))
            st.plotly_chart(h, use_container_width=True)
        with col2:
            # 1-line takeaway placeholders (adjust text to your dataset after a look)
            if c == "RevolvingUtilizationOfUnsecuredLines":
                st.markdown("- Risk rises as utilization approaches **100%**.")
            elif c == "DebtRatio":
                st.markdown("- Very high ratios are a **warning sign**.")
            elif c == "NumberOfTimes90DaysLate":
                st.markdown("- Any 90+ day late **lifts risk**; repeats lift it further.")
            elif c == "MonthlyIncome":
                st.markdown("- Lower income buckets trend **riskier** at the tails.")
            elif c == "age":
                st.markdown("- Younger borrowers show **higher risk** in the tail.")
        st.markdown('<hr class="separator" />', unsafe_allow_html=True)

# ---------------- 3) Model & Threshold ----------------
elif page == "Model & Threshold":
    big_title("Model & Threshold")

    y = df_full["SeriousDlqin2yrs"].astype(int)
    X_cols = [c for c in df_full.columns if c != "SeriousDlqin2yrs" and pd.api.types.is_numeric_dtype(df_full[c])]
    X_df = df_full[X_cols].copy().fillna(df_full[X_cols].median(numeric_only=True))

    TEST_SIZE = 0.20
    X_train_df, X_test_df, y_train, y_test = train_test_split(X_df, y, test_size=TEST_SIZE, random_state=42, stratify=y)
    scaler = StandardScaler(); X_train = scaler.fit_transform(X_train_df); X_test = scaler.transform(X_test_df)
    lr = LogisticRegression(max_iter=500, solver="liblinear", class_weight="balanced"); lr.fit(X_train, y_train)
    y_proba = lr.predict_proba(X_test)[:,1]; auc = roc_auc_score(y_test, y_proba)

    # share for other pages
    st.session_state["__model_cols__"] = X_cols
    st.session_state["__scaler_mean__"] = scaler.mean_.tolist()
    st.session_state["__scaler_scale__"] = scaler.scale_.tolist()
    st.session_state["__coef__"] = lr.coef_[0].tolist()
    st.session_state["__intercept__"] = float(lr.intercept_[0])
    st.session_state["__X_test_index__"] = X_test_df.index.tolist()
    st.session_state["__y_test__"] = y_test.tolist()
    st.session_state["__y_proba__"] = y_proba.tolist()

    section_title(f"ROC curve (AUC = {auc:.3f})")
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_fig = go.Figure()
    roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC"))
    roc_fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Baseline", line=dict(dash="dash")))
    roc_fig.update_layout(xaxis_title="False positive rate", yaxis_title="True positive rate",
                          height=380, margin=dict(l=10,r=10,t=40,b=10))
    st.plotly_chart(roc_fig, use_container_width=True)

    section_title("Threshold (maximize F1 by default)")
    ths = np.linspace(0.05, 0.95, 181)
    best_th, best_f1 = 0.50, -1
    for th in ths:
        pred = (y_proba >= th).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(y_test, pred, average="binary", zero_division=0)
        if f1 > best_f1: best_f1, best_th = f1, th
    th = st.slider("Decision threshold", 0.10, 0.90, float(best_th), step=0.01)
    y_pred = (y_proba >= th).astype(int)

    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=[0,1])
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Accuracy", f"{acc:.3f}")
    c2.metric("Precision (1)", f"{prec:.3f}")
    c3.metric("Recall (1)", f"{rec:.3f}")
    c4.metric("F1 (1)", f"{f1:.3f}")

    section_title("Confusion matrix")
    cm_fig = go.Figure(data=go.Heatmap(
        z=cm, x=["Pred 0","Pred 1"], y=["True 0","True 1"], text=cm, texttemplate="%{text}", showscale=False))
    cm_fig.update_layout(height=320, margin=dict(l=10,r=10,t=8,b=10))
    st.plotly_chart(cm_fig, use_container_width=True)
    tn, fp, fn, tp = cm.ravel()
    st.markdown(
        f"<div class='note'>"
        f"**TN:** {tn:,} non-defaulters correctly predicted. "
        f"**FP:** {fp:,} false alarms. "
        f"**FN:** {fn:,} missed defaulters. "
        f"**TP:** {tp:,} defaulters correctly flagged."
        f"</div>", unsafe_allow_html=True
    )

    # Precision–Recall + Calibration
    section_title("Precision–Recall & Calibration")
    precs, recs, _thr = precision_recall_curve(y_test, y_proba)
    pr_auc = sk_auc(recs, precs)
    pr_fig = go.Figure()
    pr_fig.add_trace(go.Scatter(x=recs, y=precs, mode="lines", name=f"PR (AUC={pr_auc:.3f})"))
    pr_fig.update_layout(xaxis_title="Recall", yaxis_title="Precision", height=300, margin=dict(l=10,r=10,t=40,b=10))

    prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins=10, strategy="quantile")
    brier = brier_score_loss(y_test, y_proba)
    cal_fig = go.Figure()
    cal_fig.add_trace(go.Scatter(x=prob_pred, y=prob_true, mode="lines+markers", name="Calibration"))
    cal_fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Perfect", line=dict(dash="dash")))
    cal_fig.update_layout(xaxis_title="Predicted probability", yaxis_title="Observed default rate",
                          height=300, margin=dict(l=10,r=10,t=40,b=10))

    cL, cR = st.columns(2)
    with cL: st.plotly_chart(pr_fig, use_container_width=True)
    with cR: st.plotly_chart(cal_fig, use_container_width=True)
    st.caption(f"Brier score: **{brier:.4f}** (lower is better).")

# ---------------- 4) Risk Buckets (A–D) ----------------
elif page == "Risk Buckets (A–D)":
    big_title("Risk Buckets (A–D) & Likely Defaulters")

    # Ensure model/scores exist
    if "__y_proba__" not in st.session_state:
        y_all = df_full["SeriousDlqin2yrs"].astype(int)
        X_cols_all = [c for c in df_full.columns if c != "SeriousDlqin2yrs" and pd.api.types.is_numeric_dtype(df_full[c])]
        X_all = df_full[X_cols_all].copy().fillna(df_full[X_cols_all].median(numeric_only=True))
        Xtr, Xte, ytr, yte = train_test_split(X_all, y_all, test_size=0.2, random_state=42, stratify=y_all)
        sc = StandardScaler(); Xtr_sc = sc.fit_transform(Xtr); Xte_sc = sc.transform(Xte)
        lr = LogisticRegression(max_iter=500, solver="liblinear", class_weight="balanced"); lr.fit(Xtr_sc, ytr)
        y_pr = lr.predict_proba(Xte_sc)[:,1]
        st.session_state["__model_cols__"] = X_cols_all
        st.session_state["__scaler_mean__"] = sc.mean_.tolist()
        st.session_state["__scaler_scale__"] = sc.scale_.tolist()
        st.session_state["__coef__"] = lr.coef_[0].tolist()
        st.session_state["__intercept__"] = float(lr.intercept_[0])
        st.session_state["__X_test_index__"] = Xte.index.tolist()
        st.session_state["__y_test__"] = yte.tolist()
        st.session_state["__y_proba__"] = y_pr.tolist()

    idx = st.session_state["__X_test_index__"]
    y_test = st.session_state["__y_test__"]
    y_proba = st.session_state["__y_proba__"]

    scored = pd.DataFrame({"prob_default": y_proba, "true_label": y_test}, index=idx)
    extras = [c for c in ["MonthlyIncome","age","DebtRatio","RevolvingUtilizationOfUnsecuredLines",
                          "NumberOfOpenCreditLinesAndLoans","NumberOfDependents"] if c in df_full.columns]
    scored = scored.join(df_full.loc[scored.index, extras])

    id_candidates = [c for c in ["ID","Id","id","CustomerID","customer_id"] if c in df_full.columns]
    id_col = id_candidates[0] if id_candidates else None
    scored["BorrowerID"] = df_full.loc[scored.index, id_col].values if id_col else scored.index

    section_title("Bucket cutoffs")
    colA,colB,colC = st.columns(3)
    cut_A = colA.number_input("A/B cutoff", 0.0, 0.99, 0.10, step=0.01)
    cut_B = colB.number_input("B/C cutoff", cut_A, 0.995, 0.25, step=0.01)
    cut_C = colC.number_input("C/D cutoff", cut_B, 1.00, 0.50, step=0.01)
    st.caption(f"A < {cut_A:.2f}  •  {cut_A:.2f}–{cut_B:.2f} = B  •  {cut_B:.2f}–{cut_C:.2f} = C  •  D ≥ {cut_C:.2f}")
    st.markdown("<div class='note'>Cutoffs split borrowers into four risk bands. Move these to balance flagged volume vs risk.</div>", unsafe_allow_html=True)

    def bucketize(p: float)->str:
        if p < cut_A: return "A"
        if p < cut_B: return "B"
        if p < cut_C: return "C"
        return "D"
    scored["bucket"] = scored["prob_default"].apply(bucketize)

    section_title("Portfolio mix by bucket")
    mix = (scored.groupby("bucket")["prob_default"]
           .agg(count="size", avg_prob="mean").reindex(["A","B","C","D"]).fillna(0))
    mix["default_rate(%)"] = (100*mix["avg_prob"]).round(1)
    mix = mix.astype({"count":int})
    st.dataframe(mix.reset_index().rename(columns={"bucket":"Bucket","count":"Borrowers",
                                                   "default_rate(%)":"Avg predicted default (%)"}),
                 use_container_width=True)
    fig = px.bar(mix.reset_index(), x="bucket", y="count", labels={"bucket":"Bucket","count":"Borrowers"})
    fig.update_layout(height=300, margin=dict(l=10,r=10,t=8,b=10))
    st.plotly_chart(fig, use_container_width=True)

    section_title("Bucket D — likely defaulters (top N)")
    n_top = st.slider("Rows to show", 5, 200, 25, step=5)
    cols_show = ["BorrowerID","prob_default","true_label"] + extras + ["bucket"]
    topD = scored[scored["bucket"]=="D"].sort_values("prob_default", ascending=False).head(n_top)
    st.dataframe(topD[cols_show], use_container_width=True)

    csv = scored[["BorrowerID","prob_default","true_label","bucket"] + extras].sort_values("prob_default", ascending=False).to_csv(index=False)
    st.download_button("Download full scored test set (CSV)", data=csv, file_name="scored_test_with_buckets.csv", mime="text/csv")

# ---------------- 5) Client Credit Check ----------------
elif page == "Client Credit Check":
    big_title("Client Credit Check — Enter borrower details")
    st.caption("Fill in the boxes below to get a GO / CAUTION / STOP verdict with a risk gauge.")

    # Ensure a model exists (reuse or build)
    if "__model_cols__" not in st.session_state:
        y_all = df_full["SeriousDlqin2yrs"].astype(int)
        X_cols_all = [c for c in df_full.columns if c != "SeriousDlqin2yrs" and pd.api.types.is_numeric_dtype(df_full[c])]
        X_all = df_full[X_cols_all].copy().fillna(df_full[X_cols_all].median(numeric_only=True))
        sc = StandardScaler().fit(X_all)
        lr = LogisticRegression(max_iter=500, solver="liblinear", class_weight="balanced").fit(sc.transform(X_all), y_all)
        st.session_state["__model_cols__"] = X_cols_all
        st.session_state["__scaler_mean__"] = sc.mean_.tolist()
        st.session_state["__scaler_scale__"] = sc.scale_.tolist()
        st.session_state["__coef__"] = lr.coef_[0].tolist()
        st.session_state["__intercept__"] = float(lr.intercept_[0])

    model_cols = st.session_state["__model_cols__"]
    scaler_mean = np.array(st.session_state["__scaler_mean__"])
    scaler_scale = np.array(st.session_state["__scaler_scale__"])
    coef = np.array(st.session_state["__coef__"])
    intercept = float(st.session_state["__intercept__"])

    def data_range(col, qlo=0.10, qhi=0.90, fallback_max=100, hard_min=0):
        if col not in df_full.columns or df_full[col].dropna().empty:
            lo, hi = hard_min, max(hard_min + 10, fallback_max)
            default = (lo + hi) // 2
            return int(lo), int(hi), int(default)
        s = pd.to_numeric(df_full[col], errors="coerce").dropna()
        ql = float(np.nanquantile(s, qlo))
        qh = float(np.nanquantile(s, qhi))
        lo = int(min(ql, qh, s.min()))
        hi = int(max(ql, qh, s.max()))
        lo = max(int(hard_min), lo)
        if hi <= lo: hi = lo + 1
        default = int((lo + hi) // 2)
        return lo, hi, default

    def clamp(val, lo, hi): return max(lo, min(hi, val))

    a_min, a_max, a_def = data_range("age", fallback_max=100, hard_min=18)
    m_min, m_max, m_def = data_range("MonthlyIncome", fallback_max=100000, hard_min=0)
    u_min, u_max, u_def = data_range("RevolvingUtilizationOfUnsecuredLines", fallback_max=200, hard_min=0)
    d_min, d_max, d_def = data_range("DebtRatio", fallback_max=300, hard_min=0)
    l_min, l_max, l_def = data_range("NumberOfOpenCreditLinesAndLoans", fallback_max=20, hard_min=0)
    dep_min, dep_max, dep_def = data_range("NumberOfDependents", fallback_max=6, hard_min=0)
    t30_max = int(max(3, float(df_full.get("NumberOfTime30-59DaysPastDueNotWorse", pd.Series([0])).max() or 0)))
    t60_max = int(max(2, float(df_full.get("NumberOfTime60-89DaysPastDueNotWorse", pd.Series([0])).max() or 0)))
    t90_max = int(max(2, float(df_full.get("NumberOfTimes90DaysLate", pd.Series([0])).max() or 0)))

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age", min_value=a_min, max_value=a_max, value=clamp(a_def, a_min, a_max), step=1, key="cc_age")
        income = st.number_input("Monthly Income", min_value=m_min, max_value=m_max, value=clamp(m_def, m_min, m_max), step=100, key="cc_inc")

        st.markdown("**Card/Line Utilization**")
        with st.expander("What is this? (tap to help)", expanded=False):
            st.markdown("""
**Card/Line Utilization = Total revolving balances ÷ Total credit limits**

- Revolving balances: current balances on credit cards/lines of credit.  
- Credit limits: total available credit across those same cards/lines.  

**Example:** balances 60,000 and limits 200,000 ⇒ 60,000 / 200,000 = **0.30 (30%)**.
Higher utilization (closer to 1.0 = 100%) usually means higher risk.
""")
            rev_bal = st.number_input("Total revolving balances", min_value=0, value=60000, step=1000, key="cc_revbal")
            tot_lim = st.number_input("Total credit limits",   min_value=1, value=200000, step=1000, key="cc_totlim")
            calc_util = round(rev_bal / tot_lim, 2)
            st.caption(f"Calculated Utilization: **{calc_util:.2f}** — copy below if helpful.")

        util = st.number_input(
            "Enter Utilization (e.g., 0.30 = 30%)",
            min_value=0.0, max_value=max(5.0, float(u_max)),
            value=float(clamp(calc_util if np.isfinite(calc_util) else 0.3, 0.0, max(5.0, float(u_max)))),
            step=0.01, key="cc_util",
            help="Utilization = total revolving balances ÷ total credit limits"
        )

    with col2:
        st.markdown("**Debt Ratio**")
        with st.expander("What is this? (tap to help)", expanded=False):
            st.markdown("""
**Debt Ratio = Total monthly debt payments ÷ Gross monthly income**

- Monthly debt payments: mortgage/rent (per policy), car/student/personal loans,
  minimum card payments, other finance plans.  
- Gross income: before-tax monthly income.

**Example:** 12,000 / 40,000 = **0.30 (30%)**.
""")
            m_pay = st.number_input("Monthly debt payments", min_value=0, value=12000, step=500, key="cc_mpay")
            g_inc = st.number_input("Gross monthly income",  min_value=1, value=40000, step=500, key="cc_ginc")
            calc_dr = round(m_pay / g_inc, 2)
            st.caption(f"Calculated Debt Ratio: **{calc_dr:.2f}** — copy below if helpful.")

        dratio = st.number_input(
            "Enter Debt Ratio (e.g., 0.30 = 30%)",
            min_value=0.0, max_value=max(1.0, float(d_max)),
            value=float(clamp(calc_dr if np.isfinite(calc_dr) else 0.3, 0.0, max(1.0, float(d_max)))),
            step=0.01, key="cc_dratio",
            help="Debt Ratio = total monthly debt payments ÷ gross monthly income"
        )

        open_lines = st.number_input("Open Credit Lines/Loans",
                                     min_value=l_min, max_value=l_max,
                                     value=clamp(l_def, l_min, l_max), step=1, key="cc_open")
        dependents = st.number_input("Dependents",
                                     min_value=dep_min, max_value=max(dep_max, dep_min+1),
                                     value=clamp(dep_def, dep_min, max(dep_max, dep_min+1)),
                                     step=1, key="cc_dep")

    with col3:
        late30 = st.number_input("Times 30–59 Days Late", min_value=0, max_value=max(1, t30_max), value=0, step=1, key="cc_30")
        late60 = st.number_input("Times 60–89 Days Late", min_value=0, max_value=max(1, t60_max), value=0, step=1, key="cc_60")
        late90 = st.number_input("Times 90+ Days Late",   min_value=0, max_value=max(1, t90_max), value=0, step=1, key="cc_90")

    # Build row & score
    med = df_full[model_cols].median(numeric_only=True)
    row = med.reindex(model_cols).astype(float)
    for k, v in {
        "age": age,
        "MonthlyIncome": income,
        "RevolvingUtilizationOfUnsecuredLines": util,
        "DebtRatio": dratio,
        "NumberOfOpenCreditLinesAndLoans": open_lines,
        "NumberOfDependents": dependents,
        "NumberOfTime30-59DaysPastDueNotWorse": late30,
        "NumberOfTime60-89DaysPastDueNotWorse": late60,
        "NumberOfTimes90DaysLate": late90,
    }.items():
        if k in row.index:
            row[k] = float(v)

    X = row.values.reshape(1, -1)
    Xs = (X - scaler_mean) / np.where(scaler_scale == 0, 1.0, scaler_scale)
    logit = float(np.dot(Xs, coef) + intercept)
    prob = float(1 / (1 + np.exp(-logit)))

    # Thresholds and verdict
    section_title("Verdict thresholds (adjust to policy)")
    cA, cB = st.columns(2)
    green_cut = cA.slider("GO if probability < ", 0.02, 0.50, 0.20, step=0.01, key="cc_green")
    red_cut   = cB.slider("STOP if probability ≥", green_cut, 0.95, 0.50, step=0.01, key="cc_red")

    if prob < green_cut:
        st.success(f"GO — predicted default probability {prob:.2%}.")
        st.markdown('<div class="note">Suitable for standard terms and amounts.</div>', unsafe_allow_html=True)
        gauge_color = "green"
    elif prob < red_cut:
        st.warning(f"CAUTION — predicted default probability {prob:.2%}.")
        st.markdown('<div class="note">Consider a smaller amount, shorter term, or extra checks.</div>', unsafe_allow_html=True)
        gauge_color = "orange"
    else:
        st.error(f"STOP — predicted default probability {prob:.2%}.")
        st.markdown('<div class="note">Lending not recommended at this time.</div>', unsafe_allow_html=True)
        gauge_color = "red"

    section_title("Risk gauge")
    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        number={"suffix": "%"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": gauge_color},
            "steps": [
                {"range": [0, green_cut * 100], "color": "#d1fae5"},
                {"range": [green_cut * 100, red_cut * 100], "color": "#fef3c7"},
                {"range": [red_cut * 100, 100], "color": "#fee2e2"},
            ],
        },
    ))
    gauge.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(gauge, use_container_width=True)

# ---------------- 6) Advanced (EDA & Diagnostics) ----------------
elif page == "Advanced (EDA & Diagnostics)":
    big_title("Advanced — EDA & Diagnostics")

    # Relationships & Segments
    section_title("Default rate by binned features")
    cols = [c for c in ["age","MonthlyIncome","DebtRatio","RevolvingUtilizationOfUnsecuredLines"] if c in df_full.columns]
    grid = st.columns(2)
    for i, c in enumerate(cols):
        with grid[i % 2]:
            figb = safe_rate_by_bin(df_full, c, bins=8)
            if figb: st.plotly_chart(figb, use_container_width=True)

    st.markdown('<hr class="separator" />', unsafe_allow_html=True)

    section_title("Debt Ratio vs Card/Line Utilization (log axes)")
    if {"DebtRatio","RevolvingUtilizationOfUnsecuredLines"}.issubset(df_full.columns):
        tmp = df_full[["DebtRatio","RevolvingUtilizationOfUnsecuredLines"]].dropna()
        s = px.scatter(tmp, x="RevolvingUtilizationOfUnsecuredLines", y="DebtRatio",
                       labels={"RevolvingUtilizationOfUnsecuredLines": DISPLAY["RevolvingUtilizationOfUnsecuredLines"],
                               "DebtRatio": DISPLAY["DebtRatio"]},
                       opacity=0.5, log_x=True, log_y=True)
        s.update_layout(height=420, margin=dict(l=10,r=10,t=8,b=10))
        st.plotly_chart(s, use_container_width=True)

    st.markdown('<hr class="separator" />', unsafe_allow_html=True)

    # Correlation
    section_title("Correlation heatmap (numeric)")
    num_cols = [c for c in df_full.columns if pd.api.types.is_numeric_dtype(df_full[c])]
    if len(num_cols) >= 3:
        corr = df_full[num_cols].corr(numeric_only=True)
        heat = go.Figure(data=go.Heatmap(z=corr.values, x=[DISPLAY.get(x,x) for x in corr.columns],
                                         y=[DISPLAY.get(y,y) for y in corr.index], zmin=-1, zmax=1,
                                         colorbar=dict(title="ρ")))
        heat.update_layout(height=520, margin=dict(l=10,r=10,t=8,b=10))
        st.plotly_chart(heat, use_container_width=True)

    st.markdown('<hr class="separator" />', unsafe_allow_html=True)

    # Modeling diagnostics that didn’t fit Core
    # Ensure scores
    if "__y_proba__" not in st.session_state:
        y_all = df_full["SeriousDlqin2yrs"].astype(int)
        X_cols_all = [c for c in df_full.columns if c != "SeriousDlqin2yrs" and pd.api.types.is_numeric_dtype(df_full[c])]
        X_all = df_full[X_cols_all].copy().fillna(df_full[X_cols_all].median(numeric_only=True))
        Xtr, Xte, ytr, yte = train_test_split(X_all, y_all, test_size=0.2, random_state=42, stratify=y_all)
        sc = StandardScaler(); Xtr_sc = sc.fit_transform(Xtr); Xte_sc = sc.transform(Xte)
        lr = LogisticRegression(max_iter=500, solver="liblinear", class_weight="balanced"); lr.fit(Xtr_sc, ytr)
        y_pr = lr.predict_proba(Xte_sc)[:,1]
        st.session_state["__model_cols__"] = X_cols_all
        st.session_state["__scaler_mean__"] = sc.mean_.tolist()
        st.session_state["__scaler_scale__"] = sc.scale_.tolist()
        st.session_state["__coef__"] = lr.coef_[0].tolist()
        st.session_state["__intercept__"] = float(lr.intercept_[0])
        st.session_state["__X_test_index__"] = Xte.index.tolist()
        st.session_state["__y_test__"] = yte.tolist()
        st.session_state["__y_proba__"] = y_pr.tolist()

    idx = st.session_state["__X_test_index__"]
    y_test = pd.Series(st.session_state["__y_test__"], index=idx)
    y_proba = np.array(st.session_state["__y_proba__"])

    # KS & Gains
    section_title("KS curve (cumulative separation) & Gains/Lift")
    df_ks = pd.DataFrame({"y": y_test.values, "p": y_proba}).sort_values("p", ascending=True)
    df_ks["cum_all"] = np.arange(1, len(df_ks)+1)/len(df_ks)
    df_ks["cum_bad"] = (df_ks["y"].cumsum())/max(1, df_ks["y"].sum())
    df_ks["cum_good"] = ((1-df_ks["y"]).cumsum())/max(1, (1-df_ks["y"]).sum())
    ks = float((df_ks["cum_bad"] - df_ks["cum_good"]).max())
    ks_fig = go.Figure()
    ks_fig.add_trace(go.Scatter(x=df_ks["cum_all"], y=df_ks["cum_bad"], mode="lines", name="Bad CDF"))
    ks_fig.add_trace(go.Scatter(x=df_ks["cum_all"], y=df_ks["cum_good"], mode="lines", name="Good CDF"))
    ks_fig.update_layout(xaxis_title="Population (sorted by score)", yaxis_title="Cumulative proportion",
                         height=360, margin=dict(l=10,r=10,t=40,b=10))

    dec = pd.qcut(y_proba, q=10, labels=False, duplicates="drop")
    gains = pd.DataFrame({"decile": dec, "y": y_test.values}).groupby("decile")["y"].agg(["count","sum"]).reset_index()
    gains = gains.sort_values("decile", ascending=False).reset_index(drop=True)
    gains["cum_bads"] = gains["sum"].cumsum()
    gains["cum_capture"] = gains["cum_bads"]/max(1, gains["sum"].sum())
    gains["pop_cum"] = gains["count"].cumsum()/max(1, gains["count"].sum())
    lift_fig = go.Figure()
    lift_fig.add_trace(go.Scatter(x=gains["pop_cum"], y=gains["cum_capture"], mode="lines+markers", name="Cumulative gains"))
    lift_fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Baseline", line=dict(dash="dash")))
    lift_fig.update_layout(xaxis_title="Cumulative population", yaxis_title="Cumulative bad capture",
                           height=360, margin=dict(l=10,r=10,t=40,b=10))

    cL, cR = st.columns(2)
    with cL: st.plotly_chart(ks_fig, use_container_width=True)
    with cR: st.plotly_chart(lift_fig, use_container_width=True)

    st.markdown('<hr class="separator" />', unsafe_allow_html=True)

    # Permutation importance (test set)
    section_title("Permutation importance (test set)")
    # Rebuild the trained objects quickly
    y_all = df_full["SeriousDlqin2yrs"].astype(int)
    X_cols_all = [c for c in df_full.columns if c != "SeriousDlqin2yrs" and pd.api.types.is_numeric_dtype(df_full[c])]
    X_all = df_full[X_cols_all].copy().fillna(df_full[X_cols_all].median(numeric_only=True))
    Xtr_df, Xte_df, ytr, yte = train_test_split(X_all, y_all, test_size=0.2, random_state=42, stratify=y_all)
    sc = StandardScaler(); Xtr_sc = sc.fit_transform(Xtr_df); Xte_sc = sc.transform(Xte_df)
    lr = LogisticRegression(max_iter=500, solver="liblinear", class_weight="balanced"); lr.fit(Xtr_sc, ytr)
    sample_idx = np.random.RandomState(42).choice(len(Xte_df), size=min(2000, len(Xte_df)), replace=False)
    X_perm = Xte_df.iloc[sample_idx]
    y_perm = yte.iloc[sample_idx]
    X_perm_sc = sc.transform(X_perm)
    pi = permutation_importance(lr, X_perm_sc, y_perm, n_repeats=8, random_state=42, scoring="roc_auc")
    pi_df = pd.DataFrame({"feature": X_cols_all, "importance": pi.importances_mean}).sort_values("importance", ascending=False).head(15)
    pi_fig = px.bar(pi_df, x="feature", y="importance")
    pi_fig.update_layout(height=360, margin=dict(l=10,r=10,t=8,b=10), xaxis_tickangle=-30)
    st.plotly_chart(pi_fig, use_container_width=True)

    st.markdown('<hr class="separator" />', unsafe_allow_html=True)

    # Saved Figures
    from PIL import Image, UnidentifiedImageError
    section_title("Saved Figures (reports/figures)")
    def load_image_safe(path: str):
        try:
            img = Image.open(path); img.load()
            if img.mode not in ("RGB", "L"): img = img.convert("RGB")
            img.thumbnail((1600, 1600))
            return img, None
        except (UnidentifiedImageError, OSError) as e:
            return None, f"Unrecognized or unreadable image: {e}"
        except Exception as e:
            return None, str(e)

    files = []
    for pat in ["reports/figures/*.png", "reports/figures/*.jpg", "reports/figures/*.jpeg"]:
        files.extend(sorted(glob.glob(pat)))
    if not files:
        st.markdown('<div class="block note">No images found in reports/figures.</div>', unsafe_allow_html=True)
    else:
        cols = st.columns(3)
        for i, p in enumerate(files):
            img, err = load_image_safe(p)
            with cols[i % 3]:
                if img is not None:
                    st.image(img, use_column_width=True, caption=os.path.basename(p))
                    try:
                        with open(p, "rb") as f:
                            st.download_button("Download", data=f.read(), file_name=os.path.basename(p),
                                               mime="image/png" if p.lower().endswith(".png") else "image/jpeg",
                                               key=f"dl_{i}_{os.path.basename(p)}")
                    except Exception: pass
                else:
                    st.markdown(f"- Could not display: `{p}` — {err or 'unknown error'}")

    st.markdown('<hr class="separator" />', unsafe_allow_html=True)

    # Data Quality
    section_title("Data Quality")
    miss = df_full.isna().sum().sort_values(ascending=False); miss_df = miss[miss>0].reset_index()
    miss_df.columns = ["Column","Missing"]
    if not miss_df.empty:
        fig = px.bar(miss_df, x="Column", y="Missing")
        fig.update_layout(height=320, margin=dict(l=10,r=10,t=8,b=10)); st.plotly_chart(fig, use_container_width=True)
    else:
        st.markdown('<div class="block note">No missing values.</div>', unsafe_allow_html=True)

# ---------------- 7) Summary & Conclusion ----------------
elif page == "Summary & Conclusion":
    big_title("Summary & Conclusion")
    st.markdown("""
Borrowers most likely to default tend to combine **very high card/line utilization**, **any past-due history**,
**very high debt ratio**, **lower income**, **younger age**, **many open lines/loans**, and **3+ dependents**.
Use A–D buckets to set policy, and use Client Credit Check to assess single cases with **GO / CAUTION / STOP**.
""")

    # Build a quick scored set to show bucket table
    if "__y_proba__" not in st.session_state:
        y_all = df_full["SeriousDlqin2yrs"].astype(int)
        X_cols_all = [c for c in df_full.columns if c != "SeriousDlqin2yrs" and pd.api.types.is_numeric_dtype(df_full[c])]
        X_all = df_full[X_cols_all].copy().fillna(df_full[X_cols_all].median(numeric_only=True))
        Xtr, Xte, ytr, yte = train_test_split(X_all, y_all, test_size=0.2, random_state=42, stratify=y_all)
        sc = StandardScaler(); Xtr_sc = sc.fit_transform(Xtr); Xte_sc = sc.transform(Xte)
        lr = LogisticRegression(max_iter=500, solver="liblinear", class_weight="balanced"); lr.fit(Xtr_sc, ytr)
        y_pr = lr.predict_proba(Xte_sc)[:,1]
        st.session_state["__X_test_index__"] = Xte.index.tolist()
        st.session_state["__y_test__"] = yte.tolist()
        st.session_state["__y_proba__"] = y_pr.tolist()

    idx = st.session_state["__X_test_index__"]
    y_test = st.session_state["__y_test__"]
    y_proba = st.session_state["__y_proba__"]
    scored = pd.DataFrame({"prob_default": y_proba, "true_label": y_test}, index=idx)
    extras = [c for c in ["MonthlyIncome","age","NumberOfOpenCreditLinesAndLoans","NumberOfDependents"] if c in df_full.columns]
    scored = scored.join(df_full.loc[scored.index, extras])
    scored["bucket"] = pd.qcut(scored["prob_default"], q=4, labels=list("ABCD"))

    section_title("Bucket table (A least risky → D most risky)")
    tbl = (scored.groupby("bucket")
           .agg(Borrowers=("prob_default","size"),
                Avg_Default_Prob=("prob_default","mean"),
                Avg_Age=("age","mean"),
                Avg_Income=("MonthlyIncome","mean"),
                Avg_Open_Loans=("NumberOfOpenCreditLinesAndLoans","mean"),
                Avg_Dependents=("NumberOfDependents","mean"))
           .reindex(list("ABCD")))
    tbl["Avg_Default_Prob"] = (100*tbl["Avg_Default_Prob"]).round(1)
    for c in ["Avg_Age","Avg_Income","Avg_Open_Loans","Avg_Dependents"]:
        if c in tbl.columns: tbl[c] = tbl[c].round(0)
    tbl = tbl.reset_index().rename(columns={"bucket":"Bucket","Avg_Default_Prob":"Avg default (%)"})
    st.dataframe(tbl, use_container_width=True)

    section_title("Operational guidance")
    st.markdown("""
- **Bucket A**: generally safe; standard terms.  
- **Bucket B**: near-safe; modest limits.  
- **Bucket C**: caution; tighter limits or extra checks.  
- **Bucket D**: high risk; small loans only if needed, or decline.
""")
