# dashboard.py
# Streamlit dashboard for the Loan Default Exam Project
# One-file app with sidebar navigation ("pages")
# - Consistent teal title boxes
# - Static EDA page (exam-style)
# - Modeling & Metrics (fixed settings)
# - Risk Buckets (A/B/C/D) + Bucket D table + CSV download
# - Interactive Lab (optional filters/threshold)
# - Saved Figures & Data Quality
# - Robust dataset auto-discovery and dirty data handling

from __future__ import annotations

import os
import glob
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
    confusion_matrix, accuracy_score
)

# ===============================
# Page setup & global style
# ===============================
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
.big-title {
  background: var(--teal);
  color: #ffffff !important;
  padding: 18px 22px;
  border-radius: var(--radius);
  box-shadow: var(--shadow);
  font-size: 1.6rem;
  font-weight: 700;
  letter-spacing: 0.2px;
  margin: 8px 0 18px 0;
}
.section-title {
  background: linear-gradient(0deg, var(--teal-light), var(--teal-light));
  border: 2px solid var(--teal);
  color: var(--ink);
  padding: 10px 14px;
  border-radius: 12px;
  box-shadow: var(--shadow);
  font-size: 1.05rem;
  font-weight: 700;
  margin: 8px 0 10px 0;
}
.small-muted { color: var(--muted); font-size: 0.9rem; }
.block { background: #ffffff; border-radius: var(--radius); box-shadow: var(--shadow); padding: 14px; }
hr.separator { border: none; height: 1px; background: #e2e8f0; margin: 18px 0; }
</style>
""", unsafe_allow_html=True)

def big_title(text: str):
    st.markdown(f'<div class="big-title">{text}</div>', unsafe_allow_html=True)

def section_title(text: str):
    st.markdown(f'<div class="section-title">{text}</div>', unsafe_allow_html=True)

def pct(x: float) -> str:
    return f"{x*100:.1f}%"

# ===============================
# Data helpers
# ===============================
REQUIRED_COLS = {
    "SeriousDlqin2yrs",
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
}

@st.cache_data(show_spinner=False)
def find_dataset() -> Optional[str]:
    patterns = ["data/*.csv","datasets/*.csv","*.csv","input/*.csv","inputs/*.csv"]
    found: List[str] = []
    for p in patterns:
        found += glob.glob(p)
    priority = ["data/clean_credit.csv","data/credit_clean.csv","data/credit.csv","cs-training.csv","train.csv"]
    ordered, seen = [], set()
    for p in priority:
        if p in found and p not in seen:
            ordered.append(p); seen.add(p)
    for f in found:
        if f not in seen:
            ordered.append(f); seen.add(f)
    for path in ordered:
        try:
            sample = pd.read_csv(path, nrows=5)
        except Exception:
            continue
        cols = set(c.strip() for c in sample.columns)
        if "SeriousDlqin2yrs" in cols and len(cols & REQUIRED_COLS) >= max(7, int(0.6*len(REQUIRED_COLS))):
            return path
    return None

def _to_numeric(df: pd.DataFrame, exclude: Optional[List[str]]=None) -> pd.DataFrame:
    exclude = exclude or []
    for c in df.columns:
        if c in exclude:
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

@st.cache_data(show_spinner=False)
def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    df = _to_numeric(df)
    if "SeriousDlqin2yrs" in df.columns:
        df = df.dropna(subset=["SeriousDlqin2yrs"])
    if "age" in df.columns:
        df = df[df["age"] > 0]
    return df

# Load once for all "pages"
csv_path = find_dataset()
if not csv_path:
    big_title("Loan Default — Dashboard")
    st.error("No compatible CSV found. Add e.g. `data/clean_credit.csv` or `cs-training.csv` and refresh.")
    st.stop()
df_full = load_data(csv_path)

# ===============================
# Sidebar navigation
# ===============================
with st.sidebar:
    st.title("Navigation")
    page = st.radio(
        "Go to",
        [
            "Overview",
            "EDA (static)",
            "Modeling & Metrics",
            "Risk Buckets (A/B/C/D)",
            "Interactive Lab",
            "Saved Figures",
            "Data Quality",
        ],
        index=0,
    )
    st.caption(f"Data source: `{csv_path}`")

# ==========================================================
# Overview
# ==========================================================
if page == "Overview":
    big_title("Loan Default — Dashboard (Overview)")
    st.caption("Stable overview. Use the nav on the left for more pages.")

    # KPIs
    section_title("Overview")
    c1, c2, c3, c4 = st.columns(4)
    total_rows = len(df_full)
    default_rate = float(df_full["SeriousDlqin2yrs"].mean()) if total_rows else 0.0
    miss_income = int(df_full["MonthlyIncome"].isna().sum()) if "MonthlyIncome" in df_full.columns else 0
    age_span = (int(df_full["age"].min()), int(df_full["age"].max())) if "age" in df_full.columns and total_rows else (None, None)
    c1.metric("Rows", f"{total_rows:,}")
    c2.metric("Default rate", pct(default_rate))
    c3.metric("Missing MonthlyIncome", f"{miss_income:,}" if "MonthlyIncome" in df_full.columns else "—")
    c4.metric("Age span", f"{age_span[0]}–{age_span[1]}" if all(age_span) else "—")

    st.markdown('<hr class="separator" />', unsafe_allow_html=True)

    # Target distribution
    section_title("Target Distribution (SeriousDlqin2yrs)")
    counts = df_full["SeriousDlqin2yrs"].value_counts(dropna=False).sort_index()
    fig = px.bar(x=[str(int(k)) for k in counts.index], y=counts.values,
                 labels={"x": "SeriousDlqin2yrs", "y": "Count"})
    fig.update_layout(margin=dict(l=10, r=10, t=8, b=10), height=320)
    st.plotly_chart(fig, use_container_width=True)

# ==========================================================
# EDA (static)
# ==========================================================
elif page == "EDA (static)":
    big_title("Exploratory Data Analysis (Static)")
    num_cols = [c for c in df_full.columns if c != "SeriousDlqin2yrs" and pd.api.types.is_numeric_dtype(df_full[c])]

    # Distributions
    section_title("Feature Distributions (colored by Default)")
    show_cols = [c for c in ["age","MonthlyIncome","DebtRatio",
                             "RevolvingUtilizationOfUnsecuredLines",
                             "NumberOfOpenCreditLinesAndLoans"] if c in num_cols]
    grid = st.columns(2)
    for i, col in enumerate(show_cols):
        with grid[i % 2]:
            fig = px.histogram(df_full, x=col, color=df_full["SeriousDlqin2yrs"].astype(str),
                               nbins=50, barmode="overlay", opacity=0.65,
                               labels={col: col, "color": "Default"})
            fig.update_layout(margin=dict(l=10, r=10, t=8, b=10), height=320, legend_title_text="Default")
            st.plotly_chart(fig, use_container_width=True)

    # Default rate by binned variables
    section_title("Default Rate by Binned Features")
    def rate_by_bin(frame: pd.DataFrame, col: str, bins: int = 8) -> Optional[go.Figure]:
        if col not in frame.columns or frame[col].dropna().empty:
            return None
        tmp = frame[[col, "SeriousDlqin2yrs"]].dropna()
        if tmp.empty:
            return None
        tmp["bin"] = pd.qcut(tmp[col], q=min(bins, tmp[col].nunique()), duplicates="drop")
        grp = tmp.groupby("bin")["SeriousDlqin2yrs"].mean().reset_index()
        grp["rate"] = grp["SeriousDlqin2yrs"] * 100.0
        fig = px.bar(grp, x="bin", y="rate", labels={"bin": col, "rate": "Default rate (%)"})
        fig.update_layout(margin=dict(l=10, r=10, t=8, b=10), height=320)
        return fig

    binnable = [c for c in ["age","MonthlyIncome","DebtRatio","RevolvingUtilizationOfUnsecuredLines"] if c in df_full.columns]
    cols = st.columns(2)
    for i, ccol in enumerate(binnable[:4]):
        figb = rate_by_bin(df_full, ccol, bins=8)
        if figb:
            with cols[i % 2]:
                st.plotly_chart(figb, use_container_width=True)

    # Correlation heatmap
    if len(num_cols) >= 3:
        section_title("Correlation Heatmap")
        corr = df_full[num_cols + ["SeriousDlqin2yrs"]].corr(numeric_only=True)
        heat = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.index, zmin=-1, zmax=1,
                                         colorbar=dict(title="ρ")))
        heat.update_layout(margin=dict(l=10, r=10, t=8, b=10), height=520)
        st.plotly_chart(heat, use_container_width=True)

# ==========================================================
# Modeling & Metrics (fixed)
# ==========================================================
elif page == "Modeling & Metrics":
    big_title("Modeling & Metrics (Fixed Settings)")

    y = df_full["SeriousDlqin2yrs"].astype(int)
    X_cols = [c for c in df_full.columns if c != "SeriousDlqin2yrs" and pd.api.types.is_numeric_dtype(df_full[c])]
    X_df = df_full[X_cols].copy().fillna(df_full[X_cols].median(numeric_only=True))

    TEST_SIZE = 0.20
    THRESH = 0.50

    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X_df, y, test_size=TEST_SIZE, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_df)
    X_test  = scaler.transform(X_test_df)

    lr = LogisticRegression(max_iter=500, solver="liblinear", class_weight="balanced")
    lr.fit(X_train, y_train)

    y_proba = lr.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)

    # ROC
    section_title(f"ROC Curve (AUC = {auc:.3f})")
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_fig = go.Figure()
    roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC"))
    roc_fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="baseline", line=dict(dash="dash")))
    roc_fig.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
                          height=380, margin=dict(l=10,r=10,t=40,b=10))
    st.plotly_chart(roc_fig, use_container_width=True)

    # Metrics @ fixed threshold
    y_pred = (y_proba >= THRESH).astype(int)
    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=[0,1])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{acc:.3f}")
    c2.metric("Precision (1)", f"{prec:.3f}")
    c3.metric("Recall (1)", f"{rec:.3f}")
    c4.metric("F1 (1)", f"{f1:.3f}")

    section_title("Confusion Matrix")
    cm_fig = go.Figure(data=go.Heatmap(z=cm, x=["Pred 0","Pred 1"], y=["True 0","True 1"],
                                       text=cm, texttemplate="%{text}", showscale=False))
    cm_fig.update_layout(height=320, margin=dict(l=10,r=10,t=8,b=10))
    st.plotly_chart(cm_fig, use_container_width=True)

    # Feature importance (|coef|)
    coef = pd.Series(lr.coef_[0], index=X_cols).abs().sort_values(ascending=False)
    topk = min(15, len(coef))
    fi = coef.head(topk).reset_index()
    fi.columns = ["feature","importance(|coef|)"]
    section_title("Top Feature Importance (|coef|)")
    fi_fig = px.bar(fi, x="feature", y="importance(|coef|)")
    fi_fig.update_layout(height=360, margin=dict(l=10,r=10,t=8,b=10), xaxis_tickangle=-30)
    st.plotly_chart(fi_fig, use_container_width=True)

    # cache for Risk Buckets page in this one-file app
    st.session_state["__X_test_index__"] = X_test_df.index.tolist()
    st.session_state["__y_test__"] = y_test.tolist()
    st.session_state["__y_proba__"] = y_proba.tolist()

# ==========================================================
# Risk Buckets (A/B/C/D)
# ==========================================================
elif page == "Risk Buckets (A/B/C/D)":
    big_title("Risk Buckets (A/B/C/D) & Likely Defaulters")

    idx = st.session_state.get("__X_test_index__")
    y_test = st.session_state.get("__y_test__")
    y_proba = st.session_state.get("__y_proba__")

    if not (idx and y_test and y_proba):
        st.info("Open 'Modeling & Metrics' first to generate predictions.")
        st.stop()

    scored = pd.DataFrame({"prob_default": y_proba, "true_label": y_test}, index=idx)

    # include ID if present
    possible_ids = [c for c in ["ID","Id","id","CustomerID","customer_id"] if c in df_full.columns]
    id_col = possible_ids[0] if possible_ids else None
    if id_col:
        scored["BorrowerID"] = df_full.loc[scored.index, id_col].values
    else:
        scored["BorrowerID"] = scored.index

    section_title("Bucket Cutoffs")
    colA, colB, colC = st.columns(3)
    cut_A = colA.number_input("A/B cutoff", min_value=0.0, max_value=0.99, value=0.10, step=0.01)
    cut_B = colB.number_input("B/C cutoff", min_value=cut_A, max_value=0.995, value=0.25, step=0.01)
    cut_C = colC.number_input("C/D cutoff", min_value=cut_B, max_value=1.0, value=0.50, step=0.01)

    def bucketize(p: float) -> str:
        if p < cut_A: return "A"
        if p < cut_B: return "B"
        if p < cut_C: return "C"
        return "D"

    scored["risk_bucket"] = scored["prob_default"].apply(bucketize)

    # Bucket counts
    section_title("Portfolio Mix by Bucket")
    bucket_counts = scored["risk_bucket"].value_counts().reindex(["A","B","C","D"]).fillna(0).astype(int).reset_index()
    bucket_counts.columns = ["bucket","count"]
    fig_b = px.bar(bucket_counts, x="bucket", y="count", labels={"count":"Borrowers"})
    fig_b.update_layout(height=300, margin=dict(l=10,r=10,t=8,b=10))
    st.plotly_chart(fig_b, use_container_width=True)

    # Likely defaulters (D)
    section_title("Bucket D — Likely Defaulters (Top N)")
    n_top = st.slider("Rows to show", 5, 200, 25, step=5)
    cols_to_show = ["BorrowerID","prob_default","risk_bucket","true_label"]
    extra_cols = [c for c in ["MonthlyIncome","age","DebtRatio","RevolvingUtilizationOfUnsecuredLines"]
                  if c in df_full.columns]
    cols_to_show += extra_cols
    top_D = (scored[scored["risk_bucket"]=="D"]
             .sort_values("prob_default", ascending=False)
             .head(n_top))
    st.dataframe(top_D[cols_to_show], use_container_width=True)

    # Download
    csv = (scored[["BorrowerID","prob_default","risk_bucket","true_label"]+extra_cols]
           .sort_values("prob_default", ascending=False).to_csv(index=False))
    st.download_button("Download full scored test set (CSV)",
                       data=csv, file_name="scored_test_with_buckets.csv", mime="text/csv")

# ==========================================================
# Interactive Lab (optional filters + threshold)
# ==========================================================
elif page == "Interactive Lab":
    big_title("Interactive Lab (Filters + Threshold)")

    # Sidebar-like controls inside the page (kept minimal)
    section_title("Filters")
    cols_f = st.columns(2)
    # Age filter
    if "age" in df_full.columns and df_full["age"].notna().any():
        amin, amax = int(df_full["age"].min()), int(df_full["age"].max())
        age_rng = cols_f[0].slider("Age range", min_value=amin, max_value=max(amax, amin+1), value=(amin, amax))
    else:
        age_rng = None
    # Monthly income filter
    if "MonthlyIncome" in df_full.columns and df_full["MonthlyIncome"].notna().any():
        mi_min, mi_max = float(df_full["MonthlyIncome"].min()), float(df_full["MonthlyIncome"].max())
        default_hi = float(np.nanpercentile(df_full["MonthlyIncome"].dropna(), 95)) if df_full["MonthlyIncome"].notna().any() else mi_max
        income_rng = cols_f[1].slider("Monthly income range",
                                      min_value=float(np.nan_to_num(mi_min, nan=0.0)),
                                      max_value=float(np.nan_to_num(mi_max, nan=100000.0)),
                                      value=(float(np.nan_to_num(mi_min, nan=0.0)),
                                             float(np.nan_to_num(default_hi, nan=mi_max))))
    else:
        income_rng = None

    delin_cols = [c for c in [
        "NumberOfTime30-59DaysPastDueNotWorse",
        "NumberOfTime60-89DaysPastDueNotWorse",
        "NumberOfTimes90DaysLate"
    ] if c in df_full.columns]
    require_any_delinquency = st.checkbox("Only rows with any delinquency > 0", value=False)

    # Apply filters
    mask = pd.Series(True, index=df_full.index)
    if age_rng and "age" in df_full.columns:
        mask &= df_full["age"].between(age_rng[0], age_rng[1], inclusive="both")
    if income_rng and "MonthlyIncome" in df_full.columns:
        mi0, mi1 = income_rng
        mask &= df_full["MonthlyIncome"].fillna(-1e12).between(mi0, mi1, inclusive="both")
    if require_any_delinquency and delin_cols:
        any_delinq = (df_full[delin_cols].fillna(0) > 0).any(axis=1)
        mask &= any_delinq
    df = df_full[mask].copy()

    # Quick model with adjustable threshold
    section_title("Quick Model — Threshold Tuner")
    y = df["SeriousDlqin2yrs"].astype(int)
    X_cols = [c for c in df.columns if c != "SeriousDlqin2yrs" and pd.api.types.is_numeric_dtype(df[c])]
    X_df = df[X_cols].copy().fillna(df[X_cols].median(numeric_only=True))

    # guard for small sets
    if len(df) < 200 or y.nunique() < 2:
        st.info("Not enough filtered rows/classes to train a model. Loosen filters.")
    else:
        X_train_df, X_test_df, y_train, y_test = train_test_split(
            X_df, y, test_size=0.2, random_state=42, stratify=y
        )
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_df)
        X_test  = scaler.transform(X_test_df)

        lr = LogisticRegression(max_iter=500, solver="liblinear", class_weight="balanced")
        lr.fit(X_train, y_train)

        y_proba = lr.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)

        th = st.slider("Decision threshold", 0.10, 0.90, 0.50, step=0.01)
        y_pred = (y_proba >= th).astype(int)

        acc = accuracy_score(y_test, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", zero_division=0)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("AUC", f"{auc:.3f}")
        c2.metric("Precision (1)", f"{prec:.3f}")
        c3.metric("Recall (1)", f"{rec:.3f}")
        c4.metric("F1 (1)", f"{f1:.3f}")

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_fig = go.Figure()
        roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC"))
        roc_fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="baseline", line=dict(dash="dash")))
        roc_fig.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
                              height=360, margin=dict(l=10,r=10,t=40,b=10))
        st.plotly_chart(roc_fig, use_container_width=True)

# ==========================================================
# Saved Figures
# ==========================================================
elif page == "Saved Figures":
    big_title("Saved Figures")
    pngs = sorted(glob.glob("reports/figures/*.png"))
    if pngs:
        section_title("reports/figures")
        cols = st.columns(3)
        for i, p in enumerate(pngs):
            with cols[i % 3]:
                st.image(p, use_container_width=True, caption=os.path.basename(p))
    else:
        st.markdown('<div class="block small-muted">No PNGs found in reports/figures.</div>', unsafe_allow_html=True)

# ==========================================================
# Data Quality
# ==========================================================
elif page == "Data Quality":
    big_title("Data Quality")

    section_title("Missing Values by Column")
    miss = df_full.isna().sum().sort_values(ascending=False)
    miss_df = miss[miss > 0].reset_index()
    miss_df.columns = ["column","missing"]
    if not miss_df.empty:
        fig = px.bar(miss_df, x="column", y="missing", labels={"missing":"Missing rows"})
        fig.update_layout(margin=dict(l=10, r=10, t=8, b=10), height=320)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.markdown('<div class="block small-muted">No missing values.</div>', unsafe_allow_html=True)

    section_title("Basic Stats")
    st.dataframe(df_full.describe(include="all").transpose(), use_container_width=True)
