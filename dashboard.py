# dashboard.py
# Loan Default Exam Project — Full EDA + Executive Summary + Modeling + Risk Buckets
# One-file app with sidebar "pages" navigation.
# - Consistent teal title boxes
# - Executive Summary (auto-written from data)
# - EDA (univariate, bivariate, segmentation, correlations, outliers, missingness)
# - Interactive Lab (filters + threshold + interactive rate-by-bin)
# - Modeling & Metrics (fixed)
# - Risk Buckets (A/B/C/D) + D-table + CSV
# - Saved Figures
# - Robust dataset auto-discovery and dirty data handling
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
from sklearn.metrics import (roc_auc_score, roc_curve, precision_recall_fscore_support,
                             confusion_matrix, accuracy_score)

# ---------------- Page + Style ----------------
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
.big-title { background: var(--teal); color: #fff!important; padding: 18px 22px;
  border-radius: var(--radius); box-shadow: var(--shadow); font-size: 1.6rem;
  font-weight: 700; margin: 8px 0 18px 0; letter-spacing: .2px; }
.section-title { background: var(--teal-light); border: 2px solid var(--teal); color: var(--ink);
  padding: 10px 14px; border-radius: 12px; box-shadow: var(--shadow); font-size: 1.05rem;
  font-weight: 700; margin: 8px 0 10px 0; }
.note { color: var(--muted); font-size: .95rem; margin: 6px 2px 14px 2px; }
.block { background: #fff; border-radius: var(--radius); box-shadow: var(--shadow); padding: 14px; }
hr.separator { border: none; height: 1px; background: #e2e8f0; margin: 18px 0; }
.kpi {font-size: 1.1rem;}
</style>
""", unsafe_allow_html=True)

def big_title(t: str): st.markdown(f'<div class="big-title">{t}</div>', unsafe_allow_html=True)
def section_title(t: str): st.markdown(f'<div class="section-title">{t}</div>', unsafe_allow_html=True)
def pct(x: float) -> str: return f"{100*x:.1f}%"

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
    return df

@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df = _to_numeric(df)
    if "SeriousDlqin2yrs" in df.columns: df = df.dropna(subset=["SeriousDlqin2yrs"])
    if "age" in df.columns: df = df[df["age"] > 0]
    return df

# Load
csv_path = find_dataset()
if not csv_path:
    big_title("Loan Default — Dashboard")
    st.error("No compatible CSV found. Add e.g. `data/clean_credit.csv` or `cs-training.csv` and refresh.")
    st.stop()
df_full = load_data(csv_path)

# ---------------- Sidebar nav ----------------
with st.sidebar:
    st.title("Navigation")
    page = st.radio("Go to", [
        "Executive Summary",
        "EDA — Univariate",
        "EDA — Bivariate & Segments",
        "EDA — Correlations & Outliers",
        "Interactive Lab",
        "Modeling & Metrics",
        "Risk Buckets (A/B/C/D)",
        "Saved Figures",
        "Data Quality"
    ], index=0)
    st.caption(f"Data source: `{csv_path}`")

# ---------------- Small shared helpers ----------------
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
        fig = px.bar(grp, x="bin", y="rate", labels={"bin": col, "rate": "Default rate (%)"})
        fig.update_layout(margin=dict(l=10,r=10,t=8,b=10), height=320)
        return fig
    except Exception:
        return None

def col_if_exists(df, name): return name if name in df.columns else None

# =======================================================
# Executive Summary
# =======================================================
if page == "Executive Summary":
    big_title("Executive Summary")

    # KPIs
    section_title("Portfolio Snapshot")
    c1,c2,c3,c4 = st.columns(4)
    rows = len(df_full)
    dr = df_full["SeriousDlqin2yrs"].mean() if rows else 0.0
    mi_missing = df_full["MonthlyIncome"].isna().sum() if "MonthlyIncome" in df_full.columns else 0
    age_min = int(df_full["age"].min()) if "age" in df_full.columns else None
    age_max = int(df_full["age"].max()) if "age" in df_full.columns else None
    c1.metric("Rows", f"{rows:,}")
    c2.metric("Default rate", pct(dr))
    c3.metric("Missing MonthlyIncome", f"{mi_missing:,}" if "MonthlyIncome" in df_full.columns else "—")
    c4.metric("Age span", f"{age_min}–{age_max}" if age_min is not None else "—")

    st.markdown('<hr class="separator" />', unsafe_allow_html=True)

    # Auto-highlights (simple heuristics)
    section_title("Key Findings")
    bullets = []
    # Utilization
    util = col_if_exists(df_full, "RevolvingUtilizationOfUnsecuredLines")
    if util:
        u99 = df_full[util].quantile(.99)
        high_util_rate = df_full[df_full[util] >= u99]["SeriousDlqin2yrs"].mean()
        bullets.append(f"Very high **revolving utilization** (≥ 99th pct: ~{u99:,.2f}) shows a default rate around **{pct(high_util_rate)}**, above portfolio average **{pct(dr)}**.")
    # Delinquency
    late90 = col_if_exists(df_full, "NumberOfTimes90DaysLate")
    if late90:
        any_late = (df_full[late90].fillna(0) > 0)
        bullets.append(f"Borrowers with **any 90-days-late history** have default rates materially higher than average (**{pct(df_full.loc[any_late,'SeriousDlqin2yrs'].mean())}** vs **{pct(dr)}**).")
    # Debt ratio
    debt = col_if_exists(df_full, "DebtRatio")
    if debt:
        d99 = df_full[debt].quantile(.99)
        bullets.append(f"Extremely high **DebtRatio** (≥ 99th pct: ~{d99:,.2f}) correlates with elevated risk relative to average.")
    # Income
    inc = col_if_exists(df_full, "MonthlyIncome")
    if inc:
        low_inc = df_full[inc] <= df_full[inc].quantile(.2)
        bullets.append(f"**Lower MonthlyIncome** segment (bottom 20%) shows higher default propensity (**{pct(df_full.loc[low_inc,'SeriousDlqin2yrs'].mean())}**).")
    if not bullets:
        bullets.append("Target default rate and outlier segments not computed due to missing columns.")

    for b in bullets:
        st.markdown(f"- {b}")

    st.markdown('<div class="note">Tip: See “Risk Buckets” for a clean A→D grouping of borrowers by predicted risk.</div>', unsafe_allow_html=True)

# =======================================================
# EDA — Univariate
# =======================================================
elif page == "EDA — Univariate":
    big_title("EDA — Univariate (Distributions)")

    # Target
    section_title("Target: SeriousDlqin2yrs (0/1)")
    counts = df_full["SeriousDlqin2yrs"].value_counts().sort_index()
    fig = px.bar(x=[str(int(k)) for k in counts.index], y=counts.values, labels={"x":"SeriousDlqin2yrs","y":"Count"})
    fig.update_layout(margin=dict(l=10,r=10,t=8,b=10), height=320)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('<div class="note">Most borrowers are non-defaulters; class imbalance is expected in this dataset.</div>', unsafe_allow_html=True)

    # Key numeric distributions (hist + box)
    section_title("Key Feature Distributions")
    show_cols = [c for c in [
        "age","MonthlyIncome","DebtRatio","RevolvingUtilizationOfUnsecuredLines",
        "NumberOfOpenCreditLinesAndLoans","NumberOfTimes90DaysLate",
        "NumberOfTime30-59DaysPastDueNotWorse","NumberOfTime60-89DaysPastDueNotWorse",
        "NumberRealEstateLoansOrLines","NumberOfDependents"
    ] if c in df_full.columns]
    for c in show_cols:
        st.markdown(f"**{c}**")
        col1, col2 = st.columns([2,1])
        with col1:
            fig = px.histogram(df_full, x=c, color=df_full["SeriousDlqin2yrs"].astype(str),
                               nbins=60, barmode="overlay", opacity=.65,
                               labels={c:c,"color":"Default"})
            fig.update_layout(height=300, margin=dict(l=10,r=10,t=8,b=10), legend_title_text="Default")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            # side boxplot (outlier glance)
            figb = px.box(df_full, y=c, points=False)
            figb.update_layout(height=300, margin=dict(l=10,r=10,t=8,b=10))
            st.plotly_chart(figb, use_container_width=True)
        st.markdown('<div class="note">Distribution by default status helps spot shifts and heavy tails/outliers.</div>', unsafe_allow_html=True)
        st.markdown('<hr class="separator" />', unsafe_allow_html=True)

# =======================================================
# EDA — Bivariate & Segments
# =======================================================
elif page == "EDA — Bivariate & Segments":
    big_title("EDA — Bivariate & Segments")

    # Default rate by bins for several variables
    section_title("Default Rate by Binned Features")
    cols = []
    for name in ["age","MonthlyIncome","DebtRatio","RevolvingUtilizationOfUnsecuredLines"]:
        if name in df_full.columns: cols.append(name)
    grid = st.columns(2)
    for i, c in enumerate(cols):
        with grid[i % 2]:
            figb = safe_rate_by_bin(df_full, c, bins=8)
            if figb: st.plotly_chart(figb, use_container_width=True)
    st.markdown('<div class="note">Binning reveals monotonic or threshold-like risk patterns (e.g., very high utilization).</div>', unsafe_allow_html=True)

    st.markdown('<hr class="separator" />', unsafe_allow_html=True)

    # Scatter vs default probability proxy (mean in hex bins)
    section_title("2D Relationship (Hexbin default rate proxy)")
    # Choose two continuous variables available
    xname = cols[0] if cols else None
    yname = cols[1] if len(cols) > 1 else None
    if xname and yname:
        tmp = df_full[[xname, yname, "SeriousDlqin2yrs"]].dropna()
        if len(tmp) > 0:
            # aggregate default rate in grid
            xb = pd.cut(tmp[xname], bins=30)
            yb = pd.cut(tmp[yname], bins=30)
            grid_df = tmp.groupby([xb, yb])["SeriousDlqin2yrs"].mean().reset_index()
            grid_df["DefaultRate"] = 100*grid_df["SeriousDlqin2yrs"]
            grid_df[xname] = grid_df[xname].astype(str)
            grid_df[yname] = grid_df[yname].astype(str)
            figh = px.density_heatmap(grid_df, x=xname, y=yname, z="DefaultRate",
                                      color_continuous_scale="Viridis",
                                      labels={"DefaultRate":"Default rate (%)"})
            figh.update_layout(height=420, margin=dict(l=10,r=10,t=8,b=10))
            st.plotly_chart(figh, use_container_width=True)
            st.markdown(f'<div class="note">Darker tiles indicate higher default rate across the {xname} × {yname} grid.</div>', unsafe_allow_html=True)
    else:
        st.info("Not enough continuous features for a 2D grid chart.")

    st.markdown('<hr class="separator" />', unsafe_allow_html=True)

    # Segmentation by delinquency flags
    section_title("Segment: Any Past-Due Flags vs Default")
    flags = [c for c in ["NumberOfTime30-59DaysPastDueNotWorse","NumberOfTime60-89DaysPastDueNotWorse","NumberOfTimes90DaysLate"] if c in df_full.columns]
    if flags:
        any_delinq = (df_full[flags].fillna(0) > 0).any(axis=1)
        seg = pd.DataFrame({"segment": np.where(any_delinq, "Has past-due", "No past-due"),
                            "SeriousDlqin2yrs": df_full["SeriousDlqin2yrs"].values})
        rate = seg.groupby("segment")["SeriousDlqin2yrs"].mean().reset_index()
        rate["rate(%)"] = 100*rate["SeriousDlqin2yrs"]
        bar = px.bar(rate, x="segment", y="rate(%)", labels={"rate(%)":"Default rate (%)"})
        bar.update_layout(height=320, margin=dict(l=10,r=10,t=8,b=10))
        st.plotly_chart(bar, use_container_width=True)
        st.markdown('<div class="note">Borrowers with any past-due history show markedly higher default rates.</div>', unsafe_allow_html=True)
    else:
        st.info("Delinquency columns not found for this segment view.")

# =======================================================
# EDA — Correlations & Outliers
# =======================================================
elif page == "EDA — Correlations & Outliers":
    big_title("EDA — Correlations & Outliers")

    # Correlation heatmap
    num_cols = [c for c in df_full.columns if pd.api.types.is_numeric_dtype(df_full[c])]
    if len(num_cols) >= 3:
        section_title("Correlation Heatmap (numeric)")
        corr = df_full[num_cols].corr(numeric_only=True)
        heat = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.index, zmin=-1, zmax=1,
                                         colorbar=dict(title="ρ")))
        heat.update_layout(height=520, margin=dict(l=10,r=10,t=8,b=10))
        st.plotly_chart(heat, use_container_width=True)
        st.markdown('<div class="note">Focus on features moderately correlated with the target or with each other to spot redundancy.</div>', unsafe_allow_html=True)

    st.markdown('<hr class="separator" />', unsafe_allow_html=True)

    # Outlier view: boxplots of key skewed vars
    section_title("Outlier Glance — Boxplots")
    show = [c for c in ["RevolvingUtilizationOfUnsecuredLines","DebtRatio","MonthlyIncome"] if c in df_full.columns]
    cols = st.columns(len(show) if show else 1)
    if show:
        for i, c in enumerate(show):
            with cols[i]:
                fig = px.box(df_full, y=c, points=False)
                fig.update_layout(height=320, margin=dict(l=10,r=10,t=8,b=10))
                st.plotly_chart(fig, use_container_width=True)
        st.markdown('<div class="note">Long upper whiskers/tails signal extreme values that often align with higher risk.</div>', unsafe_allow_html=True)
    else:
        st.info("No key skewed columns found for boxplots.")

# =======================================================
# Interactive Lab (filters + threshold + interactive rate)
# =======================================================
elif page == "Interactive Lab":
    big_title("Interactive Lab")

    section_title("Filters")
    colF1, colF2 = st.columns(2)
    if "age" in df_full.columns and df_full["age"].notna().any():
        amin, amax = int(df_full["age"].min()), int(df_full["age"].max())
        age_rng = colF1.slider("Age range", amin, max(amax, amin+1), (amin, amax))
    else: age_rng = None
    if "MonthlyIncome" in df_full.columns and df_full["MonthlyIncome"].notna().any():
        mi_min, mi_max = float(df_full["MonthlyIncome"].min()), float(df_full["MonthlyIncome"].max())
        mi_95 = float(np.nanpercentile(df_full["MonthlyIncome"].dropna(), 95)) if df_full["MonthlyIncome"].notna().any() else mi_max
        income_rng = colF2.slider("Monthly income range", float(np.nan_to_num(mi_min, nan=0.0)),
                                  float(np.nan_to_num(mi_max, nan=100000.0)),
                                  (float(np.nan_to_num(mi_min, nan=0.0)), float(np.nan_to_num(mi_95, nan=mi_max))))
    else: income_rng = None
    del_cols = [c for c in ["NumberOfTime30-59DaysPastDueNotWorse","NumberOfTime60-89DaysPastDueNotWorse","NumberOfTimes90DaysLate"] if c in df_full.columns]
    require_del = st.checkbox("Only rows with any delinquency > 0", value=False)

    mask = pd.Series(True, index=df_full.index)
    if age_rng and "age" in df_full.columns: mask &= df_full["age"].between(age_rng[0], age_rng[1], inclusive="both")
    if income_rng and "MonthlyIncome" in df_full.columns:
        mi0, mi1 = income_rng; mask &= df_full["MonthlyIncome"].fillna(-1e12).between(mi0, mi1, inclusive="both")
    if require_del and del_cols: mask &= (df_full[del_cols].fillna(0) > 0).any(axis=1)
    df = df_full[mask].copy()

    # Interactive: choose a variable to see default rate by bins
    section_title("Interactive — Default Rate by Bins")
    choices = [c for c in ["age","MonthlyIncome","DebtRatio","RevolvingUtilizationOfUnsecuredLines"] if c in df.columns]
    if choices:
        colC1, colC2 = st.columns([2,1])
        feature = colC1.selectbox("Feature", options=choices, index=0)
        bins = colC2.slider("Bins", 4, 20, 8)
        figb = safe_rate_by_bin(df, feature, bins=bins)
        if figb: st.plotly_chart(figb, use_container_width=True)
    else:
        st.info("No numeric features available for interactive binning.")

    st.markdown('<hr class="separator" />', unsafe_allow_html=True)

    # Quick model + threshold
    section_title("Quick Model — Threshold Tuner")
    if len(df) < 200 or df["SeriousDlqin2yrs"].nunique() < 2:
        st.info("Not enough filtered rows/classes to train a model. Loosen filters.")
    else:
        y = df["SeriousDlqin2yrs"].astype(int)
        X_cols = [c for c in df.columns if c != "SeriousDlqin2yrs" and pd.api.types.is_numeric_dtype(df[c])]
        X_df = df[X_cols].copy().fillna(df[X_cols].median(numeric_only=True))
        X_train_df, X_test_df, y_train, y_test = train_test_split(X_df, y, test_size=0.2, random_state=42, stratify=y)
        scaler = StandardScaler(); X_train = scaler.fit_transform(X_train_df); X_test = scaler.transform(X_test_df)
        lr = LogisticRegression(max_iter=500, solver="liblinear", class_weight="balanced"); lr.fit(X_train, y_train)
        y_proba = lr.predict_proba(X_test)[:,1]; auc = roc_auc_score(y_test, y_proba)
        th = st.slider("Decision threshold", 0.10, 0.90, 0.50, step=0.01); y_pred = (y_proba >= th).astype(int)
        acc = accuracy_score(y_test, y_pred); prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", zero_division=0)
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("AUC", f"{auc:.3f}"); c2.metric("Precision (1)", f"{prec:.3f}"); c3.metric("Recall (1)", f"{rec:.3f}"); c4.metric("F1 (1)", f"{f1:.3f}")

# =======================================================
# Modeling & Metrics (fixed)
# =======================================================
elif page == "Modeling & Metrics":
    big_title("Modeling & Metrics (Fixed Settings)")
    y = df_full["SeriousDlqin2yrs"].astype(int)
    X_cols = [c for c in df_full.columns if c != "SeriousDlqin2yrs" and pd.api.types.is_numeric_dtype(df_full[c])]
    X_df = df_full[X_cols].copy().fillna(df_full[X_cols].median(numeric_only=True))
    TEST_SIZE, THRESH = 0.20, 0.50
    X_train_df, X_test_df, y_train, y_test = train_test_split(X_df, y, test_size=TEST_SIZE, random_state=42, stratify=y)
    scaler = StandardScaler(); X_train = scaler.fit_transform(X_train_df); X_test = scaler.transform(X_test_df)
    lr = LogisticRegression(max_iter=500, solver="liblinear", class_weight="balanced"); lr.fit(X_train, y_train)
    y_proba = lr.predict_proba(X_test)[:,1]; auc = roc_auc_score(y_test, y_proba)
    section_title(f"ROC Curve (AUC = {auc:.3f})")
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_fig = go.Figure()
    roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC"))
    roc_fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="baseline", line=dict(dash="dash")))
    roc_fig.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", height=380, margin=dict(l=10,r=10,t=40,b=10))
    st.plotly_chart(roc_fig, use_container_width=True)
    y_pred = (y_proba >= THRESH).astype(int)
    acc = accuracy_score(y_test, y_pred); prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=[0,1])
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Accuracy", f"{acc:.3f}"); c2.metric("Precision (1)", f"{prec:.3f}"); c3.metric("Recall (1)", f"{rec:.3f}"); c4.metric("F1 (1)", f"{f1:.3f}")
    section_title("Confusion Matrix")
    cm_fig = go.Figure(data=go.Heatmap(z=cm, x=["Pred 0","Pred 1"], y=["True 0","True 1"], text=cm, texttemplate="%{text}", showscale=False))
    cm_fig.update_layout(height=320, margin=dict(l=10,r=10,t=8,b=10)); st.plotly_chart(cm_fig, use_container_width=True)
    coef = pd.Series(lr.coef_[0], index=X_cols).abs().sort_values(ascending=False)
    fi = coef.head(min(15,len(coef))).reset_index(); fi.columns = ["feature","importance(|coef|)"]
    section_title("Top Feature Importance (|coef|)")
    fi_fig = px.bar(fi, x="feature", y="importance(|coef|)"); fi_fig.update_layout(height=360, margin=dict(l=10,r=10,t=8,b=10), xaxis_tickangle=-30)
    st.plotly_chart(fi_fig, use_container_width=True)
    # Cache for Risk Buckets
    st.session_state["__X_test_index__"] = X_test_df.index.tolist()
    st.session_state["__y_test__"] = y_test.tolist()
    st.session_state["__y_proba__"] = y_proba.tolist()

# =======================================================
# Risk Buckets (A/B/C/D)
# =======================================================
elif page == "Risk Buckets (A/B/C/D)":
    big_title("Risk Buckets (A/B/C/D) & Likely Defaulters")
    idx = st.session_state.get("__X_test_index__"); y_test = st.session_state.get("__y_test__"); y_proba = st.session_state.get("__y_proba__")
    if not (idx and y_test and y_proba):
        st.info("Open 'Modeling & Metrics' first to generate predictions."); st.stop()
    scored = pd.DataFrame({"prob_default": y_proba, "true_label": y_test}, index=idx)
    id_candidates = [c for c in ["ID","Id","id","CustomerID","customer_id"] if c in df_full.columns]
    id_col = id_candidates[0] if id_candidates else None
    scored["BorrowerID"] = df_full.loc[scored.index, id_col].values if id_col else scored.index
    section_title("Bucket Cutoffs")
    colA,colB,colC = st.columns(3)
    cut_A = colA.number_input("A/B cutoff", 0.0, 0.99, 0.10, step=0.01)
    cut_B = colB.number_input("B/C cutoff", cut_A, 0.995, 0.25, step=0.01)
    cut_C = colC.number_input("C/D cutoff", cut_B, 1.00, 0.50, step=0.01)
    def bucketize(p: float)->str:
        if p < cut_A: return "A"
        if p < cut_B: return "B"
        if p < cut_C: return "C"
        return "D"
    scored["risk_bucket"] = scored["prob_default"].apply(bucketize)
    section_title("Portfolio Mix by Bucket")
    counts = scored["risk_bucket"].value_counts().reindex(["A","B","C","D"]).fillna(0).astype(int).reset_index()
    counts.columns = ["bucket","count"]; fig = px.bar(counts, x="bucket", y="count", labels={"count":"Borrowers"})
    fig.update_layout(height=300, margin=dict(l=10,r=10,t=8,b=10)); st.plotly_chart(fig, use_container_width=True)
    section_title("Bucket D — Likely Defaulters (Top N)")
    n_top = st.slider("Rows to show", 5, 200, 25, step=5)
    extras = [c for c in ["MonthlyIncome","age","DebtRatio","RevolvingUtilizationOfUnsecuredLines"] if c in df_full.columns]
    cols_show = ["BorrowerID","prob_default","risk_bucket","true_label"] + extras
    topD = scored[scored["risk_bucket"]=="D"].sort_values("prob_default", ascending=False).head(n_top)
    st.dataframe(topD[cols_show], use_container_width=True)
    csv = scored[["BorrowerID","prob_default","risk_bucket","true_label"]+extras].sort_values("prob_default", ascending=False).to_csv(index=False)
    st.download_button("Download full scored test set (CSV)", data=csv, file_name="scored_test_with_buckets.csv", mime="text/csv")

# =======================================================
# Saved Figures
# =======================================================
elif page == "Saved Figures":
    big_title("Saved Figures")
    pngs = sorted(glob.glob("reports/figures/*.png"))
    if pngs:
        section_title("reports/figures")
        cols = st.columns(3)
        for i, p in enumerate(pngs):
            with cols[i % 3]: st.image(p, use_container_width=True, caption=os.path.basename(p))
    else:
        st.markdown('<div class="block note">No PNGs found in reports/figures.</div>', unsafe_allow_html=True)

# =======================================================
# Data Quality
# =======================================================
elif page == "Data Quality":
    big_title("Data Quality")
    section_title("Missing Values by Column")
    miss = df_full.isna().sum().sort_values(ascending=False); miss_df = miss[miss>0].reset_index()
    miss_df.columns = ["column","missing"]
    if not miss_df.empty:
        fig = px.bar(miss_df, x="column", y="missing", labels={"missing":"Missing rows"})
        fig.update_layout(height=320, margin=dict(l=10,r=10,t=8,b=10)); st.plotly_chart(fig, use_container_width=True)
    else:
        st.markdown('<div class="block note">No missing values.</div>', unsafe_allow_html=True)
    section_title("Basic Stats")
    st.dataframe(df_full.describe(include="all").transpose(), use_container_width=True)
