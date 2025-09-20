# dashboard.py
# Loan Default Project — Full EDA + Executive Summary + Modeling + Risk Buckets
# One-file Streamlit app with a left sidebar "pages" navigation.
# - Simple, human wording throughout
# - Executive Summary (auto-written, readable)
# - EDA: Univariate, Bivariate/Segments, Correlations/Outliers
# - Easy-to-read focus on "many loans" and "dependents"
# - Interactive Lab (light controls)
# - Modeling & Metrics (fixed) + PR, KS, Gains/Lift
# - Risk Buckets (A–D) + Bucket D table + CSV
# - Saved Figures and Data Quality
# - Robust dataset auto-discovery + safe binning + no float ages

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
    confusion_matrix, accuracy_score, precision_recall_curve, auc as sk_auc
)

# ===================== Page + Style =====================
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
</style>
""", unsafe_allow_html=True)

def big_title(t: str): st.markdown(f'<div class="big-title">{t}</div>', unsafe_allow_html=True)
def section_title(t: str): st.markdown(f'<div class="section-title">{t}</div>', unsafe_allow_html=True)
def pct(x: float) -> str: return f"{100*x:.1f}%"

# ===================== Data helpers =====================
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
    # Convert everything to numeric if possible
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # Age: ensure integer-like for display
    if "age" in df.columns:
        df["age"] = df["age"].round().astype("Int64")
    return df

@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df = _to_numeric(df)
    # Drop rows with missing target; keep only positive ages
    if "SeriousDlqin2yrs" in df.columns: df = df.dropna(subset=["SeriousDlqin2yrs"])
    if "age" in df.columns: df = df[df["age"].fillna(0) > 0]
    return df

csv_path = find_dataset()
if not csv_path:
    big_title("Loan Default — Dashboard")
    st.error("No compatible CSV found. Add e.g. `data/clean_credit.csv` or `cs-training.csv` and refresh.")
    st.stop()
df_full = load_data(csv_path)

# Helper: safe bin chart
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

# ===================== Sidebar nav =====================
with st.sidebar:
    st.title("Navigation")
    page = st.radio("Go to", [
        "Executive Summary",
        "EDA — Univariate",
        "EDA — Bivariate & Segments",
        "EDA — Correlations & Outliers",
        "Interactive Lab",
        "Modeling & Metrics",
        "Risk Buckets (A–D)",
        "Saved Figures",
        "Data Quality"
    ], index=0)
    st.caption(f"Data source: `{csv_path}`")

# ===================== Executive Summary =====================
if page == "Executive Summary":
    big_title("Executive Summary")

    # KPIs
    section_title("Portfolio Snapshot")
    c1,c2,c3,c4 = st.columns(4)
    rows = len(df_full)
    dr = df_full["SeriousDlqin2yrs"].mean() if rows else 0.0
    mi_missing = df_full["MonthlyIncome"].isna().sum() if "MonthlyIncome" in df_full.columns else 0
    age_min = int(df_full["age"].min()) if "age" in df_full.columns and df_full["age"].notna().any() else None
    age_max = int(df_full["age"].max()) if "age" in df_full.columns and df_full["age"].notna().any() else None
    c1.metric("Rows", f"{rows:,}")
    c2.metric("Default rate", pct(dr))
    c3.metric("Missing MonthlyIncome", f"{mi_missing:,}" if "MonthlyIncome" in df_full.columns else "—")
    c4.metric("Age span", f"{age_min}–{age_max}" if age_min is not None else "—")

    st.markdown('<hr class="separator" />', unsafe_allow_html=True)

    # Clear, human highlights
    section_title("Key Points (plain words)")
    bullets = []

    if "RevolvingUtilizationOfUnsecuredLines" in df_full.columns:
        util = df_full["RevolvingUtilizationOfUnsecuredLines"]
        u99 = util.quantile(.99)
        high_util_rate = df_full[util >= u99]["SeriousDlqin2yrs"].mean()
        bullets.append(f"People with **very high card utilization** (top 1%) default more (**{pct(high_util_rate)}**) than the average (**{pct(dr)}**).")

    late_cols = [c for c in ["NumberOfTimes90DaysLate","NumberOfTime60-89DaysPastDueNotWorse","NumberOfTime30-59DaysPastDueNotWorse"] if c in df_full.columns]
    if late_cols:
        any_late = (df_full[late_cols].fillna(0) > 0).any(axis=1)
        bullets.append(f"Having **any past-due history** raises risk. That group’s default rate is **{pct(df_full.loc[any_late,'SeriousDlqin2yrs'].mean())}**.")

    if "DebtRatio" in df_full.columns:
        d99 = df_full["DebtRatio"].quantile(.99)
        bullets.append(f"**Very high debt ratio** (top 1%) is a warning sign (see EDA pages for the shape).")

    if "MonthlyIncome" in df_full.columns and df_full["MonthlyIncome"].notna().any():
        low_inc = df_full["MonthlyIncome"] <= df_full["MonthlyIncome"].quantile(.2)
        bullets.append(f"**Lower income** (bottom 20%) shows higher default rate (**{pct(df_full.loc[low_inc,'SeriousDlqin2yrs'].mean())}**).")

    if "NumberOfOpenCreditLinesAndLoans" in df_full.columns:
        many_loans = df_full["NumberOfOpenCreditLinesAndLoans"] >= df_full["NumberOfOpenCreditLinesAndLoans"].quantile(.8)
        bullets.append(f"Borrowers with **many open credit lines/loans** (top 20%) trend riskier than the portfolio average.")

    if "NumberOfDependents" in df_full.columns and df_full["NumberOfDependents"].notna().any():
        dep_3plus = df_full["NumberOfDependents"] >= 3
        bullets.append(f"Borrowers with **3+ dependents** show higher risk than those with fewer dependents.")

    if bullets:
        for b in bullets: st.markdown(f"- {b}")
    else:
        st.markdown("- Data loaded, but not enough columns to calculate highlights.")

    # A short “who is most risky” sentence
    section_title("Who looks most risky (simple profile)")
    parts = []
    if "RevolvingUtilizationOfUnsecuredLines" in df_full.columns: parts.append("very high card utilization")
    if late_cols: parts.append("any late payment history")
    if "DebtRatio" in df_full.columns: parts.append("very high debt ratio")
    if "MonthlyIncome" in df_full.columns: parts.append("lower income")
    if "age" in df_full.columns: parts.append("younger age")
    if "NumberOfOpenCreditLinesAndLoans" in df_full.columns: parts.append("many open credit lines/loans")
    if "NumberOfDependents" in df_full.columns: parts.append("3+ dependents")
    if parts:
        st.markdown("People who combine **" + ", ".join(parts) + "** tend to sit in the highest risk tail.")
    st.markdown('<div class="note">See “Risk Buckets (A–D)” to group borrowers by predicted risk and view the worst bucket.</div>', unsafe_allow_html=True)

# ===================== EDA — Univariate =====================
elif page == "EDA — Univariate":
    big_title("EDA — Univariate (Distributions)")

    # Target
    section_title("Target: SeriousDlqin2yrs (0 = no default, 1 = default)")
    counts = df_full["SeriousDlqin2yrs"].value_counts().sort_index()
    fig = px.bar(x=[str(int(k)) for k in counts.index], y=counts.values,
                 labels={"x":"SeriousDlqin2yrs","y":"Count"})
    fig.update_layout(margin=dict(l=10,r=10,t=8,b=10), height=320)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('<div class="note">The dataset is imbalanced: far more non-defaulters than defaulters.</div>', unsafe_allow_html=True)

    # Key numeric distributions (hist + violin + log-hist for heavy tails)
    section_title("Feature Distributions")
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
                               labels={c:c, "color":"Default"})
            fig.update_layout(height=300, margin=dict(l=10,r=10,t=8,b=10), legend_title_text="Default")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            v = px.violin(df_full, y=c, box=True, points=False)
            v.update_layout(height=145, margin=dict(l=10,r=10,t=8,b=6))
            st.plotly_chart(v, use_container_width=True)
            # auto log-hist for heavy tails
            series = df_full[c].dropna()
            if len(series) > 0 and series.max() > 50 * max(1.0, series.median() if series.median() > 0 else 1):
                hlog = px.histogram(np.log1p(series), nbins=60, labels={"value": f"log1p({c})"})
                hlog.update_layout(height=145, margin=dict(l=10,r=10,t=6,b=8))
                st.plotly_chart(hlog, use_container_width=True)

        st.markdown('<div class="note">Histogram shows class split; violin/box show outliers. '
                    'If you see a log chart, that variable has a very long right tail.</div>', unsafe_allow_html=True)
        st.markdown('<hr class="separator" />', unsafe_allow_html=True)

# ===================== EDA — Bivariate & Segments =====================
elif page == "EDA — Bivariate & Segments":
    big_title("EDA — Bivariate & Segments")

    # Default rate by binned features
    section_title("Default Rate by Binned Features")
    cols = [c for c in ["age","MonthlyIncome","DebtRatio","RevolvingUtilizationOfUnsecuredLines"] if c in df_full.columns]
    grid = st.columns(2)
    for i, c in enumerate(cols):
        with grid[i % 2]:
            figb = safe_rate_by_bin(df_full, c, bins=8)
            if figb: st.plotly_chart(figb, use_container_width=True)
    st.markdown('<div class="note">Binning helps show thresholds. For example, very high utilization often jumps in risk.</div>', unsafe_allow_html=True)

    st.markdown('<hr class="separator" />', unsafe_allow_html=True)

    # Debt vs Utilization (colored by default)
    section_title("Debt Ratio vs Card Utilization (colored by Default)")
    if {"DebtRatio","RevolvingUtilizationOfUnsecuredLines"}.issubset(df_full.columns):
        tmp = df_full[["DebtRatio","RevolvingUtilizationOfUnsecuredLines","SeriousDlqin2yrs"]].dropna()
        s = px.scatter(tmp, x="RevolvingUtilizationOfUnsecuredLines", y="DebtRatio",
                       color=tmp["SeriousDlqin2yrs"].astype(str), opacity=0.5,
                       labels={"color":"Default"})
        s.update_layout(height=420, margin=dict(l=10,r=10,t=8,b=10), legend_title_text="Default")
        st.plotly_chart(s, use_container_width=True)
        st.markdown('<div class="note">The top-right area (high utilization + high debt) holds more defaulters.</div>', unsafe_allow_html=True)

    st.markdown('<hr class="separator" />', unsafe_allow_html=True)

    # Many loans and dependents (simple and clear)
    section_title("Do defaulters have many loans and dependents?")
    cols_show = [c for c in ["NumberOfOpenCreditLinesAndLoans","NumberOfDependents"] if c in df_full.columns]
    if "NumberOfOpenCreditLinesAndLoans" in df_full.columns:
        # Bucket loan counts
        loans = df_full["NumberOfOpenCreditLinesAndLoans"].fillna(0)
        loan_band = pd.cut(loans, bins=[-0.1,0,3,6,10, loans.max()+1], labels=["0","1–3","4–6","7–10",">10"])
        rate_by_loans = df_full.groupby(loan_band)["SeriousDlqin2yrs"].mean().reset_index()
        rate_by_loans.columns = ["Loan band","Default rate"]
        rate_by_loans["Default rate (%)]"] = 100*rate_by_loans["Default rate"]
        figL = px.bar(rate_by_loans, x="Loan band", y="Default rate (%)]")
        figL.update_layout(height=320, margin=dict(l=10,r=10,t=8,b=10))
        st.plotly_chart(figL, use_container_width=True)
    if "NumberOfDependents" in df_full.columns:
        deps = df_full["NumberOfDependents"].fillna(0)
        dep_band = pd.cut(deps, bins=[-0.1,0,1,2,3, deps.max()+1], labels=["0","1","2","3",">3"])
        rate_by_dep = df_full.groupby(dep_band)["SeriousDlqin2yrs"].mean().reset_index()
        rate_by_dep.columns = ["Dependents","Default rate"]
        rate_by_dep["Default rate (%)"] = 100*rate_by_dep["Default rate"]
        figD = px.bar(rate_by_dep, x="Dependents", y="Default rate (%)")
        figD.update_layout(height=320, margin=dict(l=10,r=10,t=8,b=10))
        st.plotly_chart(figD, use_container_width=True)
    st.markdown('<div class="note">More open loans and more dependents both line up with higher default, especially at the top bands.</div>', unsafe_allow_html=True)

    st.markdown('<hr class="separator" />', unsafe_allow_html=True)

    # Delinquency intensity counts
    section_title("How common are heavy delinquency histories?")
    dl_cols = [c for c in ["NumberOfTime30-59DaysPastDueNotWorse","NumberOfTime60-89DaysPastDueNotWorse","NumberOfTimes90DaysLate"] if c in df_full.columns]
    if dl_cols:
        dd = df_full[dl_cols].fillna(0).sum(axis=1)
        band = pd.cut(dd, bins=[-0.1,0,1,3,10, dd.max()+1], labels=["0","1","2–3","4–10",">10"])
        ct = band.value_counts().sort_index().reset_index()
        ct.columns = ["Delinquency band","Count"]
        bfig = px.bar(ct, x="Delinquency band", y="Count")
        bfig.update_layout(height=300, margin=dict(l=10,r=10,t=8,b=10))
        st.plotly_chart(bfig, use_container_width=True)
        st.markdown('<div class="note">Heavy histories are rare, but those borrowers are much riskier.</div>', unsafe_allow_html=True)

# ===================== EDA — Correlations & Outliers =====================
elif page == "EDA — Correlations & Outliers":
    big_title("EDA — Correlations & Outliers")

    num_cols = [c for c in df_full.columns if pd.api.types.is_numeric_dtype(df_full[c])]
    if len(num_cols) >= 3:
        section_title("Correlation Heatmap (numeric)")
        corr = df_full[num_cols].corr(numeric_only=True)
        heat = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.index, zmin=-1, zmax=1,
                                         colorbar=dict(title="ρ")))
        heat.update_layout(height=520, margin=dict(l=10,r=10,t=8,b=10))
        st.plotly_chart(heat, use_container_width=True)
        st.markdown('<div class="note">Use this to spot overlap between features.</div>', unsafe_allow_html=True)

    # Correlation vs target (absolute)
    if "SeriousDlqin2yrs" in df_full.columns:
        section_title("Top features by absolute correlation with default")
        num_cols_target = [c for c in df_full.columns if c != "SeriousDlqin2yrs" and pd.api.types.is_numeric_dtype(df_full[c])]
        corr_t = (df_full[num_cols_target + ["SeriousDlqin2yrs"]]
                  .corr(numeric_only=True)["SeriousDlqin2yrs"]
                  .abs().drop("SeriousDlqin2yrs").sort_values(ascending=False).head(15))
        bar = px.bar(corr_t.reset_index(), x="index", y="SeriousDlqin2yrs",
                     labels={"index":"feature","SeriousDlqin2yrs":"|corr with target|"})
        bar.update_layout(height=360, margin=dict(l=10,r=10,t=8,b=10), xaxis_tickangle=-30)
        st.plotly_chart(bar, use_container_width=True)
        st.markdown('<div class="note">A quick screen of the strongest signals.</div>', unsafe_allow_html=True)

    st.markdown('<hr class="separator" />', unsafe_allow_html=True)

    section_title("Outlier Glance — Boxplots")
    show = [c for c in ["RevolvingUtilizationOfUnsecuredLines","DebtRatio","MonthlyIncome"] if c in df_full.columns]
    cols = st.columns(len(show) if show else 1)
    if show:
        for i, c in enumerate(show):
            with cols[i]:
                fig = px.box(df_full, y=c, points=False)
                fig.update_layout(height=320, margin=dict(l=10,r=10,t=8,b=10))
                st.plotly_chart(fig, use_container_width=True)
        st.markdown('<div class="note">Long upper tails often coincide with higher risk.</div>', unsafe_allow_html=True)
    else:
        st.info("No key skewed columns found for boxplots.")

# ===================== Interactive Lab =====================
elif page == "Interactive Lab":
    big_title("Interactive Lab (Filters + Default Rate by Bins + Threshold)")

    section_title("Filters")
    colF1, colF2 = st.columns(2)
    if "age" in df_full.columns and df_full["age"].notna().any():
        amin, amax = int(df_full["age"].min()), int(df_full["age"].max())
        age_rng = colF1.slider("Age range", amin, max(amax, amin+1), (amin, amax))
    else: age_rng = None
    if "MonthlyIncome" in df_full.columns and df_full["MonthlyIncome"].notna().any():
        mi_min, mi_max = float(df_full["MonthlyIncome"].min()), float(df_full["MonthlyIncome"].max())
        mi_95 = float(np.nanpercentile(df_full["MonthlyIncome"].dropna(), 95)) if df_full["MonthlyIncome"].notna().any() else mi_max
        income_rng = colF2.slider("Monthly income range",
                                  float(np.nan_to_num(mi_min, nan=0.0)),
                                  float(np.nan_to_num(mi_max, nan=100000.0)),
                                  (float(np.nan_to_num(mi_min, nan=0.0)),
                                   float(np.nan_to_num(mi_95, nan=mi_max))))
    else: income_rng = None
    del_cols = [c for c in ["NumberOfTime30-59DaysPastDueNotWorse","NumberOfTime60-89DaysPastDueNotWorse","NumberOfTimes90DaysLate"] if c in df_full.columns]
    require_del = st.checkbox("Only rows with any delinquency > 0", value=False)

    mask = pd.Series(True, index=df_full.index)
    if age_rng and "age" in df_full.columns: mask &= df_full["age"].between(age_rng[0], age_rng[1], inclusive="both")
    if income_rng and "MonthlyIncome" in df_full.columns:
        mi0, mi1 = income_rng; mask &= df_full["MonthlyIncome"].fillna(-1e12).between(mi0, mi1, inclusive="both")
    if require_del and del_cols: mask &= (df_full[del_cols].fillna(0) > 0).any(axis=1)
    df = df_full[mask].copy()

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

# ===================== Modeling & Metrics =====================
elif page == "Modeling & Metrics":
    big_title("Modeling & Metrics (Fixed Settings)")

    y = df_full["SeriousDlqin2yrs"].astype(int)
    X_cols = [c for c in df_full.columns if c != "SeriousDlqin2yrs" and pd.api.types.is_numeric_dtype(df_full[c])]
    X_df = df_full[X_cols].copy().fillna(df_full[X_cols].median(numeric_only=True))

    TEST_SIZE, THRESH = 0.20, 0.50
    X_train_df, X_test_df, y_train, y_test = train_test_split(X_df, y, test_size=TEST_SIZE, random_state=42, stratify=y)
    scaler = StandardScaler(); X_train = scaler.fit_transform(X_train_df); X_test = scaler.transform(X_test_df)

    lr = LogisticRegression(max_iter=500, solver="liblinear", class_weight="balanced")
    lr.fit(X_train, y_train)

    y_proba = lr.predict_proba(X_test)[:,1]
    auc = roc_auc_score(y_test, y_proba)

    # ROC
    section_title(f"ROC Curve (AUC = {auc:.3f})")
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_fig = go.Figure()
    roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC"))
    roc_fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Baseline", line=dict(dash="dash")))
    roc_fig.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
                          height=380, margin=dict(l=10,r=10,t=40,b=10))
    st.plotly_chart(roc_fig, use_container_width=True)

    # Metrics @ fixed threshold
    y_pred = (y_proba >= THRESH).astype(int)
    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=[0,1])

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Accuracy", f"{acc:.3f}")
    c2.metric("Precision (1)", f"{prec:.3f}")
    c3.metric("Recall (1)", f"{rec:.3f}")
    c4.metric("F1 (1)", f"{f1:.3f}")

    section_title("Confusion Matrix")
    cm_fig = go.Figure(data=go.Heatmap(z=cm, x=["Pred 0","Pred 1"], y=["True 0","True 1"], text=cm, texttemplate="%{text}", showscale=False))
    cm_fig.update_layout(height=320, margin=dict(l=10,r=10,t=8,b=10))
    st.plotly_chart(cm_fig, use_container_width=True)

    # Feature importance (|coef|)
    coef = pd.Series(lr.coef_[0], index=X_cols).abs().sort_values(ascending=False)
    fi = coef.head(min(15, len(coef))).reset_index()
    fi.columns = ["feature","importance(|coef|)"]
    section_title("Top Feature Importance (|coef|)")
    fi_fig = px.bar(fi, x="feature", y="importance(|coef|)")
    fi_fig.update_layout(height=360, margin=dict(l=10,r=10,t=8,b=10), xaxis_tickangle=-30)
    st.plotly_chart(fi_fig, use_container_width=True)

    # Precision–Recall Curve
    section_title("Precision–Recall Curve")
    precs, recs, _thr = precision_recall_curve(y_test, y_proba)
    pr_auc = sk_auc(recs, precs)
    pr_fig = go.Figure()
    pr_fig.add_trace(go.Scatter(x=recs, y=precs, mode="lines", name=f"PR (AUC={pr_auc:.3f})"))
    pr_fig.update_layout(xaxis_title="Recall", yaxis_title="Precision", height=360, margin=dict(l=10,r=10,t=40,b=10))
    st.plotly_chart(pr_fig, use_container_width=True)

    # KS curve
    section_title("KS Curve (Cumulative separation)")
    df_ks = pd.DataFrame({"y": y_test.values, "p": y_proba}).sort_values("p", ascending=True)
    df_ks["cum_all"] = np.arange(1, len(df_ks)+1)/len(df_ks)
    df_ks["cum_bad"] = (df_ks["y"].cumsum())/max(1, df_ks["y"].sum())
    df_ks["cum_good"] = ((1-df_ks["y"]).cumsum())/max(1, (1-df_ks["y"]).sum())
    ks = float((df_ks["cum_bad"] - df_ks["cum_good"]).max())
    ks_fig = go.Figure()
    ks_fig.add_trace(go.Scatter(x=df_ks["cum_all"], y=df_ks["cum_bad"], mode="lines", name="Bad CDF"))
    ks_fig.add_trace(go.Scatter(x=df_ks["cum_all"], y=df_ks["cum_good"], mode="lines", name="Good CDF"))
    ks_fig.update_layout(xaxis_title="Population proportion (sorted by score)", yaxis_title="Cumulative proportion",
                         height=360, margin=dict(l=10,r=10,t=40,b=10))
    st.plotly_chart(ks_fig, use_container_width=True)
    st.markdown(f'<div class="note">KS ≈ <b>{ks:.3f}</b>. Higher is better separation.</div>', unsafe_allow_html=True)

    # Gains & Lift (deciles)
    section_title("Gains & Lift (Deciles)")
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
    st.plotly_chart(lift_fig, use_container_width=True)

    # Save predictions to session for Risk Buckets
    st.session_state["__X_test_index__"] = X_test_df.index.tolist()
    st.session_state["__y_test__"] = y_test.tolist()
    st.session_state["__y_proba__"] = y_proba.tolist()

# ===================== Risk Buckets (A–D) =====================
elif page == "Risk Buckets (A–D)":
    big_title("Risk Buckets (A–D) & Likely Defaulters")

    idx = st.session_state.get("__X_test_index__")
    y_test = st.session_state.get("__y_test__")
    y_proba = st.session_state.get("__y_proba__")

    if not (idx and y_test and y_proba):
        st.info("Open “Modeling & Metrics” first so we can score borrowers.")
        st.stop()

    scored = pd.DataFrame({"prob_default": y_proba, "true_label": y_test}, index=idx)

    # Borrower ID if present
    id_candidates = [c for c in ["ID","Id","id","CustomerID","customer_id"] if c in df_full.columns]
    id_col = id_candidates[0] if id_candidates else None
    scored["BorrowerID"] = df_full.loc[scored.index, id_col].values if id_col else scored.index

    section_title("Bucket Cutoffs")
    colA,colB,colC = st.columns(3)
    cut_A = colA.number_input("A/B cutoff", 0.0, 0.99, 0.10, step=0.01)
    cut_B = colB.number_input("B/C cutoff", cut_A, 0.995, 0.25, step=0.01)
    cut_C = colC.number_input("C/D cutoff", cut_B, 1.00, 0.50, step=0.01)
    st.caption(f"Buckets: A < {cut_A:.2f}  •  {cut_A:.2f}–{cut_B:.2f} = B  •  {cut_B:.2f}–{cut_C:.2f} = C  •  D ≥ {cut_C:.2f}")

    def bucketize(p: float)->str:
        if p < cut_A: return "A"
        if p < cut_B: return "B"
        if p < cut_C: return "C"
        return "D"
    scored["risk_bucket"] = scored["prob_default"].apply(bucketize)

    section_title("Portfolio Mix by Bucket")
    counts = (scored["risk_bucket"].value_counts()
              .reindex(["A","B","C","D"])
              .fillna(0).astype(int).reset_index())
    counts.columns = ["bucket","count"]
    fig = px.bar(counts, x="bucket", y="count", labels={"count":"Borrowers"})
    fig.update_layout(height=300, margin=dict(l=10,r=10,t=8,b=10))
    st.plotly_chart(fig, use_container_width=True)

    section_title("Bucket D — Likely Defaulters (Top N)")
    n_top = st.slider("Rows to show", 5, 200, 25, step=5)
    extras = [c for c in ["MonthlyIncome","age","DebtRatio","RevolvingUtilizationOfUnsecuredLines",
                          "NumberOfOpenCreditLinesAndLoans","NumberOfDependents"] if c in df_full.columns]
    cols_show = ["BorrowerID","prob_default","risk_bucket","true_label"] + extras
    topD = scored[scored["risk_bucket"]=="D"].sort_values("prob_default", ascending=False).head(n_top)
    st.dataframe(topD[cols_show], use_container_width=True)

    csv = scored[["BorrowerID","prob_default","risk_bucket","true_label"]+extras].sort_values("prob_default", ascending=False).to_csv(index=False)
    st.download_button("Download full scored test set (CSV)", data=csv, file_name="scored_test_with_buckets.csv", mime="text/csv")

# ===================== Saved Figures =====================
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

# ===================== Data Quality =====================
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
    # Make sure ages display without decimals
    show_df = df_full.copy()
    if "age" in show_df.columns:
        show_df["age"] = show_df["age"].astype("Int64")
    st.dataframe(show_df.describe(include="all").transpose(), use_container_width=True)
