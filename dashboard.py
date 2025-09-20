# dashboard.py
# Streamlit dashboard for the Loan Default Exam Project
# - One-file replacement
# - Consistent teal title boxes (big + section)
# - Live Plotly charts (no missing PNGs)
# - Robust CSV discovery & dirty-data handling
# - Works locally and on Streamlit Cloud

from __future__ import annotations

import os
import glob
from typing import Optional, List

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ===============================
# Page setup
# ===============================
st.set_page_config(page_title="Loan Default — Dashboard", layout="wide")

# ===============================
# Styles (title boxes)
# ===============================
st.markdown("""
<style>
:root {
  --teal: #007c82;           /* main teal */
  --teal-light: #e6f6f7;     /* light bg tint */
  --ink: #0f172a;            /* text */
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

.small-muted {
  color: var(--muted);
  font-size: 0.9rem;
}

.block {
  background: #ffffff;
  border-radius: var(--radius);
  box-shadow: var(--shadow);
  padding: 14px;
}

hr.separator {
  border: none;
  height: 1px;
  background: #e2e8f0;
  margin: 18px 0;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="big-title">Loan Default — Interactive Dashboard</div>', unsafe_allow_html=True)
st.caption("Live charts, robust to cloud quirks. Uses Plotly for fast, reliable rendering.")

# ===============================
# Helpers & data loading
# ===============================

REQUIRED_COLS = {
    "SeriousDlqin2yrs",  # target
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
    """Find the first CSV that has the expected columns."""
    candidates_patterns: List[str] = [
        "data/*.csv",
        "datasets/*.csv",
        "*.csv",
        "input/*.csv",
        "inputs/*.csv",
    ]
    found_files: List[str] = []
    for pat in candidates_patterns:
        found_files.extend(glob.glob(pat))

    # Prefer common names if multiple are present
    priority_names = [
        "data/clean_credit.csv",
        "data/credit_clean.csv",
        "data/credit.csv",
        "cs-training.csv",
        "train.csv",
    ]
    # Move priority files to the front if present
    ordered = []
    seen = set()
    for p in priority_names:
        if p in found_files and p not in seen:
            ordered.append(p); seen.add(p)
    for f in found_files:
        if f not in seen:
            ordered.append(f); seen.add(f)

    for path in ordered:
        try:
            sample = pd.read_csv(path, nrows=5)
        except Exception:
            continue
        cols = set([c.strip() for c in sample.columns])
        # sometimes MonthlyIncome may be missing in subsets; accept if target + majority present
        overlap = len(cols.intersection(REQUIRED_COLS))
        if "SeriousDlqin2yrs" in cols and overlap >= max(7, int(0.6 * len(REQUIRED_COLS))):
            return path
    return None

def _to_numeric(df: pd.DataFrame, exclude: Optional[List[str]]=None) -> pd.DataFrame:
    """Coerce all columns to numeric where possible, to avoid 'could not convert string...' errors."""
    exclude = exclude or []
    for c in df.columns:
        if c in exclude:
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

@st.cache_data(show_spinner=False)
def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Normalize column names (strip)
    df.columns = [c.strip() for c in df.columns]
    # Coerce numerics safely (include target)
    df = _to_numeric(df)
    # Remove obvious junk rows where target is missing
    if "SeriousDlqin2yrs" in df.columns:
        df = df.dropna(subset=["SeriousDlqin2yrs"])
    # Clip weird ages if needed
    if "age" in df.columns:
        df = df[df["age"] > 0]
    return df

def has_cols(df: pd.DataFrame, cols: List[str]) -> bool:
    return all(c in df.columns for c in cols)

def pct(x: float) -> str:
    return f"{x*100:.1f}%"

# ===============================
# Load dataset (or show guidance)
# ===============================
csv_path = find_dataset()
if not csv_path:
    st.error(
        "I couldn’t find a compatible CSV in your repo. "
        "Please add one (e.g., `data/clean_credit.csv` or `cs-training.csv`) "
        "with the Give Me Some Credit columns. Then refresh."
    )
    st.stop()

df = load_data(csv_path)

# Keep only relevant cols (if extra exist, that’s fine)
available = [c for c in REQUIRED_COLS if c in df.columns]
if "SeriousDlqin2yrs" not in df.columns:
    st.error("The dataset is missing the target column `SeriousDlqin2yrs`.")
    st.stop()

# ===============================
# Sidebar filters
# ===============================
with st.sidebar:
    st.title("Filters")
    st.caption(f"Data source: `{csv_path}`")

    # Age filter
    if "age" in df.columns and df["age"].notna().any():
        amin, amax = int(df["age"].min()), int(df["age"].max())
        age_rng = st.slider("Age range", min_value=amin, max_value=max(amax, amin+1), value=(amin, amax))
    else:
        age_rng = None

    # MonthlyIncome filter
    if "MonthlyIncome" in df.columns and df["MonthlyIncome"].notna().any():
        mi_min, mi_max = float(df["MonthlyIncome"].min()), float(df["MonthlyIncome"].max())
        income_rng = st.slider("Monthly income range", min_value=float(np.nan_to_num(mi_min, nan=0.0)),
                               max_value=float(np.nan_to_num(mi_max, nan=100000.0)),
                               value=(float(np.nan_to_num(mi_min, nan=0.0)), float(np.nan_to_num(np.nanpercentile(df["MonthlyIncome"].dropna(), 95), nan=mi_max))))
    else:
        income_rng = None

    # Delinquency >0 filters (checkboxes)
    delin_cols = [c for c in [
        "NumberOfTime30-59DaysPastDueNotWorse",
        "NumberOfTime60-89DaysPastDueNotWorse",
        "NumberOfTimes90DaysLate"
    ] if c in df.columns]

    require_any_delinquency = st.checkbox("Only rows with any delinquency > 0", value=False)

# Apply filters
mask = pd.Series(True, index=df.index)
if age_rng and "age" in df.columns:
    mask &= df["age"].between(age_rng[0], age_rng[1], inclusive="both")
if income_rng and "MonthlyIncome" in df.columns:
    mi0, mi1 = income_rng
    mask &= df["MonthlyIncome"].fillna(-1e12).between(mi0, mi1, inclusive="both")
if require_any_delinquency and delin_cols:
    any_delinq = (df[delin_cols].fillna(0) > 0).any(axis=1)
    mask &= any_delinq

df_filt = df[mask].copy()

# ===============================
# Overview (KPIs)
# ===============================
st.markdown('<div class="section-title">Overview</div>', unsafe_allow_html=True)
with st.container():
    c1, c2, c3, c4 = st.columns(4)
    total_rows = len(df_filt)
    target_mean = float(df_filt["SeriousDlqin2yrs"].mean()) if total_rows > 0 else 0.0
    n_missing_income = int(df_filt["MonthlyIncome"].isna().sum()) if "MonthlyIncome" in df_filt.columns else 0
    age_span = (int(df_filt["age"].min()), int(df_filt["age"].max())) if "age" in df_filt.columns and total_rows else (None, None)

    c1.metric("Rows (after filters)", f"{total_rows:,}")
    c2.metric("Default rate", pct(target_mean))
    c3.metric("Missing MonthlyIncome", f"{n_missing_income:,}" if "MonthlyIncome" in df_filt.columns else "—")
    c4.metric("Age span", f"{age_span[0]}–{age_span[1]}" if all(age_span) else "—")

st.markdown('<hr class="separator" />', unsafe_allow_html=True)

# ===============================
# Target distribution
# ===============================
st.markdown('<div class="section-title">Target Distribution (SeriousDlqin2yrs)</div>', unsafe_allow_html=True)
with st.container():
    counts = df_filt["SeriousDlqin2yrs"].value_counts(dropna=False).sort_index()
    fig = px.bar(
        x=[str(int(k)) for k in counts.index],
        y=counts.values,
        labels={"x": "SeriousDlqin2yrs", "y": "Count"},
        title=None
    )
    fig.update_layout(margin=dict(l=10, r=10, t=8, b=10), height=320)
    st.plotly_chart(fig, use_container_width=True)

# ===============================
# Distributions (key numeric features)
# ===============================
num_cols = [c for c in df_filt.columns if c != "SeriousDlqin2yrs" and pd.api.types.is_numeric_dtype(df_filt[c])]
top_show = [c for c in [
    "age", "MonthlyIncome", "DebtRatio",
    "RevolvingUtilizationOfUnsecuredLines",
    "NumberOfOpenCreditLinesAndLoans",
] if c in num_cols]

if top_show:
    st.markdown('<div class="section-title">Feature Distributions (colored by Default)</div>', unsafe_allow_html=True)
    grid = st.columns(2)
    for i, col in enumerate(top_show):
        with grid[i % 2]:
            fig = px.histogram(
                df_filt, x=col, color=df_filt["SeriousDlqin2yrs"].astype(str),
                nbins=50, barmode="overlay", opacity=0.65,
                labels={col: col, "color": "Default"}
            )
            fig.update_layout(margin=dict(l=10, r=10, t=8, b=10), height=320, legend_title_text="Default")
            st.plotly_chart(fig, use_container_width=True)

# ===============================
# Default rate by binned variables
# ===============================
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

binnable = [c for c in ["age", "MonthlyIncome", "DebtRatio",
                        "RevolvingUtilizationOfUnsecuredLines"] if c in df_filt.columns]

if binnable:
    st.markdown('<div class="section-title">Default Rate by Binned Features</div>', unsafe_allow_html=True)
    cols = st.columns(2)
    for i, ccol in enumerate(binnable[:4]):
        fig = rate_by_bin(df_filt, ccol, bins=8)
        if fig:
            with cols[i % 2]:
                st.plotly_chart(fig, use_container_width=True)

# ===============================
# Correlation heatmap (numeric)
# ===============================
if len(num_cols) >= 3:
    st.markdown('<div class="section-title">Correlation Heatmap (Numeric Features)</div>', unsafe_allow_html=True)
    corr = df_filt[num_cols + ["SeriousDlqin2yrs"]].corr(numeric_only=True)
    heat = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.index,
        zmin=-1, zmax=1,
        colorbar=dict(title="ρ")
    ))
    heat.update_layout(margin=dict(l=10, r=10, t=8, b=10), height=520)
    st.plotly_chart(heat, use_container_width=True)

# ===============================
# Missingness
# ===============================
st.markdown('<div class="section-title">Missing Values by Column</div>', unsafe_allow_html=True)
miss = df_filt.isna().sum().sort_values(ascending=False)
miss_df = miss[miss > 0].reset_index()
miss_df.columns = ["column", "missing"]
if not miss_df.empty:
    fig = px.bar(miss_df, x="column", y="missing", labels={"missing": "Missing rows"})
    fig.update_layout(margin=dict(l=10, r=10, t=8, b=10), height=320)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.markdown('<div class="block small-muted">No missing values in the filtered data.</div>', unsafe_allow_html=True)

# ===============================
# Fallback: show any saved figure PNGs if present (optional)
# ===============================
pngs = sorted(glob.glob("reports/figures/*.png"))
if pngs:
    st.markdown('<div class="section-title">Saved Figures (from reports/figures)</div>', unsafe_allow_html=True)
    cols = st.columns(3)
    for i, p in enumerate(pngs):
        with cols[i % 3]:
            st.image(p, use_container_width=True)
