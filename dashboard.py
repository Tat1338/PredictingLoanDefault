# dashboard.py
# Loan Default Dashboard — EDA + Modeling + A–D Buckets + Client Credit Check
# - Teal styling + clear titles
# - Full navigation (Introduction → Summary & Conclusion)
# - Hero image on the first page (assets/credit_risk_hero.jpg)
# - Client Credit Check with mini-calculators (Debt Ratio, Utilization)
# - Robust EDA, Modeling, Buckets, Saved Figures, Data Quality

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

# -----------------------------------------------------------------------------
# Page config + global styles
# -----------------------------------------------------------------------------
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
.hero-img img { border-radius: 16px; box-shadow: 0 6px 18px rgba(0,0,0,0.10); }
</style>
""", unsafe_allow_html=True)

def big_title(t: str): st.markdown(f'<div class="big-title">{t}</div>', unsafe_allow_html=True)
def section_title(t: str): st.markdown(f'<div class="section-title">{t}</div>', unsafe_allow_html=True)
def pct(x: float) -> str: return f"{100*x:.1f}%"

# -----------------------------------------------------------------------------
# Display labels
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# Data helpers
# -----------------------------------------------------------------------------
HERO_PATH = "assets/credit_risk_hero.jpg"

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

# -----------------------------------------------------------------------------
# Sidebar navigation
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("Navigation")
    page = st.radio("Go to", [
        "Introduction",
        "Feature Distributions",
        "Relationships & Segments",
        "Correlations & Outliers",
        "Interactive Lab",
        "Modeling & Metrics",
        "Risk Buckets (A–D)",
        "Client Credit Check",
        "Saved Figures",
        "Data Quality",
        "Summary & Conclusion"
    ], index=0, label_visibility="visible")
    st.caption(f"Data source: `{csv_path}`")

# -----------------------------------------------------------------------------
# 1) Introduction
# -----------------------------------------------------------------------------
if page == "Introduction":
    if os.path.exists(HERO_PATH):
        st.markdown('<div class="hero-img">', unsafe_allow_html=True)
        st.image(HERO_PATH, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    big_title("Loan Default Dashboard")
    # Hero image at the top of the first page
    img_path = "assets/credit_risk_hero.jpg"
    if os.path.exists(img_path):
        st.image(img_path, use_container_width=True)

    section_title("Purpose")
    st.markdown("""
This dashboard helps a lender identify borrowers who are more likely to fall into **serious delinquency within two years**.  
The focus is practical: use a handful of familiar signals (utilization, debt ratio, past-due history, income, age, exposure) with a transparent model to support **portfolio monitoring** and **case-by-case decisions**.
""")

    section_title("How to use this")
    st.markdown("""
1. **Feature Views** — see the patterns most linked with risk.  
2. **Model & Threshold** — check discrimination (ROC/PR), calibration, and pick a threshold.  
3. **Risk Buckets (A–D)** — translate probabilities into policy buckets.  
4. **Client Credit Check** — enter a few fields and get a **GO / CAUTION / STOP** verdict.
""")

    section_title("Target & key terms")
    st.markdown(
        "- **Serious Delinquency (within 2 Years)**: target (0/1). *1 = serious delinquency occurred within two years.*\n"
        "- **Card/Line Utilization**: balances ÷ credit limits (values near 1.0 mean lines are close to maxed).\n"
        "- **Debt Ratio**: total monthly debt payments ÷ gross monthly income."
    )

    section_title("Portfolio Snapshot")
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

# -----------------------------------------------------------------------------
# 2) Feature Distributions
# -----------------------------------------------------------------------------
elif page == "Feature Distributions":
    big_title("Feature Distributions")

    section_title("Serious Delinquency within 2 Years (0/1)")
    counts = df_full["SeriousDlqin2yrs"].value_counts().sort_index()
    fig = px.bar(x=[str(int(k)) for k in counts.index], y=counts.values,
                 labels={"x": DISPLAY["SeriousDlqin2yrs"], "y":"Count"})
    fig.update_layout(margin=dict(l=10,r=10,t=8,b=10), height=320)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('<div class="note">0 = no serious delinquency within two years; 1 = serious delinquency occurred.</div>', unsafe_allow_html=True)

    section_title("Main features")
    show_cols = [c for c in [
        "age","MonthlyIncome","DebtRatio","RevolvingUtilizationOfUnsecuredLines",
        "NumberOfOpenCreditLinesAndLoans","NumberOfTimes90DaysLate",
        "NumberOfTime60-89DaysPastDueNotWorse","NumberOfTime30-59DaysPastDueNotWorse",
        "NumberOfDependents","NumberRealEstateLoansOrLines"
    ] if c in df_full.columns]

    for c in show_cols:
        st.markdown(f"**{DISPLAY.get(c, c)}**")
        col1, col2 = st.columns([2,1])
        with col1:
            fig = px.histogram(df_full, x=c, nbins=60, labels={c: DISPLAY.get(c, c)})
            fig.update_layout(height=300, margin=dict(l=10,r=10,t=8,b=10))
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            v = px.violin(df_full, y=c, box=True, points=False, labels={c: DISPLAY.get(c, c)})
            v.update_layout(height=145, margin=dict(l=10,r=10,t=8,b=6))
            st.plotly_chart(v, use_container_width=True)
            s = df_full[c].dropna()
            if len(s)>0 and s.max() > 50 * max(1.0, s.median() if s.median()>0 else 1):
                hlog = px.histogram(np.log1p(s), nbins=60, labels={"value": f"log1p({DISPLAY.get(c,c)})"})
                hlog.update_layout(height=145, margin=dict(l=10,r=10,t=6,b=8))
                st.plotly_chart(hlog, use_container_width=True)
        st.markdown('<hr class="separator" />', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 3) Relationships & Segments
# -----------------------------------------------------------------------------
elif page == "Relationships & Segments":
    big_title("Relationships & Segments")

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

    section_title("Monthly Income vs Age")
    if {"MonthlyIncome","age"}.issubset(df_full.columns):
        t2 = df_full[["MonthlyIncome","age"]].dropna()
        s2 = px.scatter(t2, x="age", y="MonthlyIncome",
                        labels={"age": DISPLAY["age"], "MonthlyIncome": DISPLAY["MonthlyIncome"]},
                        opacity=0.4)
        s2.update_layout(height=420, margin=dict(l=10,r=10,t=8,b=10))
        st.plotly_chart(s2, use_container_width=True)

    st.markdown('<hr class="separator" />', unsafe_allow_html=True)

    section_title("Do defaulters have many loans and dependents?")
    if "NumberOfOpenCreditLinesAndLoans" in df_full.columns:
        loans = df_full["NumberOfOpenCreditLinesAndLoans"].fillna(0)
        loan_band = pd.cut(loans, bins=[-0.1,0,3,6,10, loans.max()+1], labels=["0","1–3","4–6","7–10",">10"])
        rate_by_loans = df_full.groupby(loan_band)["SeriousDlqin2yrs"].mean().reset_index()
        rate_by_loans.columns = ["Loan band","Default rate"]
        rate_by_loans["Default rate (%)"] = 100*rate_by_loans["Default rate"]
        figL = px.bar(rate_by_loans, x="Loan band", y="Default rate (%)")
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
    st.markdown('<div class="note">More open lines/loans and more dependents tend to show higher risk at the top bands.</div>', unsafe_allow_html=True)

    st.markdown('<hr class="separator" />', unsafe_allow_html=True)

    section_title("How common are serious past-due histories?")
    dl_cols = [c for c in ["NumberOfTime30-59DaysPastDueNotWorse","NumberOfTime60-89DaysPastDueNotWorse","NumberOfTimes90DaysLate"] if c in df_full.columns]
    if dl_cols:
        dd = df_full[dl_cols].fillna(0).sum(axis=1)  # total past-due events
        band = pd.cut(dd, bins=[-0.1,0,1,3,10, dd.max()+1], labels=["0","1","2–3","4–10",">10"])
        ct = band.value_counts().sort_index().reset_index()
        ct.columns = ["Past-due band","Count"]
        bfig = px.bar(ct, x="Past-due band", y="Count")
        bfig.update_layout(height=300, margin=dict(l=10,r=10,t=8,b=10))
        st.plotly_chart(bfig, use_container_width=True)
        st.markdown('<div class="note">“Serious” means repeat lates or longer delays. These bands are rarer but riskier.</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 4) Correlations & Outliers
# -----------------------------------------------------------------------------
elif page == "Correlations & Outliers":
    big_title("Correlations & Outliers")
    num_cols = [c for c in df_full.columns if pd.api.types.is_numeric_dtype(df_full[c])]

    if len(num_cols) >= 3:
        section_title("Correlation heatmap (numeric)")
        corr = df_full[num_cols].corr(numeric_only=True)
        heat = go.Figure(data=go.Heatmap(z=corr.values, x=[DISPLAY.get(x,x) for x in corr.columns],
                                         y=[DISPLAY.get(y,y) for y in corr.index], zmin=-1, zmax=1,
                                         colorbar=dict(title="ρ")))
        heat.update_layout(height=520, margin=dict(l=10,r=10,t=8,b=10))
        st.plotly_chart(heat, use_container_width=True)

    if "SeriousDlqin2yrs" in df_full.columns:
        section_title("Top features by correlation with the target")
        num_cols_target = [c for c in df_full.columns if c != "SeriousDlqin2yrs" and pd.api.types.is_numeric_dtype(df_full[c])]
        corr_t = (df_full[num_cols_target + ["SeriousDlqin2yrs"]]
                  .corr(numeric_only=True)["SeriousDlqin2yrs"]
                  .abs().drop("SeriousDlqin2yrs").sort_values(ascending=False).head(15))
        bar = px.bar(corr_t.reset_index(), x="index", y="SeriousDlqin2yrs",
                     labels={"index":"Feature", "SeriousDlqin2yrs":"|Correlation with target|"})
        bar.update_layout(height=360, margin=dict(l=10,r=10,t=8,b=10), xaxis_tickangle=-30)
        st.plotly_chart(bar, use_container_width=True)

    st.markdown('<hr class="separator" />', unsafe_allow_html=True)

    section_title("Outlier glance — boxplots")
    show = [c for c in ["RevolvingUtilizationOfUnsecuredLines","DebtRatio","MonthlyIncome"] if c in df_full.columns]
    cols = st.columns(len(show) if show else 1)
    if show:
        for i, c in enumerate(show):
            with cols[i]:
                fig = px.box(df_full, y=c, points=False, labels={c: DISPLAY.get(c,c)})
                fig.update_layout(height=320, margin=dict(l=10,r=10,t=8,b=10))
                st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------------
# 5) Interactive Lab
# -----------------------------------------------------------------------------
elif page == "Interactive Lab":
    big_title("Interactive Lab (Filters + Binned Rates + Threshold)")

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
    require_del = st.checkbox("Only rows with any past-due > 0", value=False)

    mask = pd.Series(True, index=df_full.index)
    if age_rng and "age" in df_full.columns: mask &= df_full["age"].between(age_rng[0], age_rng[1], inclusive="both")
    if income_rng and "MonthlyIncome" in df_full.columns:
        mi0, mi1 = income_rng; mask &= df_full["MonthlyIncome"].fillna(-1e12).between(mi0, mi1, inclusive="both")
    if require_del and del_cols: mask &= (df_full[del_cols].fillna(0) > 0).any(axis=1)
    df = df_full[mask].copy()

    section_title("Interactive — default rate by bins")
    choices = [c for c in ["age","MonthlyIncome","DebtRatio","RevolvingUtilizationOfUnsecuredLines"] if c in df.columns]
    if choices:
        colC1, colC2 = st.columns([2,1])
        feature = colC1.selectbox("Feature", options=[(DISPLAY.get(c,c), c) for c in choices], format_func=lambda x: x[0])[1]
        bins = colC2.slider("Bins", 4, 20, 8)
        figb = safe_rate_by_bin(df, feature, bins=bins)
        if figb: st.plotly_chart(figb, use_container_width=True)
    else:
        st.info("No numeric features available for interactive binning.")

    st.markdown('<hr class="separator" />', unsafe_allow_html=True)

    section_title("Quick model — threshold tuner (on filtered data)")
    if len(df) < 200 or df["SeriousDlqin2yrs"].nunique() < 2:
        st.info("Not enough filtered rows/classes to train a model. Loosen filters.")
    else:
        y = df["SeriousDlqin2yrs"].astype(int)
        X_cols_local = [c for c in df.columns if c != "SeriousDlqin2yrs" and pd.api.types.is_numeric_dtype(df[c])]
        X_df = df[X_cols_local].copy().fillna(df[X_cols_local].median(numeric_only=True))
        X_train_df, X_test_df, y_train, y_test = train_test_split(X_df, y, test_size=0.2, random_state=42, stratify=y)
        scaler = StandardScaler(); X_train = scaler.fit_transform(X_train_df); X_test = scaler.transform(X_test_df)
        lr = LogisticRegression(max_iter=500, solver="liblinear", class_weight="balanced"); lr.fit(X_train, y_train)
        y_proba = lr.predict_proba(X_test)[:,1]; auc = roc_auc_score(y_test, y_proba)
        ths = np.linspace(0.05, 0.95, 181)
        best_th, best_f1 = 0.50, -1
        for th in ths:
            pred = (y_proba >= th).astype(int)
            _, _, f1, _ = precision_recall_fscore_support(y_test, pred, average="binary", zero_division=0)
            if f1 > best_f1: best_f1, best_th = f1, th
        th = st.slider("Decision threshold", 0.10, 0.90, float(best_th), step=0.01); y_pred = (y_proba >= th).astype(int)
        acc = accuracy_score(y_test, y_pred); prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", zero_division=0)
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("AUC", f"{auc:.3f}"); c2.metric("Precision (1)", f"{prec:.3f}"); c3.metric("Recall (1)", f"{rec:.3f}"); c4.metric("F1 (1)", f"{f1:.3f}")

    st.markdown('<hr class="separator" />', unsafe_allow_html=True)
    section_title("Export current filtered data")
    csv_data = df.to_csv(index=False)
    st.download_button("Download filtered dataset (CSV)", data=csv_data, file_name="filtered_dataset.csv", mime="text/csv")

# -----------------------------------------------------------------------------
# 6) Modeling & Metrics
# -----------------------------------------------------------------------------
elif page == "Modeling & Metrics":
    big_title("Modeling & Metrics")

    y = df_full["SeriousDlqin2yrs"].astype(int)
    X_cols = [c for c in df_full.columns if c != "SeriousDlqin2yrs" and pd.api.types.is_numeric_dtype(df_full[c])]
    X_df = df_full[X_cols].copy().fillna(df_full[X_cols].median(numeric_only=True))

    TEST_SIZE = 0.20
    X_train_df, X_test_df, y_train, y_test = train_test_split(X_df, y, test_size=TEST_SIZE, random_state=42, stratify=y)
    scaler = StandardScaler(); X_train = scaler.fit_transform(X_train_df); X_test = scaler.transform(X_test_df)
    lr = LogisticRegression(max_iter=500, solver="liblinear", class_weight="balanced"); lr.fit(X_train, y_train)
    y_proba = lr.predict_proba(X_test)[:,1]; auc = roc_auc_score(y_test, y_proba)

    # store for other pages
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

    section_title("Pick a decision threshold (maximize F1)")
    ths = np.linspace(0.05, 0.95, 181)
    best_th, best_f1 = 0.50, -1
    for th in ths:
        pred = (y_proba >= th).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(y_test, pred, average="binary", zero_division=0)
        if f1 > best_f1: best_f1, best_th = f1, th
    st.metric("Chosen threshold", f"{best_th:.2f}")
    y_pred = (y_proba >= best_th).astype(int)

    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=[0,1])

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Accuracy", f"{acc:.3f}")
    c2.metric("Precision (1)", f"{prec:.3f}")
    c3.metric("Recall (1)", f"{rec:.3f}")
    c4.metric("F1 (1)", f"{f1:.3f}")

    section_title("Confusion matrix")
    cm_fig = go.Figure(data=go.Heatmap(z=cm, x=["Pred 0","Pred 1"], y=["True 0","True 1"], text=cm, texttemplate="%{text}", showscale=False))
    cm_fig.update_layout(height=320, margin=dict(l=10,r=10,t=8,b=10))
    st.plotly_chart(cm_fig, use_container_width=True)
    tn, fp, fn, tp = cm.ravel()
    st.markdown(
        f"<div class='note'>"
        f"**True 0 (TN):** {tn:,} non-defaulters correctly predicted. "
        f"**FP:** {fp:,} false alarms. "
        f"**FN:** {fn:,} missed defaulters. "
        f"**TP:** {tp:,} defaulters correctly flagged."
        f"</div>", unsafe_allow_html=True
    )

    coef = pd.Series(lr.coef_[0], index=X_cols).abs().sort_values(ascending=False)
    fi = coef.head(min(15, len(coef))).reset_index()
    fi.columns = ["feature","importance(|coef|)"]
    section_title("Top feature importance (|coef|)")
    fi_fig = px.bar(fi, x="feature", y="importance(|coef|)")
    fi_fig.update_layout(height=360, margin=dict(l=10,r=10,t=8,b=10), xaxis_tickangle=-30)
    st.plotly_chart(fi_fig, use_container_width=True)

    section_title("Permutation importance (test set)")
    sample_idx = np.random.RandomState(42).choice(len(X_test_df), size=min(2000, len(X_test_df)), replace=False)
    X_perm = X_test_df.iloc[sample_idx]
    y_perm = y_test.iloc[sample_idx]
    X_perm_sc = scaler.transform(X_perm)
    pi = permutation_importance(lr, X_perm_sc, y_perm, n_repeats=8, random_state=42, scoring="roc_auc")
    pi_df = pd.DataFrame({"feature": X_cols, "importance": pi.importances_mean}).sort_values("importance", ascending=False).head(15)
    pi_fig = px.bar(pi_df, x="feature", y="importance")
    pi_fig.update_layout(height=360, margin=dict(l=10,r=10,t=8,b=10), xaxis_tickangle=-30)
    st.plotly_chart(pi_fig, use_container_width=True)
    st.caption("Permutation importance = drop in AUC when shuffling a feature (higher = more influential).")

    section_title("Precision–Recall curve & Calibration")
    precs, recs, _thr = precision_recall_curve(y_test, y_proba)
    pr_auc = sk_auc(recs, precs)
    pr_fig = go.Figure(); pr_fig.add_trace(go.Scatter(x=recs, y=precs, mode="lines", name=f"PR (AUC={pr_auc:.3f})"))
    pr_fig.update_layout(xaxis_title="Recall", yaxis_title="Precision", height=320, margin=dict(l=10,r=10,t=40,b=10))
    st.plotly_chart(pr_fig, use_container_width=True)

    prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins=10, strategy="quantile")
    brier = brier_score_loss(y_test, y_proba)
    cal_fig = go.Figure()
    cal_fig.add_trace(go.Scatter(x=prob_pred, y=prob_true, mode="lines+markers", name="Calibration"))
    cal_fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Perfect", line=dict(dash="dash")))
    cal_fig.update_layout(xaxis_title="Predicted probability", yaxis_title="Observed default rate",
                          height=320, margin=dict(l=10,r=10,t=40,b=10))
    st.plotly_chart(cal_fig, use_container_width=True)
    st.caption(f"Brier score: **{brier:.4f}** (lower is better).")

# -----------------------------------------------------------------------------
# 7) Risk Buckets (A–D)
# -----------------------------------------------------------------------------
elif page == "Risk Buckets (A–D)":
    big_title("Risk Buckets (A–D) & Likely Defaulters")

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

# -----------------------------------------------------------------------------
# 8) Client Credit Check
# -----------------------------------------------------------------------------
elif page == "Client Credit Check":
    big_title("Client Credit Check — Enter borrower details")
    st.caption("Fill in the boxes below to get a GO / CAUTION / STOP verdict with a risk gauge.")

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

    def rng(col, qlo=0.10, qhi=0.90, fallback_max=100):
        if col not in df_full.columns or df_full[col].dropna().empty:
            return (0, fallback_max, int(fallback_max/2))
        s = df_full[col].dropna()
        lo = int(s.quantile(qlo)); hi = int(max(s.quantile(qhi), lo+1))
        return (int(max(0, s.min())), int(max(s.max(), hi)), int((lo+hi)//2))

    a_min,a_max,a_def = rng("age", 0.10, 0.90, 100)
    m_min,m_max,m_def = rng("MonthlyIncome", 0.10, 0.90, 50000)
    u_min,u_max,u_def = rng("RevolvingUtilizationOfUnsecuredLines", 0.10, 0.90, 200)
    d_min,d_max,d_def = rng("DebtRatio", 0.10, 0.90, 300)
    l_min,l_max,l_def = rng("NumberOfOpenCreditLinesAndLoans", 0.10, 0.90, 20)
    dep_min,dep_max,dep_def = rng("NumberOfDependents", 0.10, 0.90, 6)
    t30_max = int(df_full.get("NumberOfTime30-59DaysPastDueNotWorse", pd.Series([3])).max())
    t60_max = int(df_full.get("NumberOfTime60-89DaysPastDueNotWorse", pd.Series([2])).max())
    t90_max = int(df_full.get("NumberOfTimes90DaysLate", pd.Series([2])).max())

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age", min_value=a_min, max_value=a_max, value=a_def, step=1)
        income = st.number_input("Monthly Income", min_value=m_min, max_value=m_max, value=m_def, step=1)

        st.markdown("**Card/Line Utilization**")
        with st.expander("What is this? (tap to help)"):
            st.markdown("""
**Card/Line Utilization = Total revolving balances ÷ Total credit limits**

- Revolving balances: current balances on credit cards/lines of credit.  
- Credit limits: total available limits across those same cards/lines.  

**Example:** balances 60,000 and limits 200,000 ⇒ 60,000 / 200,000 = **0.30 (30%)**.  
Higher utilization (closer to 1.0 or 100%) usually means higher risk.
            """)
            rev_bal = st.number_input("Total revolving balances", min_value=0, value=60000, step=1000, key="rev_bal_calc")
            tot_lim = st.number_input("Total credit limits", min_value=1, value=200000, step=1000, key="tot_lim_calc")
            calc_util = round(rev_bal / tot_lim, 2)
            st.caption(f"Calculated Utilization: **{calc_util:.2f}**  (use this below if helpful)")

        util = st.number_input(
            "Enter Utilization (e.g., 0.30 for 30%)",
            min_value=0.0, max_value=5.0, value=float(calc_util), step=0.01,
            help="Utilization = total revolving balances ÷ total credit limits"
        )

    with col2:
        st.markdown("**Debt Ratio**")
        with st.expander("What is this? (tap to help)"):
            st.markdown("""
**Debt Ratio = Total monthly debt payments ÷ Gross monthly income**

- Monthly debt payments: mortgage/rent (per policy), car/student/personal loans, minimum card payments, other finance plans.  
- Gross income: before-tax monthly income.

**Example:** 12,000 / 40,000 = **0.30 (30%)**.
            """)
            m_pay = st.number_input("Monthly debt payments", min_value=0, value=12000, step=500, key="m_pay_calc")
            g_inc = st.number_input("Gross monthly income", min_value=1, value=40000, step=500, key="g_inc_calc")
            calc_dr = round(m_pay / g_inc, 2)
            st.caption(f"Calculated Debt Ratio: **{calc_dr:.2f}**  (use this below if helpful)")

        dratio = st.number_input(
            "Enter Debt Ratio (e.g., 0.30 for 30%)",
            min_value=0.0, max_value=max(d_max,1), value=float(calc_dr), step=0.01,
            help="Debt Ratio = total monthly debt payments ÷ gross monthly income"
        )

        open_lines = st.number_input("Open Credit Lines/Loans", min_value=l_min, max_value=l_max, value=l_def, step=1)
        dependents = st.number_input("Dependents", min_value=dep_min, max_value=max(dep_max,0), value=dep_def, step=1)

    with col3:
        late30 = st.number_input("Times 30–59 Days Late", min_value=0, max_value=max(1,t30_max), value=0, step=1)
        late60 = st.number_input("Times 60–89 Days Late", min_value=0, max_value=max(1,t60_max), value=0, step=1)
        late90 = st.number_input("Times 90+ Days Late", min_value=0, max_value=max(1,t90_max), value=0, step=1)

    med = df_full[model_cols].median(numeric_only=True)
    row = med.reindex(model_cols).astype(float)
    for k,v in {
        "age": age, "MonthlyIncome": income, "RevolvingUtilizationOfUnsecuredLines": util,
        "DebtRatio": dratio, "NumberOfOpenCreditLinesAndLoans": open_lines,
        "NumberOfDependents": dependents, "NumberOfTime30-59DaysPastDueNotWorse": late30,
        "NumberOfTime60-89DaysPastDueNotWorse": late60, "NumberOfTimes90DaysLate": late90
    }.items():
        if k in row.index: row[k] = float(v)

    X = row.values.reshape(1,-1)
    Xs = (X - scaler_mean) / np.where(scaler_scale==0, 1.0, scaler_scale)
    logit = float(np.dot(Xs, coef) + intercept)
    prob = float(1/(1+np.exp(-logit)))

    st.caption("Verdict thresholds (adjust to policy)")
    cA, cB = st.columns(2)
    green_cut = cA.slider("GO if probability < ", 0.02, 0.50, 0.20, step=0.01)
    red_cut   = cB.slider("STOP if probability ≥", green_cut, 0.95, 0.50, step=0.01)

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
        value=prob*100,
        number={'suffix': "%"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': gauge_color},
            'steps': [
                {'range': [0, green_cut*100], 'color': '#d1fae5'},
                {'range': [green_cut*100, red_cut*100], 'color': '#fef3c7'},
                {'range': [red_cut*100, 100], 'color': '#fee2e2'},
            ]
        }
    ))
    gauge.update_layout(height=260, margin=dict(l=10,r=10,t=10,b=10))
    st.plotly_chart(gauge, use_container_width=True)

# -----------------------------------------------------------------------------
# 9) Saved Figures
# -----------------------------------------------------------------------------
elif page == "Saved Figures":
    from PIL import Image, UnidentifiedImageError
    big_title("Saved Figures")

    def load_image_safe(path: str):
        try:
            img = Image.open(path); img.load()
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")
            img.thumbnail((1600, 1600))
            return img, None
        except (UnidentifiedImageError, OSError) as e:
            return None, f"Unrecognized or unreadable image: {e}"
        except Exception as e:
            return None, str(e)

    patterns = ["reports/figures/*.png", "reports/figures/*.jpg", "reports/figures/*.jpeg"]
    files = []
    for pat in patterns: files.extend(sorted(glob.glob(pat)))

    if not files:
        st.markdown('<div class="block note">No images found in reports/figures.</div>', unsafe_allow_html=True)
    else:
        section_title("reports/figures")
        cols = st.columns(3)
        for i, p in enumerate(files):
            img, err = load_image_safe(p)
            with cols[i % 3]:
                if img is not None:
                    st.image(img, use_column_width=True, caption=os.path.basename(p))
                    try:
                        with open(p, "rb") as f:
                            st.download_button(
                                label="Download",
                                data=f.read(),
                                file_name=os.path.basename(p),
                                mime="image/png" if p.lower().endswith(".png") else "image/jpeg",
                                key=f"dl_{i}_{os.path.basename(p)}",
                            )
                    except Exception:
                        pass
                else:
                    st.markdown(f"- Could not display: `{p}` — {err or 'unknown error'}")
                    try:
                        with open(p, "rb") as f:
                            st.download_button(
                                label=f"Download {os.path.basename(p)}",
                                data=f.read(),
                                file_name=os.path.basename(p),
                                mime="application/octet-stream",
                                key=f"dl_err_{i}_{os.path.basename(p)}",
                            )
                    except Exception:
                        pass

# -----------------------------------------------------------------------------
# 10) Data Quality
# -----------------------------------------------------------------------------
elif page == "Data Quality":
    big_title("Data Quality")
    section_title("Missing values by column")
    miss = df_full.isna().sum().sort_values(ascending=False); miss_df = miss[miss>0].reset_index()
    miss_df.columns = ["Column","Missing"]
    if not miss_df.empty:
        fig = px.bar(miss_df, x="Column", y="Missing")
        fig.update_layout(height=320, margin=dict(l=10,r=10,t=8,b=10)); st.plotly_chart(fig, use_container_width=True)
    else:
        st.markdown('<div class="block note">No missing values.</div>', unsafe_allow_html=True)
    section_title("Basic stats")
    show_df = df_full.copy()
    if "age" in show_df.columns: show_df["age"] = show_df["age"].astype("Int64")
    if "MonthlyIncome" in show_df.columns: show_df["MonthlyIncome"] = show_df["MonthlyIncome"].astype("Int64")
    st.dataframe(show_df.describe(include="all").transpose(), use_container_width=True)

# -----------------------------------------------------------------------------
# 11) Summary & Conclusion
# -----------------------------------------------------------------------------
elif page == "Summary & Conclusion":
    big_title("Summary & Conclusion")
    st.markdown("Borrowers most likely to default tend to combine **very high card/line utilization**, **any past-due history**, **very high debt ratio**, **lower income**, **younger age**, **many open lines/loans**, and **3+ dependents**.")

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

    colA, colD = st.columns(2)
    with colA:
        section_title("Examples — good clients (Bucket A)")
        st.dataframe(
            scored[scored["bucket"]=="A"]
            .sort_values("prob_default", ascending=True)
            .head(10)[["prob_default"]+extras]
            .assign(**{"prob_default": lambda d: (100*d["prob_default"]).round(1)} )
            .rename(columns={"prob_default":"Pred default (%)"}),
            use_container_width=True
        )
    with colD:
        section_title("Examples — risky clients (Bucket D)")
        st.dataframe(
            scored[scored["bucket"]=="D"]
            .sort_values("prob_default", ascending=False)
            .head(10)[["prob_default"]+extras]
            .assign(**{"prob_default": lambda d: (100*d["prob_default"]).round(1)} )
            .rename(columns={"prob_default":"Pred default (%)"}),
            use_container_width=True
        )

    section_title("How to use this")
    st.markdown(
        "- **Bucket A**: generally safe; standard terms.\n"
        "- **Bucket B**: near-safe; modest limits.\n"
        "- **Bucket C**: caution; tighter limits or extra checks.\n"
        "- **Bucket D**: high risk; small loans only if needed, or decline.\n"
        "For a single case, use **Client Credit Check** (left menu) for a quick GO / CAUTION / STOP verdict."
    )
