# dashboard.py — clean intro, centered hero, sidebar diagnostics, stable pages + gauge

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Optional, Callable

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from PIL import Image

# ---------- App setup ----------
st.set_page_config(page_title="Loan Default — Dashboard", layout="wide", initial_sidebar_state="expanded")

APP_DIR = Path(__file__).parent
ASSETS_DIR = APP_DIR / "assets"
DATA_PATH = APP_DIR / "cs-training.csv"

# Try a few filenames to avoid case/extension trouble on Linux
HERO_CANDIDATES = [
    ASSETS_DIR / "credit_risk_hero.jpg",
    ASSETS_DIR / "credit_risk_hero.JPG",
    ASSETS_DIR / "credit_risk_hero.jpeg",
    ASSETS_DIR / "credit_risk_hero.png",
    ASSETS_DIR / "credit_risk_hero.PNG",
]
HERO_MAX_WIDTH = 900  # pixels (safe—older Streamlit supports width=)

# ---------- Helpers ----------
@st.cache_data(show_spinner=False)
def safe_read_csv(path: Path) -> Optional[pd.DataFrame]:
    try:
        if path.exists():
            return pd.read_csv(path)
    except Exception as e:
        st.warning(f"Could not read '{path.name}': {e}")
    return None

def load_hero_image() -> Optional[Image.Image]:
    for p in HERO_CANDIDATES:
        try:
            if p.exists():
                img = Image.open(BytesIO(p.read_bytes()))
                if img.mode != "RGB":
                    img = img.convert("RGB")
                return img
        except Exception:
            continue
    return None

def section_header(title: str, subtitle: str | None = None):
    st.markdown(
        f"""
        <div style="background:#0f766e0d;border:1px solid #0f766e22;
                    padding:16px 18px;border-radius:12px;margin:4px 0 18px 0;">
            <div style="font-size:26px;font-weight:800;color:#0f766e;">{title}</div>
            {"<div style='opacity:.85;margin-top:6px'>" + subtitle + "</div>" if subtitle else ""}
        </div>
        """,
        unsafe_allow_html=True,
    )

def run_safe(block: Callable[[], None], where: str):
    try:
        block()
    except Exception as e:
        st.error(f"Something went wrong in **{where}**.")
        st.exception(e)

def gauge(prob: float) -> go.Figure:
    pct = float(np.clip(prob, 0.0, 1.0)) * 100
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=pct,
            number={"suffix": "%"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"thickness": 0.25},
                "steps": [
                    {"range": [0, 25], "color": "#ecfdf5"},   # A
                    {"range": [25, 50], "color": "#d1fae5"},  # B
                    {"range": [50, 75], "color": "#fee2e2"},  # C
                    {"range": [75, 100], "color": "#fecaca"}  # D
                ],
            },
        )
    )
    fig.update_layout(margin=dict(l=10, r=10, t=20, b=10), height=260)
    return fig

def bucket_from_prob(prob: float) -> str:
    p = float(np.clip(prob, 0.0, 1.0))
    if p < 0.25: return "A (Low)"
    if p < 0.50: return "B (Moderate)"
    if p < 0.75: return "C (Elevated)"
    return "D (High)"

# ---------- Sidebar ----------
st.sidebar.title("Navigation")
sections = [
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
    "Summary & Conclusion",
]
choice = st.sidebar.radio("Go to", sections, index=0)

# Move diagnostics to the sidebar so it’s out of the way
diag = st.sidebar.expander("Diagnostics", expanded=False)
with diag:
    df_probe = safe_read_csv(DATA_PATH)  # light call; cached
    st.write({
        "data_exists": DATA_PATH.exists(),
        "rows": None if df_probe is None else len(df_probe),
        "cols": None if df_probe is None else df_probe.shape[1],
        "hero_found": load_hero_image() is not None,
    })
st.sidebar.caption("Files (Linux is case-sensitive):")
st.sidebar.code(f"{DATA_PATH.name}\nassets/credit_risk_hero.(jpg/png)")

# Load data once for pages
df = df_probe

# ---------- Pages ----------
if choice == "Introduction":
    def _page():
        section_header(
            "Loan Default Risk — Executive Overview",
            "A quick look at the data, a short quality check, and how we’ll talk about risk.",
        )

        # Center the hero image with a simple column trick and a safe width
        img = load_hero_image()
        if img is not None:
            left, mid, right = st.columns([1, 3, 1])
            with mid:
                w = min(HERO_MAX_WIDTH, getattr(img, "width", HERO_MAX_WIDTH))
                st.image(img, width=w)
        else:
            st.info("Hero image not found. Put `credit_risk_hero.jpg` (or .png) into **assets/**.")

        # Clean, consistent metrics row
        m1, m2, m3 = st.columns(3)
        with m1: st.metric("Rows", f"{0 if df is None else len(df):,}")
        with m2: st.metric("Columns", f"{0 if df is None else df.shape[1]}")
        with m3: st.metric("Target", "SeriousDlqin2yrs")

        st.subheader("Peek at the data")
        if df is not None:
            st.dataframe(df.head(10), use_container_width=True)
        else:
            st.warning("No data yet — place `cs-training.csv` next to this file.")
    run_safe(_page, "Introduction")

elif choice == "Feature Distributions":
    def _page():
        section_header("Feature Distributions", "Simple histograms using binned counts.")
        if df is None:
            st.warning("Data not available."); return
        num_cols = df.select_dtypes(include=["number"]).columns.tolist()
        if not num_cols:
            st.info("No numeric columns found."); return
        feature = st.selectbox("Feature", num_cols, index=0)
        bins = st.slider("Bins", 5, 60, 20)
        s = df[feature].dropna()
        counts = pd.cut(s, bins=bins).value_counts().sort_index()
        st.bar_chart(counts)
    run_safe(_page, "Feature Distributions")

elif choice == "Relationships & Segments":
    def _page():
        section_header("Relationships & Segments", "Quick scatter between two numeric fields.")
        if df is None:
            st.warning("Data not available."); return
        num = df.select_dtypes(include=["number"]).columns.tolist()
        if len(num) < 2:
            st.info("Need at least two numeric columns."); return
        c1, c2 = st.columns(2)
        with c1: x = st.selectbox("X", num, index=0, key="x_rel")
        with c2: y = st.selectbox("Y", num, index=1 if len(num) > 1 else 0, key="y_rel")
        st.scatter_chart(df[[x, y]].dropna())
    run_safe(_page, "Relationships & Segments")

elif choice == "Correlations & Outliers":
    def _page():
        section_header("Correlations & Outliers", "Pearson correlation matrix.")
        if df is None:
            st.warning("Data not available."); return
        num_df = df.select_dtypes(include=["number"])
        if num_df.empty:
            st.info("No numeric columns to correlate."); return
        try:
            corr = num_df.corr(numeric_only=True)
        except TypeError:
            corr = num_df.corr()
        st.dataframe(corr, use_container_width=True)
    run_safe(_page, "Correlations & Outliers")

elif choice == "Interactive Lab":
    run_safe(lambda: (section_header("Interactive Lab"),
                      st.info("Scratch space for quick tests.")),
             "Interactive Lab")

elif choice == "Modeling & Metrics":
    run_safe(lambda: (section_header("Modeling & Metrics"),
                      st.info("Drop in your trained model later. Page kept light to stay reliable.")),
             "Modeling & Metrics")

elif choice == "Risk Buckets (A–D)":
    run_safe(lambda: (section_header("Risk Buckets (A–D)"),
                      st.info("Describe or tune your cut-offs here. Placeholder for now.")),
             "Risk Buckets (A–D)")

elif choice == "Client Credit Check":
    def _page():
        section_header("Client Credit Check", "Quick calculator with a gauge and A–D label.")

        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("Age", 18, 120, 35, 1)
            debt_ratio = st.number_input("DebtRatio (0–5)", 0.0, 5.0, 0.30, 0.01)
            monthly_income = st.number_input("MonthlyIncome", 0, 1_000_000, 5000, 100)
        with col2:
            late_30 = st.number_input("30–59 Days Late (2y)", 0, 50, 0, 1)
            late_60 = st.number_input("60–89 Days Late (2y)", 0, 50, 0, 1)
            late_90 = st.number_input("90+ Days Late (ever)", 0, 50, 0, 1)
        with col3:
            open_loans = st.number_input("Open Credit Lines/Loans", 0, 99, 6, 1)
            utilization = st.number_input("Revolving Utilization (0–5)", 0.0, 5.0, 0.25, 0.01)
            dependents = st.number_input("Dependents", 0, 20, 0, 1)

        def clip01(x): return float(np.clip(x, 0.0, 1.0))
        parts = {
            "age": clip01((50 - age) / 50.0),
            "debt_ratio": clip01(debt_ratio / 1.0),
            "income": clip01(1.0 - (monthly_income / 15000.0)),
            "late30": clip01(late_30 / 5.0),
            "late60": clip01(late_60 / 3.0),
            "late90": clip01(late_90 / 2.0),
            "open": clip01(abs(open_loans - 7) / 14.0),
            "util": clip01(utilization / 1.0),
            "deps": clip01(dependents / 6.0),
        }
        weights = {"age":0.05,"debt_ratio":0.15,"income":0.10,"late30":0.15,
                   "late60":0.15,"late90":0.20,"open":0.05,"util":0.10,"deps":0.05}
        prob = float(np.clip(sum(parts[k]*weights[k] for k in weights), 0.0, 1.0))
        bucket = bucket_from_prob(prob)

        gcol, tcol = st.columns([1.2, 1])
        with gcol: st.plotly_chart(gauge(prob), use_container_width=True)
        with tcol:
            st.subheader("Result")
            st.metric("Estimated default probability", f"{prob*100:.1f}%")
            st.metric("Bucket", bucket)
            st.caption("What pushed the score:")
            st.json({k: round(v, 3) for k, v in parts.items()})
        st.info("Lightweight heuristic for demo purposes. Swap in a trained model later if required.")
    run_safe(_page, "Client Credit Check")

elif choice == "Saved Figures":
    run_safe(lambda: (section_header("Saved Figures"),
                      st.info("Drop any exported charts here and render them.")),
             "Saved Figures")

elif choice == "Data Quality":
    def _page():
        section_header("Data Quality")
        if df is None:
            st.warning("Data not available."); return
        st.subheader("Null counts")
        st.dataframe(df.isna().sum().to_frame("nulls"), use_container_width=True)
    run_safe(_page, "Data Quality")

elif choice == "Summary & Conclusion":
    run_safe(lambda: (section_header("Summary & Conclusion"),
                      st.success("Clean intro, centered image, sidebar diagnostics, and the calculator is back.")),
             "Summary & Conclusion")

st.caption("© Loan Default Exam Project — Streamlit build (stable).")
