# dashboard.py — stable build (no pink errors)
# - Safe, bytes-based image loading
# - Guarded pages (won’t crash if data/image missing)
# - Simplified charts (no numpy histogram pitfalls)

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

# -------------------- CONFIG --------------------
st.set_page_config(
    page_title="Loan Default — Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_DIR = Path(__file__).parent
ASSETS_DIR = APP_DIR / "assets"
DATA_PATH = APP_DIR / "cs-training.csv"                 # put your CSV next to this file
HERO_PATH = ASSETS_DIR / "credit_risk_hero.JPG"         # exact filename & case

# -------------------- HELPERS --------------------
@st.cache_data(show_spinner=False)
def safe_read_csv(path: Path) -> Optional[pd.DataFrame]:
    try:
        if path.exists():
            return pd.read_csv(path)
    except Exception as e:
        st.warning(f"Could not read data file '{path.name}': {e}")
    return None

def load_image_bytes(path: Path) -> Optional[bytes]:
    try:
        if path.exists():
            return path.read_bytes()
    except Exception as e:
        st.warning(f"Could not read image '{path.name}': {e}")
    return None

def section_header(title: str, subtitle: str | None = None):
    st.markdown(
        f"""
        <div style="background:#0f766e0d;border:1px solid #0f766e22;padding:14px 18px;border-radius:12px;margin:4px 0 16px 0;">
            <div style="font-size:24px;font-weight:700;color:#0f766e;">{title}</div>
            {"<div style='opacity:.85;margin-top:6px'>" + subtitle + "</div>" if subtitle else ""}
        </div>
        """,
        unsafe_allow_html=True,
    )

# -------------------- SIDEBAR --------------------
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

st.sidebar.caption("Data file:")
st.sidebar.code(DATA_PATH.name)

# Load data once (safe)
df = safe_read_csv(DATA_PATH)

# -------------------- PAGES --------------------
if choice == "Introduction":
    section_header(
        "Loan Default Risk — Executive Overview",
        "A streamlined view of the dataset, quality checks, and core findings.",
    )

    hero = load_image_bytes(HERO_PATH)
    if hero:
        st.image(hero, use_container_width=True)
    else:
        st.info(
            "Hero image not found. Place the file at "
            f"`assets/{HERO_PATH.name}` (remember Linux is case-sensitive)."
        )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Rows", f"{0 if df is None else len(df):,}")
    with col2:
        st.metric("Columns", f"{0 if df is None else df.shape[1]}")
    with col3:
        st.metric("Target", "SeriousDlqin2yrs")

    st.markdown("### Quick Preview")
    if df is not None:
        st.dataframe(df.head(), use_container_width=True)
    else:
        st.warning("No data loaded yet — add `cs-training.csv` to the app folder.")

elif choice == "Feature Distributions":
    section_header("Feature Distributions", "Histogram via value counts (stable).")
    if df is None:
        st.warning("Data not available.")
    else:
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        if not numeric_cols:
            st.info("No numeric columns found.")
        else:
            feature = st.selectbox("Feature", numeric_cols, index=0)
            bins = st.slider("Number of bins", 5, 60, 20)
            # Use pd.cut + value_counts (avoids numpy histogram shape/dtype issues)
            s = df[feature].dropna()
            binned = pd.cut(s, bins=bins)
            counts = binned.value_counts().sort_index()
            st.bar_chart(counts)

elif choice == "Relationships & Segments":
    section_header("Relationships & Segments", "Simple scatter (numeric only).")
    if df is None:
        st.warning("Data not available.")
    else:
        num_cols = df.select_dtypes(include=["number"]).columns.tolist()
        if len(num_cols) < 2:
            st.info("Need at least two numeric columns.")
        else:
            x = st.selectbox("X", num_cols, index=0, key="x_rel")
            y = st.selectbox("Y", num_cols, index=1 if len(num_cols) > 1 else 0, key="y_rel")
            st.scatter_chart(df[[x, y]].dropna())

elif choice == "Correlations & Outliers":
    section_header("Correlations & Outliers", "Pearson correlation matrix.")
    if df is None:
        st.warning("Data not available.")
    else:
        num_df = df.select_dtypes(include=["number"])
        if num_df.empty:
            st.info("No numeric columns to correlate.")
        else:
            # numeric_only=True is safe on modern pandas; fallback guard added
            try:
                corr = num_df.corr(numeric_only=True)
            except TypeError:
                corr = num_df.corr()
            st.dataframe(corr, use_container_width=True)

elif choice == "Interactive Lab":
    section_header("Interactive Lab")
    st.info("Add your experiments here. Page intentionally light to guarantee stability.")

elif choice == "Modeling & Metrics":
    section_header("Modeling & Metrics")
    st.info("Hook your trained model here. Stubbed to avoid runtime errors.")

elif choice == "Risk Buckets (A–D)":
    section_header("Risk Buckets (A–D)")
    st.info("Provide your A–D bucket logic here. Placeholder for now.")

elif choice == "Client Credit Check":
    section_header("Client Credit Check")
    st.info("Add form inputs to run single-client checks. Placeholder for now.")

elif choice == "Saved Figures":
    section_header("Saved Figures")
    st.info("Render exported charts/images here if present in a folder.")

elif choice == "Data Quality":
    section_header("Data Quality")
    if df is None:
        st.warning("Data not available.")
    else:
        st.write("**Null counts**")
        st.dataframe(df.isna().sum().to_frame("nulls"), use_container_width=True)

elif choice == "Summary & Conclusion":
    section_header("Summary & Conclusion")
    st.success("Stable build: menu intact, image loading hardened, pages guarded.")

# -------------------- FOOTER --------------------
st.caption("© Loan Default Exam Project — Streamlit build (stable).")
