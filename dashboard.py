# dashboard.py — stable, all-in-one exam build
# - Bytes->PIL safe image loading (fixes st.image TypeError on cloud)
# - Guarded pages so missing files never crash the app
# - Restored "Client Credit Check" with gauge + A–D bucket
# - Lightweight visuals (no exotic deps)

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Optional, Callable

import numpy as np
import pandas as pd
import streamlit as st

# Plotly is bundled on Streamlit Cloud and safe to use
import plotly.graph_objects as go
from PIL import Image

# ================== CONFIG ==================
st.set_page_config(
    page_title="Loan Default — Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_DIR = Path(__file__).parent
ASSETS_DIR = APP_DIR / "assets"
DATA_PATH = APP_DIR / "cs-training.csv"          # put your CSV next to this file

# We will try multiple hero filenames so case/extension won't break deploys
HERO_CANDIDATES = [
    ASSETS_DIR / "credit_risk_hero.jpg",
    ASSETS_DIR / "credit_risk_hero.JPG",
    ASSETS_DIR / "credit_risk_hero.jpeg",
    ASSETS_DIR / "credit_risk_hero.PNG",
    ASSETS_DIR / "credit_risk_hero.png",
]


# ================== HELPERS ==================
@st.cache_data(show_spinner=False)
def safe_read_csv(path: Path) -> Optional[pd.DataFrame]:
    """Return DataFrame or None; never throws."""
    try:
        if path.exists():
            return pd.read_csv(path)
    except Exception as e:
        st.warning(f"Could not read data file '{path.name}': {e}")
    return None


def load_hero_image() -> Optional[Image.Image]:
    """
    Open the hero image robustly:
    - Try several candidate filenames (case/extension)
    - Decode bytes with PIL
    - Convert to RGB to avoid mode issues
    Returns PIL Image or None.
    """
    for p in HERO_CANDIDATES:
        try:
            if p.exists():
                data = p.read_bytes()
                img = Image.open(BytesIO(data))
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
                    padding:14px 18px;border-radius:12px;margin:4px 0 16px 0;">
            <div style="font-size:26px;font-weight:800;color:#0f766e;">{title}</div>
            {"<div style='opacity:.85;margin-top:6px'>" + subtitle + "</div>" if subtitle else ""}
        </div>
        """,
        unsafe_allow_html=True,
    )


def run_safe(block: Callable[[], None], where: str):
    """Show a friendly, in-page error instead of a pink app crash."""
    try:
        block()
    except Exception as e:
        st.error(f"Something went wrong in **{where}**.")
        st.exception(e)


def gauge(prob: float) -> go.Figure:
    """Plotly gauge from 0–100% with risk zones."""
    pct = float(np.clip(prob, 0.0, 1.0)) * 100.0
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=pct,
            number={"suffix": "%"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"thickness": 0.25},
                "steps": [
                    {"range": [0, 25], "color": "#ecfdf5"},    # A (Low)
                    {"range": [25, 50], "color": "#d1fae5"},   # B
                    {"range": [50, 75], "color": "#fee2e2"},   # C
                    {"range": [75, 100], "color": "#fecaca"},  # D (High)
                ],
            },
            domain={"x": [0, 1], "y": [0, 1]},
        )
    )
    fig.update_layout(margin=dict(l=10, r=10, t=20, b=10), height=260)
    return fig


def bucket_from_prob(prob: float) -> str:
    """A–D buckets from probability."""
    p = float(np.clip(prob, 0.0, 1.0))
    if p < 0.25:
        return "A (Low)"
    if p < 0.50:
        return "B (Moderate)"
    if p < 0.75:
        return "C (Elevated)"
    return "D (High)"


# ================== SIDEBAR ==================
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

st.sidebar.caption("Data & Assets (Linux is case-sensitive):")
st.sidebar.code(f"{DATA_PATH.name}\nassets/credit_risk_hero.(jpg/png)")

# Load data once (safe)
df = safe_read_csv(DATA_PATH)

with st.expander("Diagnostics"):
    st.write(
        {
            "data_exists": DATA_PATH.exists(),
            "rows": None if df is None else len(df),
            "cols": None if df is None else df.shape[1],
            "hero_found": load_hero_image() is not None,
        }
    )


# ================== PAGES ==================
if choice == "Introduction":
    def _page():
        section_header(
            "Loan Default Risk — Executive Overview",
            "A streamlined view of the dataset, quality checks, and core findings.",
        )

        img = load_hero_image()
        if img is not None:
            st.image(img, use_container_width=True)
        else:
            st.info(
                "Hero image not found. Place `credit_risk_hero.jpg` (or .png) inside **assets/**."
            )

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Rows", f"{0 if df is None else len(df):,}")
        with c2:
            st.metric("Columns", f"{0 if df is None else df.shape[1]}")
        with c3:
            st.metric("Target", "SeriousDlqin2yrs")

        st.subheader("Quick Preview")
        if df is not None:
            st.dataframe(df.head(10), use_container_width=True)
        else:
            st.warning("No data loaded yet — add `cs-training.csv` next to this file.")
    run_safe(_page, "Introduction")


elif choice == "Feature Distributions":
    def _page():
        section_header("Feature Distributions", "Robust histogram via `pd.cut` + counts.")
        if df is None:
            st.warning("Data not available.")
            return
        num_cols = df.select_dtypes(include=["number"]).columns.tolist()
        if not num_cols:
            st.info("No numeric columns found.")
            return
        feature = st.selectbox("Feature", num_cols, index=0)
        bins = st.slider("Number of bins", 5, 60, 20)
        s = df[feature].dropna()
        counts = pd.cut(s, bins=bins).value_counts().sort_index()
        st.bar_chart(counts)
    run_safe(_page, "Feature Distributions")


elif choice == "Relationships & Segments":
    def _page():
        section_header("Relationships & Segments", "Simple scatter (numeric only).")
        if df is None:
            st.warning("Data not available.")
            return
        num_cols = df.select_dtypes(include=["number"]).columns.tolist()
        if len(num_cols) < 2:
            st.info("Need at least two numeric columns.")
            return
        c1, c2 = st.columns(2)
        with c1:
            x = st.selectbox("X", num_cols, index=0, key="x_rel")
        with c2:
            y = st.selectbox("Y", num_cols, index=1 if len(num_cols) > 1 else 0, key="y_rel")
        st.scatter_chart(df[[x, y]].dropna())
    run_safe(_page, "Relationships & Segments")


elif choice == "Correlations & Outliers":
    def _page():
        section_header("Correlations & Outliers", "Pearson correlation matrix.")
        if df is None:
            st.warning("Data not available.")
            return
        num_df = df.select_dtypes(include=["number"])
        if num_df.empty:
            st.info("No numeric columns to correlate.")
            return
        try:
            corr = num_df.corr(numeric_only=True)
        except TypeError:
            corr = num_df.corr()
        st.dataframe(corr, use_container_width=True)
    run_safe(_page, "Correlations & Outliers")


elif choice == "Interactive Lab":
    def _page():
        section_header("Interactive Lab")
        st.info("Sandbox area for experiments (kept light for reliability).")
    run_safe(_page, "Interactive Lab")


elif choice == "Modeling & Metrics":
    def _page():
        section_header("Modeling & Metrics")
        st.info("Hook your trained model here (placeholder to remain stable).")
    run_safe(_page, "Modeling & Metrics")


elif choice == "Risk Buckets (A–D)":
    def _page():
        section_header("Risk Buckets (A–D)")
        st.info("Add your A–D bucket logic or thresholds here (placeholder).")
    run_safe(_page, "Risk Buckets (A–D)")


elif choice == "Client Credit Check":
    def _page():
        section_header("Client Credit Check", "Quick calculator with gauge and A–D bucket.")

        # Inputs similar to cs-training.csv features (names kept simple)
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("Age", min_value=18, max_value=120, value=35, step=1)
            debt_ratio = st.number_input("DebtRatio (0–5)", min_value=0.0, max_value=5.0, value=0.3, step=0.01)
            monthly_income = st.number_input("MonthlyIncome", min_value=0, max_value=1_000_000, value=5000, step=100)
        with col2:
            late_30 = st.number_input("30–59 Days Late (last 2y)", min_value=0, max_value=50, value=0, step=1)
            late_60 = st.number_input("60–89 Days Late (last 2y)", min_value=0, max_value=50, value=0, step=1)
            late_90 = st.number_input("90+ Days Late (ever)", min_value=0, max_value=50, value=0, step=1)
        with col3:
            open_loans = st.number_input("NumberOfOpenCreditLinesAndLoans", min_value=0, max_value=99, value=6, step=1)
            utilization = st.number_input("RevolvingUtilizationOfUnsecuredLines (0–5)", min_value=0.0, max_value=5.0, value=0.25, step=0.01)
            dependents = st.number_input("NumberOfDependents", min_value=0, max_value=20, value=0, step=1)

        # Stable, transparent scoring (not a trained model; avoids dependency/risk)
        # Normalize into 0..1 contributors; higher -> higher risk
        def clip01(x): return float(np.clip(x, 0.0, 1.0))

        # Heuristics inspired by dataset semantics
        risk_parts = {
            "age": clip01((50 - age) / 50.0),  # younger slightly riskier
            "debt_ratio": clip01(debt_ratio / 1.0),  # >1 is very high; clipped later
            "income": clip01(1.0 - (monthly_income / 15000.0)),  # higher income -> lower risk
            "lates_30": clip01(late_30 / 5.0),
            "lates_60": clip01(late_60 / 3.0),
            "lates_90": clip01(late_90 / 2.0),
            "open_loans": clip01(abs(open_loans - 7) / 14.0),  # too few or too many slightly worse
            "util": clip01(utilization / 1.0),
            "deps": clip01(dependents / 6.0),
        }

        # Weighted sum -> probability
        weights = {
            "age": 0.05,
            "debt_ratio": 0.15,
            "income": 0.10,
            "lates_30": 0.15,
            "lates_60": 0.15,
            "lates_90": 0.20,
            "open_loans": 0.05,
            "util": 0.10,
            "deps": 0.05,
        }
        score = sum(risk_parts[k] * weights[k] for k in weights)  # 0..1
        prob = float(np.clip(score, 0.0, 1.0))
        bucket = bucket_from_prob(prob)

        gcol, tcol = st.columns([1.2, 1])
        with gcol:
            st.plotly_chart(gauge(prob), use_container_width=True)
        with tcol:
            st.subheader("Risk Result")
            st.metric("Estimated default probability", f"{prob*100:.1f}%")
            st.metric("Bucket", bucket)

            st.caption("Breakdown (higher values = more risk):")
            st.json({k: round(v, 3) for k, v in risk_parts.items()})

        st.info(
            "This is a simple heuristic scorer (exam-safe). Replace with your trained model later if needed."
        )
    run_safe(_page, "Client Credit Check")


elif choice == "Saved Figures":
    def _page():
        section_header("Saved Figures")
        st.info("Render exported charts/images here if present in a folder.")
    run_safe(_page, "Saved Figures")


elif choice == "Data Quality":
    def _page():
        section_header("Data Quality")
        if df is None:
            st.warning("Data not available.")
            return
        st.subheader("Null counts")
        nulls = df.isna().sum().to_frame("nulls")
        st.dataframe(nulls, use_container_width=True)
    run_safe(_page, "Data Quality")


elif choice == "Summary & Conclusion":
    def _page():
        section_header("Summary & Conclusion")
        st.success("Stable build: menu intact, hero image hardened, calculator restored.")
    run_safe(_page, "Summary & Conclusion")


# ================== FOOTER ==================
st.caption("© Loan Default Exam Project — Streamlit build (stable).")
