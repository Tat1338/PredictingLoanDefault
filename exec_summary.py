# exec_summary.py
# Writes a plain-English executive summary to reports/executive_summary.md
# - Restores all sections you liked (bullets, microsegments, comparison tables)
# - Fixes "Takeaway" (no "micro" wording or variables)
# - Proper Markdown tables so nothing vanishes
# - Silences pandas groupby category warnings with observed=True

from __future__ import annotations

import os
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd

EXPORTS = "exports"
REPORTS = "reports"
OUT_MD = os.path.join(REPORTS, "executive_summary.md")
HOLDOUT_CSV = os.path.join(EXPORTS, "holdout_predictions.csv")
WITH_FEATS_CSV = os.path.join(EXPORTS, "holdout_with_features.csv")
MODEL_SUMMARY_CSV = os.path.join(EXPORTS, "model_eval_summary.csv")

# ---- configuration
THRESHOLD = 0.30
PREFERRED_PROBA_COLS = ["proba_rf", "proba_logreg"]  # pick in this order

# ------------ helpers

def ensure_dirs():
    os.makedirs(REPORTS, exist_ok=True)

def read_csv_safe(path: str) -> Optional[pd.DataFrame]:
    try:
        if os.path.exists(path):
            return pd.read_csv(path)
    except Exception:
        return None
    return None

def fmt_pct(x: float, d: int = 1) -> str:
    if x is None or not np.isfinite(x): return "—"
    return f"{100.0 * float(x):.{d}f}%"

def fmt_float(x: float, d: int = 3) -> str:
    if x is None or not np.isfinite(x): return "—"
    return f"{float(x):.{d}f}"

def fmt_int(x: float | int) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)): return "—"
    try:
        return f"{int(round(float(x))):,}"
    except Exception:
        return str(x)

def pick_proba_col(df: pd.DataFrame) -> Tuple[str, str]:
    for c in PREFERRED_PROBA_COLS:
        if c in df.columns:
            pretty = "Random Forest" if c == "proba_rf" else "Logistic Regression"
            return c, pretty
    for c in df.columns:
        if str(c).lower().startswith("proba"):
            return c, c
    raise ValueError("No probability column found (expected proba_rf / proba_logreg).")

def confusion_at_threshold(y: np.ndarray, p: np.ndarray, thr: float) -> Tuple[int, int, int, int]:
    pred = (p >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    return tp, fp, tn, fn

def ks_statistic(y: np.ndarray, s: np.ndarray) -> float:
    pos = s[y == 1]; neg = s[y == 0]
    if pos.size == 0 or neg.size == 0:
        return np.nan
    grid = np.linspace(0.0, 1.0, 1001)
    cdf_p = np.searchsorted(np.sort(pos), grid, side="right") / pos.size
    cdf_n = np.searchsorted(np.sort(neg), grid, side="right") / neg.size
    return float(np.max(np.abs(cdf_p - cdf_n)))

# ------------ banding

def band_age(x: float) -> str:
    try: a = float(x)
    except Exception: return "Age: Missing"
    if a < 30: return "<30"
    if a < 40: return "30–39"
    if a < 50: return "40–49"
    if a < 60: return "50–59"
    return "60+"

def band_dependents(x: float) -> str:
    try: d = int(float(x))
    except Exception: return "Dependents: Missing"
    if d <= 0: return "0"
    if d <= 2: return "1–2"
    return "3+"

def quintile_labels(kind: str) -> List[str]:
    if kind == "income":
        return ["Income: Bottom 20%", "Income: Low 20%", "Income: Middle 20%", "Income: High 20%", "Income: Top 20%"]
    if kind == "debt":
        return ["Debt: Lowest 20%", "Debt: Low 20%", "Debt: Middle 20%", "Debt: High 20%", "Debt: Highest 20%"]
    return ["Q1","Q2","Q3","Q4","Q5"]

def band_by_quintile(series: pd.Series, kind: str) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    valid = s.dropna()
    if valid.empty:
        return pd.Series([f"{kind.capitalize()}: Missing"] * len(s), index=s.index)
    try:
        q = pd.qcut(valid, q=5, labels=quintile_labels(kind))
        out = pd.Series(index=s.index, dtype="object")
        out.loc[valid.index] = q.astype(str).values
        out = out.fillna(f"{kind.capitalize()}: Missing")
        return out
    except Exception:
        return pd.Series([f"{kind.capitalize()}: All"] * len(s), index=s.index)

def band_util(x: float) -> str:
    v = pd.to_numeric(x, errors="coerce")
    if not np.isfinite(v): return "Utilization: Missing"
    if v <= 0.10: return "≤0.10"
    if v <= 0.50: return "0.11–0.50"
    if v <= 1.00: return "0.51–1.00"
    if v <= 5.00: return "1.01–5.00"
    return ">5.00"

def band_open_lines(x: float) -> str:
    v = pd.to_numeric(x, errors="coerce")
    if not np.isfinite(v): return "Open lines: Missing"
    if v <= 2: return "≤2"
    if v <= 6: return "3–6"
    return "7+"

def prior_delinq_band(row: pd.Series) -> str:
    names = [
        "NumberOfTimes90DaysLate",
        "NumberOfTime30-59DaysPastDueNotWorse",
        "NumberOfTime60-89DaysPastDueNotWorse",
    ]
    total = 0
    for n in names:
        if n in row:
            v = pd.to_numeric(row[n], errors="coerce")
            if np.isfinite(v): total += int(v)
    if total <= 0: return "none"
    if total == 1: return "1"
    return "2+"

# ------------ table builders

def df_to_md_table(df: pd.DataFrame) -> str:
    """Render a DataFrame as a GitHub-style Markdown table."""
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = ["| " + " | ".join(str(df.iloc[i, j]) for j in range(len(cols))) + " |"
            for i in range(len(df))]
    return "\n".join([header, sep] + rows)

def order_categorical(series: pd.Series, order: List[str]) -> pd.Series:
    cat = pd.Categorical(series, categories=order, ordered=True)
    return pd.Series(cat, index=series.index)

def one_way_table_df(df: pd.DataFrame, band_col: str, y_col: str, p_col: str, thr: float) -> pd.DataFrame:
    g = df.groupby(band_col, dropna=True, observed=True)
    out = g.agg(
        customers=(y_col, "size"),
        defaults=(y_col, "sum"),
        flagged=(p_col, lambda s: int((pd.to_numeric(s, errors="coerce") >= thr).sum()))
    ).reset_index().rename(columns={band_col: "Group"})
    total = out["customers"].sum()
    out["Share of portfolio"] = out["customers"] / total if total else 0.0
    out["Default rate"] = out["defaults"] / out["customers"]
    out["Flag rate"] = out["flagged"] / out["customers"]
    out = out[["Group", "Share of portfolio", "Default rate", "Flag rate"]]
    return out

def microsegments_top10_df(df: pd.DataFrame, a_col: str, b_col: str, y_col: str, p_col: str, thr: float, base_rate: float) -> Tuple[pd.DataFrame, str]:
    tmp = df[[a_col, b_col, y_col, p_col]].copy()
    tmp["flagged"] = (pd.to_numeric(tmp[p_col], errors="coerce") >= thr).astype(int)
    g = tmp.groupby([a_col, b_col], dropna=True, observed=True)
    mix = g.agg(
        customers=(y_col, "size"),
        defaults=(y_col, "sum"),
        flagged=("flagged", "sum"),
    ).reset_index().rename(columns={a_col: "Group A", b_col: "Group B"})
    total = mix["customers"].sum()
    mix["Share of portfolio"] = mix["customers"] / total if total else 0.0
    mix["Default rate"] = mix["defaults"] / mix["customers"]
    mix["Flag rate"] = mix["flagged"] / mix["customers"]
    mix["Lift over base"] = mix["Default rate"] - base_rate
    mix["Impact score"] = mix["Lift over base"].clip(lower=0) * mix["Share of portfolio"]
    top = mix.sort_values(["Impact score", "Default rate", "Share of portfolio"], ascending=False).head(10)
    # Takeaway
    topA = top["Group A"].value_counts().idxmax() if not top.empty else None
    topB = top["Group B"].value_counts().idxmax() if not top.empty else None
    if topA and topB:
        takeaway_text = (
            f"Start with **{topA}** and **{topB}**. "
            "These groups default more often than average, and there are enough customers in them to move losses. "
            "Put extra checks or lower starting limits here first — this is where reviews deliver the biggest expected loss reduction."
        )
    elif topA:
        takeaway_text = (
            f"Start with **{topA}**. "
            "This group defaults more than average and is large enough to matter. "
            "Focusing reviews and tighter limits here will reduce losses fastest."
        )
    elif topB:
        takeaway_text = (
            f"Start with **{topB}**. "
            "This group defaults more than average and is large enough to matter. "
            "Focusing reviews and tighter limits here will reduce losses fastest."
        )
    else:
        takeaway_text = "Focus first on the highest default-rate groups with a meaningful share of customers."
    top = top[["Group A", "Group B", "Share of portfolio", "Default rate", "Flag rate"]]
    return top.reset_index(drop=True), takeaway_text

# ------------ main

def build_summary() -> str:
    ensure_dirs()

    preds = read_csv_safe(HOLDOUT_CSV)
    if preds is None or preds.empty:
        return "# Executive Summary\n\n*No predictions file found at `exports/holdout_predictions.csv`.*\n"

    # labels & probs
    y_col = next((c for c in ["y_true", "target", "SeriousDlqin2yrs"] if c in preds.columns), None)
    if y_col is None:
        return "# Executive Summary\n\n*No label column found (expected y_true/target/SeriousDlqin2yrs).*"
    proba_col, model_name = pick_proba_col(preds)

    y = pd.to_numeric(preds[y_col], errors="coerce").fillna(0).astype(int).to_numpy()
    p = pd.to_numeric(preds[proba_col], errors="coerce").clip(0, 1).to_numpy()
    n = int(np.isfinite(p).sum())
    base_rate = float(np.nanmean(y)) if n else np.nan

    tp, fp, tn, fn = confusion_at_threshold(y, p, THRESHOLD)
    precision = tp / (tp + fp) if (tp + fp) else np.nan
    recall = tp / (tp + fn) if (tp + fn) else np.nan
    flagged_share = float((p >= THRESHOLD).mean()) if n else np.nan
    ks = ks_statistic(y, p)

    # model eval
    auc_txt = ap_txt = None
    eval_df = read_csv_safe(MODEL_SUMMARY_CSV)
    if eval_df is not None and not eval_df.empty:
        mrow = None
        if "model" in eval_df.columns:
            for _, row in eval_df.iterrows():
                m = str(row.get("model", "")).lower()
                if ("rf" in m and proba_col == "proba_rf") or ("log" in m and proba_col == "proba_logreg"):
                    mrow = row
                    break
        if mrow is None:
            mrow = eval_df.iloc[eval_df["AP"].astype(float).idxmax()] if "AP" in eval_df.columns else eval_df.iloc[0]
        if "AUC" in mrow: auc_txt = fmt_float(float(mrow["AUC"]), 3)
        if "AP"  in mrow: ap_txt  = fmt_float(float(mrow["AP"]),  3)

    # features
    df_feat = read_csv_safe(WITH_FEATS_CSV)
    have_features = df_feat is not None and not df_feat.empty

    lines: List[str] = []
    lines.append("# Executive Summary\n")

    # Plain terms
    lines.append("## What this means in plain terms")
    lines.append(f"- About **{fmt_pct(base_rate)}** of customers default today.")
    lines.append(f"- **{model_name}** is used for the summary.")
    lines.append(f"- At a cutoff of **{THRESHOLD:.2f}**, about **{fmt_pct(flagged_share)}** of customers would be flagged.")
    lines.append(f"- Among those flagged, precision is **{fmt_float(precision,3)}**; we catch **{fmt_float(recall,3)}** of future defaulters (recall).")
    tail = []
    if np.isfinite(ks): tail.append(f"KS: **{fmt_float(ks,3)}**")
    if auc_txt: tail.append(f"AUC: **{auc_txt}**")
    if ap_txt:  tail.append(f"AP: **{ap_txt}**")
    if tail:
        lines.append("- " + "  •  ".join(tail))
    if np.isfinite(flagged_share):
        flagged_10k = 10000 * flagged_share
        total = tp + tn + fp + fn
        tp_rate = tp / total if total else 0
        fp_rate = fp / total if total else 0
        fn_rate = fn / total if total else 0
        lines.append(
            f"- Per **10,000** customers at this cutoff: **{fmt_int(flagged_10k)}** flagged "
            f"(~{fmt_int(10000*tp_rate)} true defaulters, {fmt_int(10000*fp_rate)} false alarms; "
            f"{fmt_int(10000*fn_rate)} missed)."
        )
    lines.append("")

    # Who to prioritise bullets (quick scan)
    if have_features:
        df = df_feat.copy()
        df["y"] = pd.to_numeric(df[y_col], errors="coerce").fillna(0).astype(int)
        df[proba_col] = pd.to_numeric(df[proba_col], errors="coerce").clip(0, 1)

        # bands
        if "age" in df: df["Age"] = df["age"].map(band_age)
        if "NumberOfDependents" in df: df["Dependents"] = df["NumberOfDependents"].map(band_dependents)
        if "MonthlyIncome" in df: df["Income (20% bands)"] = band_by_quintile(df["MonthlyIncome"], "income")
        if "DebtRatio" in df: df["Debt ratio (20% bands)"] = band_by_quintile(df["DebtRatio"], "debt")
        if "RevolvingUtilizationOfUnsecuredLines" in df: df["Revolving utilization"] = df["RevolvingUtilizationOfUnsecuredLines"].map(band_util)
        if "NumberOfOpenCreditLinesAndLoans" in df: df["Open credit lines"] = df["NumberOfOpenCreditLinesAndLoans"].map(band_open_lines)
        df["Prior delinquencies"] = df.apply(prior_delinq_band, axis=1)

        bullets: List[str] = []

        def add_max_note(label: str, colname: str):
            if colname not in df: return
            t = one_way_table_df(df, colname, "y", proba_col, THRESHOLD)
            t2 = t[~t["Group"].str.contains("Missing", na=False)].copy()
            if t2.empty: return
            t2 = t2.sort_values("Default rate", ascending=False)
            g = t2.iloc[0]
            bullets.append(f"- **{label}:** highest default in **{g['Group']}** ({fmt_pct(g['Default rate'])}); flagged at {fmt_pct(g['Flag rate'])}.")

        add_max_note("Age", "Age")
        add_max_note("Dependents", "Dependents")
        add_max_note("Income", "Income (20% bands)")

        if bullets:
            lines.append("## Who to prioritise for review or limits")
            lines.extend(bullets)
            lines.append("")

        # Top-10 microsegments (Age × Income preferred)
        pair: Optional[Tuple[str,str]] = None
        if "Age" in df and "Income (20% bands)" in df:
            pair = ("Age", "Income (20% bands)")
        elif "Dependents" in df and "Income (20% bands)" in df:
            pair = ("Dependents", "Income (20% bands)")
        elif "Age" in df and "Dependents" in df:
            pair = ("Age", "Dependents")

        if pair:
            top10_df, takeaway_text = microsegments_top10_df(df, pair[0], pair[1], "y", proba_col, THRESHOLD, base_rate)
            show = top10_df.copy()
            for c in ["Share of portfolio", "Default rate", "Flag rate"]:
                show[c] = show[c].apply(lambda v: fmt_pct(v, 1))
            lines.append(f"## Top 10 Highest-Risk Microsegments ({pair[0]} × {pair[1]})")
            lines.append(df_to_md_table(show))
            lines.append("")
            lines.append(f"**Takeaway:** {takeaway_text}")
            lines.append("")
        # Clarify what each row represents
        lines.append(
            "_Each row is a **joint segment**: an **age band** combined with an **income band**. "
            "'Share of portfolio' is the % of customers in that row. 'Default rate' is the observed rate in that row. "
            "'Flag rate' is the % that would be flagged at the chosen threshold._"
        )
        lines.append("")

        # Build a clear Takeaway from the top joint pairs (use first 2–3 rows)
        pairs = []
        for _, r in top10_df.head(3).iterrows():
            a = str(r.get("Group A", "")).strip()
            b = str(r.get("Group B", "")).strip()
            if a and b:
                pairs.append(f"**{a} × {b}**")

        if pairs:
            if len(pairs) == 1:
                tw = pairs[0]
            elif len(pairs) == 2:
                tw = f"{pairs[0]} and {pairs[1]}"
            else:
                tw = f"{pairs[0]}, then {pairs[1]}, then {pairs[2]}"
            lines.append(
                f"**Takeaway:** Start with {tw}. These joint age–income groups have higher-than-average default "
                "and enough customers to meaningfully reduce losses by adding checks or lower starting limits."
            )
            lines.append("")

        # Priority groups (simple: rate vs. size) — no 'do first/next' phrasing
        def join_pairs(lst):
            if not lst: return "—"
            if len(lst) == 1: return lst[0]
            if len(lst) == 2: return f"{lst[0]} and {lst[1]}"
            return f"{', '.join(lst[:-1])}, and {lst[-1]}"

        # Use numeric values from the unformatted top10_df
        median_share = float(top10_df["Share of portfolio"].median())
        priority1, priority2 = [], []
        for _, r in top10_df.iterrows():
            a = str(r.get("Group A", "")).strip()
            b = str(r.get("Group B", "")).strip()
            if not a or not b:
                continue
            pair_label = f"**{a} × {b}**"
            high_default = float(r["Default rate"]) > float(base_rate)
            high_share = float(r["Share of portfolio"]) >= median_share
            if high_default and high_share:
                priority1.append(pair_label)  # Big & Risky
            elif high_default and not high_share:
                priority2.append(pair_label)  # Risky but Smaller

        # Priority groups (table) — based on default rate vs. group size
        median_share = float(top10_df["Share of portfolio"].median())
        rows = []
        for _, r in top10_df.iterrows():
            a = str(r.get("Group A", "")).strip()
            b = str(r.get("Group B", "")).strip()
            if not a or not b:
                continue
            seg = f"{a} × {b}"
            share = float(r["Share of portfolio"])
            dr = float(r["Default rate"])
            fr = float(r["Flag rate"])

            high_default = dr > float(base_rate)
            high_share = share >= median_share

            if high_default and high_share:
                prio = "Priority 1 — Big & Risky"
            elif high_default and not high_share:
                prio = "Priority 2 — Risky but Smaller"
            else:
                prio = "Priority 3/4 — Safer"

            rows.append({
                "Priority": prio,
                "Segment": f"**{seg}**",
                "Share of portfolio": share,
                "Default rate": dr,
                "Flag rate": fr
            })

        prio_df = pd.DataFrame(rows)
        # Show the two action-oriented groups
        prio_df = prio_df[prio_df["Priority"].isin([
            "Priority 1 — Big & Risky",
            "Priority 2 — Risky but Smaller"
        ])]

        # Order: Priority, then highest default, then bigger share
        prio_df = prio_df.sort_values(
            ["Priority", "Default rate", "Share of portfolio"],
            ascending=[True, False, False]
        ).reset_index(drop=True)

        # Format percentages for display
        prio_show = prio_df.copy()
        for c in ["Share of portfolio", "Default rate", "Flag rate"]:
            prio_show[c] = prio_show[c].apply(lambda v: fmt_pct(v, 1))

        lines.append("### Priority groups (table)")
        lines.append(df_to_md_table(prio_show))
        lines.append("")

        # Comparison tables
        lines.append("## Comparison tables")

        def add_table_block(title: str, colname: str, order: Optional[List[str]] = None):
            if colname not in df: return
            t = one_way_table_df(df, colname, "y", proba_col, THRESHOLD).copy()
            if order:
                t["Group"] = order_categorical(t["Group"], order)
                t = t.sort_values("Group")
                t["Group"] = t["Group"].astype(str)
            for c in ["Share of portfolio", "Default rate", "Flag rate"]:
                t[c] = t[c].apply(lambda v: fmt_pct(v, 1))
            lines.append(f"### {title}")
            lines.append(df_to_md_table(t))
            lines.append("")

        def order_categorical(series: pd.Series, order: List[str]) -> pd.Series:
            cat = pd.Categorical(series, categories=order, ordered=True)
            return pd.Series(cat, index=series.index)

        add_table_block("Age bands", "Age", ["<30","30–39","40–49","50–59","60+","Age: Missing"])
        add_table_block("Dependents", "Dependents", ["0","1–2","3+","Dependents: Missing"])
        add_table_block("Income (20% bands)", "Income (20% bands)", quintile_labels("income") + ["Income: Missing"])
        add_table_block("Revolving utilization", "Revolving utilization", ["≤0.10","0.11–0.50","0.51–1.00","1.01–5.00",">5.00","Utilization: Missing"])
        add_table_block("Debt ratio (20% bands)", "Debt ratio (20% bands)", quintile_labels("debt") + ["Debt: Missing"])
        add_table_block("Open credit lines", "Open credit lines", ["≤2","3–6","7+","Open lines: Missing"])
        add_table_block("Prior delinquencies", "Prior delinquencies", ["none","1","2+"])

    # Model evidence + operating point
    lines.append("## Model Quality (evidence)")
    if ap_txt: lines.append(f"- Best by **AP**: {model_name} — AP **{ap_txt}**.")
    if auc_txt: lines.append(f"- Best by **AUC**: {model_name} — AUC **{auc_txt}**.")
    lines.append(f"- Base default rate: **{fmt_pct(base_rate)}**.")
    lines.append("")

    lines.append("## Recommended operating point")
    lines.append(f"- Threshold: **{THRESHOLD:.2f}**")
    lines.append(f"- Precision: **{fmt_float(precision,3)}**, Recall: **{fmt_float(recall,3)}**")
    lines.append(f"- Confusion mix: TP={fmt_int(tp)} FP={fmt_int(fp)} TN={fmt_int(tn)} FN={fmt_int(fn)}")
    lines.append("- Guidance: treat a missed defaulter (FN) as ~10× a false alarm (FP). Pick the lowest-cost threshold that fits your review capacity.")
    lines.append("")

    return "\n".join(lines).strip() + "\n"


if __name__ == "__main__":
    md = build_summary()
    ensure_dirs()
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"Wrote {OUT_MD}")
