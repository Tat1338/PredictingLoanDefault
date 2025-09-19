# exec_summary.py — Executive Summary generator (plain-English + deep insights)

# - Age/Dependents/Income (plain English)
# - Risk tiers (A–E)
# - Threshold scenarios (capacity vs precision/recall)
# - Comparison tables (with "Share of portfolio")
# - Two-way patterns (e.g., Age×Income)
# - Policy rules (draft)
# - Top 10 Highest-Risk Microsegments (+ "Take away" + clear explanation)
#
# Usage (from your project root, with the app still running):
#   .\.venv\Scripts\python.exe exec_summary.py
# Then click Rerun in the Streamlit app → Executive Summary page.

from __future__ import annotations
import os, textwrap
import numpy as np
import pandas as pd

EXPORTS = "exports"
REPORTS = "reports"
os.makedirs(REPORTS, exist_ok=True)

# ---------- helpers ----------
def num(x, d=3):
    return "—" if x is None or pd.isna(x) else f"{float(x):.{d}f}"

def pct(x, d=1):
    return "—" if x is None or pd.isna(x) else f"{100*float(x):.{d}f}%"

def fmt_int(x):
    if x is None: return "—"
    if isinstance(x, float) and pd.isna(x): return "—"
    try: return f"{int(x):,}"
    except Exception: return "—"

def ks_stat(y, p):
    y = np.asarray(y).astype(int); p = np.asarray(p).astype(float)
    if y.size == 0 or p.size == 0: return np.nan
    pos = np.sort(p[y==1]); neg = np.sort(p[y==0])
    if len(pos)==0 or len(neg)==0: return np.nan
    grid = np.linspace(0,1,1001)
    cdf_p = np.searchsorted(pos, grid, side="right")/len(pos)
    cdf_n = np.searchsorted(neg, grid, side="right")/len(neg)
    return float(np.max(np.abs(cdf_p-cdf_n)))

def brier(y, p):
    y = np.asarray(y).astype(float); p = np.asarray(p).astype(float)
    if len(y)!=len(p) or len(y)==0: return np.nan
    return float(np.mean((y-p)**2))

def get_first_col(df: pd.DataFrame, names: list[str]) -> str | None:
    for n in names:
        if n in df.columns: return n
    low = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in low: return low[n.lower()]
    return None

def qbands(s: pd.Series, q=5, labels=None):
    s = pd.to_numeric(s, errors="coerce")
    if labels is None: labels = [f"Q{i}" for i in range(1, q+1)]
    try:
        return pd.qcut(s, q=q, labels=labels, duplicates="drop")
    except Exception:
        s_clean = s.replace([np.inf, -np.inf], np.nan).dropna()
        if s_clean.empty: return pd.Series([np.nan]*len(s), index=s.index)
        bins = np.linspace(s_clean.min(), s_clean.max(), q+1)
        try:
            return pd.cut(s, bins=bins, labels=labels, include_lowest=True)
        except Exception:
            return pd.Series([np.nan]*len(s), index=s.index)

# ---------- load data ----------
perf = pd.read_csv(os.path.join(EXPORTS, "model_eval_summary.csv")) if os.path.exists(os.path.join(EXPORTS,"model_eval_summary.csv")) else None

hold = None
for _p in [os.path.join(EXPORTS,"holdout_with_features.csv"),
           os.path.join(EXPORTS,"holdout_predictions.csv")]:
    if os.path.exists(_p):
        hold = pd.read_csv(_p)
        break
if hold is None or hold.empty:
    raise SystemExit("No holdout file found in exports/ (need holdout_with_features.csv or holdout_predictions.csv)")

thr  = pd.read_csv(os.path.join(EXPORTS, "threshold_metrics.csv")) if os.path.exists(os.path.join(EXPORTS,"threshold_metrics.csv")) else None

# ---------- find columns ----------
ycol = get_first_col(hold, ["y_true","target","SeriousDlqin2yrs"])
proba_col = "proba_rf" if "proba_rf" in hold.columns else next((c for c in hold.columns if str(c).lower().startswith("proba")), None)

age_col   = get_first_col(hold, ["age","Age"])
inc_col   = get_first_col(hold, ["MonthlyIncome","monthly_income","income"])
dep_col   = get_first_col(hold, ["NumberOfDependents","Dependents","dependents"])
util_col  = get_first_col(hold, ["RevolvingUtilizationOfUnsecuredLines","revolving_utilization","ruul"])
debt_col  = get_first_col(hold, ["DebtRatio","debt_ratio"])
open_col  = get_first_col(hold, ["NumberOfOpenCreditLinesAndLoans","open_credit_lines","open_lines"])
late30    = get_first_col(hold, ["NumberOfTime30-59DaysPastDueNotWorse","NumberOfTimes30-59DaysPastDueNotWorse","times_30_59"])
late60    = get_first_col(hold, ["NumberOfTime60-89DaysPastDueNotWorse","NumberOfTimes60-89DaysPastDueNotWorse","times_60_89"])
late90    = get_first_col(hold, ["NumberOfTimes90DaysLate","times_90_plus"])

# ---------- headline numbers ----------
base_rate = float(pd.to_numeric(hold[ycol], errors="coerce").mean()) if ycol else np.nan

best_ap_name=best_auc_name=None; best_ap=best_auc=np.nan
if perf is not None and not perf.empty:
    if "AUC" in perf.columns: perf["AUC"] = pd.to_numeric(perf["AUC"], errors="coerce")
    if "AP"  in perf.columns: perf["AP"]  = pd.to_numeric(perf["AP"],  errors="coerce")
    if "AP" in perf.columns and perf["AP"].notna().any():
        i=int(perf["AP"].idxmax()); best_ap=float(perf.loc[i,"AP"]); best_ap_name=str(perf.loc[i,"model"])
    if "AUC" in perf.columns and perf["AUC"].notna().any():
        i=int(perf["AUC"].idxmax()); best_auc=float(perf.loc[i,"AUC"]); best_auc_name=str(perf.loc[i,"model"])

# threshold
thr_star=0.30; prec=rec=np.nan; tp=fp=tn=fn=None
if thr is not None and not thr.empty:
    df=thr.copy()
    pick=None
    for c in ["F1","f1"]:
        if c in df.columns and df[c].notna().any():
            pick=df.sort_values(c, ascending=False).iloc[0]; break
    if pick is None and {"precision","recall"}.issubset(df.columns):
        cand=df[df["precision"]>=0.4].sort_values("recall", ascending=False)
        pick=cand.iloc[0] if not cand.empty else df.iloc[0]
    if pick is None: pick=df.iloc[0]
    thr_star=float(pick.get("threshold",thr_star))
    prec=pick.get("precision",pick.get("Precision",np.nan))
    rec =pick.get("recall",   pick.get("Recall",   np.nan))
    tp  =pick.get("tp",None); fp=pick.get("fp",None); tn=pick.get("tn",None); fn=pick.get("fn",None)

# distribution & quality
scores = pd.to_numeric(hold[proba_col], errors="coerce").clip(0,1) if proba_col else pd.Series(dtype=float)
n = int(scores.notna().sum()) if proba_col else 0
flagged = int((scores>=thr_star).sum()) if proba_col else 0
flag_rate = flagged/n if n else np.nan

ks_val=brier_rf=mean_pos=mean_neg=np.nan
if ycol and proba_col:
    y=pd.to_numeric(hold[ycol], errors="coerce").fillna(0).astype(int)
    ks_val = ks_stat(y, scores.to_numpy())
    brier_rf = brier(y, scores.to_numpy())
    mean_pos=float(scores[y==1].mean()) if (y==1).any() else np.nan
    mean_neg=float(scores[y==0].mean()) if (y==0).any() else np.nan
    if any(v is None for v in [tp,fp,tn,fn]):
        pred=(scores>=thr_star).astype(int)
        tp=int(((pred==1)&(y==1)).sum()); fp=int(((pred==1)&(y==0)).sum())
        tn=int(((pred==0)&(y==0)).sum()); fn=int(((pred==0)&(y==1)).sum())

# ---------- build bands for slices ----------
def build_bands(df: pd.DataFrame):
    bands = {}
    if age_col:
        age = pd.to_numeric(df[age_col], errors="coerce")
        bands["age_band"] = pd.cut(age, bins=[0,30,40,50,60,200],
                                   labels=["<30","30–39","40–49","50–59","60+"], include_lowest=True)
    if inc_col:
        inc = pd.to_numeric(df[inc_col], errors="coerce")
        # Rename away from Q1..Q5 to avoid “quarter” confusion
        bands["inc_q"] = qbands(
            inc, q=5,
            labels=["Income: Bottom 20%","Income: Low 20%","Income: Middle 20%","Income: High 20%","Income: Top 20%"]
        )
    if dep_col:
        dep = pd.to_numeric(df[dep_col], errors="coerce")
        bands["dep_band"] = pd.cut(dep.fillna(0), bins=[-1,0,2,100],
                                   labels=["0","1–2","3+"], include_lowest=True)
    if util_col:
        u = pd.to_numeric(df[util_col], errors="coerce")
        bands["util_band"] = pd.cut(u, bins=[-0.001,0.1,0.5,1.0,5.0, np.inf],
                                    labels=["≤0.10","0.11–0.50","0.51–1.00","1.01–5.00",">5.00"], include_lowest=True)
    if debt_col:
        dr = pd.to_numeric(df[debt_col], errors="coerce")
        bands["dr_q"] = qbands(
            dr, q=5,
            labels=["Debt: Lowest 20%","Debt: Low 20%","Debt: Middle 20%","Debt: High 20%","Debt: Highest 20%"]
        )
    if open_col:
        op = pd.to_numeric(df[open_col], errors="coerce")
        bands["open_band"] = pd.cut(op.fillna(0), bins=[-1,2,6,100],
                                    labels=["≤2","3–6","7+"], include_lowest=True)
    if any([late30, late60, late90]):
        def to_num(c):
            return pd.to_numeric(df[c], errors="coerce") if c else pd.Series([0]*len(df))
        l30 = to_num(late30); l60 = to_num(late60); l90 = to_num(late90)
        total_late = l30.fillna(0) + l60.fillna(0) + l90.fillna(0)
        bands["late_band"] = pd.cut(total_late.fillna(0), bins=[-0.1,0.5,1.5,100],
                                    labels=["none","1","2+"], include_lowest=True)
    return bands

bands = build_bands(hold)

# ---------- plain-English headline ----------
plain=[]
plain.append(f"About **{pct(base_rate)}** of customers default today.")
if best_ap_name: plain.append(f"**{best_ap_name}** performs best (AP **{num(best_ap,3)}**, AUC **{num(best_auc,3)}**).")
if not pd.isna(flag_rate): plain.append(f"At cutoff **{num(thr_star,2)}**, about **{pct(flag_rate)}** of customers are flagged.")
if not pd.isna(prec) and not pd.isna(rec): plain.append(f"Among flagged, ~**{num(prec,3)}** are true defaulters; we catch **{num(rec,3)}** of all defaulters.")
if not pd.isna(ks_val): plain.append(f"Separation is strong (**KS {num(ks_val,3)}**); probabilities align (**Brier {num(brier_rf,3)}**).")

counts_line=""
if n:
    scale=10000/n
    flagged_10k=int(round(flagged*scale)); tp_10k=int(round((tp or 0)*scale)); fp_10k=int(round((fp or 0)*scale)); fn_10k=int(round((fn or 0)*scale))
    counts_line=f"Per **10,000** customers: **{fmt_int(flagged_10k)} flagged** (~**{fmt_int(tp_10k)}** true, **{fmt_int(fp_10k)}** false alarms; **{fmt_int(fn_10k)}** missed)."

# ---------- Age, Dependents & Income — plain English ----------
easy_lines = []
if ycol and "age_band" in bands:
    df_age = pd.DataFrame({"y": pd.to_numeric(hold[ycol], errors="coerce"),
                           "band": bands["age_band"]}).dropna()
    if not df_age.empty:
        r = df_age.groupby("band")["y"].mean()
        hi, lo = r.idxmax(), r.idxmin()
        easy_lines.append(f"**By age:** highest default in **{hi}** at **{pct(r.loc[hi])}**; lowest in **{lo}** at **{pct(r.loc[lo])}**.")
if ycol and "dep_band" in bands:
    df_dep = pd.DataFrame({"y": pd.to_numeric(hold[ycol], errors="coerce"),
                           "band": bands["dep_band"]}).dropna()
    if not df_dep.empty:
        r = df_dep.groupby("band")["y"].mean()
        hi, lo = r.idxmax(), r.idxmin()
        easy_lines.append(f"**Dependents:** highest default with **{hi}** dependents at **{pct(r.loc[hi])}**; lowest in **{lo}** at **{pct(r.loc[lo])}**.")
if ycol and "inc_q" in bands:
    df_inc = pd.DataFrame({"y": pd.to_numeric(hold[ycol], errors="coerce"),
                           "band": bands["inc_q"]}).dropna()
    if not df_inc.empty:
        r = df_inc.groupby("band")["y"].mean()
        hi, lo = r.idxmax(), r.idxmin()
        easy_lines.append(f"**Income:** highest default in **{hi}** at **{pct(r.loc[hi])}**; lowest in **{lo}** at **{pct(r.loc[lo])}**.")

# ---------- risk tiers (A–E) ----------
tiers_txt=""
if proba_col and ycol:
    qs = scores.rank(pct=True)
    tier = pd.Series(index=scores.index, dtype="object")
    tier[qs>0.60] = "E"
    tier[(qs>0.40)&(qs<=0.60)]="D"
    tier[(qs>0.20)&(qs<=0.40)]="C"
    tier[(qs>0.10)&(qs<=0.20)]="B"
    tier[(qs<=0.10)]="A"
    df_t = pd.DataFrame({"tier":tier, "y":y, "s":scores})
    g = df_t.groupby("tier", dropna=True)
    out=[]
    for t in ["A","B","C","D","E"]:
        if t in g.groups:
            gi=g.get_group(t); rate=pd.to_numeric(gi["y"], errors="coerce").mean(); share=len(gi)/len(df_t)
            out.append(f"{t}: default **{pct(rate)}**, population **{pct(share)}**")
    if out: tiers_txt = " • ".join(out)

# ---------- threshold scenarios ----------
scenarios_txt=""
df_s = None
if proba_col and ycol:
    yv = y.to_numpy(); sv = scores.to_numpy()
    rows=[]
    for t in [0.20,0.25,0.30,0.35,0.40]:
        pred=(sv>=t).astype(int)
        TP=int(((pred==1)&(yv==1)).sum()); FP=int(((pred==1)&(yv==0)).sum())
        TN=int(((pred==0)&(yv==0)).sum()); FN=int(((pred==0)&(yv==1)).sum())
        precision = TP/(TP+FP) if TP+FP>0 else np.nan
        recall    = TP/(TP+FN) if TP+FN>0 else np.nan
        flagged   = TP+FP
        cost10 = 10*FN + 1*FP
        rows.append((t, flagged/len(yv), precision, recall, TP, FP, cost10))
    df_s = pd.DataFrame(rows, columns=["threshold","flag_rate","precision","recall","TP","FP","cost_F1_10"])
    scenarios_txt = " | ".join([f"t={r.threshold:.2f}: flag {pct(r.flag_rate)}, P={num(r.precision,3)}, R={num(r.recall,3)}, TP={fmt_int(r.TP)}, FP={fmt_int(r.FP)}, cost={fmt_int(r.cost_F1_10)}" for _,r in df_s.iterrows()])

# ---------- comparison tables ----------
def _mk_table_for_band(band_series: pd.Series, band_title: str) -> str | None:
    if band_series is None:
        return None
    tmp = pd.DataFrame({"band": band_series})
    if ycol:
        tmp["y"] = pd.to_numeric(hold[ycol], errors="coerce")
    if proba_col:
        tmp["flag"] = (scores >= thr_star).astype(float)
    tmp = tmp.dropna(subset=["band"])
    if tmp.empty or ("y" not in tmp.columns):
        return None

    g = tmp.groupby("band", dropna=True)
    pop = g.size() / len(tmp)
    def_rate = g["y"].mean()
    flag_rate = g["flag"].mean() if "flag" in tmp.columns else None

    lines = []
    lines.append(f"**{band_title}**\n")
    lines.append("| Group | Share of portfolio | Default rate | Flag rate |")
    lines.append("|---|---:|---:|---:|")
    for lab in [str(x) for x in pop.index.tolist()]:
        pop_pct = pct(pop.get(lab))
        def_pct = pct(def_rate.get(lab))
        flag_pct = pct(flag_rate.get(lab)) if flag_rate is not None else "—"
        lines.append(f"| {lab} | {pop_pct} | {def_pct} | {flag_pct} |")
    lines.append("")
    return "\n".join(lines)

tables = []
try:
    if "age_band"  in bands: tables.append(_mk_table_for_band(bands["age_band"], "Age bands"))
    if "dep_band"  in bands: tables.append(_mk_table_for_band(bands["dep_band"], "Dependents"))
    if "inc_q"     in bands: tables.append(_mk_table_for_band(bands["inc_q"], "Income (20% bands)"))
    if "util_band" in bands: tables.append(_mk_table_for_band(bands["util_band"], "Revolving utilization"))
    if "dr_q"      in bands: tables.append(_mk_table_for_band(bands["dr_q"], "Debt ratio (20% bands)"))
    if "open_band" in bands: tables.append(_mk_table_for_band(bands["open_band"], "Open credit lines"))
    if "late_band" in bands: tables.append(_mk_table_for_band(bands["late_band"], "Prior delinquencies"))
except Exception:
    pass

# ---------- two-way patterns ----------
def two_way_top(a: str, b: str, min_n: int = 200, top_k: int = 3):
    if ycol is None or a not in bands or b not in bands: return []
    tmp = pd.DataFrame({
        "a": bands[a],
        "b": bands[b],
        "y": pd.to_numeric(hold[ycol], errors="coerce")
    })
    if proba_col:
        tmp["flag"] = (scores >= thr_star).astype(float)
    tmp = tmp.dropna(subset=["a","b"])
    g = tmp.groupby(["a","b"])
    stat = g.agg(n=("y","size"),
                 rate=("y", "mean"),
                 flag=("flag","mean") if "flag" in tmp.columns else ("y","mean")).reset_index()
    stat = stat[stat["n"] >= min_n]
    stat = stat.sort_values("rate", ascending=False).head(top_k)
    out = []
    for _, r in stat.iterrows():
        out.append(f"{r['a']} × {r['b']}: default **{pct(r['rate'])}** (flag {pct(r['flag']) if 'flag' in stat.columns else '—'}, n={fmt_int(r['n'])})")
    return out

two_way_lines = []
two_way_lines += [f"- {s}" for s in two_way_top("age_band","inc_q")]
two_way_lines += [f"- {s}" for s in two_way_top("dep_band","inc_q")]
two_way_lines += [f"- {s}" for s in two_way_top("util_band","dr_q")]

# ---------- policy rules (draft) ----------
policy_lines = []
def add_rule(name: str, mask: pd.Series):
    if ycol is None: return
    yv = pd.to_numeric(hold[ycol], errors="coerce")
    nmask = int(mask.sum())
    if nmask < 200: return
    rate = yv[mask].mean()
    share = nmask / len(hold)
    if pd.isna(rate): return
    if rate >= (base_rate or 0)*1.20:  # 20%+ over baseline
        action = "Manual review + tighter limits" if rate >= (base_rate or 0)*1.50 else "Tighter limits or secondary checks"
        policy_lines.append(f"- **{name}** — default **{pct(rate)}** (portfolio **{pct(share)}**). **Action:** {action}.")

if "inc_q" in bands and "age_band" in bands:
    add_rule("Low income (bottom 20%) & age <30", (bands["inc_q"]=="Income: Bottom 20%") & (bands["age_band"]=="<30"))
if "util_band" in bands:
    if "late_band" in bands:
        add_rule("High utilization (≥1.01) with prior delinquency", bands["util_band"].isin(["1.01–5.00",">5.00"]) & (bands["late_band"]!="none"))
    add_rule("Extreme utilization (>5.00)", bands["util_band"]==">5.00")
if "dr_q" in bands and "open_band" in bands:
    add_rule("Debt ratio highest 20% with ≤2 open lines", (bands["dr_q"]=="Debt: Highest 20%") & (bands["open_band"]=="≤2"))
if "dep_band" in bands and "inc_q" in bands:
    add_rule("No dependents & low income (bottom 20%)", (bands["dep_band"]=="0") & (bands["inc_q"]=="Income: Bottom 20%"))

# ---------- Top 10 Highest-Risk Microsegments (with Take away & explanation) ----------
def top_microsegments(bands_dict: dict, min_n: int = 300, top_k: int = 10):
    pairs_order = [
        ("age_band", "inc_q"),
        ("dep_band", "inc_q"),
        ("util_band", "dr_q"),
        ("open_band", "dr_q"),
        ("late_band", "inc_q"),
    ]
    rows_num = []
    weightA, weightB = {}, {}

    if ycol is None or proba_col is None:
        return [], ""

    yv = pd.to_numeric(hold[ycol], errors="coerce")
    flags = (scores >= thr_star).astype(float)

    for a, b in pairs_order:
        if a not in bands_dict or b not in bands_dict:
            continue
        df = pd.DataFrame({
            "A": bands_dict[a],
            "B": bands_dict[b],
            "y": yv,
            "flag": flags
        }).dropna(subset=["A","B"])
        if df.empty:
            continue

        g = df.groupby(["A","B"], dropna=True)
        stat = g.agg(n=("y","size"), rate=("y", "mean"), flag_rate=("flag","mean")).reset_index()
        stat = stat[stat["n"] >= min_n]
        if stat.empty:
            continue

        # totals for weights within this pair set
        total_n = stat["n"].sum()
        for _, r in stat.iterrows():
            rate = float(r["rate"])
            lift = rate / (base_rate if base_rate else np.nan)
            pop_share_num = float(r["n"]) / float(total_n) if total_n else 0.0
            flag_rate_num = float(r["flag_rate"])
            seg = f"{r['A']} × {r['B']}"

            rows_num.append({
                "seg": seg,
                "rate_num": rate,
                "lift_num": lift,
                "pop_share_num": pop_share_num,
                "flag_rate_num": flag_rate_num,
                "n_num": int(r["n"]),
                "_A": str(r["A"]),
                "_B": str(r["B"]),
            })

            # weights (risk × size) for take away
            w = (lift if not pd.isna(lift) else 0.0) * pop_share_num
            weightA[r["A"]] = weightA.get(r["A"], 0.0) + w
            weightB[r["B"]] = weightB.get(r["B"], 0.0) + w

    if not rows_num:
        return [], ""

    # global top_k by default rate
    rows_num = sorted(rows_num, key=lambda d: d["rate_num"], reverse=True)[:top_k]

    formatted = []
    for r in rows_num:
        formatted.append({
            "Segment": r["seg"],
            "Share of portfolio": pct(r["pop_share_num"]),
            "Default rate": pct(r["rate_num"]),
            "Lift vs base": num(r["lift_num"], 2),
            "Flag rate": pct(r["flag_rate_num"]),
            "n": fmt_int(r["n_num"]),
        })

    topA = max(weightA.items(), key=lambda kv: kv[1])[0] if weightA else None
    topB = max(weightB.items(), key=lambda kv: kv[1])[0] if weightB else None
    takeaway = ""
    if topA and topB:
        takeaway = f"Risk concentrates in **{topA}** combined with **{topB}**."
    elif topA:
        takeaway = f"Risk concentrates in **{topA}**."
    elif topB:
        takeaway = f"Risk concentrates in **{topB}**."

    return formatted, takeaway

micro_rows, micro_takeaway = top_microsegments(bands, min_n=300, top_k=10)

# ---------- compose markdown ----------
parts=[]
parts.append("# Executive Summary")

parts.append("\n## What this means in plain terms")
parts.append("- " + "\n- ".join(plain))
if counts_line: parts.append("\n" + counts_line)

if easy_lines:
    parts.append("\n## Age, Dependents & Income — plain English")
    parts.append("- " + "\n- ".join(easy_lines))
    parts.append("**How to use this:** keep decisions score-driven and policy-compliant; use these slices to plan reviews and limits, not as stand-alone rules.")

if tiers_txt:
    parts.append("\n## Risk tiers (A–E)")
    parts.append(f"- {tiers_txt}")
    parts.append("**Use:** focus manual reviews on **A/B**, moderate limits on **C**, lighter touch on **D/E**.")

if scenarios_txt:
    parts.append("\n## Threshold scenarios (capacity & trade-offs)")
    parts.append(scenarios_txt)
    parts.append("\n**Tip:** choose the row with the **lowest cost** that fits your review capacity.")

if two_way_lines:
    parts.append("\n## Two-way patterns (who is consistently riskier)")
    parts.extend(two_way_lines)

if policy_lines:
    parts.append("\n## Policy rules (draft) — who to review or limit")
    parts.extend(policy_lines)

if any(tables):
    parts.append("\n## Comparison tables")
    parts.extend([t for t in tables if t])

if micro_rows:
    parts.append("\n## Top 10 Highest-Risk Microsegments")
    parts.append("| Segment | Share of portfolio | Default rate | Lift vs base | Flag rate | n |")
    parts.append("|---|---:|---:|---:|---:|---:|")
    for r in micro_rows:
        parts.append(f"| {r['Segment']} | {r['Share of portfolio']} | {r['Default rate']} | {r['Lift vs base']} | {r['Flag rate']} | {r['n']} |")
    parts.append("")
    if micro_takeaway:
        parts.append(f"**Take away:** {micro_takeaway}")
    parts.append(textwrap.dedent("""
    **What this shows:** small, specific groups where risk clusters when two fields are combined (e.g., age with income, or utilization with debt ratio).

    **How to read it**
    - **Share of portfolio** — how many customers sit in that group.
    - **Default rate** — loss risk inside the group.
    - **Lift vs base** — how much worse than the overall average (1.20× = 20% higher than average).
    - **Flag rate** — at the current cutoff, what share would be sent to review.

    **How to act**
    - Prioritise control actions (extra docs, manual checks, lower starting limits, pricing add-ons) for segments with **high default rate/lift** and a **meaningful share of the portfolio**.
    - Keep final decisions **score-driven** and **policy-compliant**; segments focus your effort, not replace the score.
    """).strip())

parts.append("\n## Model Quality (evidence)")
mq=[]
if best_ap_name: mq.append(f"**Best by AP:** {best_ap_name} — AP **{num(best_ap,3)}**.")
if best_auc_name: mq.append(f"**Best by AUC:** {best_auc_name} — AUC **{num(best_auc,3)}**.")
if not pd.isna(base_rate) and not pd.isna(best_ap) and (base_rate or 0)>0:
    mq.append(f"Base default rate: **{pct(base_rate)}**. Best AP ≈ **{num(best_ap,3)}** (~**{num(best_ap/(base_rate or np.nan),2)}×** random baseline).")
if not pd.isna(ks_val): mq.append(f"Separation: **KS {num(ks_val,3)}**.")
if not pd.isna(brier_rf): mq.append(f"Calibration: **Brier {num(brier_rf,3)}**.")
parts.append("- " + "\n- ".join(mq) if mq else "- (add exports files to populate)")

parts.append("\n## Recommended operating point")
parts.append(f"- Threshold: **{num(thr_star,2)}**")
parts.append(f"- Precision: **{num(prec,3)}**, Recall: **{num(rec,3)}**")
if not any(v is None for v in [tp,fp,tn,fn]):
    parts.append(f"- Confusion mix: TP={fmt_int(tp)} FP={fmt_int(fp)} TN={fmt_int(tn)} FN={fmt_int(fn)}")
parts.append(textwrap.dedent("""
**Costed impact (example):** treat a missed defaulter (FN) as ~10× a false alarm (FP). Pick the lowest-cost threshold that your team can review.
""").strip())

parts.append("\n## Recommendation")
parts.append("Pilot for 4–6 weeks at the chosen cutoff. Monitor **precision, recall, Brier, KS, flag rate** weekly; recalibrate quarterly or on drift.")

# ---------- write ----------
out_path = os.path.join(REPORTS,"executive_summary.md")
with open(out_path, "w", encoding="utf-8") as f:
    f.write("\n".join(parts).strip() + "\n")
print(f"Wrote {out_path}")
