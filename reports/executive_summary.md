# Executive Summary

## What this means in plain terms
- About **6.7%** of customers default today.
- **Random Forest** performs best (AP **0.376**, AUC **0.851**).
- At cutoff **0.30**, about **9.1%** of customers are flagged.
- Among flagged, ~**0.377** are true defaulters; we catch **0.511** of all defaulters.
- Separation is strong (**KS 0.560**); probabilities align (**Brier 0.053**).

Per **10,000** customers: **908 flagged** (~**343** true, **566** false alarms; **327** missed).

## Age, Dependents & Income — plain English
- **By age:** highest default in **30–39** at **7.1%**; lowest in **50–59** at **6.5%**.
- **Dependents:** highest default with **1–2** dependents at **6.8%**; lowest in **3+** at **6.3%**.
- **Income:** highest default in **Income: Bottom 20%** at **7.0%**; lowest in **Income: Top 20%** at **6.5%**.
**How to use this:** keep decisions score-driven and policy-compliant; use these slices to plan reviews and limits, not as stand-alone rules.

## Risk tiers (A–E)
- A: default **0.5%**, population **10.0%** • B: default **1.2%**, population **10.0%** • C: default **1.3%**, population **20.0%** • D: default **2.4%**, population **20.0%** • E: default **14.5%**, population **40.0%**
**Use:** focus manual reviews on **A/B**, moderate limits on **C**, lighter touch on **D/E**.

## Threshold scenarios (capacity & trade-offs)
t=0.20: flag 14.5%, P=0.292, R=0.631, TP=1,263, FP=3,056, cost=10,446 | t=0.25: flag 11.2%, P=0.338, R=0.567, TP=1,135, FP=2,220, cost=10,890 | t=0.30: flag 9.1%, P=0.377, R=0.511, TP=1,024, FP=1,690, cost=11,470 | t=0.35: flag 7.5%, P=0.405, R=0.453, TP=907, FP=1,331, cost=12,281 | t=0.40: flag 6.1%, P=0.437, R=0.397, TP=794, FP=1,023, cost=13,103

**Tip:** choose the row with the **lowest cost** that fits your review capacity.

## Two-way patterns (who is consistently riskier)
- 30–39 × Income: High 20%: default **8.6%** (flag 9.5%, n=849)
- <30 × Income: Low 20%: default **7.9%** (flag 9.1%, n=419)
- 30–39 × Income: Bottom 20%: default **7.5%** (flag 9.2%, n=942)
- 1–2 × Income: Low 20%: default **7.5%** (flag 8.8%, n=1,508)
- 3+ × Income: Bottom 20%: default **7.2%** (flag 8.8%, n=318)
- 1–2 × Income: Middle 20%: default **7.2%** (flag 8.9%, n=1,777)
- 0.51–1.00 × Debt: Highest 20%: default **7.8%** (flag 9.8%, n=1,245)
- 0.11–0.50 × Debt: Highest 20%: default **7.4%** (flag 10.7%, n=1,517)
- ≤0.10 × Debt: Low 20%: default **7.3%** (flag 8.7%, n=2,675)

## Policy rules (draft) — who to review or limit
- **Debt ratio highest 20% with ≤2 open lines** — default **9.2%** (portfolio **2.0%**). **Action:** Tighter limits or secondary checks.

## Comparison tables
**Age bands**

| Group | Share of portfolio | Default rate | Flag rate |
|---|---:|---:|---:|
| <30 | 7.1% | 7.0% | 9.2% |
| 30–39 | 16.3% | 7.1% | 9.5% |
| 40–49 | 23.4% | 6.6% | 8.9% |
| 50–59 | 23.3% | 6.5% | 8.9% |
| 60+ | 29.9% | 6.6% | 9.1% |

**Dependents**

| Group | Share of portfolio | Default rate | Flag rate |
|---|---:|---:|---:|
| 0 | 60.5% | 6.7% | 9.2% |
| 1–2 | 30.4% | 6.8% | 8.9% |
| 3+ | 9.1% | 6.3% | 8.9% |

**Income (20% bands)**

| Group | Share of portfolio | Default rate | Flag rate |
|---|---:|---:|---:|
| Income: Bottom 20% | 20.9% | 7.0% | 9.2% |
| Income: Low 20% | 19.1% | 6.6% | 8.5% |
| Income: Middle 20% | 20.1% | 6.9% | 9.5% |
| Income: High 20% | 19.9% | 6.5% | 8.6% |
| Income: Top 20% | 20.0% | 6.5% | 9.1% |

**Revolving utilization**

| Group | Share of portfolio | Default rate | Flag rate |
|---|---:|---:|---:|
| ≤0.10 | 42.7% | 6.7% | 8.9% |
| 0.11–0.50 | 29.5% | 6.9% | 9.4% |
| 0.51–1.00 | 25.6% | 6.5% | 8.9% |
| 1.01–5.00 | 2.0% | 8.0% | 9.6% |
| >5.00 | 0.1% | 4.8% | 16.7% |

**Debt ratio (20% bands)**

| Group | Share of portfolio | Default rate | Flag rate |
|---|---:|---:|---:|
| Debt: Lowest 20% | 20.0% | 6.6% | 8.9% |
| Debt: Low 20% | 20.0% | 6.9% | 9.0% |
| Debt: Middle 20% | 20.0% | 6.4% | 9.2% |
| Debt: High 20% | 20.1% | 6.8% | 8.7% |
| Debt: Highest 20% | 19.9% | 6.8% | 9.7% |

**Open credit lines**

| Group | Share of portfolio | Default rate | Flag rate |
|---|---:|---:|---:|
| ≤2 | 8.8% | 7.5% | 9.8% |
| 3–6 | 31.6% | 6.6% | 8.8% |
| 7+ | 59.6% | 6.6% | 9.1% |

**Prior delinquencies**

| Group | Share of portfolio | Default rate | Flag rate |
|---|---:|---:|---:|
| none | 79.8% | 6.7% | 9.1% |
| 1 | 11.7% | 6.7% | 9.0% |
| 2+ | 8.4% | 7.1% | 9.2% |


## Top 10 Highest-Risk Microsegments
| Segment | Share of portfolio | Default rate | Lift vs base | Flag rate | n |
|---|---:|---:|---:|---:|---:|
| ≤2 × Debt: Highest 20% | 2.1% | 9.2% | 1.37 | 11.0% | 611 |
| 30–39 × Income: High 20% | 3.6% | 8.6% | 1.28 | 9.5% | 849 |
| 2+ × Income: High 20% | 1.6% | 8.0% | 1.20 | 8.6% | 374 |
| <30 × Income: Low 20% | 1.8% | 7.9% | 1.18 | 9.1% | 419 |
| 0.51–1.00 × Debt: Highest 20% | 4.3% | 7.8% | 1.16 | 9.8% | 1,245 |
| 2+ × Income: Bottom 20% | 2.0% | 7.8% | 1.16 | 8.6% | 488 |
| 30–39 × Income: Bottom 20% | 4.0% | 7.5% | 1.12 | 9.2% | 942 |
| 1–2 × Income: Low 20% | 6.3% | 7.5% | 1.12 | 8.8% | 1,508 |
| 2+ × Income: Top 20% | 1.3% | 7.5% | 1.12 | 10.1% | 307 |
| 40–49 × Income: Middle 20% | 5.2% | 7.4% | 1.11 | 9.1% | 1,226 |

**Take away:** Risk concentrates in **none** combined with **Income: Bottom 20%**.
**What this shows:** small, specific groups where risk clusters when two fields are combined (e.g., age with income, or utilization with debt ratio).

**How to read it**
- **Share of portfolio** — how many customers sit in that group.
- **Default rate** — loss risk inside the group.
- **Lift vs base** — how much worse than the overall average (1.20× = 20% higher than average).
- **Flag rate** — at the current cutoff, what share would be sent to review.

**How to act**
- Prioritise control actions (extra docs, manual checks, lower starting limits, pricing add-ons) for segments with **high default rate/lift** and a **meaningful share of the portfolio**.
- Keep final decisions **score-driven** and **policy-compliant**; segments focus your effort, not replace the score.

## Model Quality (evidence)
- **Best by AP:** Random Forest — AP **0.376**.
- **Best by AUC:** Random Forest — AUC **0.851**.
- Base default rate: **6.7%**. Best AP ≈ **0.376** (~**5.62×** random baseline).
- Separation: **KS 0.560**.
- Calibration: **Brier 0.053**.

## Recommended operating point
- Threshold: **0.30**
- Precision: **0.377**, Recall: **0.511**
- Confusion mix: TP=1,024 FP=1,690 TN=26,186 FN=978
**Costed impact (example):** treat a missed defaulter (FN) as ~10× a false alarm (FP). Pick the lowest-cost threshold that your team can review.

## Recommendation
Pilot for 4–6 weeks at the chosen cutoff. Monitor **precision, recall, Brier, KS, flag rate** weekly; recalibrate quarterly or on drift.
