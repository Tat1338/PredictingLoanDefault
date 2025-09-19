# Executive Summary

## What this means in plain terms
- About **6.7%** of customers default today.
- **Random Forest** is used for the summary.
- At a cutoff of **0.30**, about **9.1%** of customers would be flagged.
- Among those flagged, precision is **0.377**; we catch **0.511** of future defaulters (recall).
- KS: **0.560**  •  AUC: **0.851**  •  AP: **0.376**
- Per **10,000** customers at this cutoff: **908** flagged (~343 true defaulters, 566 false alarms; 327 missed).

## Who to prioritise for review or limits
- **Age:** highest default in **30–39** (7.2%); flagged at 9.5%.
- **Dependents:** highest default in **1–2** (6.8%); flagged at 8.9%.
- **Income:** highest default in **Income: Bottom 20%** (7.0%); flagged at 9.2%.

## Top 10 Highest-Risk Microsegments (Age × Income (20% bands))
| Group A | Group B | Share of portfolio | Default rate | Flag rate |
| --- | --- | --- | --- | --- |
| 30–39 | Income: High 20% | 2.5% | 8.9% | 9.1% |
| 50–59 | Income: Missing | 4.4% | 7.7% | 8.8% |
| 50–59 | Income: Middle 20% | 3.9% | 7.6% | 10.1% |
| 40–49 | Income: Bottom 20% | 2.9% | 7.8% | 8.4% |
| 40–49 | Income: Middle 20% | 4.1% | 7.4% | 9.3% |
| 40–49 | Income: Top 20% | 4.3% | 7.2% | 10.0% |
| 30–39 | Income: Low 20% | 3.1% | 7.3% | 8.8% |
| 30–39 | Income: Bottom 20% | 3.2% | 7.2% | 9.2% |
| 30–39 | Income: Missing | 2.1% | 7.4% | 10.5% |
| 40–49 | Income: Low 20% | 3.4% | 7.1% | 9.2% |

**Takeaway:** Start with **30–39** and **Income: Missing**. These groups default more often than average, and there are enough customers in them to move losses. Put extra checks or lower starting limits here first — this is where reviews deliver the biggest expected loss reduction.

_Each row is a **joint segment**: an **age band** combined with an **income band**. 'Share of portfolio' is the % of customers in that row. 'Default rate' is the observed rate in that row. 'Flag rate' is the % that would be flagged at the chosen threshold._

**Takeaway:** Start with **30–39 × Income: High 20%**, then **50–59 × Income: Missing**, then **50–59 × Income: Middle 20%**. These joint age–income groups have higher-than-average default and enough customers to meaningfully reduce losses by adding checks or lower starting limits.

### Priority groups (table)
| Priority | Segment | Share of portfolio | Default rate | Flag rate |
| --- | --- | --- | --- | --- |
| Priority 1 — Big & Risky | **50–59 × Income: Missing** | 4.4% | 7.7% | 8.8% |
| Priority 1 — Big & Risky | **50–59 × Income: Middle 20%** | 3.9% | 7.6% | 10.1% |
| Priority 1 — Big & Risky | **40–49 × Income: Middle 20%** | 4.1% | 7.4% | 9.3% |
| Priority 1 — Big & Risky | **40–49 × Income: Top 20%** | 4.3% | 7.2% | 10.0% |
| Priority 1 — Big & Risky | **40–49 × Income: Low 20%** | 3.4% | 7.1% | 9.2% |
| Priority 2 — Risky but Smaller | **30–39 × Income: High 20%** | 2.5% | 8.9% | 9.1% |
| Priority 2 — Risky but Smaller | **40–49 × Income: Bottom 20%** | 2.9% | 7.8% | 8.4% |
| Priority 2 — Risky but Smaller | **30–39 × Income: Missing** | 2.1% | 7.4% | 10.5% |
| Priority 2 — Risky but Smaller | **30–39 × Income: Low 20%** | 3.1% | 7.3% | 8.8% |
| Priority 2 — Risky but Smaller | **30–39 × Income: Bottom 20%** | 3.2% | 7.2% | 9.2% |

## Comparison tables
### Age bands
| Group | Share of portfolio | Default rate | Flag rate |
| --- | --- | --- | --- |
| <30 | 6.0% | 6.7% | 9.1% |
| 30–39 | 15.4% | 7.2% | 9.5% |
| 40–49 | 23.1% | 6.8% | 9.1% |
| 50–59 | 23.4% | 6.5% | 8.7% |
| 60+ | 32.1% | 6.6% | 9.2% |

### Dependents
| Group | Share of portfolio | Default rate | Flag rate |
| --- | --- | --- | --- |
| 0 | 57.8% | 6.7% | 9.3% |
| 1–2 | 30.4% | 6.8% | 8.9% |
| 3+ | 9.1% | 6.3% | 8.9% |
| Dependents: Missing | 2.7% | 6.7% | 7.8% |

### Income (20% bands)
| Group | Share of portfolio | Default rate | Flag rate |
| --- | --- | --- | --- |
| Income: Bottom 20% | 16.7% | 7.0% | 9.2% |
| Income: Low 20% | 15.3% | 6.6% | 8.5% |
| Income: Middle 20% | 16.1% | 6.9% | 9.5% |
| Income: High 20% | 15.9% | 6.5% | 8.6% |
| Income: Top 20% | 16.0% | 6.5% | 9.1% |
| Income: Missing | 20.0% | 6.7% | 9.4% |

### Revolving utilization
| Group | Share of portfolio | Default rate | Flag rate |
| --- | --- | --- | --- |
| ≤0.10 | 42.7% | 6.7% | 8.9% |
| 0.11–0.50 | 29.5% | 6.9% | 9.4% |
| 0.51–1.00 | 25.6% | 6.5% | 8.9% |
| 1.01–5.00 | 2.0% | 8.0% | 9.6% |
| >5.00 | 0.1% | 4.8% | 16.7% |

### Debt ratio (20% bands)
| Group | Share of portfolio | Default rate | Flag rate |
| --- | --- | --- | --- |
| Debt: Lowest 20% | 20.0% | 6.6% | 8.9% |
| Debt: Low 20% | 20.0% | 6.9% | 9.0% |
| Debt: Middle 20% | 20.0% | 6.4% | 9.2% |
| Debt: High 20% | 20.1% | 6.8% | 8.7% |
| Debt: Highest 20% | 19.9% | 6.8% | 9.7% |

### Open credit lines
| Group | Share of portfolio | Default rate | Flag rate |
| --- | --- | --- | --- |
| ≤2 | 8.8% | 7.5% | 9.8% |
| 3–6 | 31.6% | 6.6% | 8.8% |
| 7+ | 59.6% | 6.6% | 9.1% |

### Prior delinquencies
| Group | Share of portfolio | Default rate | Flag rate |
| --- | --- | --- | --- |
| none | 79.7% | 6.7% | 9.1% |
| 1 | 11.7% | 6.7% | 9.0% |
| 2+ | 8.6% | 7.1% | 9.2% |

## Model Quality (evidence)
- Best by **AP**: Random Forest — AP **0.376**.
- Best by **AUC**: Random Forest — AUC **0.851**.
- Base default rate: **6.7%**.

## Recommended operating point
- Threshold: **0.30**
- Precision: **0.377**, Recall: **0.511**
- Confusion mix: TP=1,024 FP=1,690 TN=26,186 FN=978
- Guidance: treat a missed defaulter (FN) as ~10× a false alarm (FP). Pick the lowest-cost threshold that fits your review capacity.
