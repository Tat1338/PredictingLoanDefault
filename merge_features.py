# merge_features.py
# Join raw features to holdout predictions so the summary can analyze slices.

from __future__ import annotations
import os
import pandas as pd

ROOT = os.path.dirname(__file__)
EXP = os.path.join(ROOT, "exports")
pred_path = os.path.join(EXP, "holdout_predictions.csv")

if not os.path.exists(pred_path):
    raise SystemExit("Missing exports/holdout_predictions.csv")

# Put your raw data CSV (with Age/MonthlyIncome/Dependents/etc.) in *one* of these places:
candidates = [
    os.path.join(ROOT, "cs-training.csv"),
    os.path.join(ROOT, "data", "cs-training.csv"),
    os.path.join(EXP,  "holdout_features.csv"),
    os.path.join(ROOT, "data", "holdout_features.csv"),
]

feat_path = next((p for p in candidates if os.path.exists(p)), None)
if feat_path is None:
    raise SystemExit(
        "Could not find a features CSV.\n"
        "Add your raw data as one of:\n"
        " - cs-training.csv\n - data/cs-training.csv\n - exports/holdout_features.csv\n - data/holdout_features.csv"
    )

print(f"Using features file: {feat_path}")

preds = pd.read_csv(pred_path)
feats = pd.read_csv(feat_path)

# Keep common credit-risk columns if present (falls back to 'all' if none match)
wanted = [
    "Age","age",
    "MonthlyIncome","income",
    "NumberOfDependents","Dependents","dependents",
    "RevolvingUtilizationOfUnsecuredLines",
    "DebtRatio",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberOfTime30-59DaysPastDueNotWorse","NumberOfTimes30-59DaysPastDueNotWorse",
    "NumberOfTime60-89DaysPastDueNotWorse","NumberOfTimes60-89DaysPastDueNotWorse",
]
keep = [c for c in wanted if c in feats.columns] or list(feats.columns)

# Align by row order (assumes holdout made from the same rows)
n = min(len(preds), len(feats))
if len(preds) != len(feats):
    print(f"Note: length mismatch — preds={len(preds)}, feats={len(feats)}. Using first {n} rows.")

combined = pd.concat(
    [feats.iloc[:n][keep].reset_index(drop=True),
     preds.iloc[:n].reset_index(drop=True)],
    axis=1
)

out_path = os.path.join(EXP, "holdout_with_features.csv")
combined.to_csv(out_path, index=False)
print(f"Wrote {out_path}")
print("Next: I’ll show you one command to run this, then we’ll refresh the summary.")
