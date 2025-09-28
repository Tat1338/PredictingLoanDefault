import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# 1) load data (adjust the filename if yours is different)
csv = Path("cs-training.csv")
if not csv.exists():
    raise FileNotFoundError("Put cs-training.csv in the project root or change the path in this file.")

df = pd.read_csv(csv)
df = df[df["age"] > 0].copy()

# 2) make age groups
bins   = [18, 25, 35, 45, 55, 65, 120]
labels = ["18–24","25–34","35–44","45–54","55–64","65+"]
df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels, right=False)

# 3) default % per age group
rate = df.groupby("age_group")["SeriousDlqin2yrs"].mean() * 100

# 4) clean bar chart — ONLY % labels (no client counts)
fig, ax = plt.subplots(figsize=(8,5))
bars = ax.bar(rate.index, rate.values)
for b, p in zip(bars, rate.values):
    ax.text(b.get_x()+b.get_width()/2, p+0.3, f"{p:.1f}%", ha="center", va="bottom", fontsize=10)

ax.set_title("Share late by age group")
ax.set_ylabel("% of clients late")
ax.set_xlabel("Age group")
ax.set_ylim(0, max(rate.values)+3)
fig.tight_layout()

# 5) overwrite your existing figure
out = Path("reports/figures/age_group_default_rate.png")
fig.savefig(out, dpi=200, bbox_inches="tight")
print("Saved:", out)
