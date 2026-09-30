import json
import matplotlib.pyplot as plt
import os

os.makedirs("results/charts", exist_ok=True)

files = [
    "results/data/synonym_invariance.json",
    "results/data/negation_dir.json",
    "results/data/irrelevant_insertion.json",
    "results/data/typo_robustness.json",
]

labels = []
rates = []

for filepath in files:
    with open(filepath) as f:
        data = json.load(f)
        labels.append(data["relation"])
        rates.append(data["violation_rate"])

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(labels, rates, color="#C44E52", width=0.5)

ax.set_ylabel("Violation Rate (%)")
ax.set_title("Metamorphic Testing: Violation Rate by Relation")
ax.set_ylim(0, max(rates) + 15)
plt.xticks(rotation=15, ha="right")

for bar, rate in zip(bars, rates):
    ax.text(bar.get_x() + bar.get_width() / 2, rate + 1, f"{rate:.0f}%",
             ha="center", fontsize=10, fontweight="bold")

plt.tight_layout()
plt.savefig("results/charts/metamorphic_violation_rates.png", dpi=150)
print("Chart saved to results/charts/metamorphic_violation_rates.png")