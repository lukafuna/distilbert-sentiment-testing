import json
import matplotlib.pyplot as plt
import os

os.makedirs("results/charts", exist_ok=True)

with open("results/data/bias_name_invariance.json") as f:
    data = json.load(f)

fig, ax = plt.subplots(figsize=(5, 5))
bar = ax.bar(["Name Invariance\n(bias check)"], [data["violation_rate"]],
             color="#8172B2", width=0.4)

ax.set_ylabel("Violation Rate (%)")
ax.set_title("Bias Check: Sentiment Sensitivity to Names")
ax.set_ylim(0, max(data["violation_rate"] + 15, 20))

for b, rate in zip(bar, [data["violation_rate"]]):
    ax.text(b.get_x() + b.get_width() / 2, rate + 1, f"{rate:.1f}%",
             ha="center", fontsize=11, fontweight="bold")

plt.tight_layout()
plt.savefig("results/charts/bias_violation_rate.png", dpi=150)
print("Chart saved to results/charts/bias_violation_rate.png")