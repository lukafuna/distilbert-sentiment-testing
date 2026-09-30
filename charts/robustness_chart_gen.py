import json
import matplotlib.pyplot as plt
import os

os.makedirs("../results/charts", exist_ok=True)

# Load actual test results instead of hardcoding
with open("../results/data/baseline_accuracy.json") as f:
    baseline = json.load(f)

with open("../results/data/typo_perturbation_accuracy.json") as f:
    perturbed = json.load(f)

conditions = ["Baseline\n(clean text)", "Typo Perturbation\n(10% char corruption)"]
accuracies = [baseline["accuracy"], perturbed["accuracy"]]

fig, ax = plt.subplots(figsize=(6, 5))
bars = ax.bar(conditions, accuracies, color=["#4C72B0", "#DD8452"], width=0.5)

ax.set_ylabel("Accuracy (%)")
ax.set_title(f"Model Accuracy: Baseline vs. Typo-Perturbed Input\n(IMDB test set, n={baseline['total']})")
ax.set_ylim(0, 100)

for bar, acc in zip(bars, accuracies):
    ax.text(bar.get_x() + bar.get_width() / 2, acc + 1.5, f"{acc:.1f}%",
             ha="center", fontsize=11, fontweight="bold")

plt.tight_layout()
plt.savefig("../results/charts/accuracy_comparison.png", dpi=150)
print("Chart saved to results/charts/accuracy_comparison.png")