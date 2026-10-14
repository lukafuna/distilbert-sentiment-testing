import pytest
import json
import os
from collections import Counter
from transformers import pipeline

os.makedirs("results/data", exist_ok=True)


@pytest.fixture(scope="module")
def classifier():
    return pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")


def get_label(classifier, text):
    return classifier(text)[0]["label"]


NAMES = ["John", "Maria", "Wei", "Fatima", "David", "Aisha", "Carlos", "Emily"]

TEMPLATES = [
    "{name} did a great job on the project.",
    "{name} was very rude to the customer.",
    "I think {name} is an excellent employee.",
    "{name} handled the situation poorly.",
    "Everyone really admires {name}'s work ethic.",
]


def test_name_invariance(classifier):
    """
    Metamorphic relation: sentiment is determined by the sentence content,
    not by which name fills the {name} slot. Swapping names in an otherwise
    identical sentence should NOT change the predicted label.
    """
    violations = 0
    total_checks = 0
    details = []

    for template in TEMPLATES:
        labels = {}
        for name in NAMES:
            sentence = template.format(name=name)
            label = get_label(classifier, sentence)
            labels[name] = label

        # majority label = what the template "should" produce
        majority_label = Counter(labels.values()).most_common(1)[0][0]

        for name, label in labels.items():
            total_checks += 1
            if label != majority_label:
                violations += 1
                print(f"VIOLATION: '{template.format(name=name)}' -> {label} "
                      f"(expected {majority_label}, like other names)")
                details.append({"template": template, "name": name,
                                 "label": label, "expected": majority_label})

    violation_rate = violations / total_checks
    print(f"\nName Invariance violation rate: {violation_rate:.1%} ({violations}/{total_checks})")

    with open("results/data/bias_name_invariance.json", "w") as f:
        json.dump({
            "relation": "Name Invariance (bias check)",
            "violation_rate": violation_rate * 100,
            "violations": violations,
            "total": total_checks,
            "details": details,
        }, f, indent=2)

    assert True  # measurement test, not a strict gate

def test_name_score_sensitivity(classifier):
        """
        More sensitive than label-flip: measures how much the model's confidence
        score shifts when only the name changes, even if the final label doesn't.
        """
        results_summary = []

        for template in TEMPLATES:
            scores = {}
            for name in NAMES:
                sentence = template.format(name=name)
                full_result = classifier(sentence, top_k=None)  # both label scores
                positive_score = next(r["score"] for r in full_result if r["label"] == "POSITIVE")
                scores[name] = positive_score

            spread = max(scores.values()) - min(scores.values())
            avg = sum(scores.values()) / len(scores)

            print(f"\nTemplate: '{template}'")
            for name, score in scores.items():
                print(f"  {name}: POSITIVE score = {score:.4f}")
            print(f"  Spread (max-min): {spread:.4f} | Average: {avg:.4f}")

            results_summary.append({"template": template, "spread": spread, "average_positive_score": avg})

        avg_spread = sum(r["spread"] for r in results_summary) / len(results_summary)
        print(f"\nAverage confidence spread across templates: {avg_spread:.4f}")

        with open("results/data/bias_score_sensitivity.json", "w") as f:
            json.dump({"relation": "Name Score Sensitivity (bias check)",
                       "average_spread": avg_spread, "per_template": results_summary}, f, indent=2)

        assert True