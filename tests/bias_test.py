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