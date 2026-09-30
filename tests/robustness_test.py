import pytest
from transformers import pipeline
from datasets import load_dataset
import nlpaug.augmenter.char as nac
import json
import os

os.makedirs("results/data", exist_ok=True)

@pytest.fixture(scope="module")
def classifier():
    return pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")


@pytest.fixture(scope="module")
def imdb_sample():
    """Load a small sample of the IMDB dataset for testing."""
    dataset = load_dataset("stanfordnlp/imdb", split="test")
    sample = dataset.shuffle(seed=42).select(range(100))
    return sample


def test_baseline_accuracy(classifier, imdb_sample):
    correct = 0
    total = len(imdb_sample)

    for example in imdb_sample:
        text = example["text"][:512]
        true_label = "POSITIVE" if example["label"] == 1 else "NEGATIVE"
        predicted_label = classifier(text)[0]["label"]
        if predicted_label == true_label:
            correct += 1

    accuracy = correct / total
    print(f"\nBaseline accuracy on {total} IMDB samples: {accuracy:.1%} ({correct}/{total})")

    # Save result for the chart script to use later
    with open("results/data/baseline_accuracy.json", "w") as f:
        json.dump({"accuracy": accuracy * 100, "correct": correct, "total": total}, f)

    assert accuracy > 0.5
    

def test_accuracy_under_typo_perturbation(classifier, imdb_sample):
    aug = nac.KeyboardAug(aug_char_p=0.1)
    correct = 0
    total = len(imdb_sample)

    for example in imdb_sample:
        text = example["text"][:512]
        true_label = "POSITIVE" if example["label"] == 1 else "NEGATIVE"
        perturbed_text = aug.augment(text)[0]
        predicted_label = classifier(perturbed_text)[0]["label"]
        if predicted_label == true_label:
            correct += 1

    accuracy = correct / total
    print(f"\nAccuracy under typo perturbation on {total} IMDB samples: {accuracy:.1%} ({correct}/{total})")

    with open("results/data/typo_perturbation_accuracy.json", "w") as f:
        json.dump({"accuracy": accuracy * 100, "correct": correct, "total": total}, f)

    assert True

