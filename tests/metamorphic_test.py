import pytest
from transformers import pipeline


@pytest.fixture(scope="module")
def classifier():
    return pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")


def get_label(classifier, text):
    """label string (POSITIVE/NEGATIVE) for a sentence."""
    return classifier(text)[0]["label"]


SYNONYM_PAIRS = [
    ("This movie was great", "This movie was fantastic"),
    ("I loved this film", "I adored this film"),
    ("The food was terrible", "The food was awful"),
    ("This is a bad product", "This is a poor product"),
    ("The service was excellent", "The service was outstanding"),
    ("I hated the ending", "I despised the ending"),
    ("This book is boring", "This book is dull"),
    ("The staff was friendly", "The staff was kind"),
    ("What a disappointing experience", "What a letdown experience"),
    ("This is amazing work", "This is superb work"),
]


def test_synonym_invariance(classifier):
    violations = 0
    total = len(SYNONYM_PAIRS)

    for original, variant in SYNONYM_PAIRS:
        label_original = get_label(classifier, original)
        label_variant = get_label(classifier, variant)

        if label_original != label_variant:
            violations += 1
            print(f"VIOLATION: '{original}' -> {label_original} | '{variant}' -> {label_variant}")

    violation_rate = violations / total
    print(f"\nSynonym Invariance violation rate: {violation_rate:.1%} ({violations}/{total})")

    assert violation_rate <= 0.3, f"Too many synonym invariance violations: {violation_rate:.1%}"

NEGATION_PAIRS = [
    ("I don't dislike this movie", "I dislike this movie"),  # double negative flip
    ("This isn't bad at all", "This is bad"),
    ("It's not like I hated it", "I hated it"),
    ("I can't say I enjoyed it", "I enjoyed it"),
    ("Not everyone will love this film", "Everyone will love this film"),
]


def test_negation_flips_sentiment(classifier):
    violations = 0
    total = len(NEGATION_PAIRS)

    for original, negated in NEGATION_PAIRS:
        label_original = get_label(classifier, original)
        label_negated = get_label(classifier, negated)


        if label_original == label_negated:
            violations += 1
            print(f"VIOLATION (no flip): '{original}' -> {label_original} | '{negated}' -> {label_negated}")

    violation_rate = violations / total
    print(f"\nNegation DIR violation rate: {violation_rate:.1%} ({violations}/{total})")

    assert violation_rate <= 0.5, f"Too many negation violations: {violation_rate:.1%}"


IRRELEVANT_INSERTION_PAIRS = [
    ("This movie was great", "This movie was great. I watched it on a Tuesday."),
    ("The food was terrible", "The food was terrible. The restaurant is downtown."),
    ("I loved this book", "I loved this book. It has 300 pages."),
    ("This product is excellent", "This product is excellent. It arrived on time."),
    ("The service was bad", "The service was bad. The building has two floors."),
]

def test_irrelevant_insertion_invariance(classifier):
    violations = 0
    total = len(IRRELEVANT_INSERTION_PAIRS)

    for original, extended in IRRELEVANT_INSERTION_PAIRS:
        label_original = get_label(classifier, original)
        label_extended = get_label(classifier, extended)

        if label_original != label_extended:
            violations += 1
            print(f"VIOLATION: '{original}' -> {label_original} | '{extended}' -> {label_extended}")

    violation_rate = violations / total
    print(f"\nIrrelevant insertion violation rate: {violation_rate:.1%} ({violations}/{total})")

    assert violation_rate <= 0.3, f"Too many insertion violations: {violation_rate:.1%}"


CONTRAST_PAIRS = [
    ("The acting was great, but the plot was terrible", None),
    ("I expected to hate it, but I loved it", None),
    ("The service was slow, but the food was amazing", None),
]

SARCASM_EXAMPLES = [
    "Oh great, another Monday.",
    "Wow, what a fantastic way to ruin my day.",
    "Just what I needed, more traffic.",
]

def test_contrast_sentences_logging(classifier):

    print("\n--- Contrastive sentence behavior ---")
    for sentence, _ in CONTRAST_PAIRS:
        label = get_label(classifier, sentence)
        print(f"'{sentence}' -> {label}")

    assert True


def test_sarcasm_logging(classifier):

    print("\n--- Sarcasm behavior ---")
    for sentence in SARCASM_EXAMPLES:
        label = get_label(classifier, sentence)
        print(f"'{sentence}' -> {label}")

    assert True

import nlpaug.augmenter.char as nac

def test_typo_robustness(classifier):
    aug = nac.KeyboardAug(aug_char_p=0.15)
    sentences = [
        "This movie was absolutely fantastic",
        "I really hated this restaurant",
        "The customer service was excellent",
        "This book was incredibly boring",
        "I would highly recommend this product",
        "The plot was confusing and slow",
        "Great performance by the lead actor",
        "I was disappointed by the ending",
        "This is the best purchase I've made",
        "Terrible quality, would not buy again",
    ]

    violations = 0
    total_runs = 3

    for sentence in sentences:
        original_label = get_label(classifier, sentence)
        for _ in range(total_runs):
            typo_version = aug.augment(sentence)[0]
            typo_label = get_label(classifier, typo_version)
            if original_label != typo_label:
                violations += 1

    total_checks = len(sentences) * total_runs
    rate = violations / total_checks
    print(f"\nTypo robustness violation rate: {rate:.1%} ({violations}/{total_checks})")
    assert True