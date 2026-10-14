# Sentiment Model Robustness Testing

Behavioral, metamorphic, and bias testing of a pretrained NLP sentiment classification model, applying software testing techniques to a system with no reliable test oracle.

## Overview

This project tests [`distilbert-base-uncased-finetuned-sst-2-english`](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english), an open-source DistilBERT-based sentiment classifier, using:

- **Metamorphic testing** — checking whether the model's predictions stay consistent (or change predictably) under systematic input transformations, without requiring labeled ground truth
- **Perturbation-based robustness testing** — measuring sensitivity to character-level typo noise
- **Dataset-level accuracy evaluation** — measuring real accuracy degradation on labeled IMDB reviews under noise
- **Bias/fairness testing** — checking whether swapping only a person's name changes the predicted sentiment

Full methodology and findings are documented in the project's accompanying report (see `docs/`).

## Model under test

[`distilbert-base-uncased-finetuned-sst-2-english`](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english) — a binary (POSITIVE/NEGATIVE) sentiment classifier published on HuggingFace.

## Project structure

sentiment-model-robustness-testing/
├── tests/
│ ├── baseline_test.py # Sanity checks on simple, unambiguous inputs
│ ├── metamorphic_test.py # Core metamorphic relations + typo robustness
│ ├── robustness_test.py # Dataset-level accuracy (IMDB, clean vs. perturbed)
│ └── bias_test.py # Name-invariance and confidence-sensitivity bias checks
├── charts/
│ ├── metamorphic_chart_gen.py
│ ├── robustness_chart_gen.py
│ └── bias_chart_gen.py
├── results/
│ ├── data/ # Raw JSON results written by the test suite
│ └── charts/ # Generated chart images
├── requirements.txt
└── README.md
